# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import json
import os
import re
from collections.abc import AsyncGenerator
from typing import Any

import google.auth
from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.apps import App, ResumabilityConfig
from google.adk.events.event import Event, EventActions
from google.adk.events.request_input import RequestInput
from google.adk.workflow import START, FunctionNode, Workflow
from google.genai import types
from pydantic import BaseModel, Field

from expense_agent.config import MODEL_NAME, THRESHOLD

# Setup local authentication environment for Vertex AI if API key is not present
if "GEMINI_API_KEY" not in os.environ and "GOOGLE_API_KEY" not in os.environ:
    try:
        _, project_id = google.auth.default()
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")
    except Exception:
        pass


# ==============================================================================
# Pydantic Schemas
# ==============================================================================


class Expense(BaseModel):
    """Represents a validated expense report."""

    amount: float = Field(description="The transaction value of the expense.")
    submitter: str = Field(description="The employee who submitted the expense.")
    category: str = Field(description="The category of the expense.")
    description: str = Field(description="Description details of the expense.")
    date: str = Field(description="The date of the transaction.")


class RiskAnalysis(BaseModel):
    """LLM Compliance Risk Evaluation Schema."""

    risk_score: int = Field(
        description="Compliance risk score from 1 (low) to 5 (high)"
    )
    flagged_items: list[str] = Field(
        description="Specific items flagged for compliance concern"
    )
    justification: str = Field(description="Detailed justification for the risk score")
    policy_violations: bool = Field(
        description="Whether the expense violates standard policies"
    )


# ==============================================================================
# Workflow Node Implementations
# ==============================================================================


def parse_input(node_input: Any) -> Event:
    """Parses incoming JSON payloads (Pub/Sub base64 or plain json)."""
    payload_str = ""
    payload_dict = {}

    if isinstance(node_input, str):
        payload_str = node_input
    elif isinstance(node_input, dict):
        payload_dict = node_input
    elif hasattr(node_input, "parts"):
        # Handle google.genai.types.Content input from playground/CLI run
        payload_str = "".join([p.text for p in node_input.parts if p.text])
    else:
        payload_str = str(node_input)

    if payload_str:
        try:
            payload_dict = json.loads(payload_str)
        except Exception:
            pass

    # Extract base64 encoded Pub/Sub message data if present
    if "data" in payload_dict:
        data_val = payload_dict["data"]
        if isinstance(data_val, str):
            try:
                decoded = base64.b64decode(data_val).decode("utf-8")
                payload_dict = json.loads(decoded)
            except Exception:
                try:
                    payload_dict = json.loads(data_val)
                except Exception:
                    pass
        elif isinstance(data_val, dict):
            payload_dict = data_val

    expense = Expense(
        amount=float(payload_dict.get("amount", 0.0)),
        submitter=payload_dict.get("submitter", "Unknown"),
        category=payload_dict.get("category", "General"),
        description=payload_dict.get("description", ""),
        date=payload_dict.get("date", ""),
    )

    # Store in session state for later stages to access
    return Event(
        output=expense,
        actions=EventActions(state_delta={"expense": expense.model_dump()}),
    )


def triage_expense(node_input: Expense) -> Event:
    """Routes the expense dynamically based on transaction value."""
    if node_input.amount < THRESHOLD:
        return Event(output=node_input, actions=EventActions(route="low_value"))
    else:
        return Event(output=node_input, actions=EventActions(route="high_value"))


def auto_approve(node_input: Expense) -> AsyncGenerator[Event, None]:
    """Deterministically auto-approves low-value expenses."""
    result = {
        "status": "APPROVED",
        "justification": f"Auto-approved: expense amount ${node_input.amount:.2f} is under the ${THRESHOLD:.2f} threshold.",
        "expense": node_input.model_dump(),
    }
    yield Event(
        content=types.Content(
            role="model",
            parts=[
                types.Part.from_text(
                    text=f"✅ Low-value expense of ${node_input.amount:.2f} approved instantly."
                )
            ],
        )
    )
    yield Event(output=result)


def security_screen(node_input: Expense) -> Event:
    """Pre-LLM screening to redact email, SSN, and credit card PII and detect prompt injection."""
    desc = node_input.description

    # Redact email addresses
    desc_clean = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[REDACTED EMAIL]", desc)

    # SSN pattern: xxx-xx-xxxx or unhyphenated 9-11 digits
    ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b|\b\d{9,11}\b"
    # Credit Card pattern: 13-16 digits with optional spaces or hyphens
    cc_pattern = r"\b(?:\d[ -]*?){13,16}\b"

    redacted_categories = []
    if re.search(ssn_pattern, desc_clean):
        desc_clean = re.sub(ssn_pattern, "[REDACTED SSN]", desc_clean)
        redacted_categories.append("SSN")
    if re.search(cc_pattern, desc_clean):
        desc_clean = re.sub(cc_pattern, "[REDACTED CREDIT CARD]", desc_clean)
        redacted_categories.append("Credit Card")

    # Check for prompt injection heuristic
    lower_desc = desc_clean.lower()
    injection_detected = any(
        phrase in lower_desc
        for phrase in [
            "ignore previous",
            "system prompt",
            "bypass instructions",
            "override rules",
            "developer instruction",
            "ignore instructions",
            "bypass all rules",
            "bypass rules",
            "auto-approve",
        ]
    )

    screened = node_input.model_copy(update={"description": desc_clean})

    # Save to state so human review payload is clean as well
    state_delta = {
        "expense": screened.model_dump(),
        "redacted_categories": redacted_categories,
    }

    if injection_detected:
        state_delta["security_alert"] = "Prompt injection detected"
        return Event(
            output=screened,
            actions=EventActions(
                route="flagged_injection",
                state_delta=state_delta,
            ),
        )

    state_delta["security_alert"] = "None"
    return Event(
        output=screened,
        actions=EventActions(
            route="passed",
            state_delta=state_delta,
        ),
    )


# LLM Risk Judgment Node
llm_reviewer = LlmAgent(
    name="llm_reviewer",
    model=MODEL_NAME,
    instruction=(
        "You are an expense compliance reviewer. "
        "Analyze the provided corporate expense report. Identify any policy violations, "
        "suspicious descriptions, or other compliance concerns. "
        "Rate the risk on a scale of 1 to 5 and list your reasons."
    ),
    output_schema=RiskAnalysis,
    output_key="risk_analysis",
)


async def human_review(
    ctx: Context, node_input: RiskAnalysis | Expense
) -> AsyncGenerator[Event | RequestInput, None]:
    """Pauses the workflow for manual human review and decision authorization."""
    expense_dict = ctx.state.get("expense", {})
    expense_amount = expense_dict.get("amount", 0.0)
    expense_submitter = expense_dict.get("submitter", "Unknown")

    if isinstance(node_input, Expense):
        # LLM bypassed due to security event
        risk_analysis_dict = {
            "risk_score": 5,
            "flagged_items": ["Security Alert: Prompt Injection"],
            "justification": "Bypassed LLM reviewer: Prompt injection attempt detected in description!",
            "policy_violations": True,
        }
        llm_risk_score = 5
        llm_justification = (
            "Bypassed LLM reviewer: Prompt injection attempt detected in description!"
        )
        policy_violations = True
    else:
        risk_analysis_dict = node_input.model_dump()
        llm_risk_score = node_input.risk_score
        llm_justification = node_input.justification
        policy_violations = node_input.policy_violations

    if not ctx.resume_inputs or "decision" not in ctx.resume_inputs:
        msg = (
            f"⚠️  High-Value Expense Pending Review:\n"
            f"Submitter: {expense_submitter}\n"
            f"Amount: ${expense_amount:.2f}\n"
            f"LLM Risk Score: {llm_risk_score}/5\n"
            f"LLM Justification: {llm_justification}\n"
            f"Policy Violations: {policy_violations}\n\n"
            f"Please approve or reject this expense."
        )
        yield RequestInput(interrupt_id="decision", message=msg)
        return

    decision_val = ctx.resume_inputs["decision"]
    if isinstance(decision_val, dict):
        decision_raw = (
            decision_val.get("decision") or next(iter(decision_val.values()), "") or ""
        )
    else:
        decision_raw = str(decision_val)

    decision_raw = decision_raw.strip().lower()
    if "approve" in decision_raw:
        status = "APPROVED"
        msg_out = f"✅ Expense of ${expense_amount:.2f} by {expense_submitter} approved by human reviewer."
    else:
        status = "REJECTED"
        msg_out = f"❌ Expense of ${expense_amount:.2f} by {expense_submitter} rejected by human reviewer."

    result = {
        "status": status,
        "justification": f"Human review outcome: {status}",
        "expense": expense_dict,
        "risk_analysis": risk_analysis_dict,
    }
    yield Event(
        content=types.Content(role="model", parts=[types.Part.from_text(text=msg_out)])
    )
    yield Event(output=result)


# Human Review FunctionNode definition with rerun_on_resume=True
human_review_node = FunctionNode(
    func=human_review,
    name="human_review",
    rerun_on_resume=True,
)


# ==============================================================================
# Workflow definition
# ==============================================================================

root_agent = Workflow(
    name="expense_approval_workflow",
    edges=[
        (START, parse_input),
        (parse_input, triage_expense),
        # Route by transaction value using a RoutingMap dictionary
        (
            triage_expense,
            {
                "low_value": auto_approve,
                "high_value": security_screen,
            },
        ),
        # Route based on security screen output
        (
            security_screen,
            {
                "passed": llm_reviewer,
                "flagged_injection": human_review_node,
            },
        ),
        # Route high-value expense to human approval gate
        (llm_reviewer, human_review_node),
    ],
    description="Processes, triages, and routes employee expense reports with human-in-the-loop gating.",
)

app = App(
    root_agent=root_agent,
    name="expense_agent",
    resumability_config=ResumabilityConfig(is_resumable=True),
)
