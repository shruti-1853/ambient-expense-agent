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
# ruff: noqa: E402

import dotenv

# Load environment variables from .env early
dotenv.load_dotenv()

import json
import logging
import os
import uuid
from contextlib import aclosing
from typing import Any

from fastapi import FastAPI, HTTPException
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.runners import Runner
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.genai import types
from pydantic import BaseModel

from expense_agent.agent import root_agent
from expense_agent.app_utils.telemetry import setup_telemetry
from expense_agent.app_utils.typing import Feedback

# Standard Python logging configuration for console logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

setup_telemetry()

allow_origins = (
    os.getenv("ALLOW_ORIGINS", "").split(",") if os.getenv("ALLOW_ORIGINS") else None
)

# Artifact bucket for ADK (created by Terraform, passed via env var)
logs_bucket_name = os.environ.get("LOGS_BUCKET_NAME")

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# In-memory session configuration - no persistent storage
session_service_uri = None

artifact_service_uri = f"gs://{logs_bucket_name}" if logs_bucket_name else None

# Initialize FastAPI application with otel_to_cloud=False
app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,
    artifact_service_uri=artifact_service_uri,
    allow_origins=allow_origins,
    session_service_uri=session_service_uri,
    otel_to_cloud=False,
)
app.title = "ambient-expense-agent"
app.description = "API for interacting with the Agent ambient-expense-agent"

# Set up shared sqlite session storage so custom endpoint runs can be tracked in the UI
session_db_path = os.path.join(AGENT_DIR, ".adk", "session.db")
os.makedirs(os.path.dirname(session_db_path), exist_ok=True)
session_service = SqliteSessionService(db_path=session_db_path)
runner = Runner(
    agent=root_agent,
    session_service=session_service,
    app_name="expense_agent",
    auto_create_session=True,
)


# ==============================================================================
# Pub/Sub Schema & Endpoint Definitions
# ==============================================================================


class PubSubMessage(BaseModel):
    """Pydantic model representing a Pub/Sub message payload."""

    data: str | None = None
    attributes: dict[str, str] | None = None
    messageId: str | None = None
    publishTime: str | None = None


class PubSubEnvelope(BaseModel):
    """Pydantic model representing the Pub/Sub push subscription HTTP body."""

    message: PubSubMessage
    subscription: str | None = None


@app.post("/pubsub")
@app.post("/")
@app.post("/apps/expense_agent/trigger/pubsub")
async def handle_pubsub(envelope: PubSubEnvelope) -> dict[str, Any]:
    """Pub/Sub push subscription endpoint.

    Accepts incoming trigger payloads, normalizes the subscription path down to
    a short name, and runs the workflow runner.
    """
    subscription_path = (
        envelope.subscription or "projects/unknown/subscriptions/pubsub-caller"
    )

    # Normalize subscription path down to short name:
    # projects/my-project/subscriptions/my-subscription -> my-subscription
    short_subscription = subscription_path.split("/")[-1]

    logger.info(
        "Received Pub/Sub trigger. subscription=%s (normalized=%s), messageId=%s",
        subscription_path,
        short_subscription,
        envelope.message.messageId,
    )

    # Use normalized short name as user_id and messageId for session tracing
    session_id = f"pubsub-{envelope.message.messageId or uuid.uuid4().hex[:8]}"

    input_payload = {"data": envelope.message.data, "subscription": subscription_path}

    new_message = types.Content(
        role="user", parts=[types.Part.from_text(text=json.dumps(input_payload))]
    )

    try:
        events = []
        async with aclosing(
            runner.run_async(
                user_id=short_subscription,
                session_id=session_id,
                new_message=new_message,
            )
        ) as agen:
            async for event in agen:
                events.append(event)

        final_output = None
        for event in events:
            if event.output is not None:
                final_output = event.output

        session = await session_service.get_session(
            app_name="expense_agent",
            user_id=short_subscription,
            session_id=session_id,
        )
        state = session.state if session else {}

        if state.get("security_alert") == "Prompt injection detected":
            raise HTTPException(
                status_code=400,
                detail="Notice that the SSN is fully redacted in the description, the security warning is raised, the LLM is bypassed, and the workflow is paused awaiting your review decision.",
            )

        logger.info(
            "Pub/Sub execution complete for session %s. Final Output: %s",
            session_id,
            final_output,
        )
        return {"status": "success", "session_id": session_id, "output": final_output}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("Error processing Pub/Sub message: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Agent processing failed: {e!s}"
        ) from e


@app.post("/feedback")
def collect_feedback(feedback: Feedback) -> dict[str, str]:
    """Collect and log feedback.

    Args:
        feedback: The feedback data to log

    Returns:
        Success message
    """
    logger.info("Feedback received: %s", feedback.model_dump())
    return {"status": "success"}


# Main execution serving on port 8080
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
