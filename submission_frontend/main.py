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

import os
import re
import json
import logging
from pathlib import Path
from typing import Optional

import google.auth
import vertexai
from google.adk.sessions.vertex_ai_session_service import VertexAiSessionService
from google.cloud import aiplatform_v1beta1
from vertexai.preview.reasoning_engines import ReasoningEngine
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Manager Dashboard Service")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dashboard")

# Read GCP configuration from environment variables
PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION", "us-east1")
AGENT_RUNTIME_ID = os.environ.get("AGENT_RUNTIME_ID")

# Fallback project detection using google auth
if not PROJECT_ID:
    try:
        _, PROJECT_ID = google.auth.default()
    except Exception:
        PROJECT_ID = "shruti-499518"

# Fallback agent runtime ID using the project's deployed engine
if not AGENT_RUNTIME_ID:
    AGENT_RUNTIME_ID = "projects/1046977482237/locations/us-east1/reasoningEngines/3854975727912878080"

logger.info(f"Using Project ID: {PROJECT_ID}")
logger.info(f"Using Region/Location: {LOCATION}")
logger.info(f"Using Agent Runtime ID: {AGENT_RUNTIME_ID}")

# Initialize Vertex AI SDK
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Extract numeric engine id from the resource path for Session Service
numeric_id = AGENT_RUNTIME_ID
if AGENT_RUNTIME_ID and "/" in AGENT_RUNTIME_ID:
    match = re.search(r'reasoningEngines/(\d+)', AGENT_RUNTIME_ID)
    if match:
        numeric_id = match.group(1)
    else:
        numeric_id = AGENT_RUNTIME_ID.split("/")[-1]

logger.info(f"Extracted Numeric Engine ID: {numeric_id}")

# Initialize Session Service to query and traverse session histories
session_service = VertexAiSessionService(
    project=PROJECT_ID,
    location=LOCATION,
    agent_engine_id=numeric_id
)

# Dynamic Patch to address the google-cloud-aiplatform dynamic registration bug
# When AdkApp exposes "async" methods (e.g. async_create_session), it raises an
# "Unsupported api mode: async" ValueError, which blocks client-side SDK method registration.
if not hasattr(ReasoningEngine, "query"):
    logger.info("Bypassing client-side SDK bug: Patching ReasoningEngine.query dynamically...")
    def manual_query(self, **kwargs):
        # Call the streaming endpoint since AdkApp executes query through stream_query
        response_stream = self.execution_api_client.stream_query_reasoning_engine(
            request={
                "name": self.resource_name,
                "input": kwargs,
                "class_method": "stream_query",
            }
        )
        return list(response_stream)
    ReasoningEngine.query = manual_query


class ActionRequest(BaseModel):
    interrupt_id: str
    approved: bool


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the manager dashboard HTML page."""
    html_path = Path(__file__).parent / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard UI not found.")
    return html_path.read_text(encoding="utf-8")


@app.get("/api/pending")
async def get_pending_approvals():
    """
    Queries all sessions, traverses their histories, and identifies unresolved
    adk_request_input function calls. Returns session ID, interrupt ID, and expense payload.
    Supports falling back to local SQLite session database if Vertex AI fails.
    """
    try:
        sessions = []
        use_sqlite = False
        sqlite_service = None

        # Try Vertex AI first, if configured and active
        try:
            if not AGENT_RUNTIME_ID or "projects/unknown" in AGENT_RUNTIME_ID:
                raise Exception("Agent runtime unconfigured, skipping to SQLite fallback.")
            list_response = await session_service.list_sessions(app_name=numeric_id)
            sessions = list_response.sessions
        except Exception as ve:
            logger.info("Vertex AI Session Service failed or unconfigured. Falling back to local SQLite...")
            db_path = Path(__file__).parent.parent / "ambient-expense-submission" / "ambient-expense-agent" / ".adk" / "session.db"
            if not db_path.exists():
                db_path = Path(__file__).parent.parent / "ambient-expense-agent" / ".adk" / "session.db"
            
            if db_path.exists():
                logger.info(f"Using local SQLite DB path: {db_path}")
                from google.adk.sessions.sqlite_session_service import SqliteSessionService
                sqlite_service = SqliteSessionService(db_path=str(db_path))
                list_response = await sqlite_service.list_sessions(app_name="expense_agent")
                sessions = list_response.sessions
                use_sqlite = True
            else:
                logger.error(f"Local SQLite database not found at: {db_path}")
                return []
        
        pending_approvals = []
        for session in sessions:
            try:
                # Fetch full session detail with chronological events
                if use_sqlite and sqlite_service:
                    session_detail = await sqlite_service.get_session(
                        app_name="expense_agent",
                        user_id=session.user_id,
                        session_id=session.id
                    )
                else:
                    session_detail = await session_service.get_session(
                        app_name=numeric_id,
                        user_id=session.user_id,
                        session_id=session.id
                    )

                if not session_detail:
                    continue
                
                events = session_detail.events
                pending_found = False
                for idx in range(len(events) - 1, -1, -1):
                    if pending_found:
                        break
                    event = events[idx]
                    fcs = event.get_function_calls()
                    for fc in fcs:
                        if fc.name == "adk_request_input":
                            interrupt_id = fc.id
                            
                            # Check if a matching FunctionResponse exists in the session
                            resolved = False
                            for other_event in events:
                                frs = other_event.get_function_responses()
                                if any(fr.name == "adk_request_input" and fr.id == interrupt_id for fr in frs):
                                    resolved = True
                                    break
                            
                            if not resolved:
                                # Retrieve closest preceding expense details (amount, description)
                                amount = 0.0
                                description = "No description provided"
                                
                                for prev_event in reversed(events[:idx]):
                                    if prev_event.output:
                                        if isinstance(prev_event.output, dict):
                                            if "amount" in prev_event.output and "description" in prev_event.output:
                                                amount = float(prev_event.output["amount"])
                                                description = str(prev_event.output["description"])
                                                break
                                        elif hasattr(prev_event.output, "amount") and hasattr(prev_event.output, "description"):
                                            amount = float(getattr(prev_event.output, "amount"))
                                            description = str(getattr(prev_event.output, "description"))
                                            break
                                            
                                pending_approvals.append({
                                    "session_id": session.id,
                                    "user_id": session.user_id,
                                    "interrupt_id": interrupt_id,
                                    "amount": amount,
                                    "description": description
                                })
                                pending_found = True
                                break
            except Exception as se:
                logger.error(f"Error checking session {session.id}: {se}")
                continue
                
        return pending_approvals
    except Exception as e:
        logger.exception("Error listing pending approvals")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/action/{session_id}")
async def resume_session(session_id: str, request: ActionRequest):
    """
    Resumes a paused session on Agent Runtime or local uvicorn backend.
    """
    try:
        use_local_run = False
        
        # Try remote Vertex AI first if configured
        try:
            if not AGENT_RUNTIME_ID or "projects/unknown" in AGENT_RUNTIME_ID or "3854975727912878080" not in AGENT_RUNTIME_ID:
                raise Exception("Agent runtime unconfigured, switching to local uvicorn run.")
                
            remote_app = ReasoningEngine(AGENT_RUNTIME_ID)
            
            # Build the exact resume payload expected by the ADK runner
            resume_payload = {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "id": request.interrupt_id,
                            "name": "adk_request_input",
                            "response": {
                                "approved": request.approved
                            }
                        }
                    }
                ]
            }
            
            logger.info(f"Resuming session {session_id} on engine {AGENT_RUNTIME_ID}")
            # Run query with user_id strictly set to default-user
            response_stream = remote_app.query(
                message=resume_payload,
                user_id="default-user",
                session_id=session_id
            )
            events = list(response_stream)
        except Exception as e:
            logger.info("Vertex AI run failed or bypassed. Resuming session locally via local agent API...")
            use_local_run = True

        if use_local_run:
            import requests
            
            # Find the user_id from SQLite session database
            db_path = Path(__file__).parent.parent / "ambient-expense-submission" / "ambient-expense-agent" / ".adk" / "session.db"
            if not db_path.exists():
                db_path = Path(__file__).parent.parent / "ambient-expense-agent" / ".adk" / "session.db"
                
            user_id = "pubsub-caller"
            if db_path.exists():
                import sqlite3
                try:
                    with sqlite3.connect(str(db_path)) as conn:
                        conn.row_factory = sqlite3.Row
                        cursor = conn.cursor()
                        cursor.execute("SELECT user_id FROM sessions WHERE id=?", (session_id,))
                        row = cursor.fetchone()
                        if row:
                            user_id = row["user_id"]
                except Exception as se:
                    logger.error(f"Error querying SQLite for user_id: {se}")
            
            resume_payload = {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "id": request.interrupt_id,
                            "name": "adk_request_input",
                            "response": {
                                "approved": request.approved
                            }
                        }
                    }
                ]
            }
            
            run_url = "http://127.0.0.1:8080/run"
            run_payload = {
                "appName": "expense_agent",
                "userId": user_id,
                "sessionId": session_id,
                "newMessage": resume_payload
            }
            
            logger.info(f"POSTing resume query to local agent: {run_url}")
            resp = requests.post(run_url, json=run_payload)
            resp.raise_for_status()
            events = resp.json()

        # Process returned events to extract final compliance remarks and status
        final_message = ""
        status = "approved" if request.approved else "rejected"
        
        for chunk in events:
            # Handle both local JSON dicts and remote chunk objects
            evt = chunk if isinstance(chunk, dict) else {}
            if not evt and hasattr(chunk, "data") and chunk.data:
                try:
                    data_str = chunk.data.decode("utf-8") if isinstance(chunk.data, bytes) else str(chunk.data)
                    for line in data_str.strip().split("\n"):
                        if line:
                            evt = json.loads(line)
                            break
                except Exception:
                    pass
            
            if not evt:
                continue

            # Extract error conditions
            if evt.get("error_message") or evt.get("error_code"):
                final_message = evt.get("error_message") or evt.get("error_code")
                status = "error"
                
            # Extract structured status/message output
            out = evt.get("output")
            if out and isinstance(out, dict):
                if "status" in out:
                    status = out["status"]
                if "message" in out:
                    final_message = out["message"]
                    
            # Fallback: extract model response text
            content = evt.get("content")
            if content and isinstance(content, dict):
                parts = content.get("parts")
                if parts and isinstance(parts, list):
                    for p in parts:
                        if isinstance(p, dict) and p.get("text"):
                            final_message = p.get("text")
                    
        if not final_message:
            action_word = "approved" if request.approved else "rejected"
            final_message = f"Expense has been successfully {action_word}."
            
        return {
            "status": status,
            "message": final_message,
            "session_id": session_id,
            "interrupt_id": request.interrupt_id
        }
    except Exception as e:
        logger.exception(f"Error taking action on session {session_id}")
        raise HTTPException(status_code=500, detail=str(e))

