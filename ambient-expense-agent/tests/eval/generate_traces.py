import asyncio
import json
import os
from contextlib import aclosing

from google.adk.runners import Runner
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.genai import types

from expense_agent.agent import root_agent

# Define constants
DATASET_PATH = "tests/eval/datasets/basic-dataset.json"
OUTPUT_PATH = "artifacts/traces/generated_traces.json"


async def run_scenario(runner, session_service, case):
    case_id = case["eval_id"]
    prompt_text = case["prompt"]["parts"][0]["text"]
    session_id = f"eval-{case_id}"

    # Clean existing session
    await session_service.delete_session(
        app_name="expense_agent", user_id="eval-user", session_id=session_id
    )

    new_message = types.Content(
        role="user", parts=[types.Part.from_text(text=prompt_text)]
    )

    # First run
    interrupt_id = None
    async with aclosing(
        runner.run_async(
            user_id="eval-user",
            session_id=session_id,
            new_message=new_message,
        )
    ) as agen:
        async for event in agen:
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if (
                        part.function_call
                        and part.function_call.name == "adk_request_input"
                    ):
                        interrupt_id = part.function_call.id

    # Resume if interrupted
    if interrupt_id:
        # Determine the decision
        prompt_dict = json.loads(prompt_text)
        desc = prompt_dict.get("description", "").lower()
        if "ignore" in desc or "bypass" in desc or "auto-approve" in desc:
            decision = "reject"
        else:
            decision = "approve"

        print(f"[{case_id}] Interrupted. Automating decision: {decision}")
        resume_message = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        name="adk_request_input",
                        id=interrupt_id,
                        response={"decision": decision},
                    )
                )
            ],
        )
        async with aclosing(
            runner.run_async(
                user_id="eval-user",
                session_id=session_id,
                new_message=resume_message,
            )
        ) as agen:
            async for _event in agen:
                pass

    # Read final session events
    session = await session_service.get_session(
        app_name="expense_agent", user_id="eval-user", session_id=session_id
    )
    if not session:
        raise ValueError(f"Session not found for {case_id}")

    # Map session events to eval turns
    turns = []
    current_turn = None
    for event in session.events:
        is_user_input = False
        if event.content and event.content.role == "user":
            is_user_input = True

        if is_user_input:
            if current_turn is not None:
                turns.append(current_turn)
            current_turn = {"turn_index": len(turns), "events": []}

        if current_turn is not None:
            if event.content:
                content_dict = event.content.model_dump(exclude_none=True)
            elif event.output is not None:
                text_val = (
                    json.dumps(event.output)
                    if isinstance(event.output, (dict, list))
                    else str(event.output)
                )
                content_dict = {"role": "model", "parts": [{"text": text_val}]}
            else:
                content_dict = {"role": "model", "parts": [{"text": ""}]}

            event_dict = {
                "author": event.author if event.author else "user",
                "content": content_dict,
            }
            current_turn["events"].append(event_dict)

    if current_turn is not None:
        turns.append(current_turn)

    # Extract final text response
    final_response_content = None
    for event in reversed(session.events):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    final_response_content = event.content
                    break
            if final_response_content:
                break

    if not final_response_content:
        for event in reversed(session.events):
            if event.content:
                final_response_content = event.content
                break

    responses = []
    if final_response_content:
        responses.append(
            {"response": final_response_content.model_dump(exclude_none=True)}
        )

    return {
        "eval_id": case_id,
        "prompt": case.get("prompt"),
        "responses": responses,
        "agent_data": {
            "agents": {
                "expense_approval_workflow": {
                    "agent_id": "expense_approval_workflow",
                    "instruction": "Processes, triages, and routes employee expense reports with human-in-the-loop gating.",
                },
                "llm_reviewer": {
                    "agent_id": "llm_reviewer",
                    "instruction": "Analyze compliance of high value expense reports.",
                },
            },
            "turns": turns,
        },
    }


async def main():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset {DATASET_PATH} not found.")
        return

    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    session_service = SqliteSessionService(db_path=".adk/eval_traces.db")
    runner = Runner(
        agent=root_agent,
        session_service=session_service,
        app_name="expense_agent",
        auto_create_session=True,
    )

    eval_cases_traces = []
    for case in dataset["eval_cases"]:
        print(f"Running scenario: {case['eval_id']}")
        trace_case = await run_scenario(runner, session_service, case)
        eval_cases_traces.append(trace_case)

    # Ensure target directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"eval_cases": eval_cases_traces}, f, indent=2)

    print(f"Successfully generated traces and saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
