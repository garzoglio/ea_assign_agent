# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google.adk.agents.llm_agent import LlmAgent
# Import SequentialAgent to orchestrate the parallel and merge steps
from google.adk.agents.sequential_agent import SequentialAgent
# Use InMemoryRunner for local testing/prototyping
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types
from dotenv import load_dotenv

# Load environment variables from a .env file in the same directory.
load_dotenv()

# --- Configuration ---
APP_NAME = "ea_assign_planner"
USER_ID = "ea_manager"
SESSION_ID = "plan_ea_assignment"
GEMINI_MODEL = "gemini-2.0-flash"


# --8<-- [start:init]
# Part of agent.py --> Follow https://google.github.io/adk-docs/get-started/quickstart/ to learn the setup
# --- 1. Define Planner Agents to run first ---

# Researcher 1: Renewable Energy
planner_agent = LlmAgent(
    name="EAAssignmentPlanner",
    model=GEMINI_MODEL,
    instruction="""- You are a manager for the Enterprise Architecture (EA) team. A user can present you with questions related to the team or with a new opportunity for assignment.
- When presented with a new opportunity, you want to plan the list of the questions necessary to find the best fit for the job from the EA team. Ideally the user should provide the name of the account, the target number of hours per week for the assignment, the duration of the assignment, optionally the timezone of the account.
    - The list of questions to ask to gather all the knowledge required to make an assignment are below. The key parameters provided by the user are written in all capital letters. If the ACCOUNT_TIMEZONE is not specified, leave out question (4).
        - 1) What are the detailed information for the 7 least busy members of the team over the next NUMBER_OF_WEEKS?
        - 2) Add NUMBER_OF_HOURS_PER_WEEK to the assignments for each week of each team member
        - 3) Did anyone from the team already worked for ACCOUNT_NAME over the past 2 years?
        - 4) Who lives within 1 hour of the ACCOUNT_TIMEZONE?
- Present the results only providing the questions without any introduction or closure information. For example, for a question like "I have an opportunity with Motorola for 8h/w for 3 weeks in the central timezone", you shoould return the following
- "
- 1) What are the detailed information for the 7 least busy members of the team over the next 3 weeks?
- 2) Add 8 hours per week to the assignments for each week of each team member
- 3) Did anyone from the team already worked for Motorola over the past 2 years?
- 4) Who lives within 1 hour of the central timezone?
- "
""",
    description="Plan the questions to determine the assignment of an Enterprise Architect (EA) team members to an opportunity",
    tools=[],
    # Store result in state for the merger agent
    output_key="ea_assignment_plan_result"
)

# Since we only have one agent, we can set it as the root agent directly
# without needing a SequentialAgent or ParallelAgent to orchestrate it.
root_agent = planner_agent
# --8<-- [end:init]

# --- 5. Running the Agent (Using InMemoryRunner for local testing) This works in Notebooks and script file ---

# Use InMemoryRunner: Ideal for quick prototyping and local testing
runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
print(f"InMemoryRunner created for agent '{root_agent.name}'.")

# Session creation is moved into the async helper function below.


async def call_planner_agent(query: str, user_id: str, session_id: str):
    """
    Helper async function to call the planner agent.
    Prints the final response.
    """
    # The session must be created within an async context before running the agent.
    session_service = runner.session_service
    await session_service.create_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )
    print(f"Session '{session_id}' created for direct run.")

    print(f"\n--- Running Planner Agent for query: '{query}' ---")
    content = types.Content(role='user', parts=[types.Part(text=query)])
    final_response_text = None

    print("Starting agent...")
    try:
        async for event in runner.run_async(
            user_id=user_id, session_id=session_id, new_message=content
        ):
            author_name = event.author or "System"
            is_final = event.is_final_response()
            print(f"  [Event] From: {author_name}, Final: {is_final}")  # Basic event logging

            # Check if it's the final response from our agent
            if is_final and author_name == planner_agent.name and event.content and event.content.parts:
                 final_response_text = event.content.parts[0].text.strip()
                 print(f"\n<<< Final Response from {author_name}:\n{final_response_text}")
                 # We can break after the final response from our single agent.
                 break

            elif event.is_error():
                 print(f"  -> Error from {author_name}: {event.error_message}")

        if final_response_text is None:
             print("<<< Pipeline finished but did not produce the expected final text response.")

    except Exception as e:
        print(f"\n❌ An error occurred during agent execution: {e}")



initial_trigger_query = "I have an opportunity with motorola over the next 2 weeks for 10h/w in the central timezone."

# # In Colab/Jupyter:
# await call_planner_agent(initial_trigger_query, user_id=USER_ID, session_id=SESSION_ID)

# In a standalone Python script or if await is not supported/failing:
import asyncio
asyncio.run(call_planner_agent(initial_trigger_query, user_id=USER_ID, session_id=SESSION_ID))