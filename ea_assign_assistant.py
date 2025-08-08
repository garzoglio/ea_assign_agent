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

import google.auth
import google.auth.transport.requests
import google.oauth2.id_token
from google.adk.agents.llm_agent import LlmAgent
# Use InMemoryRunner for local testing/prototyping
from google.adk.runners import InMemoryRunner
from google.adk.tools.openapi_tool import OpenAPIToolset
from fastapi.openapi.models import HTTPBearer
from google.adk.auth.auth_credential import (
    AuthCredential,
    AuthCredentialTypes,
    ServiceAccount,
    ServiceAccountCredential,
)
from google.genai import types
from dotenv import load_dotenv
import logging
import os
import httpx
import asyncio
import json
import uuid

# Enable detailed logging for the HTTP client to see request headers.
# This is useful for debugging authentication issues.
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.DEBUG)

# Load environment variables from a .env file in the same directory.
load_dotenv()

# --- Configuration ---
APP_NAME = "ea_assign_agent"
USER_ID = "ea_manager"
SESSION_ID = "get_ea_team_info"
GEMINI_MODEL = "gemini-2.0-flash"


# --8<-- [start:init]
# --- 1. Define Planner Agent (not used in this flow, but kept for future use) ---
planner_agent = LlmAgent(
    name="EAAssignmentPlanner",
    model=GEMINI_MODEL,
    instruction="""- You are a manager for the Enterprise Architecture (EA) team. A user can present you with questions related to the team or with a new opportunity for assignment.
- When presented with a new opportunity, you want to plan the list of the questions necessary to find the best fit for the job from the EA team. Ideally the user should provide the name of the account, the target number of hours per week for the assignment, the duration of the assignment, optionally the timezone of the account.
    - The list of questions to ask to gather all the knowledge required to make an assignment are below. The key parameters provided by the user are written in all capital letters. If the ACCOUNT_TIMEZONE is not specified, leave out question (4).
        - 1) What are the detailed information for the 5 least busy members of the team over the next NUMBER_OF_WEEKS?
        - 2) Add NUMBER_OF_HOURS_PER_WEEK to the assignments for each week of each team member
        - 3) Who from the team already worked for ACCOUNT_NAME over the past 2 years, if anyone?
        - 4) Who lives within 1 hour of the ACCOUNT_TIMEZONE?
- Present the results only providing the questions without any introduction or closure information. For example, for a question like "I have an opportunity with Motorola for 8h/w for 3 weeks in the central timezone", you shoould return the following
- "
- 1) What are the detailed information for the 5 least busy members of the team over the next 3 weeks?
- 2) Add 8 hours per week to the assignments for each week of each team member
- 3) Who from the team already worked for motorola over the past 2 years, if anyone?
- 4) Who lives within 1 hour of the central timezone?
- "
""",
    description="Plan the questions to determine the assignment of an Enterprise Architect (EA) team members to an opportunity",
    tools=[],
    # Store result in state for the merger agent
    output_key="ea_assignment_plan_result"
)

# --- 1b. Define Effort Update Agent ---
effort_update_agent = LlmAgent(
    name="EffortUpdateAgent",
    model=GEMINI_MODEL,
    instruction="""- You are an assistant that updates team member schedules.
- You will be given a current schedule and a request to add a specific number of hours to each weekly assignment for each person.
- Your task is to find every mention of hours in the schedule, parse the number, add the requested number of hours to it, and replace the old number with the new total.
- You must present the updated schedule in the exact same format as the input schedule.
- You will also be given a specific introductory sentence to start your response with. Use that sentence verbatim.

For example, if the request is:
"Based on the following schedule, add 8 hours to each weekly assignment for each person."
Present the result in the same format, starting with the phrase: "Adding an extra 8h/w over the next 3 weeks would result in the following schedule for the 5 least busy team members:"

Current Schedule:
OK. Here are the details for the 5 least busy team members over the next 3 weeks:

*   **Abhilash Thumma**: Week of August 16, 2025: 18 hours across 3 projects (located in America/Chicago)
*   **Hadrian Knotz**: (located in America/Los_Angeles)
    *   Week of August 16, 2025: 40 hours across 3 projects 
    *   Week of August 23, 2025: 8 hours across 2 projects
*   **Sundar Mudupalli**: (located in America/Los_Angeles)
    *   Week of August 16, 2025: 25.6 hours across 3 projects 

Your output should be:
"Adding an extra 8h/w over the next 3 weeks would result in the following schedule for the 5 least busy team members:

*   **Abhilash Thumma**: Week of August 16, 2025: 26 hours across 3 projects (located in America/Chicago)
*   **Hadrian Knotz**: (located in America/Los_Angeles)
    *   Week of August 16, 2025: 48 hours across 3 projects 
    *   Week of August 23, 2025: 16 hours across 2 projects
*   **Sundar Mudupalli**: (located in America/Los_Angeles)
    *   Week of August 16, 2025: 33.6 hours across 3 projects"
""",
    description="Adds a given number of hours to each team member's weekly schedule and formats the output.",
    tools=[],
    output_key="effort_update_result",
)

# --- 1c. Define Recommender Agent ---
recommender_agent = LlmAgent(
    name="EARecommenderAgent",
    model=GEMINI_MODEL,
    instruction="""- You are a senior manager for the Enterprise Architecture (EA) team.
- Your task is to recommend the best candidates for a new assignment based on several factors.
- You will be provided with the following information:
    1.  An updated schedule showing team members' availability after adding the new assignment's hours i.e. you do not have to add the aditional hours anymore, as the projected number of hours are already included in your input.
    2.  A list of team members who have previously worked for the account.
    3.  A list of team members who are in a compatible timezone.
- Based on this information, you must provide a ranked list of the top 3-5 candidates.
- For each candidate, provide a brief justification for their ranking. Their availability is the main priority with lower hours being better; followed by previous experience with the account; followed by timezone compatibility, if provided. Don't include considerations around timezone if it is not provided in your input.
- Present the final recommendation in a clear, easy-to-read format.

- Example 1:
Input:
"Based on the following information, please recommend the best candidates for the assignment.

1. Updated Schedule:
Here are the details for the 5 least busy team members over the next 3 weeks:

Abhilash Thumma, located in the America/Chicago timezone, is assigned to 3 projects for a total of 26 hours for the week ending August 16, 2025.

Hadrian Knotz, located in the America/Los_Angeles timezone, is assigned to 3 projects for a total of 48 hours for the week ending August 16, 2025. He is then assigned to 2 projects for 16 hours each for the weeks ending August 23, 2025, and August 30, 2025.

Kyle Romell, located in the America/Chicago timezone, is assigned to 3 projects for a total of 34 hours for the week ending August 16, 2025. He is then assigned to 1 project for 28 hours for the week ending August 23, 2025.

Laurence Chiu, located in the America/Toronto timezone, is assigned to 2 projects for a total of 20 hours for the week ending August 16, 2025. He is then assigned to 3 projects for 48 hours for the week ending August 23, 2025, and to 2 projects for 20 hours the week ending August 30, 2025.

Valavan Rajakumar, located in the America/Toronto timezone, is assigned to 1 project for a total of 16 hours for the week ending August 16, 2025.

2. Past Involvement with the Account:
Over the past two years, Abhilash Thumma and Gabriele Garzoglio have worked for Motorola Solutions, Inc.

3. Timezone Compatibility:
The following team members live within one hour of the central timezone: Robin Aubrey, Carl Franklin, Alex Maclinovsky, Gabriele Garzoglio, Kyle Romell, Abhilash Thumma, Andrés Olarte, Creighton Swank, Will Weber, Eric Poon, Harish Murthy, Steve Kluger, Kevin Winters, Parag Doshi, Michelle Sollicito, Laurence Chiu, Valavan Rajakumar, and Ben Swenka.

Provide a ranked list of the top candidates with your reasoning.
"

Output:
"Here's my recommendation for the top candidates for the new assignment, ranked by suitability:

**Top Candidates:**

1.  **Abhilash Thumma:**
    *   **Justification:** Abhilash has experience working with Motorola Solutions, which is a significant advantage. His workload for the week ending Aug 16, is projected to be 26 hours. He is also timezone compatible.

2.  **Valavan Rajakumar:**
    *   **Justification:** Valavan has the lightest workload for the week ending Aug 16, 2025, at 16 hours. Valavan is timezone compatible but lacks prior experience with the account.

3.  **Kyle Romell:**
    *   **Justification:** Kyle has a manageable workload for the week ending Aug 16 and Aug 23, 2025, at 34 and 28 hours respectively, and he's in a compatible timezone. While he doesn't have prior experience with Motorola Solutions, his availability makes him a good option.

4.  **Laurence Chiu:**
    *   **Justification:** Laurence has the second lightest workload for the week ending Aug 16, 2025, at 20 hours, but he becomes busy on Aug 23 with 48h. He is also timezone compatible, but lacks prior experience with the account.

**Summary Table:**

| Rank | Candidate          | Account Exp.| Close Timezone| 8/16 Hours | 8/23 Hours | 8/30 Hours | Rationale                                                                               |
| ---- | ------------------ | ----------- | ------------- | ---------- | ---------- | ---------- | --------------------------------------------------------------------------------------- |
| 1    | Abhilash Thumma    | Yes         | Yes           | 26         |  0         |  0         | Prior experience with Motorola, good availability, and timezone compatible.             |
| 2    | Valavan Rajakumar  | No          | Yes           | 16         |  0         |  0         |  Very light workload, timezone compatible, but no prior experience with Motorola.       |
| 3    | Kyle Romell        | No          | Yes           | 34         | 28         |  0         |  Manageable workload, timezone compatible, but no prior experience with Motorola.       |
| 4    | Laurence Chiu      | No          | Yes           | 20         | 48         | 20         |  Light workload, timezone compatible, but no prior experience with Motorola.            |"


- Example 2:
Input:
"Based on the following information, please recommend the best candidates for the assignment.

1. Updated Schedule:
Adding 20 hours per week to the assignments for each week of each team member would result in:

Here are the 5 least busy members of the team for the week ending August 16, 2025:

*   Abhilash Thumma, located in America/Chicago, is assigned to 38 hours across 3 projects.
*   Kyle Romell, located in America/Chicago, is assigned to 46 hours across 3 projects.
*   Laurence Chiu, located in America/Toronto, is assigned to 32 hours across 2 projects.
*   Sundar Mudupalli, located in America/Los\_Angeles, is assigned to 45.6 hours across 3 projects.
*   Valavan Rajakumar, located in America/Toronto, is assigned to 28 hours across 1 project.

2. Past Involvement with the Account:
Over the past two years, Hadrian Knotz, Ben Swenka, Laurence Chiu, Harish Murthy, Alex Maclinovsky, Kevin Winters, Blake Dubois, and Carl Franklin have been assigned to Paypal Holdings, Inc.

3. Timezone Compatibility:
The following team members live within one hour of the Pacific timezone: Will Weber, Ravinder Lota, Hadrian Knotz, Sundar Mudupalli, and Wilton Wong.

Provide a ranked list of the top candidates with your reasoning."

Output:
"Here's my recommendation for the top candidates for the new assignment, ranked by suitability:

**Top Candidates:**

1.  **Laurence Chiu:**
    *   **Justification:** Laurence has experience working with Paypal, which is a significant advantage. He is projected to work 32 hours but he is not timezone compatible.

2.  **Valavan Rajakumar:**
    *   **Justification:** Valavan is projected to work only 28 hours. He is also not timezone compatible and lacks prior experience with the account.

3.  **Abhilash Thumma:**
    *   **Justification:** Abhilash is not timezone compatible and is projected to work 38 hours. He lacks prior experience with the account.

4.  **Sundar Mudupalli:**
    *   **Justification:** Sundar is timezone compatible, but he is projected to work 45.6 hours, beyond a regular 40 hours work week.

5.  **Kyle Romell:**
    *   **Justification:** Kyle is not timezone compatible and he is projected to work 46 hours, beyond a regular 40 hours work week.



**Summary Table:**

| Rank | Candidate          | Account Exp.| Close Timezone| Hours | Rationale                                                                               |
| ---- | ------------------ | ----------- | ------------- | ---------- | --------------------------------------------------------------------------------------- |
| 4    | Laurence Chiu      | Yes         | No            | 32         |  Prior experience with Paypal, timezone compatible, but projected to work 32 hours.            |
| 5    | Valavan Rajakumar  | No          | No            | 28         |  Projected to work 28 hours, timezone compatible, but no prior experience with Motorola.            |"
| 2    | Abhilash Thumma    | No          | No            | 38         |  Timezone compatible, but projected to work 38 hours.       |
| 1    | Sundar Mudupalli   | No          | Yes           | 45.6       |  Timezone compatible, but projected to work 45.6 hours.             |
| 3    | Kyle Romell        | No          | No            | 46         |  Timezone compatible, but projected to work 46 hours.       |

""",
    description="Recommends the best EA team members for an assignment based on availability, past experience, and timezone.",
    tools=[],
    output_key="recommender_result",
)




# --- 4. Set the root agent to run ---
# We are testing the planner_agent agent, so we set it as the root agent.
root_agent = planner_agent
# --8<-- [end:init]

# --- 5. Running the Agent (Using InMemoryRunner for local testing) This works in Notebooks and script file ---

# Use InMemoryRunner: Ideal for quick prototyping and local testing
runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
print(f"InMemoryRunner created for agent '{root_agent.name}'.")

# Session creation is moved into the async helper function below.


async def call_ea_team_info_agent(query: str, user_id: str, session_id: str):
    """
    Helper async function to call the EATeamInfo agent.
    Prints the final response.
    """
    # The session must be created within an async context before running the agent.
    session_service = runner.session_service
    await session_service.create_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )
    print(f"Session '{session_id}' created for direct run.")

    print(f"\n--- Running EATeamInfo Agent for query: '{query}' ---")
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
                 # We can break after the final response from our single agent.
                 break

            elif event.error_code is not None:
                 error_message = event.error.message if event.error else "Unknown error"
                 if event.error and event.error.details:
                     error_message += f"\n     Details: {event.error.details}"
                 print(f"  -> ❌ Error from {author_name}: {error_message}")

        if final_response_text is None:
             print("<<< Pipeline finished but did not produce the expected final text response.")
        else:
            print(f"\n<<< Raw Final Response from {planner_agent.name}:\n{final_response_text}")

            # Parse the multi-line response into a dictionary
            lines = [line.strip() for line in final_response_text.split('\n') if line.strip()]

            cleaned_lines = []
            for line in lines:
                if line.startswith('- '):
                    cleaned_lines.append(line[2:].strip())
                else:
                    cleaned_lines.append(line)

            keys = ["ask_availability", "ask_total_effort", "ask_past_involvement", "ask_timezone"]
            plan_dict = dict.fromkeys(keys)

            for i, line in enumerate(cleaned_lines):
                if i < len(keys):
                    plan_dict[keys[i]] = line

            print("\n--- Parsed Plan Dictionary ---")
            print(json.dumps(plan_dict, indent=2))

            # --- New section for parallel API calls ---
            print("\n--- Invoking backend APIs in parallel to gather data ---")

            # The base URL for the Dialogflow detectIntent API.
            # A new session ID will be generated for each invocation.
            # DF Agent: EA Info Assistant Task
            #  Playbook: EA Info Assistant Task Playbook
            api_base_url = "https://us-central1-dialogflow.googleapis.com/v3/projects/test-project-26133-466015/locations/us-central1/agents/4812b694-5a2a-4b40-8736-8224627b7e80"

            # The project ID to use for quota and billing. This is required when using
            # user credentials (Application Default Credentials) to call Google Cloud APIs.
            quota_project_id = "test-project-26133-466015"

            try:
                # 1. Get OAuth2 access token using Application Default Credentials.
                # Standard googleapis.com endpoints use OAuth2 access tokens for authentication,
                # unlike the previous Cloud Run endpoint which used OIDC ID tokens.
                # google.auth.default() finds the credentials, and credentials.refresh()
                # gets a valid access token.
                print("Fetching access token for Google Cloud API...")
                credentials, project_id = google.auth.default()
                auth_req = google.auth.transport.requests.Request()
                credentials.refresh(auth_req)
                access_token = credentials.token
                if not access_token:
                    raise Exception("Failed to obtain access token.")
                print("Access token fetched successfully.")

                headers = {
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                    "x-goog-user-project": quota_project_id,
                }

                # 2. Prepare parallel API calls for the relevant questions
                queries_to_run = {
                    "availability": plan_dict.get("ask_availability"),
                    "past_involvement": plan_dict.get("ask_past_involvement"),
                    "timezone_match": plan_dict.get("ask_timezone"),
                }

                tasks = []
                # Use a single client session for all requests
                async with httpx.AsyncClient(headers=headers, timeout=90.0) as client:
                    for key, q in queries_to_run.items():
                        if q:
                            # Generate a new random session ID for each call to treat them as separate conversations.
                            dialogflow_session_id = str(uuid.uuid4())
                            api_endpoint = f"{api_base_url}/sessions/{dialogflow_session_id}:detectIntent"
                            print(
                                f"Using Dialogflow session ID {dialogflow_session_id} for task '{key}'"
                            )

                            # The Dialogflow detectIntent API expects a specific JSON payload.
                            json_payload = {
                                "queryInput": {
                                    "text": {"text": q},
                                    "languageCode": "en",
                                },
                                "queryParams": {
                                    "timeZone": "America/Chicago"
                                },
                            }
                            task = asyncio.create_task(
                                client.post(api_endpoint, json=json_payload), name=key
                            )
                            tasks.append(task)

                    if tasks:
                        # 3. Execute calls in parallel and wait for all to complete
                        print(f"Executing {len(tasks)} API calls in parallel...")
                        responses = await asyncio.gather(*tasks, return_exceptions=True)
                        print("All API calls completed.")

                        # 4. Process and categorize all responses
                        print("\n--- API Call Results ---")
                        success_results = {}
                        failures = []
                        for i, response in enumerate(responses):
                            task_name = tasks[i].get_name()
                            if isinstance(response, Exception):
                                failures.append(f"  - Task '{task_name}' failed with exception: {response}")
                                continue
                            try:
                                response.raise_for_status()
                                json_response = response.json()
                                success_results[task_name] = json_response
                            except httpx.HTTPStatusError as e:
                                failures.append(f"  - Task '{task_name}' failed with HTTP status {e.response.status_code}: {e.response.text}")
                            except json.JSONDecodeError:
                                failures.append(f"  - Task '{task_name}' failed with JSON Decode Error. Response: {response.text}")

                        # Print all results, starting with successes
                        for task_name, result in success_results.items():
                            print(f"✅ Result for '{task_name}':")
                            print(json.dumps(result, indent=2))

                        for failure_message in failures:
                            # The failure message already has the task name and details.
                            print(f"❌{failure_message}")

                        # 5. Abort if there were any failures, otherwise extract answers.
                        if failures:
                            print("\n❌ One or more API calls failed. Aborting before final synthesis.")
                            raise Exception("Aborting due to critical API call failures.")

                        # This part only runs if all calls were successful.
                        extracted_answers = {}
                        for task_name, json_response in success_results.items():
                            # Safely extract the text from the nested response message
                            answer_text = json_response.get("queryResult", {}).get("responseMessages", [{}])[0].get("text", {}).get("text", [""])[0]
                            extracted_answers[task_name] = answer_text.strip() if answer_text else "No answer text found in response."
                        
                        print("\n--- Extracted Answers from APIs ---")
                        print(json.dumps(extracted_answers, indent=2))

                        # 6. Run Effort Update Agent if applicable
                        ask_total_effort_str = plan_dict.get("ask_total_effort")
                        ask_availability_str = plan_dict.get("ask_availability")
                        availability_result_str = extracted_answers.get("availability")

                        if ask_total_effort_str and ask_availability_str and availability_result_str:
                            print("\n--- Running Effort Update Agent ---")

                            # Use the planner's instruction as the core of the prompt, and combine it
                            # with the formatting instructions and the data for the agent.
                            update_effort_query = f"""{ask_total_effort_str}

Current Schedule:
{availability_result_str}
"""
                            effort_runner = InMemoryRunner(agent=effort_update_agent, app_name=APP_NAME)
                            effort_session_id = f"effort-update-{session_id}"
                            await effort_runner.session_service.create_session(
                                app_name=APP_NAME, user_id=user_id, session_id=effort_session_id
                            )

                            effort_content = types.Content(role='user', parts=[types.Part(text=update_effort_query)])
                            effort_response_text = None

                            async for event in effort_runner.run_async(
                                user_id=user_id, session_id=effort_session_id, new_message=effort_content
                            ):
                                if event.is_final_response() and event.author == effort_update_agent.name and event.content and event.content.parts:
                                    effort_response_text = event.content.parts[0].text.strip()
                                    break
                            
                            if effort_response_text:
                                print(f"\n--- Updated Effort from {effort_update_agent.name} ---")
                                print(effort_response_text)

                                # 7. Run Recommender Agent
                                print("\n--- Running Recommender Agent ---")

                                past_involvement_str = extracted_answers.get("past_involvement", "No information available.")
                                timezone_match_str = extracted_answers.get("timezone_match", "No information available.")

                                recommender_query = f"""Based on the following information, please recommend the best candidates for the assignment.

1. Updated Schedule:
{effort_response_text}

2. Past Involvement with the Account:
{past_involvement_str}

3. Timezone Compatibility:
{timezone_match_str}

Provide a ranked list of the top candidates with your reasoning.
"""
                                print(f"\n--- Input to the Final Recommendation from {recommender_agent.name} ---")
                                print(f"\n{recommender_query}")

                                recommender_runner = InMemoryRunner(agent=recommender_agent, app_name=APP_NAME)
                                recommender_session_id = f"recommender-{session_id}"
                                await recommender_runner.session_service.create_session(
                                    app_name=APP_NAME, user_id=user_id, session_id=recommender_session_id
                                )

                                recommender_content = types.Content(role='user', parts=[types.Part(text=recommender_query)])
                                recommender_response_text = None

                                async for event in recommender_runner.run_async(
                                    user_id=user_id, session_id=recommender_session_id, new_message=recommender_content
                                ):
                                    if event.is_final_response() and event.author == recommender_agent.name and event.content and event.content.parts:
                                        recommender_response_text = event.content.parts[0].text.strip()
                                        break

                                if recommender_response_text:
                                    print(f"\n--- Final Recommendation from {recommender_agent.name} ---")
                                    print(recommender_response_text)
                                else:
                                    print("<<< Recommender agent did not produce a response.")
                            else:
                                print("<<< Effort update agent did not produce a response.")
                    else:
                        print("No questions to ask the backend API.")

            except Exception as e:
                print(f"\n❌ An error occurred during API invocation: {e}")

    except Exception as e:
        print(f"\n❌ An error occurred during agent execution: {e}")


# Test sets
initial_trigger_query = "I have an opportunity with motorola for 8h/w for 3 weeks in the central timezone. Who are the best EA candidate for assignment?" 
#initial_trigger_query = "I have an opportunity with motorola for 8h/w for 3 weeks. Who are the best EA candidate for assignment?" 
#initial_trigger_query = "I have an opportunity with Paypal for 20h/w for next week. Paypal is in the pacific timezone. Who are the best EA candidate for assignment?" 

# # In Colab/Jupyter:
# await call_ea_team_info_agent(initial_trigger_query, user_id=USER_ID, session_id=SESSION_ID)

# In a standalone Python script or if await is not supported/failing:
asyncio.run(call_ea_team_info_agent(initial_trigger_query, user_id=USER_ID, session_id=SESSION_ID))