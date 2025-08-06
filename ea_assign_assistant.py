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
*   **Hadrian Knotz**:
    *   Week of August 16, 2025: 40 hours across 3 projects (located in America/Los_Angeles)
    *   Week of August 23, 2025: 8 hours across 2 projects (located in America/Los_Angeles)
*   **Sundar Mudupalli**:
    *   Week of August 16, 2025: 25.6 hours across 3 projects (located in America/Los_Angeles)

Your output should be:
"Adding an extra 8h/w over the next 3 weeks would result in the following schedule for the 5 least busy team members:

*   **Abhilash Thumma**: Week of August 16, 2025: 26 hours across 3 projects (located in America/Chicago)
*   **Hadrian Knotz**:
    *   Week of August 16, 2025: 48 hours across 3 projects (located in America/Los_Angeles)
    *   Week of August 23, 2025: 16 hours across 2 projects (located in America/Los_Angeles)
*   **Sundar Mudupalli**:
    *   Week of August 16, 2025: 33.6 hours across 3 projects (located in America/Los_Angeles)"
""",
    description="Adds a given number of hours to each team member's weekly schedule and formats the output.",
    tools=[],
    output_key="effort_update_result",
)

# --- 2. Create a tool from the BigQuery API OpenAPI spec ---
# This tool will allow the agent to execute SQL queries.
# with open("bq_api.yaml", "r") as f:
#     bq_api_yaml_str = f.read()

# Configure authentication using Application Default Credentials (ADC)
# to generate an OIDC token for the Cloud Run service.

# try:
#     credentials, project_id = google.auth.default()
#     print(f"Using credentials type: {type(credentials)}")
#     print(f"Detected project ID: {project_id}")
#     auth_req = google.auth.transport.requests.Request()
#     audience="https://bq-api-service-kz5lpdkcca-uc.a.run.app"
#     print(f"Fetching ID token for audience: {audience}")
#     id_token = google.oauth2.id_token.fetch_id_token(auth_req, audience)
#     #headers = {"Authorization": f"Bearer {id_token}"}


# except google.auth.exceptions.DefaultCredentialsError as e:
#     print(f"Error: Could not find default credentials. Please ensure you are authenticated.")
#     print(f"If running locally, try 'gcloud auth application-default login'.")
#     print(f"If running on GCP, ensure your service has a service account with appropriate permissions.")
#     print(e)
#     raise
# except requests.exceptions.RequestException as e:
#     print(f"Error making HTTP request: {e}")
#     raise
# except Exception as e:
#     print(f"An unexpected error occurred: {e}")
#     raise

# service_account = ServiceAccount(
#     use_default_credential=True,
#     # The audience is the URL of the Cloud Run service to be invoked.
#     audience="https://bq-api-service-kz5lpdkcca-uc.a.run.app/query",
#     # Scopes are for OAuth2 access tokens. For OIDC ID tokens (needed for
#     # Cloud Run), 'audience' is used. Pass an empty list for scopes.
#     scopes=[],
# )

# auth_credential = AuthCredential(
#     auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
#     service_account=service_account,
# )

# auth_scheme = HTTPBearer(bearerFormat="JWT")

#service_account_auth_credential = ServiceAccountCredential(
#    audience="https://bq-api-service-kz5lpdkcca-uc.a.run.app"
#)

# auth_scheme, auth_credential = service_account_dict_to_scheme_credential(
#     config=service_account_cred,
#     scopes=["https://www.googleapis.com/auth/cloud-platform"],
# )

# bq_tool = OpenAPIToolset(
#     spec_str=bq_api_yaml_str,
#     spec_str_type="yaml",
#     auth_scheme=auth_scheme,
#     auth_credential=auth_credential,
# )

# # --- 3. Define the EATeamInfo agent ---
# # This agent uses the BigQuery tool to answer questions about the team.
# ea_team_info_agent = LlmAgent(
#     name="EATeamInfo",
#     model=GEMINI_MODEL,
#     instruction="""- You are a manager for the Enterprise Architecture (EA) team. You provide information about the team members and their assignments.
# - If asked to get information about the team, you should call the `bq_tool` tool. The tool queries tables in BigQuery and accepts a json paylod in the form  '{"query": "SQLQUERY"}' When building your queries, always use a syntax based on LIKE "%value%" constratins.
#     - When asked to get information about the team member you shoould use table test-project-26133-466015.EA_Assistant_Data.EA_info . The table has the following schema:
#         - [{"name":"preferred_name","type":"STRING","mode":"NULLABLE"},{"name":"user_name","type":"STRING","mode":"NULLABLE"},{"name":"manager_user_name","type":"STRING","mode":"NULLABLE"},{"name":"is_manager","type":"INTEGER","mode":"NULLABLE"},{"name":"manager_hierarchy_user_names","type":"STRING","mode":"NULLABLE"},{"name":"business_title","type":"STRING","mode":"NULLABLE"},{"name":"visible_job_family","type":"STRING","mode":"NULLABLE"},{"name":"location_site","type":"STRING","mode":"NULLABLE"},{"name":"location_timezone","type":"STRING","mode":"NULLABLE"},{"name":"physical_location_site","type":"STRING","mode":"NULLABLE"},{"name":"hire_date_str","type":"DATE","mode":"NULLABLE"},{"name":"cost_center_num","type":"STRING","mode":"NULLABLE"},{"name":"cost_center_name","type":"STRING","mode":"NULLABLE"}]
#         - For example, to get the list of the team members, the SQLQUERY='SELECT preferred_name FROM `test-project-26133-466015.EA_Assistant_Data.EA_info`', therefore the full json payload to input in the API is
#         - {"query":"SELECT preferred_name FROM `test-project-26133-466015.EA_Assistant_Data.EA_info`"}
#         - For example, to get the list of team members in a given timezone, use location_timezone from table test-project-26133-466015.EA_Assistant_Data.EA_info , considering that America/New_York is in the Eastern Time Zone (ET);America/Toronto is in the Eastern Time Zone (ET); America/Chicago is in the Central Time Zone (CT); America/Denver is in the Mountain Time Zone (MT); US/Arizona is in the Mountain Time Zone (MT); America/Los_Angeles is in the Pacific Time Zone (PT); ET is one hour ahead of CT, which is an hour ahead of MT, which is an hour ahear of PT; so, this also means that, ET is two hours ahead of MT and three hours ahead of PT; CT is two hours ahead of PT.
#             - So to get the list of team members within one hour of central timzone, you can use SQLQUERY: 'SELECT preferred_name FROM `test-project-26133-466015.EA_Assistant_Data.EA_info` WHERE location_timezone in (\"America/Chicago\", \"America/Denver\", \"US/Arizona\", \"America/Toronto\", \"America/New_York\")'
#         - For example, when asking about the direct reports of a manager by their preferred_name e.g. PREFERRED_MANAGER_NAME, you can use SQLQUERY: 'SELECT preferred_name, user_name FROM `test-project-26133-466015.EA_Assistant_Data.EA_info` WHERE manager_user_name = '(SELECT user_name FROM `test-project-26133-466015.EA_Assistant_Data.EA_info` WHERE LOWER(preferred_name) LIKE "%preferred_manager_name%")' AND is_manager = 0;
#         - For example, when asking information about team members, only the first name (lower or upper case), instead of the full name, in the preferred_name might be used, therefore use LIKE in your constraints. The SQLQUERY would result in something like 'SELECT manager_user_name, location_site, location_timezone FROM `test-project-26133-466015.EA_Assistant_Data.EA_info` WHERE LOWER(preferred_name) LIKE '%first_name%';
#     - When asked to get information about the team short term assignments, you shoould use table test-project-26133-466015.EA_Assistant_Data.EA_sum_week_assign . The table has the following schema:
#         - [{"name":"resource_name","type":"STRING","mode":"NULLABLE"},{"name":"week_ending_date","type":"DATE","mode":"NULLABLE"},{"name":"SUM of assigned_hours","type":"FLOAT","mode":"NULLABLE"},{"name":"COUNTA of project_name","type":"INTEGER","mode":"NULLABLE"}]
#             - For example, if asked who are the top 5 people that are less busy next week, you could show the assignments week by week (no need to aggregate by resource_name), use the SQLQUERY='SELECT resource_name, `SUM of assigned_hours` FROM `test-project-26133-466015.EA_Assistant_Data.EA_sum_week_assign` WHERE week_ending_date = DATE_ADD(CURRENT_DATE(), INTERVAL (7 - EXTRACT(DAYOFWEEK FROM CURRENT_DATE()) + 7) DAY) ORDER BY resource_name ASC LIMIT 5;'
#             - For example, if asked who the least busy people are over the next 2 weeks that are not managers, you could show the assignments week by week (no need to aggregate by resource_name), using SQLQUERY='SELECT resource_name, `SUM of assigned_hours` FROM `test-project-26133-466015.EA_Assistant_Data.EA_sum_week_assign` WHERE (week_ending_date = DATE_ADD(CURRENT_DATE(), INTERVAL (7 - EXTRACT(DAYOFWEEK FROM CURRENT_DATE()) + 7) DAY) OR week_ending_date = DATE_ADD(CURRENT_DATE(), INTERVAL (7 - EXTRACT(DAYOFWEEK FROM CURRENT_DATE()) + 14) DAY) ) AND resource_name IN (SELECT preferred_name FROM `test-project-26133-466015.EA_Assistant_Data.EA_info` WHERE is_manager = 0) ORDER BY resource_name ASC LIMIT 5;'
#             - For example, if asked the details (...key word: details...) of the least busy people over the next 2 weeks, run something like the following, joining additional information for the team emmerbs (such as timezone) and providing week by week data for assignments and number of projets: SQLQUERY='WITH LeastBusyPeople AS (SELECT resource_name FROM `test-project-26133-466015.EA_Assistant_Data.EA_sum_week_assign` WHERE week_ending_date IN (DATE_ADD(CURRENT_DATE(), INTERVAL (7 - EXTRACT(DAYOFWEEK FROM CURRENT_DATE()) + 7) DAY), DATE_ADD(CURRENT_DATE(), INTERVAL (7 - EXTRACT(DAYOFWEEK FROM CURRENT_DATE()) + 14) DAY)) AND resource_name IN (SELECT preferred_name FROM `test-project-26133-466015.EA_Assistant_Data.EA_info` WHERE is_manager = 0) GROUP BY resource_name ORDER BY SUM(`SUM of assigned_hours`) ASC LIMIT 5) SELECT t1.resource_name, t2.location_timezone, t1.week_ending_date, t1.`SUM of assigned_hours`, t1.`COUNTA of project_name` FROM `test-project-26133-466015.EA_Assistant_Data.EA_sum_week_assign` AS t1 JOIN `test-project-26133-466015.EA_Assistant_Data.EA_info` AS t2 ON t1.resource_name = t2.preferred_name WHERE t1.resource_name IN (SELECT resource_name FROM LeastBusyPeople) AND t1.week_ending_date IN (DATE_ADD(CURRENT_DATE(), INTERVAL (7 - EXTRACT(DAYOFWEEK FROM CURRENT_DATE()) + 7) DAY), DATE_ADD(CURRENT_DATE(), INTERVAL (7 - EXTRACT(DAYOFWEEK FROM CURRENT_DATE()) + 14) DAY)) ORDER BY t1.resource_name, t1.week_ending_date ASC;'
#     - When asked to get information about the team historical assignments, you shoould use table test-project-26133-466015.EA_Assistant_Data.EA_hist_assign . The table has the following schema:
#         - [{"name":"account_name","type":"STRING","mode":"NULLABLE"},{"name":"project_name","type":"STRING","mode":"NULLABLE"},{"name":"resource_name","type":"STRING","mode":"NULLABLE"},{"name":"resource_role","type":"STRING","mode":"NULLABLE"},{"name":"project_start_date","type":"DATE","mode":"NULLABLE"},{"name":"project_end_date","type":"DATE","mode":"NULLABLE"}]
#             - For example, if asked who was assigned to account MY_ACCOUNT_NAME over the past year, you can use the SQLQUERY='SELECT DISTINCT resource_name, account_name, project_start_date FROM `test-project-26133-466015.EA_Assistant_Data.EA_hist_assign` WHERE LOWER(account_name) LIKE '%my_account_name%' AND project_start_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 YEAR);'
#     - When asked to get information about the project assignments and account for a week from one month in the past to one month one in the future, you shoould use table test-project-26133-466015.EA_Assistant_Data.EA_week_assign . The table has the following schema:
#         - [{"name":"week_ending_date","type":"DATE","mode":"NULLABLE"},{"name":"account_name","type":"STRING","mode":"NULLABLE"},{"name":"project_id","type":"STRING","mode":"NULLABLE"},{"name":"project_name","type":"STRING","mode":"NULLABLE"},{"name":"project_type","type":"STRING","mode":"NULLABLE"},{"name":"assigned_hours","type":"FLOAT","mode":"NULLABLE"},{"name":"resource_name","type":"STRING","mode":"NULLABLE"},{"name":"resource_role","type":"STRING","mode":"NULLABLE"}]
#             - For example, When asked who is on vacation (or out of office or OOO) you can look for project_name='PTO'
#             - For example, when asking information about team members, only the first name (lower or upper case), instead of the full name, in the preferred_name might be used, therefore use LIKE in your constraints. For a question like "what accounts is First_name assigned to this week?", the SQLQUERY would be something like
#                 - 'SELECT account_name FROM `test-project-26133-466015.EA_Assistant_Data.EA_week_assign` WHERE LOWER(resource_name) LIKE "%first_name%" AND week_ending_date = DATE_ADD(CURRENT_DATE(), INTERVAL (7 - EXTRACT(DAYOFWEEK FROM CURRENT_DATE())) DAY)'
#                 - Note that next week will have a constraint 'week_ending_date = DATE_ADD(CURRENT_DATE(), INTERVAL (7 - EXTRACT(DAYOFWEEK FROM CURRENT_DATE()) +7 ) DAY)' and last week 'week_ending_date = DATE_ADD(CURRENT_DATE(), INTERVAL (7 - EXTRACT(DAYOFWEEK FROM CURRENT_DATE()) -7 ) DAY)'
# - Return only the results from the tool.
#     """,
#     description="Identify the characteristics and assignments of the Enterprise Architect (EA) team members",
#     tools=[bq_tool],
# )


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
                 print(f"  -> 🔴 Error from {author_name}: {error_message}")

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
                        availability_result_str = extracted_answers.get("availability")

                        if ask_total_effort_str and availability_result_str:
                            print("\n--- Running Effort Update Agent ---")

                            # Append the schedule from the API call to the original request from the planner.
                            update_effort_query = f"""{ask_total_effort_str}

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
                            else:
                                print("<<< Effort update agent did not produce a response.")
                    else:
                        print("No questions to ask the agent.")

            except Exception as e:
                print(f"\n❌ An error occurred during API invocation: {e}")

    except Exception as e:
        print(f"\n❌ An error occurred during agent execution: {e}")



initial_trigger_query = "I have an opportunity with motorola for 8h/w for 3 weeks in the central timezone. Who are the best EA candidate for assignment?" 

# # In Colab/Jupyter:
# await call_ea_team_info_agent(initial_trigger_query, user_id=USER_ID, session_id=SESSION_ID)

# In a standalone Python script or if await is not supported/failing:
asyncio.run(call_ea_team_info_agent(initial_trigger_query, user_id=USER_ID, session_id=SESSION_ID))