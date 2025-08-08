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
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.DEBUG)

# Load environment variables from a .env file in the same directory.
load_dotenv()

# --- Configuration ---
APP_NAME = "ea_assign_agent"
USER_ID = "ea_manager"
SESSION_ID = "get_ea_team_info"
GEMINI_MODEL = "gemini-2.0-flash"
PROJECT_ID = "test-project-26133-466015"


# --- 2. Create a tool from the BigQuery API OpenAPI spec ---

# Configure authentication using Application Default Credentials (ADC)
# to generate an OIDC token for the Cloud Run service.

service_account = ServiceAccount(
    use_default_credential=True,
    # The audience is the URL of the Cloud Run service to be invoked.
    audience="https://bq-api-service-kz5lpdkcca-uc.a.run.app",
    # Scopes are for OAuth2 access tokens. For OIDC ID tokens (needed for
    # Cloud Run), 'audience' is used. Pass an empty list for scopes.
    scopes=[],
)

auth_credential = AuthCredential(
    auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
    service_account=service_account,
)

auth_scheme = HTTPBearer(bearerFormat="JWT")


# This tool will allow the agent to execute SQL queries.
with open("bq_api.yaml", "r") as f:
    bq_api_yaml_str = f.read()

bq_tool = OpenAPIToolset(
    spec_str=bq_api_yaml_str,
    spec_str_type="yaml",
    auth_scheme=auth_scheme,
    auth_credential=auth_credential,
)

# --- 3. Define the EATeamInfo agent ---
# This agent uses the BigQuery tool to answer questions about the team.
ea_team_info_agent = LlmAgent(
    name="EATeamInfo",
    model=GEMINI_MODEL,
    instruction="""- You are a manager for the Enterprise Architecture (EA) team. You provide information about the team members and their assignments.
- If asked to get information about the team, you should call the `bq_tool` tool. The tool queries tables in BigQuery and accepts a json paylod in the form  '{"query": "SQLQUERY"}' When building your queries, always use a syntax based on LIKE "%value%" constratins.
    - When asked to get information about the team member you shoould use table test-project-26133-466015.EA_Assistant_Data.EA_info . The table has the following schema:
        - [{"name":"preferred_name","type":"STRING","mode":"NULLABLE"},{"name":"user_name","type":"STRING","mode":"NULLABLE"},{"name":"manager_user_name","type":"STRING","mode":"NULLABLE"},{"name":"is_manager","type":"INTEGER","mode":"NULLABLE"},{"name":"manager_hierarchy_user_names","type":"STRING","mode":"NULLABLE"},{"name":"business_title","type":"STRING","mode":"NULLABLE"},{"name":"visible_job_family","type":"STRING","mode":"NULLABLE"},{"name":"location_site","type":"STRING","mode":"NULLABLE"},{"name":"location_timezone","type":"STRING","mode":"NULLABLE"},{"name":"physical_location_site","type":"STRING","mode":"NULLABLE"},{"name":"hire_date_str","type":"DATE","mode":"NULLABLE"},{"name":"cost_center_num","type":"STRING","mode":"NULLABLE"},{"name":"cost_center_name","type":"STRING","mode":"NULLABLE"}]
        - For example, to get the list of the team members, the SQLQUERY='SELECT preferred_name FROM `test-project-26133-466015.EA_Assistant_Data.EA_info`', therefore the full json payload to input in the API is
        - {"query":"SELECT preferred_name FROM `test-project-26133-466015.EA_Assistant_Data.EA_info`"}
        - For example, to get the list of team members in a given timezone, use location_timezone from table test-project-26133-466015.EA_Assistant_Data.EA_info , considering that America/New_York is in the Eastern Time Zone (ET);America/Toronto is in the Eastern Time Zone (ET); America/Chicago is in the Central Time Zone (CT); America/Denver is in the Mountain Time Zone (MT); US/Arizona is in the Mountain Time Zone (MT); America/Los_Angeles is in the Pacific Time Zone (PT); ET is one hour ahead of CT, which is an hour ahead of MT, which is an hour ahear of PT; so, this also means that, ET is two hours ahead of MT and three hours ahead of PT; CT is two hours ahead of PT.
            - So to get the list of team members within one hour of central timzone, you can use SQLQUERY: 'SELECT preferred_name FROM `test-project-26133-466015.EA_Assistant_Data.EA_info` WHERE location_timezone in (\"America/Chicago\", \"America/Denver\", \"US/Arizona\", \"America/Toronto\", \"America/New_York\")'
        - For example, when asking about the direct reports of a manager by their preferred_name e.g. PREFERRED_MANAGER_NAME, you can use SQLQUERY: 'SELECT preferred_name, user_name FROM `test-project-26133-466015.EA_Assistant_Data.EA_info` WHERE manager_user_name = '(SELECT user_name FROM `test-project-26133-466015.EA_Assistant_Data.EA_info` WHERE LOWER(preferred_name) LIKE "%preferred_manager_name%")' AND is_manager = 0;
        - For example, when asking information about team members, only the first name (lower or upper case), instead of the full name, in the preferred_name might be used, therefore use LIKE in your constraints. The SQLQUERY would result in something like 'SELECT manager_user_name, location_site, location_timezone FROM `test-project-26133-466015.EA_Assistant_Data.EA_info` WHERE LOWER(preferred_name) LIKE '%first_name%';
    - When asked to get information about the team short term assignments, you shoould use table test-project-26133-466015.EA_Assistant_Data.EA_sum_week_assign . The table has the following schema:
        - [{"name":"resource_name","type":"STRING","mode":"NULLABLE"},{"name":"week_ending_date","type":"DATE","mode":"NULLABLE"},{"name":"SUM of assigned_hours","type":"FLOAT","mode":"NULLABLE"},{"name":"COUNTA of project_name","type":"INTEGER","mode":"NULLABLE"}]
            - For example, if asked who are the top 5 people that are less busy next week, you could show the assignments week by week (no need to aggregate by resource_name), use the SQLQUERY='SELECT resource_name, `SUM of assigned_hours` FROM `test-project-26133-466015.EA_Assistant_Data.EA_sum_week_assign` WHERE week_ending_date = DATE_ADD(CURRENT_DATE(), INTERVAL (7 - EXTRACT(DAYOFWEEK FROM CURRENT_DATE()) + 7) DAY) ORDER BY resource_name ASC LIMIT 5;'
            - For example, if asked who the least busy people are over the next 2 weeks that are not managers, you could show the assignments week by week (no need to aggregate by resource_name), using SQLQUERY='SELECT resource_name, `SUM of assigned_hours` FROM `test-project-26133-466015.EA_Assistant_Data.EA_sum_week_assign` WHERE (week_ending_date = DATE_ADD(CURRENT_DATE(), INTERVAL (7 - EXTRACT(DAYOFWEEK FROM CURRENT_DATE()) + 7) DAY) OR week_ending_date = DATE_ADD(CURRENT_DATE(), INTERVAL (7 - EXTRACT(DAYOFWEEK FROM CURRENT_DATE()) + 14) DAY) ) AND resource_name IN (SELECT preferred_name FROM `test-project-26133-466015.EA_Assistant_Data.EA_info` WHERE is_manager = 0) ORDER BY resource_name ASC LIMIT 5;'
            - For example, if asked the details (...key word: details...) of the least busy people over the next 2 weeks, run something like the following, joining additional information for the team emmerbs (such as timezone) and providing week by week data for assignments and number of projets: SQLQUERY='WITH LeastBusyPeople AS (SELECT resource_name FROM `test-project-26133-466015.EA_Assistant_Data.EA_sum_week_assign` WHERE week_ending_date IN (DATE_ADD(CURRENT_DATE(), INTERVAL (7 - EXTRACT(DAYOFWEEK FROM CURRENT_DATE()) + 7) DAY), DATE_ADD(CURRENT_DATE(), INTERVAL (7 - EXTRACT(DAYOFWEEK FROM CURRENT_DATE()) + 14) DAY)) AND resource_name IN (SELECT preferred_name FROM `test-project-26133-466015.EA_Assistant_Data.EA_info` WHERE is_manager = 0) GROUP BY resource_name ORDER BY SUM(`SUM of assigned_hours`) ASC LIMIT 5) SELECT t1.resource_name, t2.location_timezone, t1.week_ending_date, t1.`SUM of assigned_hours`, t1.`COUNTA of project_name` FROM `test-project-26133-466015.EA_Assistant_Data.EA_sum_week_assign` AS t1 JOIN `test-project-26133-466015.EA_Assistant_Data.EA_info` AS t2 ON t1.resource_name = t2.preferred_name WHERE t1.resource_name IN (SELECT resource_name FROM LeastBusyPeople) AND t1.week_ending_date IN (DATE_ADD(CURRENT_DATE(), INTERVAL (7 - EXTRACT(DAYOFWEEK FROM CURRENT_DATE()) + 7) DAY), DATE_ADD(CURRENT_DATE(), INTERVAL (7 - EXTRACT(DAYOFWEEK FROM CURRENT_DATE()) + 14) DAY)) ORDER BY t1.resource_name, t1.week_ending_date ASC;'
    - When asked to get information about the team historical assignments, you shoould use table test-project-26133-466015.EA_Assistant_Data.EA_hist_assign . The table has the following schema:
        - [{"name":"account_name","type":"STRING","mode":"NULLABLE"},{"name":"project_name","type":"STRING","mode":"NULLABLE"},{"name":"resource_name","type":"STRING","mode":"NULLABLE"},{"name":"resource_role","type":"STRING","mode":"NULLABLE"},{"name":"project_start_date","type":"DATE","mode":"NULLABLE"},{"name":"project_end_date","type":"DATE","mode":"NULLABLE"}]
            - For example, if asked who was assigned to account MY_ACCOUNT_NAME over the past year, you can use the SQLQUERY='SELECT DISTINCT resource_name, account_name, project_start_date FROM `test-project-26133-466015.EA_Assistant_Data.EA_hist_assign` WHERE LOWER(account_name) LIKE '%my_account_name%' AND project_start_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 YEAR);'
    - When asked to get information about the project assignments and account for a week from one month in the past to one month one in the future, you shoould use table test-project-26133-466015.EA_Assistant_Data.EA_week_assign . The table has the following schema:
        - [{"name":"week_ending_date","type":"DATE","mode":"NULLABLE"},{"name":"account_name","type":"STRING","mode":"NULLABLE"},{"name":"project_id","type":"STRING","mode":"NULLABLE"},{"name":"project_name","type":"STRING","mode":"NULLABLE"},{"name":"project_type","type":"STRING","mode":"NULLABLE"},{"name":"assigned_hours","type":"FLOAT","mode":"NULLABLE"},{"name":"resource_name","type":"STRING","mode":"NULLABLE"},{"name":"resource_role","type":"STRING","mode":"NULLABLE"}]
            - For example, When asked who is on vacation (or out of office or OOO) you can look for project_name='PTO'
            - For example, when asking information about team members, only the first name (lower or upper case), instead of the full name, in the preferred_name might be used, therefore use LIKE in your constraints. For a question like "what accounts is First_name assigned to this week?", the SQLQUERY would be something like
                - 'SELECT account_name FROM `test-project-26133-466015.EA_Assistant_Data.EA_week_assign` WHERE LOWER(resource_name) LIKE "%first_name%" AND week_ending_date = DATE_ADD(CURRENT_DATE(), INTERVAL (7 - EXTRACT(DAYOFWEEK FROM CURRENT_DATE())) DAY)'
                - Note that next week will have a constraint 'week_ending_date = DATE_ADD(CURRENT_DATE(), INTERVAL (7 - EXTRACT(DAYOFWEEK FROM CURRENT_DATE()) +7 ) DAY)' and last week 'week_ending_date = DATE_ADD(CURRENT_DATE(), INTERVAL (7 - EXTRACT(DAYOFWEEK FROM CURRENT_DATE()) -7 ) DAY)'
- Return only the results from the tool.
    """,
    description="Identify the characteristics and assignments of the Enterprise Architect (EA) team members",
    tools=[bq_tool],
)

# We are testing the planner_agent agent, so we set it as the root agent.
root_agent = ea_team_info_agent
# --8<-- [end:init]

# --- 5. Running the Agent (Using InMemoryRunner for local testing) This works in Notebooks and script file ---

# Use InMemoryRunner: Ideal for quick prototyping and local testing
runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
print(f"InMemoryRunner created for agent '{root_agent.name}'.")

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
            if is_final and author_name == ea_team_info_agent.name and event.content and event.content.parts:
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
            print(f"\n<<< Raw Final Response from {ea_team_info_agent.name}:\n{final_response_text}")
        
    except Exception as e:
        print(f"\n❌ An error occurred during agent execution: {e}")

# Test sets
initial_trigger_query = "Who are the least busy team members over the next 3 weeks?" 

# In a standalone Python script or if await is not supported/failing:
asyncio.run(call_ea_team_info_agent(initial_trigger_query, user_id=USER_ID, session_id=SESSION_ID))