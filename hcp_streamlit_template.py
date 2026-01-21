"""
HCP (Healthcare Professional/Doctor) Data Steward Application Template
======================================================================
This is a reusable template for searching doctors and their hospital affiliations.
Adapt the configuration section below for your specific use case.

Key Features:
- Snowflake Cortex Analyst integration for natural language search
- Perplexity API for web enrichment
- LLM-powered affiliation priority ranking
- Comparison report UI
"""

import json
import time
import os
import requests
import pandas as pd
import streamlit as st
from snowflake.core import Root
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col
from snowflake.cortex import Complete
from urllib.parse import urlparse
from typing import List, Optional
from pydantic import BaseModel, Field
from perplexity import Perplexity

# ============================================================================
# CONFIGURATION SECTION - Customize these for your application
# ============================================================================

# App Configuration
APP_CONFIG = {
    "page_title": "HCP Data Steward",
    "page_layout": "wide",
    "search_placeholder": "Search for a Doctor/Healthcare Professional",
    "entity_name": "HCP",  # HCP, Doctor, Provider, etc.
    "entity_name_plural": "HCPs",
    "affiliation_name": "Hospital",  # Hospital, Facility, Organization
    "affiliation_name_plural": "Hospitals",
}

# Column mappings - Map your database columns to display names
# Format: "DISPLAY_NAME": "DB_COLUMN_NAME"
ENTITY_COLUMNS = {
    "ID": "ID",
    "Name": "NAME",
    "NPI": "NPI",
    "Specialty": "SPECIALTY",
    "Address Line1": "ADDRESS1",
    "Address Line2": "ADDRESS2",
    "City": "CITY",
    "State": "STATE",
    "ZIP": "ZIP",
}

# Affiliation columns for the comparison table
AFFILIATION_COLUMNS = {
    "Hospital ID": "HOSPITAL_ID",
    "Hospital Name": "HOSPITAL_NAME",
    "Hospital NPI": "HOSPITAL_NPI",
    "Hospital Address": "HOSPITAL_ADDRESS",
    "Hospital City": "HOSPITAL_CITY",
    "Hospital State": "HOSPITAL_STATE",
    "Hospital ZIP": "HOSPITAL_ZIP",
}

# Search results table columns (subset shown in main search results)
SEARCH_RESULT_COLUMNS = ["ID", "Name", "NPI", "Specialty", "City", "State"]
SEARCH_RESULT_COL_SIZES = (0.8, 2, 1.2, 1.5, 1.2, 0.8)

# Cortex Analyst Configuration
CORTEX_CONFIG = {
    "database": "YOUR_DATABASE",
    "schema": "YOUR_SCHEMA", 
    "stage": "YOUR_STAGE",
    "semantic_model_file": "YOUR_MODEL.yaml",
}

# ============================================================================
# END CONFIGURATION SECTION
# ============================================================================

# Set Perplexity API key from Streamlit secrets
os.environ["PERPLEXITY_API_KEY"] = st.secrets["perplexity"]["api_key"]


# --- SNOWFLAKE CONNECTION ---
@st.cache_resource
def get_snowflake_session():
    """Creates a Snowflake session using credentials from Streamlit secrets."""
    connection_parameters = {
        "account": st.secrets["snowflake"]["account"],
        "user": st.secrets["snowflake"]["user"],
        "password": st.secrets["snowflake"]["password"],
        "warehouse": st.secrets["snowflake"]["warehouse"],
        "database": st.secrets["snowflake"]["database"],
        "schema": st.secrets["snowflake"]["schema"],
        "role": st.secrets["snowflake"]["role"],
    }
    return Session.builder.configs(connection_parameters).create()


# --- LLM PRIORITY RANKING (Snowflake Cortex REST API with Structured Output) ---
def get_affiliation_priorities_from_llm(session, selected_entity_data: dict, affiliations: list, entity_type: str = "doctor", affiliation_type: str = "hospital") -> dict:
    """
    Calls Snowflake Cortex REST API with structured JSON output to rank affiliations by priority.
    
    Args:
        session: Snowflake session
        selected_entity_data: Dict with selected entity details (Name, Address, City, State, ZIP)
        affiliations: List of (key, affiliation_data) tuples
        entity_type: Type of entity (e.g., "doctor", "provider", "HCP")
        affiliation_type: Type of affiliation (e.g., "hospital", "facility", "organization")
    
    Returns:
        Dict mapping affiliation key to {"priority": int, "reason": str}
    """
    if not affiliations:
        return {}
    
    # Build the prompt for the LLM
    selected_info = f"""
Selected {entity_type.title()}:
- Name: {selected_entity_data.get('Name', 'N/A')}
- Address: {selected_entity_data.get('Address Line1', '')} {selected_entity_data.get('Address Line2', '')}
- City: {selected_entity_data.get('City', 'N/A')}
- State: {selected_entity_data.get('State', 'N/A')}
- ZIP: {selected_entity_data.get('ZIP', 'N/A')}
"""
    
    affiliations_info = f"{affiliation_type.title()} affiliations to rank:\n"
    for idx, (key, aff) in enumerate(affiliations):
        affiliations_info += f"""
{affiliation_type.title()} {idx + 1} (Key: {key}):
- Name: {aff.get('HOSPITAL_NAME', aff.get('HCO NAME', 'N/A'))}
- Address: {aff.get('HOSPITAL_ADDRESS', aff.get('HCO ADDRESS', 'N/A'))}
- City: {aff.get('HOSPITAL_CITY', aff.get('HCO CITY', 'N/A'))}
- State: {aff.get('HOSPITAL_STATE', aff.get('HCO STATE', 'N/A'))}
- ZIP: {aff.get('HOSPITAL_ZIP', aff.get('HCO ZIP', 'N/A'))}
- Source: {aff.get('SOURCE', 'N/A')}
"""
    
    prompt = f"""You are a healthcare data analyst. Analyze the following {entity_type} and their potential {affiliation_type} affiliations. 
Rank each {affiliation_type} by priority (1 being highest priority/best match) based on:
1. Geographic proximity (same city, state, ZIP code area)
2. Name similarity or relationship (same health system, known affiliations)
3. Address proximity

{selected_info}

{affiliations_info}

Use the exact keys provided for each {affiliation_type} in your rankings."""

    try:
        # Use Snowflake Cortex REST API with structured JSON output
        account = st.secrets["snowflake"]["account"]
        account_url = account.replace("_", "-").replace(".", "-")
        api_url = f"https://{account_url}.snowflakecomputing.com/api/v2/cortex/inference:complete"
        
        token = session.connection.rest.token
        
        headers = {
            "Authorization": f"Snowflake Token=\"{token}\"",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        json_schema = {
            "type": "object",
            "properties": {
                "rankings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string"},
                            "priority": {"type": "number"},
                            "reason": {"type": "string"}
                        },
                        "required": ["key", "priority", "reason"]
                    }
                }
            },
            "required": ["rankings"]
        }
        
        request_body = {
            "model": "claude-3-5-sonnet",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
            "response_format": {
                "type": "json",
                "schema": json_schema
            }
        }
        
        resp = requests.post(api_url, headers=headers, json=request_body, timeout=60)
        
        if resp.status_code >= 400:
            raise Exception(f"API request failed with status {resp.status_code}: {resp.text}")
        
        # Parse streaming response
        response_text = ""
        for line in resp.text.strip().split("\n"):
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        response_text += content
                except json.JSONDecodeError:
                    continue
        
        result = json.loads(response_text.strip())
        
        priority_map = {}
        for ranking in result.get("rankings", []):
            priority_map[str(ranking["key"])] = {
                "priority": ranking["priority"],
                "reason": ranking["reason"]
            }
        return priority_map
        
    except Exception as e:
        st.warning(f"Could not get LLM priority ranking: {e}")
        return {str(key): {"priority": idx + 1, "reason": "Default ordering (LLM unavailable)"} 
                for idx, (key, _) in enumerate(affiliations)}


# --- POPUP FUNCTIONS ---
def show_popup_without_button(popup_placeholder, message_type, record_info):
    """Renders a custom popup message that auto-dismisses after a delay."""
    with popup_placeholder.container():
        st.markdown("""
            <style>
                .st-popup-container {
                    position: fixed;
                    top: 20%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    z-index: 9999;
                    padding: 2rem;
                    border-radius: 10px;
                    background-color: #ffffff;
                    color: #000000;
                    border: 2px solid #4CAF50;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
                    text-align: center;
                    min-width: 350px;
                }
                .st-popup-container h4 {
                    color: #4CAF50;
                    font-size: 1.5rem;
                    margin-bottom: 0.5rem;
                }
                .st-popup-container p {
                    font-size: 1rem;
                }
            </style>
            """, unsafe_allow_html=True)
        
        if message_type == "update_success" and 'message' in record_info:
            title = "Update Successful! ✅"
            message = record_info['message']
        elif message_type == "primary_success":
            title = f"Primary {APP_CONFIG['affiliation_name']} Updated! ✅"
            message = f"Primary {APP_CONFIG['affiliation_name']} is set with ID: {record_info.get('affiliation_id')}."
        else:
            title = "Success! ✅"
            message = "Operation completed successfully."
            
        st.markdown(
            f"""
            <div class="st-popup-container">
                <h4>{title}</h4>
                <p>{message}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    time.sleep(2)
    st.session_state.show_popup = False
    st.session_state.popup_message_info = None
    st.rerun()


# --- PYDANTIC MODELS FOR PERPLEXITY RESPONSE ---
class EntityData(BaseModel):
    """Model for the main entity (HCP/Doctor) data from web search."""
    Name: list[str]
    address_line_1: List[str] = Field(..., alias="Address Line1")
    address_line_2: List[str] = Field(..., alias="Address Line2")
    ZIP: list[str]
    City: list[str]
    State: list[str]
    Specialty: list[str] = []
    NPI: list[str] = []

class AffiliationData(BaseModel):
    """Model for affiliation (Hospital) data from web search."""
    Affiliation_ID: list[str]
    Affiliation_Name: list[str]
    Affiliation_NPI: list[str] = []
    Affiliation_Address1: list[str]
    Affiliation_City: list[str]
    Affiliation_State: list[str]
    Affiliation_ZIP: list[str]

class SearchResponse(BaseModel):
    """Combined response model for Perplexity API."""
    entity_data: EntityData
    affiliation_data: AffiliationData


# --- PERPLEXITY WEB SEARCH ---
def get_consolidated_data_for_entity(entity_data, model_name="sonar-pro", use_pro_search=True, entity_type="doctor", affiliation_type="hospital"):
    """
    Searches the web for entity information and affiliations using Perplexity API.
    
    Args:
        entity_data: Dict or pandas Series with entity details
        model_name: Perplexity model to use
        use_pro_search: Whether to use pro search
        entity_type: Type of entity being searched
        affiliation_type: Type of affiliations to find
    """
    client = Perplexity()
    
    if hasattr(entity_data, 'to_dict'):
        entity_data = entity_data.to_dict()
    
    def get_val(key):
        if isinstance(entity_data, dict):
            val = entity_data.get(key, '')
            if not val:
                val = entity_data.get(f'HCP_{key}', entity_data.get(f'DOCTOR_{key}', ''))
            return val
        return str(entity_data)
    
    entity_name = get_val('NAME')
    entity_npi = get_val('NPI')
    entity_specialty = get_val('SPECIALTY')
    entity_address1 = get_val('ADDRESS1')
    entity_city = get_val('CITY')
    entity_state = get_val('STATE')
    entity_zip = get_val('ZIP')
    
    user_query = f"""
    You are a healthcare data research specialist. Search the web thoroughly for information about this US {entity_type}:
    
    **{entity_type.title()} to Research:**
    - Name: {entity_name}
    - NPI: {entity_npi}
    - Specialty: {entity_specialty}
    - Address: {entity_address1}
    - City: {entity_city}
    - State: {entity_state}
    - ZIP: {entity_zip}

    **IMPORTANT INSTRUCTIONS:**
    1. You MUST search the web and find COMPLETE information for ALL fields requested below
    2. Do NOT return "N/A" for address fields if the {entity_type} or {affiliation_type} exists
    3. Search NPI registries, hospital websites, and medical directories

    **Part 1 - {entity_type.title()} Details (verify/update from web sources):**
    - Name: Full name of the {entity_type}
    - NPI: 10-digit NPI number
    - Specialty: Medical specialty
    - Address Line 1: Practice street address
    - Address Line 2: Suite/unit number (or empty string if none)
    - City: City name in ALL CAPS
    - State: 2-letter US state code
    - ZIP: 5-digit zipcode

    **Part 2 - {affiliation_type.title()} Affiliations:**
    Search for {affiliation_type}s where this {entity_type} practices or has privileges.
    
    For each {affiliation_type}, provide:
    - Affiliation_ID: The NPI number of the {affiliation_type} (10 digits), or "N/A" if not found
    - Affiliation_Name: Full name of the {affiliation_type}
    - Affiliation_NPI: NPI number of the {affiliation_type}
    - Affiliation_Address1: Street address of the {affiliation_type}
    - Affiliation_City: City in ALL CAPS
    - Affiliation_State: 2-letter state code
    - Affiliation_ZIP: 5-digit zipcode

    **CRITICAL:** 
    - Search thoroughly for {affiliation_type} affiliations
    - Include ALL known {affiliation_type}s where the {entity_type} practices
    - Provide complete addresses for each {affiliation_type}
    """

    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": user_query}],
        web_search_options={
            "search_type": "pro" if use_pro_search else "fast"
        },  
        response_format={
            "type": "json_schema",
            "json_schema": {
                "schema": SearchResponse.model_json_schema()
            }
        }
    )

    return json.loads(completion.choices[0].message.content)


def standardize_value_lengths(dictionary):
    """Standardize list lengths in a dictionary by padding shorter lists."""
    valid_lists = [v for v in dictionary.values() if isinstance(v, list) and len(v) > 0]
    if not valid_lists:
        return dictionary

    max_length = max(len(v) for v in valid_lists)

    for key, value in dictionary.items():
        if not isinstance(value, list):
            continue
        if len(value) == 0:
            dictionary[key] = [None] * max_length
        elif len(value) < max_length:
            dictionary[key].extend([value[0]] * (max_length - len(value)))

    return dictionary


# --- CORTEX ANALYST HELPER FUNCTIONS ---
def send_message(session, prompt: str) -> dict:
    """Send a message to Cortex Analyst and receive a response."""
    root = Root(session)
    cortex_analyst_service = (
        root.databases[CORTEX_CONFIG["database"]]
        .schemas[CORTEX_CONFIG["schema"]]
        .cortex_analyst_services[CORTEX_CONFIG["stage"]]
    )
    
    resp = cortex_analyst_service.send_message(
        model=CORTEX_CONFIG["semantic_model_file"],
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    )
    return resp


def process_message(session, prompt: str):
    """Process a user message and update session state."""
    st.session_state.messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
    with st.spinner("Searching..."):
        response = send_message(session, prompt=prompt)
        content = response["message"]["content"]
        st.session_state.messages.append({"role": "assistant", "content": content})


def display_interpretation(content):
    """Display the assistant's interpretation of the query."""
    for item in content:
        if item["type"] == "text":
            st.markdown(f'You searched : "{st.session_state.last_prompt}"')
            st.markdown(f'This is our interpretation of your question : "{item["text"]}"')


def ensure_join_in_sql(sql: str) -> str:
    """Modify SQL to include necessary joins if needed (customize for your schema)."""
    # Add your custom SQL modification logic here if needed
    return sql


# --- UI HELPER FUNCTIONS ---
def get_safe_value(record, key, prefix=""):
    """Safely get a value from a record with optional prefix fallback."""
    val = record.get(key)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        val = record.get(f"{prefix}{key}" if prefix else key)
    return val if val is not None and not (isinstance(val, float) and pd.isna(val)) else 'N/A'


# --- CUSTOM CSS ---
CUSTOM_CSS = """
<style>
    div[data-testid="stHorizontalBlock"]:has(div.cell-content),
    div[data-testid="stHorizontalBlock"]:has(div.affiliation-cell) { border-bottom: 1px solid #e6e6e6; }
    div[data-testid="stHorizontalBlock"]:has(div.cell-content):hover,
    div[data-testid="stHorizontalBlock"]:has(div.affiliation-cell):hover { background-color: #f8f9fa; }
    .cell-content, .affiliation-cell { padding: 0.3rem 0.5rem; font-size: 14px; display: flex; align-items: center; height: 48px; }
    .report-header, .affiliation-header { font-weight: bold; color: #4f4f4f; padding: 0.5rem; }
    .affiliation-header { border-bottom: 2px solid #ccc; }
    .report-proposed-column { border-left: 2px solid #D3D3D3; padding-left: 1.5rem; }
    .checkbox-container { width: 100%; text-align: center; }
    .checkbox-container div[data-testid="stCheckbox"] { padding-top: 12px; }
    div[data-testid="stExpander"] button { margin-top: -1rem; }
    .affiliation-cell div[data-testid="stButton"] button { padding: 0.2rem 0.5rem; font-size: 12px; height: 30px; }
</style>
"""


# --- MAIN PAGE RENDER FUNCTION ---
def render_main_page(session):
    """Render the main search page."""
    st.title(f"🏥 {APP_CONFIG['page_title']}")
    st.markdown(f"Search for {APP_CONFIG['entity_name_plural']} and manage their {APP_CONFIG['affiliation_name'].lower()} affiliations.")
    
    # --- MAIN INPUT LOGIC ---
    freeze_container = st.container(border=True)
    with freeze_container:
        user_input_text = st.chat_input(APP_CONFIG["search_placeholder"])
        current_prompt = user_input_text

        if current_prompt and current_prompt != st.session_state.get("last_prompt"):
            process_message(session, prompt=current_prompt)
            st.session_state.last_prompt = current_prompt

    # --- DISPLAY LOGIC ---
    if st.session_state.messages:
        assistant_messages = [msg for msg in st.session_state.messages if msg["role"] == "assistant"]
        if assistant_messages:
            st.markdown("---")
            display_interpretation(content=assistant_messages[-1]["content"])

            # Search Results Table
            response_container = st.container(border=True)
            with response_container:
                st.markdown("## Search Results")
                
                content = assistant_messages[-1]["content"]
                sql_item_found = False
                
                for item in content:
                    if item["type"] == "sql":
                        sql_item_found = True
                        original_sql = item["statement"]
                        sql_to_run = ensure_join_in_sql(original_sql)
                        df = session.sql(sql_to_run).to_pandas()
                        
                        if not df.empty:
                            st.session_state.results_df = df
                            st.write("Please select a record from the table to proceed:")
                            
                            # Render table headers
                            cols = st.columns(SEARCH_RESULT_COL_SIZES)
                            headers = ["Select"] + SEARCH_RESULT_COLUMNS[1:]  # Skip ID in display, keep for selection
                            for col_obj, header_name in zip(cols, ["Select"] + SEARCH_RESULT_COLUMNS[1:]):
                                col_obj.markdown(f"**{header_name}**")

                            # Render table rows
                            for index, row in df.iterrows():
                                row_id = row.get("ID") if "ID" in row.index else row.get(f"{APP_CONFIG['entity_name']}_ID")
                                if row_id is None or pd.isna(row_id):
                                    row_id = index
                                is_selected = row_id == st.session_state.get("selected_entity_id")
                                row_cols = st.columns(SEARCH_RESULT_COL_SIZES)

                                if is_selected:
                                    row_cols[0].write("🔘")
                                else:
                                    if row_cols[0].button("", key=f"select_{row_id}"):
                                        st.session_state.selected_entity_id = row_id
                                        st.rerun()
                                
                                # Display remaining columns
                                for i, col_name in enumerate(SEARCH_RESULT_COLUMNS[1:], 1):
                                    db_col = ENTITY_COLUMNS.get(col_name, col_name.upper())
                                    row_cols[i].write(row.get(db_col, "N/A"))
                        else:
                            st.info("We couldn't find any records matching your search.", icon="ℹ️")
                            st.markdown("")
                            if st.button("🔍 Still want to proceed with Web Search?", key="web_search_empty", type="primary"):
                                create_empty_record_and_redirect()
                
                if not sql_item_found:
                    st.info("The assistant did not return a SQL query for this prompt. It may be a greeting or a clarifying question.")
                    st.markdown("")
                    if st.button("🔍 Still want to proceed with Web Search?", key="web_search_no_sql", type="primary"):
                        create_empty_record_and_redirect()
            
            # Selected Record Details
            if st.session_state.get("selected_entity_id") and st.session_state.results_df is not None:
                display_selected_record_details(session)


def create_empty_record_and_redirect():
    """Create an empty record and redirect to enrichment page."""
    st.session_state.empty_record_for_enrichment = {
        'ID': '',
        'NAME': '',
        'NPI': '',
        'SPECIALTY': '',
        'ADDRESS1': '',
        'ADDRESS2': '',
        'CITY': '',
        'STATE': '',
        'ZIP': '',
    }
    st.session_state.web_search_query = st.session_state.get('last_prompt', '')
    st.session_state.selected_entity_id = 'empty_record'
    st.session_state.current_view = "enrichment_page"
    st.rerun()


def display_selected_record_details(session):
    """Display details for the selected record."""
    st.markdown("---")
    st.markdown(f"### Selected {APP_CONFIG['entity_name']} Details")
    
    id_col = "ID" if "ID" in st.session_state.results_df.columns else f"{APP_CONFIG['entity_name']}_ID"
    selected_id_str = str(st.session_state.selected_entity_id)
    selected_record = st.session_state.results_df[
        st.session_state.results_df[id_col].astype(str) == selected_id_str
    ]
    
    if not selected_record.empty:
        record = selected_record.iloc[0]
        
        # Display record details in columns
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Name:** {get_safe_value(record, 'NAME')}")
            st.markdown(f"**NPI:** {get_safe_value(record, 'NPI')}")
            st.markdown(f"**Specialty:** {get_safe_value(record, 'SPECIALTY')}")
        with col2:
            st.markdown(f"**Address:** {get_safe_value(record, 'ADDRESS1')}")
            st.markdown(f"**City, State ZIP:** {get_safe_value(record, 'CITY')}, {get_safe_value(record, 'STATE')} {get_safe_value(record, 'ZIP')}")
        
        st.divider()
        
        # Enrich button
        button_col, _ = st.columns([0.25, 0.75])
        with button_col:
            if st.button("Enrich with AI Assistant 🚀", type="primary"):
                st.session_state.current_view = "enrichment_page"
                st.rerun()


# --- ENRICHMENT PAGE RENDER FUNCTION ---
def render_enrichment_page(session, selected_df):
    """Render the enrichment/comparison page."""
    # Back button
    _, btn_col = st.columns([4, 1])
    with btn_col:
        if st.button("⬅️ Back to Search Results"):
            st.session_state.current_view = "main"
            st.session_state.selected_entity_id = None
            st.rerun()

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(f"<h3>📑 Current vs. Proposed Comparison Report</h3>", unsafe_allow_html=True)
    
    if selected_df.empty:
        st.warning(f"No {APP_CONFIG['entity_name']} data was provided for enrichment.")
        st.stop()
    
    # Get enriched data from API
    @st.cache_data(ttl=600)
    def get_enriched_data(_session, entity_df):
        if entity_df.empty:
            return {}
        selected_record = entity_df.iloc[0].to_dict()
        try:
            return get_consolidated_data_for_entity(
                selected_record, 
                model_name="sonar-pro", 
                use_pro_search=True,
                entity_type=APP_CONFIG['entity_name'].lower(),
                affiliation_type=APP_CONFIG['affiliation_name'].lower()
            )
        except Exception as e:
            st.error(f"Error during AI enrichment: {e}")
            return {}
    
    # Build current data dict
    selected_record = selected_df.iloc[0]
    current_data_dict = {}
    for display_name, db_col in ENTITY_COLUMNS.items():
        val = selected_record.get(db_col)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            val = ''
        current_data_dict[display_name] = val
    
    current_df = pd.DataFrame([current_data_dict])
    
    # Placeholder for reason popup dialog
    if st.session_state.get('show_reason_popup'):
        popup_data = st.session_state.get('reason_popup_data', {})
        affiliation_name = popup_data.get('affiliation_name', 'Unknown')
        priority = popup_data.get('priority', '-')
        reason = popup_data.get('reason', 'No reason available')
        
        @st.dialog("🎯 Priority Reasoning")
        def show_reason_dialog():
            st.markdown(f"**{APP_CONFIG['affiliation_name']}:** {affiliation_name}")
            st.markdown(f"<span style='display: inline-block; background-color: #4CAF50; color: white; padding: 0.25rem 0.75rem; border-radius: 15px; font-weight: bold;'>Priority {priority}</span>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown(f"""
            <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 5px; border-left: 4px solid #1f77b4;'>
                <strong>Reason:</strong><br>{reason}
            </div>
            """, unsafe_allow_html=True)
            if st.button("Close", key="close_dialog_btn", use_container_width=True):
                st.session_state.show_reason_popup = False
                st.session_state.reason_popup_data = None
                st.rerun()
        
        show_reason_dialog()
    
    # Get enriched data
    with st.spinner(f"🔍 Searching web for {APP_CONFIG['entity_name']} information and {APP_CONFIG['affiliation_name'].lower()} affiliations..."):
        api_response = get_enriched_data(session, selected_df)
    
    if not api_response:
        st.error("Could not retrieve enriched data.")
        st.stop()
    
    # Process and display the data...
    st.success(f"✅ Found enriched data for {APP_CONFIG['entity_name']}!")
    
    # Display comparison sections here (adapt from original code)
    # ... Add your comparison UI logic ...


# ============================================================================
# MAIN APPLICATION ENTRY POINT
# ============================================================================

# Set page config
st.set_page_config(layout=APP_CONFIG["page_layout"], page_title=APP_CONFIG["page_title"])

# Initialize session state
SESSION_STATE_DEFAULTS = {
    "messages": [],
    "results_df": None,
    "selected_entity_id": None,
    "current_view": "main",
    "last_prompt": None,
    "show_popup": False,
    "popup_message_info": None,
    "show_confirm_dialog": False,
    "show_primary_confirm_dialog": False,
    "show_reason_popup": False,
    "reason_popup_data": None,
    "priority_reasons": {},
    "priority_rankings_cache": {},
}

for key, default_value in SESSION_STATE_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# Get session
session = get_snowflake_session()

# Page router
if st.session_state.current_view == "main":
    render_main_page(session)
elif st.session_state.current_view == "enrichment_page":
    popup_placeholder = st.empty()
    if st.session_state.show_popup:
        show_popup_without_button(popup_placeholder, st.session_state.popup_message_info['type'], st.session_state.popup_message_info)
    
    # Handle empty record flow
    if st.session_state.selected_entity_id == 'empty_record' and st.session_state.get('empty_record_for_enrichment'):
        empty_record = st.session_state.empty_record_for_enrichment
        selected_record_df = pd.DataFrame([empty_record])
        if not st.session_state.show_popup:
            render_enrichment_page(session, selected_record_df)
    elif st.session_state.selected_entity_id and st.session_state.results_df is not None:
        id_col = "ID" if "ID" in st.session_state.results_df.columns else f"{APP_CONFIG['entity_name']}_ID"
        selected_id_str = str(st.session_state.selected_entity_id)
        selected_record_df = st.session_state.results_df[
            st.session_state.results_df[id_col].astype(str) == selected_id_str
        ]
        if not st.session_state.show_popup:
            render_enrichment_page(session, selected_record_df)
    else:
        st.warning(f"Please select a {APP_CONFIG['entity_name']} record from the main page first.")
        if st.button("Back to Main Page"):
            st.session_state.current_view = "main"
            st.rerun()
