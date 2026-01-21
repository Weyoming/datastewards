"""
HCP (Healthcare Professional/Doctor) Data Steward Application Template
======================================================================
Enhanced template with improved composability and integrated features.

Key Features:
- Snowflake Cortex Analyst integration for natural language search
- Perplexity API for web enrichment
- LLM-powered affiliation priority ranking
- Comparison report UI with detailed dialogs
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
from urllib.parse import urlparse
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field
from perplexity import Perplexity

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

APP_CONFIG = {
    "page_title": "HCP Data Steward",
    "page_layout": "wide",
    "search_placeholder": "Search for a Doctor/Healthcare Professional",
    "entity_name": "HCP",
    "entity_name_plural": "HCPs",
    "affiliation_name": "Hospital",
    "affiliation_name_plural": "Hospitals",
}

# Column mappings - <Display Name>: <DB Column>
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

AFFILIATION_COLUMNS = {
    "Affiliation ID": "HOSPITAL_ID",
    "Affiliation Name": "HOSPITAL_NAME",
    "Affiliation NPI": "HOSPITAL_NPI",
    "Affiliation Address": "HOSPITAL_ADDRESS",
    "Affiliation City": "HOSPITAL_CITY",
    "Affiliation State": "HOSPITAL_STATE",
    "Affiliation ZIP": "HOSPITAL_ZIP",
}

# Search results table configuration
SEARCH_RESULT_COLUMNS = ["ID", "Name", "NPI", "Specialty", "City", "State"]
SEARCH_RESULT_COL_SIZES = (0.8, 2, 1.2, 1.5, 1.2, 0.8)

# Cortex Analyst Configuration
CORTEX_CONFIG = {
    "database": st.secrets["snowflake"]["database"],
    "schema": st.secrets["snowflake"]["schema"], 
    "stage": "YOUR_STAGE",
    "semantic_model_file": st.secrets["snowflake"]["semantic_model_file"],
}

# Database table name (customize for your schema)
TABLE_NAME = "NPI"

# ============================================================================
# END CONFIGURATION SECTION
# ============================================================================

os.environ["PERPLEXITY_API_KEY"] = st.secrets["perplexity"]["api_key"]

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class EntityData(BaseModel):
    """Model for entity (HCP/Doctor) data from web search."""
    Name: List[str]
    address_line_1: List[str] = Field(..., alias="Address Line1")
    address_line_2: List[str] = Field(..., alias="Address Line2")
    ZIP: List[str]
    City: List[str]
    State: List[str]
    Specialty: List[str] = []
    NPI: List[str] = []

class AffiliationData(BaseModel):
    """Model for affiliation (Hospital) data from web search."""
    Affiliation_ID: List[str]
    Affiliation_Name: List[str]
    Affiliation_NPI: List[str] = []
    Affiliation_Address1: List[str]
    Affiliation_City: List[str]
    Affiliation_State: List[str]
    Affiliation_ZIP: List[str]

class SearchResponse(BaseModel):
    """Combined response model for Perplexity API."""
    entity_data: EntityData
    affiliation_data: AffiliationData

# ============================================================================
# SNOWFLAKE CONNECTION
# ============================================================================

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

# ============================================================================
# CORTEX ANALYST FUNCTIONS
# ============================================================================

def send_cortex_message(session, prompt: str) -> dict:
    """Send message to Cortex Analyst via REST API."""
    account = st.secrets["snowflake"]["account"]
    account_url = account.replace("_", "-").replace(".", "-")
    api_url = f"https://{account_url}.snowflakecomputing.com/api/v2/cortex/analyst/message"
    
    headers = {
        "Authorization": f"Snowflake Token=\"{session.connection.rest.token}\"",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    
    request_body = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "semantic_model_file": f"@{CORTEX_CONFIG['database']}.{CORTEX_CONFIG['schema']}.{CORTEX_CONFIG['stage']}/{CORTEX_CONFIG['semantic_model_file']}",
    }

    resp = requests.post(api_url, headers=headers, json=request_body, timeout=30)
    
    if resp.status_code < 400:
        return resp.json()
    else:    
        raise Exception(f"Failed request with status {resp.status_code}: {resp.text}")

def process_cortex_message(session, prompt: str):
    """Process user message and update session state."""
    st.session_state.messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    
    with st.spinner("Searching..."):
        response = send_cortex_message(session, prompt)
        question_item = {"type": "text", "text": prompt.strip()}
        response["message"]["content"].insert(0, question_item)
        st.session_state.messages.append({"role": "assistant", "content": response["message"]["content"]})

# ============================================================================
# LLM PRIORITY RANKING
# ============================================================================

def get_affiliation_priorities(session, entity_data: dict, affiliations: List[Tuple], 
                               entity_type: str = None, affiliation_type: str = None) -> dict:
    """
    Calls Snowflake Cortex REST API to rank affiliations by priority.
    Returns dict mapping affiliation key to {"priority": int, "reason": str}
    """
    if not affiliations:
        return {}
    
    entity_type = entity_type or APP_CONFIG['entity_name'].lower()
    affiliation_type = affiliation_type or APP_CONFIG['affiliation_name'].lower()
    
    # Build prompt
    entity_info = f"""
Selected {entity_type.title()}:
- Name: {entity_data.get('Name', 'N/A')}
- Address: {entity_data.get('Address Line1', '')} {entity_data.get('Address Line2', '')}
- City: {entity_data.get('City', 'N/A')}
- State: {entity_data.get('State', 'N/A')}
- ZIP: {entity_data.get('ZIP', 'N/A')}
"""
    
    affiliations_info = f"{affiliation_type.title()} affiliations to rank:\n"
    for idx, (key, aff) in enumerate(affiliations):
        aff_name = aff.get('HOSPITAL_NAME') or aff.get(f'{affiliation_type.upper()}_NAME', 'N/A')
        aff_addr = aff.get('HOSPITAL_ADDRESS') or aff.get(f'{affiliation_type.upper()}_ADDRESS', 'N/A')
        aff_city = aff.get('HOSPITAL_CITY') or aff.get(f'{affiliation_type.upper()}_CITY', 'N/A')
        aff_state = aff.get('HOSPITAL_STATE') or aff.get(f'{affiliation_type.upper()}_STATE', 'N/A')
        aff_zip = aff.get('HOSPITAL_ZIP') or aff.get(f'{affiliation_type.upper()}_ZIP', 'N/A')
        
        affiliations_info += f"""
{affiliation_type.title()} {idx + 1} (Key: {key}):
- Name: {aff_name}
- Address: {aff_addr}
- City: {aff_city}
- State: {aff_state}
- ZIP: {aff_zip}
- Source: {aff.get('SOURCE', 'N/A')}
"""
    
    prompt = f"""You are a healthcare data analyst. Rank each {affiliation_type} by priority (1=highest) based on:
1. Geographic proximity (same city, state, ZIP)
2. Name similarity (same health system)
3. Address proximity

{entity_info}

{affiliations_info}

Use the exact keys provided."""

    try:
        account = st.secrets["snowflake"]["account"]
        account_url = account.replace("_", "-").replace(".", "-")
        api_url = f"https://{account_url}.snowflakecomputing.com/api/v2/cortex/inference:complete"
        
        headers = {
            "Authorization": f"Snowflake Token=\"{session.connection.rest.token}\"",
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
            "response_format": {"type": "json", "schema": json_schema}
        }
        
        resp = requests.post(api_url, headers=headers, json=request_body, timeout=60)
        
        if resp.status_code >= 400:
            raise Exception(f"API failed: {resp.status_code}")
        
        # Parse streaming response
        response_text = ""
        for line in resp.text.strip().split("\n"):
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if "choices" in data and data["choices"]:
                        response_text += data["choices"][0].get("delta", {}).get("content", "")
                except json.JSONDecodeError:
                    continue
        
        result = json.loads(response_text.strip())
        return {str(r["key"]): {"priority": r["priority"], "reason": r["reason"]} 
                for r in result.get("rankings", [])}
        
    except Exception as e:
        st.warning(f"Could not get LLM priority ranking: {e}")
        return {str(key): {"priority": idx + 1, "reason": "Default ordering (LLM unavailable)"} 
                for idx, (key, _) in enumerate(affiliations)}

# ============================================================================
# PERPLEXITY WEB SEARCH
# ============================================================================

def search_entity_web(entity_data, model_name="sonar-pro", use_pro_search=True, 
                     entity_type: str = None, affiliation_type: str = None,
                     search_query: str = None):
    """Search web for entity and affiliation information."""
    client = Perplexity()
    
    entity_type = entity_type or APP_CONFIG['entity_name'].lower()
    affiliation_type = affiliation_type or APP_CONFIG['affiliation_name'].lower()
    
    # Convert to dict if needed
    if hasattr(entity_data, 'to_dict'):
        entity_data = entity_data.to_dict()
    
    def get_val(key):
        val = entity_data.get(key, '')
        if not val:
            # Try with entity name prefix
            val = entity_data.get(f"{APP_CONFIG['entity_name']}_{key}", '')
        return val or ''
    
    entity_name = get_val('NAME') or search_query or ''
    
    user_query = f"""
You are a healthcare data research specialist. Search the web for this US {entity_type}:

**{entity_type.title()} to Research:**
- Name: {entity_name}
- NPI: {get_val('NPI')}
- Specialty: {get_val('SPECIALTY')}
- Address: {get_val('ADDRESS1')}
- City: {get_val('CITY')}
- State: {get_val('STATE')}
- ZIP: {get_val('ZIP')}

**CRITICAL INSTRUCTIONS:**
1. Search thoroughly - do NOT return "N/A" for addresses if entity exists
2. Search NPI registries, {affiliation_type} websites, medical directories

**Part 1 - {entity_type.title()} Details:**
- Name, NPI, Specialty
- Address Line1, Address Line2 (or empty)
- City (ALL CAPS), State (2-letter), ZIP (5-digit)

**Part 2 - {affiliation_type.title()} Affiliations:**
For each {affiliation_type} where {entity_type} practices:
- Affiliation_ID: NPI number (10 digits) or "N/A"
- Affiliation_Name, Affiliation_NPI
- Affiliation_Address1, City (CAPS), State (2-letter), ZIP (5-digit)

Include ALL known {affiliation_type}s with complete addresses.
"""

    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": user_query}],
        web_search_options={"search_type": "pro" if use_pro_search else "fast"},  
        response_format={
            "type": "json_schema",
            "json_schema": {"schema": SearchResponse.model_json_schema()}
        }
    )

    return json.loads(completion.choices[0].message.content)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_safe_value(record, key, prefix=""):
    """Safely get value from record with optional prefix fallback."""
    val = record.get(key)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        val = record.get(f"{prefix}{key}" if prefix else key)
    return val if val is not None and not (isinstance(val, float) and pd.isna(val)) else 'N/A'

def show_success_popup(popup_placeholder, message_type: str, record_info: dict):
    """Display auto-dismissing success popup."""
    with popup_placeholder.container():
        st.markdown("""
            <style>
                .st-popup-container {
                    position: fixed; top: 20%; left: 50%; transform: translate(-50%, -50%);
                    z-index: 9999; padding: 2rem; border-radius: 10px;
                    background-color: #ffffff; color: #000000;
                    border: 2px solid #4CAF50; box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                    text-align: center; min-width: 350px;
                }
                .st-popup-container h4 { color: #4CAF50; font-size: 1.5rem; margin-bottom: 0.5rem; }
            </style>
        """, unsafe_allow_html=True)
        
        if message_type == "update_success":
            title, message = "Update Successful! ✅", record_info.get('message', '')
        elif message_type == "primary_success":
            title = f"Primary {APP_CONFIG['affiliation_name']} Updated! ✅"
            message = f"Primary set to ID: {record_info.get('affiliation_id')}."
        else:
            title, message = "Success! ✅", "Operation completed."
            
        st.markdown(f'<div class="st-popup-container"><h4>{title}</h4><p>{message}</p></div>', 
                   unsafe_allow_html=True)

    time.sleep(2)
    st.session_state.show_popup = False
    st.session_state.popup_message_info = None
    st.rerun()

# ============================================================================
# CSS STYLES
# ============================================================================

CUSTOM_CSS = """
<style>
    div[data-testid="stHorizontalBlock"]:has(div.cell-content),
    div[data-testid="stHorizontalBlock"]:has(div.affiliation-cell) { 
        border-bottom: 1px solid #e6e6e6; 
    }
    div[data-testid="stHorizontalBlock"]:has(div.cell-content):hover,
    div[data-testid="stHorizontalBlock"]:has(div.affiliation-cell):hover { 
        background-color: #f8f9fa; 
    }
    .cell-content, .affiliation-cell { 
        padding: 0.3rem 0.5rem; font-size: 14px; 
        display: flex; align-items: center; height: 48px; 
    }
    .report-header, .affiliation-header { 
        font-weight: bold; color: #4f4f4f; padding: 0.5rem; 
    }
    .affiliation-header { border-bottom: 2px solid #ccc; }
    .report-proposed-column { 
        border-left: 2px solid #D3D3D3; padding-left: 1.5rem; 
    }
    .checkbox-container { width: 100%; text-align: center; }
    .checkbox-container div[data-testid="stCheckbox"] { padding-top: 12px; }
    .affiliation-cell div[data-testid="stButton"] button { 
        padding: 0.2rem 0.5rem; font-size: 12px; height: 30px; 
    }
    .detail-key { font-weight: bold; color: #4F8BE7; margin-top: 0.5rem; }
    .detail-value { padding-bottom: 0.5rem; }
</style>
"""

# ============================================================================
# UI COMPONENTS
# ============================================================================

@st.dialog("🎯 Priority Reasoning")
def show_priority_dialog():
    """Display priority reasoning dialog."""
    popup_data = st.session_state.get('reason_popup_data', {})
    
    st.markdown(f"**{APP_CONFIG['affiliation_name']}:** {popup_data.get('affiliation_name', 'Unknown')}")
    priority = popup_data.get('priority', '-')
    st.markdown(
        f"<span style='display: inline-block; background-color: #4CAF50; color: white; "
        f"padding: 0.25rem 0.75rem; border-radius: 15px; font-weight: bold;'>"
        f"Priority {priority}</span>", 
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown(f"""
    <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 5px; 
                border-left: 4px solid #1f77b4;'>
        <strong>Reason:</strong><br>{popup_data.get('reason', 'N/A')}
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Close", use_container_width=True):
        st.session_state.show_reason_popup = False
        st.session_state.reason_popup_data = None
        st.rerun()

def display_search_results(session, df: pd.DataFrame):
    """Display search results table with selection."""
    if df.empty:
        st.info(f"No {APP_CONFIG['entity_name_plural'].lower()} found.", icon="ℹ️")
        if st.button(f"🔍 Proceed with Web Search?", type="primary"):
            create_empty_record_and_redirect()
        return
    
    st.session_state.results_df = df
    st.write("Select a record:")
    
    # Headers
    cols = st.columns(SEARCH_RESULT_COL_SIZES)
    for col_obj, header in zip(cols, ["Select"] + SEARCH_RESULT_COLUMNS[1:]):
        col_obj.markdown(f"**{header}**")
    
    # Rows
    for _, row in df.iterrows():
        row_id = row.get("ID") or row.get(f"{APP_CONFIG['entity_name']}_ID") or _
        is_selected = row_id == st.session_state.get("selected_entity_id")
        row_cols = st.columns(SEARCH_RESULT_COL_SIZES)
        
        if is_selected:
            row_cols[0].write("📘")
        else:
            if row_cols[0].button("", key=f"select_{row_id}"):
                st.session_state.selected_entity_id = row_id
                st.rerun()
        
        for i, col_name in enumerate(SEARCH_RESULT_COLUMNS[1:], 1):
            db_col = ENTITY_COLUMNS.get(col_name, col_name.upper())
            row_cols[i].write(get_safe_value(row, db_col))

def display_selected_record(session, record):
    """Display details for selected record."""
    st.markdown(f"### Selected {APP_CONFIG['entity_name']} Details")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Name:** {get_safe_value(record, 'NAME')}")
        st.markdown(f"**NPI:** {get_safe_value(record, 'NPI')}")
        st.markdown(f"**Specialty:** {get_safe_value(record, 'SPECIALTY')}")
    with col2:
        st.markdown(f"**Address:** {get_safe_value(record, 'ADDRESS1')}")
        city = get_safe_value(record, 'CITY')
        state = get_safe_value(record, 'STATE')
        zip_code = get_safe_value(record, 'ZIP')
        st.markdown(f"**City, State ZIP:** {city}, {state} {zip_code}")
    
    st.divider()
    
    col_btn, _ = st.columns([0.25, 0.75])
    with col_btn:
        if st.button("Enrich with AI Assistant 🚀", type="primary"):
            st.session_state.current_view = "enrichment_page"
            st.rerun()

def create_empty_record_and_redirect():
    """Create empty record and redirect to enrichment."""
    st.session_state.empty_record_for_enrichment = {
        k: '' for k in ENTITY_COLUMNS.values()
    }
    st.session_state.web_search_query = st.session_state.get('last_prompt', '')
    st.session_state.selected_entity_id = 'empty_record'
    st.session_state.current_view = "enrichment_page"
    st.rerun()

# ============================================================================
# MAIN PAGE
# ============================================================================

def render_main_page(session):
    """Render the main search page."""
    st.title(f"🏥 {APP_CONFIG['page_title']}")
    st.markdown(f"Search for {APP_CONFIG['entity_name_plural']} and manage {APP_CONFIG['affiliation_name'].lower()} affiliations.")
    
    # Search input
    with st.container(border=True):
        user_input = st.chat_input(APP_CONFIG["search_placeholder"])
        if user_input and user_input != st.session_state.get("last_prompt"):
            process_cortex_message(session, user_input)
            st.session_state.last_prompt = user_input
    
    # Display results
    if st.session_state.messages:
        assistant_msgs = [m for m in st.session_state.messages if m["role"] == "assistant"]
        if assistant_msgs:
            st.markdown("---")
            content = assistant_msgs[-1]["content"]
            
            # Show interpretation
            if len(content) > 1 and content[1].get("type") == "text":
                st.markdown(f'You searched: "{content[0].get("text", "")}"')
                st.markdown(f'Interpretation: "{content[1].get("text", "")}"')
            
            # Search results
            with st.container(border=True):
                st.markdown("## Search Results")
                
                sql_found = False
                for item in content:
                    if item["type"] == "sql":
                        sql_found = True
                        df = session.sql(item["statement"]).to_pandas()
                        display_search_results(session, df)
                        break
                
                if not sql_found:
                    st.info("No SQL query returned. Try rephrasing your search.")
                    if st.button("🔍 Proceed with Web Search?", type="primary"):
                        create_empty_record_and_redirect()
            
            # Selected record details
            if st.session_state.get("selected_entity_id") and st.session_state.results_df is not None:
                st.markdown("---")
                id_col = "ID" if "ID" in st.session_state.results_df.columns else f"{APP_CONFIG['entity_name']}_ID"
                selected = st.session_state.results_df[
                    st.session_state.results_df[id_col].astype(str) == str(st.session_state.selected_entity_id)
                ]
                if not selected.empty:
                    display_selected_record(session, selected.iloc[0])

# ============================================================================
# ENRICHMENT PAGE (Placeholder - Implement as needed)
# ============================================================================

def render_enrichment_page(session, selected_df):
    """Render enrichment/comparison page."""
    _, btn_col = st.columns([4, 1])
    with btn_col:
        if st.button("⬅️ Back to Search"):
            st.session_state.current_view = "main"
            st.session_state.selected_entity_id = None
            st.rerun()
    
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown("### 🔒 Current vs. Proposed Comparison Report")
    
    # Show priority dialog if needed
    if st.session_state.get('show_reason_popup'):
        show_priority_dialog()
    
    if selected_df.empty:
        st.warning(f"No {APP_CONFIG['entity_name']} data provided.")
        return
    
    # Get enriched data
    @st.cache_data(ttl=600)
    def get_enriched_data(_session, entity_df, search_query=None):
        if entity_df.empty:
            return {}
        try:
            return search_entity_web(
                entity_df.iloc[0].to_dict(),
                search_query=search_query
            )
        except Exception as e:
            st.error(f"Enrichment error: {e}")
            return {}
    
    with st.spinner(f"🔍 Searching web for {APP_CONFIG['entity_name']} info..."):
        search_query = st.session_state.get('web_search_query')
        api_response = get_enriched_data(session, selected_df, search_query)
    
    if api_response:
        st.success(f"✅ Found enriched data for {APP_CONFIG['entity_name']}!")
        # Add your comparison UI logic here
        st.info("Implement comparison UI with demographic and affiliation sections")

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

st.set_page_config(layout=APP_CONFIG["page_layout"], page_title=APP_CONFIG["page_title"])

# Initialize session state
SESSION_DEFAULTS = {
    "messages": [],
    "results_df": None,
    "selected_entity_id": None,
    "current_view": "main",
    "last_prompt": None,
    "show_popup": False,
    "popup_message_info": None,
    "show_reason_popup": False,
    "reason_popup_data": None,
    "priority_rankings_cache": {},
}

for key, default in SESSION_DEFAULTS.items():
    st.session_state.setdefault(key, default)

# Get session
session = get_snowflake_session()

# Route to appropriate page
if st.session_state.current_view == "main":
    render_main_page(session)
elif st.session_state.current_view == "enrichment_page":
    popup_placeholder = st.empty()
    
    if st.session_state.show_popup:
        show_success_popup(
            popup_placeholder, 
            st.session_state.popup_message_info['type'], 
            st.session_state.popup_message_info
        )
    
    # Handle empty record or selected record
    if st.session_state.selected_entity_id == 'empty_record':
        empty_df = pd.DataFrame([st.session_state.get('empty_record_for_enrichment', {})])
        if not st.session_state.show_popup:
            render_enrichment_page(session, empty_df)
    elif st.session_state.selected_entity_id and st.session_state.results_df is not None:
        id_col = "ID" if "ID" in st.session_state.results_df.columns else f"{APP_CONFIG['entity_name']}_ID"
        selected_df = st.session_state.results_df[
            st.session_state.results_df[id_col].astype(str) == str(st.session_state.selected_entity_id)
        ]
        if not st.session_state.show_popup:
            render_enrichment_page(session, selected_df)
    else:
        st.warning(f"Select a {APP_CONFIG['entity_name']} record first.")
        if st.button("Back to Main"):
            st.session_state.current_view = "main"
            st.rerun()
