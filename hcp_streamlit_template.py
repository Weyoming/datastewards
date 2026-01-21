"""
HCP (Healthcare Professional/Doctor) Data Steward Application Template
======================================================================
Enhanced template with improved composability and integrated features.
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
from typing import List, Dict, Optional, Tuple, Any
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

# Main Page Configuration
MAIN_PAGE_TABLES_CONFIG = {
    "search_results": [{}],
    "demographics": [{}],
    "affiliations": [{}],
}

# Enrichment Page Configuration
ENRICHMENT_PAGE_TABLES_CONFIG = {
    # Demographics Comparison Table
    "demographics": [
        {"label": "Name", "db_col": "NAME", "api_field": "Name"},
        {"label": "NPI", "db_col": "NPI", "api_field": "NPI"},
        {"label": "Specialty", "db_col": "SPECIALTY", "api_field": "Specialty"},
        {"label": "Address Line 1", "db_col": "ADDRESS1", "api_field": "Address Line1"},
        {"label": "Address Line 2", "db_col": "ADDRESS2", "api_field": "Address Line2"},
        {"label": "City", "db_col": "CITY", "api_field": "City"},
        {"label": "State", "db_col": "STATE", "api_field": "State"},
        {"label": "ZIP", "db_col": "ZIP", "api_field": "ZIP"},
    ],
    # Adjusted widths: Label, Current, Proposed, Source, Action
    "demographics_col_sizes": (1.5, 2.5, 2.5, 1.5, 1), 
    
    # Affiliation Table
    "affiliations_headers": [
        "Status", "Source", "ID", "Name", "Address", "City", "State", "ZIP", "Priority", "Details"
    ],
    "affiliations_col_sizes": (0.8, 1, 1, 2.5, 2, 1.2, 0.6, 0.8, 0.8, 0.8),
}

# Cortex Analyst Configuration
CORTEX_CONFIG = {
    "database": st.secrets["snowflake"]["database"],
    "schema": st.secrets["snowflake"]["schema"], 
    "stage": st.secrets["snowflake"]["semantic_model_stage"],
    "semantic_model_file": st.secrets["snowflake"]["semantic_model_file"],
}

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
    
    affiliations_info = ""
    for idx, (key, aff) in enumerate(affiliations):
        aff_name = aff.get('Affiliation_Name', aff.get('HOSPITAL_NAME', 'N/A'))
        aff_addr = aff.get('Affiliation_Address', aff.get('HOSPITAL_ADDRESS', 'N/A'))
        aff_city = aff.get('Affiliation_City', aff.get('HOSPITAL_CITY', 'N/A'))
        
        affiliations_info += f"""
{affiliation_type.title()} {idx + 1} (Key: {key}):
- Name: {aff_name}
- Address: {aff_addr}, {aff_city}
- Source: {aff.get('SOURCE', 'N/A')}
"""
    
    prompt = f"""You are a healthcare data analyst. Rank each {affiliation_type} by priority (1=highest) based on:
1. Geographic proximity (same city, state, ZIP)
2. Name similarity (same health system)
3. Address proximity

{entity_info}

{affiliations_info}

Return JSON only: {{ "rankings": [ {{ "key": "...", "priority": 1, "reason": "..." }} ] }}"""

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
    
    if hasattr(entity_data, 'to_dict'):
        entity_data = entity_data.to_dict()
    
    def get_val(key):
        val = entity_data.get(key, '')
        if not val:
            val = entity_data.get(f"{APP_CONFIG['entity_name']}_{key}", '')
        return val or ''
    
    entity_name = get_val('NAME') or search_query or ''
    
    user_query = f"""
You are a healthcare data research specialist. Search the web for this US {entity_type}:

**{entity_type.title()} to Research:**
- Name: {entity_name}
- NPI: {get_val('NPI')}
- Address: {get_val('ADDRESS1')}
- City: {get_val('CITY')}
- State: {get_val('STATE')}

**Part 1 - {entity_type.title()} Details:**
- Name, NPI, Specialty
- Address Line1, Address Line2 (or empty)
- City (ALL CAPS), State (2-letter), ZIP (5-digit)

**Part 2 - {affiliation_type.title()} Affiliations:**
For each {affiliation_type} where {entity_type} practices:
- Affiliation_ID: NPI number (10 digits) or "N/A"
- Affiliation_Name, Affiliation_NPI
- Affiliation_Address1, City (CAPS), State (2-letter), ZIP (5-digit)
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

def standardize_list(val):
    if isinstance(val, list):
        return val[0] if val else ""
    return val

# ============================================================================
# CSS STYLES (Enhanced for Dark Mode)
# ============================================================================

CUSTOM_CSS = """
<style>
    /* Table row separator - subtle in both modes */
    .report-row-separator {
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        margin-top: 5px;
        margin-bottom: 5px;
    }

    /* Header text styling - inherit color for dark/light compatibility */
    .report-header {
        font-weight: 600;
        opacity: 0.9;
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
    }

    /* Proposed value highlighting - Dark Mode Friendly 
       Using transparent background + border instead of solid pastel
    */
    .proposed-val {
        color: #4CAF50; /* Brighter green visible on dark & light */
        border: 1px solid #4CAF50;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: 600;
        background-color: rgba(76, 175, 80, 0.1); /* Subtle transparent background */
        font-size: 0.9rem;
    }
    
    /* Standard cell text */
    .cell-text {
        font-size: 0.9rem;
        word-wrap: break-word;
        opacity: 0.9;
    }

    /* Remove button padding quirks */
    div[data-testid="stButton"] button {
        border-radius: 6px;
    }
    
    /* Fix checkbox alignment in columns */
    div[data-testid="stCheckbox"] {
        padding-top: 0px; 
    }
</style>
"""

# ============================================================================
# UI COMPONENTS
# ============================================================================

@st.dialog("🎯 Priority Reasoning")
def show_priority_dialog():
    """Display priority reasoning dialog."""
    popup_data = st.session_state.get('reason_popup_data', {})
    
    st.subheader(f"{popup_data.get('affiliation_name', 'Unknown')}")
    priority = popup_data.get('priority', '-')
    
    st.markdown(
        f"<span style='display: inline-block; border: 1px solid #4CAF50; color: #4CAF50; "
        f"padding: 0.25rem 0.75rem; border-radius: 15px; font-weight: bold; background-color: rgba(76, 175, 80, 0.1);'>"
        f"Priority {priority}</span>", 
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.write(popup_data.get('reason', 'N/A'))

@st.dialog("⚠️ Confirm Updates")
def show_confirm_update_dialog():
    """Dialog to confirm changes before DB update."""
    changes = st.session_state.get('pending_changes', [])
    
    if not changes:
        st.info("No changes selected.")
        if st.button("Close"):
            st.session_state.show_confirm_dialog = False
            st.rerun()
        return

    st.warning("Are you sure you want to apply these updates?", icon="⚠️")
    
    st.markdown("### Pending Changes")
    for change in changes:
        # Use vertical_alignment="center" here too
        col1, col2, col3 = st.columns([1.5, 2, 2], vertical_alignment="center")
        col1.markdown(f"**{change['field']}**")
        col2.markdown(f"Current: `{change['current']}`")
        col3.markdown(f"New: :green[`{change['proposed']}`]")
        st.divider()
        
    col_submit, col_cancel = st.columns(2)
    with col_submit:
        if st.button("✅ Yes, Update Database", type="primary", use_container_width=True):
            st.toast("Updated successfully! (Simulation)", icon="💾")
            st.session_state.show_confirm_dialog = False
            st.session_state.pending_changes = []
            st.rerun()
            
    with col_cancel:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.show_confirm_dialog = False
            st.rerun()

# ============================================================================
# MAIN PAGE
# ============================================================================

def render_main_page(session):
    """Render the main search page."""
    st.title(f"🏥 {APP_CONFIG['page_title']}")
    
    with st.container():
        user_input = st.chat_input(APP_CONFIG["search_placeholder"])
        if user_input and user_input != st.session_state.get("last_prompt"):
            process_cortex_message(session, user_input)
            st.session_state.last_prompt = user_input
    
    if st.session_state.messages:
        assistant_msgs = [m for m in st.session_state.messages if m["role"] == "assistant"]
        if assistant_msgs:
            st.markdown("---")
            content = assistant_msgs[-1]["content"]
            
            if len(content) > 1 and content[1].get("type") == "text":
                st.markdown(f'**Interpretation:** "{content[1].get("text", "")}"')
            
            # Removed border=True
            with st.container():
                st.subheader("Search Results")
                
                sql_found = False
                for item in content:
                    if item["type"] == "sql":
                        sql_found = True
                        df = session.sql(item["statement"]).to_pandas()
                        
                        if df.empty:
                            st.info("No records found in database.")
                        else:
                            st.session_state.results_df = df
                            display_search_results(session, df)
                        break
                
                if not sql_found:
                    st.info("No SQL query returned. Try rephrasing your search.")
            
            if not sql_found or (st.session_state.results_df is not None and st.session_state.results_df.empty):
                st.markdown("---")
                if st.button(f"🔍 Proceed with Web Search for '{st.session_state.last_prompt}'?", type="primary"):
                    create_empty_record_and_redirect()

def display_search_results(session, df: pd.DataFrame):
    """Display search results table with selection."""
    st.write("Select a record to enrich:")
    
    # Vertical alignment for main search table too
    cols = st.columns(SEARCH_RESULT_COL_SIZES, vertical_alignment="bottom")
    for col_obj, header in zip(cols, ["Select"] + SEARCH_RESULT_COLUMNS[1:]):
        col_obj.markdown(f"**{header}**")
    
    for _, row in df.iterrows():
        row_id = row.get("ID") or row.get(f"{APP_CONFIG['entity_name']}_ID")
        is_selected = str(row_id) == str(st.session_state.get("selected_entity_id"))
        
        row_cols = st.columns(SEARCH_RESULT_COL_SIZES, vertical_alignment="center")
        
        if is_selected:
            row_cols[0].write("✅")
        else:
            if row_cols[0].button("Select", key=f"select_{row_id}"):
                st.session_state.selected_entity_id = row_id
                st.rerun()
        
        for i, col_name in enumerate(SEARCH_RESULT_COLUMNS[1:], 1):
            db_col = ENTITY_COLUMNS.get(col_name, col_name.upper())
            row_cols[i].write(get_safe_value(row, db_col))
        
        st.markdown("<div class='report-row-separator'></div>", unsafe_allow_html=True)

    if st.session_state.get("selected_entity_id"):
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Enrich Selected Record 🚀", type="primary"):
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
# ENRICHMENT PAGE
# ============================================================================

def render_enrichment_page(session, selected_df):
    """Render the modular enrichment comparison page."""
    
    # Navigation
    _, btn_col = st.columns([4, 1])
    with btn_col:
        if st.button("⬅️ Back to Search"):
            st.session_state.current_view = "main"
            st.session_state.priority_rankings = {} 
            st.rerun()

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    if st.session_state.get('show_confirm_dialog'):
        show_confirm_update_dialog()
        
    if selected_df.empty:
        st.error("No record data available.")
        return

    current_record = selected_df.iloc[0].to_dict()
    record_id = get_safe_value(current_record, "ID", "")
    
    @st.cache_data(ttl=600)
    def get_enriched_data(_session, record_dict, search_query=None):
        try:
            return search_entity_web(record_dict, search_query=search_query)
        except Exception as e:
            st.error(f"Enrichment error: {e}")
            return None

    with st.spinner(f"🔍 Searching web for {APP_CONFIG['entity_name']} info..."):
        search_query = st.session_state.get('web_search_query')
        api_response = get_enriched_data(session, current_record, search_query)

    if not api_response:
        st.warning("Could not retrieve data from web search.")
        return

    proposed_entity = api_response.get("entity_data", {})
    raw_aff_data = api_response.get("affiliation_data", {})
    proposed_affiliations = []
    
    if raw_aff_data and "Affiliation_ID" in raw_aff_data:
        count = len(raw_aff_data["Affiliation_ID"])
        for i in range(count):
            row = {}
            for key, val_list in raw_aff_data.items():
                if isinstance(val_list, list) and len(val_list) > i:
                    row[key] = val_list[i]
                else:
                    row[key] = ""
            proposed_affiliations.append(row)

    st.title("📑 Comparison Report")
    st.info(f"Comparing Database Record (ID: {record_id}) vs. Web Search Results")
    
    # Removed border=True
    with st.container():
        render_comparison_section(current_record, proposed_entity, record_id)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 Update Selected Fields", type="primary"):
            changes = []
            for row in ENRICHMENT_PAGE_TABLES_CONFIG["demographics"]:
                key = f"approve_{record_id}_{row['db_col']}"
                if st.session_state.get(key, False):
                    changes.append({
                        "field": row['label'],
                        "id": record_id,
                        "current": get_safe_value(current_record, row['db_col']),
                        "proposed": standardize_list(proposed_entity.get(row['api_field']))
                    })
            
            st.session_state.pending_changes = changes
            st.session_state.show_confirm_dialog = True
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Removed border=True
    with st.container():
        render_affiliation_section(session, current_record, proposed_affiliations)

def render_comparison_section(current_record: dict, proposed_data: dict, record_id: str):
    """
    Renders the demographic comparison table.
    Uses vertical_alignment="center" for alignment.
    """
    st.subheader("👤 Demographic Details")
    
    # Headers
    headers = ["Field", "Current Value", "Proposed Value", "Source", "Update?"]
    cols = st.columns(ENRICHMENT_PAGE_TABLES_CONFIG["demographics_col_sizes"], vertical_alignment="bottom")
    for col_obj, header in zip(cols, headers):
        col_obj.markdown(f"<div class='report-header'>{header}</div>", unsafe_allow_html=True)
        
    st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)

    # Rows
    for row_config in ENRICHMENT_PAGE_TABLES_CONFIG["demographics"]:
        label = row_config["label"]
        db_col = row_config["db_col"]
        api_field = row_config["api_field"]
        
        current_val = get_safe_value(current_record, db_col)
        proposed_raw = proposed_data.get(api_field, "")
        proposed_val = standardize_list(proposed_raw)
        
        # Render Row with Vertical Alignment
        r_cols = st.columns(ENRICHMENT_PAGE_TABLES_CONFIG["demographics_col_sizes"], vertical_alignment="center")
        
        r_cols[0].markdown(f"**{label}**")
        r_cols[1].markdown(f"<span class='cell-text'>{current_val}</span>", unsafe_allow_html=True)
        
        # Highlight logic
        if str(current_val).strip() != str(proposed_val).strip() and proposed_val:
            r_cols[2].markdown(f"<span class='proposed-val cell-text'>{proposed_val}</span>", unsafe_allow_html=True)
        else:
            r_cols[2].markdown(f"<span class='cell-text'>{proposed_val}</span>", unsafe_allow_html=True)
            
        r_cols[3].caption("Web Search")
        
        checkbox_key = f"approve_{record_id}_{db_col}"
        r_cols[4].checkbox("Approve", key=checkbox_key, label_visibility="collapsed")
        
        # Row separator
        st.markdown("<div class='report-row-separator'></div>", unsafe_allow_html=True)

def render_affiliation_section(session, current_record: dict, proposed_affiliations: List[dict]):
    """
    Renders the affiliations table.
    Uses vertical_alignment="center" for alignment.
    """
    st.subheader(f"🏥 {APP_CONFIG['affiliation_name']} Affiliations")
    
    # Data Processing (Same as before)
    all_affiliations = {}
    
    # Add Proposed (AI) Affiliations
    for item in proposed_affiliations:
        aff_id = standardize_list(item.get("Affiliation_ID", "N/A"))
        key = aff_id if aff_id != "N/A" else f"new_{item.get('Affiliation_Name', [''])[0]}"
        
        all_affiliations[key] = {
            "ID": aff_id,
            "Name": standardize_list(item.get("Affiliation_Name")),
            "Address": standardize_list(item.get("Affiliation_Address1")),
            "City": standardize_list(item.get("Affiliation_City")),
            "State": standardize_list(item.get("Affiliation_State")),
            "ZIP": standardize_list(item.get("Affiliation_ZIP")),
            "Source": "AI Found"
        }

    # Analysis Button
    if all_affiliations and st.button("🎯 Analyze Priority Order with AI"):
        with st.spinner("Analyzing priorities..."):
            aff_list_for_api = [(k, v) for k, v in all_affiliations.items()]
            rankings = get_affiliation_priorities(session, current_record, aff_list_for_api)
            st.session_state.priority_rankings = rankings
            st.rerun()

    rankings = st.session_state.get('priority_rankings', {})

    # Table Header - aligned bottom
    cols = st.columns(ENRICHMENT_PAGE_TABLES_CONFIG["affiliations_col_sizes"], vertical_alignment="bottom")
    for col, header in zip(cols, ENRICHMENT_PAGE_TABLES_CONFIG["affiliations_headers"]):
        col.markdown(f"**{header}**")
        
    st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)

    # Render Rows
    sorted_items = sorted(
        all_affiliations.items(),
        key=lambda x: rankings.get(x[0], {}).get("priority", 999)
    )

    for key, data in sorted_items:
        # Vertical Alignment Center for Rows
        r_cols = st.columns(ENRICHMENT_PAGE_TABLES_CONFIG["affiliations_col_sizes"], vertical_alignment="center")
        
        r_cols[0].markdown(":green[**New**]")
        r_cols[1].caption(data['Source'])
        r_cols[2].write(data['ID'])
        r_cols[3].write(data['Name'])
        r_cols[4].write(data['Address'])
        r_cols[5].write(data['City'])
        r_cols[6].write(data['State'])
        r_cols[7].write(data['ZIP'])
        
        p_info = rankings.get(key, {})
        priority = p_info.get("priority", "-")
        r_cols[8].markdown(f"**{priority}**" if priority != "-" else "-")
        
        if p_info:
            if r_cols[9].button("ℹ️", key=f"reason_{key}"):
                st.session_state.reason_popup_data = {
                    "affiliation_name": data['Name'],
                    "priority": priority,
                    "reason": p_info.get("reason")
                }
                show_priority_dialog()
        
        st.markdown("<div class='report-row-separator'></div>", unsafe_allow_html=True)

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

st.set_page_config(layout=APP_CONFIG["page_layout"], page_title=APP_CONFIG["page_title"])

SESSION_DEFAULTS = {
    "messages": [],
    "results_df": None,
    "selected_entity_id": None,
    "current_view": "main",
    "last_prompt": None,
    "priority_rankings": {},
    "show_confirm_dialog": False,
    "pending_changes": []
}

for key, default in SESSION_DEFAULTS.items():
    st.session_state.setdefault(key, default)

session = get_snowflake_session()

if st.session_state.current_view == "main":
    render_main_page(session)

elif st.session_state.current_view == "enrichment_page":
    if st.session_state.selected_entity_id == 'empty_record':
        empty_df = pd.DataFrame([st.session_state.get('empty_record_for_enrichment', {})])
        render_enrichment_page(session, empty_df)
    elif st.session_state.selected_entity_id and st.session_state.results_df is not None:
        id_col = "ID" if "ID" in st.session_state.results_df.columns else f"{APP_CONFIG['entity_name']}_ID"
        selected_df = st.session_state.results_df[
            st.session_state.results_df[id_col].astype(str) == str(st.session_state.selected_entity_id)
        ]
        render_enrichment_page(session, selected_df)
    else:
        st.warning("No record selected.")
        if st.button("Return to Search"):
            st.session_state.current_view = "main"
            st.rerun()
