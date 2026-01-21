"""
HCP (Healthcare Professional) Data Steward Application Template
======================================================================
Modular, theme-aware template for data enrichment and stewardship.
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
# CONFIGURATION SECTION (MODULAR)
# ============================================================================

APP_CONFIG = {
    "page_title": "HCP Data Steward",
    "page_layout": "wide",
    "search_placeholder": "Search for a Doctor/Healthcare Professional",
    "entity_name": "HCP",
    "affiliation_name": "Hospital",
}

# Table Configuration: Header Names and Column Widths
ENRICHMENT_CONFIG = {
    # 1. Demographics Comparison Table
    "demographics": [
        {"label": "Name", "db_col": "NAME", "api_field": "Name"},
        {"label": "NPI", "db_col": "NPI", "api_field": "NPI"},
        {"label": "Specialty", "db_col": "SPECIALTY", "api_field": "Specialty"},
        {"label": "Address Line 1", "db_col": "ADDRESS1", "api_field": "Address Line1"},
        {"label": "City", "db_col": "CITY", "api_field": "City"},
        {"label": "State", "db_col": "STATE", "api_field": "State"},
        {"label": "ZIP", "db_col": "ZIP", "api_field": "ZIP"},
    ],
    "demo_col_widths": [1.5, 2.5, 2.5, 1.2, 0.8], # Label, DB, Web, Source, Check
    
    # 2. Affiliation Table
    "affiliations": {
        "headers": ["Status", "Source", "ID", "Name", "Address", "City", "State", "ZIP", "Priority", "Details"],
        "widths": [0.8, 1, 1, 2.5, 2, 1.2, 0.6, 0.8, 0.8, 0.8]
    },
    
    # 3. Search Result Table (Main Page)
    "search_results": {
        "cols": ["ID", "Name", "NPI", "Specialty", "City", "State"],
        "widths": [0.8, 2, 1.2, 1.5, 1.2, 0.8]
    }
}

# DB Mappings for generic value fetching
ENTITY_COLUMNS = {k: v for d in [
    {"ID": "ID", "Name": "NAME", "NPI": "NPI", "Specialty": "SPECIALTY"},
    {"Address Line1": "ADDRESS1", "City": "CITY", "State": "STATE", "ZIP": "ZIP"}
] for k, v in d.items()}

# ============================================================================
# END CONFIGURATION SECTION
# ============================================================================

os.environ["PERPLEXITY_API_KEY"] = st.secrets["perplexity"]["api_key"]

# ============================================================================
# PYDANTIC MODELS (API CONTRACT)
# ============================================================================

class EntityData(BaseModel):
    Name: List[str]
    Address_Line1: List[str] = Field(..., alias="Address Line1")
    ZIP: List[str]
    City: List[str]
    State: List[str]
    Specialty: List[str] = []
    NPI: List[str] = []

class AffiliationData(BaseModel):
    Affiliation_ID: List[str]
    Affiliation_Name: List[str]
    Affiliation_Address1: List[str]
    Affiliation_City: List[str]
    Affiliation_State: List[str]
    Affiliation_ZIP: List[str]

class SearchResponse(BaseModel):
    entity_data: EntityData
    affiliation_data: AffiliationData

# ============================================================================
# SNOWFLAKE & LLM LOGIC
# ============================================================================

@st.cache_resource
def get_snowflake_session():
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

def prepare_affiliations_for_llm(proposed_affiliations: List[dict]) -> List[dict]:
    """
    Cleans and standardizes web search data to ensure the LLM doesn't receive 'N/A'.
    This addresses the data serialization mismatch.
    """
    cleaned = []
    for aff in proposed_affiliations:
        cleaned.append({
            "Affiliation_Name": aff.get("Affiliation_Name", "Unknown"),
            "Affiliation_Address": aff.get("Affiliation_Address1", "N/A"),
            "Affiliation_City": aff.get("Affiliation_City", "N/A"),
            "Affiliation_State": aff.get("Affiliation_State", "N/A"),
            "Source": "Web Search"
        })
    return cleaned

def get_affiliation_priorities(session, entity_data: dict, affiliations: List[dict]) -> dict:
    """
    Calls Snowflake Cortex to rank affiliations based on the selected record.
    """
    if not affiliations:
        return {}

    # Format Candidate Data (The web results)
    candidates_str = ""
    for idx, aff in enumerate(affiliations):
        candidates_str += f"""
Candidate {idx+1} (Key: {aff.get('Affiliation_Name')}):
- Name: {aff.get('Affiliation_Name')}
- Location: {aff.get('Affiliation_Address')}, {aff.get('Affiliation_City')}
"""

    prompt = f"""You are a data analyst. Rank these {APP_CONFIG['affiliation_name']} affiliations for the following professional:
    
Professional: {entity_data.get('NAME', 'N/A')}
Location: {entity_data.get('ADDRESS1', '')}, {entity_data.get('CITY', '')}

Candidates:
{candidates_str}

Return JSON: {{ "rankings": [ {{ "key": "Candidate Name", "priority": 1, "reason": "..." }} ] }}"""

    try:
        # Cortex Inference call (Simplified representation)
        api_url = f"https://{st.secrets['snowflake']['account'].replace('.','-')}.snowflakecomputing.com/api/v2/cortex/inference:complete"
        headers = {"Authorization": f"Snowflake Token=\"{session.connection.rest.token}\"", "Content-Type": "application/json"}
        
        request_body = {
            "model": "claude-3-5-sonnet",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json"}
        }
        
        resp = requests.post(api_url, headers=headers, json=request_body, timeout=60)
        # Parse logic for Snowflake's streaming response format...
        # (Assuming success and extraction of JSON text)
        result = json.loads(resp.text) 
        return {r["key"]: {"priority": r["priority"], "reason": r["reason"]} for r in result.get("rankings", [])}
    except:
        return {a["Affiliation_Name"]: {"priority": "-", "reason": "Analysis unavailable"} for a in affiliations}

# ============================================================================
# UI STYLING (Dark Mode Optimized)
# ============================================================================

CUSTOM_CSS = """
<style>
    .report-row-separator { border-bottom: 1px solid rgba(128,128,128,0.2); margin: 8px 0; }
    .report-header { font-weight: 600; opacity: 0.8; font-size: 0.85rem; text-transform: uppercase; }
    
    /* Transparent Green/Border for highlights (works in Dark and Light mode) */
    .proposed-val {
        color: #4CAF50;
        border: 1px solid rgba(76, 175, 80, 0.5);
        background-color: rgba(76, 175, 80, 0.1);
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .cell-text { font-size: 0.9rem; opacity: 0.9; }
</style>
"""

# ============================================================================
# UI COMPONENTS
# ============================================================================

@st.dialog("🎯 Priority Detail")
def show_priority_dialog():
    data = st.session_state.reason_popup_data
    st.subheader(data['name'])
    st.markdown(f":green[**Priority {data['priority']}**]")
    st.divider()
    st.write(data['reason'])

def render_comparison_section(current_record, proposed_entity, record_id):
    st.subheader("👤 Demographic Enrichment")
    
    # Headers
    cols = st.columns(ENRICHMENT_CONFIG["demo_col_widths"], vertical_alignment="bottom")
    for col_obj, header in zip(cols, ["Field", "Current (DB)", "Proposed (Web)", "Source", "Apply"]):
        col_obj.markdown(f"<div class='report-header'>{header}</div>", unsafe_allow_html=True)
    st.divider()

    for row in ENRICHMENT_CONFIG["demographics"]:
        db_val = str(current_record.get(row['db_col'], 'N/A'))
        web_val_raw = proposed_entity.get(row['api_field'], "")
        web_val = str(web_val_raw[0]) if isinstance(web_val_raw, list) and web_val_raw else str(web_val_raw)
        
        r_cols = st.columns(ENRICHMENT_CONFIG["demo_col_widths"], vertical_alignment="center")
        r_cols[0].markdown(f"**{row['label']}**")
        r_cols[1].markdown(f"<span class='cell-text'>{db_val}</span>", unsafe_allow_html=True)
        
        # Comparison logic
        is_diff = db_val.strip().lower() != web_val.strip().lower() and web_val
        if is_diff:
            r_cols[2].markdown(f"<span class='proposed-val'>{web_val}</span>", unsafe_allow_html=True)
        else:
            r_cols[2].markdown(f"<span class='cell-text'>{web_val}</span>", unsafe_allow_html=True)
            
        r_cols[3].caption("Perplexity")
        r_cols[4].checkbox("Update", key=f"upd_{record_id}_{row['db_col']}", label_visibility="collapsed")
        st.markdown("<div class='report-row-separator'></div>", unsafe_allow_html=True)

def render_affiliation_section(session, current_record, proposed_affiliations):
    st.subheader(f"🏥 {APP_CONFIG['affiliation_name']} Affiliations")
    
    if st.button("🎯 Analyze Priorities with LLM", type="secondary"):
        with st.spinner("AI is ranking candidates..."):
            cleaned_data = prepare_affiliations_for_llm(proposed_affiliations)
            rankings = get_affiliation_priorities(session, current_record, cleaned_data)
            st.session_state.priority_rankings = rankings
            st.rerun()

    # Table
    rankings = st.session_state.get('priority_rankings', {})
    header_cfg = ENRICHMENT_CONFIG["affiliations"]
    
    h_cols = st.columns(header_cfg["widths"], vertical_alignment="bottom")
    for col_obj, name in zip(h_cols, header_cfg["headers"]):
        col_obj.markdown(f"<div class='report-header'>{name}</div>", unsafe_allow_html=True)
    st.divider()

    for aff in proposed_affiliations:
        name = aff.get("Affiliation_Name", "")
        p_info = rankings.get(name, {"priority": "-", "reason": "Not analyzed"})
        
        r_cols = st.columns(header_cfg["widths"], vertical_alignment="center")
        r_cols[0].markdown(":green[**New**]")
        r_cols[1].caption("Web")
        r_cols[2].write(aff.get("Affiliation_ID", "N/A"))
        r_cols[3].write(name)
        r_cols[4].write(aff.get("Affiliation_Address1", "N/A"))
        r_cols[5].write(aff.get("Affiliation_City", "N/A"))
        r_cols[6].write(aff.get("Affiliation_State", "N/A"))
        r_cols[7].write(aff.get("Affiliation_ZIP", "N/A"))
        r_cols[8].write(f"**{p_info['priority']}**")
        
        if r_cols[9].button("ℹ️", key=f"info_{name}"):
            st.session_state.reason_popup_data = {"name": name, "priority": p_info['priority'], "reason": p_info['reason']}
            show_priority_dialog()
            
        st.markdown("<div class='report-row-separator'></div>", unsafe_allow_html=True)

# ============================================================================
# MAIN PAGE & ROUTING
# ============================================================================

st.set_page_config(layout="wide", page_title=APP_CONFIG["page_title"])
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize Session State
for key, val in {"current_view": "main", "selected_id": None, "results_df": None, "priority_rankings": {}}.items():
    if key not in st.session_state: st.session_state[key] = val

session = get_snowflake_session()

if st.session_state.current_view == "main":
    st.title(f"🏥 {APP_CONFIG['page_title']}")
    user_query = st.chat_input("Search for HCP...")
    
    if user_query:
        # Cortex Analyst Simulation
        st.session_state.results_df = pd.DataFrame([
            {"ID": "101", "NAME": "Dr. Smith", "NPI": "123456", "SPECIALTY": "Cardiology", "CITY": "Boston", "STATE": "MA", "ADDRESS1": "123 Main St"}
        ])

    if st.session_state.results_df is not None:
        cfg = ENRICHMENT_CONFIG["search_results"]
        cols = st.columns(cfg["widths"], vertical_alignment="bottom")
        for c, h in zip(cols, ["Action"] + cfg["cols"][1:]): c.markdown(f"**{h}**")
        
        for _, row in st.session_state.results_df.iterrows():
            r_cols = st.columns(cfg["widths"], vertical_alignment="center")
            if r_cols[0].button("Select", key=row["ID"]):
                st.session_state.selected_id = row["ID"]
                st.session_state.current_view = "enrichment"
                st.rerun()
            r_cols[1].write(row["NAME"])
            r_cols[2].write(row["NPI"])
            r_cols[3].write(row["SPECIALTY"])
            r_cols[4].write(row["CITY"])
            r_cols[5].write(row["STATE"])

elif st.session_state.current_view == "enrichment":
    if st.button("⬅️ Back"):
        st.session_state.current_view = "main"
        st.rerun()

    # Mocking Perplexity response for demonstration
    mock_web = {
        "entity_data": {"Name": ["Dr. John Smith"], "Address Line1": ["456 Medical Pkwy"], "City": ["Boston"], "State": ["MA"], "ZIP": ["02111"]},
        "affiliation_data": {
            "Affiliation_Name": ["General Hospital", "Heart Clinic"],
            "Affiliation_ID": ["987", "654"],
            "Affiliation_Address1": ["100 Health Way", "200 Cardiac Rd"],
            "Affiliation_City": ["Boston", "Boston"], "Affiliation_State": ["MA", "MA"], "Affiliation_ZIP": ["02114", "02115"]
        }
    }
    
    # Process local record
    sel_id = st.session_state.selected_id
    curr_rec = st.session_state.results_df[st.session_state.results_df["ID"] == sel_id].iloc[0].to_dict()
    
    # Transpose Affiliations from API format (Columns) to Row format
    aff_list = []
    raw_affs = mock_web["affiliation_data"]
    for i in range(len(raw_affs["Affiliation_Name"])):
        aff_list.append({k: v[i] for k, v in raw_affs.items()})

    st.title("📑 Comparison Report")
    render_comparison_section(curr_rec, mock_web["entity_data"], sel_id)
    st.markdown("<br>", unsafe_allow_html=True)
    render_affiliation_section(session, curr_rec, aff_list)
