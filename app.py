"""
app.py - DataSense AI: Intelligent Data Analyst Agent.

Enterprise SaaS Interface:
- Palette: #F6F8FB canvas, #142338 dark navy sidebar, #FFFFFF cards, #17243A deep navy text, #159A91 teal primary accent, #5369D8 secondary accent.
- Preserves full existing backend: LangChain tools, LangGraph state graph, Pandas analytics, and ChromaDB RAG.
- Sidebar Navigation: Clearly distinguishable navigation for Ask Data, Data Explorer, and Document Library.
- Fixed Bottom Chat Input: Chat input remains permanently anchored at the bottom, never moving or splitting messages.
"""

import os
import io
import json
import re
from typing import Any, List, Dict
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from rag_utils import load_pdf_from_path_or_stream, split_documents, build_vector_store
from agent import build_analyst_agent, SYSTEM_PROMPT

# Load environment variables
load_dotenv()

# Page Setup
st.set_page_config(
    page_title="DataSense AI - Intelligent Data Analyst",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Enterprise SaaS Styling
st.markdown("""
<style>
    /* Font & Global Reset */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Viewport clearance: prevents any top header cut-off */
    .block-container {
        padding-top: 4.2rem !important;
        padding-bottom: 5.5rem !important;
        max-width: 1120px !important;
    }

    /* Core Canvas */
    .stApp {
        background-color: #F6F8FB;
    }

    /* Dark Navy Enterprise Sidebar (#142338) */
    [data-testid="stSidebar"] {
        background-color: #142338 !important;
        border-right: 1px solid #1E314B;
    }
    [data-testid="stSidebar"] * {
        color: #E2E8F0;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
        color: #F8FAFC !important;
        font-size: 0.76rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        margin-top: 1.1rem !important;
        margin-bottom: 0.5rem !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
        color: #94A3B8 !important;
        font-size: 0.84rem !important;
    }

    /* Top Navigation Segmented Control Styling */
    [data-testid="stSegmentedControl"] {
        display: flex;
        justify-content: flex-end;
    }
    [data-testid="stSegmentedControl"] button {
        border-radius: 8px !important;
        font-size: 0.86rem !important;
        font-weight: 600 !important;
        padding: 6px 16px !important;
        transition: all 0.15s ease-in-out !important;
    }

    /* Sidebar Inputs */
    [data-testid="stSidebar"] div[data-baseweb="input"],
    [data-testid="stSidebar"] div[data-baseweb="base-input"] {
        background-color: #1A2C44 !important;
        border: 1px solid #273E5D !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    [data-testid="stSidebar"] div[data-baseweb="input"]:focus-within {
        border-color: #159A91 !important;
        box-shadow: 0 0 0 1px #159A91 !important;
    }
    [data-testid="stSidebar"] div[data-baseweb="input"] input {
        background-color: transparent !important;
        color: #F8FAFC !important;
        border: none !important;
        font-size: 0.85rem !important;
        padding: 8px 12px !important;
    }
    [data-testid="stSidebar"] div[data-baseweb="input"] button {
        background-color: transparent !important;
        color: #94A3B8 !important;
        border: none !important;
    }
    [data-testid="stSidebar"] div[data-baseweb="input"] button:hover {
        background-color: transparent !important;
        color: #F8FAFC !important;
    }

    /* Primary Action Buttons in Sidebar */
    .stSidebar [data-testid="stButton"] button {
        background-color: #1A2C44 !important;
        color: #E2E8F0 !important;
        border: 1px solid #273E5D !important;
        border-radius: 8px !important;
        font-size: 0.84rem !important;
        font-weight: 500 !important;
        padding: 0.55rem 0.85rem !important;
        transition: all 0.15s ease-in-out !important;
    }
    .stSidebar [data-testid="stButton"] button:hover {
        background-color: #159A91 !important;
        border-color: #159A91 !important;
        color: #FFFFFF !important;
    }

    /* Main Surface Cards */
    .saas-card {
        background-color: #FFFFFF;
        border: 1px solid #E6EAF0;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }

    /* Sidebar Brand Unit */
    .brand-unit {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 6px 0 14px 0;
        border-bottom: 1px solid #1E314B;
        margin-bottom: 14px;
    }
    .brand-logo {
        background: linear-gradient(135deg, #159A91 0%, #0E7C75 100%);
        color: #FFFFFF;
        width: 36px;
        height: 36px;
        border-radius: 9px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 17px;
        letter-spacing: -0.02em;
        box-shadow: 0 2px 6px rgba(21, 154, 145, 0.35);
    }
    .brand-name {
        font-size: 1.15rem;
        font-weight: 700;
        color: #F8FAFC !important;
        line-height: 1.2;
    }
    .brand-sub {
        font-size: 0.76rem;
        color: #94A3B8 !important;
        font-weight: 500;
    }

    /* Sidebar Source Card */
    .source-card {
        background-color: #18283E;
        border: 1px solid #223755;
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 8px;
    }
    .source-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.8rem;
        font-weight: 600;
        color: #F1F5F9;
        margin-bottom: 4px;
    }
    .source-card-meta {
        font-size: 0.74rem;
        color: #94A3B8;
    }

    /* Status Pills */
    .pill {
        display: inline-block;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .pill-teal {
        background-color: rgba(21, 154, 145, 0.18);
        color: #2DD4BF !important;
        border: 1px solid rgba(21, 154, 145, 0.35);
    }
    .pill-muted {
        background-color: #1A2C44;
        color: #64748B !important;
        border: 1px solid #273E5D;
    }
    .pill-active-light {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }

    /* Hero Banner */
    .hero-panel {
        background: linear-gradient(180deg, #FFFFFF 0%, #FAFBFD 100%);
        border: 1px solid #E6EAF0;
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 22px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .hero-h1 {
        font-size: 1.55rem;
        font-weight: 700;
        color: #17243A;
        letter-spacing: -0.02em;
        margin-bottom: 6px;
    }
    .hero-p {
        font-size: 0.92rem;
        color: #778399;
        margin-bottom: 16px;
        line-height: 1.5;
    }
    .tag-container {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }
    .engine-tag {
        background-color: #F1F4F9;
        color: #33455E;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid #E2E8F0;
    }

    /* Suggested Query Cards (Main Area) */
    .query-card-header {
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 10px;
    }
    .qc-sales { color: #159A91; }
    .qc-docs { color: #2878C7; }
    .qc-comb { color: #5369D8; }

    /* Custom Query Buttons */
    .main [data-testid="stColumn"] button {
        background-color: #FFFFFF !important;
        border: 1px solid #E6EAF0 !important;
        border-radius: 10px !important;
        color: #17243A !important;
        padding: 14px 16px !important;
        text-align: left !important;
        font-size: 0.85rem !important;
        line-height: 1.4 !important;
        font-weight: 500 !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
        min-height: 72px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: flex-start !important;
        transition: all 0.15s ease-in-out !important;
    }
    .main [data-testid="stColumn"] button:hover {
        border-color: #159A91 !important;
        background-color: #FAFDFD !important;
        box-shadow: 0 3px 6px rgba(21, 154, 145, 0.08) !important;
        transform: translateY(-1px);
    }

    /* KPI Display Cards */
    .kpi-grid-card {
        background-color: #FFFFFF;
        border: 1px solid #E6EAF0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .kpi-title-label {
        font-size: 0.74rem;
        font-weight: 600;
        color: #778399;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .kpi-value-num {
        font-size: 1.7rem;
        font-weight: 700;
        color: #17243A;
        letter-spacing: -0.02em;
    }
    .kpi-sub-text {
        font-size: 0.75rem;
        color: #94A3B8;
        margin-top: 2px;
    }

    /* Pipeline Step Tracker */
    .pipeline-bar {
        display: flex;
        align-items: center;
        background-color: #FFFFFF;
        border: 1px solid #E6EAF0;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 14px 0 20px 0;
        gap: 12px;
        flex-wrap: wrap;
    }
    .pipeline-step {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .step-dot-active {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #159A91;
    }
    .step-dot-muted {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #CBD5E1;
    }
    .step-arrow {
        color: #94A3B8;
        font-size: 0.8rem;
    }

    /* Execution Badges */
    .op-badge {
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 4px;
        margin-right: 6px;
    }
    .op-pandas { background-color: #EFF6FF; color: #1D4ED8; border: 1px solid #BFDBFE; }
    .op-rag { background-color: #F0FDF4; color: #15803D; border: 1px solid #BBF7D0; }
    .op-chart { background-color: #FAF5FF; color: #7E22CE; border: 1px solid #E9D5FF; }

    /* Bottom Bar & Chat Input Docking */
    [data-testid="stBottom"] {
        background-color: #F6F8FB !important;
        border-top: 1px solid #E6EAF0 !important;
        padding-top: 10px !important;
        padding-bottom: 14px !important;
    }
    [data-testid="stBottom"] > div {
        max-width: 1120px !important;
        margin: 0 auto !important;
    }
    [data-testid="stChatInput"] {
        background-color: #FFFFFF !important;
        border: 1px solid #D5DCE5 !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04) !important;
        transition: all 0.15s ease-in-out !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #159A91 !important;
        box-shadow: 0 0 0 2px rgba(21, 154, 145, 0.2) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #17243A !important;
        font-size: 0.9rem !important;
    }
    [data-testid="stChatInput"] button {
        color: #159A91 !important;
    }

    /* Edit Button on User Chat Messages */
    [data-testid="stChatMessage"] button[kind="secondary"] {
        background-color: transparent !important;
        border: 1px solid #CBD5E1 !important;
        color: #64748B !important;
        font-size: 0.72rem !important;
        padding: 2px 8px !important;
        border-radius: 6px !important;
        min-height: 26px !important;
        line-height: 1 !important;
        transition: all 0.15s ease !important;
    }
    [data-testid="stChatMessage"] button[kind="secondary"]:hover {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
        border-color: #94A3B8 !important;
    }
</style>
""", unsafe_allow_html=True)


def extract_text_content(content: Any) -> str:
    """Safely extract plain text from LLM response blocks, stripping internal chart comments."""
    text = ""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
            elif hasattr(block, "text"):
                parts.append(str(block.text))
        text = "".join(parts) if parts else str(content)
    else:
        text = str(content)

    return re.sub(r"<!-- CHART_JSON:.*? -->", "", text, flags=re.DOTALL).strip()


# Session State Management
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "raw_docs" not in st.session_state:
    st.session_state.raw_docs = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
if "processed_csv_name" not in st.session_state:
    st.session_state.processed_csv_name = None
if "processed_pdf_name" not in st.session_state:
    st.session_state.processed_pdf_name = None
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "editing_idx" not in st.session_state:
    st.session_state.editing_idx = None
if "active_view" not in st.session_state:
    st.session_state.active_view = "Ask Data"


def compile_agent_if_ready():
    """Compile or re-compile the LangGraph agent when state or credentials update."""
    if st.session_state.api_key.strip():
        try:
            if st.session_state.retriever is None and st.session_state.raw_docs is not None:
                chunks = split_documents(st.session_state.raw_docs)
                _, retriever = build_vector_store(chunks, api_key=st.session_state.api_key)
                st.session_state.retriever = retriever

            st.session_state.agent = build_analyst_agent(
                df=st.session_state.df,
                retriever=st.session_state.retriever,
                api_key=st.session_state.api_key
            )
            return True
        except Exception:
            return False
    return False


# ==============================================================================
# SIDEBAR: Premium Dark Navy Workspace (#142338)
# ==============================================================================
with st.sidebar:
    # 1. Branding
    st.markdown("""
    <div class="brand-unit">
        <div class="brand-logo">D</div>
        <div>
            <div class="brand-name">DataSense AI</div>
            <div class="brand-sub">Intelligent Data Analyst</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. LLM Engine Configuration
    st.markdown("#### LLM Configuration")
    user_key = st.text_input(
        "Google Gemini API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="Paste your API key...",
        label_visibility="collapsed"
    )
    st.caption("Press Enter after typing your key to apply.")

    if user_key != st.session_state.api_key:
        st.session_state.api_key = user_key.strip()
        st.session_state.agent = None
        compile_agent_if_ready()
        st.rerun()

    # Live Connection Badge
    if st.session_state.api_key.strip():
        st.markdown("<span class='pill pill-teal'>Engine Connected</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='pill pill-muted'>Key Not Configured</span>", unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # 4. Data Sources Section
    st.markdown("#### Data Sources")
    
    # Quick Load Demo Action (runs with 0 API key required)
    if st.button("Load Demo Dataset & Handbook", use_container_width=True):
        sample_csv = os.path.join("sample_data", "sales_data.csv")
        sample_pdf = os.path.join("sample_data", "company_handbook.pdf")
        if os.path.exists(sample_csv) and os.path.exists(sample_pdf):
            try:
                st.session_state.df = pd.read_csv(sample_csv)
                st.session_state.processed_csv_name = "sales_data.csv"
                st.session_state.raw_docs = load_pdf_from_path_or_stream(sample_pdf, filename="company_handbook.pdf")
                st.session_state.processed_pdf_name = "company_handbook.pdf"
                compile_agent_if_ready()
                st.rerun()
            except Exception as e:
                st.error(f"Error loading demo files: {str(e)}")

    # Distinct CSV Source Card
    if st.session_state.df is not None:
        csv_badge = f"<span class='pill pill-teal'>{len(st.session_state.df)} rows</span>"
        st.markdown(f"""
        <div class="source-card">
            <div class="source-card-header">
                <span>CSV: {st.session_state.processed_csv_name}</span>
                {csv_badge}
            </div>
            <div class="source-card-meta">Transactions & Revenue Metrics</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="source-card">
            <div class="source-card-header">
                <span>Structured Data</span>
                <span class='pill pill-muted'>None</span>
            </div>
            <div class="source-card-meta">Upload CSV or load demo data</div>
        </div>
        """, unsafe_allow_html=True)

    # Distinct PDF Source Card
    if st.session_state.raw_docs is not None:
        pdf_badge = "<span class='pill pill-teal'>Indexed</span>" if st.session_state.retriever else "<span class='pill pill-teal'>Parsed</span>"
        st.markdown(f"""
        <div class="source-card">
            <div class="source-card-header">
                <span>PDF: {st.session_state.processed_pdf_name}</span>
                {pdf_badge}
            </div>
            <div class="source-card-meta">{len(st.session_state.raw_docs)} pages • Policies & Terms</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="source-card">
            <div class="source-card-header">
                <span>Document Knowledge</span>
                <span class='pill pill-muted'>None</span>
            </div>
            <div class="source-card-meta">Upload PDF or load demo data</div>
        </div>
        """, unsafe_allow_html=True)

    # Custom Upload Dropzone (Collapsible)
    with st.expander("Upload Custom Files", expanded=False):
        uploaded_csv = st.file_uploader("Upload CSV Dataset", type=["csv"])
        uploaded_pdf = st.file_uploader("Upload PDF Document", type=["pdf"])

        if st.button("Process Files", use_container_width=True):
            if not uploaded_csv and not uploaded_pdf:
                st.warning("Please select at least one file.")
            else:
                try:
                    if uploaded_csv is not None:
                        st.session_state.df = pd.read_csv(uploaded_csv)
                        st.session_state.processed_csv_name = uploaded_csv.name
                    if uploaded_pdf is not None:
                        st.session_state.raw_docs = load_pdf_from_path_or_stream(uploaded_pdf, filename=uploaded_pdf.name)
                        st.session_state.processed_pdf_name = uploaded_pdf.name
                    compile_agent_if_ready()
                    st.rerun()
                except Exception as e:
                    st.error(f"Upload error: {str(e)}")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # 4. Session Controls
    st.markdown("#### Session Controls")
    c_col1, c_col2 = st.columns(2)
    with c_col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.editing_idx = None
            st.rerun()
    with c_col2:
        if st.button("Reset Data", use_container_width=True):
            st.session_state.df = None
            st.session_state.raw_docs = None
            st.session_state.retriever = None
            st.session_state.agent = None
            st.session_state.processed_csv_name = None
            st.session_state.processed_pdf_name = None
            st.session_state.messages = []
            st.session_state.editing_idx = None
            st.rerun()


# ==============================================================================
# MAIN WORKSPACE: Header Banner & Top Horizontal Navigation
# ==============================================================================

agent_ready = st.session_state.agent is not None
readiness_badge = "<span class='pill pill-active-light'>Agent Ready</span>" if agent_ready else "<span class='pill' style='background:#F1F4F9; color:#475569;'>Awaiting Key / Data</span>"
data_summary = f"{st.session_state.processed_csv_name or 'No CSV'} &bull; {st.session_state.processed_pdf_name or 'No PDF'}"

if "active_view" not in st.session_state:
    st.session_state.active_view = "Ask Data"

# Main Top Header Bar with Horizontal View Switcher
head_col1, head_col2 = st.columns([5, 5])
with head_col1:
    st.markdown(f"""
    <div style="padding-top: 3px;">
        <div style="font-size: 1.35rem; font-weight: 700; color: #17243A; letter-spacing: -0.02em;">DataSense AI</div>
        <div style="font-size: 0.82rem; color: #778399; margin-top: 2px;">
            Context: <strong style="color:#17243A;">{data_summary}</strong> &bull; {readiness_badge}
        </div>
    </div>
    """, unsafe_allow_html=True)

with head_col2:
    selected_nav = st.segmented_control(
        "Workspace View Navigation",
        options=["Ask Data", "Data Explorer", "Document Library"],
        default=st.session_state.active_view,
        label_visibility="collapsed",
        key="top_segmented_nav"
    )
    if selected_nav:
        st.session_state.active_view = selected_nav

active_view = st.session_state.active_view
st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)


# Helper: Dynamic Chart & Visual Widget Rendering
def render_dynamic_visuals(tools_called: list, df: pd.DataFrame):
    """Render dynamic charts, KPI cards, or tables based on executed tools."""
    if df is None:
        return

    for tool_call in tools_called:
        output_str = tool_call.get("output", "")
        tool_name = tool_call.get("name", "")

        # 1. Custom chart from plot_chart tool
        chart_match = re.search(r"<!-- CHART_JSON:(.*?) -->", output_str, flags=re.DOTALL)
        if chart_match:
            try:
                payload = json.loads(chart_match.group(1))
                chart_type = payload.get("chart_type", "bar")
                title = payload.get("title", "Data Visualization")
                x_label = payload.get("x_label", "Category")
                y_label = payload.get("y_label", "Value")
                chart_df = pd.DataFrame(payload.get("data", []))

                if not chart_df.empty and "x" in chart_df.columns and "y" in chart_df.columns:
                    st.caption(f"**{title}** ({y_label} by {x_label})")
                    chart_data = chart_df.set_index("x")[["y"]]
                    chart_data.columns = [y_label]

                    if chart_type == "line":
                        st.line_chart(chart_data, color="#159A91", use_container_width=True)
                    elif chart_type == "area":
                        st.area_chart(chart_data, color="#5369D8", use_container_width=True)
                    else:
                        st.bar_chart(chart_data, color="#159A91", use_container_width=True)
                    continue
            except Exception:
                pass

        # 2. Revenue Analysis Widget
        if tool_name == "get_revenue_analysis":
            try:
                total_rev = df["revenue"].sum() if "revenue" in df.columns else 0
                st.markdown(f"""
                <div class="kpi-grid-card" style="display: inline-block; min-width: 230px; margin: 8px 0;">
                    <div class="kpi-title-label">Total Revenue</div>
                    <div class="kpi-value-num">${total_rev:,.2f}</div>
                    <div class="kpi-sub-text">Across {len(df)} transactions</div>
                </div>
                """, unsafe_allow_html=True)
                if "category" in df.columns and "revenue" in df.columns:
                    cat_rev = df.groupby("category")["revenue"].sum().reset_index().set_index("category")
                    cat_rev.columns = ["Revenue ($)"]
                    st.caption("**Revenue Breakdown by Category**")
                    st.bar_chart(cat_rev, color="#159A91", use_container_width=True)
            except Exception:
                pass

        # 3. Monthly Revenue Trend Widget
        elif tool_name == "get_monthly_revenue_trend":
            try:
                date_col = next((c for c in ["order_date", "date", "OrderDate"] if c in df.columns), None)
                if date_col and "revenue" in df.columns:
                    df_copy = df.copy()
                    df_copy["pdate"] = pd.to_datetime(df_copy[date_col], errors="coerce")
                    df_copy = df_copy.dropna(subset=["pdate"])
                    df_copy["Month"] = df_copy["pdate"].dt.strftime("%Y-%m")
                    m_rev = df_copy.groupby("Month")["revenue"].sum().reset_index().set_index("Month")
                    m_rev.columns = ["Revenue ($)"]
                    st.caption("**Monthly Revenue Trajectory (Chronological)**")
                    st.line_chart(m_rev, color="#159A91", use_container_width=True)
            except Exception:
                pass

        # 4. Top Products Widget
        elif tool_name == "get_top_products":
            try:
                if "product" in df.columns and "revenue" in df.columns:
                    top_prods = (
                        df.groupby(["product", "category"] if "category" in df.columns else ["product"])["revenue"]
                        .sum()
                        .reset_index()
                        .sort_values(by="revenue", ascending=False)
                        .head(5)
                    )
                    top_prods = top_prods.rename(columns={"product": "Product", "category": "Category", "revenue": "Revenue ($)"})
                    top_prods["Revenue ($)"] = top_prods["Revenue ($)"].apply(lambda x: f"${x:,.2f}")
                    st.caption("**Top Revenue Generating Products**")
                    st.dataframe(top_prods, hide_index=True, use_container_width=True)
            except Exception:
                pass


# Helper: Factual Execution Details Formatter
def render_execution_details(tools_called: list):
    """Render factual tool execution details in a clean expandable section."""
    with st.expander(f"How the agent handled this query ({len(tools_called)} operations)", expanded=False):
        for tool_call in tools_called:
            name = tool_call.get("name", "tool")
            clean_output = re.sub(r"<!-- CHART_JSON:.*? -->", "", tool_call.get("output", ""), flags=re.DOTALL).strip()

            if name == "retrieve_document_knowledge":
                badge = "<span class='op-badge op-rag'>Document RAG</span>"
                desc = "Retrieved policy chunks and page citations from ChromaDB vector store."
            elif name == "plot_chart":
                badge = "<span class='op-badge op-chart'>Data Visualization</span>"
                desc = "Computed aggregated metrics and dynamically plotted chart from dataset."
            else:
                badge = "<span class='op-badge op-pandas'>CSV / Pandas Analysis</span>"
                desc = f"Executed deterministic analytical computation via Pandas tool '{name}'."

            st.markdown(f"{badge} **{name}**", unsafe_allow_html=True)
            st.caption(desc)
            st.code(clean_output, language="text")


# ==============================================================================
# VIEW 1: ASK DATA (Conversational Workspace)
# ==============================================================================
if active_view == "Ask Data":
    # 1. Always-Available Example Inquiries Dropdown (Never vanishes!)
    demo_questions = [
        "-- Select an example inquiry to try --",
        "[Sales] Which product generated the highest revenue?",
        "[Sales] Show the monthly revenue trend.",
        "[Sales] Which region generated the highest sales?",
        "[Sales] What is the average transaction value across categories?",
        "[Policy] What is the return policy for electronics?",
        "[Policy] What are the warranty terms and conditions?",
        "[Policy] What are the volume bulk discount rules?",
        "[Combined] Which electronics product generated the highest revenue, and what is its return policy?",
        "[Combined] Which category has the highest revenue, and does it have a special discount policy?",
        "[Plot] Plot revenue by category as a bar chart",
        "[Plot] Plot monthly revenue trajectory as a line chart",
        "[Plot] Plot regional revenue comparison"
    ]

    col_demo_select, col_demo_btn = st.columns([5, 1])
    with col_demo_select:
        chosen_demo = st.selectbox(
            "Example Inquiries",
            options=demo_questions,
            index=0,
            label_visibility="collapsed",
            key="demo_inquiry_dropdown"
        )
    with col_demo_btn:
        btn_run_demo = st.button("Run Query", use_container_width=True, key="btn_run_demo")

    if btn_run_demo and chosen_demo != demo_questions[0]:
        clean_prompt = re.sub(r"^\[.*?\]\s*", "", chosen_demo).strip()
        st.session_state.messages.append({"role": "user", "content": clean_prompt})
        st.rerun()

    # Process pending query (e.g. from buttons or edit actions)
    if st.session_state.pending_query:
        st.session_state.messages.append({"role": "user", "content": st.session_state.pending_query})
        st.session_state.pending_query = None
        st.rerun()

    # Starter Welcome Hero Panel (Shown when conversation is empty)
    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div class="hero-panel" style="margin-top: 10px;">
            <div class="hero-h1">Your data has answers. Let's find them.</div>
            <div class="hero-p">Ask complex questions across your sales transactions and operational PDF handbooks using natural language.</div>
            <div class="tag-container">
                <span class="engine-tag">LangGraph State Machine</span>
                <span class="engine-tag">ChromaDB Vector Retrieval</span>
                <span class="engine-tag">Deterministic Pandas Analysis</span>
                <span class="engine-tag">Dynamic Visualization Engine</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="query-card-header qc-sales">Sales Analytics</div>', unsafe_allow_html=True)
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            if st.button("Top Product by Revenue\nIdentify the highest earning product", key="btn_q_top", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "Which product generated the highest revenue?"})
                st.rerun()
        with col_s2:
            if st.button("Monthly Revenue Trend\nView chronological sales trajectory", key="btn_q_month", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "Show the monthly revenue trend."})
                st.rerun()
        with col_s3:
            if st.button("Regional Sales Comparison\nCompare performance across regions", key="btn_q_reg", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "Which region generated the highest sales?"})
                st.rerun()

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="query-card-header qc-docs">Document Search</div>', unsafe_allow_html=True)
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            if st.button("Electronics Return Policy\nCheck return window and condition rules", key="btn_q_ret", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "What is the return policy for electronics?"})
                st.rerun()
        with col_d2:
            if st.button("Warranty Terms & Conditions\nInspect hardware and battery coverage", key="btn_q_warr", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "What are the warranty conditions?"})
                st.rerun()

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="query-card-header qc-comb">Combined Intelligence (Multi-Source Reasoning)</div>', unsafe_allow_html=True)
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("Top Product & Return Policy\nJoint analysis finding top product and its policy", key="btn_q_comb1", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "Which electronics product generated the highest revenue, and what is its return policy?"})
                st.rerun()
        with col_c2:
            if st.button("Category Sales & Volume Discounts\nCheck highest earning category and bulk discounts", key="btn_q_comb2", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "Which category has the highest revenue, and does it have a special discount policy?"})
                st.rerun()

        st.markdown("---")

    # 2. Main Chat Message Container (Always strictly ABOVE the chat input)
    chat_container = st.container()
    with chat_container:
        for idx, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                with st.chat_message("user"):
                    if st.session_state.editing_idx == idx:
                        # ChatGPT Style In-Place Query Editing
                        st.caption("Edit your question:")
                        edited_query = st.text_area(
                            "Edit query",
                            value=msg["content"],
                            key=f"edit_area_{idx}",
                            label_visibility="collapsed"
                        )
                        col_save, col_cancel, _ = st.columns([1, 1, 4])
                        with col_save:
                            if st.button("Save & Submit", key=f"btn_save_{idx}", type="primary"):
                                st.session_state.messages = st.session_state.messages[:idx]
                                st.session_state.editing_idx = None
                                st.session_state.messages.append({"role": "user", "content": edited_query})
                                st.rerun()
                        with col_cancel:
                            if st.button("Cancel", key=f"btn_cancel_{idx}"):
                                st.session_state.editing_idx = None
                                st.rerun()
                    else:
                        col_text, col_edit = st.columns([11, 1])
                        with col_text:
                            st.markdown(msg["content"])
                        with col_edit:
                            if st.button("Edit", key=f"btn_edit_{idx}", help="Edit this question"):
                                st.session_state.editing_idx = idx
                                st.rerun()

            elif msg["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.markdown(msg["content"])

                    if msg.get("tools_called") and st.session_state.df is not None:
                        render_dynamic_visuals(msg["tools_called"], st.session_state.df)

                    if msg.get("tools_called"):
                        render_execution_details(msg["tools_called"])

    # 3. Robust In-Flight Query Handling (Resumes seamlessly even if user switched tabs!)
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        unanswered_query = st.session_state.messages[-1]["content"]

        if not st.session_state.api_key.strip():
            with chat_container:
                st.warning("Please enter your Google Gemini API Key in the sidebar to run the agent.")
        else:
            if st.session_state.agent is None:
                compile_agent_if_ready()

            if st.session_state.agent is None:
                with chat_container:
                    st.error("Agent initialization failed. Please verify your Gemini API key.")
            else:
                with chat_container:
                    with st.chat_message("assistant"):
                        with st.spinner("Analyzing structured transactions and document knowledge..."):
                            try:
                                graph_messages = [SystemMessage(content=SYSTEM_PROMPT)]
                                for past_msg in st.session_state.messages[:-1]:
                                    if past_msg["role"] == "user":
                                        graph_messages.append(HumanMessage(content=past_msg["content"]))
                                    elif past_msg["role"] == "assistant":
                                        graph_messages.append(AIMessage(content=past_msg["content"]))

                                graph_messages.append(HumanMessage(content=unanswered_query))

                                result = st.session_state.agent.invoke({"messages": graph_messages})

                                all_msgs = result["messages"]
                                final_ai_msg = all_msgs[-1]
                                response_text = extract_text_content(final_ai_msg.content)

                                tools_called = []
                                for m in all_msgs:
                                    if hasattr(m, "name") and m.name:
                                        tools_called.append({
                                            "name": m.name,
                                            "output": str(m.content)
                                        })

                                st.markdown(response_text)

                                if tools_called and st.session_state.df is not None:
                                    render_dynamic_visuals(tools_called, st.session_state.df)

                                if tools_called:
                                    render_execution_details(tools_called)

                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": response_text,
                                    "tools_called": tools_called
                                })
                                st.rerun()

                            except Exception as e:
                                err = f"An error occurred: {str(e)}"
                                st.error(err)
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": err,
                                    "tools_called": []
                                })
                                st.rerun()

    # 4. Chat Composer Input (Permanently Anchored at Bottom in stBottom)
    chat_prompt = st.chat_input("Ask an analytical question or search company policy handbook...")
    if chat_prompt:
        st.session_state.messages.append({"role": "user", "content": chat_prompt})
        st.rerun()


# ==============================================================================
# VIEW 2: DATA EXPLORER (Business Intelligence Dashboard)
# ==============================================================================
elif active_view == "Data Explorer":
    if st.session_state.df is not None:
        df = st.session_state.df

        # 4 Prominent KPI Metric Cards
        st.markdown("#### Key Performance Indicators")
        k_col1, k_col2, k_col3, k_col4 = st.columns(4)

        with k_col1:
            st.markdown(f"""
            <div class="kpi-grid-card">
                <div class="kpi-title-label">Total Records</div>
                <div class="kpi-value-num">{len(df):,}</div>
                <div class="kpi-sub-text">Transactions logged</div>
            </div>
            """, unsafe_allow_html=True)

        with k_col2:
            st.markdown(f"""
            <div class="kpi-grid-card">
                <div class="kpi-title-label">Total Attributes</div>
                <div class="kpi-value-num">{df.shape[1]}</div>
                <div class="kpi-sub-text">Dataset columns</div>
            </div>
            """, unsafe_allow_html=True)

        with k_col3:
            total_rev = df["revenue"].sum() if "revenue" in df.columns else 0
            st.markdown(f"""
            <div class="kpi-grid-card">
                <div class="kpi-title-label">Total Revenue</div>
                <div class="kpi-value-num">${total_rev:,.2f}</div>
                <div class="kpi-sub-text">Cumulative sales value</div>
            </div>
            """, unsafe_allow_html=True)

        with k_col4:
            avg_order = df["revenue"].mean() if "revenue" in df.columns else 0
            st.markdown(f"""
            <div class="kpi-grid-card">
                <div class="kpi-title-label">Avg Transaction</div>
                <div class="kpi-value-num">${avg_order:,.2f}</div>
                <div class="kpi-sub-text">Mean ticket size</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

        # Balanced 2-Column Chart Layout
        st.markdown("#### Exploratory Visualizations")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("**Chart A: Monthly Revenue Trend**")
            date_col = next((c for c in ["order_date", "date", "OrderDate"] if c in df.columns), None)
            if date_col and "revenue" in df.columns:
                df_copy = df.copy()
                df_copy["pdate"] = pd.to_datetime(df_copy[date_col], errors="coerce")
                df_copy = df_copy.dropna(subset=["pdate"])
                df_copy["Month"] = df_copy["pdate"].dt.strftime("%Y-%m")
                monthly_chart = df_copy.groupby("Month")["revenue"].sum().reset_index().set_index("Month")
                monthly_chart.columns = ["Revenue ($)"]
                st.line_chart(monthly_chart, color="#159A91", use_container_width=True)
            else:
                st.caption("Date or revenue column not found.")

        with chart_col2:
            st.markdown("**Chart B: Category-wise Revenue**")
            if "category" in df.columns and "revenue" in df.columns:
                cat_chart = df.groupby("category")["revenue"].sum().reset_index().set_index("category")
                cat_chart.columns = ["Revenue ($)"]
                st.bar_chart(cat_chart, color="#5369D8", use_container_width=True)
            else:
                st.caption("Category or revenue column not found.")

        st.markdown("---")

        # Dataset Preview & Export Section
        st.markdown(f"#### Active Dataset: `{st.session_state.processed_csv_name}`")
        st.dataframe(df, use_container_width=True, height=280)

        # CSV Download Action
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Dataset as CSV",
            data=csv_bytes,
            file_name=f"export_{st.session_state.processed_csv_name or 'sales_data.csv'}",
            mime="text/csv"
        )

        st.markdown("---")

        # Column Schema & Missing Values
        st.markdown("#### Schema & Data Integrity")
        schema_df = pd.DataFrame({
            "Column Name": df.columns,
            "Data Type": [str(t) for t in df.dtypes],
            "Non-Null Count": df.notnull().sum().values,
            "Missing Values": df.isnull().sum().values,
            "Sample Value": [str(df[c].iloc[0]) if len(df) > 0 else "" for c in df.columns]
        })
        st.dataframe(schema_df, hide_index=True, use_container_width=True)

    else:
        st.markdown("""
        <div class="saas-card" style="text-align: center; padding: 48px 24px;">
            <div style="font-size: 1.15rem; font-weight: 700; color: #17243A; margin-bottom: 8px;">No Structured Dataset Loaded</div>
            <div style="color: #778399; font-size: 0.9rem;">Click 'Load Demo Dataset & Handbook' in the sidebar or upload a CSV file to inspect data and charts.</div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# VIEW 3: DOCUMENT LIBRARY (Document Intelligence Workspace)
# ==============================================================================
elif active_view == "Document Library":
    if st.session_state.raw_docs is not None:
        docs = st.session_state.raw_docs
        chunks = split_documents(docs)
        is_indexed = st.session_state.retriever is not None

        # Document Collection Card
        st.markdown(f"#### Document Collection: `{st.session_state.processed_pdf_name}`")

        doc_kpi1, doc_kpi2, doc_kpi3 = st.columns(3)
        with doc_kpi1:
            st.markdown(f"""
            <div class="kpi-grid-card">
                <div class="kpi-title-label">Extracted Pages</div>
                <div class="kpi-value-num">{len(docs)}</div>
                <div class="kpi-sub-text">Parsed from PDF</div>
            </div>
            """, unsafe_allow_html=True)
        with doc_kpi2:
            st.markdown(f"""
            <div class="kpi-grid-card">
                <div class="kpi-title-label">Indexed Chunks</div>
                <div class="kpi-value-num">{len(chunks)}</div>
                <div class="kpi-sub-text">Recursive character splits</div>
            </div>
            """, unsafe_allow_html=True)
        with doc_kpi3:
            idx_label = "ChromaDB Active" if is_indexed else "Parsed (Awaiting Key)"
            idx_sub = "Vector retrieval enabled" if is_indexed else "Enter API key to generate vectors"
            st.markdown(f"""
            <div class="kpi-grid-card">
                <div class="kpi-title-label">Index Status</div>
                <div class="kpi-value-num" style="font-size: 1.35rem;">{idx_label}</div>
                <div class="kpi-sub-text">{idx_sub}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Accurate 5-Stage Indexing Pipeline Tracker
        dot_up = "<span class='step-dot-active'></span>"
        dot_ext = "<span class='step-dot-active'></span>"
        dot_chk = "<span class='step-dot-active'></span>"
        dot_emb = "<span class='step-dot-active'></span>" if is_indexed else "<span class='step-dot-muted'></span>"
        dot_idx = "<span class='step-dot-active'></span>" if is_indexed else "<span class='step-dot-muted'></span>"
        emb_label = "Embeddings Generated" if is_indexed else "Embeddings (Awaiting Key)"
        idx_step_label = "Vector Indexed" if is_indexed else "ChromaDB Indexing"

        st.markdown(f"""
        <div class="pipeline-bar">
            <div class="pipeline-step">{dot_up} PDF Uploaded</div>
            <span class="step-arrow">&rarr;</span>
            <div class="pipeline-step">{dot_ext} Text Extracted</div>
            <span class="step-arrow">&rarr;</span>
            <div class="pipeline-step">{dot_chk} Chunked ({len(chunks)})</div>
            <span class="step-arrow">&rarr;</span>
            <div class="pipeline-step">{dot_emb} {emb_label}</div>
            <span class="step-arrow">&rarr;</span>
            <div class="pipeline-step">{dot_idx} {idx_step_label}</div>
        </div>
        """, unsafe_allow_html=True)

        # Split Document Viewer
        st.markdown("#### Document Content Viewer")
        split_col1, split_col2 = st.columns([1, 2])

        with split_col1:
            st.markdown("**Document Index & Page Selector**")
            page_labels = [f"Page {doc.metadata.get('page', i + 1)}" for i, doc in enumerate(docs)]
            selected_page_label = st.radio("Select Page to Inspect", options=page_labels, label_visibility="collapsed")
            selected_idx = page_labels.index(selected_page_label)
            selected_doc = docs[selected_idx]

            st.caption(f"Source: `{selected_doc.metadata.get('source', 'document.pdf')}`")
            st.caption(f"Character length: {len(selected_doc.page_content)} chars")

        with split_col2:
            st.markdown(f"**Content Preview ({selected_page_label})**")
            st.markdown(f"""
            <div class="saas-card" style="background-color: #FFFFFF; font-family: -apple-system, BlinkMacSystemFont, monospace; font-size: 0.88rem; line-height: 1.6; white-space: pre-wrap; height: 340px; overflow-y: auto;">
{selected_doc.page_content}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Chunk Sample Inspector
        st.markdown("#### Vector Chunk Inspector")
        st.caption("Inspect how RecursiveCharacterTextSplitter partitions document pages into retrieval units.")
        chunk_preview_df = pd.DataFrame([
            {
                "Chunk #": i + 1,
                "Page": c.metadata.get("page", 1),
                "Characters": len(c.page_content),
                "Text Excerpt": c.page_content[:120] + "..."
            }
            for i, c in enumerate(chunks[:6])
        ])
        st.dataframe(chunk_preview_df, hide_index=True, use_container_width=True)

    else:
        st.markdown("""
        <div class="saas-card" style="text-align: center; padding: 48px 24px;">
            <div style="font-size: 1.15rem; font-weight: 700; color: #17243A; margin-bottom: 8px;">No Document Knowledge Base Loaded</div>
            <div style="color: #778399; font-size: 0.9rem;">Click 'Load Demo Dataset & Handbook' in the sidebar or upload a PDF to inspect pages and RAG indexing.</div>
        </div>
        """, unsafe_allow_html=True)
