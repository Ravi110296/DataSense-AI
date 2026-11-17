"""
app.py - DataSense AI: Intelligent Data Analyst Agent.

Redesigned Enterprise SaaS Interface:
- Modern, clean, minimal SaaS dashboard aesthetic (#F7F9FC canvas, #F0F3F7 sidebar, #172B4D deep navy text, #238B9E teal accent).
- Preserves full existing backend: LangChain tools, LangGraph state graph, Pandas analytics, and ChromaDB RAG.
- Sidebar with Branding, LLM Configuration, Data Sources, Real-time Workspace Status, and Session Controls.
- Main Workspace with 3 Tabs: 'Ask Data' (Chat & Interactive Query Cards), 'Data Explorer' (Preview & Schema), and 'Document Library' (Indexed Chunks & Pages).
- Factual execution trace in 'How the agent handled this query'.
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

# Streamlit Page Setup
st.set_page_config(
    page_title="DataSense AI - Intelligent Data Analyst",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise CSS (Clean SaaS Palette: #F7F9FC canvas, #172B4D navy, #238B9E teal accent, #E2E8F0 borders)
st.markdown("""
<style>
    /* Font & Global Reset */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Layout Clearance: Generous top padding prevents header cut-off */
    .block-container {
        padding-top: 3.8rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1080px !important;
    }

    /* Enterprise Palette */
    .stApp {
        background-color: #F7F9FC;
    }
    [data-testid="stSidebar"] {
        background-color: #F0F3F7 !important;
        border-right: 1px solid #E2E8F0;
    }

    /* Typography Hierarchy */
    h1, h2, h3, h4, h5, h6 {
        color: #172B4D !important;
        font-weight: 600 !important;
    }
    p, span, label {
        color: #172B4D;
    }
    .muted-text {
        color: #738095 !important;
        font-size: 0.85rem;
    }

    /* Brand Header in Sidebar */
    .brand-container {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 4px 0 16px 0;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 16px;
    }
    .brand-icon {
        background-color: #238B9E;
        color: #FFFFFF;
        width: 36px;
        height: 36px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 18px;
        letter-spacing: -0.02em;
    }
    .brand-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #172B4D;
        line-height: 1.2;
    }
    .brand-subtitle {
        font-size: 0.78rem;
        color: #738095;
        font-weight: 500;
    }

    /* Clean Card Component */
    .saas-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }

    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.74rem;
        font-weight: 600;
    }
    .status-active {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }
    .status-inactive {
        background-color: #F1F5F9;
        color: #64748B;
        border: 1px solid #CBD5E1;
    }

    /* Workspace Status Container */
    .status-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid #E2E8F0;
        font-size: 0.85rem;
    }
    .status-row:last-child {
        border-bottom: none;
    }

    /* KPI Metric Box */
    .kpi-container {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 10px 0;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
        display: inline-block;
        min-width: 220px;
    }
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #738095;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #172B4D;
        letter-spacing: -0.02em;
    }

    /* Clean Prompt Buttons */
    div[data-testid="stColumn"] button {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        color: #172B4D !important;
        padding: 12px 14px !important;
        text-align: left !important;
        font-size: 0.84rem !important;
        line-height: 1.4 !important;
        font-weight: 500 !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
        min-height: 58px !important;
        transition: all 0.15s ease !important;
    }
    div[data-testid="stColumn"] button:hover {
        border-color: #238B9E !important;
        color: #238B9E !important;
        background-color: #F8FAFC !important;
        box-shadow: 0 2px 4px rgba(35, 139, 158, 0.08) !important;
    }

    /* Execution Step Badge */
    .exec-badge {
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 4px;
        margin-right: 6px;
    }
    .exec-pandas { background-color: #EFF6FF; color: #1D4ED8; border: 1px solid #BFDBFE; }
    .exec-rag { background-color: #F0FDF4; color: #15803D; border: 1px solid #BBF7D0; }
    .exec-chart { background-color: #FAF5FF; color: #7E22CE; border: 1px solid #E9D5FF; }

    /* Custom Streamlit Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 18px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.92rem;
        font-weight: 500;
        color: #738095;
        padding: 8px 12px;
    }
    .stTabs [aria-selected="true"] {
        color: #238B9E !important;
        border-bottom-color: #238B9E !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def extract_text_content(content: Any) -> str:
    """Safely extract plain text from LLM response blocks, stripping internal chart payload comments."""
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


# Session State Initialization
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


# Helper to recompile agent when state changes
def ensure_agent_compiled():
    if st.session_state.api_key.strip():
        try:
            # Build retriever if document raw text is loaded but retriever not yet built
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
# SIDEBAR: Workspace & Configuration
# ==============================================================================
with st.sidebar:
    # 1. Branding
    st.markdown("""
    <div class="brand-container">
        <div class="brand-icon">D</div>
        <div>
            <div class="brand-title">DataSense AI</div>
            <div class="brand-subtitle">Intelligent Data Analyst</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. LLM Configuration
    st.markdown("#### LLM Configuration")
    input_key = st.text_input(
        "API Key (Google Gemini)",
        value=st.session_state.api_key,
        type="password",
        placeholder="Enter your API key...",
        label_visibility="collapsed"
    )
    st.caption("Press Enter after entering your key to apply.")
    
    if input_key != st.session_state.api_key:
        st.session_state.api_key = input_key.strip()
        st.session_state.agent = None
        ensure_agent_compiled()
        st.rerun()

    # 3. Data Sources Section
    st.markdown("#### Data Sources")
    if st.button("Load Sample Dataset & Handbook", use_container_width=True):
        sample_csv_path = os.path.join("sample_data", "sales_data.csv")
        sample_pdf_path = os.path.join("sample_data", "company_handbook.pdf")

        if os.path.exists(sample_csv_path) and os.path.exists(sample_pdf_path):
            try:
                # Load CSV (no key required)
                st.session_state.df = pd.read_csv(sample_csv_path)
                st.session_state.processed_csv_name = "sales_data.csv"

                # Extract PDF text (no key required)
                st.session_state.raw_docs = load_pdf_from_path_or_stream(sample_pdf_path, filename="company_handbook.pdf")
                st.session_state.processed_pdf_name = "company_handbook.pdf"

                # If key already configured, compile agent
                ensure_agent_compiled()
                st.success("Sample dataset & handbook loaded.")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")
        else:
            st.error("Sample files not found on disk.")

    with st.expander("Upload Custom Files", expanded=False):
        uploaded_csv = st.file_uploader("Upload CSV Data", type=["csv"])
        uploaded_pdf = st.file_uploader("Upload PDF Document", type=["pdf"])

        if st.button("Process Uploaded Files", use_container_width=True):
            if not uploaded_csv and not uploaded_pdf:
                st.warning("Please select at least one file to upload.")
            else:
                try:
                    if uploaded_csv is not None:
                        st.session_state.df = pd.read_csv(uploaded_csv)
                        st.session_state.processed_csv_name = uploaded_csv.name
                    if uploaded_pdf is not None:
                        st.session_state.raw_docs = load_pdf_from_path_or_stream(uploaded_pdf, filename=uploaded_pdf.name)
                        st.session_state.processed_pdf_name = uploaded_pdf.name

                    ensure_agent_compiled()
                    st.success("Custom files processed successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"File processing error: {str(e)}")

    # 4. Workspace Status Section
    st.markdown("#### Workspace Status")
    
    # LLM Status
    llm_status = "<span class='status-badge status-active'>Connected</span>" if st.session_state.api_key else "<span class='status-badge status-inactive'>Not Configured</span>"
    st.markdown(f"""
    <div class="status-row">
        <span><b>LLM Engine:</b> Gemini</span>
        {llm_status}
    </div>
    """, unsafe_allow_html=True)

    # CSV Status
    if st.session_state.df is not None:
        csv_badge = "<span class='status-badge status-active'>Loaded</span>"
        csv_info = f"{st.session_state.processed_csv_name} ({len(st.session_state.df)} rows, {st.session_state.df.shape[1]} cols)"
    else:
        csv_badge = "<span class='status-badge status-inactive'>Not Loaded</span>"
        csv_info = "No CSV file loaded"
    st.markdown(f"""
    <div class="status-row">
        <span><b>Structured CSV:</b></span>
        {csv_badge}
    </div>
    <div class="muted-text" style="padding-bottom: 6px;">{csv_info}</div>
    """, unsafe_allow_html=True)

    # PDF Status
    if st.session_state.raw_docs is not None:
        pdf_badge = "<span class='status-badge status-active'>Indexed</span>" if st.session_state.retriever else "<span class='status-badge status-active'>Parsed</span>"
        pdf_info = f"{st.session_state.processed_pdf_name} ({len(st.session_state.raw_docs)} pages)"
    else:
        pdf_badge = "<span class='status-badge status-inactive'>Not Indexed</span>"
        pdf_info = "No PDF document loaded"
    st.markdown(f"""
    <div class="status-row">
        <span><b>Document Base:</b></span>
        {pdf_badge}
    </div>
    <div class="muted-text" style="padding-bottom: 8px;">{pdf_info}</div>
    """, unsafe_allow_html=True)

    # 5. Session Controls
    st.markdown("#### Session Controls")
    col_sc1, col_sc2 = st.columns(2)
    with col_sc1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col_sc2:
        if st.button("Reset Data", use_container_width=True):
            st.session_state.df = None
            st.session_state.raw_docs = None
            st.session_state.retriever = None
            st.session_state.agent = None
            st.session_state.processed_csv_name = None
            st.session_state.processed_pdf_name = None
            st.session_state.messages = []
            st.rerun()


# ==============================================================================
# MAIN WORKSPACE: Header & Tabs
# ==============================================================================

# Compact Header
st.markdown("""
<div class="saas-card" style="padding: 16px 20px; margin-bottom: 18px;">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
        <div>
            <div style="font-size: 1.35rem; font-weight: 700; color: #172B4D;">DataSense AI</div>
            <div style="font-size: 0.88rem; color: #738095;">Explore your structured data and business documents using natural-language questions.</div>
        </div>
        <div style="display: flex; gap: 8px;">
            <span class="status-badge status-active">LangGraph Orchestrated</span>
            <span class="status-badge status-active">ChromaDB RAG</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 3 Primary Workspace Navigation Tabs
tab_ask, tab_explorer, tab_docs = st.tabs(["Ask Data", "Data Explorer", "Document Library"])


# Dynamic Visual Widgets Helper
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
                title = payload.get("title", "Chart")
                x_label = payload.get("x_label", "Category")
                y_label = payload.get("y_label", "Value")
                chart_df = pd.DataFrame(payload.get("data", []))

                if not chart_df.empty and "x" in chart_df.columns and "y" in chart_df.columns:
                    st.caption(f"**{title}** ({y_label} by {x_label})")
                    chart_data = chart_df.set_index("x")[["y"]]
                    chart_data.columns = [y_label]

                    if chart_type == "line":
                        st.line_chart(chart_data, color="#238B9E", use_container_width=True)
                    elif chart_type == "area":
                        st.area_chart(chart_data, color="#2878C7", use_container_width=True)
                    else:
                        st.bar_chart(chart_data, color="#238B9E", use_container_width=True)
                    continue
            except Exception:
                pass

        # 2. Revenue analysis fallback visual
        if tool_name == "get_revenue_analysis":
            try:
                total_rev = df["revenue"].sum() if "revenue" in df.columns else 0
                st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-label">Total Revenue</div>
                    <div class="kpi-value">${total_rev:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
                if "category" in df.columns and "revenue" in df.columns:
                    cat_rev = df.groupby("category")["revenue"].sum().reset_index().set_index("category")
                    cat_rev.columns = ["Revenue ($)"]
                    st.caption("**Revenue Breakdown by Category**")
                    st.bar_chart(cat_rev, color="#238B9E", use_container_width=True)
            except Exception:
                pass

        # 3. Monthly trend fallback visual
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
                    st.caption("**Monthly Revenue Trend (2024)**")
                    st.line_chart(m_rev, color="#238B9E", use_container_width=True)
            except Exception:
                pass

        # 4. Top products fallback visual
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


# Execution Details Formatter
def render_execution_details(tools_called: list):
    """Render factual tool execution details in a clean expandable section."""
    with st.expander(f"How the agent handled this query ({len(tools_called)} operations)", expanded=False):
        for tool_call in tools_called:
            name = tool_call.get("name", "tool")
            clean_output = re.sub(r"<!-- CHART_JSON:.*? -->", "", tool_call.get("output", ""), flags=re.DOTALL).strip()
            
            # Categorize operation
            if name == "retrieve_document_knowledge":
                badge = "<span class='exec-badge exec-rag'>Document RAG</span>"
                desc = "Retrieved policy chunks and source page metadata from ChromaDB vector store."
            elif name == "plot_chart":
                badge = "<span class='exec-badge exec-chart'>Data Visualization</span>"
                desc = "Computed aggregated metrics and dynamically plotted chart from dataset."
            else:
                badge = "<span class='exec-badge exec-pandas'>CSV / Pandas Analysis</span>"
                desc = f"Executed deterministic analytical computation via Pandas tool '{name}'."

            st.markdown(f"{badge} **{name}**", unsafe_allow_html=True)
            st.caption(desc)
            st.code(clean_output, language="text")


# ==============================================================================
# TAB 1: ASK DATA (Conversational Interface)
# ==============================================================================
with tab_ask:
    # Empty State: Suggested Queries in 3 Categories
    if len(st.session_state.messages) == 0:
        st.markdown('<div class="muted-text" style="font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 10px;">Suggested Inquiries</div>', unsafe_allow_html=True)
        
        sq_col1, sq_col2, sq_col3 = st.columns(3)
        
        with sq_col1:
            st.markdown("**Sales Analytics**")
            if st.button("Top Product by Revenue\nIdentify highest earning product", key="sq_top_prod", use_container_width=True):
                st.session_state.pending_query = "Which product generated the highest revenue?"
            if st.button("Monthly Revenue Trend\nView chronological sales trajectory", key="sq_monthly", use_container_width=True):
                st.session_state.pending_query = "Show the monthly revenue trend."
            if st.button("Regional Sales Comparison\nCompare performance across regions", key="sq_region", use_container_width=True):
                st.session_state.pending_query = "Which region generated the highest sales?"

        with sq_col2:
            st.markdown("**Document Search**")
            if st.button("Electronics Return Policy\nCheck return window & restocking fees", key="sq_ret_policy", use_container_width=True):
                st.session_state.pending_query = "What is the return policy for electronics?"
            if st.button("Warranty Terms & Conditions\nInspect hardware and battery coverage", key="sq_warranty", use_container_width=True):
                st.session_state.pending_query = "What are the warranty conditions?"

        with sq_col3:
            st.markdown("**Combined Intelligence**")
            if st.button("Top Product & Return Policy\nJoint analysis of transactions and PDF", key="sq_comb_prod", use_container_width=True):
                st.session_state.pending_query = "Which electronics product generated the highest revenue, and what is its return policy?"
            if st.button("Category Sales & Discounts\nCheck top category and bulk discount terms", key="sq_comb_disc", use_container_width=True):
                st.session_state.pending_query = "Which category has the highest revenue, and does it have a special discount policy?"

        st.markdown("---")

    # Display Active Message History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg.get("tools_called") and st.session_state.df is not None:
                render_dynamic_visuals(msg["tools_called"], st.session_state.df)

            if msg.get("tools_called"):
                render_execution_details(msg["tools_called"])

    # Chat Input
    user_query = st.chat_input("Ask a question about sales transactions, company policies, or both...")

    # Process pending query triggered by suggested card
    if st.session_state.pending_query:
        user_query = st.session_state.pending_query
        st.session_state.pending_query = None

    if user_query:
        if not st.session_state.api_key.strip():
            st.error("Please enter and apply your Gemini API Key in the sidebar before asking questions.")
            st.stop()

        if st.session_state.agent is None:
            ensure_agent_compiled()
            if st.session_state.agent is None:
                st.error("Failed to compile agent. Verify your API key and data sources.")
                st.stop()

        # Append and Display User Query
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Execute Agent Workflow
        with st.chat_message("assistant"):
            with st.spinner("Analyzing structured data and document knowledge..."):
                try:
                    graph_messages = [SystemMessage(content=SYSTEM_PROMPT)]
                    for past_msg in st.session_state.messages:
                        if past_msg["role"] == "user":
                            graph_messages.append(HumanMessage(content=past_msg["content"]))
                        elif past_msg["role"] == "assistant":
                            graph_messages.append(AIMessage(content=past_msg["content"]))

                    result = st.session_state.agent.invoke({"messages": graph_messages})

                    all_msgs = result["messages"]
                    final_ai_msg = all_msgs[-1]
                    response_text = extract_text_content(final_ai_msg.content)

                    # Collect tools called
                    tools_called = []
                    for m in all_msgs:
                        if hasattr(m, "name") and m.name:
                            tools_called.append({
                                "name": m.name,
                                "output": str(m.content)
                            })

                    st.markdown(response_text)

                    # Dynamic Visualizations
                    if tools_called and st.session_state.df is not None:
                        render_dynamic_visuals(tools_called, st.session_state.df)

                    # Execution Details
                    if tools_called:
                        render_execution_details(tools_called)

                    # Persist response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "tools_called": tools_called
                    })

                except Exception as e:
                    error_msg = f"An error occurred: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "tools_called": []
                    })


# ==============================================================================
# TAB 2: DATA EXPLORER
# ==============================================================================
with tab_explorer:
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Summary KPI Cards
        st.markdown("#### Dataset Overview")
        k_col1, k_col2, k_col3, k_col4 = st.columns(4)
        with k_col1:
            st.metric("Total Records", f"{len(df):,}")
        with k_col2:
            st.metric("Total Columns", f"{df.shape[1]}")
        with k_col3:
            total_rev = df["revenue"].sum() if "revenue" in df.columns else 0
            st.metric("Total Revenue", f"${total_rev:,.2f}")
        with k_col4:
            avg_order = df["revenue"].mean() if "revenue" in df.columns else 0
            st.metric("Average Transaction", f"${avg_order:,.2f}")

        st.markdown("---")

        # Scrollable Dataset Preview
        st.markdown(f"#### Active Dataset: `{st.session_state.processed_csv_name}`")
        st.dataframe(df, use_container_width=True, height=280)

        # Download CSV Action
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Dataset as CSV",
            data=csv_bytes,
            file_name=f"export_{st.session_state.processed_csv_name or 'sales_data.csv'}",
            mime="text/csv",
            help="Download the active dataset directly to your local computer."
        )

        st.markdown("---")

        # Schema & Column Metadata
        st.markdown("#### Column Metadata & Integrity")
        schema_df = pd.DataFrame({
            "Column Name": df.columns,
            "Data Type": [str(t) for t in df.dtypes],
            "Non-Null Count": df.notnull().sum().values,
            "Missing Values": df.isnull().sum().values,
            "Sample Value": [str(df[c].iloc[0]) if len(df) > 0 else "" for c in df.columns]
        })
        st.dataframe(schema_df, hide_index=True, use_container_width=True)

        st.markdown("---")

        # Built-in Exploratory Charts
        st.markdown("#### Exploratory Visualizations")
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            if "category" in df.columns and "revenue" in df.columns:
                st.caption("Revenue by Category")
                cat_chart = df.groupby("category")["revenue"].sum().reset_index().set_index("category")
                st.bar_chart(cat_chart, color="#238B9E")
        with c_col2:
            if "region" in df.columns and "revenue" in df.columns:
                st.caption("Revenue by Geographic Region")
                reg_chart = df.groupby("region")["revenue"].sum().reset_index().set_index("region")
                st.bar_chart(reg_chart, color="#2878C7")

    else:
        st.markdown("""
        <div class="saas-card" style="text-align: center; padding: 36px 20px;">
            <div style="font-size: 1.1rem; font-weight: 600; color: #172B4D; margin-bottom: 6px;">No Structured Dataset Loaded</div>
            <div class="muted-text">Upload a CSV file or click 'Load Sample Dataset & Handbook' in the sidebar to begin analyzing your data.</div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# TAB 3: DOCUMENT LIBRARY
# ==============================================================================
with tab_docs:
    if st.session_state.raw_docs is not None:
        docs = st.session_state.raw_docs
        
        st.markdown(f"#### Active Document: `{st.session_state.processed_pdf_name}`")
        
        d_col1, d_col2, d_col3 = st.columns(3)
        with d_col1:
            st.metric("Total Pages Extracted", f"{len(docs)}")
        with d_col2:
            chunk_count = len(split_documents(docs))
            st.metric("Indexed Chunks", f"{chunk_count}")
        with d_col3:
            v_status = "ChromaDB Connected" if st.session_state.retriever else "Parsed (Awaiting Key)"
            st.metric("Vector Index", v_status)

        st.markdown("---")

        # Document Page Viewer
        st.markdown("#### Document Content Viewer")
        page_options = [f"Page {doc.metadata.get('page', idx + 1)}" for idx, doc in enumerate(docs)]
        selected_page_str = st.selectbox("Select Page to Inspect", options=page_options)
        
        selected_idx = page_options.index(selected_page_str)
        selected_doc = docs[selected_idx]

        st.markdown(f"""
        <div class="saas-card" style="background-color: #FFFFFF; font-family: monospace; font-size: 0.88rem; line-height: 1.6; white-space: pre-wrap;">
{selected_doc.page_content}
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="saas-card" style="text-align: center; padding: 36px 20px;">
            <div style="font-size: 1.1rem; font-weight: 600; color: #172B4D; margin-bottom: 6px;">No Document Knowledge Base Loaded</div>
            <div class="muted-text">Upload a PDF handbook or click 'Load Sample Dataset & Handbook' in the sidebar to enable document retrieval.</div>
        </div>
        """, unsafe_allow_html=True)
