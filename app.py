"""
app.py - Enterprise UI for Intelligent Data Analyst Agent with RAG.

Features:
- Professional SaaS design matching enterprise analytics dashboards.
- Zero emojis; clean typography and minimal slate/navy visual hierarchy.
- Dark sidebar with configuration, file uploaders, and system metrics.
- Top navigation bar with dashboard metadata.
- Interactive chat interface with automated visual widgets (KPI cards, charts, and tables).
- Expandable technical execution trace and document citations.
"""

import os
import io
from typing import Any
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from rag_utils import load_pdf_from_path_or_stream, split_documents, build_vector_store
from agent import build_analyst_agent, SYSTEM_PROMPT

# Load environment variables
load_dotenv()

# Streamlit Page Configuration
st.set_page_config(
    page_title="AI Analyst - Enterprise Data & Document Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Theme (Matching Enterprise SaaS Dashboard)
st.markdown("""
<style>
    /* Global Font & Reset */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Background & Container */
    .stApp {
        background-color: #F8FAFC;
    }
    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    /* Dark Enterprise Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B;
    }
    [data-testid="stSidebar"] * {
        color: #F1F5F9;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #F8FAFC !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        margin-top: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
        color: #94A3B8 !important;
        font-size: 0.85rem !important;
    }
    [data-testid="stSidebar"] input {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
        font-size: 0.85rem !important;
    }
    [data-testid="stSidebar"] input:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 1px #3B82F6 !important;
    }
    [data-testid="stSidebar"] button {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        padding: 0.45rem 0.75rem !important;
        transition: all 0.15s ease-in-out !important;
    }
    [data-testid="stSidebar"] button:hover {
        background-color: #2563EB !important;
        border-color: #2563EB !important;
        color: #FFFFFF !important;
    }

    /* Top Navigation Bar */
    .enterprise-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #0F172A;
        color: #FFFFFF;
        padding: 0.85rem 1.4rem;
        border-radius: 8px;
        margin-bottom: 1.25rem;
        border: 1px solid #1E293B;
    }
    .header-left {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .header-back {
        color: #64748B;
        font-weight: 400;
        cursor: default;
    }
    .header-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #FFFFFF;
        letter-spacing: -0.01em;
    }
    .header-badge {
        background-color: #1E293B;
        color: #38BDF8;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        border: 1px solid #334155;
    }
    .header-right {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        font-size: 0.85rem;
        color: #94A3B8;
    }
    .header-nav-item {
        color: #94A3B8;
        text-decoration: none;
        font-weight: 500;
    }
    .header-nav-item:hover {
        color: #FFFFFF;
    }

    /* KPI Metric Card */
    .kpi-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin: 0.8rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        display: inline-block;
        min-width: 240px;
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
    }

    /* Quick Inquiries Row */
    .quick-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }

    /* Status Pill */
    .status-pill {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .pill-active {
        background-color: #1E293B;
        color: #38BDF8;
        border: 1px solid #334155;
    }
    .pill-muted {
        background-color: #1E293B;
        color: #64748B;
        border: 1px solid #334155;
    }

    /* Clean Streamlit Chat Elements */
    [data-testid="stChatMessage"] {
        padding: 0.75rem 0.25rem !important;
        background-color: transparent !important;
    }
    
    /* Document Citation Box */
    .citation-box {
        background-color: #F8FAFC;
        border-left: 3px solid #2563EB;
        padding: 0.6rem 0.9rem;
        margin-top: 0.6rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.82rem;
        color: #475569;
    }
</style>
""", unsafe_allow_html=True)


def extract_text_content(content: Any) -> str:
    """Safely extract plain text from Gemini message content, handling string or block lists."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
            elif hasattr(block, "text"):
                parts.append(str(block.text))
        return "".join(parts) if parts else str(content)
    return str(content)


# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "processed_csv_name" not in st.session_state:
    st.session_state.processed_csv_name = None
if "processed_pdf_name" not in st.session_state:
    st.session_state.processed_pdf_name = None
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# Sidebar: Enterprise Controls
with st.sidebar:
    st.markdown("### Authentication")
    api_key = st.text_input(
        "Google Gemini API Key",
        value="",
        type="password",
        placeholder="Paste API key here...",
        help="Free tier key from Google AI Studio (aistudio.google.com)"
    )
    
    st.markdown("---")
    
    st.markdown("### Quick Setup")
    if st.button("Load Sample Dataset & Handbook", use_container_width=True):
        try:
            sample_csv_path = os.path.join("sample_data", "sales_data.csv")
            sample_pdf_path = os.path.join("sample_data", "company_handbook.pdf")
            
            if not os.path.exists(sample_csv_path) or not os.path.exists(sample_pdf_path):
                st.error("Sample files not found on disk.")
            elif not api_key.strip():
                st.error("Enter Gemini API key above.")
            else:
                with st.spinner("Indexing sample data..."):
                    # Load CSV
                    df = pd.read_csv(sample_csv_path)
                    st.session_state.df = df
                    st.session_state.processed_csv_name = "sales_data.csv"
                    
                    # Load and Index PDF
                    raw_docs = load_pdf_from_path_or_stream(sample_pdf_path, filename="company_handbook.pdf")
                    chunks = split_documents(raw_docs)
                    _, retriever = build_vector_store(chunks, api_key=api_key)
                    st.session_state.retriever = retriever
                    st.session_state.processed_pdf_name = "company_handbook.pdf"
                    
                    # Compile Agent
                    st.session_state.agent = build_analyst_agent(
                        df=st.session_state.df,
                        retriever=st.session_state.retriever,
                        api_key=api_key
                    )
                st.success("Sample data ready.")
                st.rerun()
        except Exception as e:
            st.error(f"Setup failed: {str(e)}")

    st.markdown("---")
    
    st.markdown("### Data Upload")
    uploaded_csv = st.file_uploader("Structured Data (CSV)", type=["csv"])
    uploaded_pdf = st.file_uploader("Document Knowledge (PDF)", type=["pdf"])
    
    if st.button("Process Files", use_container_width=True):
        if not api_key.strip():
            st.error("Enter Gemini API key above.")
        elif not uploaded_csv and not uploaded_pdf:
            st.warning("Upload at least one CSV or PDF file.")
        else:
            with st.spinner("Processing files..."):
                try:
                    if uploaded_csv is not None:
                        df = pd.read_csv(uploaded_csv)
                        st.session_state.df = df
                        st.session_state.processed_csv_name = uploaded_csv.name
                    
                    if uploaded_pdf is not None:
                        raw_docs = load_pdf_from_path_or_stream(uploaded_pdf, filename=uploaded_pdf.name)
                        chunks = split_documents(raw_docs)
                        _, retriever = build_vector_store(chunks, api_key=api_key)
                        st.session_state.retriever = retriever
                        st.session_state.processed_pdf_name = uploaded_pdf.name
                    
                    st.session_state.agent = build_analyst_agent(
                        df=st.session_state.df,
                        retriever=st.session_state.retriever,
                        api_key=api_key
                    )
                    st.success("Files processed successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Processing error: {str(e)}")

    st.markdown("---")

    st.markdown("### System Context")
    if st.session_state.df is not None:
        st.markdown(
            f"**Dataset:** `{st.session_state.processed_csv_name}`<br/>"
            f"<span class='status-pill pill-active'>{st.session_state.df.shape[0]} Rows &times; {st.session_state.df.shape[1]} Columns</span>",
            unsafe_allow_html=True
        )
    else:
        st.markdown("<span class='status-pill pill-muted'>No dataset loaded</span>", unsafe_allow_html=True)

    st.write("")
    if st.session_state.retriever is not None:
        st.markdown(
            f"**Knowledge Base:** `{st.session_state.processed_pdf_name}`<br/>"
            f"<span class='status-pill pill-active'>Vector Store Indexed</span>",
            unsafe_allow_html=True
        )
    else:
        st.markdown("<span class='status-pill pill-muted'>No document indexed</span>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Top Navigation Bar (SaaS Header)
st.markdown("""
<div class="enterprise-header">
    <div class="header-left">
        <span class="header-back">&larr;</span>
        <span class="header-title">AI Analyst</span>
        <span class="header-badge">LangGraph &bull; RAG</span>
    </div>
    <div class="header-right">
        <span class="header-nav-item">Dashboard</span>
        <span class="header-nav-item">Reports</span>
        <span class="header-nav-item">Settings</span>
    </div>
</div>
""", unsafe_allow_html=True)


# Quick Inquiries Bar
st.markdown('<div class="quick-label">Suggested Inquiries</div>', unsafe_allow_html=True)
pill_col1, pill_col2, pill_col3, pill_col4 = st.columns(4)

with pill_col1:
    if st.button("Top Products by Revenue", use_container_width=True):
        st.session_state.pending_query = "Which product generated the highest revenue?"
with pill_col2:
    if st.button("Monthly Revenue Trend", use_container_width=True):
        st.session_state.pending_query = "What is the monthly revenue trend?"
with pill_col3:
    if st.button("Electronics Return Policy", use_container_width=True):
        st.session_state.pending_query = "What is the return policy for electronics?"
with pill_col4:
    if st.button("Combined Analysis", use_container_width=True):
        st.session_state.pending_query = "Which electronics product generated the highest revenue, and what is its return policy?"


# Render Visual Widgets Based on Executed Tools
def render_tool_visuals(tools_called: list, df: pd.DataFrame):
    """Render clean charts, KPI cards, or tables matching the user reference design."""
    tool_names = [t["name"] for t in tools_called]
    
    # 1. Monthly Revenue Trend -> Render Clean Line Chart
    if "get_monthly_revenue_trend" in tool_names and df is not None:
        try:
            date_col = next((c for c in ["order_date", "date", "OrderDate"] if c in df.columns), None)
            if date_col and "revenue" in df.columns:
                df_copy = df.copy()
                df_copy["parsed_date"] = pd.to_datetime(df_copy[date_col], errors="coerce")
                df_copy = df_copy.dropna(subset=["parsed_date"])
                df_copy["Month"] = df_copy["parsed_date"].dt.strftime("%Y-%m")
                monthly_rev = df_copy.groupby("Month")["revenue"].sum().reset_index()
                monthly_rev = monthly_rev.rename(columns={"revenue": "Revenue ($)"})
                st.caption("Sales by Month")
                st.line_chart(monthly_rev.set_index("Month"), color="#2563EB", use_container_width=True)
        except Exception:
            pass

    # 2. Revenue Analysis -> Render Clean KPI Card & Category Bar Chart
    if "get_revenue_analysis" in tool_names and df is not None:
        try:
            total_rev = df["revenue"].sum() if "revenue" in df.columns else 0
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Revenue</div>
                <div class="kpi-value">${total_rev:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if "category" in df.columns and "revenue" in df.columns:
                cat_rev = df.groupby("category")["revenue"].sum().reset_index()
                cat_rev = cat_rev.rename(columns={"category": "Category", "revenue": "Revenue ($)"})
                st.caption("Sales by Category")
                st.bar_chart(cat_rev.set_index("Category"), color="#3B82F6", use_container_width=True)
        except Exception:
            pass

    # 3. Top Products -> Render Clean Data Table
    if "get_top_products" in tool_names and df is not None:
        try:
            top_df = (
                df.groupby(["product", "category"] if "category" in df.columns else ["product"])["revenue"]
                .sum()
                .reset_index()
                .sort_values(by="revenue", ascending=False)
                .head(5)
            )
            top_df = top_df.rename(columns={"product": "Product", "category": "Category", "revenue": "Revenue ($)"})
            top_df["Revenue ($)"] = top_df["Revenue ($)"].apply(lambda x: f"${x:,.2f}")
            st.caption("Top Revenue Generating Products")
            st.dataframe(top_df, hide_index=True, use_container_width=True)
        except Exception:
            pass


# Display Message History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Render visual charts / KPI widgets if tools were run
        if msg.get("tools_called") and st.session_state.df is not None:
            render_tool_visuals(msg["tools_called"], st.session_state.df)
            
        # Technical execution trace
        if msg.get("tools_called"):
            with st.expander("Execution Trace & Retrieved Context", expanded=False):
                for tool_call in msg["tools_called"]:
                    st.markdown(f"**Tool:** `{tool_call['name']}`")
                    st.code(tool_call.get("output", ""), language="text")


# Chat Input
user_query = st.chat_input("Ask a question about sales data, company policies, or both...")

# Handle pending query from quick action buttons
if st.session_state.pending_query:
    user_query = st.session_state.pending_query
    st.session_state.pending_query = None

if user_query:
    # 1. Validate API Key
    if not api_key.strip():
        st.error("Please enter your Gemini API Key in the sidebar to proceed.")
        st.stop()

    # 2. Build Agent If Needed
    if st.session_state.agent is None:
        try:
            st.session_state.agent = build_analyst_agent(
                df=st.session_state.df,
                retriever=st.session_state.retriever,
                api_key=api_key
            )
        except Exception as e:
            st.error(f"Agent initialization error: {str(e)}")
            st.stop()

    # 3. Record User Query
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 4. Invoke Agent
    with st.chat_message("assistant"):
        with st.spinner("Analyzing data and documents..."):
            try:
                graph_messages = [SystemMessage(content=SYSTEM_PROMPT)]
                for past_msg in st.session_state.messages:
                    if past_msg["role"] == "user":
                        graph_messages.append(HumanMessage(content=past_msg["content"]))
                    elif past_msg["role"] == "assistant":
                        graph_messages.append(AIMessage(content=past_msg["content"]))

                # Execute LangGraph
                result = st.session_state.agent.invoke({"messages": graph_messages})
                
                # Extract text and tools
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

                # Display Text Response
                st.markdown(response_text)

                # Render Visual Widgets (Charts, KPI Cards, Tables)
                if tools_called and st.session_state.df is not None:
                    render_tool_visuals(tools_called, st.session_state.df)

                # Display Execution Trace
                if tools_called:
                    with st.expander("Execution Trace & Retrieved Context", expanded=False):
                        for tool_call in tools_called:
                            st.markdown(f"**Tool:** `{tool_call['name']}`")
                            st.code(tool_call.get("output", ""), language="text")

                # Persist to Session State
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "tools_called": tools_called
                })

            except Exception as e:
                error_msg = f"An error occurred while processing the request: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "tools_called": []
                })
