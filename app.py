"""
app.py - Intelligent Data Analyst Agent with RAG.

Clean, professional enterprise interface for querying structured CSV data and unstructured PDF documents.
Supports deterministic analytical tools, dynamic data visualization (charts), and document retrieval.
"""

import os
import json
import re
from typing import Any
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from rag_utils import load_pdf_from_path_or_stream, split_documents, build_vector_store
from agent import build_analyst_agent, SYSTEM_PROMPT

# Load environment variables if available
load_dotenv()

# Streamlit Page Setup
st.set_page_config(
    page_title="Intelligent Data Analyst Agent",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, Professional Styling (Cohesive light-slate palette, high contrast, zero broken CSS)
st.markdown("""
<style>
    /* Clean layout */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }
    
    /* Header section */
    .app-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.25rem;
        line-height: 1.4;
    }
    
    /* Status banner */
    .status-container {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.85rem;
    }
    .status-tag {
        font-weight: 600;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
    }
    .status-ready {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }
    .status-waiting {
        background-color: #FFFBEB;
        color: #92400E;
        border: 1px solid #FDE68A;
    }

    /* KPI metric card */
    .kpi-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin: 0.6rem 0;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
        display: inline-block;
        min-width: 220px;
    }
    .kpi-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-val {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)


def extract_text_content(content: Any) -> str:
    """Extract plain text from LLM response blocks, stripping internal chart comments."""
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

    # Clean out internal chart JSON comments from text display
    text = re.sub(r"<!-- CHART_JSON:.*? -->", "", text, flags=re.DOTALL).strip()
    return text


# Initialize Session State
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
if "processed_csv_name" not in st.session_state:
    st.session_state.processed_csv_name = None
if "processed_pdf_name" not in st.session_state:
    st.session_state.processed_pdf_name = None
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# Sidebar: Clean & Functional Controls
with st.sidebar:
    st.subheader("API Configuration")
    api_key = st.text_input(
        "Google Gemini API Key",
        value="",
        type="password",
        placeholder="Paste your Gemini API key here..."
    )
    st.caption("Press Enter after entering your key to apply.")

    st.divider()

    st.subheader("Sample Data")
    st.caption("Test the application with pre-built synthetic transactions and policy handbook.")
    if st.button("Load Sample Dataset & Handbook", use_container_width=True):
        sample_csv_path = os.path.join("sample_data", "sales_data.csv")
        sample_pdf_path = os.path.join("sample_data", "company_handbook.pdf")
        
        if not os.path.exists(sample_csv_path) or not os.path.exists(sample_pdf_path):
            st.error("Sample data files not found on disk. Run generate_sample_data.py first.")
        else:
            with st.spinner("Loading sample data..."):
                try:
                    # 1. Load CSV (no API key required)
                    df = pd.read_csv(sample_csv_path)
                    st.session_state.df = df
                    st.session_state.processed_csv_name = "sales_data.csv"
                    
                    # 2. Extract PDF text (no API key required)
                    raw_docs = load_pdf_from_path_or_stream(sample_pdf_path, filename="company_handbook.pdf")
                    st.session_state.raw_docs = raw_docs
                    st.session_state.processed_pdf_name = "company_handbook.pdf"
                    
                    # 3. If API key is present, build retriever and agent immediately
                    if api_key.strip():
                        chunks = split_documents(raw_docs)
                        _, retriever = build_vector_store(chunks, api_key=api_key)
                        st.session_state.retriever = retriever
                        st.session_state.agent = build_analyst_agent(
                            df=st.session_state.df,
                            retriever=st.session_state.retriever,
                            api_key=api_key
                        )
                        st.success("Sample data loaded and AI agent initialized.")
                    else:
                        st.success("Sample data loaded. Enter your API key above whenever you are ready to query.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading sample data: {str(e)}")

    st.divider()

    st.subheader("Upload Custom Files")
    uploaded_csv = st.file_uploader("Upload CSV Data", type=["csv"])
    uploaded_pdf = st.file_uploader("Upload PDF Document", type=["pdf"])

    if st.button("Process Uploaded Files", use_container_width=True):
        if not uploaded_csv and not uploaded_pdf:
            st.warning("Please upload a CSV or PDF file to process.")
        else:
            with st.spinner("Processing files..."):
                try:
                    if uploaded_csv is not None:
                        st.session_state.df = pd.read_csv(uploaded_csv)
                        st.session_state.processed_csv_name = uploaded_csv.name

                    if uploaded_pdf is not None:
                        st.session_state.raw_docs = load_pdf_from_path_or_stream(uploaded_pdf, filename=uploaded_pdf.name)
                        st.session_state.processed_pdf_name = uploaded_pdf.name

                    # Build agent if key is present
                    if api_key.strip():
                        retriever = None
                        if st.session_state.raw_docs:
                            chunks = split_documents(st.session_state.raw_docs)
                            _, retriever = build_vector_store(chunks, api_key=api_key)
                            st.session_state.retriever = retriever

                        st.session_state.agent = build_analyst_agent(
                            df=st.session_state.df,
                            retriever=st.session_state.retriever,
                            api_key=api_key
                        )
                        st.success("Files processed and agent compiled.")
                    else:
                        st.success("Files loaded. Enter API key above to query.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to process files: {str(e)}")

    st.divider()

    # Data Sources & Status Panel
    st.subheader("Data Sources & Status")
    
    if st.session_state.df is not None:
        st.markdown(f"**Structured Data:** `{st.session_state.processed_csv_name}`")
        st.caption(f"{len(st.session_state.df)} records | Columns: {', '.join(list(st.session_state.df.columns)[:4])}...")
    else:
        st.caption("Structured Data: None loaded")

    if st.session_state.raw_docs is not None:
        st.markdown(f"**Document:** `{st.session_state.processed_pdf_name}`")
        st.caption(f"{len(st.session_state.raw_docs)} pages extracted")
    else:
        st.caption("Document: None loaded")

    if st.session_state.agent is not None:
        st.markdown("<span class='status-tag status-ready'>Agent Ready</span>", unsafe_allow_html=True)
    elif api_key.strip() and (st.session_state.df is not None or st.session_state.raw_docs is not None):
        st.markdown("<span class='status-tag status-ready'>Ready to initialize</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='status-tag status-waiting'>Awaiting API Key / Data</span>", unsafe_allow_html=True)

    st.write("")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Main Application Area
st.markdown('<div class="app-title">Intelligent Data Analyst Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Autonomous reasoning agent built with LangChain, LangGraph, and RAG. '
    'Answers analytical queries using structured CSV transactions and policy queries using PDF documentation.</div>',
    unsafe_allow_html=True
)

# Suggested Inquiries (Clean Clickable Buttons)
st.caption("SUGGESTED INQUIRIES")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Highest Revenue Product", use_container_width=True):
        st.session_state.pending_query = "Which product generated the highest total revenue?"
with col2:
    if st.button("Monthly Revenue Trend", use_container_width=True):
        st.session_state.pending_query = "What is the monthly revenue trend?"
with col3:
    if st.button("Electronics Return Policy", use_container_width=True):
        st.session_state.pending_query = "What is the return policy for electronics?"
with col4:
    if st.button("Combined: Revenue & Policy", use_container_width=True):
        st.session_state.pending_query = "Which electronics product generated the highest revenue, and what is its return policy?"


# Dynamic Chart & Widget Rendering
def render_dynamic_visuals(tools_called: list, df: pd.DataFrame):
    """
    Renders generalizable charts and visual cards dynamically based on tool outputs.
    Works for any chart requested via plot_chart or standard analytical tools.
    """
    if df is None:
        return

    for tool_call in tools_called:
        output_str = tool_call.get("output", "")
        tool_name = tool_call.get("name", "")

        # 1. Check for dynamic chart payload generated by plot_chart
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
                    st.caption(f"{title} ({y_label} by {x_label})")
                    chart_data = chart_df.set_index("x")[["y"]]
                    chart_data.columns = [y_label]
                    
                    if chart_type == "line":
                        st.line_chart(chart_data, color="#2563EB", use_container_width=True)
                    elif chart_type == "area":
                        st.area_chart(chart_data, color="#3B82F6", use_container_width=True)
                    else:
                        st.bar_chart(chart_data, color="#2563EB", use_container_width=True)
                    continue
            except Exception:
                pass

        # 2. Fallback visual for get_revenue_analysis
        if tool_name == "get_revenue_analysis":
            try:
                total_rev = df["revenue"].sum() if "revenue" in df.columns else 0
                st.markdown(f"""
                <div class="kpi-box">
                    <div class="kpi-title">Total Revenue</div>
                    <div class="kpi-val">${total_rev:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
                if "category" in df.columns and "revenue" in df.columns:
                    cat_rev = df.groupby("category")["revenue"].sum().reset_index().set_index("category")
                    cat_rev.columns = ["Revenue ($)"]
                    st.caption("Revenue by Category")
                    st.bar_chart(cat_rev, color="#2563EB", use_container_width=True)
            except Exception:
                pass

        # 3. Fallback visual for get_monthly_revenue_trend
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
                    st.caption("Chronological Monthly Sales Trend")
                    st.line_chart(m_rev, color="#2563EB", use_container_width=True)
            except Exception:
                pass

        # 4. Fallback visual for get_top_products
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
                    st.caption("Top Products Ranked by Revenue")
                    st.dataframe(top_prods, hide_index=True, use_container_width=True)
            except Exception:
                pass


# Display Existing Conversation History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Render dynamic visual charts or KPI cards
        if msg.get("tools_called") and st.session_state.df is not None:
            render_dynamic_visuals(msg["tools_called"], st.session_state.df)

        # Technical Execution Trace
        if msg.get("tools_called"):
            with st.expander("Execution Trace & Retrieved Context", expanded=False):
                for tool_call in msg["tools_called"]:
                    clean_output = re.sub(r"<!-- CHART_JSON:.*? -->", "", tool_call.get("output", ""), flags=re.DOTALL).strip()
                    st.markdown(f"**Tool:** `{tool_call['name']}`")
                    st.code(clean_output, language="text")


# Chat Input Handling
user_query = st.chat_input("Ask an analytical question or search policy handbook...")

# Check for button click triggers
if st.session_state.pending_query:
    user_query = st.session_state.pending_query
    st.session_state.pending_query = None

if user_query:
    # 1. Validation
    if not api_key.strip():
        st.error("Please enter your Gemini API Key in the sidebar to run the agent.")
        st.stop()

    # 2. Ensure Agent is Compiled
    if st.session_state.agent is None:
        try:
            # Build retriever if document is loaded
            if st.session_state.retriever is None and st.session_state.raw_docs is not None:
                chunks = split_documents(st.session_state.raw_docs)
                _, retriever = build_vector_store(chunks, api_key=api_key)
                st.session_state.retriever = retriever

            st.session_state.agent = build_analyst_agent(
                df=st.session_state.df,
                retriever=st.session_state.retriever,
                api_key=api_key
            )
        except Exception as e:
            st.error(f"Agent initialization error: {str(e)}")
            st.stop()

    # 3. Add and Display User Message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 4. Run Agent Workflow
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
                
                all_msgs = result["messages"]
                final_ai_msg = all_msgs[-1]
                response_text = extract_text_content(final_ai_msg.content)

                # Collect tool calls
                tools_called = []
                for m in all_msgs:
                    if hasattr(m, "name") and m.name:
                        tools_called.append({
                            "name": m.name,
                            "output": str(m.content)
                        })

                # Display text response
                st.markdown(response_text)

                # Render dynamic charts or tables if tools were executed
                if tools_called and st.session_state.df is not None:
                    render_dynamic_visuals(tools_called, st.session_state.df)

                # Display Execution Trace
                if tools_called:
                    with st.expander("Execution Trace & Retrieved Context", expanded=False):
                        for tool_call in tools_called:
                            clean_output = re.sub(r"<!-- CHART_JSON:.*? -->", "", tool_call.get("output", ""), flags=re.DOTALL).strip()
                            st.markdown(f"**Tool:** `{tool_call['name']}`")
                            st.code(clean_output, language="text")

                # Store assistant response in session state
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
