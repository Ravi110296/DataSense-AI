"""
app.py - Streamlit User Interface for Intelligent Data Analyst Agent.

Provides:
- Sidebar configuration (Gemini API key, CSV & PDF uploaders, Load Sample Data button).
- File status indicators and dataset statistics.
- Interactive chat interface with message history.
- Tool invocation transparency and retrieved document citations.
"""

import os
import io
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from rag_utils import load_pdf_from_path_or_stream, split_documents, build_vector_store
from agent import build_analyst_agent, SYSTEM_PROMPT

# Load environment variables from .env
load_dotenv()

# Streamlit Page Configuration
st.set_page_config(
    page_title="Intelligent Data Analyst Agent with RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 0.375rem;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-success { background-color: #DEF7EC; color: #03543F; }
    .badge-warning { background-color: #FEF08A; color: #854D0E; }
    .tool-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 0.6rem 0.8rem;
        margin-top: 0.4rem;
        border-radius: 0.25rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


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


# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration & Data")
    
    # 1. API Key Setup
    env_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    api_key = st.text_input(
        "Google Gemini API Key",
        value=env_api_key,
        type="password",
        help="Free tier key from https://aistudio.google.com/"
    )
    
    # Free-Tier Model Selection
    selected_model = st.selectbox(
        "Gemini Model (Free Tier Compatible)",
        options=["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"],
        index=0,
        help="Standard free tier models with zero cost on Google AI Studio."
    )
    
    st.divider()
    
    # 2. Sample Data Quick-Loader
    st.subheader("📁 Quick Start")
    if st.button("🚀 Load Sample Dataset & Handbook", use_container_width=True):
        try:
            sample_csv_path = os.path.join("sample_data", "sales_data.csv")
            sample_pdf_path = os.path.join("sample_data", "company_handbook.pdf")
            
            if not os.path.exists(sample_csv_path) or not os.path.exists(sample_pdf_path):
                st.error("Sample files not found. Please run generate_sample_data.py first.")
            elif not api_key.strip():
                st.error("Please enter a valid Gemini API Key first.")
            else:
                with st.spinner("Processing sample CSV and PDF..."):
                    # Load CSV
                    df = pd.read_csv(sample_csv_path)
                    st.session_state.df = df
                    st.session_state.processed_csv_name = "sales_data.csv (Sample)"
                    
                    # Load and Index PDF
                    raw_docs = load_pdf_from_path_or_stream(sample_pdf_path, filename="company_handbook.pdf")
                    chunks = split_documents(raw_docs)
                    _, retriever = build_vector_store(chunks, api_key=api_key)
                    st.session_state.retriever = retriever
                    st.session_state.processed_pdf_name = "company_handbook.pdf (Sample)"
                    
                    # Compile Agent
                    st.session_state.agent = build_analyst_agent(
                        df=st.session_state.df,
                        retriever=st.session_state.retriever,
                        api_key=api_key,
                        model_name=selected_model
                    )
                st.success("Sample data loaded & agent initialized!")
                st.rerun()
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")

    st.divider()
    
    # 3. File Uploaders
    st.subheader("📤 Upload Custom Files")
    uploaded_csv = st.file_uploader("Upload Sales Data (CSV)", type=["csv"])
    uploaded_pdf = st.file_uploader("Upload Company Document (PDF)", type=["pdf"])
    
    if st.button("Process Uploaded Files", use_container_width=True):
        if not api_key.strip():
            st.error("Please provide your Gemini API Key above.")
        elif not uploaded_csv and not uploaded_pdf:
            st.warning("Please upload at least a CSV or PDF file.")
        else:
            with st.spinner("Processing uploaded files..."):
                try:
                    # Process CSV
                    if uploaded_csv is not None:
                        df = pd.read_csv(uploaded_csv)
                        st.session_state.df = df
                        st.session_state.processed_csv_name = uploaded_csv.name
                    
                    # Process PDF
                    if uploaded_pdf is not None:
                        raw_docs = load_pdf_from_path_or_stream(uploaded_pdf, filename=uploaded_pdf.name)
                        chunks = split_documents(raw_docs)
                        _, retriever = build_vector_store(chunks, api_key=api_key)
                        st.session_state.retriever = retriever
                        st.session_state.processed_pdf_name = uploaded_pdf.name
                    
                    # Compile Agent with new state
                    st.session_state.agent = build_analyst_agent(
                        df=st.session_state.df,
                        retriever=st.session_state.retriever,
                        api_key=api_key,
                        model_name=selected_model
                    )
                    st.success("Files processed & agent compiled successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to process files: {str(e)}")

    st.divider()

    # 4. Status Overview
    st.subheader("📊 System Status")
    if st.session_state.df is not None:
        st.markdown(
            f"**CSV Data:** `{st.session_state.processed_csv_name}`<br/>"
            f"<span class='status-badge badge-success'>Active: {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} cols</span>",
            unsafe_allow_html=True
        )
    else:
        st.markdown("<span class='status-badge badge-warning'>No CSV loaded</span>", unsafe_allow_html=True)

    st.write("")
    if st.session_state.retriever is not None:
        st.markdown(
            f"**PDF Knowledge:** `{st.session_state.processed_pdf_name}`<br/>"
            f"<span class='status-badge badge-success'>Vector Store Indexed</span>",
            unsafe_allow_html=True
        )
    else:
        st.markdown("<span class='status-badge badge-warning'>No PDF indexed</span>", unsafe_allow_html=True)

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Main Application Area
st.markdown('<div class="main-header">Intelligent Data Analyst Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Autonomous Agent powered by <b>LangChain</b>, <b>LangGraph</b>, and <b>ChromaDB RAG</b> '
    'to analyze structured CSV transactions and unstructured PDF business policies.</div>',
    unsafe_allow_html=True
)

# Suggested Questions Expander
with st.expander("💡 Example Questions to Try", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Structured Data Analysis**")
        st.markdown("- *Which product generated the highest revenue?*")
        st.markdown("- *What is the monthly revenue trend?*")
        st.markdown("- *Which region generated the highest revenue?*")
    with col2:
        st.markdown("**Document Retrieval (RAG)**")
        st.markdown("- *What is the company's return policy for electronics?*")
        st.markdown("- *What is the warranty period for furniture?*")
        st.markdown("- *Are there special conditions for returning kitchen appliances?*")
    with col3:
        st.markdown("**Combined Multi-Tool Query**")
        st.markdown("- *Which electronics product generated the highest revenue, and what is its return policy?*")
        st.markdown("- *Which category has the highest sales, and does the company offer discounts for it?*")

# Display Message History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "tools_called" in msg and msg["tools_called"]:
            with st.expander("🛠️ Tools Executed & Retrieved Context", expanded=False):
                for tool_call in msg["tools_called"]:
                    st.markdown(f"**Tool Invoked:** `{tool_call['name']}`")
                    st.code(tool_call.get("output", ""), language="text")

# Chat Input & Agent Execution
user_query = st.chat_input("Ask a question about the sales data, company policies, or both...")

if user_query:
    # 1. Validate Prerequisites
    if not api_key.strip():
        st.error("Please enter your Gemini API Key in the sidebar to proceed.")
        st.stop()

    if st.session_state.agent is None:
        # Build agent if not yet built
        try:
            st.session_state.agent = build_analyst_agent(
                df=st.session_state.df,
                retriever=st.session_state.retriever,
                api_key=api_key,
                model_name=selected_model
            )
        except Exception as e:
            st.error(f"Initialization error: {str(e)}")
            st.stop()

    # 2. Display User Message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 3. Execute LangGraph Agent
    with st.chat_message("assistant"):
        with st.spinner("Analyzing and reasoning..."):
            try:
                # Prepare conversation messages for LangGraph
                graph_messages = [SystemMessage(content=SYSTEM_PROMPT)]
                for past_msg in st.session_state.messages:
                    if past_msg["role"] == "user":
                        graph_messages.append(HumanMessage(content=past_msg["content"]))
                    elif past_msg["role"] == "assistant":
                        graph_messages.append(AIMessage(content=past_msg["content"]))

                # Invoke LangGraph
                result = st.session_state.agent.invoke({"messages": graph_messages})
                
                # Extract final response
                all_msgs = result["messages"]
                final_ai_msg = all_msgs[-1]
                response_text = final_ai_msg.content

                # Extract intermediate tool executions
                tools_called = []
                for m in all_msgs:
                    # Check for ToolMessage (output of tools)
                    if hasattr(m, "name") and m.name:
                        tools_called.append({
                            "name": m.name,
                            "output": str(m.content)
                        })

                # Display response
                st.markdown(response_text)

                # Display tool execution details if any tools were invoked
                if tools_called:
                    with st.expander("🛠️ Tools Executed & Retrieved Context", expanded=False):
                        for tool_call in tools_called:
                            st.markdown(f"**Tool Invoked:** `{tool_call['name']}`")
                            st.code(tool_call.get("output", ""), language="text")

                # Save assistant response to state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "tools_called": tools_called
                })

            except Exception as e:
                error_msg = f"An error occurred while processing your query: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "tools_called": []
                })
