"""
agent.py - Core Agent definition using LangChain, LangGraph, and Pandas Tools.

Implements:
- Structured data analysis tools using Pandas (Overview, Revenue, Top Products, Monthly Trend, Regional).
- Unstructured document retrieval tool (RAG).
- System prompt guiding tool selection and multi-tool synthesis.
- LangGraph compiled workflow: START -> Agent -> Tools -> Agent -> END.
"""

import os
from typing import Optional, List, Dict, Any
import pandas as pd

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from rag_utils import format_chunks_with_sources


SYSTEM_PROMPT = """You are an Intelligent Data Analyst Assistant capable of answering questions using two sources of truth:
1. Structured CSV data (sales transactions, revenue, product lines, regional metrics).
2. Unstructured PDF document knowledge (company handbook, warranties, return policies, discounts).

Guidelines:
- Analyze user requests carefully to determine which tool or combination of tools is required.
- For quantitative, numerical, or sales inquiries (revenue, top products, trends, regions), invoke the appropriate data analysis tool.
- For business policies, return conditions, warranties, discounts, or terms, invoke the 'retrieve_document_knowledge' tool.
- For COMBINED queries (e.g., 'Which electronics product generated the highest revenue, and what is its return policy?'):
  1. First invoke the data analysis tool to discover the specific product or category.
  2. Next invoke the document retrieval tool with that specific product or category to retrieve its policy.
  3. Synthesize both findings into a clear, unified, and coherent response.
- Ground all answers strictly in the tool outputs. Do not fabricate numerical data or policy rules.
- If retrieved document information is cited, include the source file name and page number.
- If information is missing from either the dataset or the document, clearly state that it is not available.
"""


def create_analyst_tools(df: Optional[pd.DataFrame], retriever: Optional[Any]) -> list:
    """
    Create tool functions bound to the current DataFrame and Document Retriever.
    """

    @tool
    def get_dataset_overview() -> str:
        """
        Get high-level summary of the sales dataset: row count, column count,
        column names, and missing-value counts.
        """
        if df is None or df.empty:
            return "No dataset loaded. Please upload a CSV file."

        num_rows, num_cols = df.shape
        columns = list(df.columns)
        missing_counts = df.isnull().sum().to_dict()
        missing_summary = ", ".join([f"{k}: {v}" for k, v in missing_counts.items() if v > 0])
        if not missing_summary:
            missing_summary = "None (clean dataset)"

        return (
            f"Dataset Overview:\n"
            f"- Total Records (Rows): {num_rows}\n"
            f"- Total Columns: {num_cols}\n"
            f"- Column Names: {', '.join(columns)}\n"
            f"- Missing Values: {missing_summary}"
        )

    @tool
    def get_revenue_analysis() -> str:
        """
        Compute total company revenue, overall transaction count, average order value,
        and revenue breakdown by product category.
        """
        if df is None or df.empty:
            return "No dataset loaded. Please upload a CSV file."

        if "revenue" not in df.columns:
            return "Column 'revenue' is missing from the dataset."

        total_rev = df["revenue"].sum()
        total_orders = len(df)
        avg_rev = df["revenue"].mean()

        lines = [
            f"Revenue Summary:",
            f"- Total Revenue: ${total_rev:,.2f}",
            f"- Total Transactions: {total_orders}",
            f"- Average Transaction Value: ${avg_rev:,.2f}\n",
            "Revenue by Category:"
        ]

        if "category" in df.columns:
            cat_group = (
                df.groupby("category")["revenue"]
                .agg(["sum", "count"])
                .rename(columns={"sum": "total_revenue", "count": "order_count"})
                .sort_values(by="total_revenue", ascending=False)
            )
            for cat, row in cat_group.iterrows():
                pct = (row["total_revenue"] / total_rev) * 100 if total_rev > 0 else 0
                lines.append(
                    f"  * {cat}: ${row['total_revenue']:,.2f} ({pct:.1f}% of total, {int(row['order_count'])} orders)"
                )
        else:
            lines.append("  (Category column not found)")

        return "\n".join(lines)

    @tool
    def get_top_products(n: int = 5, category: Optional[str] = None) -> str:
        """
        Identify top revenue-generating products.
        Can optionally filter by category (e.g. 'Electronics', 'Furniture').
        Returns the top N products ranked by total revenue.
        """
        if df is None or df.empty:
            return "No dataset loaded. Please upload a CSV file."

        if "revenue" not in df.columns or "product" not in df.columns:
            return "Required columns ('product', 'revenue') not found in dataset."

        filtered_df = df
        filter_msg = ""
        if category and "category" in df.columns:
            # Case-insensitive category match
            mask = df["category"].str.strip().str.lower() == category.strip().lower()
            filtered_df = df[mask]
            filter_msg = f" in category '{category}'"
            if filtered_df.empty:
                available_cats = ", ".join(df["category"].dropna().unique())
                return f"No products found for category '{category}'. Available categories: {available_cats}"

        prod_group = (
            filtered_df.groupby(["product", "category"] if "category" in filtered_df.columns else ["product"])
            .agg(total_revenue=("revenue", "sum"), total_qty=("quantity", "sum") if "quantity" in filtered_df.columns else ("revenue", "count"))
            .sort_values(by="total_revenue", ascending=False)
            .head(n)
        )

        lines = [f"Top {len(prod_group)} Products{filter_msg} by Revenue:"]
        rank = 1
        for idx, row in prod_group.iterrows():
            prod_name = idx[0] if isinstance(idx, tuple) else idx
            cat_name = f" ({idx[1]})" if isinstance(idx, tuple) else ""
            qty_info = f", Units Sold: {int(row['total_qty'])}" if "quantity" in filtered_df.columns else ""
            lines.append(f"  {rank}. {prod_name}{cat_name}: ${row['total_revenue']:,.2f}{qty_info}")
            rank += 1

        return "\n".join(lines)

    @tool
    def get_monthly_revenue_trend() -> str:
        """
        Aggregate revenue by month in chronological order to observe business sales trends.
        """
        if df is None or df.empty:
            return "No dataset loaded. Please upload a CSV file."

        # Detect date column
        date_col = next((c for c in ["order_date", "date", "OrderDate"] if c in df.columns), None)
        if not date_col:
            return "No recognized date column (e.g. 'order_date') found in the dataset."

        df_copy = df.copy()
        df_copy["parsed_date"] = pd.to_datetime(df_copy[date_col], errors="coerce")
        df_copy = df_copy.dropna(subset=["parsed_date"])

        if df_copy.empty:
            return "Date column contains invalid or unparseable date values."

        df_copy["month_period"] = df_copy["parsed_date"].dt.to_period("M")
        monthly = (
            df_copy.groupby("month_period")["revenue"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "total_revenue", "count": "order_count"})
            .sort_index()
        )

        lines = ["Monthly Revenue Trend:"]
        for month, row in monthly.iterrows():
            lines.append(f"  * {str(month)}: ${row['total_revenue']:,.2f} ({int(row['order_count'])} orders)")

        return "\n".join(lines)

    @tool
    def get_regional_analysis() -> str:
        """
        Compare revenue performance and transaction volume across geographic regions.
        """
        if df is None or df.empty:
            return "No dataset loaded. Please upload a CSV file."

        reg_col = next((c for c in ["region", "Region", "territory"] if c in df.columns), None)
        if not reg_col:
            return "No regional column (e.g. 'region') found in the dataset."

        total_rev = df["revenue"].sum()
        reg_group = (
            df.groupby(reg_col)["revenue"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "total_revenue", "count": "order_count"})
            .sort_values(by="total_revenue", ascending=False)
        )

        lines = ["Regional Performance Comparison:"]
        for reg, row in reg_group.iterrows():
            pct = (row["total_revenue"] / total_rev) * 100 if total_rev > 0 else 0
            lines.append(
                f"  * {reg}: ${row['total_revenue']:,.2f} ({pct:.1f}% of total, {int(row['order_count'])} orders)"
            )

        return "\n".join(lines)

    @tool
    def retrieve_document_knowledge(query: str) -> str:
        """
        Retrieve relevant business policies, return guidelines, warranty details,
        or product information from uploaded PDF documents.
        """
        if retriever is None:
            return "No PDF document loaded. Please upload a PDF file to enable document search."

        try:
            docs = retriever.invoke(query)
            if not docs:
                return f"No relevant information found in the document for '{query}'."
            return format_chunks_with_sources(docs)
        except Exception as e:
            return f"Error during document retrieval: {str(e)}"

    return [
        get_dataset_overview,
        get_revenue_analysis,
        get_top_products,
        get_monthly_revenue_trend,
        get_regional_analysis,
        retrieve_document_knowledge
    ]


def build_analyst_agent(
    df: Optional[pd.DataFrame] = None,
    retriever: Optional[Any] = None,
    api_key: Optional[str] = None,
    model_name: str = "gemini-1.5-flash"
):
    """
    Construct and compile the LangGraph workflow:
    START -> Agent Node -> Tools Node (conditional) -> Agent Node -> END
    """
    gemini_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        raise ValueError("Missing Gemini API Key. Provide it in .env or via the sidebar.")

    # 1. Initialize tools with active data references
    tools = create_analyst_tools(df, retriever)

    # 2. Initialize LLM with tool binding
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.0,
        google_api_key=gemini_key
    )
    llm_with_tools = llm.bind_tools(tools)

    # 3. Define Graph nodes
    def agent_node(state: MessagesState) -> Dict[str, Any]:
        messages = state["messages"]
        # Ensure system prompt is first message
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    # 4. Construct StateGraph
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    compiled_agent = workflow.compile()
    return compiled_agent
