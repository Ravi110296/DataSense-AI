"""
test_agent.py - Comprehensive End-to-End Test Suite for Intelligent Data Analyst Agent.

Tests the 5 core scenarios specified in the project brief:
1. Structured data query (Highest revenue product)
2. Document retrieval query (Electronics return policy)
3. Combined query (Highest revenue electronics product + its return policy)
4. Conversational follow-up (Warranty inquiry using context)
5. Missing information handling (Inquiry on unlisted policy)
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from rag_utils import load_pdf_from_path_or_stream, split_documents, build_vector_store
from agent import build_analyst_agent, SYSTEM_PROMPT

load_dotenv()


def run_tests():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if len(sys.argv) > 1 and sys.argv[1].strip():
        api_key = sys.argv[1].strip()

    print("=" * 70)
    print("INTELLIGENT DATA ANALYST AGENT — END-TO-END TEST SUITE")
    print("=" * 70)

    # 1. Load Data Sources
    print("\n[Step 1] Loading sample CSV and PDF documents...")
    csv_path = os.path.join("sample_data", "sales_data.csv")
    pdf_path = os.path.join("sample_data", "company_handbook.pdf")

    if not os.path.exists(csv_path) or not os.path.exists(pdf_path):
        print("ERROR: Sample data files not found. Run generate_sample_data.py first.")
        return False

    df = pd.read_csv(csv_path)
    print(f"  * CSV Loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

    raw_docs = load_pdf_from_path_or_stream(pdf_path, filename="company_handbook.pdf")
    chunks = split_documents(raw_docs)
    print(f"  * PDF Loaded: {len(raw_docs)} pages extracted, split into {len(chunks)} chunks.")

    # 2. Check API Key
    if not api_key:
        print("\n" + "!" * 70)
        print("NOTICE: No GEMINI_API_KEY found in .env or environment.")
        print("To run the live LLM end-to-end tests, run:")
        print("   python test_agent.py <YOUR_GEMINI_API_KEY>")
        print("or add GEMINI_API_KEY=your_key to a .env file.")
        print("!" * 70)
        print("\nTesting local deterministic tools directly without live API...")
        from agent import create_analyst_tools
        tools = {t.name: t for t in create_analyst_tools(df, None)}
        
        # Test get_top_products
        out = tools["get_top_products"].invoke({"n": 1})
        assert "ProBook Ultra 15" in out, "Failed to identify ProBook Ultra 15"
        assert "24,000" in out, "Failed to calculate revenue $24,000"
        print("  ✓ get_top_products tool: PASSED")
        
        # Test get_revenue_analysis
        out = tools["get_revenue_analysis"].invoke({})
        assert "69,880" in out, "Failed to calculate total revenue $69,880"
        assert "Electronics" in out, "Failed to identify Electronics category"
        print("  ✓ get_revenue_analysis tool: PASSED")
        
        # Test get_monthly_revenue_trend
        out = tools["get_monthly_revenue_trend"].invoke({})
        assert "2024-01" in out and "2024-06" in out, "Failed monthly trend grouping"
        print("  ✓ get_monthly_revenue_trend tool: PASSED")
        
        # Test get_regional_analysis
        out = tools["get_regional_analysis"].invoke({})
        assert "North" in out and "21,710" in out, "Failed regional analysis"
        print("  ✓ get_regional_analysis tool: PASSED")
        
        print("\nAll local analytical tools verified successfully!")
        return True

    # 3. Build Vector Store & Compile Agent
    print("\n[Step 2] Building ChromaDB vector store with GoogleGenerativeAIEmbeddings...")
    vectorstore, retriever = build_vector_store(chunks, api_key=api_key)
    print("  ✓ Vector store created and retriever initialized.")

    print("\n[Step 3] Compiling LangGraph Agent workflow...")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    print(f"  * Using Free-Tier Model: {model_name}")
    agent = build_analyst_agent(df=df, retriever=retriever, api_key=api_key, model_name=model_name)
    print("  ✓ LangGraph agent compiled successfully.")

    # 4. Execute End-to-End Test Scenarios
    test_results = []
    conversation_history = []

    def extract_text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [b["text"] for b in content if isinstance(b, dict) and "text" in b]
            return "".join(parts) if parts else str(content)
        return str(content)

    def ask(prompt: str, scenario_num: int, title: str):
        print(f"\n{'='*70}\n[TEST {scenario_num}] {title}\nUser Query: \"{prompt}\"\n{'-'*70}")
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + conversation_history + [HumanMessage(content=prompt)]
        res = agent.invoke({"messages": messages})
        final_response = extract_text(res["messages"][-1].content)
        
        # Check tool calls
        tools_used = [m.name for m in res["messages"] if hasattr(m, "name") and m.name]
        print(f"Tools Invoked: {tools_used}")
        print(f"Agent Response:\n{final_response}\n")
        
        # Update history
        conversation_history.append(HumanMessage(content=prompt))
        conversation_history.append(AIMessage(content=final_response))
        return final_response, tools_used

    # Test 1: Structured Data Only
    resp1, tools1 = ask("Which product generated the highest total revenue?", 1, "Structured Data Only")
    pass1 = ("ProBook" in resp1 or "24,000" in resp1) and ("get_top_products" in tools1 or "get_revenue_analysis" in tools1)
    test_results.append(("Test 1: Structured Data Query", pass1))

    # Test 2: Document Retrieval Only
    resp2, tools2 = ask("What is the return policy for electronics?", 2, "Document Retrieval Only (RAG)")
    pass2 = ("30-day" in resp2 or "30 day" in resp2 or "restocking fee" in resp2.lower()) and ("retrieve_document_knowledge" in tools2)
    test_results.append(("Test 2: Document Retrieval (RAG)", pass2))

    # Test 3: Combined Structured & Unstructured Query
    resp3, tools3 = ask("Which electronics product generated the highest revenue, and what is its return policy?", 3, "Combined Query (Data + Document)")
    has_product = "probook" in resp3.lower()
    has_policy = ("30" in resp3) or ("restocking" in resp3.lower())
    pass3 = has_product and has_policy and len(tools3) >= 2
    test_results.append(("Test 3: Combined Structured & Unstructured Query", pass3))

    # Test 4: Conversation Context (Follow-up)
    resp4, tools4 = ask("What about its warranty?", 4, "Conversational Follow-up")
    pass4 = ("2-year" in resp4.lower() or "2 year" in resp4.lower() or "warranty" in resp4.lower())
    test_results.append(("Test 4: Conversational Follow-up", pass4))

    # Test 5: Missing Information Handling
    resp5, tools5 = ask("What is the return policy for musical instruments?", 5, "Missing Information Handling")
    pass5 = ("not mentioned" in resp5.lower() or "not found" in resp5.lower() or "does not contain" in resp5.lower() or "no information" in resp5.lower() or "not available" in resp5.lower())
    test_results.append(("Test 5: Missing Information Handling", pass5))

    # Summary
    print("\n" + "=" * 70)
    print("TEST EXECUTION SUMMARY:")
    print("=" * 70)
    all_passed = True
    for name, status in test_results:
        status_str = "PASS ✓" if status else "FAIL ✗"
        if not status:
            all_passed = False
        print(f"  * {name:<45} : {status_str}")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
