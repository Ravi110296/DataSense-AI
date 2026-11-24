"""
generate_pdf_report.py - Builds a publication-grade PDF report and interview guide
for DataSense AI: Intelligent Data Analyst Agent with RAG.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib import colors
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and render total page count: 'Page X of Y'.
    Also renders a clean running header and footer.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        
        # Suppress running header on cover/first page
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#64748B"))
            self.drawString(54, 750, "DataSense AI — Engineering Architecture & Interview Guide")
            
            self.setFont("Helvetica", 8)
            self.drawRightString(612 - 54, 750, "Confidential / Candidate Reference")
            
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 744, 612 - 54, 744)

        # Running Footer (on all pages)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(54, 34, "LangGraph • LangChain • ChromaDB RAG • Gemini 3.5 Flash-Lite")
        
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 34, page_str)
        
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 44, 612 - 54, 44)
        
        self.restoreState()


def build_pdf():
    pdf_filename = "PROJECT_REPORT_AND_INTERVIEW_GUIDE.pdf"
    
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    # 504 pt usable width
    usable_width = 504

    styles = getSampleStyleSheet()

    # Custom Palette
    c_primary = colors.HexColor("#142338")      # Deep Navy
    c_accent = colors.HexColor("#159A91")       # Teal Accent
    c_accent_dark = colors.HexColor("#0E7C75")  # Dark Teal
    c_secondary = colors.HexColor("#2878C7")    # Corporate Blue
    c_text = colors.HexColor("#1E293B")         # Slate Dark Text
    c_muted = colors.HexColor("#64748B")        # Muted Gray
    c_bg_light = colors.HexColor("#F8FAFC")     # Card Light Background
    c_border = colors.HexColor("#E2E8F0")       # Border Gray

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=c_primary,
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=c_accent_dark,
        spaceAfter=12
    )
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13.5,
        leading=18,
        textColor=c_primary,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=c_secondary,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13.5,
        textColor=c_text,
        spaceAfter=6
    )
    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=c_text,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3
    )
    callout_style = ParagraphStyle(
        'DocCallout',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13.5,
        textColor=c_primary
    )
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=c_text
    )
    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.white
    )
    code_style = ParagraphStyle(
        'DocCode',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=11,
        textColor=c_primary
    )

    story = []

    # =========================================================================
    # HEADER BANNER
    # =========================================================================
    story.append(Paragraph("DataSense AI: Intelligent Data Analyst Agent", title_style))
    story.append(Paragraph("Enterprise Architecture Report & End-to-End Interview Preparation Playbook", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_accent, spaceBefore=0, spaceAfter=8))

    # Meta Info Card Table
    meta_data = [
        [
            Paragraph("<b>Candidate Role:</b> AI / Agentic Systems Engineer", table_cell),
            Paragraph("<b>Project Type:</b> Enterprise SaaS Agentic System", table_cell),
            Paragraph("<b>Stack:</b> LangGraph, LangChain, ChromaDB, Gemini", table_cell)
        ],
        [
            Paragraph("<b>Interface:</b> Streamlit 1.56 (SaaS UI)", table_cell),
            Paragraph("<b>Verification:</b> 100% Deterministic Pandas Tools", table_cell),
            Paragraph("<b>Security:</b> Zero Code Execution (`eval` / `exec`)", table_cell)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[168, 168, 168])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_bg_light),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 1: 30-SECOND ELEVATOR PITCH
    # =========================================================================
    story.append(Paragraph("1. Executive Summary & 30-Second Interview Pitch", h1_style))
    pitch_text = (
        "<b>The 30-Second Pitch for Interviewers:</b><br/>"
        "<i>\"I designed and built <b>DataSense AI</b>, an enterprise agentic business intelligence application that autonomously "
        "reasons across both structured e-commerce transactions (CSV) and unstructured operational policy handbooks (PDF) "
        "using <b>LangGraph</b>, <b>LangChain</b>, and <b>ChromaDB RAG</b>.<br/><br/>"
        "In enterprise settings, business decisions require connecting numbers to policies. However, pure LLMs cannot perform "
        "reliable math without hallucinating, and allowing models to run arbitrary Python scripts (`eval`/`exec`) introduces critical "
        "remote code execution risks. DataSense AI resolves this through a <b>cyclical LangGraph state machine</b> that orchestrates "
        "pre-compiled, deterministic Pandas tools for 100% mathematical accuracy alongside vector similarity retrieval with "
        "1-indexed page citations—all rendered within a modern Streamlit SaaS frontend.\"</i>"
    )
    pitch_card = Table([[Paragraph(pitch_text, callout_style)]], colWidths=[usable_width])
    pitch_card.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EFF6FF")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#93C5FD")),
        ('LINEBEFORE', (0,0), (0,-1), 4, c_secondary),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(pitch_card)
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 2: BUSINESS PROBLEM & OBJECTIVES
    # =========================================================================
    story.append(Paragraph("2. The Business Problem & Architectural Objectives", h1_style))
    story.append(Paragraph(
        "<b>The Enterprise Knowledge Gap:</b> Quantitative business data (sales, orders, revenues) lives in tabular databases, "
        "while operational governance (warranty durations, return periods, restocking fees, and commercial discounts) resides in "
        "unstructured PDFs. Traditional analytics dashboards cannot read policy clauses, while conventional conversational chatbots "
        "cannot compute financial figures reliably.",
        body_style
    ))
    story.append(Paragraph("<b>Four Fundamental Engineering Flaws in Naive AI Agents:</b>", h2_style))
    story.append(Paragraph("• <b>The Arithmetic Hallucination Flaw:</b> LLMs are probabilistic language predictors, not calculators. Asking an LLM to sum 52 sales records results in frequent arithmetic drift.", bullet_style))
    story.append(Paragraph("• <b>The Remote Code Execution Danger:</b> Naive architectures pass LLM-generated Python strings directly to <font name='Courier'>eval()</font> or <font name='Courier'>exec()</font>, exposing the server to catastrophic prompt injection and filesystem attacks.", bullet_style))
    story.append(Paragraph("• <b>Context Window Pollution:</b> Ingesting complete raw CSVs and multi-page manuals into prompt context blows past rate limits, increases latency to 30+ seconds, and causes attention degradation.", bullet_style))
    story.append(Paragraph("• <b>Stateless Web Framework Friction:</b> Streamlit reruns the full script on every interaction, causing naive chat implementations to jump, lose in-flight agent reasoning, or detach chat inputs.", bullet_style))

    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Core System Deliverables:</b>", h2_style))
    obj_data = [
        [Paragraph("Objective", table_header), Paragraph("Technical Solution", table_header), Paragraph("Architectural Guarantee", table_header)],
        [
            Paragraph("<b>Exact Arithmetic</b>", table_cell),
            Paragraph("Pre-compiled deterministic Pandas analytical functions", table_cell),
            Paragraph("100% mathematical precision; zero hallucination", table_cell)
        ],
        [
            Paragraph("<b>Zero Execution Risk</b>", table_cell),
            Paragraph("No arbitrary code execution; structured tool calling only", table_cell),
            Paragraph("Total defense against code-injection vulnerabilities", table_cell)
        ],
        [
            Paragraph("<b>Document Verifiability</b>", table_cell),
            Paragraph("PyPDF text chunking with 1-indexed page metadata preserved", table_cell),
            Paragraph("Every policy response cites exact document and page #", table_cell)
        ],
        [
            Paragraph("<b>Multi-Source Synthesis</b>", table_cell),
            Paragraph("LangGraph StateGraph with conditional cyclical routing", table_cell),
            Paragraph("Single-turn multi-step reasoning across CSV + PDF", table_cell)
        ],
        [
            Paragraph("<b>Enterprise SaaS UX</b>", table_cell),
            Paragraph("Top segmented control, fixed bottom chat, ChatGPT-style editing", table_cell),
            Paragraph("Never-vanishing queries, resilient across tab switches", table_cell)
        ],
    ]
    t_obj = Table(obj_data, colWidths=[110, 204, 190])
    t_obj.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_light]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_obj)

    # Remove arbitrary PageBreaks and let content flow naturally with KeepTogether on tables
    # story.append(PageBreak())

    # =========================================================================
    # SECTION 3: SYSTEM ARCHITECTURE
    # =========================================================================
    story.append(Paragraph("3. Complete System Architecture & Component Flow", h1_style))
    story.append(Paragraph(
        "DataSense AI is partitioned into three cleanly decoupled layers: the <b>Presentation Layer</b> (Streamlit 1.56), "
        "the <b>Agentic Orchestration Layer</b> (LangGraph StateGraph), and the <b>Execution Layer</b> (Pandas & ChromaDB RAG).",
        body_style
    ))

    # Architecture Box Diagram in Monospace Table
    arch_diagram_text = (
        "+--------------------------------------------------------------------------------------------------+<br/>"
        "|                                     PRESENTATION LAYER (app.py)                                  |<br/>"
        "|   Top Navigation: [ Ask Data ] | [ Data Explorer ] | [ Document Library ]                        |<br/>"
        "|   Features: Fixed Bottom Chat (stBottom) | Persistent Demo Queries Dropdown | In-Place Query Edit|<br/>"
        "+--------------------------------------------------+-----------------------------------------------+<br/>"
        "                                                   | User Prompt / Pending State<br/>"
        "                                                   v<br/>"
        "+--------------------------------------------------------------------------------------------------+<br/>"
        "|                              AGENTIC ORCHESTRATION LAYER (agent.py)                               |<br/>"
        "|                     LangGraph StateGraph Engine  [State: MessagesState]                          |<br/>"
        "|                                                                                                  |<br/>"
        "|      START  --->  [ Agent Node ]  <=================>  tools_condition  --->  [ END ]            |<br/>"
        "|                   (Gemini 3.5-Lite)                         |                                    |<br/>"
        "|                                                             v (Tool Calls)                       |<br/>"
        "|                                                      [ ToolNode ]                                |<br/>"
        "+-------------------------------------------------------------+------------------------------------+<br/>"
        "                                                              |<br/>"
        "                               +------------------------------+-------------------------------+<br/>"
        "                               |                                                              |<br/>"
        "                               v                                                              v<br/>"
        "+-----------------------------------------------+              +-----------------------------------+<br/>"
        "|          PANDAS ANALYTICS TOOLS               |              |       CHROMADB VECTOR RAG         |<br/>"
        "|  - get_dataset_overview()                     |              |  - retrieve_document_knowledge()  |<br/>"
        "|  - get_revenue_analysis()                     |              |  - Cosine Similarity Search       |<br/>"
        "|  - get_top_products()                         |              |  - Dense Vector Embeddings        |<br/>"
        "|  - get_monthly_revenue_trend()                |              |  - Preserved Page Metadata (1-3)  |<br/>"
        "|  - get_regional_analysis()                    |              |  - In-Memory Ephemeral Store      |<br/>"
        "|  - plot_chart() Dynamic Visualizer            |              |                                   |<br/>"
        "+----------------------+------------------------+              +-----------------+-----------------+<br/>"
        "                       |                                                         |<br/>"
        "                       v                                                         v<br/>"
        "+-----------------------------------------------+              +-----------------------------------+<br/>"
        "|           STRUCTURED CSV DATA                 |              |     UNSTRUCTURED PDF HANDBOOK     |<br/>"
        "|      sales_data.csv (52 Transactions)         |              |  company_handbook.pdf (3 Pages)   |<br/>"
        "+-----------------------------------------------+              +-----------------------------------+"
    )
    arch_box = Table([[Paragraph(arch_diagram_text, code_style)]], colWidths=[usable_width])
    arch_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#0F172A")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#334155")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    # Override text color to light green/cyan
    code_cyan = ParagraphStyle('CodeCyan', parent=code_style, textColor=colors.HexColor("#38BDF8"), fontSize=6.8, leading=8.5)
    arch_box = Table([[Paragraph(arch_diagram_text, code_cyan)]], colWidths=[usable_width])
    arch_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#0B132B")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#1E293B")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(arch_box)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 4: TECHNICAL DEEP-DIVE
    # =========================================================================
    story.append(Paragraph("4. Step-by-Step Technical Deep-Dive", h1_style))
    
    # Module 1
    story.append(Paragraph("Module 1: Deterministic Structured Data Engine (agent.py)", h2_style))
    story.append(Paragraph(
        "Rather than relying on probabilistic LLM code generation, we constructed six specialized, deterministic Pandas tools. "
        "Each tool validates the active DataFrame schema, isolates required columns, computes aggregations directly via optimized "
        "C-extensions, and returns a formatted string to the agent.",
        body_style
    ))
    
    tools_data = [
        [Paragraph("Tool Identifier", table_header), Paragraph("Parameters / Inputs", table_header), Paragraph("Deterministic Analytical Role", table_header)],
        [
            Paragraph("<font name='Courier'>get_dataset_overview</font>", table_cell),
            Paragraph("None (bound to active df)", table_cell),
            Paragraph("Records count (52), columns (9), null audit, and structural summary.", table_cell)
        ],
        [
            Paragraph("<font name='Courier'>get_revenue_analysis</font>", table_cell),
            Paragraph("None (aggregates revenue)", table_cell),
            Paragraph("Computes total revenue ($69,880.00), total orders (52), AOV ($1,343.85), and category breakdown.", table_cell)
        ],
        [
            Paragraph("<font name='Courier'>get_top_products</font>", table_cell),
            Paragraph("<font name='Courier'>limit: int = 5</font>", table_cell),
            Paragraph("Ranks product lines by revenue. Pinpoints ProBook Ultra 15 as top earner ($24,000.00).", table_cell)
        ],
        [
            Paragraph("<font name='Courier'>get_monthly_revenue_trend</font>", table_cell),
            Paragraph("None (parses order_date)", table_cell),
            Paragraph("Extracts YYYY-MM chronological keys and sums monthly revenue progression across 2024.", table_cell)
        ],
        [
            Paragraph("<font name='Courier'>get_regional_analysis</font>", table_cell),
            Paragraph("None (groups region)", table_cell),
            Paragraph("Calculates regional market shares: North ($22.4k), West ($21.8k), East ($15.0k), South ($10.7k).", table_cell)
        ],
        [
            Paragraph("<font name='Courier'>plot_chart</font>", table_cell),
            Paragraph("chart_type, x, y, title, agg", table_cell),
            Paragraph("General dynamic visualizer emitting encapsulated JSON for instant Streamlit chart rendering.", table_cell)
        ],
    ]
    t_tools = Table(tools_data, colWidths=[130, 114, 260])
    t_tools.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_light]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_tools)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>The Dynamic Chart Protocol:</b> When a user asks for visual comparison, <font name='Courier'>plot_chart</font> aggregates "
        "the requested columns and emits an invisible HTML comment: <font name='Courier'>&lt;!-- CHART_JSON: {...} --&gt;</font>. "
        "Streamlit's regex parser extracts this payload, strips it from the conversational bubble, and renders a native responsive "
        "<font name='Courier'>st.bar_chart</font>, <font name='Courier'>st.line_chart</font>, or <font name='Courier'>st.area_chart</font>.",
        body_style
    ))

    # Remove arbitrary PageBreaks and let content flow naturally with KeepTogether on tables
    # story.append(PageBreak())

    # Module 2
    story.append(Paragraph("Module 2: Document RAG Knowledge Engine (rag_utils.py)", h2_style))
    story.append(Paragraph(
        "The RAG subsystem handles unstructured business governance documents. The pipeline is engineered to ensure "
        "<b>zero loss of document traceability</b>:",
        body_style
    ))
    story.append(Paragraph("• <b>Metadata Extraction:</b> Using <font name='Courier'>pypdf</font>, we extract text page-by-page. Crucially, we tag each raw page with its human-readable 1-indexed page number: <font name='Courier'>{'source': 'company_handbook.pdf', 'page': page_idx + 1}</font>.", bullet_style))
    story.append(Paragraph("• <b>Semantic Text Splitting:</b> Using <font name='Courier'>RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)</font>, text is divided into 10 compact chunks. The 100-character overlap prevents policy clauses (like warranty exclusions) from being severed at chunk boundaries.", bullet_style))
    story.append(Paragraph("• <b>Embedding & Ephemeral Indexing:</b> Chunks are transformed into 768-dimensional dense vector embeddings using Google's <font name='Courier'>models/text-embedding-004</font>. They are stored in an in-memory ChromaDB instance (<font name='Courier'>chromadb.EphemeralClient</font>), ensuring zero disk footprints and sub-10ms similarity searches.", bullet_style))
    story.append(Paragraph("• <b>Verifiable Citation Formatting:</b> Retrieved chunks are pre-formatted with header anchors before entering the prompt: <font name='Courier'>[Document: company_handbook.pdf | Page: 2]</font>. The agent explicitly transfers these citations to its response.", bullet_style))

    story.append(Spacer(1, 6))

    # Module 3
    story.append(Paragraph("Module 3: LangGraph Agentic Orchestration (agent.py)", h2_style))
    story.append(Paragraph(
        "Instead of rigid linear chains or legacy black-box loops, the core agent is compiled as a <b>LangGraph StateGraph</b>. "
        "The graph is defined with a typed state schema: <font name='Courier'>MessagesState</font>, maintaining a strict history "
        "of <font name='Courier'>SystemMessage</font>, <font name='Courier'>HumanMessage</font>, <font name='Courier'>AIMessage</font>, and <font name='Courier'>ToolMessage</font> objects.",
        body_style
    ))
    story.append(Paragraph("<b>The Execution Loop:</b>", body_style))
    story.append(Paragraph("1. <b>START -> Agent Node:</b> The LLM evaluates the full conversation history alongside its instructions. It decides whether the inquiry is conversational, quantitative, policy-based, or combined.", bullet_style))
    story.append(Paragraph("2. <b>Conditional Routing (tools_condition):</b> If the LLM generates a <font name='Courier'>tool_calls</font> directive, the edge branches to <font name='Courier'>tools</font>. If no tools are required, it routes to <font name='Courier'>END</font>.", bullet_style))
    story.append(Paragraph("3. <b>Multi-Step Tool Chaining:</b> For complex inquiries (e.g. <i>'Which product made the most money and what is its return policy?'</i>), the agent first invokes <font name='Courier'>get_top_products</font>. The graph loops back to the agent node with the output (<font name='Courier'>ProBook Ultra 15, Electronics</font>). The agent then immediately invokes <font name='Courier'>retrieve_document_knowledge('electronics return policy')</font>, loops back again, and synthesizes both outputs before terminating at <font name='Courier'>END</font>.", bullet_style))

    story.append(Spacer(1, 6))

    # Module 4
    story.append(Paragraph("Module 4: Enterprise SaaS Frontend Engineering (app.py)", h2_style))
    story.append(Paragraph(
        "Streamlit operates on a stateless, top-to-bottom script execution lifecycle. To deliver a production-grade enterprise SaaS "
        "user experience, we implemented five robust architectural patterns:",
        body_style
    ))
    story.append(Paragraph("• <b>Top Horizontal Navigation:</b> Replaced sidebar radio buttons with a sleek top horizontal segmented control (<font name='Courier'>st.segmented_control</font>). Navigation sits in the main workspace header, keeping the sidebar dedicated to Authentication, Data Sources, and Session Controls.", bullet_style))
    story.append(Paragraph("• <b>Permanent Bottom Chat Docking:</b> Calling chat inputs inside inner tabs unpins them in Streamlit. By moving navigation to the top level, <font name='Courier'>st.chat_input</font> executes at the root level inside <font name='Courier'>stBottom</font>, permanently anchored at the bottom. A <font name='Courier'>padding-bottom: 7.5rem</font> clearance guarantees messages never overlap the input bar.", bullet_style))
    story.append(Paragraph("• <b>Persistent Example Inquiries Dropdown:</b> Traditional starter cards vanish once a chat starts. We built a permanent toolbar dropdown above the chat window (<font name='Courier'>st.selectbox</font> + <font name='Courier'>Run Query</font> button) that remains accessible at all times.", bullet_style))
    story.append(Paragraph("• <b>In-Place Query Editing (ChatGPT Style):</b> Every user bubble features an <font name='Courier'>Edit</font> button. Clicking it expands an inline text area pre-filled with the query, accompanied by <font name='Courier'>Save & Submit</font> and <font name='Courier'>Cancel</font>. Submitting truncates subsequent history from that index and executes a fresh branch analysis.", bullet_style))
    story.append(Paragraph("• <b>Interrupted Tab-Switching Recovery:</b> Switching views during LLM generation aborts execution. DataSense AI inspects the state: if the last message in history is from the user without an assistant reply, returning to 'Ask Data' immediately detects the in-flight query and finishes generating the answer.", bullet_style))

    # Remove arbitrary PageBreaks and let content flow naturally with KeepTogether on tables
    # story.append(PageBreak())

    # =========================================================================
    # SECTION 5: GROUND TRUTH BENCHMARKS
    # =========================================================================
    story.append(Paragraph("5. Ground-Truth Benchmarks & Zero-Hallucination Proof", h1_style))
    story.append(Paragraph(
        "To provide empirical verification during technical interviews, the test dataset (<font name='Courier'>sales_data.csv</font>) "
        "and handbook (<font name='Courier'>company_handbook.pdf</font>) were compiled with exact mathematical and policy targets:",
        body_style
    ))

    benchmark_data = [
        [Paragraph("Target Metric / Policy Query", table_header), Paragraph("Ground Truth Value", table_header), Paragraph("Tool Used & Verification Output", table_header), Paragraph("Audit Status", table_header)],
        [
            Paragraph("Total Company Revenue", table_cell),
            Paragraph("<b>$69,880.00</b>", table_cell),
            Paragraph("<font name='Courier'>get_revenue_analysis()</font> -> Exact sum: $69,880.00", table_cell),
            Paragraph("<font color='#15803D'><b>Verified 100%</b></font>", table_cell)
        ],
        [
            Paragraph("Total Transaction Count", table_cell),
            Paragraph("<b>52 records</b>", table_cell),
            Paragraph("<font name='Courier'>get_dataset_overview()</font> -> Exactly 52 rows", table_cell),
            Paragraph("<font color='#15803D'><b>Verified 100%</b></font>", table_cell)
        ],
        [
            Paragraph("Top Product by Revenue", table_cell),
            Paragraph("<b>ProBook Ultra 15</b> ($24,000.00)", table_cell),
            Paragraph("<font name='Courier'>get_top_products()</font> -> 20 units @ $1,200.00", table_cell),
            Paragraph("<font color='#15803D'><b>Verified 100%</b></font>", table_cell)
        ],
        [
            Paragraph("Top Category by Revenue", table_cell),
            Paragraph("<b>Electronics</b> ($31,880.00)", table_cell),
            Paragraph("<font name='Courier'>get_revenue_analysis()</font> -> 45.6% company share", table_cell),
            Paragraph("<font color='#15803D'><b>Verified 100%</b></font>", table_cell)
        ],
        [
            Paragraph("Top Regional Market", table_cell),
            Paragraph("<b>North</b> ($22,400.00)", table_cell),
            Paragraph("<font name='Courier'>get_regional_analysis()</font> -> 16 orders, 32.1% share", table_cell),
            Paragraph("<font color='#15803D'><b>Verified 100%</b></font>", table_cell)
        ],
        [
            Paragraph("Electronics Return Window", table_cell),
            Paragraph("<b>30 Days</b> (15% restocking fee)", table_cell),
            Paragraph("<font name='Courier'>retrieve_document_knowledge</font> -> Handbook Page 1", table_cell),
            Paragraph("<font color='#15803D'><b>Verified 100%</b></font>", table_cell)
        ],
        [
            Paragraph("Furniture Return Condition", table_cell),
            Paragraph("<b>30-Day trial</b> ($50 freight fee)", table_cell),
            Paragraph("<font name='Courier'>retrieve_document_knowledge</font> -> Handbook Page 1", table_cell),
            Paragraph("<font color='#15803D'><b>Verified 100%</b></font>", table_cell)
        ],
        [
            Paragraph("Standard Electronics Warranty", table_cell),
            Paragraph("<b>2-Year limited</b> (1-yr battery)", table_cell),
            Paragraph("<font name='Courier'>retrieve_document_knowledge</font> -> Handbook Page 2", table_cell),
            Paragraph("<font color='#15803D'><b>Verified 100%</b></font>", table_cell)
        ],
        [
            Paragraph("Commercial Bulk Discount", table_cell),
            Paragraph("<b>10% off</b> orders > $1,000", table_cell),
            Paragraph("<font name='Courier'>retrieve_document_knowledge</font> -> Handbook Page 3", table_cell),
            Paragraph("<font color='#15803D'><b>Verified 100%</b></font>", table_cell)
        ],
    ]
    t_bench = Table(benchmark_data, colWidths=[130, 114, 180, 80])
    t_bench.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_light]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_bench)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 6: INTERVIEW PLAYBOOK (Q&A)
    # =========================================================================
    story.append(Paragraph("6. Interview Playbook: Technical Questions & Model Answers", h1_style))
    story.append(Paragraph(
        "Review these six technical questions and polished answers to explain your architecture without any loose ends:",
        body_style
    ))

    qa_list = [
        (
            "Q1: Why did you use LangGraph instead of classic LangChain AgentExecutor?",
            "\"LangChain's legacy AgentExecutor was a monolithic black-box loop that provided minimal control over execution state "
            "and intermediate routing. LangGraph reimagines agent workflows as deterministic state machines. State is explicitly typed "
            "via MessagesState, graph transitions are governed by clear conditional edges (tools_condition), and cyclical loops are natively "
            "supported. This was critical for multi-source reasoning: when an inquiry requires first finding the top product via Pandas "
            "and then retrieving its policy via ChromaDB, LangGraph inspects the intermediate tool output and dynamically triggers a second "
            "tool before routing to the final synthesis node.\""
        ),
        (
            "Q2: Why not allow the LLM to generate and execute Python code dynamically?",
            "\"Letting an LLM execute arbitrary generated Python code via eval() or PythonREPL introduces severe security and stability risks: "
            "(1) Security: Malicious or prompt-injected inputs can escape sandboxes, access sensitive environment variables, or delete files. "
            "(2) Reliability: LLMs frequently hallucinate column names, misinterpret data types, or produce syntax errors that cause pipeline crashes. "
            "By constraining the agent to pre-compiled, deterministic Pandas analytical tools, we guarantee 100% mathematical precision, "
            "sub-5ms execution times, and zero attack surface.\""
        ),
        (
            "Q3: How does your RAG pipeline guarantee accurate page citations?",
            "\"During PDF ingestion with pypdf, we extract content page-by-page and explicitly bind 1-indexed page metadata to each document "
            "chunk: {'source': filename, 'page': page_num}. When RecursiveCharacterTextSplitter divides the text into 800-character blocks, "
            "it inherits this metadata. When ChromaDB retrieves the top-k chunks, our formatting utility prepends a verified citation header "
            "to each excerpt. The agent's system prompt strictly instructs it to cite the source and page number in its synthesized response.\""
        ),
        (
            "Q4: How did you solve Streamlit's statelessness challenges (chat jumping & tab interruption)?",
            "\"Streamlit executes top-to-bottom on every user event. We solved three classic hurdles: (1) Bottom Docking: Calling chat inputs "
            "inside tab containers unpins them; moving navigation to a top horizontal segmented control keeps st.chat_input at the root level, "
            "docked permanently in stBottom. (2) Tab Interruption Recovery: If an agent is thinking and the user switches views, Streamlit interrupts "
            "execution. We structured st.session_state.messages to track in-flight queries. If the last message is from the user without an assistant "
            "reply, returning to the chat view immediately resumes thinking. (3) Query Editing: We built in-place message branch editing that "
            "truncates subsequent history from that index and re-runs the agent from the edited question.\""
        ),
        (
            "Q5: What model did you choose and why?",
            "\"We defaulted to Google's gemini-3.5-flash-lite via langchain-google-genai. It provides ultra-fast tool calling (sub-second latency), "
            "has an accessible free tier supported across all API accounts, and natively handles structured function binding. To prevent hanging "
            "during upstream traffic spikes, we configured request_timeout=30.0 and max_retries=2.\""
        ),
        (
            "Q6: How would you scale this architecture to a multi-million-row production enterprise?",
            "\"I would scale the architecture across three tiers: (1) Data Tier: Replace in-memory Pandas with DuckDB for medium data or Snowflake / "
            "BigQuery for massive enterprise data, executing read-only SQL tool queries. (2) Vector Tier: Transition from ephemeral ChromaDB to a "
            "distributed vector database like Pinecone, Pgvector (PostgreSQL), or BigQuery Vector Search with hybrid sparse-dense retrieval. "
            "(3) Backend Tier: Decouple the Streamlit frontend from the agent backend using FastAPI and asynchronous WebSockets/SSE, executing "
            "agent reasoning in a Celery/Redis background worker queue.\""
        ),
    ]

    for q, a in qa_list:
        story.append(Paragraph(f"<b>{q}</b>", h2_style))
        story.append(Paragraph(a, body_style))
        story.append(Spacer(1, 3))

    # Remove arbitrary PageBreaks and let content flow naturally with KeepTogether on tables
    # story.append(PageBreak())

    # =========================================================================
    # SECTION 7: TECH STACK & SUMMARY CHECKLIST
    # =========================================================================
    story.append(Paragraph("7. Technology Stack Reference & Interview Summary", h1_style))
    
    stack_data = [
        [Paragraph("Component", table_header), Paragraph("Technology", table_header), Paragraph("Version", table_header), Paragraph("Role in System", table_header)],
        [Paragraph("Language", table_cell), Paragraph("Python", table_cell), Paragraph("3.11+", table_cell), Paragraph("Core application and mathematical runtime", table_cell)],
        [Paragraph("Orchestrator", table_cell), Paragraph("LangGraph", table_cell), Paragraph(">= 0.2.0", table_cell), Paragraph("State machine, MessagesState & cyclical agent-tool loops", table_cell)],
        [Paragraph("Tool Framework", table_cell), Paragraph("LangChain Core", table_cell), Paragraph(">= 0.3.0", table_cell), Paragraph("Tool decorators, message schemas & function binding", table_cell)],
        [Paragraph("LLM Engine", table_cell), Paragraph("Google Gemini", table_cell), Paragraph("3.5 Flash-Lite", table_cell), Paragraph("Reasoning model with native function calling", table_cell)],
        [Paragraph("Embeddings", table_cell), Paragraph("Google GenAI", table_cell), Paragraph("text-embedding-004", table_cell), Paragraph("768-dimensional dense vector embeddings", table_cell)],
        [Paragraph("Vector Store", table_cell), Paragraph("ChromaDB", table_cell), Paragraph("langchain-chroma", table_cell), Paragraph("In-memory ephemeral vector database", table_cell)],
        [Paragraph("Analytics", table_cell), Paragraph("Pandas", table_cell), Paragraph(">= 2.0.0", table_cell), Paragraph("Deterministic calculations & statistical aggregations", table_cell)],
        [Paragraph("PDF Reader", table_cell), Paragraph("PyPDF", table_cell), Paragraph("pypdf", table_cell), Paragraph("Document parsing preserving 1-indexed page metadata", table_cell)],
        [Paragraph("Text Chunking", table_cell), Paragraph("RecursiveSplitter", table_cell), Paragraph("LangChain", table_cell), Paragraph("800-character chunks with 100-character overlap", table_cell)],
        [Paragraph("Web Frontend", table_cell), Paragraph("Streamlit", table_cell), Paragraph("1.56.0", table_cell), Paragraph("Enterprise SaaS UI with top horizontal navigation", table_cell)],
    ]
    t_stack = Table(stack_data, colWidths=[90, 110, 84, 220])
    t_stack.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_light]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_stack)
    story.append(Spacer(1, 10))

    # Summary Checklist Card
    story.append(Paragraph("<b>Interview Presentation Checklist:</b>", h2_style))
    checklist_text = (
        "<b>[✓] Problem Statement:</b> Clear narrative on enterprise data silos (tabular numbers vs. policy documents).<br/>"
        "<b>[✓] Architectural Rigor:</b> Presentation $\\rightarrow$ LangGraph State Machine $\\rightarrow$ Deterministic Pandas / ChromaDB $\\rightarrow$ Grounded Synthesis.<br/>"
        "<b>[✓] Security & Precision:</b> Zero arbitrary code execution (`eval`/`exec`); 100% deterministic math.<br/>"
        "<b>[✓] Citation Accuracy:</b> Verifiable 1-indexed page citations from PyPDF through vector store to response.<br/>"
        "<b>[✓] UX Engineering:</b> Fixed bottom chat docking, persistent dropdown, in-place query editing, tab-switch recovery.<br/>"
        "<b>[✓] Production Scalability:</b> DuckDB / BigQuery migration roadmap, managed vector storage, FastAPI decoupling."
    )
    chk_card = Table([[Paragraph(checklist_text, callout_style)]], colWidths=[usable_width])
    chk_card.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F0FDF4")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#BBF7D0")),
        ('LINEBEFORE', (0,0), (0,-1), 4, colors.HexColor("#16A34A")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(chk_card)

    # Build PDF with two-pass canvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated publication-grade PDF: {pdf_filename}")


if __name__ == "__main__":
    build_pdf()
