"""
rag_utils.py - Retrieval-Augmented Generation (RAG) pipeline for PDF documents.

Handles:
- Loading and extracting text from PDF files with page-level metadata.
- Text chunking using RecursiveCharacterTextSplitter.
- Embedding generation using GoogleGenerativeAIEmbeddings.
- ChromaDB vector store creation and document retrieval.
- Formatted source citations.
"""

import os
import uuid
from typing import List, Tuple, Optional
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import chromadb


def load_pdf_from_path_or_stream(file_source, filename: str = "document.pdf") -> List[Document]:
    """
    Extract text from a PDF file path or file-like object (e.g. Streamlit UploadedFile).
    Preserves document name and 1-indexed page numbers in metadata.
    """
    reader = PdfReader(file_source)
    documents: List[Document] = []
    
    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": filename, "page": page_idx + 1}
                )
            )
            
    if not documents:
        raise ValueError(f"No readable text could be extracted from '{filename}'.")
        
    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 100
) -> List[Document]:
    """
    Split long document pages into smaller, cohesive chunks for vector search.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    return chunks


def build_vector_store(
    chunks: List[Document],
    api_key: Optional[str] = None,
    embedding_model: str = "models/gemini-embedding-001"
) -> Tuple[Chroma, any]:
    """
    Build an in-memory Chroma vector store populated with document chunks.
    Uses Google AI Studio free-tier compatible embeddings (models/gemini-embedding-001 with fallback).
    A fresh ephemeral collection is created to prevent stale results across uploads.
    """
    gemini_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        raise ValueError("Missing Gemini API Key. Please provide it in .env or via the sidebar.")
        
    # Attempt with specified model, with fallback to other free-tier embedding IDs
    candidate_models = [embedding_model, "models/gemini-embedding-2", "models/text-embedding-004"]
    embeddings = None
    last_err = None
    
    for model_candidate in candidate_models:
        try:
            candidate_emb = GoogleGenerativeAIEmbeddings(
                model=model_candidate,
                google_api_key=gemini_key
            )
            # Test a minimal dummy call to ensure the candidate model works with this key
            candidate_emb.embed_query("test")
            embeddings = candidate_emb
            break
        except Exception as e:
            last_err = e
            continue

    if embeddings is None:
        raise ValueError(f"Failed to initialize embedding model: {last_err}")
    
    # In-memory ephemeral client so no stale data persists on disk
    client = chromadb.Client()
    collection_name = f"rag_{uuid.uuid4().hex[:10]}"
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name=collection_name
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return vectorstore, retriever


def format_chunks_with_sources(docs: List[Document]) -> str:
    """
    Format retrieved chunks with clear source document and page number headers.
    """
    if not docs:
        return "No relevant information found in the document."
        
    formatted_chunks = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Document")
        page = doc.metadata.get("page", "Unknown")
        formatted_chunks.append(
            f"[Source {i} | File: {source} | Page: {page}]\n{doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(formatted_chunks)
