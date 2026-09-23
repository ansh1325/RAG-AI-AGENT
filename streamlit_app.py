import asyncio
import base64
from pathlib import Path
import time
import os
import requests

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Automatically bridge Streamlit secrets to environment variables
try:
    if hasattr(st, "secrets"):
        for key, val in st.secrets.items():
            if isinstance(val, str) and key not in os.environ:
                os.environ[key] = val
except Exception:
    pass

import inngest
from google import genai
from google.genai import types

from data_loader import embed_texts, EMBED_DIM
from vector_db import QdrantStorage

st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")


def get_gemini_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            pass
    return key or ""


def get_inngest_client() -> inngest.Inngest:
    is_prod = bool(os.getenv("INNGEST_SIGNING_KEY") or os.getenv("INNGEST_EVENT_KEY")) and os.getenv("INNGEST_DEV", "0") != "1"
    return inngest.Inngest(app_id="rag-app", is_production=is_prod)


async def send_rag_ingest_event(file_name: str, file_bytes: bytes) -> None:
    client = get_inngest_client()
    pdf_b64 = base64.b64encode(file_bytes).decode("utf-8")
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_base64": pdf_b64,
                "source_id": file_name,
            },
        )
    )


async def send_rag_query_event(question: str, top_k: int) -> None:
    try:
        client = get_inngest_client()
        await client.send(
            inngest.Event(
                name="rag/query_pdf_ai",
                data={
                    "question": question,
                    "top_k": top_k,
                },
            )
        )
    except Exception:
        pass


def execute_grounded_rag_query(question: str, top_k: int) -> dict:
    gemini_key = get_gemini_api_key()
    if not gemini_key:
        return {
            "answer": "⚠️ Error: GEMINI_API_KEY is not configured in Streamlit Secrets. Please add it to your app settings.",
            "sources": []
        }

    # Embed and search vector store
    query_vec = embed_texts([question])[0]
    store = QdrantStorage(dim=EMBED_DIM)
    found = store.search(query_vec, top_k)
    contexts = found.get("contexts", [])
    sources = found.get("sources", [])

    if not contexts:
        return {
            "answer": "No relevant context found in the uploaded documents. Please ingest PDF files first.",
            "sources": []
        }

    context_block = "\n\n".join(f"- {c}" for c in contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely and accurately using the context above."
    )

    client = genai.Client(api_key=gemini_key)
    models_to_try = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-flash-latest"]
    last_err = None

    for model_name in models_to_try:
        try:
            res = client.models.generate_content(
                model=model_name,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction="You are a Retrieval Augmented Generation agent. You give accurate answers based on the context provided to you.",
                    temperature=0.2,
                    max_output_tokens=1024,
                )
            )
            if res and res.text:
                return {"answer": res.text.strip(), "sources": sources}
        except Exception as e:
            last_err = e
            continue

    return {
        "answer": f"Unable to generate response. Error: {last_err}",
        "sources": sources
    }


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    .main .block-container {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .gradient-text {
        background: linear-gradient(135deg, #a855f7 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    
    .status-badge {
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 5px;
    }
    .status-active {
        background-color: rgba(34, 197, 94, 0.1);
        color: rgb(34, 197, 94);
        border: 1px solid rgba(34, 197, 94, 0.2);
    }
    
    .answer-card {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        font-size: 1.05rem;
        line-height: 1.6;
    }
    
    div[data-baseweb="input"] {
        border-radius: 8px !important;
    }
    
    button[kind="primaryFormSubmit"], button[kind="secondary"] {
        background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    button[kind="primaryFormSubmit"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="gradient-text">⚡ Resilient RAG Agent Portal</p>', unsafe_allow_html=True)
st.caption("Powered by Inngest, Qdrant, & Google Gemini 3.6 Flash")

# Sidebar status display
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/database.png", width=100)
    st.markdown("### ⚙️ System Status")
    
    is_cloud = bool(os.getenv("INNGEST_SIGNING_KEY"))
    inngest_status = "Cloud Active" if is_cloud else "Local (Port 8288)"
    qdrant_status = "Cloud Cluster" if os.getenv("QDRANT_API_KEY") else "Local (Port 6333)"
    
    st.markdown("**Inngest Engine**")
    st.markdown(f'<span class="status-badge status-active">● {inngest_status}</span>', unsafe_allow_html=True)
    
    st.markdown("**Qdrant Vector DB**")
    st.markdown(f'<span class="status-badge status-active">● {qdrant_status}</span>', unsafe_allow_html=True)
    
    st.markdown("**Core AI Models**")
    st.info("LLM: gemini-3.6-flash\n\nEmbed: gemini-embedding-2")
    
    st.divider()
    st.caption("Asynchronous, step-orchestrated RAG production architecture.")

# Tabs configuration
tab_upload, tab_query = st.tabs(["📤 Ingest Document", "💬 Ask Agent"])

with tab_upload:
    st.markdown("### 📄 Document Ingestion Pipeline")
    st.write("Upload a PDF to split, embed, and index text chunks inside Qdrant asynchronously.")
    uploaded = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    
    if uploaded:
        with st.spinner("Uploading and dispatching event to Inngest workflow..."):
            for file in uploaded:
                file_bytes = file.getvalue()
                asyncio.run(send_rag_ingest_event(file.name, file_bytes))
                st.success(f"🎉 Triggered background ingestion workflow for: **{file.name}**")
            time.sleep(0.3)
        st.info("Inngest is currently orchestrating the chunking, embedding, and vector upserts in background steps.")

with tab_query:
    st.markdown("### 💬 Grounded Semantic Query")
    st.write("Ask questions based on your indexed documents. Gemini will formulate a grounded answer using retrieved vector context.")
    
    with st.form("rag_query_form"):
        question = st.text_input("Your question", placeholder="e.g., What are the key takeaways from this document?")
        top_k = st.number_input("How many chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
        submitted = st.form_submit_button("Ask Agent")
        
        if submitted and question.strip():
            with st.spinner("Retrieving vector context and generating answer..."):
                # Asynchronously track event in Inngest Cloud
                asyncio.run(send_rag_query_event(question.strip(), int(top_k)))
                # Execute grounded response
                output = execute_grounded_rag_query(question.strip(), int(top_k))
                answer = output.get("answer", "")
                sources = output.get("sources", [])
            
            st.markdown("---")
            st.markdown("### 💡 Agent Response")
            st.markdown(f'<div class="answer-card">{answer or "(No answer received)"}</div>', unsafe_allow_html=True)
            
            if sources:
                st.markdown("#### 📚 Referenced Sources")
                for s in sources:
                    st.markdown(f"- 📁 `{s}`")
