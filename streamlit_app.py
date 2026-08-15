import asyncio
from pathlib import Path
import time

import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests

load_dotenv()

st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)


def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path


async def send_rag_ingest_event(pdf_path: Path) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )


async def send_rag_query_event(question: str, top_k: int) -> None:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )
    return result[0]


def _inngest_api_base() -> str:
    # Local dev server default; configurable via env
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


def fetch_run_details(run_id: str) -> dict:
    url = f"{_inngest_api_base()}/runs/{run_id}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", {})


def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            
            # Check for completed status and ensure output is populated
            if status in ("Completed", "Succeeded", "Success", "Finished", "COMPLETED", "SUCCEEDED", "SUCCESS", "FINISHED"):
                output = run.get("output")
                if output:
                    return output
                
                # Fallback to fetching details if not in list
                run_id = run.get("run_id") or run.get("id")
                try:
                    details = fetch_run_details(run_id)
                    output = details.get("output")
                    if output:
                        return output
                except Exception:
                    pass
                
                # If output is not yet populated, keep polling
            
            # Check for failure status
            if status in ("Failed", "Cancelled", "FAILED", "CANCELLED"):
                raise RuntimeError(f"Function run failed with status: {status}")
                
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for run output (last status: {last_status})")
        time.sleep(poll_interval_s)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* App styling */
    .main .block-container {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Header decoration */
    .gradient-text {
        background: linear-gradient(135deg, #a855f7 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    
    /* Status indicators */
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
    
    /* Answer box */
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
    
    /* Input fields and components styling */
    div[data-baseweb="input"] {
        border-radius: 8px !important;
    }
    
    /* Customize Streamlit Buttons */
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
st.caption("Powered by Inngest, Qdrant, & Google Gemini 3.5 Flash")

# Sidebar status display
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/database.png", width=100)
    st.markdown("### ⚙️ System Status")
    
    st.markdown("**Inngest Dev Server**")
    st.markdown('<span class="status-badge status-active">● Active (Port 8288)</span>', unsafe_allow_html=True)
    
    st.markdown("**Qdrant DB**")
    st.markdown('<span class="status-badge status-active">● Running (Port 6333)</span>', unsafe_allow_html=True)
    
    st.markdown("**Core Models**")
    st.info("LLM: gemini-3.5-flash\n\nEmbed: gemini-embedding-2")
    
    st.divider()
    st.caption("A modern asynchronous retrieval augmented generation portal.")

# Tabs configuration
tab_upload, tab_query = st.tabs(["📤 Ingest Document", "💬 Ask Agent"])

with tab_upload:
    st.markdown("### 📄 Document Ingestion Pipeline")
    st.write("Upload a PDF to split, embed, and index text chunks inside Qdrant dynamically.")
    uploaded = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    
    if uploaded:
        with st.spinner("Uploading and triggering ingestion..."):
            for file in uploaded:
                path = save_uploaded_pdf(file)
                asyncio.run(send_rag_ingest_event(path))
                st.success(f"🎉 Successfully triggered ingestion workflow for: **{path.name}**")
            time.sleep(0.3)
        st.info("The background worker is currently chunking and indexing your files. You can monitor the progress on the Inngest Dev Console.")

with tab_query:
    st.markdown("### 💬 Grounded Semantic Query")
    st.write("Ask questions based on your uploaded documents. Gemini will formulate a grounded answer using the retrieved context.")
    
    with st.form("rag_query_form"):
        question = st.text_input("Your question", placeholder="e.g., What is this document about?")
        top_k = st.number_input("How many chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
        submitted = st.form_submit_button("Ask Agent")
        
        if submitted and question.strip():
            with st.spinner("Retrieving context and generating answer..."):
                event_id = asyncio.run(send_rag_query_event(question.strip(), int(top_k)))
                output = wait_for_run_output(event_id)
                answer = output.get("answer", "")
                sources = output.get("sources", [])
            
            st.markdown("---")
            st.markdown("### 💡 Agent Response")
            st.markdown(f'<div class="answer-card">{answer or "(No answer)"}</div>', unsafe_allow_html=True)
            
            if sources:
                st.markdown("#### 📚 Referenced Sources")
                for s in sources:
                    st.markdown(f"- 📁 `{s}`")
