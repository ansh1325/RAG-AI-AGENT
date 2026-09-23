import logging
import os
import uuid
import base64
import tempfile
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
import inngest
import inngest.fast_api
from google import genai
from google.genai import types

from data_loader import load_and_chunk, embed_texts, EMBED_DIM
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGQueryResult, RAGSearchResult, RAGUpsertResult

is_production = bool(os.getenv("INNGEST_SIGNING_KEY") or os.getenv("INNGEST_EVENT_KEY")) and os.getenv("INNGEST_DEV", "0") != "1"

inngest_client = inngest.Inngest(
    app_id="rag-app",
    logger=logging.getLogger("uvicorn"),
    is_production=is_production,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: Ingest Pdf",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        if "pdf_base64" in ctx.event.data:
            pdf_bytes = base64.b64decode(ctx.event.data["pdf_base64"])
            source_id = ctx.event.data.get("source_id", "uploaded_doc.pdf")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name
            try:
                chunks = load_and_chunk(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            return RAGChunkAndSrc(chunks=chunks, source_id=source_id)
        else:
            pdf_path = ctx.event.data['pdf_path']
            source_id = ctx.event.data.get("source_id", pdf_path)
            chunks = load_and_chunk(pdf_path)
            return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, name=f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{'source': source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage(dim=EMBED_DIM).upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load_and_chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query Pdf",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5):
        query_vec = embed_texts([question])[0]
        store = QdrantStorage(dim=EMBED_DIM)
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found['sources'])

    question = ctx.event.data['question']
    top_k = int(ctx.event.data.get('top_k', 5))

    found = await ctx.step.run('embed_and_search', lambda: _search(question, top_k), output_type=RAGSearchResult)
    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    def _generate_answer(user_content: str) -> str:
        client = genai.Client()
        models = ['gemini-3.6-flash', 'gemini-3.5-flash', 'gemini-flash-latest', 'gemini-2.5-flash']
        for model_name in models:
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
                    return res.text.strip()
            except Exception as e:
                logging.warning(f"Model {model_name} failed: {e}. Trying next fallback...")
        return "Sorry, unable to generate an answer at this time."

    answer = await ctx.step.run('llm-answer', lambda: _generate_answer(user_content))
    return {'answer': answer, "sources": found.sources, 'num_contexts': len(found.contexts)}


app = FastAPI(title="RAG AI Agent API")

@app.get("/")
def health():
    return {"status": "ok", "message": "RAG AI Agent backend is running"}

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])