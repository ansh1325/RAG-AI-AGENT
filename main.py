import logging
from fastapi import FastAPI
from dotenv import load_dotenv
load_dotenv()
import inngest
import inngest.fast_api
from google import genai
from google.genai import types
import uuid
import os
import datetime
from data_loader import load_and_chunk,embed_texts,EMBED_DIM
from vector_db import QdrantStorage

from custom_types import RAGChunkAndSrc,RAGQueryResult,RAGSearchResult,RAGUpsertResult

inngest_client=inngest.Inngest(
    app_id="rag-app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)
@inngest_client.create_function(
    fn_id="RAG: Ingest Pdf",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx:inngest.Context):
    def _load(ctx:inngest.Context)-> RAGChunkAndSrc:
        pdf_path=ctx.event.data['pdf_path']
        source_id=ctx.event.data.get("source_id",pdf_path)
        chunks=load_and_chunk(pdf_path)
        return RAGChunkAndSrc(chunks=chunks,source_id=source_id)



    def _upsert(chunks_and_src:RAGChunkAndSrc)-> RAGUpsertResult:

        chunks=chunks_and_src.chunks
        source_id=chunks_and_src.source_id
        vecs=embed_texts(chunks)
        ids=[str(uuid.uuid5(uuid.NAMESPACE_URL,name=f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads=[{'source':source_id,"text":chunks[i]} for i in range(len(chunks))]
        QdrantStorage(dim=EMBED_DIM).upsert(ids,vecs,payloads)
        return RAGUpsertResult(ingested=len(chunks))


              
    chunks_and_src=await ctx.step.run("load_and_chunk", lambda: _load(ctx),output_type=RAGChunkAndSrc)
    ingested=await ctx.step.run("embed-and-upsert",lambda:_upsert(chunks_and_src),output_type=RAGUpsertResult)

    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query Pdf",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)

async def rag_query_pdf_ai(ctx:inngest.Context):
    def _search(question:str,top_k:int=5):
        query_vec=embed_texts([question])[0]
        store=QdrantStorage(dim=EMBED_DIM)
        found=store.search(query_vec,top_k)
        return RAGSearchResult(contexts=found["contexts"],sources=found['sources'])
    question=ctx.event.data['question']
    top_k=int(ctx.event.data.get('top_k',5))

    found=await ctx.step.run('embed_and_search',lambda:_search(question,top_k),output_type=RAGSearchResult)
    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
    "Use the following context to answer the question.\n\n"
    f"Context:\n{context_block}\n\n"
    f"Question: {question}\n"
    "Answer concisely using the context above.")

    def _generate_answer(user_content: str) -> str:
        client = genai.Client()
        res = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction="You are a Retrival augmented generation agent. You give answers based on the Context Provided to you,",
                temperature=0.2,
                max_output_tokens=1024,
            )
        )
        return res.text.strip() if res.text else ""

    answer = await ctx.step.run('llm-answer', lambda: _generate_answer(user_content))
    return {'answer':answer,"sources":found.sources,'num_contexts':len(found.contexts)}




app=FastAPI()
inngest.fast_api.serve(app,inngest_client,[rag_ingest_pdf,rag_query_pdf_ai])