import os
from google import genai
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "gemini-embedding-2"
EMBED_DIM = 3072
splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def get_genai_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)
    return genai.Client()

def load_and_chunk(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    client = get_genai_client()
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
    )
    return [item.values for item in response.embeddings]