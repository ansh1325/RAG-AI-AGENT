# ⚡ Production-Grade PDF RAG Agent with Gemini & Inngest

[🎬 Watch the RAG Pipeline Demo Video on LinkedIn](https://lnkd.in/p/g2MV85E5)

A modern, Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **Inngest** (for asynchronous step-by-step workflow orchestration), **Qdrant** (for vector storage), and **Streamlit** (for the interactive frontend), powered by the **Google GenAI SDK** (`gemini-3.5-flash` and `gemini-embedding-2`).

---

## 🚀 Why This Project Stands Out 
Unlike typical, naive RAG implementations that process everything synchronously (blocking the user and easily failing on large documents), this system uses a **production-ready distributed workflow pattern**:
* **Resilient Event-Driven Execution**: Uses **Inngest** to decouple document ingestion and querying into background step-functions. If a step fails (e.g., API limit reached or network blip), Inngest automatically retries only the failed step with exponential backoff.
* **Modern Vector Search**: Leverages Qdrant's new, unified **Query API** (`query_points`) instead of legacy search endpoints, using state-of-the-art `gemini-embedding-2` to embed documents into 3,072-dimensional space.
* **Next-Gen Python Tooling**: Uses **`uv`**, the ultra-fast Python package installer and resolver, keeping dependencies locked and environment creation instantaneous.

---

This application is split into two asynchronous, event-driven pipelines managed by **Inngest**:

### 1. Document Ingestion Pipeline (`rag/ingest_pdf`)
* **Upload**: The user uploads a PDF in the Streamlit UI.
* **Extraction & Chunking**: The backend loads the PDF and splits the text into manageable chunks.
* **Vector Embeddings**: Text chunks are embedded into **3,072-dimensional** vectors using Google's `gemini-embedding-2` model.
* **Vector Storage**: Vectors are upserted into the **Qdrant Vector Database** along with their payload (source name and raw text).

### 2. Retrieval-Augmented Query Pipeline (`rag/query_pdf_ai`)
* **User Query**: The user asks a question via the Streamlit UI.
* **Query Embedding**: The question is embedded using the same `gemini-embedding-2` model.
* **Semantic Search**: The system queries Qdrant using the unified `query_points` API to retrieve the most relevant text chunks.
* **Response Generation**: The query and retrieved context chunks are wrapped in a grounded prompt and sent to `gemini-3.5-flash` to generate a concise, factual answer.
* **Real-time Polling**: Streamlit polls the Inngest Dev Server's `/runs/{id}` endpoint to seamlessly retrieve and display the final answer.

---

## 🧰 Tech Stack
* **Language & Package Management**: Python 3.12, [uv](https://github.com/astral-sh/uv)
* **Frontend**: Streamlit
* **Backend**: FastAPI
* **Workflow Orchestrator**: Inngest
* **Vector Database**: Qdrant (Dockerized)
* **AI & LLM Services**: Google GenAI SDK (`gemini-3.5-flash` for reasoning/generation, `gemini-embedding-2` for embeddings)

---

## 🏁 Getting Started Guide

Follow these steps to clone this repository and run the application locally on your machine.

### 📋 Prerequisites
Ensure you have the following installed:
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) (to run Qdrant)
* [Node.js v18+](https://nodejs.org/) (to run the Inngest Dev CLI)
* [uv](https://github.com/astral-sh/uv#installation) (for ultra-fast Python environment management)

---

### 🛠️ Step-by-Step Installation

#### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd RAG-AI-AGENT
```

#### 2. Configure Environment Variables
Copy the example environment file and add your Gemini API Key:
```bash
cp .env.example .env
```
Open `.env` and replace `your_gemini_api_key_here` with a valid key from [Google AI Studio](https://aistudio.google.com/).

#### 3. Run Qdrant Vector Database
Start a local Qdrant instance using Docker:
```bash
docker run -d -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage" --name qdrant-rag qdrant/qdrant
```

#### 4. Activate Virtual Environment & Install Dependencies
Using `uv`, set up your virtual environment and install packages in one step:
* **PowerShell**:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
  .venv\Scripts\Activate.ps1
  uv sync
  ```
* **Command Prompt (cmd)**:
  ```cmd
  .venv\Scripts\activate.bat
  uv sync
  ```
* **Linux/macOS**:
  ```bash
  source .venv/bin/activate
  uv sync
  ```

---

### 🚀 Running the Application

You will need **three** terminal windows active (with your virtual environment activated in each terminal).

#### **Terminal 1: Start the Inngest Dev Server**
Inngest orchestrates the steps and acts as the event dispatcher:
```bash
npx inngest-cli@latest dev
```
*Observability Dashboard will be available at:* `http://localhost:8288`

#### **Terminal 2: Start the FastAPI Backend**
Run the worker/backend API:
```bash
uvicorn main:app --reload
```
*FastAPI API documentation will be available at:* `http://localhost:8000/docs`

#### **Terminal 3: Start the Streamlit UI**
Launch the graphical interface:
```bash
streamlit run streamlit_app.py
```
*The web page will open automatically at:* `http://localhost:8501`

---

## 🔍 How to Test
1. Access the Streamlit UI.
2. Upload a sample PDF document. Wait for the page to show `Triggered ingestion for: <your_file>.pdf`.
3. Go to the **Functions** tab on your Inngest Dev Server (`http://localhost:8288`) to watch the ingestion step function (`rag_ingest_pdf`) load, chunk, embed, and upsert vectors in real time.
4. Go back to Streamlit, write a question about your PDF, and press **Ask**. You will see the response ground itself specifically using the text inside the PDF!
