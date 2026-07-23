# Production-Grade RAG & Safety Evaluation Pipeline

<p float="left">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-%2300BAFF?style=flat-square&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenAI-API-black?style=flat-square&logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-Persistent-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/FastAPI-000000?style=flat-square&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white" />
</p>

A **Retrieval-Augmented Generation (RAG) pipeline** with multi-strategy chunking,
persistent vector storage via ChromaDB, and strategy comparison. Upload documents
(PDF/TXT), select chunking strategies, and ask questions with full source
attribution and confidence scoring.

![Demo](assets/demo_gif_rag_chatbot.gif)

---

## Architecture

```
┌──────────────┐       ┌──────────────┐       ┌─────────────────────┐
│  Streamlit   │──────▶│   FastAPI    │──────▶│      ChromaDB       │
│  Frontend    │◀──────│   Backend    │◀──────│  (Docker, persistent)│
└──────────────┘       └──────────────┘       └─────────────────────┘
                              │
                              ▼
                       ┌──────────────┐
                       │  OpenAI API  │
                       │ (embed + LLM)│
                       └──────────────┘
```

**Vector collections** (one per chunking strategy):
- `chunks_fixed` — fixed character-length chunks
- `chunks_sentence` — sentence-boundary chunks
- `chunks_semantic` — embedding-similarity-based semantic chunks

---

## Features

- **Persistent vector storage** via ChromaDB (Docker) — data survives restarts
- **Per-strategy collections** — no cross-contamination between chunking methods
- Upload multiple documents with structured metadata payloads
- Three chunking strategies: `fixed`, `sentence`, `semantic`
- Strategy comparison mode (side-by-side answers)
- Confidence scoring and source attribution
- Health endpoint (`GET /health`) for monitoring
- Streamlit frontend with session state management

---

## Quick Start

### 1. Start ChromaDB

```bash
docker compose up -d
```

This launches a persistent ChromaDB instance on port `8100`.

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

Optional overrides:
```env
CHROMA_HOST=localhost
CHROMA_PORT=8100
```

### 4. Run the backend

```bash
uvicorn backend.main:app --reload
```

### 5. Run the frontend

```bash
streamlit run frontend/app.py
```

---

## How to Use

1. Upload a PDF or TXT document in the Streamlit UI.
2. Select one or more chunking strategies to compare.
3. Ask a question about the document.
4. View per-strategy answers, confidence scores, and source chunks.

---

## API Endpoints

| Method | Path      | Description                  |
|--------|-----------|------------------------------|
| POST   | `/upload` | Upload and chunk a document  |
| POST   | `/chat`   | Ask a question (RAG)         |
| GET    | `/health` | ChromaDB connectivity check  |

---

## Project Structure

```
backend/
├── api/            # FastAPI route handlers
├── chunking/       # Chunking strategies (fixed, sentence, semantic)
├── rag/            # RAG pipeline (retriever, generator)
├── storage/        # In-memory document registry
├── utils/          # Embedding & parsing utilities
├── vectorstore/    # ChromaDB persistent store
├── config.py       # Environment & model config
├── main.py         # FastAPI app entry point
└── schemas.py      # Pydantic models
frontend/
└── app.py          # Streamlit UI
docker-compose.yml  # ChromaDB container
```

---

## License

MIT License — feel free to use and modify.
