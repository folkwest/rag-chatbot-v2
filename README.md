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

The project also includes a standalone **chunking evaluation pipeline** (`testing/eval/`)
that benchmarks four chunking strategies against a fixed test corpus with ground-truth
retrieval scoring — useful for empirically choosing the best chunking approach for
different document types.

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

**App vector collections** (one per chunking strategy, used by the chatbot):
- `chunks_fixed` — fixed character-length chunks
- `chunks_sentence` — sentence-boundary chunks
- `chunks_semantic` — embedding-similarity-based semantic chunks

---

## Features

- **Persistent vector storage** via ChromaDB (Docker) — data survives restarts
- **Per-strategy collections** — no cross-contamination between chunking methods
- Upload multiple documents with structured metadata payloads
- Three app-level chunking strategies: `fixed`, `sentence`, `semantic`
- Four eval-level chunking strategies: `fixed_size`, `recursive`, `semantic`, `section_aware`
- Strategy comparison mode (side-by-side answers)
- Confidence scoring and source attribution
- Health endpoint (`GET /health`) for monitoring
- Streamlit frontend with session state management
- Standalone chunking evaluation pipeline with position-based ground-truth scoring

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
testing/
├── eval/           # Chunking strategy evaluation pipeline
│   ├── chunkers/   # Eval-specific chunker implementations
│   ├── results/    # CSV output (gitignored)
│   ├── run_eval.py # Pipeline orchestrator
│   ├── ingest.py   # Document parsers with position metadata
│   ├── retrieval.py# Top-k cosine similarity retrieval
│   ├── scoring.py  # Hit/miss scoring + integrity flags
│   ├── output.py   # CSV + summary output
│   └── questions.json # Eval questions with ground-truth locations
├── openapi-callbacks.md
├── attention_is_all_you_need.pdf
├── pride_and_prejudice.txt
└── sec-filing.html
docker-compose.yml  # ChromaDB container
```

---

## Chunking Strategy Evaluation

A self-contained evaluation pipeline under `testing/eval/` that quantitatively compares chunking strategies by measuring retrieval accuracy against known ground-truth locations. This pipeline is independent of the main chatbot app — it uses its own chunker implementations and does not require the FastAPI backend or ChromaDB to be running.

The eval pipeline tests four strategies (fixed-size, recursive, semantic, section-aware) which overlap with but are not identical to the app's three strategies (fixed, sentence, semantic). The eval strategies are purpose-built for benchmarking with position-metadata tracking.

### What It Does

1. **Ingests** 4 test documents with position metadata (line numbers, page numbers, SEC Item IDs)
2. **Chunks** each document with 4 strategies (fixed-size, recursive, semantic, section-aware)
3. **Retrieves** top-3 chunks per eval question via cosine similarity (OpenAI embeddings)
4. **Scores** whether retrieved chunks overlap the ground-truth location
5. **Reports** hit rates, average chunk sizes, and integrity flags (code/table splits)

### Test Corpus

| Document | Type | Stresses |
|----------|------|----------|
| `openapi-callbacks.md` | Markdown | Headers + embedded code blocks |
| `attention_is_all_you_need.pdf` | PDF | Dense technical paper, equations |
| `pride_and_prejudice.txt` | Plain text | Long-form narrative, no headers |
| `sec-filing.html` | SEC 10-K/A (HTML) | Financial tables, Item sections |

### Running the Evaluation

```bash
pip install -r testing/eval/requirements.txt
python -m testing.eval
```

Requires `OPENAI_API_KEY` in your `.env` file (loaded automatically via python-dotenv).

### Sample Results

```
Hit Rate (per strategy per document):
----------------------------------------------------------------------
Strategy             Document                       Hit Rate
----------------------------------------------------------------------
fixed_size           attention_is_all_you_need.pdf  2/3 (66.7%)
fixed_size           openapi-callbacks.md           3/3 (100.0%)
fixed_size           pride_and_prejudice.txt        2/3 (66.7%)
fixed_size           sec-filing.html                3/3 (100.0%)
recursive            attention_is_all_you_need.pdf  2/3 (66.7%)
recursive            openapi-callbacks.md           3/3 (100.0%)
recursive            pride_and_prejudice.txt        2/3 (66.7%)
recursive            sec-filing.html                3/3 (100.0%)
section_aware        attention_is_all_you_need.pdf  2/3 (66.7%)
section_aware        openapi-callbacks.md           2/3 (66.7%)
section_aware        pride_and_prejudice.txt        1/3 (33.3%)
section_aware        sec-filing.html                3/3 (100.0%)
semantic             attention_is_all_you_need.pdf  3/3 (100.0%)
semantic             openapi-callbacks.md           2/3 (66.7%)
semantic             pride_and_prejudice.txt        1/3 (33.3%)
semantic             sec-filing.html                3/3 (100.0%)
```

### Key Findings

- **Fixed-size and recursive** perform consistently well (75% overall) — the recursive chunker's merge-after-split approach produces similarly-sized chunks while respecting structural boundaries.
- **Section-aware** underperforms on unstructured text (P&P at 33.3%) since it falls back to paragraph splitting when no headers exist. Its 100% on sec-filing.html is a "recall via brute size" artifact — a 3,217-token single-Item chunk trivially contains any answer.
- **Semantic** excels on the technical PDF (100%) where topic shifts align well with information boundaries, but struggles with narrative text where topic continuity is high.
- **Integrity flags**: fixed-size and semantic each produced 1 chunk that splits a code fence — the recursive and section-aware strategies avoid this by treating code blocks as atomic units.

### Caveats

- **n=3 questions per document** is not enough for statistical confidence. Each question represents 33.3 percentage points. Expand to 5–6 per doc before drawing strong conclusions.
- **Oversized chunks inflate hit rate.** Section-aware on sec-filing.html scores 100% because it produces one massive chunk per Item — not because its boundaries are meaningfully better.
- **Semantic threshold sensitivity.** The default threshold (0.4) can be tuned via `EVAL_SEMANTIC_THRESHOLD` env var. Higher values produce smaller chunks; lower values produce larger ones.

### Configuration

| Env Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | (required) | OpenAI API key for embeddings |
| `EVAL_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `EVAL_SEMANTIC_THRESHOLD` | `0.4` | Cosine similarity threshold for semantic chunker |
| `EVAL_TOP_K` | `3` | Number of chunks to retrieve per question |
| `EVAL_CHUNK_SIZE` | `500` | Default chunk size (tokens) for fixed/recursive |
| `EVAL_OVERLAP` | `50` | Token overlap for fixed-size chunker |

---

## License

MIT License — feel free to use and modify.
