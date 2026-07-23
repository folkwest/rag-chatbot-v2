import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import upload, chat
from backend.vectorstore.chroma_store import vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Chatbot", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(chat.router)


@app.get("/health")
def health():
    chroma_ok = vector_store.healthcheck()
    return {
        "status": "healthy" if chroma_ok else "degraded",
        "chromadb": "connected" if chroma_ok else "unreachable",
    }
