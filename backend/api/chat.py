import logging

from fastapi import APIRouter, HTTPException
from backend.schemas import ChatRequest, ChatResponse
from backend.rag.pipeline import run_rag
from backend.vectorstore.chroma_store import vector_store
from backend.storage.document_store import doc_store

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Validate document exists
    if not doc_store.exists(req.document_id):
        raise HTTPException(status_code=404, detail="Document not found.")

    strategies = req.chunking_strategy
    if isinstance(strategies, str):
        strategies = [strategies]

    all_results = []

    for strat in strategies:
        answer, confidence, retrieved = run_rag(
            req.question,
            vector_store,
            req.document_id,
            chunking_strategy=strat,
            retrieval_strategy=req.retrieval_strategy,
        )

        sources = [
            {
                "text": r["text"],
                "score": r["score"],
                "doc_id": r["metadata"]["doc_id"],
                "filename": r["metadata"]["filename"],
                "chunking_strategy": r["metadata"]["chunking_strategy"],
                "reranker_score": r.get("reranker_score"),
            }
            for r in retrieved
        ]

        all_results.append({
            "strategy": strat,
            "answer": answer,
            "confidence": confidence,
            "sources": sources,
        })

    return {"results": all_results}
