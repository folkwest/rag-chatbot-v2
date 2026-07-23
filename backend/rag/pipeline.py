import logging
from typing import List, Tuple

from backend.utils.embeddings import embed_texts
from backend.rag.generator import generate_answer
from backend.config import TOP_K
from backend.retrieval.hybrid_retriever import hybrid_retrieve

logger = logging.getLogger(__name__)


def run_rag(
    question: str,
    vector_store,
    document_id: str = None,
    chunking_strategy: str = "fixed",
    retrieval_strategy: str = "vector",
) -> Tuple[str, float, List[dict]]:
    """Run the RAG pipeline: embed query, retrieve from strategy collection, generate.

    Args:
        question: The user's question.
        vector_store: ChromaStore instance for vector retrieval.
        document_id: Optional document ID to filter results.
        chunking_strategy: Chunking strategy name (e.g., "fixed", "sentence", "semantic").
        retrieval_strategy: Retrieval strategy - "vector" for cosine-similarity only,
            "hybrid" for BM25 + dense + RRF + cross-encoder reranking.

    Returns:
        Tuple of (answer, confidence, source_chunks).
    """

    query_embedding = embed_texts([question])[0]

    if retrieval_strategy == "hybrid":
        retrieved = _hybrid_path(
            question=question,
            query_embedding=query_embedding,
            vector_store=vector_store,
            chunking_strategy=chunking_strategy,
            document_id=document_id,
        )
    else:
        # Default "vector" path — existing cosine-similarity retrieval unchanged
        retrieved = vector_store.search(
            query_embedding=query_embedding,
            strategy=chunking_strategy,
            top_k=TOP_K,
            doc_id=document_id,
        )

    if not retrieved:
        return "No relevant context found for this question.", 0.0, []

    context_chunks = [r["text"] for r in retrieved]
    answer = generate_answer(question, context_chunks)

    # Compute confidence score
    if retrieval_strategy == "hybrid":
        confidence = _compute_hybrid_confidence(retrieved)
    else:
        # Existing confidence: inverse distance, normalized
        # ChromaDB cosine distance is in [0, 2], so 1/(1+d) gives [0.33, 1.0]
        confidence = min(
            1.0,
            sum(1.0 / (1.0 + r["score"]) for r in retrieved) / len(retrieved),
        )

    return answer, confidence, retrieved


def _hybrid_path(
    question: str,
    query_embedding: List[float],
    vector_store,
    chunking_strategy: str,
    document_id: str = None,
) -> List[dict]:
    """Execute hybrid retrieval and format results to match the vector path contract.

    Returns list of dicts with: text, score, metadata, reranker_score.
    """
    results = hybrid_retrieve(
        query=question,
        query_embedding=query_embedding,
        vector_store=vector_store,
        strategy=chunking_strategy,
        doc_id=document_id,
    )

    # Format hybrid results into the same structure as the vector path
    formatted: List[dict] = []
    for item in results:
        # Build metadata dict from available fields
        metadata = item.get("metadata", {})
        # If metadata is not nested (hybrid_retriever flattens it), reconstruct
        if not metadata:
            metadata = {
                k: v
                for k, v in item.items()
                if k not in ("text", "score", "reranker_score", "rrf_score", "chunk_id")
            }

        # Ensure required metadata fields are present (BM25-only results may lack them)
        metadata.setdefault("doc_id", document_id or "")
        metadata.setdefault("chunking_strategy", chunking_strategy)
        metadata.setdefault("filename", metadata.get("source_filename", ""))

        formatted.append({
            "text": item.get("text", ""),
            "score": item.get("score", 0.0),
            "metadata": metadata,
            "reranker_score": item.get("reranker_score"),
        })

    return formatted


def _compute_hybrid_confidence(retrieved: List[dict]) -> float:
    """Compute confidence from reranker scores when available.

    Uses the average of reranker_scores if present, otherwise falls back
    to the standard inverse-distance method.
    """
    reranker_scores = [
        r["reranker_score"]
        for r in retrieved
        if r.get("reranker_score") is not None
    ]

    if reranker_scores:
        # Reranker scores are typically in [0, 1] range from sigmoid
        # Use average as confidence
        return min(1.0, sum(reranker_scores) / len(reranker_scores))

    # Fallback to inverse-distance method if no reranker scores available
    return min(
        1.0,
        sum(1.0 / (1.0 + r["score"]) for r in retrieved) / len(retrieved),
    )
