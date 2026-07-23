"""Hybrid retriever: BM25 + dense + RRF + cross-encoder reranking."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from backend.config import CANDIDATE_SET_SIZE, FINAL_CONTEXT_SIZE, RRF_CONSTANT
from backend.retrieval.bm25_registry import bm25_registry
from backend.retrieval.reranker import reranker
from backend.retrieval.rrf import rrf_combine

logger = logging.getLogger(__name__)


def _format_dense_results(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert ChromaDB search results into the format expected by rrf_combine.

    ChromaDB results have: {"text", "score", "metadata": {"doc_id", "chunk_id", ...}}
    RRF expects: {"chunk_id", "text", ...metadata}
    """
    formatted = []
    for hit in hits:
        metadata = hit.get("metadata", {})
        item: Dict[str, Any] = {
            "chunk_id": metadata.get("chunk_id", ""),
            "text": hit["text"],
            "score": hit.get("score", 0.0),
        }
        # Flatten metadata into the item
        for key, value in metadata.items():
            if key not in item:
                item[key] = value
        formatted.append(item)
    return formatted


def hybrid_retrieve(
    query: str,
    query_embedding: List[float],
    vector_store,
    strategy: str,
    doc_id: Optional[str] = None,
    candidate_set_size: int = CANDIDATE_SET_SIZE,
    final_context_size: int = FINAL_CONTEXT_SIZE,
    rrf_k: int = RRF_CONSTANT,
) -> List[Dict[str, Any]]:
    """
    Execute hybrid retrieval: BM25 + dense + RRF + cross-encoder reranking.

    Returns final_context_size chunks with reranker_score metadata.
    Falls back to RRF-only ranking if reranker fails.

    Args:
        query: The user's search query text.
        query_embedding: Precomputed dense embedding for the query.
        vector_store: ChromaStore instance for dense retrieval.
        strategy: Chunking strategy name (e.g., "fixed", "sentence", "semantic").
        doc_id: Optional document ID to filter results.
        candidate_set_size: Number of candidates to retrieve before reranking.
        final_context_size: Number of final chunks to return after reranking.
        rrf_k: RRF constant for rank fusion formula.

    Returns:
        List of chunk dicts with reranker_score metadata. Each dict contains:
        "text", "score", "reranker_score", "metadata" (with doc_id, filename, etc.)
    """
    # Execute BM25 and dense search concurrently (Requirement 7.1)
    bm25_results: List[Dict[str, Any]] = []
    dense_results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        bm25_future = executor.submit(
            bm25_registry.search,
            strategy,
            query,
            top_n=candidate_set_size,
            doc_id=doc_id,
        )
        dense_future = executor.submit(
            vector_store.search,
            query_embedding,
            strategy,
            top_k=candidate_set_size,
            doc_id=doc_id,
        )

        for future in as_completed([bm25_future, dense_future]):
            if future == bm25_future:
                bm25_results = future.result()
            else:
                dense_results = future.result()

    # Format dense results for RRF (needs chunk_id at top level)
    formatted_dense = _format_dense_results(dense_results)

    # Combine via Reciprocal Rank Fusion
    # Dense results first so their richer metadata (doc_id, filename, etc.) takes priority
    rrf_results = rrf_combine(
        [formatted_dense, bm25_results],
        k=rrf_k,
        top_n=candidate_set_size,
    )

    # Attempt cross-encoder reranking; fall back to RRF-only on failure
    try:
        reranked = reranker.rerank(query, rrf_results, top_k=final_context_size)
    except Exception as e:
        logger.warning(
            "Reranker failed, falling back to RRF-only ranking: %s", str(e)
        )
        # Fall back to top-K from RRF results with reranker_score=None
        reranked = []
        for item in rrf_results[:final_context_size]:
            result = dict(item)
            result["reranker_score"] = None
            reranked.append(result)

    return reranked
