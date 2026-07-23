"""Reciprocal Rank Fusion (RRF) combiner for merging multiple ranked lists."""

from typing import Any, Dict, List


def rrf_combine(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = 60,
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    Each item in a ranked list must have a "chunk_id" key.
    Uses 1-based ranking: score = sum(1 / (k + rank_i)) across all lists
    a chunk appears in.

    Args:
        ranked_lists: List of ranked lists. Each list contains dicts with at
            minimum "chunk_id" and "text" keys, plus optional metadata.
        k: RRF constant (default 60). Higher values reduce the impact of
            high-ranking items.
        top_n: Maximum number of results to return.

    Returns:
        Combined list sorted by descending RRF score, limited to top_n.
        Each result includes: "chunk_id", "text", "rrf_score", plus any
        metadata from the original items.
    """
    # Map chunk_id -> accumulated score and metadata
    scores: Dict[str, float] = {}
    metadata: Dict[str, Dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        for rank_index, item in enumerate(ranked_list):
            chunk_id = item["chunk_id"]
            rank = rank_index + 1  # 1-based ranking
            rrf_score = 1.0 / (k + rank)

            scores[chunk_id] = scores.get(chunk_id, 0.0) + rrf_score

            # Store metadata from the first occurrence we see
            if chunk_id not in metadata:
                metadata[chunk_id] = {
                    key: value
                    for key, value in item.items()
                    if key != "chunk_id"
                }

    # Build result list with scores and metadata
    results = []
    for chunk_id, rrf_score in scores.items():
        result: Dict[str, Any] = {"chunk_id": chunk_id, "rrf_score": rrf_score}
        result.update(metadata[chunk_id])
        results.append(result)

    # Sort by descending RRF score
    results.sort(key=lambda x: x["rrf_score"], reverse=True)

    return results[:top_n]
