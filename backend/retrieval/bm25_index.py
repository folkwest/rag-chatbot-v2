"""BM25 sparse index for a single chunking strategy collection."""

from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

from backend.retrieval.tokenizer import tokenize


class BM25Index:
    """In-memory BM25 index for a single chunking strategy collection."""

    def __init__(self) -> None:
        self._documents: Dict[str, Dict[str, List[str]]] = {}
        self._corpus: List[List[str]] = []
        self._chunk_id_map: List[str] = []
        self._text_map: List[str] = []
        self._bm25: Optional[BM25Okapi] = None

    def add_documents(self, doc_id: str, chunks: List[str], chunk_ids: List[str]) -> None:
        """Add chunks for a document to the index. Incremental update."""
        if len(chunks) != len(chunk_ids):
            raise ValueError("chunks and chunk_ids must have the same length")

        self._documents[doc_id] = {
            "chunk_ids": list(chunk_ids),
            "texts": list(chunks),
        }
        self.rebuild()

    def remove_document(self, doc_id: str) -> None:
        """Remove all chunks belonging to a document from the index."""
        if doc_id in self._documents:
            del self._documents[doc_id]
            self.rebuild()

    def search(
        self,
        query: str,
        top_n: int = 20,
        doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the index and return ranked results.
        Returns: [{"chunk_id": str, "text": str, "score": float, "rank": int}, ...]
        """
        if self._bm25 is None or len(self._corpus) == 0:
            return []

        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)

        # Build candidate indices, optionally filtering by doc_id
        if doc_id is not None:
            # Get the chunk_ids belonging to the specified document
            if doc_id not in self._documents:
                return []
            valid_chunk_ids = set(self._documents[doc_id]["chunk_ids"])
            candidate_indices = [
                i for i, cid in enumerate(self._chunk_id_map)
                if cid in valid_chunk_ids
            ]
        else:
            candidate_indices = list(range(len(self._chunk_id_map)))

        # Filter to only positive scores and sort descending
        scored_candidates = [
            (i, scores[i]) for i in candidate_indices if scores[i] > 0
        ]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Limit to top_n
        scored_candidates = scored_candidates[:top_n]

        results = []
        for rank, (idx, score) in enumerate(scored_candidates, start=1):
            results.append({
                "chunk_id": self._chunk_id_map[idx],
                "text": self._text_map[idx],
                "score": float(score),
                "rank": rank,
            })

        return results

    def rebuild(self) -> None:
        """Rebuild the internal BM25Okapi instance from current document store."""
        self._corpus = []
        self._chunk_id_map = []
        self._text_map = []

        for doc_id, doc_data in self._documents.items():
            for chunk_id, text in zip(doc_data["chunk_ids"], doc_data["texts"]):
                self._chunk_id_map.append(chunk_id)
                self._text_map.append(text)
                self._corpus.append(tokenize(text))

        if self._corpus:
            self._bm25 = BM25Okapi(self._corpus)
        else:
            self._bm25 = None

    @property
    def document_count(self) -> int:
        """Return the number of documents in the index."""
        return len(self._documents)
