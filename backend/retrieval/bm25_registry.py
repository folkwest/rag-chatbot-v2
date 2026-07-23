"""Registry managing per-strategy BM25Index instances."""

from typing import Dict, List, Optional

from backend.retrieval.bm25_index import BM25Index


class BM25Registry:
    """Registry of BM25 indices, one per chunking strategy."""

    def __init__(self) -> None:
        self._indices: Dict[str, BM25Index] = {}

    def get_index(self, strategy: str) -> BM25Index:
        """Get or lazily create a BM25Index for the given strategy."""
        if strategy not in self._indices:
            self._indices[strategy] = BM25Index()
        return self._indices[strategy]

    def add_documents(
        self, strategy: str, doc_id: str, chunks: List[str], chunk_ids: List[str]
    ) -> None:
        """Add document chunks to the index for the given strategy."""
        index = self.get_index(strategy)
        index.add_documents(doc_id, chunks, chunk_ids)

    def remove_document(self, doc_id: str) -> None:
        """Remove a document from ALL strategy indices."""
        for index in self._indices.values():
            index.remove_document(doc_id)

    def search(
        self,
        strategy: str,
        query: str,
        top_n: int = 20,
        doc_id: Optional[str] = None,
    ) -> List[Dict]:
        """Search the BM25 index for the given strategy."""
        index = self.get_index(strategy)
        return index.search(query, top_n=top_n, doc_id=doc_id)


bm25_registry: BM25Registry = BM25Registry()
