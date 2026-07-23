"""
ChromaDB-backed persistent vector store with per-strategy collections.

Each chunking strategy gets its own collection to prevent cross-contamination
and enable clean side-by-side comparison.

Supports two modes:
  - "server" mode: connects to a remote/Docker ChromaDB instance via HTTP
  - "local" mode (default): uses a persistent local directory (no Docker needed)
"""

import logging
from typing import List, Optional, Dict

import chromadb
from chromadb.config import Settings

from backend.config import CHROMA_HOST, CHROMA_PORT, CHROMA_MODE, CHROMA_PERSIST_DIR
from backend.retrieval.bm25_registry import bm25_registry

logger = logging.getLogger(__name__)

# Supported strategies — each gets a dedicated collection
STRATEGIES = ["fixed", "sentence", "semantic"]


class ChromaStore:
    """Persistent vector store backed by ChromaDB with per-strategy collections."""

    def __init__(
        self,
        mode: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        persist_dir: Optional[str] = None,
    ):
        self.mode = mode or CHROMA_MODE
        self.host = host or CHROMA_HOST
        self.port = port or CHROMA_PORT
        self.persist_dir = persist_dir or CHROMA_PERSIST_DIR
        self._client = None
        self._collections: Dict[str, chromadb.Collection] = {}

    @property
    def client(self):
        if self._client is None:
            if self.mode == "server":
                self._client = chromadb.HttpClient(
                    host=self.host,
                    port=self.port,
                    settings=Settings(anonymized_telemetry=False),
                )
                logger.info(f"Connected to ChromaDB server at {self.host}:{self.port}")
            else:
                self._client = chromadb.PersistentClient(
                    path=self.persist_dir,
                    settings=Settings(anonymized_telemetry=False),
                )
                logger.info(f"Using local persistent ChromaDB at {self.persist_dir}")
        return self._client

    def _get_collection(self, strategy: str) -> chromadb.Collection:
        """Get or create a collection for the given chunking strategy."""
        if strategy not in self._collections:
            collection_name = f"chunks_{strategy}"
            self._collections[strategy] = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Collection '{collection_name}' ready")
        return self._collections[strategy]

    def add(
        self,
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[dict],
        strategy: str,
    ) -> None:
        """Add vectors to the strategy-specific collection."""
        collection = self._get_collection(strategy)

        # ChromaDB requires string IDs
        ids = [
            f"{meta['doc_id']}__{strategy}__{meta['chunk_id']}"
            for meta in metadatas
        ]

        # Batch insert (ChromaDB handles deduplication by ID)
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            batch_end = i + batch_size
            collection.add(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end],
                documents=texts[i:batch_end],
                metadatas=metadatas[i:batch_end],
            )

        logger.info(
            f"Added {len(texts)} vectors to collection 'chunks_{strategy}'"
        )

    def search(
        self,
        query_embedding: List[float],
        strategy: str,
        top_k: int = 5,
        doc_id: Optional[str] = None,
    ) -> List[dict]:
        """Search within a strategy-specific collection."""
        collection = self._get_collection(strategy)

        where_filter = None
        if doc_id:
            where_filter = {"doc_id": doc_id}

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
        )

        # Flatten ChromaDB's nested response format
        hits: List[dict] = []
        if results and results["ids"] and results["ids"][0]:
            for idx in range(len(results["ids"][0])):
                hits.append({
                    "text": results["documents"][0][idx],
                    "score": results["distances"][0][idx] if results["distances"] else 0.0,
                    "metadata": results["metadatas"][0][idx],
                })

        return hits

    def delete_document(self, doc_id: str) -> None:
        """Delete all chunks for a document across all strategy collections."""
        for strategy in STRATEGIES:
            collection = self._get_collection(strategy)
            collection.delete(where={"doc_id": doc_id})
        # Also remove from BM25 indices
        bm25_registry.remove_document(doc_id)
        logger.info(f"Deleted all chunks for doc_id={doc_id}")

    def count(self, strategy: str) -> int:
        """Return the number of vectors in a strategy collection."""
        return self._get_collection(strategy).count()

    def healthcheck(self) -> bool:
        """Verify ChromaDB connectivity."""
        try:
            self.client.heartbeat()
            return True
        except Exception:
            return False


# Module-level singleton
vector_store = ChromaStore()
