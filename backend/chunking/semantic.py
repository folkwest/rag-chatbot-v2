import logging
from typing import List

import numpy as np

from backend.utils.embeddings import embed_texts
from .base import Chunker

logger = logging.getLogger(__name__)


class SemanticChunker(Chunker):
    """Chunks text by detecting semantic breakpoints using embedding similarity."""

    def __init__(self, initial_chunk_size: int = 3, similarity_threshold: float = 0.78):
        self.initial_chunk_size = initial_chunk_size  # sentences per initial segment
        self.similarity_threshold = similarity_threshold

    def chunk(self, text: str) -> List[str]:
        # Split text into sentences first
        text = text.replace("\n", " ").replace("\r", " ").strip()
        sentences = [s.strip() for s in text.split(". ") if s.strip()]

        if len(sentences) <= self.initial_chunk_size:
            return [text] if text else []

        # Group sentences into initial small segments
        segments: List[str] = []
        for i in range(0, len(sentences), self.initial_chunk_size):
            segment = ". ".join(sentences[i:i + self.initial_chunk_size])
            if segment:
                segments.append(segment)

        if len(segments) <= 1:
            return segments

        # Embed all segments
        try:
            embeddings = embed_texts(segments)
        except Exception as e:
            logger.warning(f"Embedding failed, falling back to fixed chunking: {e}")
            # Fallback: return segments as-is
            return segments

        embeddings_np = np.array(embeddings)

        # Compute cosine similarity between consecutive segments
        chunks: List[str] = []
        current_chunk = segments[0]

        for i in range(1, len(segments)):
            sim = self._cosine_similarity(embeddings_np[i - 1], embeddings_np[i])

            if sim >= self.similarity_threshold:
                # Merge with current chunk (semantically similar)
                current_chunk += " " + segments[i]
            else:
                # Breakpoint detected — start a new chunk
                chunks.append(current_chunk.strip())
                current_chunk = segments[i]

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        logger.info(f"SemanticChunker produced {len(chunks)} chunks from {len(segments)} segments")
        return chunks

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
