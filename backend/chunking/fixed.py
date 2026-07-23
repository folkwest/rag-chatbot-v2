from typing import List
from .base import Chunker


class FixedChunker(Chunker):
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        chunks = []
        step = max(1, self.chunk_size - self.overlap)
        for i in range(0, len(text), step):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks
