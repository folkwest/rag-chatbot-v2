import logging
from typing import List

import nltk
from .base import Chunker

logger = logging.getLogger(__name__)

# Download punkt tokenizer data (no-op if already present)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

from nltk.tokenize import sent_tokenize


class SentenceChunker(Chunker):
    def __init__(self, max_chars: int = 500):
        self.max_chars = max_chars

    def chunk(self, text: str) -> List[str]:
        sentences = sent_tokenize(text)
        chunks: List[str] = []
        current = ""

        for sent in sentences:
            if current and len(current) + len(sent) + 1 > self.max_chars:
                chunks.append(current.strip())
                current = sent
            else:
                current = f"{current} {sent}" if current else sent

        if current.strip():
            chunks.append(current.strip())

        return chunks
