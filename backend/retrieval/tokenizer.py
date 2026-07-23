"""Tokenization utilities for BM25 retrieval."""

import re
from typing import List


def tokenize(text: str) -> List[str]:
    """Tokenize text for BM25: lowercase, split on non-alphanumeric."""
    return re.findall(r'\w+', text.lower())
