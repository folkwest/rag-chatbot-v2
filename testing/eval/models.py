"""Data models for the chunking evaluation pipeline."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PositionMetadata:
    """Position reference within a source document.

    Uses optional fields to support different document types:
    - line_number_start/end for .md/.txt (1-based)
    - page_number_start/end for .pdf (1-based)
    - item_id for SEC HTML filings (e.g., "Item 1A")
    """

    line_number_start: Optional[int] = None
    line_number_end: Optional[int] = None
    page_number_start: Optional[int] = None
    page_number_end: Optional[int] = None
    item_id: Optional[str] = None


@dataclass
class RawUnit:
    """A pre-chunked text block from the ingestion module."""

    text: str
    doc_id: str
    position_metadata: PositionMetadata


@dataclass
class Chunk:
    """A text segment produced by a chunking strategy."""

    chunk_id: str
    chunk_text: str
    doc_id: str
    strategy: str
    position_metadata_range: PositionMetadata
    token_count: int
    integrity_flags: list[str] = field(default_factory=list)


@dataclass
class EvalQuestion:
    """An evaluation question with ground-truth location."""

    question_id: str
    question_text: str
    document_id: str
    ground_truth_location: Optional[PositionMetadata]


@dataclass
class RetrievalResult:
    """Result of retrieving chunks for one question under one strategy."""

    strategy: str
    document_id: str
    question_id: str
    question_text: str
    retrieved_chunks: list[Chunk]
    similarity_scores: list[float]


@dataclass
class ScoredResult:
    """Final scored result for one strategy-question pair."""

    strategy: str
    doc_id: str
    question_text: str
    hit_or_miss: Optional[bool]
    status: str
    retrieved_chunk_ids: list[str]
    num_chunks_for_doc: int
    avg_chunk_size_tokens: float
    error_message: Optional[str] = None


def count_tokens(text: str) -> int:
    """Count tokens in text using whitespace splitting.

    This provides a simple, predictable token count without requiring
    external tokenizer libraries.
    """
    return len(text.split())
