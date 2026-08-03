"""Chunking strategies registry.

Exports a STRATEGIES dict mapping strategy names to chunker class instances
(using default configuration), and re-exports the BaseChunker interface.
"""

from testing.eval.chunkers.base import BaseChunker
from testing.eval.chunkers.fixed import FixedSizeChunker
from testing.eval.chunkers.recursive import RecursiveChunker
from testing.eval.chunkers.section_aware import SectionAwareChunker
from testing.eval.chunkers.semantic import SemanticChunker

STRATEGIES: dict[str, BaseChunker] = {
    "fixed_size": FixedSizeChunker(),
    "recursive": RecursiveChunker(),
    "semantic": SemanticChunker(),
    "section_aware": SectionAwareChunker(),
}

__all__ = [
    "BaseChunker",
    "FixedSizeChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "SectionAwareChunker",
    "STRATEGIES",
]
