"""Abstract base class for chunking strategies."""

from abc import ABC, abstractmethod

from testing.eval.models import Chunk, RawUnit


class BaseChunker(ABC):
    """Base interface for all chunking strategies.

    Subclasses must implement the `chunk` method to convert a list of
    RawUnits into positioned Chunks, and the `strategy_name` property
    to identify the strategy.
    """

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the name identifying this chunking strategy."""
        ...

    @abstractmethod
    def chunk(self, raw_units: list[RawUnit]) -> list[Chunk]:
        """Accept raw units and produce positioned chunks."""
        ...
