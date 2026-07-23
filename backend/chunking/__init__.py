from .fixed import FixedChunker
from .sentence import SentenceChunker
from .semantic import SemanticChunker


def get_chunker(strategy: str):
    if strategy == "sentence":
        return SentenceChunker()
    elif strategy == "semantic":
        return SemanticChunker()
    # default
    return FixedChunker()
