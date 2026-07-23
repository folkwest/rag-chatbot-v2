from pydantic import BaseModel
from typing import List, Optional, Union


class ChatRequest(BaseModel):
    question: str
    document_id: str
    chunking_strategy: Union[str, List[str]] = "fixed"
    retrieval_strategy: str = "vector"


class SourceChunk(BaseModel):
    text: str
    score: float
    doc_id: str
    filename: str
    chunking_strategy: str
    reranker_score: Optional[float] = None


class StrategyResult(BaseModel):
    strategy: str
    answer: str
    confidence: float
    sources: List[SourceChunk]


class ChatResponse(BaseModel):
    results: List[StrategyResult]
