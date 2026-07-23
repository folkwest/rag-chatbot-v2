"""Cross-encoder reranker with lazy model loading.

NOTE: The cross-encoder requires a compatible PyTorch installation.
If it cannot load (e.g., segfault on Python 3.13+), the hybrid retriever
automatically falls back to RRF-only ranking which still provides
significant improvement over vector-only search.

To enable reranking, ensure torch + sentence-transformers work on your system:
    python -c "from sentence_transformers import CrossEncoder; print('OK')"
"""

import logging
import math
import subprocess
import sys
import threading
from typing import Any, Dict, List, Optional

from backend.config import RERANKER_MODEL

logger = logging.getLogger(__name__)


def _check_torch_available() -> bool:
    """Check if torch/sentence-transformers can load without segfaulting.
    
    Runs a quick subprocess test to avoid crashing the main process.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", "from sentence_transformers import CrossEncoder; print('ok')"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0 and "ok" in result.stdout
    except Exception:
        return False


class CrossEncoderReranker:
    """Cross-encoder reranker with safe lazy model loading."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self._model_name = model_name or RERANKER_MODEL
        self._model = None
        self._available = None  # None = unchecked, True/False = checked
        self._ready_event = threading.Event()
        self._lock = threading.Lock()

    def _check_and_load(self) -> bool:
        """Check availability and load model if possible. Thread-safe."""
        with self._lock:
            if self._ready_event.is_set():
                return True
            if self._available is False:
                return False

            # First time: check if torch/sentence-transformers work
            if self._available is None:
                logger.info("Checking cross-encoder availability...")
                self._available = _check_torch_available()
                if not self._available:
                    logger.warning(
                        "Cross-encoder reranker unavailable (torch/sentence-transformers "
                        "incompatible with this Python version). "
                        "Hybrid retrieval will use RRF-only ranking."
                    )
                    return False

            # torch is available, try loading the model
            try:
                logger.info("Loading cross-encoder model: %s", self._model_name)
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self._model_name, device="cpu")
                self._ready_event.set()
                logger.info("Cross-encoder model loaded successfully")
                return True
            except Exception as e:
                logger.warning("Failed to load cross-encoder model: %s", str(e))
                self._available = False
                return False

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Score and rerank candidates.
        Returns top-k candidates sorted by descending cross-encoder score.
        Each result includes original metadata plus "reranker_score" field.
        Raises RuntimeError if model is not available (caller handles fallback).
        """
        if not self._ready_event.is_set():
            if not self._check_and_load():
                raise RuntimeError(
                    "Cross-encoder reranker not available on this system"
                )

        if not candidates:
            return []

        # Truncate text to avoid exceeding model's max token length (512 tokens ≈ 2000 chars)
        max_chars = 2000
        pairs = [(query[:500], candidate["text"][:max_chars]) for candidate in candidates]

        # Get scores from the cross-encoder
        scores = self._model.predict(pairs)

        # Combine candidates with their scores, clamping to avoid inf/-inf
        scored_candidates = []
        for candidate, score in zip(candidates, scores):
            s = float(score)
            if math.isinf(s) or math.isnan(s):
                s = 0.0
            scored_candidates.append((candidate, s))

        # Sort by descending score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Return top_k results with reranker_score field added
        results = []
        for candidate, score in scored_candidates[:top_k]:
            result = dict(candidate)
            result["reranker_score"] = float(score)
            results.append(result)

        return results

    @property
    def is_loaded(self) -> bool:
        """Return whether the model is loaded and ready."""
        return self._ready_event.is_set()


# Module-level singleton
reranker = CrossEncoderReranker()
