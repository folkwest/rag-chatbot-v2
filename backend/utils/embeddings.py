import logging
from typing import List

from openai import OpenAI
from backend.config import OPENAI_API_KEY, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts using OpenAI's embedding API."""
    if not texts:
        return []

    # OpenAI API has a limit on batch size; chunk if needed
    batch_size = 100
    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        all_embeddings.extend(e.embedding for e in response.data)

    return all_embeddings
