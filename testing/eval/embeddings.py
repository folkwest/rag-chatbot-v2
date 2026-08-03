"""Self-contained OpenAI embedding client for the evaluation pipeline.

Provides embedding generation via the OpenAI API and a pure-Python
cosine similarity function (no numpy dependency).
"""

import math

from testing.eval import config


def embed_texts(
    texts: list[str],
    model: str = None,
    api_key: str = None,
) -> list[list[float]]:
    """Embed texts in batches using the OpenAI API.

    Args:
        texts: List of text strings to embed.
        model: Embedding model name. Defaults to config.EMBEDDING_MODEL.
        api_key: OpenAI API key. Defaults to config.OPENAI_API_KEY.

    Returns:
        List of embedding vectors (one per input text).

    Raises:
        RuntimeError: If the OpenAI API call fails, with a descriptive message.
        ValueError: If no API key is available.
    """
    import openai

    if model is None:
        model = config.EMBEDDING_MODEL
    if api_key is None:
        api_key = config.OPENAI_API_KEY

    if not api_key:
        raise ValueError(
            "OpenAI API key is required. Set the OPENAI_API_KEY environment variable."
        )

    client = openai.OpenAI(api_key=api_key)

    batch_size = 100
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            response = client.embeddings.create(input=batch, model=model)
        except Exception as e:
            raise RuntimeError(
                f"OpenAI embedding API call failed for batch starting at index {i}: {e}"
            ) from e

        # Sort by index to ensure ordering matches input
        sorted_data = sorted(response.data, key=lambda x: x.index)
        all_embeddings.extend([item.embedding for item in sorted_data])

    return all_embeddings


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Uses only the math standard library module (no numpy dependency).

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        Cosine similarity value between -1.0 and 1.0.
        Returns 0.0 if either vector has zero norm.
    """
    dot_product = sum(x * y for x, y in zip(a, b))

    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)
