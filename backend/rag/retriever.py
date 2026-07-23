from typing import List


def retrieve(vector_store, query_embedding: List[float], top_k: int = 5) -> List[dict]:
    """Retrieve top-k results from the vector store."""
    return vector_store.search(query_embedding, top_k)
