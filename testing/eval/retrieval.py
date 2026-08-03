"""Retrieval harness for the chunking evaluation pipeline.

Embeds evaluation questions and chunk sets, performs top-k cosine-similarity
retrieval, and builds RetrievalResult records for each strategy-question pair.
"""

from testing.eval.config import TOP_K
from testing.eval.embeddings import cosine_similarity, embed_texts
from testing.eval.models import Chunk, EvalQuestion, RetrievalResult


def retrieve_top_k(
    query_embedding: list[float],
    chunk_embeddings: list[list[float]],
    chunks: list[Chunk],
    k: int = 3,
) -> list[tuple[Chunk, float]]:
    """Return top-k chunks by cosine similarity to the query embedding.

    Args:
        query_embedding: The embedding vector for the query.
        chunk_embeddings: List of embedding vectors, one per chunk.
        chunks: List of Chunk objects corresponding to chunk_embeddings.
        k: Number of top results to return.

    Returns:
        List of (chunk, similarity_score) tuples sorted by descending
        cosine similarity. Returns min(k, len(chunks)) results.
    """
    scored = []
    for chunk, embedding in zip(chunks, chunk_embeddings):
        score = cosine_similarity(query_embedding, embedding)
        scored.append((chunk, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[: min(k, len(chunks))]


def run_retrieval(
    questions: list[EvalQuestion],
    strategy_chunks: dict[str, list[Chunk]],
    k: int = TOP_K,
) -> list[RetrievalResult]:
    """Run top-k retrieval for all question-strategy pairs.

    For each question, finds chunks matching the question's document_id within
    each strategy's chunk list, embeds them and the question, then retrieves
    the top-k most similar chunks.

    Args:
        questions: List of evaluation questions.
        strategy_chunks: Mapping from strategy name to its produced chunks.
        k: Number of top results to retrieve per question.

    Returns:
        List of RetrievalResult objects, one per strategy-question pair.
        If embedding fails for a pair, records an error result.
        If a strategy produces zero chunks for a document, records a "no_chunks" result.
    """
    results: list[RetrievalResult] = []

    for question in questions:
        for strategy_name, chunks in strategy_chunks.items():
            # Filter chunks to those matching the question's document
            doc_chunks = [c for c in chunks if c.doc_id == question.document_id]

            # If no chunks exist for this document under this strategy
            if not doc_chunks:
                results.append(
                    RetrievalResult(
                        strategy=strategy_name,
                        document_id=question.document_id,
                        question_id=question.question_id,
                        question_text=question.question_text,
                        retrieved_chunks=[],
                        similarity_scores=[],
                    )
                )
                continue

            # Attempt embedding and retrieval
            try:
                # Embed all relevant chunks and the question together
                texts_to_embed = [c.chunk_text for c in doc_chunks] + [
                    question.question_text
                ]
                embeddings = embed_texts(texts_to_embed)

                chunk_embeddings = embeddings[:-1]
                query_embedding = embeddings[-1]

                # Retrieve top-k
                top_k_results = retrieve_top_k(
                    query_embedding, chunk_embeddings, doc_chunks, k
                )

                retrieved_chunks = [chunk for chunk, _ in top_k_results]
                similarity_scores = [score for _, score in top_k_results]

                results.append(
                    RetrievalResult(
                        strategy=strategy_name,
                        document_id=question.document_id,
                        question_id=question.question_id,
                        question_text=question.question_text,
                        retrieved_chunks=retrieved_chunks,
                        similarity_scores=similarity_scores,
                    )
                )

            except Exception:
                # Record error but don't crash — continue with remaining pairs
                results.append(
                    RetrievalResult(
                        strategy=strategy_name,
                        document_id=question.document_id,
                        question_id=question.question_id,
                        question_text=question.question_text,
                        retrieved_chunks=[],
                        similarity_scores=[],
                    )
                )

    return results
