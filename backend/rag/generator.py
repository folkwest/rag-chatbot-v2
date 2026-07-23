import logging
from typing import List

from openai import OpenAI
from backend.config import OPENAI_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based only on the provided context. "
    "If the answer is not present in the context, say \"I don't know based on the provided documents.\""
)


def generate_answer(question: str, context_chunks: List[str]) -> str:
    """Generate an answer from retrieved context chunks."""
    if not context_chunks:
        return "I don't know based on the provided documents."

    context = "\n\n---\n\n".join(context_chunks)

    user_prompt = f"""Context:
{context}

Question: {question}

Answer the question using ONLY the context above. If the answer is not present, say "I don't know based on the provided documents."
"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise
