"""Scoring module for the chunking evaluation pipeline.

Determines hit/miss by checking position overlap between retrieved chunks
and ground-truth locations. Also checks chunk integrity (whether chunks
split code fences or HTML tables).
"""

import re
from typing import Optional

from testing.eval.models import (
    Chunk,
    EvalQuestion,
    PositionMetadata,
    RetrievalResult,
    ScoredResult,
    count_tokens,
)


def score_hit(retrieved_chunks: list[Chunk], ground_truth: PositionMetadata) -> bool:
    """Determine if any retrieved chunk overlaps the ground-truth location.

    Returns True if at least one chunk's position_metadata_range overlaps
    with the ground_truth PositionMetadata.

    Overlap logic:
    - line_number ranges: overlap if chunk.line_number_end >= gt.line_number_start
      AND chunk.line_number_start <= gt.line_number_end
    - page_number ranges: overlap if chunk.page_number_end >= gt.page_number_start
      AND chunk.page_number_start <= gt.page_number_end
    - item_id: overlap if chunk.item_id == gt.item_id (exact match or contains)
    """
    for chunk in retrieved_chunks:
        chunk_range = chunk.position_metadata_range

        # Check line_number overlap
        if (
            ground_truth.line_number_start is not None
            and ground_truth.line_number_end is not None
            and chunk_range.line_number_start is not None
            and chunk_range.line_number_end is not None
        ):
            if (
                chunk_range.line_number_end >= ground_truth.line_number_start
                and chunk_range.line_number_start <= ground_truth.line_number_end
            ):
                return True

        # Check page_number overlap
        if (
            ground_truth.page_number_start is not None
            and ground_truth.page_number_end is not None
            and chunk_range.page_number_start is not None
            and chunk_range.page_number_end is not None
        ):
            if (
                chunk_range.page_number_end >= ground_truth.page_number_start
                and chunk_range.page_number_start <= ground_truth.page_number_end
            ):
                return True

        # Check item_id overlap (exact match or contains)
        if ground_truth.item_id is not None and chunk_range.item_id is not None:
            if (
                chunk_range.item_id == ground_truth.item_id
                or ground_truth.item_id in chunk_range.item_id
                or chunk_range.item_id in ground_truth.item_id
            ):
                return True

    return False


def check_integrity(chunk: Chunk, full_document_text: str) -> list[str]:
    """Check if a chunk starts or ends inside a code fence or HTML table.

    Uses the chunk's position metadata to determine approximate location,
    then checks whether the corresponding region in the full document text
    crosses code fence or table boundaries.

    For chunks produced by token-based splitters (where exact text matching
    fails due to whitespace normalization), falls back to checking whether
    a significant portion of the chunk's first/last tokens can be found
    in the document text within code fences or tables.

    Returns a list of flag strings (e.g., ["code_fence_split", "table_split"])
    or an empty list if the chunk is clean.
    """
    flags: list[str] = []

    # Strategy: find the chunk's approximate location by searching for
    # a short prefix/suffix of the chunk text (first/last ~50 chars)
    # This handles whitespace normalization better than full text match
    chunk_text = chunk.chunk_text
    
    # Try to find a unique prefix (first 80 non-whitespace chars worth of tokens)
    prefix_tokens = chunk_text.split()[:8]
    suffix_tokens = chunk_text.split()[-8:]
    
    prefix_snippet = prefix_tokens[0] if prefix_tokens else ""
    suffix_snippet = suffix_tokens[-1] if suffix_tokens else ""
    
    # Find approximate start position using the first token
    chunk_start = full_document_text.find(prefix_snippet)
    # Find approximate end position using the last token  
    chunk_end = full_document_text.rfind(suffix_snippet)
    
    if chunk_start == -1 or chunk_end == -1:
        # Can't locate chunk in document — skip integrity check
        return flags
    
    chunk_end += len(suffix_snippet)

    # Check for code fence split
    text_before = full_document_text[:chunk_start]
    text_after = full_document_text[chunk_end:]

    fence_count_before = len(re.findall(r"```", text_before))
    if fence_count_before % 2 == 1:
        # Odd number of fences before = chunk starts inside a code block
        flags.append("code_fence_split")
    else:
        # Check if chunk ends inside a code block
        chunk_region = full_document_text[chunk_start:chunk_end]
        fence_count_in_chunk = len(re.findall(r"```", chunk_region))
        total_before_and_in = fence_count_before + fence_count_in_chunk
        if total_before_and_in % 2 == 1:
            # Chunk ends inside a code block
            flags.append("code_fence_split")

    # Check for HTML table split
    table_opens_before = len(re.findall(r"<table", text_before, re.IGNORECASE))
    table_closes_before = len(re.findall(r"</table>", text_before, re.IGNORECASE))

    if table_opens_before > table_closes_before:
        # More opens than closes before = chunk starts inside a table
        flags.append("table_split")
    else:
        # Check if chunk ends inside a table
        chunk_region = full_document_text[chunk_start:chunk_end]
        table_opens_in_chunk = len(
            re.findall(r"<table", chunk_region, re.IGNORECASE)
        )
        table_closes_in_chunk = len(
            re.findall(r"</table>", chunk_region, re.IGNORECASE)
        )
        total_opens = table_opens_before + table_opens_in_chunk
        total_closes = table_closes_before + table_closes_in_chunk
        if total_opens > total_closes:
            # Chunk ends inside a table
            flags.append("table_split")

    return flags


def score_results(
    retrieval_results: list[RetrievalResult],
    questions: list[EvalQuestion],
    all_chunks_by_strategy: dict[str, list[Chunk]],
    document_texts: dict[str, str],
) -> list[ScoredResult]:
    """Score all retrieval results against ground-truth locations.

    For each RetrievalResult:
    - Determines hit/miss using score_hit
    - If ground_truth_location is None, sets status="unevaluated", hit_or_miss=None
    - Computes num_chunks_for_doc and avg_chunk_size_tokens from all_chunks_by_strategy
    - Checks integrity flags on retrieved chunks

    Args:
        retrieval_results: Results from the retrieval harness.
        questions: All evaluation questions (for ground-truth lookup).
        all_chunks_by_strategy: Mapping of strategy name to all chunks produced.
        document_texts: Mapping of document_id to full document text.

    Returns:
        List of ScoredResult objects with scoring outcomes.
    """
    # Build a lookup for questions by question_id
    question_lookup: dict[str, EvalQuestion] = {
        q.question_id: q for q in questions
    }

    scored_results: list[ScoredResult] = []

    for result in retrieval_results:
        question = question_lookup.get(result.question_id)
        ground_truth: Optional[PositionMetadata] = None
        if question is not None:
            ground_truth = question.ground_truth_location

        # Compute num_chunks_for_doc and avg_chunk_size_tokens
        strategy_chunks = all_chunks_by_strategy.get(result.strategy, [])
        doc_chunks = [c for c in strategy_chunks if c.doc_id == result.document_id]
        num_chunks_for_doc = len(doc_chunks)

        if num_chunks_for_doc > 0:
            avg_chunk_size_tokens = sum(
                c.token_count for c in doc_chunks
            ) / num_chunks_for_doc
        else:
            avg_chunk_size_tokens = 0.0

        # Check integrity flags on retrieved chunks
        doc_text = document_texts.get(result.document_id, "")
        for chunk in result.retrieved_chunks:
            if doc_text:
                integrity_flags = check_integrity(chunk, doc_text)
                # Update chunk's integrity_flags in place
                chunk.integrity_flags = integrity_flags

        # Determine hit/miss status
        if ground_truth is None:
            # No ground truth — unevaluated
            scored_results.append(
                ScoredResult(
                    strategy=result.strategy,
                    doc_id=result.document_id,
                    question_text=result.question_text,
                    hit_or_miss=None,
                    status="unevaluated",
                    retrieved_chunk_ids=[c.chunk_id for c in result.retrieved_chunks],
                    num_chunks_for_doc=num_chunks_for_doc,
                    avg_chunk_size_tokens=avg_chunk_size_tokens,
                )
            )
        elif num_chunks_for_doc == 0:
            # No chunks for this document
            scored_results.append(
                ScoredResult(
                    strategy=result.strategy,
                    doc_id=result.document_id,
                    question_text=result.question_text,
                    hit_or_miss=False,
                    status="no_chunks",
                    retrieved_chunk_ids=[],
                    num_chunks_for_doc=0,
                    avg_chunk_size_tokens=0.0,
                )
            )
        else:
            # Score the hit
            hit = score_hit(result.retrieved_chunks, ground_truth)
            scored_results.append(
                ScoredResult(
                    strategy=result.strategy,
                    doc_id=result.document_id,
                    question_text=result.question_text,
                    hit_or_miss=hit,
                    status="scored",
                    retrieved_chunk_ids=[c.chunk_id for c in result.retrieved_chunks],
                    num_chunks_for_doc=num_chunks_for_doc,
                    avg_chunk_size_tokens=avg_chunk_size_tokens,
                )
            )

    return scored_results
