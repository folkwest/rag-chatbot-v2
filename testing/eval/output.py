"""Output module for the chunking evaluation pipeline.

Provides CSV writing and stdout summary printing for evaluation results.
"""

import csv
import os
from collections import defaultdict

from testing.eval.models import ScoredResult


def write_csv(results: list[ScoredResult], output_path: str) -> None:
    """Write evaluation results to a CSV file.

    Columns: strategy, doc, question, hit_or_miss, retrieved_chunk_ids,
             num_chunks_for_doc, avg_chunk_size_tokens

    Creates the output directory if it doesn't exist.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fieldnames = [
        "strategy",
        "doc",
        "question",
        "hit_or_miss",
        "retrieved_chunk_ids",
        "num_chunks_for_doc",
        "avg_chunk_size_tokens",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            hit_or_miss_value = _format_hit_or_miss(result)
            retrieved_ids = ",".join(result.retrieved_chunk_ids)

            writer.writerow({
                "strategy": result.strategy,
                "doc": result.doc_id,
                "question": result.question_text,
                "hit_or_miss": hit_or_miss_value,
                "retrieved_chunk_ids": retrieved_ids,
                "num_chunks_for_doc": result.num_chunks_for_doc,
                "avg_chunk_size_tokens": result.avg_chunk_size_tokens,
            })


def print_summary(
    results: list[ScoredResult], integrity_flags: dict[str, int]
) -> None:
    """Print a readable summary table to stdout.

    Shows:
    - Hit rate per strategy per document (fraction and percentage)
    - Integrity flag count per strategy
    - Average chunk size (tokens) per strategy per document
    """
    # Group results by strategy and document
    strategy_doc_results: dict[str, dict[str, list[ScoredResult]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for result in results:
        strategy_doc_results[result.strategy][result.doc_id].append(result)

    strategies = sorted(strategy_doc_results.keys())

    # Print header
    print("=" * 70)
    print("CHUNKING EVALUATION SUMMARY")
    print("=" * 70)

    # Hit rate per strategy per document
    print("\nHit Rate (per strategy per document):")
    print("-" * 70)
    print(f"{'Strategy':<20} {'Document':<30} {'Hit Rate':<20}")
    print("-" * 70)

    for strategy in strategies:
        docs = sorted(strategy_doc_results[strategy].keys())
        for doc in docs:
            doc_results = strategy_doc_results[strategy][doc]
            scored = [
                r for r in doc_results if r.status == "scored"
            ]
            hits = sum(1 for r in scored if r.hit_or_miss is True)
            total = len(scored)
            if total > 0:
                pct = (hits / total) * 100
                hit_rate = f"{hits}/{total} ({pct:.1f}%)"
            else:
                hit_rate = "N/A (no scored results)"
            print(f"{strategy:<20} {doc:<30} {hit_rate:<20}")

    # Integrity flags per strategy
    print("\nIntegrity Flags (per strategy):")
    print("-" * 70)
    print(f"{'Strategy':<20} {'Flag Count':<20}")
    print("-" * 70)

    for strategy in strategies:
        count = integrity_flags.get(strategy, 0)
        print(f"{strategy:<20} {count:<20}")

    # Average chunk size per strategy per document
    print("\nAverage Chunk Size in Tokens (per strategy per document):")
    print("-" * 70)
    print(f"{'Strategy':<20} {'Document':<30} {'Avg Tokens':<20}")
    print("-" * 70)

    for strategy in strategies:
        docs = sorted(strategy_doc_results[strategy].keys())
        for doc in docs:
            doc_results = strategy_doc_results[strategy][doc]
            # Use the avg_chunk_size_tokens from any result for this strategy-doc pair
            # (they should all be the same since it's a per-doc metric)
            if doc_results:
                avg_size = doc_results[0].avg_chunk_size_tokens
                print(f"{strategy:<20} {doc:<30} {avg_size:<20.1f}")

    print("\n" + "=" * 70)


def _format_hit_or_miss(result: ScoredResult) -> str:
    """Format the hit_or_miss value for CSV output.

    Returns "hit" or "miss" for scored results, or the status string
    for non-scored results (unevaluated, error, no_chunks).
    """
    if result.status in ("unevaluated", "error", "no_chunks"):
        return result.status
    if result.hit_or_miss is True:
        return "hit"
    return "miss"
