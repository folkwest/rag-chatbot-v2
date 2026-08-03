"""Analyze retrieval misses — shows question, ground-truth, and actual retrieved chunks side-by-side.

Usage: python testing/eval/analyze_misses.py [--strategy STRATEGY] [--doc DOCUMENT]

Filters are optional. Without them, all misses are shown.
"""

import argparse
import csv
import json
import os
import sys
import textwrap

# Paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(_SCRIPT_DIR, "results", "eval_results.csv")
QUESTIONS_FILE = os.path.join(_SCRIPT_DIR, "questions.json")


def load_questions() -> dict[str, dict]:
    """Load questions keyed by question_text for lookup."""
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {q["question_text"]: q for q in data}


def load_results() -> list[dict]:
    """Load the CSV results."""
    with open(RESULTS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def build_chunk_index() -> dict[str, str]:
    """Rebuild all chunks and return a dict mapping chunk_id -> chunk_text.

    This re-runs ingestion and chunking to get the actual text content.
    """
    # Ensure project root is on sys.path for imports
    project_root = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from testing.eval.ingest import ingest_document
    from testing.eval.chunkers import STRATEGIES

    corpus_dir = os.path.join(_SCRIPT_DIR, "..")
    docs = [
        "openapi-callbacks.md",
        "attention_is_all_you_need.pdf",
        "pride_and_prejudice.txt",
        "sec-filing.html",
    ]

    chunk_index: dict[str, str] = {}

    for doc_filename in docs:
        file_path = os.path.join(corpus_dir, doc_filename)
        raw_units = ingest_document(file_path)

        for strategy_name, strategy in STRATEGIES.items():
            try:
                chunks = strategy.chunk(raw_units)
                for chunk in chunks:
                    chunk_index[chunk.chunk_id] = chunk.chunk_text
            except Exception as e:
                print(f"  Warning: {strategy_name} on {doc_filename} failed: {e}")

    return chunk_index


def format_ground_truth(gt: dict | None) -> str:
    """Format ground-truth location for display."""
    if gt is None:
        return "null (unevaluated)"
    parts = []
    if gt.get("line_number_start"):
        parts.append(f"lines {gt['line_number_start']}–{gt['line_number_end']}")
    if gt.get("page_number_start"):
        parts.append(f"pages {gt['page_number_start']}–{gt['page_number_end']}")
    if gt.get("item_id"):
        parts.append(f"item_id: {gt['item_id']}")
    return ", ".join(parts) if parts else "unknown"


def get_source_context(doc_id: str, gt: dict) -> str:
    """Extract the actual ground-truth text from the source document."""
    corpus_dir = os.path.join(_SCRIPT_DIR, "..")

    if gt.get("line_number_start"):
        file_path = os.path.join(corpus_dir, doc_id)
        if not os.path.exists(file_path):
            return "(source file not found)"
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        start = gt["line_number_start"] - 1  # 0-indexed
        end = gt["line_number_end"]
        excerpt = "".join(lines[start:end])
        if len(excerpt) > 600:
            excerpt = excerpt[:300] + "\n  [...]\n" + excerpt[-300:]
        return excerpt.strip()

    elif gt.get("page_number_start"):
        try:
            import fitz
        except ImportError:
            return "(pymupdf not available)"
        file_path = os.path.join(corpus_dir, doc_id)
        if not os.path.exists(file_path):
            return "(source file not found)"
        doc = fitz.open(file_path)
        pages = range(gt["page_number_start"] - 1, gt["page_number_end"])
        text = ""
        for p in pages:
            text += doc[p].get_text()
        doc.close()
        if len(text) > 600:
            text = text[:300] + "\n  [...]\n" + text[-300:]
        return text.strip()

    elif gt.get("item_id"):
        return f"(entire {gt['item_id']} section)"

    return "(unknown format)"


def truncate(text: str, max_chars: int = 300) -> str:
    """Truncate text to max_chars, adding ellipsis if needed."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def main():
    parser = argparse.ArgumentParser(description="Analyze retrieval misses")
    parser.add_argument("--strategy", "-s", help="Filter by strategy name")
    parser.add_argument("--doc", "-d", help="Filter by document name")
    parser.add_argument("--no-chunks", action="store_true",
                        help="Skip rebuilding chunks (faster, shows IDs only)")
    args = parser.parse_args()

    if not os.path.exists(RESULTS_CSV):
        print(f"Error: Results file not found at {RESULTS_CSV}")
        print("Run the eval pipeline first: python -m testing.eval")
        sys.exit(1)

    questions = load_questions()
    results = load_results()

    # Filter to misses only
    misses = [r for r in results if r["hit_or_miss"] == "miss"]

    if args.strategy:
        misses = [r for r in misses if r["strategy"] == args.strategy]
    if args.doc:
        misses = [r for r in misses if args.doc in r["doc"]]

    if not misses:
        print("No misses found with the given filters.")
        return

    # Build chunk text index (skip if --no-chunks)
    chunk_index: dict[str, str] = {}
    if not args.no_chunks:
        print("Rebuilding chunks to get text content (use --no-chunks to skip)...")
        chunk_index = build_chunk_index()

    print(f"\n{'='*80}")
    print(f"RETRIEVAL MISS ANALYSIS — {len(misses)} misses")
    print(f"{'='*80}")

    for i, miss in enumerate(misses, 1):
        strategy = miss["strategy"]
        doc = miss["doc"]
        question_text = miss["question"]
        chunk_ids_str = miss["retrieved_chunk_ids"]
        chunk_ids = [cid.strip() for cid in chunk_ids_str.split(",") if cid.strip()]

        # Look up ground truth
        q_data = questions.get(question_text, {})
        gt = q_data.get("ground_truth_location")

        print(f"\n{'─'*80}")
        print(f"Miss #{i}")
        print(f"{'─'*80}")
        print(f"  Strategy:     {strategy}")
        print(f"  Document:     {doc}")
        print(f"  Question:     {question_text}")
        print(f"  Ground truth: {format_ground_truth(gt)}")
        print(f"  Retrieved:    {chunk_ids_str}")

        if gt:
            print(f"\n  ┌─ EXPECTED (ground-truth excerpt):")
            context = get_source_context(doc, gt)
            for line in textwrap.wrap(context, width=74):
                print(f"  │ {line}")
            print(f"  └─")

        # Show actual retrieved chunk text
        print(f"\n  ┌─ ACTUALLY RETRIEVED:")
        if chunk_index:
            for j, cid in enumerate(chunk_ids, 1):
                text = chunk_index.get(cid)
                if text:
                    print(f"  │ [{j}] {cid}")
                    print(f"  │     {truncate(text, 250)}")
                    print(f"  │")
                else:
                    print(f"  │ [{j}] {cid} — (text not available, likely semantic strategy)")
        else:
            print(f"  │ {chunk_ids_str}")
            print(f"  │ (use without --no-chunks to see actual text)")
        print(f"  └─")

    # Summary
    print(f"\n{'='*80}")
    print("MISS SUMMARY BY STRATEGY + DOCUMENT")
    print(f"{'='*80}")

    from collections import Counter
    miss_counts = Counter((m["strategy"], m["doc"]) for m in misses)
    for (strategy, doc), count in sorted(miss_counts.items()):
        print(f"  {strategy:20s} {doc:35s} {count} miss(es)")

    print()


if __name__ == "__main__":
    main()
