"""Main orchestrator for the chunking evaluation pipeline.

Ties together ingestion, chunking, retrieval, scoring, and output modules
to run the full evaluation pipeline end-to-end.
"""

import json
import os
import sys

try:
    from testing.eval import config
    from testing.eval.models import EvalQuestion, PositionMetadata
    from testing.eval.ingest import ingest_document
    from testing.eval.chunkers import STRATEGIES
    from testing.eval.retrieval import run_retrieval
    from testing.eval.scoring import score_results
    from testing.eval.output import write_csv, print_summary
except ImportError as e:
    print(
        f"Error: Missing required dependency — {e}. "
        "Please install dependencies with: pip install -r testing/eval/requirements.txt",
        file=sys.stderr,
    )
    sys.exit(1)


# Document filenames in the test corpus
_CORPUS_DOCUMENTS = [
    "openapi-callbacks.md",
    "attention_is_all_you_need.pdf",
    "pride_and_prejudice.txt",
    "sec-filing.html",
]


def main() -> None:
    """Run the full evaluation pipeline."""
    # 1. Check that OPENAI_API_KEY is set
    if not config.OPENAI_API_KEY:
        print(
            "Error: OPENAI_API_KEY environment variable is not set. "
            "Please set it before running the evaluation pipeline.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 2. Load and validate questions from questions.json
    questions = _load_questions()

    # 3. Ingest all 4 test corpus documents
    document_texts: dict[str, str] = {}
    all_raw_units: dict[str, list] = {}

    for doc_filename in _CORPUS_DOCUMENTS:
        file_path = os.path.join(config.CORPUS_DIR, doc_filename)
        print(f"Ingesting: {doc_filename}...")
        raw_units = ingest_document(file_path)
        all_raw_units[doc_filename] = raw_units

        # Store full text for integrity checking
        document_texts[doc_filename] = "\n".join(unit.text for unit in raw_units)

    # 4. For each chunking strategy, chunk all documents
    strategy_chunks: dict[str, list] = {}

    for strategy_name, strategy in STRATEGIES.items():
        print(f"Chunking with strategy: {strategy_name}...")
        chunks = []
        for doc_filename in _CORPUS_DOCUMENTS:
            raw_units = all_raw_units[doc_filename]
            doc_chunks = strategy.chunk(raw_units)
            chunks.extend(doc_chunks)
        strategy_chunks[strategy_name] = chunks

    # 5. Run retrieval
    print("Running retrieval...")
    retrieval_results = run_retrieval(questions, strategy_chunks)

    # 6. Score results
    print("Scoring results...")
    scored_results = score_results(
        retrieval_results, questions, strategy_chunks, document_texts
    )

    # 7. Compute integrity_flags dict: check ALL chunks against their source document
    print("Checking chunk integrity...")
    from testing.eval.scoring import check_integrity
    integrity_flags: dict[str, int] = {}
    for strategy_name, chunks in strategy_chunks.items():
        flag_count = 0
        for chunk in chunks:
            doc_text = document_texts.get(chunk.doc_id, "")
            if doc_text:
                flags = check_integrity(chunk, doc_text)
                chunk.integrity_flags = flags
                if flags:
                    flag_count += 1
        integrity_flags[strategy_name] = flag_count

    # 8. Write CSV
    output_path = os.path.join(config.RESULTS_DIR, "eval_results.csv")
    print(f"Writing results to: {output_path}")
    write_csv(scored_results, output_path)

    # 9. Print summary
    print_summary(scored_results, integrity_flags)

    print("\nEvaluation complete.")


def _load_questions() -> list[EvalQuestion]:
    """Load and validate evaluation questions from questions.json.

    Reads the JSON file at config.QUESTIONS_FILE, validates all entries
    have question_text and document_id, and converts to EvalQuestion objects.

    Returns:
        List of EvalQuestion objects.

    Raises:
        FileNotFoundError: If questions.json is missing.
        ValueError: If any entry is missing required fields.
    """
    if not os.path.exists(config.QUESTIONS_FILE):
        raise FileNotFoundError(
            f"Questions file not found: {config.QUESTIONS_FILE}"
        )

    with open(config.QUESTIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Invalid questions file: expected a JSON array, got {type(data).__name__}"
        )

    questions: list[EvalQuestion] = []

    for i, entry in enumerate(data):
        # Validate required fields
        if "question_text" not in entry:
            raise ValueError(
                f"Invalid question entry at index {i}: missing 'question_text'"
            )
        if "document_id" not in entry:
            raise ValueError(
                f"Invalid question entry at index {i}: missing 'document_id'"
            )

        # Construct PositionMetadata from ground_truth_location if present
        ground_truth: PositionMetadata | None = None
        gt_data = entry.get("ground_truth_location")
        if gt_data is not None:
            ground_truth = PositionMetadata(
                line_number_start=gt_data.get("line_number_start"),
                line_number_end=gt_data.get("line_number_end"),
                page_number_start=gt_data.get("page_number_start"),
                page_number_end=gt_data.get("page_number_end"),
                item_id=gt_data.get("item_id"),
            )

        questions.append(
            EvalQuestion(
                question_id=entry.get("question_id", f"q_{i}"),
                question_text=entry["question_text"],
                document_id=entry["document_id"],
                ground_truth_location=ground_truth,
            )
        )

    return questions


if __name__ == "__main__":
    main()
