"""Document ingestion module for the chunking evaluation pipeline.

Parses each document type into RawUnit objects with position metadata.
Supports: Markdown (.md), plain text (.txt), PDF (.pdf), SEC HTML (.html).
"""

import os
import re

from testing.eval.models import PositionMetadata, RawUnit


# Supported file extensions and their handler mapping
_SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".html"}


def ingest_document(file_path: str) -> list[RawUnit]:
    """Route to the correct parser based on file extension.

    Args:
        file_path: Path to the document file to ingest.

    Returns:
        A list of RawUnit objects with position metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError: If the file cannot be read due to permissions.
        ValueError: If the file extension is not supported.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: '{file_path}' — file not found"
        )

    if not os.access(file_path, os.R_OK):
        raise PermissionError(
            f"Permission denied: '{file_path}' — permission denied"
        )

    ext = os.path.splitext(file_path)[1].lower()

    if ext not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format: '{file_path}' — format not supported. "
            f"Supported extensions: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
        )

    if ext == ".md":
        return _ingest_markdown(file_path)
    elif ext == ".txt":
        return _ingest_text(file_path)
    elif ext == ".pdf":
        return _ingest_pdf(file_path)
    elif ext == ".html":
        return _ingest_html_sec(file_path)

    # Should not reach here due to extension check above
    return []


def _ingest_markdown(file_path: str) -> list[RawUnit]:
    """Ingest a Markdown file by splitting on consecutive blank lines.

    Splits at paragraph boundaries (one or more consecutive blank lines)
    and assigns 1-based line_number_start and line_number_end to each RawUnit.

    Args:
        file_path: Path to the Markdown file.

    Returns:
        List of RawUnit objects with line_number position metadata.
    """
    return _split_by_blank_lines(file_path)


def _ingest_text(file_path: str) -> list[RawUnit]:
    """Ingest a plain text file by splitting on consecutive blank lines.

    Same logic as markdown: splits at paragraph boundaries.

    Args:
        file_path: Path to the text file.

    Returns:
        List of RawUnit objects with line_number position metadata.
    """
    return _split_by_blank_lines(file_path)


def _split_by_blank_lines(file_path: str) -> list[RawUnit]:
    """Split a text file into RawUnits at paragraph boundaries.

    A paragraph boundary is one or more consecutive blank lines
    (lines containing only whitespace).

    Args:
        file_path: Path to the file.

    Returns:
        List of RawUnit objects with line_number_start and line_number_end
        position metadata (1-based).
    """
    doc_id = os.path.basename(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    raw_units: list[RawUnit] = []
    paragraph_lines: list[str] = []
    paragraph_start_line: int | None = None

    for i, line in enumerate(lines):
        line_number = i + 1  # 1-based

        if line.strip() == "":
            # Blank line — if we have accumulated paragraph content, flush it
            if paragraph_lines:
                text = "".join(paragraph_lines).strip()
                if text:
                    raw_units.append(
                        RawUnit(
                            text=text,
                            doc_id=doc_id,
                            position_metadata=PositionMetadata(
                                line_number_start=paragraph_start_line,
                                line_number_end=line_number - 1,
                            ),
                        )
                    )
                paragraph_lines = []
                paragraph_start_line = None
        else:
            # Non-blank line — accumulate
            if paragraph_start_line is None:
                paragraph_start_line = line_number
            paragraph_lines.append(line)

    # Flush remaining paragraph at end of file
    if paragraph_lines:
        text = "".join(paragraph_lines).strip()
        if text:
            raw_units.append(
                RawUnit(
                    text=text,
                    doc_id=doc_id,
                    position_metadata=PositionMetadata(
                        line_number_start=paragraph_start_line,
                        line_number_end=len(lines),
                    ),
                )
            )

    return raw_units


def _ingest_pdf(file_path: str) -> list[RawUnit]:
    """Ingest a PDF file by extracting text per page.

    Uses PyMuPDF (fitz) to extract text from each page. Produces one
    RawUnit per page that contains extractable text. Skips empty pages.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of RawUnit objects with page_number position metadata (1-based).

    Raises:
        RuntimeError: If PDF parsing fails.
    """
    try:
        import fitz
    except ImportError:
        raise ImportError(
            "PyMuPDF (fitz) is required for PDF ingestion. "
            "Install it with: pip install pymupdf"
        )

    doc_id = os.path.basename(file_path)
    raw_units: list[RawUnit] = []

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        raise RuntimeError(
            f"PDF parse failure: '{file_path}' — {str(e)}"
        )

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text().strip()

            if not text:
                # Skip empty pages
                continue

            page_number = page_num + 1  # 1-based
            raw_units.append(
                RawUnit(
                    text=text,
                    doc_id=doc_id,
                    position_metadata=PositionMetadata(
                        page_number_start=page_number,
                        page_number_end=page_number,
                    ),
                )
            )
    finally:
        doc.close()

    return raw_units


def _ingest_html_sec(file_path: str) -> list[RawUnit]:
    """Ingest an SEC 10-K HTML filing by parsing Item headings.

    Uses BeautifulSoup to extract text, then identifies Item headings
    with regex r"Item\\s+(\\d+[A-Z]?)" (case-insensitive). Assigns item_id
    to each RawUnit representing text between one Item heading and the next
    distinct Item heading.

    When the same Item is referenced multiple times (e.g., "Item 15" appearing
    as both a heading and in cross-references), only the distinct Item boundaries
    are used for splitting. Text between duplicate references of the same Item
    is merged into a single section.

    Returns empty list if no Item headings are found.

    Args:
        file_path: Path to the HTML file.

    Returns:
        List of RawUnit objects with item_id position metadata.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(
            "BeautifulSoup is required for HTML ingestion. "
            "Install it with: pip install beautifulsoup4"
        )

    doc_id = os.path.basename(file_path)

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")
    full_text = soup.get_text(separator="\n")

    # Find all Item headings with their positions in the text
    item_pattern = re.compile(r"Item\s+(\d+[A-Z]?)", re.IGNORECASE)
    matches = list(item_pattern.finditer(full_text))

    if not matches:
        return []

    # Deduplicate: keep only the first occurrence of each distinct Item
    seen_items: dict[str, int] = {}  # item_id -> index in deduped list
    deduped_matches: list[tuple[str, int]] = []  # (item_id, start_pos)

    for match in matches:
        item_id = f"Item {match.group(1).upper()}"
        if item_id not in seen_items:
            seen_items[item_id] = len(deduped_matches)
            deduped_matches.append((item_id, match.start()))

    # Build raw units from deduplicated item boundaries
    raw_units: list[RawUnit] = []

    for i, (item_id, start_pos) in enumerate(deduped_matches):
        # End position is start of next distinct item or end of text
        if i + 1 < len(deduped_matches):
            end_pos = deduped_matches[i + 1][1]
        else:
            end_pos = len(full_text)

        section_text = full_text[start_pos:end_pos].strip()

        if section_text:
            raw_units.append(
                RawUnit(
                    text=section_text,
                    doc_id=doc_id,
                    position_metadata=PositionMetadata(
                        item_id=item_id,
                    ),
                )
            )

    return raw_units
