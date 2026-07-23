import logging
import os

from pypdf import PdfReader

logger = logging.getLogger(__name__)


def parse_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    try:
        reader = PdfReader(file_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages)
        if not text.strip():
            raise ValueError(f"No extractable text in PDF: {file_path}")
        return text
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to parse PDF '{file_path}': {e}") from e


def parse_txt(file_path: str) -> str:
    """Read a plain text file."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Text file not found: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        # Fallback to latin-1 which never fails
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read()
