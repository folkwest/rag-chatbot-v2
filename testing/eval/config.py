"""Configuration module for the chunking evaluation pipeline.

Reads environment variables and defines defaults for chunking parameters,
embedding model, retrieval settings, and file paths.
"""

import os

# Load .env file from project root if python-dotenv is available
try:
    from dotenv import load_dotenv

    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
except ImportError:
    pass  # python-dotenv not installed — rely on environment variables directly

# Embedding
EMBEDDING_MODEL = os.getenv("EVAL_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Required at runtime, not at import

# Chunking defaults
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50
SEMANTIC_THRESHOLD = float(os.getenv("EVAL_SEMANTIC_THRESHOLD", "0.4"))
MAX_SENTENCES_PER_CHUNK = 20

# Retrieval
TOP_K = int(os.getenv("EVAL_TOP_K", "3"))

# Paths (relative to this module's location)
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(_MODULE_DIR, "..")
QUESTIONS_FILE = os.path.join(_MODULE_DIR, "questions.json")
RESULTS_DIR = os.path.join(_MODULE_DIR, "results")
