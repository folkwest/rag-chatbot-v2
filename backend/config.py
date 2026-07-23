from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is not set. "
        "Please create a .env file with your OpenAI API key."
    )

# Models
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# Chunking defaults
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# RAG
TOP_K = 5

# ChromaDB
CHROMA_MODE = os.getenv("CHROMA_MODE", "local")  # "local" or "server"
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8100"))
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", ".chroma_data")

# Hybrid retrieval configuration
RRF_CONSTANT = int(os.getenv("RRF_CONSTANT", "60"))
CANDIDATE_SET_SIZE = int(os.getenv("CANDIDATE_SET_SIZE", "20"))
FINAL_CONTEXT_SIZE = int(os.getenv("FINAL_CONTEXT_SIZE", "5"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

if CANDIDATE_SET_SIZE <= FINAL_CONTEXT_SIZE:
    raise ValueError(
        f"CANDIDATE_SET_SIZE ({CANDIDATE_SET_SIZE}) must be greater than "
        f"FINAL_CONTEXT_SIZE ({FINAL_CONTEXT_SIZE})"
    )
