"""
Central configuration for ResearchFlow.

Everything that might change (model names, folder paths, chunk sizes)
lives here so you never have to hunt through the codebase to tweak it.
"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent          # .../backend
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploaded_papers"
CHROMA_DIR = DATA_DIR / "chroma_db"
STORE_DIR = DATA_DIR / "store"                              # papers.json, chunks.json

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
STORE_DIR.mkdir(parents=True, exist_ok=True)

PAPERS_REGISTRY_FILE = STORE_DIR / "papers.json"
CHUNKS_STORE_FILE = STORE_DIR / "chunks.json"

# ---------------------------------------------------------------------------
# Ollama (local LLM + embeddings — no API key, no internet call)
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("RESEARCHFLOW_CHAT_MODEL", "llama3.2")
EMBED_MODEL = os.getenv("RESEARCHFLOW_EMBED_MODEL", "nomic-embed-text")

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE_WORDS = 220          # ~roughly 1-2 paragraphs
CHUNK_OVERLAP_WORDS = 40

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
SEMANTIC_TOP_K = 15
KEYWORD_TOP_K = 15
FINAL_TOP_K = 6
SEMANTIC_WEIGHT = 0.6
KEYWORD_WEIGHT = 0.4

# Absolute (pre-normalization) relevance floors. A chunk must clear at least
# ONE of these on its raw score to be considered real evidence at all.
# These are tunable — raise them if you see irrelevant chunks still leaking
# through; lower them if legitimate answers start losing their sources.
SEMANTIC_ABS_MIN = 0.45   # raw cosine similarity (1 - distance), range ~0-1
KEYWORD_ABS_MIN = 1.0     # raw BM25 score, unbounded, dataset-dependent

# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
MAX_HISTORY_TURNS = 6            # how many past turns we keep for context
COLLECTION_NAME = "researchflow_papers"