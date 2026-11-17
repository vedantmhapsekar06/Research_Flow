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
# RESEARCHFLOW_DATA_DIR lets tests (and Docker users) point storage elsewhere
# without touching real data.
DATA_DIR = Path(os.getenv("RESEARCHFLOW_DATA_DIR", str(BASE_DIR / "data")))
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
# If fewer than this many NEW words would remain after a chunk, the chunk is
# extended to absorb them instead of emitting a tiny tail chunk that is
# mostly overlap with the previous one.
CHUNK_MIN_TAIL_WORDS = 30

# An overlap >= chunk size would make the chunking loop never advance.
if not (0 <= CHUNK_OVERLAP_WORDS < CHUNK_SIZE_WORDS):
    raise ValueError("CHUNK_OVERLAP_WORDS must be >= 0 and < CHUNK_SIZE_WORDS")

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
SEMANTIC_TOP_K = 15
KEYWORD_TOP_K = 15
FINAL_TOP_K = 6

# Fusion is Reciprocal Rank Fusion (RRF): each retriever contributes
# weight / (RRF_K + rank). Unlike min-max normalising the truncated top-K
# lists, RRF doesn't hand a lone weak BM25 hit a full score of 1.0 and doesn't
# treat "not in the other list's top-K" as a score of exactly 0.
RRF_K = 60
SEMANTIC_WEIGHT = 0.6
KEYWORD_WEIGHT = 0.4

# Absolute (raw-score) relevance floors, applied BEFORE fusion. A chunk only
# enters a retriever's ranking if it clears that retriever's floor, so a chunk
# survives overall only if it clears at least ONE floor. The defaults are
# deliberately lenient: run `python inspect_scores.py` against your own
# papers and raise them if irrelevant chunks still leak through. Set a floor
# to 0 to disable it. The LLM's NOT_FOUND decision is the second line of
# defence, not the only one.
SEMANTIC_ABS_MIN = 0.30   # raw cosine similarity (1 - distance), range ~0-1
KEYWORD_ABS_MIN = 1.0     # raw BM25 score, unbounded, corpus-dependent

# Multi-paper balance: after fusion, the best chunk of every paper whose best
# fused score is at least this fraction of the overall best is guaranteed a
# slot in the final Top-K, so one paper can't crowd out the others on
# comparison questions. Papers that are clearly less relevant are not forced in.
PAPER_BALANCE_RATIO = 0.5

# nomic-embed-text was trained with task prefixes; without them retrieval
# quality is noticeably worse. Applied only when the model name contains "nomic".
EMBED_USE_TASK_PREFIXES = True

# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
MAX_HISTORY_TURNS = 6            # past TURNS (1 turn = question + answer) kept for context
CONDENSE_MAX_ANSWER_CHARS = 300  # prior answers are truncated this much in the condenser prompt
COLLECTION_NAME = "researchflow_papers"

# ---------------------------------------------------------------------------
# Uploads / server
# ---------------------------------------------------------------------------
MAX_UPLOAD_MB = int(os.getenv("RESEARCHFLOW_MAX_UPLOAD_MB", "50"))

# The app serves its own frontend, so same-origin requests need no CORS at
# all. A wildcard would let ANY website you visit call this local API and read
# your papers, so only explicit origins are allowed. Override with a
# comma-separated RESEARCHFLOW_ALLOWED_ORIGINS if you host the UI elsewhere.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "RESEARCHFLOW_ALLOWED_ORIGINS",
        "http://localhost:8500,http://127.0.0.1:8500,http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")
    if o.strip()
]