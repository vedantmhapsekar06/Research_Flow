"""
storage.py
----------
Small helper for two lightweight JSON "tables" that live alongside ChromaDB:

1. papers.json  -> registry of every uploaded paper and its status
2. chunks.json  -> raw chunk text + metadata, needed for BM25 keyword search
                   (ChromaDB is vector-only, so we keep our own copy of the
                   text to build a BM25 index from).

Concurrency notes (these were real bugs before):
- The lock is re-entrant and is held across the WHOLE read-modify-write of
  every mutator. Previously it only covered each individual read or write, so
  two threads updating the registry could silently overwrite each other.
- Writes go to a temp file and are swapped in with os.replace(), so a crash
  mid-write can't leave a truncated, unparseable JSON file behind.
- claim_papers_for_processing() flips statuses atomically, so two concurrent
  /process calls can never both pick up the same paper.
"""

import json
import os
from threading import RLock

from core.config import PAPERS_REGISTRY_FILE, CHUNKS_STORE_FILE

_lock = RLock()
_chunks_version = 0   # bumped on every chunk-store write; lets the retriever cache BM25


def _read_json(path, default):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Papers registry
# ---------------------------------------------------------------------------
def load_papers() -> dict:
    """Returns {paper_id: {filename, status, pages, error, sha256, retryable}}"""
    with _lock:
        return _read_json(PAPERS_REGISTRY_FILE, {})


def save_papers(papers: dict) -> None:
    with _lock:
        _write_json(PAPERS_REGISTRY_FILE, papers)


def upsert_paper(paper_id: str, **fields) -> None:
    with _lock:
        papers = load_papers()
        papers.setdefault(paper_id, {})
        papers[paper_id].update(fields)
        save_papers(papers)


def update_paper(paper_id: str, **fields) -> bool:
    """Like upsert_paper but never resurrects a deleted record. Returns False if missing."""
    with _lock:
        papers = load_papers()
        if paper_id not in papers:
            return False
        papers[paper_id].update(fields)
        save_papers(papers)
        return True


def paper_exists(paper_id: str) -> bool:
    return paper_id in load_papers()


def delete_paper_record(paper_id: str) -> None:
    with _lock:
        papers = load_papers()
        papers.pop(paper_id, None)
        save_papers(papers)


def find_paper_by_hash(sha256: str):
    """Returns (paper_id, fields) of an already-uploaded identical file, or None."""
    for pid, fields in load_papers().items():
        if fields.get("sha256") == sha256:
            return pid, fields
    return None


def claim_papers_for_processing() -> list[tuple[str, str]]:
    """
    Atomically marks every paper that needs processing as 'processing' and
    returns [(paper_id, filename), ...] for the caller to work on.

    Eligible: freshly uploaded papers, and failed papers whose failure was
    transient (Ollama down / timed out). Permanent failures (corrupt PDF, no
    extractable text) are NOT retried — they would just fail again.
    """
    with _lock:
        papers = load_papers()
        claimed = []
        for pid, f in papers.items():
            status = f.get("status")
            if status == "uploaded" or (status == "failed" and f.get("retryable")):
                f["status"] = "processing"
                f["error"] = None
                claimed.append((pid, f["filename"]))
        if claimed:
            save_papers(papers)
        return claimed


def reset_stale_processing() -> int:
    """
    Called once at server start-up: nothing can genuinely be 'processing' in a
    fresh process, so those papers were interrupted by a crash/restart. Put
    them back to 'uploaded' so the next /process call picks them up.
    """
    with _lock:
        papers = load_papers()
        n = 0
        for f in papers.values():
            if f.get("status") == "processing":
                f["status"] = "uploaded"
                n += 1
        if n:
            save_papers(papers)
        return n


# ---------------------------------------------------------------------------
# Chunk store (text corpus for BM25)
# ---------------------------------------------------------------------------
def load_chunks() -> dict:
    """Returns {chunk_id: {text, paper_id, paper_name, page, section}}"""
    with _lock:
        return _read_json(CHUNKS_STORE_FILE, {})


def save_chunks(chunks: dict) -> None:
    global _chunks_version
    with _lock:
        _write_json(CHUNKS_STORE_FILE, chunks)
        _chunks_version += 1


def chunks_version() -> int:
    return _chunks_version


def add_chunks_to_store(chunks: list[dict]) -> None:
    with _lock:
        store = load_chunks()
        for c in chunks:
            store[c["chunk_id"]] = {
                "text": c["text"],
                "paper_id": c["paper_id"],
                "paper_name": c["paper_name"],
                "page": c["page"],
                "section": c["section"],
            }
        save_chunks(store)


def remove_paper_chunks_from_store(paper_id: str) -> None:
    with _lock:
        store = load_chunks()
        store = {cid: c for cid, c in store.items() if c["paper_id"] != paper_id}
        save_chunks(store)
