"""
storage.py
----------
Small helper for two lightweight JSON "tables" that live alongside
ChromaDB:

1. papers.json  -> registry of every uploaded paper and its status
2. chunks.json  -> raw chunk text + metadata, needed for BM25 keyword
                   search (ChromaDB is vector-only; it doesn't do
                   keyword search, so we keep our own copy of the
                   text to build a BM25 index from).

For a resume project this beats spinning up a real database for
something this small — but these functions are the only place that
touches these files, so swapping in a real DB later only means
rewriting this one module.
"""

import json
from threading import Lock
from core.config import PAPERS_REGISTRY_FILE, CHUNKS_STORE_FILE

_lock = Lock()


def _read_json(path, default):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Papers registry
# ---------------------------------------------------------------------------
def load_papers() -> dict:
    """Returns {paper_id: {filename, status, pages, error}}"""
    with _lock:
        return _read_json(PAPERS_REGISTRY_FILE, {})


def save_papers(papers: dict) -> None:
    with _lock:
        _write_json(PAPERS_REGISTRY_FILE, papers)


def upsert_paper(paper_id: str, **fields) -> None:
    papers = load_papers()
    papers.setdefault(paper_id, {})
    papers[paper_id].update(fields)
    save_papers(papers)


def delete_paper_record(paper_id: str) -> None:
    papers = load_papers()
    papers.pop(paper_id, None)
    save_papers(papers)


# ---------------------------------------------------------------------------
# Chunk store (text corpus for BM25)
# ---------------------------------------------------------------------------
def load_chunks() -> dict:
    """Returns {chunk_id: {text, paper_id, paper_name, page, section}}"""
    with _lock:
        return _read_json(CHUNKS_STORE_FILE, {})


def save_chunks(chunks: dict) -> None:
    with _lock:
        _write_json(CHUNKS_STORE_FILE, chunks)


def add_chunks_to_store(chunks: list[dict]) -> None:
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
    store = load_chunks()
    store = {cid: c for cid, c in store.items() if c["paper_id"] != paper_id}
    save_chunks(store)