"""
retriever.py
------------
The heart of the RAG pipeline's "R" (Retrieval).

Combines two retrieval strategies and fuses their rankings:

1. Semantic search (ChromaDB + Ollama embeddings)
   - Good at: conceptual/paraphrased matches
   - Weak at: exact technical terms, acronyms, numbers

2. BM25 keyword search (rank_bm25)
   - Good at: exact term matches
   - Weak at: paraphrasing / conceptual similarity

Pipeline:
  raw scores -> absolute relevance floors -> Reciprocal Rank Fusion
  -> multi-paper balancing -> Top-K

Why this differs from the earlier min-max fusion:
- Min-max over a truncated top-K list always maps the best hit to 1.0 and the
  worst to 0.0, so a lone weak BM25 match got full weight and an irrelevant
  query still "had" a best chunk. RRF only uses ranks, and the absolute floors
  (applied to the RAW scores) are what decide whether a chunk is real evidence.
- A chunk outside one retriever's top-K used to get exactly 0 from it; with
  RRF it simply contributes nothing from that retriever.
- The floors are lenient by default (see config.py) because raw scores are
  corpus-dependent; the LLM's NOT_FOUND judgement in chat_service.py remains
  the final check.
"""

import re
from threading import Lock

from rank_bm25 import BM25Okapi

from core.config import (
    SEMANTIC_TOP_K, KEYWORD_TOP_K, FINAL_TOP_K,
    SEMANTIC_WEIGHT, KEYWORD_WEIGHT, RRF_K,
    SEMANTIC_ABS_MIN, KEYWORD_ABS_MIN, PAPER_BALANCE_RATIO,
    CHUNKS_STORE_FILE,
)
from services.embeddings import get_embedding
from services.vector_store import semantic_search, count as vector_count
from services import storage


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# BM25 index (cached; rebuilt only when the chunk store actually changes)
# ---------------------------------------------------------------------------
_bm25_lock = Lock()
_bm25_cache = {"key": None, "value": (None, [], {})}


def _store_key():
    try:
        st = CHUNKS_STORE_FILE.stat()
        sig = (st.st_mtime_ns, st.st_size)
    except FileNotFoundError:
        sig = None
    return (storage.chunks_version(), sig)


def _build_bm25_index():
    """Returns (bm25 | None, chunk_ids, store). Cached across queries."""
    key = _store_key()
    with _bm25_lock:
        if _bm25_cache["key"] == key:
            return _bm25_cache["value"]

        store = storage.load_chunks()
        chunk_ids = list(store.keys())
        if not chunk_ids:
            value = (None, [], {})
        else:
            corpus = [_tokenize(store[cid]["text"]) for cid in chunk_ids]
            value = (BM25Okapi(corpus), chunk_ids, store)

        _bm25_cache["key"] = key
        _bm25_cache["value"] = value
        return value


# ---------------------------------------------------------------------------
# Pure helpers (no I/O — unit-tested directly)
# ---------------------------------------------------------------------------
def fuse_rankings(semantic_ranked: list[str], keyword_ranked: list[str],
                  w_sem: float = SEMANTIC_WEIGHT, w_kw: float = KEYWORD_WEIGHT,
                  k: int = RRF_K) -> dict[str, float]:
    """
    Weighted Reciprocal Rank Fusion. Inputs are chunk ids ordered best-first.
    score(id) = w_sem/(k+rank_sem) + w_kw/(k+rank_kw), missing lists contribute 0.
    """
    fused: dict[str, float] = {}
    for rank, cid in enumerate(semantic_ranked, start=1):
        fused[cid] = fused.get(cid, 0.0) + w_sem / (k + rank)
    for rank, cid in enumerate(keyword_ranked, start=1):
        fused[cid] = fused.get(cid, 0.0) + w_kw / (k + rank)
    return fused


def select_with_paper_balance(ranked: list[str], fused: dict[str, float],
                              paper_of: dict[str, str], k: int,
                              ratio: float = PAPER_BALANCE_RATIO) -> list[str]:
    """
    Picks k chunk ids from `ranked` (best-first). First, the best chunk of each
    paper whose best fused score is >= ratio * the overall best is guaranteed a
    slot (so comparison questions see every relevant paper); remaining slots go
    to the highest-ranked leftovers. Output keeps overall fused-rank order.
    """
    if not ranked:
        return []

    top_score = fused[ranked[0]]
    chosen: list[str] = []
    seen_papers: set[str] = set()

    for cid in ranked:
        paper = paper_of[cid]
        if paper in seen_papers:
            continue
        seen_papers.add(paper)
        if fused[cid] >= ratio * top_score and len(chosen) < k:
            chosen.append(cid)

    for cid in ranked:
        if len(chosen) >= k:
            break
        if cid not in chosen:
            chosen.append(cid)

    order = {cid: i for i, cid in enumerate(ranked)}
    return sorted(chosen, key=order.get)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def hybrid_search(query: str, paper_id: str | None = None) -> list[dict]:
    bm25, chunk_ids, store = _build_bm25_index()

    # Nothing indexed at all: skip the embedding call entirely.
    if not store and vector_count() == 0:
        return []

    # --- Semantic -----------------------------------------------------------
    query_embedding = get_embedding(query, is_query=True)
    semantic_results = semantic_search(query_embedding, top_k=SEMANTIC_TOP_K, paper_id=paper_id)
    semantic_meta = {r["chunk_id"]: r for r in semantic_results}
    semantic_sims = {r["chunk_id"]: 1.0 - r["distance"] for r in semantic_results}
    # Chroma returns ascending distance, i.e. best-first already. Apply the floor.
    semantic_ranked = [
        cid for cid in semantic_meta if semantic_sims[cid] >= SEMANTIC_ABS_MIN
    ]

    # --- Keyword ------------------------------------------------------------
    keyword_scores: dict[str, float] = {}
    if bm25 is not None:
        raw_scores = bm25.get_scores(_tokenize(query))
        candidates = [
            (cid, float(score))
            for cid, score in zip(chunk_ids, raw_scores)
            if paper_id is None or store[cid]["paper_id"] == paper_id
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        for cid, score in candidates[:KEYWORD_TOP_K]:
            if score > 0 and score >= KEYWORD_ABS_MIN:
                keyword_scores[cid] = score
    keyword_ranked = list(keyword_scores)   # insertion order == best-first

    # --- Fuse + select ------------------------------------------------------
    fused = fuse_rankings(semantic_ranked, keyword_ranked)
    if not fused:
        return []

    def _lookup(cid: str) -> dict | None:
        if cid in semantic_meta:
            m = semantic_meta[cid]
            return {
                "text": m["text"],
                "paper_id": m["metadata"]["paper_id"],
                "paper_name": m["metadata"]["paper_name"],
                "page": m["metadata"]["page"],
                "section": m["metadata"].get("section") or None,
            }
        if cid in store:
            c = store[cid]
            return {
                "text": c["text"], "paper_id": c["paper_id"],
                "paper_name": c["paper_name"], "page": c["page"],
                "section": c.get("section") or None,
            }
        return None

    info = {cid: _lookup(cid) for cid in fused}
    info = {cid: v for cid, v in info.items() if v is not None}
    ranked = sorted((c for c in fused if c in info), key=fused.get, reverse=True)
    paper_of = {cid: info[cid]["paper_id"] for cid in ranked}

    top_ids = select_with_paper_balance(ranked, fused, paper_of, FINAL_TOP_K)

    # Display relevance in 0-1: fused score relative to the best possible score
    # (rank 1 in both retrievers).
    max_possible = (SEMANTIC_WEIGHT + KEYWORD_WEIGHT) / (RRF_K + 1)

    results = []
    for cid in top_ids:
        results.append({
            "chunk_id": cid,
            **info[cid],
            "score": round(min(1.0, fused[cid] / max_possible), 4),
            "semantic_sim": round(semantic_sims[cid], 4) if cid in semantic_sims else None,
            "bm25": round(keyword_scores[cid], 4) if cid in keyword_scores else None,
        })
    return results
