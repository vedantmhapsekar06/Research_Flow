"""
retriever.py
------------
The heart of the RAG pipeline's "R" (Retrieval).

Combines two different retrieval strategies and fuses their results:

1. Semantic search (ChromaDB + Ollama embeddings)
   - Good at: conceptual/paraphrased matches
   - Weak at: exact technical terms, acronyms, numbers

2. BM25 keyword search (rank_bm25)
   - Good at: exact term matches
   - Weak at: paraphrasing / conceptual similarity

We run both, normalize their scores to a comparable 0-1 range, and
combine them with a weighted sum.

Note on relevance filtering: we do NOT filter chunks by a raw score
threshold here. Testing showed that for a small, topically-narrow
corpus, semantic similarity for irrelevant queries sits in nearly the
same numeric range as similarity for relevant ones (a known embedding
model quirk), and BM25 scores stay high whenever a query reuses common
domain vocabulary (e.g. "search", "retrieval") even if the specific
subject is absent. A numeric floor can't reliably tell these apart.
Instead, "is this actually relevant" is decided by the LLM itself in
chat_service.py, which is far better at judging semantic relevance
than a raw similarity score is.
"""

import re
from rank_bm25 import BM25Okapi

from core.config import (
    SEMANTIC_TOP_K, KEYWORD_TOP_K, FINAL_TOP_K,
    SEMANTIC_WEIGHT, KEYWORD_WEIGHT,
)
from services.embeddings import get_embedding
from services.vector_store import semantic_search
from services.storage import load_chunks


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_bm25_index():
    store = load_chunks()
    chunk_ids = list(store.keys())
    corpus = [_tokenize(store[cid]["text"]) for cid in chunk_ids]
    if not corpus:
        return None, [], {}
    bm25 = BM25Okapi(corpus)
    return bm25, chunk_ids, store


def _normalize(scores: dict) -> dict:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def hybrid_search(query: str, paper_id: str | None = None) -> list[dict]:
    query_embedding = get_embedding(query)
    semantic_results = semantic_search(query_embedding, top_k=SEMANTIC_TOP_K)

    semantic_scores = {r["chunk_id"]: 1.0 - r["distance"] for r in semantic_results}
    semantic_meta = {r["chunk_id"]: r for r in semantic_results}

    bm25, chunk_ids, store = _build_bm25_index()
    keyword_scores = {}
    if bm25 is not None:
        tokenized_query = _tokenize(query)
        raw_scores = bm25.get_scores(tokenized_query)
        ranked = sorted(zip(chunk_ids, raw_scores), key=lambda x: x[1], reverse=True)
        for cid, score in ranked[:KEYWORD_TOP_K]:
            if score > 0:
                keyword_scores[cid] = score

    norm_semantic = _normalize(semantic_scores)
    norm_keyword = _normalize(keyword_scores)

    all_ids = set(norm_semantic) | set(norm_keyword)
    fused = {}
    for cid in all_ids:
        s = norm_semantic.get(cid, 0.0)
        k = norm_keyword.get(cid, 0.0)
        fused[cid] = SEMANTIC_WEIGHT * s + KEYWORD_WEIGHT * k

    top_ids = sorted(fused, key=fused.get, reverse=True)[:FINAL_TOP_K]

    results = []
    for cid in top_ids:
        if cid in semantic_meta:
            meta = semantic_meta[cid]
            results.append({
                "chunk_id": cid,
                "text": meta["text"],
                "paper_id": meta["metadata"]["paper_id"],
                "paper_name": meta["metadata"]["paper_name"],
                "page": meta["metadata"]["page"],
                "section": meta["metadata"]["section"] or None,
                "score": round(fused[cid], 4),
            })
        elif cid in store:
            c = store[cid]
            results.append({
                "chunk_id": cid,
                "text": c["text"],
                "paper_id": c["paper_id"],
                "paper_name": c["paper_name"],
                "page": c["page"],
                "section": c["section"] or None,
                "score": round(fused[cid], 4),
            })

    if paper_id:
        results = [r for r in results if r["paper_id"] == paper_id]

    return results