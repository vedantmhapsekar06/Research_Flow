"""
One-off diagnostic: prints the RAW (pre-normalization) semantic and
BM25 scores for both a real question and a nonsense question, so we
can pick threshold constants based on actual numbers instead of
guessing.
"""

from services.embeddings import get_embedding
from services.vector_store import semantic_search
from services.retriever import _build_bm25_index, _tokenize
from core.config import SEMANTIC_TOP_K, KEYWORD_TOP_K


def inspect(query: str):
    print(f"\n{'='*70}\nQUERY: {query}\n{'='*70}")

    query_embedding = get_embedding(query)
    semantic_results = semantic_search(query_embedding, top_k=SEMANTIC_TOP_K)
    print("\n-- Semantic (raw similarity = 1 - distance) --")
    for r in semantic_results[:8]:
        sim = round(1.0 - r["distance"], 4)
        print(f"  {sim}  |  page {r['metadata']['page']}  |  {r['text'][:70]}")

    bm25, chunk_ids, store = _build_bm25_index()
    if bm25:
        tokenized = _tokenize(query)
        scores = bm25.get_scores(tokenized)
        ranked = sorted(zip(chunk_ids, scores), key=lambda x: x[1], reverse=True)
        print("\n-- BM25 (raw score) --")
        for cid, score in ranked[:8]:
            if score > 0:
                print(f"  {round(score, 4)}  |  page {store[cid]['page']}  |  {store[cid]['text'][:70]}")


inspect("Does the PaperQA paper mention using Redis for caching search results?")
inspect("How many questions are in the LitQA dataset?")