from services.retriever import hybrid_search

results = hybrid_search("what is BM25 used for")

for r in results:
    print(f"Page {r['page']} | score {r['score']} | {r['text'][:100]}")