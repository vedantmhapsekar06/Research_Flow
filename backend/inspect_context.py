"""
One-off diagnostic: prints the EXACT chunk text that gets sent to the
LLM for a given query, so we can see whether retrieval is actually
returning the right content, or whether the fact got cut awkwardly
across a chunk boundary.
"""

from services.retriever import hybrid_search
from services.chat_service import _build_context_block

query = "How many questions are in the LitQA dataset, and what percentage of PaperQA's citations were found to be hallucinated?"

chunks = hybrid_search(query)
print(f"Retrieved {len(chunks)} chunks\n")

context_block = _build_context_block(chunks)
print(context_block)