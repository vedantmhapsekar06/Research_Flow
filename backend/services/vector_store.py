"""
vector_store.py
----------------
Thin wrapper around ChromaDB — our local vector database.

ChromaDB persists to disk under backend/data/chroma_db, so your
knowledge base survives restarts. We only ever query embeddings we
generated ourselves via Ollama (see embeddings.py), so this file
never talks to the internet either.
"""

import logging
from core.config import CHROMA_DIR, COLLECTION_NAME
import chromadb
from chromadb.config import Settings

# Known ChromaDB 0.5.x issue: its telemetry code is incompatible with the
# installed posthog version and logs a "Failed to send telemetry event"
# error on every operation. It's purely cosmetic (nothing is actually
# broken), but it clutters output, so we silence that specific logger.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR),
    settings=Settings(anonymized_telemetry=False),
)
_collection = _client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def add_chunks(chunks: list[dict], embeddings: list[list[float]]) -> None:
    """
    chunks: list of chunk dicts (see chunker.py) — each needs a unique chunk_id.
    embeddings: parallel list of embedding vectors, same order as chunks.
    """
    if not chunks:
        return

    _collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[
            {
                "paper_id": c["paper_id"],
                "paper_name": c["paper_name"],
                "page": c["page"],
                "section": c["section"] or "",
            }
            for c in chunks
        ],
    )


def semantic_search(query_embedding: list[float], top_k: int) -> list[dict]:
    """
    Returns the top_k most semantically similar chunks to the query.
    Each result: {"chunk_id", "text", "metadata", "distance"}
    """
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    if not results["ids"] or not results["ids"][0]:
        return []

    out = []
    for i in range(len(results["ids"][0])):
        out.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return out


def delete_paper(paper_id: str) -> None:
    """
    Removes every chunk belonging to a paper. Called when the user
    deletes an already-processed paper, so it can never influence
    future answers again.
    """
    _collection.delete(where={"paper_id": paper_id})


def count() -> int:
    return _collection.count()