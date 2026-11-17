"""
embeddings.py
-------------
Talks to your local Ollama server to turn text into vectors (embeddings).

Nothing here calls the internet — Ollama runs entirely on your machine
at http://localhost:11434 once you've done `ollama pull nomic-embed-text`.
"""

import requests
from core.config import OLLAMA_BASE_URL, EMBED_MODEL, EMBED_USE_TASK_PREFIXES
from services.errors import OllamaConnectionError  # re-exported for callers

__all__ = ["OllamaConnectionError", "get_embedding", "get_embeddings_batch"]

# One pooled HTTP session: avoids a new TCP connection per chunk when a paper
# has hundreds of chunks.
_session = requests.Session()


def _with_task_prefix(text: str, is_query: bool) -> str:
    """nomic-embed-text expects 'search_query: ' / 'search_document: ' prefixes."""
    if EMBED_USE_TASK_PREFIXES and "nomic" in EMBED_MODEL.lower():
        return ("search_query: " if is_query else "search_document: ") + text
    return text


def get_embedding(text: str, is_query: bool = False) -> list[float]:
    """
    Returns a single embedding vector for the given text.
    is_query=True for search queries, False (default) for stored documents.
    """
    try:
        response = _session.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": _with_task_prefix(text, is_query)},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except requests.exceptions.ConnectionError as e:
        raise OllamaConnectionError(
            f"Could not reach Ollama at {OLLAMA_BASE_URL}. Is it running? Try `ollama serve`."
        ) from e
    except requests.exceptions.Timeout as e:
        raise OllamaConnectionError(
            f"Ollama timed out while embedding with '{EMBED_MODEL}'."
        ) from e
    except requests.exceptions.HTTPError as e:
        raise OllamaConnectionError(
            f"Ollama returned an error for model '{EMBED_MODEL}'. "
            f"Did you run `ollama pull {EMBED_MODEL}`? Details: {e}"
        ) from e


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Embeds document chunks. Ollama's /api/embeddings endpoint handles one
    prompt at a time, so we loop over a pooled session.
    """
    return [get_embedding(t, is_query=False) for t in texts]
