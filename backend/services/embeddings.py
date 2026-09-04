"""
embeddings.py
-------------
Talks to your local Ollama server to turn text into vectors (embeddings).

Nothing here calls the internet — Ollama runs entirely on your machine
at http://localhost:11434 once you've done `ollama pull nomic-embed-text`.
"""

import requests
from core.config import OLLAMA_BASE_URL, EMBED_MODEL


class OllamaConnectionError(Exception):
    """Raised when Ollama isn't running or the model isn't pulled."""
    pass


def get_embedding(text: str) -> list[float]:
    """
    Returns a single embedding vector for the given text.
    """
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except requests.exceptions.ConnectionError as e:
        raise OllamaConnectionError(
            f"Could not reach Ollama at {OLLAMA_BASE_URL}. Is it running? Try `ollama serve`."
        ) from e
    except requests.exceptions.HTTPError as e:
        raise OllamaConnectionError(
            f"Ollama returned an error for model '{EMBED_MODEL}'. "
            f"Did you run `ollama pull {EMBED_MODEL}`? Details: {e}"
        ) from e


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Ollama's /api/embeddings endpoint handles one prompt at a time,
    so we simply loop. For a resume project this is plenty fast for
    a handful of papers — batching/parallelism is listed as a possible
    future optimization.
    """
    return [get_embedding(t) for t in texts]