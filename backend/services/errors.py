"""
errors.py
---------
Shared exception types. Previously llm.py and embeddings.py each defined
their own `OllamaConnectionError`, so `except OllamaConnectionError` in one
module silently failed to catch the other module's error. One class, one
import path.
"""


class OllamaConnectionError(Exception):
    """Raised when Ollama is unreachable, times out, or rejects a request."""
