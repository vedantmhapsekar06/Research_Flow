"""
llm.py
------
Talks to your local Ollama server to generate chat responses.

Two responsibilities:
1. chat_completion() — the actual answer-generation call to llama3.2
2. condense_question() — rewrites a follow-up question ("what about
   page 2?") into a standalone question using the conversation history, so
   the retriever gets something it can actually search for.
"""

import re
import requests
from core.config import OLLAMA_BASE_URL, CHAT_MODEL, CONDENSE_MAX_ANSWER_CHARS
from services.errors import OllamaConnectionError  # re-exported for callers

__all__ = ["OllamaConnectionError", "chat_completion", "condense_question"]


def chat_completion(messages: list[dict], temperature: float = 0.2) -> str:
    """
    messages: list of {"role": "system"|"user"|"assistant", "content": "..."}
    Returns the model's reply text.
    """
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": CHAT_MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError as e:
        raise OllamaConnectionError(
            f"Could not reach Ollama at {OLLAMA_BASE_URL}. Is it running?"
        ) from e
    except requests.exceptions.Timeout as e:
        raise OllamaConnectionError(
            f"Ollama timed out while generating with '{CHAT_MODEL}'."
        ) from e
    except requests.exceptions.HTTPError as e:
        raise OllamaConnectionError(
            f"Ollama returned an error for model '{CHAT_MODEL}'. "
            f"Did you run `ollama pull {CHAT_MODEL}`? Details: {e}"
        ) from e


def sanitize_rewrite(original: str, rewritten: str) -> str:
    """
    Small models sometimes ignore "reply with ONLY the rewritten question" and
    answer it, explain themselves, or return nothing. Using that text as the
    search query would silently wreck retrieval, so anything that doesn't look
    like a single short question falls back to the user's original wording.
    """
    text = (rewritten or "").strip()
    text = re.sub(r"^(standalone question|rewritten question)\s*:\s*", "", text, flags=re.I)
    text = text.strip().strip('"').strip("'").strip()

    if not text:
        return original
    if "\n" in text:                                   # multi-line = explanation / answer
        return original
    if len(text) > max(300, 4 * len(original)):        # far too long to be a rewrite
        return original
    return text


def condense_question(history: list[dict], question: str) -> str:
    """
    Given recent conversation history ([{"role","content"}, ...]) and a new
    (possibly ambiguous) follow-up question, asks the LLM to rewrite it as a
    standalone question that makes sense without the prior context.

    If there's no history yet, we skip the extra LLM call entirely — the first
    question in a conversation is always already standalone.
    """
    if not history:
        return question

    def _clip(turn: dict) -> str:
        content = turn["content"]
        if turn["role"] == "assistant" and len(content) > CONDENSE_MAX_ANSWER_CHARS:
            content = content[:CONDENSE_MAX_ANSWER_CHARS] + "..."
        return f"{turn['role']}: {content}"

    history_text = "\n".join(_clip(t) for t in history)

    prompt = (
        "Given the conversation history and a follow-up question, rewrite the "
        "follow-up question to be a standalone question that includes all "
        "necessary context. If the follow-up question is already standalone, "
        "return it unchanged. Do NOT answer the question. Reply with ONLY the "
        "rewritten question on a single line, nothing else.\n\n"
        f"Conversation history:\n{history_text}\n\n"
        f"Follow-up question: {question}\n\n"
        "Standalone question:"
    )

    rewritten = chat_completion(
        [{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return sanitize_rewrite(question, rewritten)
