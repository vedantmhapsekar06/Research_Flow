"""
llm.py
------
Talks to your local Ollama server to generate chat responses.

Two responsibilities:
1. chat_completion() — the actual answer-generation call to llama3.2
2. condense_question() — rewrites a follow-up question ("what about
   page 2?") into a standalone question ("what does page 2 of the
   hybrid RAG paper discuss?") using the conversation history, so the
   retriever gets something it can actually search for.
"""

import requests
from core.config import OLLAMA_BASE_URL, CHAT_MODEL


class OllamaConnectionError(Exception):
    pass


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
    except requests.exceptions.HTTPError as e:
        raise OllamaConnectionError(
            f"Ollama returned an error for model '{CHAT_MODEL}'. "
            f"Did you run `ollama pull {CHAT_MODEL}`? Details: {e}"
        ) from e


def condense_question(history: list[dict], question: str) -> str:
    """
    Given recent conversation history and a new (possibly ambiguous)
    follow-up question, asks the LLM to rewrite it as a standalone
    question that makes sense without the prior context.

    If there's no history yet, we skip the extra LLM call entirely —
    the first question in a conversation is always already standalone.
    """
    if not history:
        return question

    history_text = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)

    prompt = (
        "Given the conversation history and a follow-up question, rewrite the "
        "follow-up question to be a standalone question that includes all "
        "necessary context. If the follow-up question is already standalone, "
        "return it unchanged. Reply with ONLY the rewritten question, nothing else.\n\n"
        f"Conversation history:\n{history_text}\n\n"
        f"Follow-up question: {question}\n\n"
        "Standalone question:"
    )

    rewritten = chat_completion(
        [{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return rewritten.strip().strip('"')