"""
Test harness. Everything runs WITHOUT Ollama: embeddings are a deterministic
hashed bag-of-words vector and the chat model is a stub the tests can script.
Storage is redirected to a temp directory before any app module is imported.
"""
import hashlib
import math
import os
import re
import sys
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="researchflow_test_")
os.environ["RESEARCHFLOW_DATA_DIR"] = _TMP
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

DIM = 256


def fake_embed(text: str, is_query: bool = False):
    vec = [0.0] * DIM
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@pytest.fixture(autouse=True)
def fake_ollama(monkeypatch):
    import services.embeddings as emb
    import services.retriever as retr
    import api.papers as papers_api
    import services.chat_service as chat_service

    monkeypatch.setattr(emb, "get_embedding", fake_embed)
    monkeypatch.setattr(retr, "get_embedding", fake_embed)
    # The toy hashed embeddings score lower than real nomic vectors and the
    # corpora here are tiny (BM25 IDF ~ 0), so disable the floors by default;
    # test_relevance_floor_* re-enables them explicitly.
    monkeypatch.setattr(retr, "SEMANTIC_ABS_MIN", 0.0)
    monkeypatch.setattr(retr, "KEYWORD_ABS_MIN", 0.0)
    monkeypatch.setattr(papers_api, "get_embeddings_batch",
                        lambda texts: [fake_embed(t) for t in texts])

    state = {"reply": "NOT_FOUND: stub", "calls": [], "condense": None}

    def fake_chat(messages, temperature=0.2):
        state["calls"].append(messages)
        return state["reply"]

    monkeypatch.setattr(chat_service, "chat_completion", fake_chat)
    monkeypatch.setattr(chat_service, "condense_question",
                        lambda history, q: state["condense"] or q)
    return state


@pytest.fixture()
def clean_state():
    """Wipe all papers/chunks/vectors/sessions between tests."""
    from services import storage, vector_store
    import services.chat_service as cs
    for pid in list(storage.load_papers()):
        vector_store.delete_paper(pid)
    storage.save_papers({})
    storage.save_chunks({})
    cs._sessions.clear()
    yield
