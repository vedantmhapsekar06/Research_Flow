"""End-to-end through the real FastAPI app, real PDFs, real Chroma + BM25 — only Ollama is faked."""
import fitz
import pytest
from fastapi.testclient import TestClient

from main import app
from services import storage

client = TestClient(app)


def make_pdf(pages: list[str]) -> bytes:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(40, 40, 550, 800), text, fontsize=9)
    data = doc.tobytes()
    doc.close()
    return data


PAPER_A = [
    "Introduction\n" + "PaperQA uses hybrid retrieval with BM25 and dense vectors for scientific question answering. " * 12,
    "Results\n" + "The LitQA benchmark contains 52 questions and citation accuracy reached 69 percent. " * 12,
]
PAPER_B = [
    "Introduction\n" + "Vanilla RAG baselines use only dense embeddings with no keyword index at all. " * 12,
]


_PDF_CACHE: dict[int, bytes] = {}


def upload(name, pages):
    # PyMuPDF embeds a fresh document ID each time, so build bytes once per content
    # (otherwise "the same file" would hash differently and defeat the dedupe test).
    data = _PDF_CACHE.setdefault(id(pages), make_pdf(pages))
    return client.post("/api/papers/upload", files={"file": (name, data, "application/pdf")})


def test_upload_process_and_list(clean_state):
    r = upload("a.pdf", PAPER_A)
    assert r.status_code == 200 and r.json()["status"] == "uploaded"
    res = client.post("/api/papers/process").json()
    assert len(res["processed"]) == 1 and res["failed"] == []
    papers = client.get("/api/papers").json()
    assert papers[0]["status"] == "processed" and papers[0]["chunk_count"] > 0 and papers[0]["pages"] == 2


def test_rejects_non_pdf_and_fake_pdf_and_empty(clean_state):
    assert client.post("/api/papers/upload", files={"file": ("a.txt", b"hello", "text/plain")}).status_code == 400
    assert client.post("/api/papers/upload", files={"file": ("a.pdf", b"not really a pdf", "application/pdf")}).status_code == 400
    assert client.post("/api/papers/upload", files={"file": ("a.pdf", b"", "application/pdf")}).status_code == 400
    assert client.get("/api/papers").json() == []


def test_duplicate_upload_returns_existing_record(clean_state):
    a = upload("a.pdf", PAPER_A).json()
    b = upload("copy.pdf", PAPER_A).json()
    assert a["paper_id"] == b["paper_id"]
    assert len(client.get("/api/papers").json()) == 1


def test_size_limit(clean_state, monkeypatch):
    import api.papers as p
    monkeypatch.setattr(p, "_MAX_BYTES", 100)
    assert upload("a.pdf", PAPER_A).status_code == 413
    assert client.get("/api/papers").json() == []


def test_process_is_not_repeated_and_claims_atomically(clean_state):
    upload("a.pdf", PAPER_A)
    claimed1 = storage.claim_papers_for_processing()
    claimed2 = storage.claim_papers_for_processing()
    assert len(claimed1) == 1 and claimed2 == []           # second caller gets nothing


def test_transient_failure_is_retried_permanent_is_not(clean_state, monkeypatch):
    import api.papers as p
    from services.errors import OllamaConnectionError

    upload("a.pdf", PAPER_A)

    def boom(texts):
        raise OllamaConnectionError("Ollama down")
    monkeypatch.setattr(p, "get_embeddings_batch", boom)
    res = client.post("/api/papers/process").json()
    assert len(res["failed"]) == 1
    paper = client.get("/api/papers").json()[0]
    assert paper["status"] == "failed" and "Ollama" in paper["error"]
    assert storage.load_chunks() == {}                     # nothing half-indexed

    # Ollama comes back -> a plain /process call recovers it (old code skipped 'failed' forever).
    from tests.conftest import fake_embed
    monkeypatch.setattr(p, "get_embeddings_batch", lambda t: [fake_embed(x) for x in t])
    res = client.post("/api/papers/process").json()
    assert len(res["processed"]) == 1

    # A permanent failure (image-only / textless PDF) is NOT retried.
    blank = fitz.open(); blank.new_page(); data = blank.tobytes()
    client.post("/api/papers/upload", files={"file": ("blank.pdf", data, "application/pdf")})
    assert len(client.post("/api/papers/process").json()["failed"]) == 1
    assert client.post("/api/papers/process").json() == {"processed": [], "failed": []}


def test_reset_stale_processing_on_startup(clean_state):
    upload("a.pdf", PAPER_A)
    storage.claim_papers_for_processing()                   # simulates crash mid-processing
    assert storage.reset_stale_processing() == 1
    assert client.get("/api/papers").json()[0]["status"] == "uploaded"


def test_delete_blocked_while_processing_then_cleans_everything(clean_state):
    pid = upload("a.pdf", PAPER_A).json()["paper_id"]
    storage.claim_papers_for_processing()
    assert client.delete(f"/api/papers/{pid}").status_code == 409
    storage.update_paper(pid, status="uploaded")
    client.post("/api/papers/process")
    assert client.delete(f"/api/papers/{pid}").status_code == 200
    assert storage.load_chunks() == {} and client.get("/api/papers").json() == []
    assert client.delete(f"/api/papers/{pid}").status_code == 404


def test_update_paper_does_not_resurrect_deleted_record(clean_state):
    assert storage.update_paper("ghost", status="processed") is False
    assert "ghost" not in storage.load_papers()


# ------------------------------------------------------------- retrieval
def test_hybrid_search_finds_right_page_and_scores_are_0_to_1(clean_state):
    from services.retriever import hybrid_search
    upload("a.pdf", PAPER_A); client.post("/api/papers/process")
    res = hybrid_search("How many questions are in the LitQA benchmark?")
    assert res and res[0]["page"] == 2
    assert all(0.0 <= r["score"] <= 1.0 for r in res)


def test_paper_filter_is_applied_inside_the_search(clean_state):
    from services.retriever import hybrid_search
    a = upload("a.pdf", PAPER_A).json()["paper_id"]
    b = upload("b.pdf", PAPER_B).json()["paper_id"]
    client.post("/api/papers/process")
    res = hybrid_search("dense embeddings retrieval", paper_id=b)
    assert res and {r["paper_id"] for r in res} == {b}


def test_empty_corpus_skips_embedding_and_returns_nothing(clean_state, monkeypatch):
    import services.retriever as retr
    monkeypatch.setattr(retr, "get_embedding", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not embed")))
    assert retr.hybrid_search("anything") == []


def test_bm25_cache_invalidates_after_new_upload(clean_state):
    from services.retriever import hybrid_search
    upload("a.pdf", PAPER_A); client.post("/api/papers/process")
    hybrid_search("LitQA")                                  # warms cache
    upload("b.pdf", PAPER_B); client.post("/api/papers/process")
    res = hybrid_search("Vanilla RAG baselines keyword index")
    assert any(r["paper_name"] == "b.pdf" for r in res)


def test_relevance_floor_drops_everything_when_nothing_matches(clean_state, monkeypatch):
    import services.retriever as retr
    upload("a.pdf", PAPER_A); client.post("/api/papers/process")
    monkeypatch.setattr(retr, "SEMANTIC_ABS_MIN", 0.99)
    monkeypatch.setattr(retr, "KEYWORD_ABS_MIN", 1e9)
    assert retr.hybrid_search("completely unrelated quantum gravity") == []


# ------------------------------------------------------------------ chat
def _setup_chat():
    upload("a.pdf", PAPER_A); client.post("/api/papers/process")


def test_chat_returns_only_cited_sources_renumbered(clean_state, fake_ollama):
    _setup_chat()
    fake_ollama["reply"] = "The benchmark has 52 questions [Source 2]."
    r = client.post("/api/chat", json={"question": "How many questions in LitQA?"}).json()
    assert len(r["sources"]) == 1
    assert r["answer"] == "The benchmark has 52 questions [Source 1]."
    assert r["sources"][0]["score"] is not None            # UI relevance bar needs this


def test_chat_not_found_has_no_sources_and_is_not_fed_back_as_history(clean_state, fake_ollama):
    _setup_chat()
    fake_ollama["reply"] = "NOT_FOUND: Redis is never mentioned."
    r = client.post("/api/chat", json={"question": "Does it use Redis for LitQA caching?"}).json()
    assert r["sources"] == [] and not r["answer"].startswith("NOT_FOUND")

    fake_ollama["reply"] = "52 [Source 1]."
    client.post("/api/chat", json={"question": "How many questions in LitQA?"})
    sent = fake_ollama["calls"][-1]
    assert all("Redis" not in m["content"] for m in sent)   # refusal excluded from model context
    # ...but the UI-restore history still has both turns.
    hist = client.get("/api/chat/history").json()["history"]
    assert len(hist) == 4


def test_chat_valid_answer_with_negative_wording_keeps_sources(clean_state, fake_ollama):
    _setup_chat()
    fake_ollama["reply"] = "The paper does not mention Redis but reports 52 questions [Source 1]."
    r = client.post("/api/chat", json={"question": "How many questions in LitQA?"}).json()
    assert len(r["sources"]) == 1


def test_chat_uncited_answer_falls_back_to_retrieved_sources(clean_state, fake_ollama):
    _setup_chat()
    fake_ollama["reply"] = "There are 52 questions."
    r = client.post("/api/chat", json={"question": "How many questions in LitQA?"}).json()
    assert len(r["sources"]) >= 1


def test_history_window_is_measured_in_turns(clean_state, fake_ollama):
    import services.chat_service as cs
    _setup_chat()
    fake_ollama["reply"] = "52 [Source 1]."
    for i in range(cs.MAX_HISTORY_TURNS + 3):
        client.post("/api/chat", json={"question": f"How many questions in LitQA? #{i}"})
    assert len(client.get("/api/chat/history").json()["history"]) == cs.MAX_HISTORY_TURNS * 2
    sent = fake_ollama["calls"][-1]
    # system + (MAX_HISTORY_TURNS-1 prior grounded turns kept at call time)*2 + current user turn
    assert len([m for m in sent if m["role"] == "assistant"]) == cs.MAX_HISTORY_TURNS


def test_stale_citations_are_stripped_from_history_sent_to_model(clean_state, fake_ollama):
    _setup_chat()
    fake_ollama["reply"] = "52 questions [Source 2]."
    client.post("/api/chat", json={"question": "How many questions in LitQA?"})
    client.post("/api/chat", json={"question": "And the accuracy in LitQA?"})
    prior = [m for m in fake_ollama["calls"][-1] if m["role"] == "assistant"]
    assert prior and all("[Source" not in m["content"] for m in prior)


def test_chat_empty_corpus_message(clean_state, fake_ollama):
    r = client.post("/api/chat", json={"question": "anything?"}).json()
    assert r["sources"] == [] and "couldn't find" in r["answer"]


def test_chat_ollama_down_returns_503_not_500(clean_state, monkeypatch):
    import services.chat_service as cs
    from services.errors import OllamaConnectionError
    _setup_chat()
    monkeypatch.setattr(cs, "chat_completion", lambda *a, **k: (_ for _ in ()).throw(OllamaConnectionError("down")))
    assert client.post("/api/chat", json={"question": "How many questions in LitQA?"}).status_code == 503


def test_chat_validation(clean_state):
    assert client.post("/api/chat", json={"question": ""}).status_code == 422
    assert client.post("/api/chat", json={"question": "   "}).status_code == 400


# ------------------------------------------------------------------ misc
def test_cors_is_not_wildcard():
    r = client.options("/api/papers", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"})
    assert r.headers.get("access-control-allow-origin") != "*"
    assert "evil.example" not in r.headers.get("access-control-allow-origin", "")
    ok = client.get("/api/health", headers={"Origin": "http://localhost:8500"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:8500"
