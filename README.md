# ResearchFlow

A fully local, hybrid-retrieval RAG chatbot for querying multiple research papers — no paid APIs, no cloud dependency. Upload PDFs, ask natural-language questions, get answers grounded in the actual paper text with page-level citations.

Built as a portfolio project to explore retrieval-augmented generation end-to-end: chunking strategy, hybrid semantic + keyword search, citation grounding, and hallucination resistance — all running on a local LLM via Ollama.

---

## Why local?

Every part of this stack runs on your own machine, for free:
- **LLM + embeddings**: [Ollama](https://ollama.com) (`llama3.2` for chat, `nomic-embed-text` for embeddings)
- **Vector store**: ChromaDB (persisted to disk)
- **Keyword search**: BM25 (`rank_bm25`)
- **PDF parsing**: PyMuPDF

No API keys, no per-request cost, no data leaving your machine.

---

## Architecture

```
PDF Upload → Page-level text extraction → Overlapping chunking (220 words, 40-word overlap)
    → Embedding (Ollama / nomic-embed-text) → ChromaDB vector store
                                             → BM25 keyword index

User Question → Follow-up resolution (LLM condenses conversational context into a standalone query)
    → Hybrid retrieval (semantic search + BM25, weighted score fusion: 0.6 / 0.4)
    → Top-K chunks passed to LLM with strict grounding instructions
    → Answer + page-cited sources returned to UI
```

**Backend**: FastAPI, serving both the REST API and the static frontend from a single process.
**Frontend**: Single-file HTML/CSS/JS — no framework, no build step.

---

## Features

- Multi-PDF upload with async processing (extract → chunk → embed → index)
- Hybrid retrieval: dense semantic search (catches paraphrasing) + BM25 (catches exact terms, acronyms, numbers)
- Page-level source citations with relevance scores, clickable to highlight the supporting excerpt
- Conversational memory — follow-up questions are automatically resolved into standalone queries before retrieval
- Explicit hallucination resistance: the system is instructed to say "not found in the papers" rather than answer from the LLM's own training knowledge, even for well-known papers

---

## Setup

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.com) installed and running.

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python main.py                # runs on http://localhost:8500
```

Open `http://localhost:8500/` — upload a PDF, wait for it to process, and start asking questions.

---

## Testing methodology

Rather than eyeballing a few chat responses, I built a **six-category test suite** targeting the specific failure modes RAG systems are known for, and ran it against multiple paper sets — including well-known papers (*Attention Is All You Need*, *XGBoost*) specifically chosen because a local LLM is likely to have memorized real facts about them, making grounding failures easier to surface.

| # | Category | What it verifies | Result |
|---|---|---|---|
| 1 | Basic retrieval | Simple single-paper factual questions | ✅ Pass |
| 2 | Exact factual retrieval | Precise numbers survive chunking + retrieval | ✅ Pass |
| 3 | Cross-paper retrieval | A single answer correctly synthesizes chunks from two different PDFs | ✅ Pass |
| 4 | Follow-up questions | Conversational memory resolves pronouns/context across turns | ✅ Pass |
| 5 | Citation accuracy | Cited page actually contains the claimed fact — tested 3x independently | ✅ Pass (3/3) |
| 6 | Hallucination resistance | System refuses to answer fabricated premises, even when tempting real details are nearby in the same paper | ✅ Pass (2/2 traps, different papers) |

### A real debugging story worth noting

Early testing surfaced a subtle but important failure: asked about a fabricated topic ("does this paper mention using Redis for caching?"), the system correctly said no — but still displayed 6 source citations, as if the answer were grounded. Digging in:

1. **First attempt**: added a fixed relevance-score threshold to filter weak retrieval matches. This didn't work — for a small, topically-narrow corpus, semantic similarity for *irrelevant* queries sits in nearly the same numeric range as similarity for *relevant* ones (a known embedding-model behavior called anisotropy). No single threshold could separate the two distributions.
2. **Second attempt**: moved the "is this actually relevant" decision from a numeric score to the LLM itself, via an explicit `NOT_FOUND:` marker convention in the system prompt. This worked for the case that surfaced it — but a later test showed the LLM would sometimes reach the *correct* conclusion while phrasing it differently ("There is no information about...") instead of using the exact marker string, silently bypassing the check.
3. **Final fix**: broadened the detection to catch common natural-language refusal phrasings in addition to the strict marker, rather than relying on exact string matching.

This ended up being a good lesson in a subtlety of building grounded LLM systems: correct LLM *reasoning* and reliable *machine-readable signaling* of that reasoning are two separate problems, and testing needs to catch both.

---

## Known limitations

- **Table/figure structure is lost during PDF text extraction.** PyMuPDF flattens tables into linear text, so a dense results table can become hard for a small local model to parse correctly (e.g. distinguishing which number belongs to which row). This is a general, unsolved-in-the-small problem for naive PDF-to-text RAG pipelines — production systems typically use dedicated table-extraction models.
- **Recall on broad, comparative questions is imperfect.** A question like "compare X and Y's approaches" may not retrieve every relevant chunk, since relevant content can be spread thinly across many non-adjacent chunks. This is normal hybrid-retrieval behavior, not a bug — no retrieval system has perfect recall.
- **Small local LLM (3B parameters)**: `llama3.2` handles single- and multi-source synthesis well in testing, but is more likely than a larger model to need explicit, carefully-worded instructions to stay reliably grounded. Swapping to a larger local model (e.g. `llama3.1:8b`) is a one-line config change if needed.

---

## Project structure

```
backend/
├── main.py                  # FastAPI entrypoint, serves API + static frontend
├── core/config.py           # All settings: models, chunk sizes, retrieval weights
├── models/schemas.py        # Pydantic request/response models
├── services/
│   ├── pdf_processor.py     # PDF → per-page text extraction
│   ├── chunker.py           # Page text → overlapping chunks
│   ├── embeddings.py        # Ollama embedding client
│   ├── vector_store.py      # ChromaDB wrapper
│   ├── storage.py           # JSON-backed paper registry + chunk store
│   ├── retriever.py         # Hybrid (semantic + BM25) retrieval
│   ├── llm.py                # Ollama chat client + query condensing
│   └── chat_service.py      # Orchestrates retrieval + generation + memory
└── api/
    ├── papers.py             # Upload / list / delete / process endpoints
    └── chat.py                # Chat endpoint
frontend/
└── index.html                 # Single-file chat UI
```

---

## Possible future improvements

- Dedicated table extraction for PDFs with dense results tables
- Cross-encoder reranking stage for improved recall on broad/comparative questions
- Persist conversation history to disk (currently in-memory, resets on server restart)
- Embedded PDF viewer in the sources panel (currently shows excerpt snippets only)
```
