"""
chat_service.py
---------------
Orchestrates a single chat turn end-to-end.

Grounding strategy: retrieval hands the LLM a numbered set of excerpts; the
LLM must either answer with [Source N] citations or begin with NOT_FOUND:.

Logic fixes over the first version:
- Not-found is decided ONLY by the explicit NOT_FOUND: marker at the START of
  the reply (plus a narrow refusal-opening check when the reply has no
  citations at all). The old whole-answer substring match on phrases like
  "does not mention" threw away sources for perfectly valid answers such as
  "Paper A does not mention X, but Paper B uses Y [Source 3]".
- Returned sources are the chunks the answer actually CITED, and the
  [Source N] markers in the answer text are renumbered to match the returned
  list (the UI numbers source cards 1..n in list order). Previously all
  retrieved chunks were returned, cited or not.
- History is measured in TURNS everywhere (it was 6 messages in one place and
  12 in another), and ungrounded turns/old citation markers are kept out of
  the model's context.
"""

import re
from threading import Lock

from core.config import MAX_HISTORY_TURNS
from services.retriever import hybrid_search
from services.llm import chat_completion, condense_question

# session_id -> list of turns: {"question", "answer", "grounded"}
_sessions: dict[str, list[dict]] = {}
_sessions_lock = Lock()

NOT_FOUND_MARKER = "NOT_FOUND:"
NO_CHUNKS_ANSWER = (
    "I couldn't find anything relevant in the uploaded papers to "
    "answer that. Try rephrasing, or make sure the relevant paper "
    "has been uploaded and processed."
)

SYSTEM_PROMPT = (
    "You are ResearchFlow, an assistant that answers questions strictly using "
    "the provided excerpts from research papers. Rules:\n"
    "1. Only use information found in the provided context excerpts — even if "
    "you recognize the paper or believe you already know the answer from your "
    "own training. Your own prior knowledge, and earlier turns of the "
    "conversation, are NOT valid sources of facts.\n"
    "2. Questions may require combining facts from MULTIPLE different Source "
    "excerpts (e.g. one number from Source 2 and another from Source 5). Read "
    "ALL excerpts carefully before deciding the answer is missing.\n"
    f"3. If the excerpts, even combined, do not contain what is needed to "
    f"answer (including when the question assumes something the excerpts do "
    f"not state), begin your ENTIRE response with exactly '{NOT_FOUND_MARKER}' "
    "followed by one short sentence saying what is missing. Do not guess.\n"
    "4. Support every factual statement with its source number in the form "
    "[Source N] (or [Source 2, Source 5]). Only cite a source whose text "
    "actually contains the fact.\n"
    "5. Be concise and precise — this is for a researcher, not a general audience."
)

# ---------------------------------------------------------------------------
# Not-found detection
# ---------------------------------------------------------------------------
_NOT_FOUND_RE = re.compile(r"^\W*NOT[_\s-]?FOUND\b\W*", re.IGNORECASE)

# Only consulted when the reply has NO citations at all and only against its
# opening words — a last resort for models that refuse without the marker.
_REFUSAL_OPENINGS = (
    "i cannot", "i can't", "i could not", "i couldn't", "i do not", "i don't",
    "there is no", "there's no", "no information", "the excerpts do not",
    "the excerpts don't", "the context does not", "the context doesn't",
    "the provided", "not found",
)

# ---------------------------------------------------------------------------
# Citation parsing
# ---------------------------------------------------------------------------
# [Source 2]  (Source 2)  [Source 2, Source 5]  [Sources 2 and 5]  [Source 2: Paper A, p.4]
_CITE_GROUP_RE = re.compile(r"[\[(]\s*sources?\b[^\])]*[\])]", re.IGNORECASE)
_CITE_IDS_RE = re.compile(
    r"sources?\s*#?\s*(\d+(?:\s*(?:,|and|&)\s*(?:sources?\s*#?\s*)?\d+)*)",
    re.IGNORECASE,
)


def _ids_in_group(group_text: str) -> list[int]:
    m = _CITE_IDS_RE.search(group_text)
    return [int(x) for x in re.findall(r"\d+", m.group(1))] if m else []


def strip_citations(text: str) -> str:
    """Removes [Source N] markers (used for history, whose numbering is stale next turn)."""
    out = _CITE_GROUP_RE.sub(lambda m: "" if _ids_in_group(m.group(0)) else m.group(0), text)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return re.sub(r" +([.,;:])", r"\1", out).strip()


def remap_citations(answer: str, n_chunks: int) -> tuple[str, list[int]]:
    """
    Finds the [Source N] markers the model used, keeps only valid ones
    (1..n_chunks), and renumbers them 1..m in order of first appearance.

    Returns (rewritten_answer, original_indices_in_new_order) where the indices
    are 0-based positions into the retrieved chunk list.
    """
    order: list[int] = []
    for m in _CITE_GROUP_RE.finditer(answer):
        for sid in _ids_in_group(m.group(0)):
            if 1 <= sid <= n_chunks and sid not in order:
                order.append(sid)

    mapping = {orig: new for new, orig in enumerate(order, start=1)}

    def _repl(m: re.Match) -> str:
        ids = _ids_in_group(m.group(0))
        if not ids:                       # e.g. "[Source code]" — not a citation, leave alone
            return m.group(0)
        new_ids: list[int] = []
        for sid in ids:
            if sid in mapping and mapping[sid] not in new_ids:
                new_ids.append(mapping[sid])
        if not new_ids:                   # cites a source that doesn't exist — drop it
            return ""
        return "[" + ", ".join(f"Source {i}" for i in new_ids) + "]"

    rewritten = _CITE_GROUP_RE.sub(_repl, answer)
    rewritten = re.sub(r"[ \t]{2,}", " ", rewritten)
    rewritten = re.sub(r" +([.,;:])", r"\1", rewritten).strip()
    return rewritten, [o - 1 for o in order]


def is_not_found(raw_answer: str) -> bool:
    text = raw_answer.strip()
    if _NOT_FOUND_RE.match(text):
        return True
    # Fallback: refusal-style opening AND no citations anywhere in the reply.
    opening = text[:160].lower()
    has_citation = any(_ids_in_group(m.group(0)) for m in _CITE_GROUP_RE.finditer(text))
    return (not has_citation) and opening.startswith(_REFUSAL_OPENINGS)


def _clean_not_found(raw_answer: str) -> str:
    text = _NOT_FOUND_RE.sub("", raw_answer.strip(), count=1).strip()
    return text or "The uploaded papers don't contain the information needed to answer that."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_context_block(chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(
            f"[Source {i}] (Paper: {c['paper_name']}, Page: {c['page']})\n{c['text']}"
        )
    return "\n\n".join(blocks)


def _grounded_history(turns: list[dict]) -> list[dict]:
    """Flat [{"role","content"}] of the recent GROUNDED turns, stale citations removed."""
    msgs = []
    for t in turns[-MAX_HISTORY_TURNS:]:
        if not t["grounded"]:
            continue
        msgs.append({"role": "user", "content": t["question"]})
        msgs.append({"role": "assistant", "content": strip_citations(t["answer"])})
    return msgs


def _snapshot(session_id: str) -> list[dict]:
    with _sessions_lock:
        return list(_sessions.get(session_id, []))


def _record_turn(session_id: str, question: str, answer: str, grounded: bool) -> None:
    with _sessions_lock:
        turns = _sessions.setdefault(session_id, [])
        turns.append({"question": question, "answer": answer, "grounded": grounded})
        del turns[:-MAX_HISTORY_TURNS]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def answer_question(question: str, session_id: str = "default") -> dict:
    """
    Returns {"answer": str, "sources": [ {paper_id, paper_name, page, section, snippet, score}, ... ]}
    """
    turns = _snapshot(session_id)
    llm_history = _grounded_history(turns)

    standalone_query = condense_question(llm_history, question)
    chunks = hybrid_search(standalone_query)

    if not chunks:
        answer, sources, grounded = NO_CHUNKS_ANSWER, [], False
    else:
        user_turn = (
            f"Context excerpts:\n\n{_build_context_block(chunks)}\n\n"
            f"Question: {question}"
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(llm_history)
        messages.append({"role": "user", "content": user_turn})

        raw_answer = chat_completion(messages)

        if is_not_found(raw_answer):
            answer, sources, grounded = _clean_not_found(raw_answer), [], False
        else:
            answer, cited = remap_citations(raw_answer, len(chunks))
            if not cited:
                # The model answered but didn't use the [Source N] format.
                # Fall back to everything it was shown rather than show an
                # answer with no provenance at all.
                cited = list(range(len(chunks)))
            sources = [
                {
                    "paper_id": chunks[i]["paper_id"],
                    "paper_name": chunks[i]["paper_name"],
                    "page": chunks[i]["page"],
                    "section": chunks[i]["section"],
                    "snippet": chunks[i]["text"][:220] + ("..." if len(chunks[i]["text"]) > 220 else ""),
                    "score": chunks[i]["score"],
                }
                for i in cited
            ]
            grounded = True

    _record_turn(session_id, question, answer, grounded)
    return {"answer": answer, "sources": sources}


def get_history(session_id: str = "default") -> list[dict]:
    """Full display history as [{"role","content"}] — includes refusals, so the UI can restore the chat."""
    out = []
    for t in _snapshot(session_id):
        out.append({"role": "user", "content": t["question"]})
        out.append({"role": "assistant", "content": t["answer"]})
    return out
