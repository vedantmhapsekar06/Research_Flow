"""
chat_service.py
---------------
Orchestrates a single chat turn end-to-end.

Key design decision: instead of trying to detect "irrelevant retrieval"
with a numeric score threshold (which testing showed doesn't work
reliably for a small, topically-narrow corpus — see retriever.py's
docstring), we ask the LLM itself to explicitly flag when the
retrieved context doesn't actually answer the question.

We check for this in two ways:
1. A strict marker string (NOT_FOUND_MARKER) the system prompt asks
   the LLM to start its response with.
2. A fallback list of common "I don't know" phrasings, since small
   local models don't always follow the marker instruction exactly
   even when their underlying reasoning is correct — testing showed
   the model would often correctly conclude "not in the papers" but
   phrase it in its own words instead of the literal marker string.
   Catching both means a correctly-reasoned refusal doesn't
   accidentally still show sources just because of wording.
"""

from core.config import MAX_HISTORY_TURNS
from services.retriever import hybrid_search
from services.llm import chat_completion, condense_question

_sessions: dict[str, list[dict]] = {}

NOT_FOUND_MARKER = "NOT_FOUND:"

# Fallback phrases that indicate the LLM concluded the answer isn't
# present, even if it didn't use the exact NOT_FOUND_MARKER string.
# Checked as a case-insensitive substring match against the full answer.
NOT_FOUND_PHRASES = [
    "no information about",
    "no info about",
    "does not mention",
    "do not mention",
    "doesn't mention",
    "don't mention",
    "not mentioned in",
    "no mention of",
    "cannot answer",
    "can't answer",
    "not contain",
    "not explicitly",
    "does not contain",
    "not found in",
    "no relevant information",
]

SYSTEM_PROMPT = (
    "You are ResearchFlow, an assistant that answers questions strictly using "
    "the provided excerpts from research papers. Rules:\n"
    "1. Only use information found in the provided context excerpts below — "
    "even if you recognize the paper or believe you already know the answer "
    "from your own training. Your own prior knowledge is NOT a valid source.\n"
    "2. Questions may require combining facts from MULTIPLE different Source "
    "excerpts (e.g. one number from Source 2 and another from Source 5). "
    "Read ALL provided excerpts carefully before deciding whether the answer "
    "is present — do not conclude something is missing just because one "
    "single excerpt doesn't contain the full answer by itself.\n"
    f"3. Only if NONE of the excerpts, even combined, contain information "
    f"relevant to the question, start your ENTIRE response with exactly "
    f"'{NOT_FOUND_MARKER}' followed by a brief explanation. This should be "
    "rare — most retrieved excerpts contain at least partial relevant info.\n"
    "4. When you state a fact, cite the specific Source number it came from. "
    "Only cite a source if that source's text actually contains the fact — "
    "never attach a citation to a source that doesn't support it.\n"
    "5. Be concise and precise — this is for a researcher, not a general audience."
)


def _get_history(session_id: str) -> list[dict]:
    return _sessions.setdefault(session_id, [])


def _build_context_block(chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(
            f"[Source {i}] (Paper: {c['paper_name']}, Page: {c['page']})\n{c['text']}"
        )
    return "\n\n".join(blocks)


def _is_not_found_answer(raw_answer: str) -> bool:
    """
    Returns True if the LLM's answer indicates the context didn't
    actually contain relevant information — checking both the strict
    marker and common natural-language "I don't know" phrasings.
    """
    text = raw_answer.strip()
    if text.startswith(NOT_FOUND_MARKER):
        return True

    lowered = text.lower()
    return any(phrase in lowered for phrase in NOT_FOUND_PHRASES)


def answer_question(question: str, session_id: str = "default") -> dict:
    """
    Returns {"answer": str, "sources": [ {paper_id, paper_name, page, section, snippet}, ... ]}
    """
    history = _get_history(session_id)

    standalone_query = condense_question(history, question)
    chunks = hybrid_search(standalone_query)

    if not chunks:
        # Retrieval found literally nothing at all (empty corpus, etc.)
        answer = (
            "I couldn't find anything relevant in the uploaded papers to "
            "answer that. Try rephrasing, or make sure the relevant paper "
            "has been uploaded and processed."
        )
        sources = []
    else:
        context_block = _build_context_block(chunks)
        user_turn = (
            f"Context excerpts:\n\n{context_block}\n\n"
            f"Question: {question}"
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history[-MAX_HISTORY_TURNS:])
        messages.append({"role": "user", "content": user_turn})

        raw_answer = chat_completion(messages)

        if _is_not_found_answer(raw_answer):
            # The LLM judged the retrieved chunks don't actually answer
            # the question — even though something was retrieved. Strip
            # the marker if present, and correctly show NO sources,
            # since none of the retrieved chunks were actually used.
            answer = raw_answer.strip()
            if answer.startswith(NOT_FOUND_MARKER):
                answer = answer[len(NOT_FOUND_MARKER):].strip()
            sources = []
        else:
            answer = raw_answer
            sources = [
                {
                    "paper_id": c["paper_id"],
                    "paper_name": c["paper_name"],
                    "page": c["page"],
                    "section": c["section"],
                    "snippet": c["text"][:220] + ("..." if len(c["text"]) > 220 else ""),
                }
                for c in chunks
            ]

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    _sessions[session_id] = history[-(MAX_HISTORY_TURNS * 2):]

    return {"answer": answer, "sources": sources}


def get_history(session_id: str = "default") -> list[dict]:
    return _sessions.get(session_id, [])