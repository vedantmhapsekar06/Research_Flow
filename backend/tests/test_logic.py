"""Pure-logic tests: chunking, cleaning, fusion, citations, not-found, condenser guard."""
import pytest

from services.chunker import chunk_pages
from services.pdf_processor import _clean_text
from services.retriever import fuse_rankings, select_with_paper_balance
from services.chat_service import (
    is_not_found, remap_citations, strip_citations, _clean_not_found,
)
from services.llm import sanitize_rewrite


# ---------------------------------------------------------------- chunker
def _words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_chunker_overlap_and_coverage():
    pages = [{"page": 1, "text": _words(500)}]
    chunks = chunk_pages(pages, "p", "P.pdf")
    covered = set()
    for c in chunks:
        covered.update(c["text"].split())
    assert covered == set(_words(500).split())           # nothing lost
    assert chunks[0]["text"].split()[-40:] == chunks[1]["text"].split()[:40]  # 40-word overlap


def test_chunker_no_tiny_tail_chunk():
    # 230 words: naive loop would emit a 2nd chunk of 50 words that is 40 overlap + 10 new.
    chunks = chunk_pages([{"page": 1, "text": _words(230)}], "p", "P.pdf")
    assert len(chunks) == 1 and len(chunks[0]["text"].split()) == 230


def test_chunker_keeps_page_numbers():
    pages = [{"page": 3, "text": _words(10)}, {"page": 7, "text": _words(10)}]
    assert [c["page"] for c in chunk_pages(pages, "p", "P")] == [3, 7]


def test_section_is_the_one_in_effect_at_chunk_start():
    # Old code stamped every chunk on a page with the LAST header on that page.
    text = "Introduction\n" + _words(300) + "\nResults\n" + _words(300, "r")
    chunks = chunk_pages([{"page": 1, "text": text}], "p", "P")
    assert chunks[0]["section"] == "Introduction"
    assert chunks[-1]["section"] == "Results"


def test_section_carries_across_pages():
    pages = [{"page": 1, "text": "Methods\n" + _words(20)}, {"page": 2, "text": _words(20)}]
    assert chunk_pages(pages, "p", "P")[1]["section"] == "Methods"


# -------------------------------------------------------------- cleaning
def test_clean_keeps_numeric_table_cells_but_drops_page_number():
    raw = "Method Recall\nPaperQA\n100\n22\nVanilla\n6\n12"
    out = _clean_text(raw)
    assert "100" in out and "22" in out and "6" in out
    assert _clean_text("Some text here\nmore text\n8") == "Some text here\nmore text"
    assert _clean_text("3\nSome text").startswith("Some text")
    assert _clean_text("text\nPage 4 of 12") == "text"


# ---------------------------------------------------------------- fusion
def test_rrf_lone_keyword_hit_is_not_given_full_weight():
    fused = fuse_rankings(["a", "b", "c"], ["z"])
    # Old min-max gave a lone BM25 hit keyword score 1.0 => 0.4, beating b and c outright.
    assert fused["z"] < fused["a"]
    assert fused["a"] > fused["b"] > fused["c"]


def test_rrf_agreement_beats_single_source():
    fused = fuse_rankings(["a", "b"], ["b", "x"])
    assert fused["b"] > fused["a"]


def test_rrf_empty_inputs():
    assert fuse_rankings([], []) == {}


# --------------------------------------------------------- paper balance
def test_balance_guarantees_second_relevant_paper():
    ranked = ["A1", "A2", "A3", "A4", "B1"]
    fused = {"A1": 1.0, "A2": 0.9, "A3": 0.8, "A4": 0.75, "B1": 0.6}
    paper_of = {c: c[0] for c in ranked}
    out = select_with_paper_balance(ranked, fused, paper_of, k=3, ratio=0.5)
    assert "B1" in out and len(out) == 3
    assert out == sorted(out, key=ranked.index)            # keeps rank order


def test_balance_does_not_force_irrelevant_paper():
    ranked = ["A1", "A2", "B1"]
    fused = {"A1": 1.0, "A2": 0.9, "B1": 0.1}
    out = select_with_paper_balance(ranked, fused, {c: c[0] for c in ranked}, k=2, ratio=0.5)
    assert out == ["A1", "A2"]


# ------------------------------------------------------------- not-found
@pytest.mark.parametrize("text", [
    "NOT_FOUND: nothing about Redis.", "not_found: x", "NOT FOUND - nope", "  NOT_FOUND:",
])
def test_not_found_marker(text):
    assert is_not_found(text)


def test_valid_answer_with_negative_phrase_is_not_discarded():
    # The old substring list ("does not mention") classified this as not-found.
    ans = "Paper A does not mention caching, but Paper B uses Redis [Source 3]."
    assert not is_not_found(ans)


def test_uncited_refusal_opening_is_caught_but_cited_one_is_not():
    assert is_not_found("I cannot find that in the excerpts.")
    assert not is_not_found("I don't think it was 5%, the paper says 7% [Source 1].")


def test_clean_not_found_strips_marker():
    assert _clean_not_found("NOT_FOUND: Redis isn't discussed.") == "Redis isn't discussed."
    assert "don't contain" in _clean_not_found("NOT_FOUND:")


# ------------------------------------------------------------- citations
def test_remap_renumbers_and_selects_only_cited():
    ans = "X is 5 [Source 4]. Y is 7 (Source 2). Again X [Source 4, Source 2]."
    new, idx = remap_citations(ans, n_chunks=6)
    assert idx == [3, 1]                                   # first-cited order, 0-based
    assert new == "X is 5 [Source 1]. Y is 7 [Source 2]. Again X [Source 1, Source 2]."


def test_remap_drops_out_of_range_and_ignores_page_numbers():
    new, idx = remap_citations("Fact [Source 9]. Other [Source 2: Paper A, Page 4].", 3)
    assert idx == [1]
    assert new == "Fact. Other [Source 1]."


def test_remap_leaves_non_citations_alone():
    new, idx = remap_citations("See [Source code] for details.", 3)
    assert new == "See [Source code] for details." and idx == []


def test_strip_citations():
    assert strip_citations("It is 5 [Source 2]. Done.") == "It is 5. Done."


# ------------------------------------------------------ condenser guard
def test_sanitize_rewrite():
    q = "why?"
    assert sanitize_rewrite(q, 'Standalone question: "Why did Paper A use BM25?"') == "Why did Paper A use BM25?"
    assert sanitize_rewrite(q, "") == q
    assert sanitize_rewrite(q, "Line one\nLine two") == q
    assert sanitize_rewrite(q, "x" * 2000) == q
