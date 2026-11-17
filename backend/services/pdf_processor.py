"""
pdf_processor.py
-----------------
Turns a PDF file on disk into a list of (page_number, cleaned_text) pairs.

Why page-level extraction matters: ResearchFlow's whole value proposition
is "answer + cite the exact page it came from". If we just concatenated
the entire PDF into one blob of text, we'd lose the ability to say
"Page 8" later. So we always keep text tagged to the page it came from.
"""

import re
import fitz  # PyMuPDF


def extract_pages(pdf_path: str) -> list[dict]:
    """
    Opens a PDF and returns a list like:
        [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]

    Pages are 1-indexed because that's how humans refer to pages
    ("see page 8"), not how programmers index lists.
    """
    pages = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            raw_text = page.get_text("text")
            cleaned = _clean_text(raw_text)
            if cleaned.strip():
                pages.append({"page": i + 1, "text": cleaned})
    return pages


_PAGE_NUMBER_RE = re.compile(r"(?:page\s+)?\d{1,4}(?:\s*(?:/|of)\s*\d{1,4})?", re.IGNORECASE)


def _clean_text(text: str) -> str:
    """
    Light cleaning of text extracted from academic PDFs:
    - removes hyphenation line-breaks like "atten-\\ntion" -> "attention"
    - collapses excessive whitespace/newlines that PDF extraction introduces
    - strips a page number ONLY when it is the first or last line of the page.
      (The old version deleted every digits-only line, which also destroyed
      numeric table cells like "100" or "22" — exactly the exact-number facts
      a research Q&A tool needs to keep.)
    """
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    if lines and _PAGE_NUMBER_RE.fullmatch(lines[0]):
        lines.pop(0)
    if lines and _PAGE_NUMBER_RE.fullmatch(lines[-1]):
        lines.pop()
    return "\n".join(lines).strip()


SECTION_HEADERS = [
    "abstract", "introduction", "related work", "background",
    "methodology", "methods", "materials and methods", "approach",
    "experiments", "experimental setup", "results", "evaluation",
    "discussion", "limitations", "conclusion", "conclusions",
    "references", "acknowledgments",
]


def detect_section(text_line: str) -> str | None:
    """
    Very lightweight heuristic: if a line is short and matches a known
    section header (case-insensitive, ignoring numbering like '3.'),
    treat it as a section title. A full layout-aware parser (e.g. GROBID)
    is a possible future enhancement — this is intentionally simple.
    """
    stripped = re.sub(r"^\d+(\.\d+)*\.?\s*", "", text_line).strip().lower()
    if stripped in SECTION_HEADERS and len(text_line) < 40:
        return stripped.title()
    return None