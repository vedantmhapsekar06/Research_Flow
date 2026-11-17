"""
chunker.py
----------
Splits page text into overlapping word-based chunks and attaches
metadata (page number, detected section) to each chunk.

Why overlap? If a sentence describing a key result gets cut exactly in
half between two chunks, neither chunk alone captures the full meaning.
A small overlap (default 40 words) means the boundary sentence usually
appears whole in at least one chunk.

Chunks deliberately never cross page boundaries: every chunk keeps exactly
one page number, so a citation like "Page 8" always points at text that is
really on page 8.
"""

from core.config import CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS, CHUNK_MIN_TAIL_WORDS
from services.pdf_processor import detect_section


def chunk_pages(pages: list[dict], paper_id: str, paper_name: str) -> list[dict]:
    """
    pages: [{"page": 1, "text": "..."}, ...]  (output of pdf_processor.extract_pages)

    Returns a flat list of chunk dicts:
        {
          "chunk_id": "paperid_p1_c0",
          "text": "...",
          "paper_id": "...",
          "paper_name": "...",
          "page": 1,
          "section": "Introduction" | None,
        }
    """
    chunks = []
    current_section = None   # carries across pages until the next header

    for page in pages:
        page_num = page["page"]

        # Tag every word with the section in effect where it appears, so each
        # chunk gets the section at ITS start — not whatever header happened
        # to be last on the page.
        words: list[str] = []
        word_sections: list[str | None] = []
        for line in page["text"].split("\n"):
            header = detect_section(line)
            if header:
                current_section = header
            for w in line.split():
                words.append(w)
                word_sections.append(current_section)

        if not words:
            continue

        start = 0
        chunk_index = 0
        while start < len(words):
            end = min(start + CHUNK_SIZE_WORDS, len(words))
            # Absorb a tiny remainder instead of emitting a tail chunk that is
            # almost entirely overlap with this one.
            if len(words) - end < CHUNK_MIN_TAIL_WORDS:
                end = len(words)

            chunks.append({
                "chunk_id": f"{paper_id}_p{page_num}_c{chunk_index}",
                "text": " ".join(words[start:end]),
                "paper_id": paper_id,
                "paper_name": paper_name,
                "page": page_num,
                "section": word_sections[start],
            })

            chunk_index += 1
            if end == len(words):
                break
            start = end - CHUNK_OVERLAP_WORDS   # config guarantees overlap < size => always advances

    return chunks
