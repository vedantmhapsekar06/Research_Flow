"""
chunker.py
----------
Splits page text into overlapping word-based chunks and attaches
metadata (page number, detected section) to each chunk.

Why overlap? If a sentence describing a key result gets cut exactly in
half between two chunks, neither chunk alone captures the full meaning.
A small overlap (default 40 words) means the boundary sentence usually
appears whole in at least one chunk.
"""

from core.config import CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS
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
    current_section = None

    for page in pages:
        page_num = page["page"]
        text = page["text"]

        for line in text.split("\n"):
            section = detect_section(line)
            if section:
                current_section = section

        words = text.split()
        if not words:
            continue

        start = 0
        chunk_index = 0
        while start < len(words):
            end = min(start + CHUNK_SIZE_WORDS, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunk_id = f"{paper_id}_p{page_num}_c{chunk_index}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "paper_id": paper_id,
                "paper_name": paper_name,
                "page": page_num,
                "section": current_section,
            })

            chunk_index += 1
            if end == len(words):
                break
            start = end - CHUNK_OVERLAP_WORDS

    return chunks