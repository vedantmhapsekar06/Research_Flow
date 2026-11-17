"""
api/papers.py
-------------
Endpoints for managing the paper library.
"""

import hashlib
import uuid
from collections import Counter

from fastapi import APIRouter, UploadFile, File, HTTPException

from core.config import UPLOAD_DIR, MAX_UPLOAD_MB
from models.schemas import PaperOut, ProcessResult
from services import storage
from services.pdf_processor import extract_pages
from services.chunker import chunk_pages
from services.embeddings import get_embeddings_batch, OllamaConnectionError
from services.vector_store import add_chunks, delete_paper as vector_delete_paper

router = APIRouter(prefix="/api/papers", tags=["papers"])

_READ_BLOCK = 1024 * 1024
_MAX_BYTES = MAX_UPLOAD_MB * 1024 * 1024


@router.post("/upload", response_model=PaperOut)
def upload_paper(file: UploadFile = File(...)):
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    paper_id = uuid.uuid4().hex[:12]
    dest_path = UPLOAD_DIR / f"{paper_id}.pdf"

    # Stream to disk in blocks: enforce the size limit, verify it really is a
    # PDF (an extension alone proves nothing), and hash it for de-duplication.
    hasher = hashlib.sha256()
    size = 0
    try:
        with open(dest_path, "wb") as out:
            first = True
            while True:
                block = file.file.read(_READ_BLOCK)
                if not block:
                    break
                if first:
                    if b"%PDF-" not in block[:1024]:
                        raise HTTPException(400, "File does not look like a valid PDF.")
                    first = False
                size += len(block)
                if size > _MAX_BYTES:
                    raise HTTPException(413, f"File exceeds the {MAX_UPLOAD_MB} MB upload limit.")
                hasher.update(block)
                out.write(block)
        if size == 0:
            raise HTTPException(400, "The uploaded file is empty.")
    except HTTPException:
        dest_path.unlink(missing_ok=True)
        raise

    digest = hasher.hexdigest()

    # Re-uploading the same file would double its chunks and skew retrieval.
    # Hand back the existing record instead of creating a duplicate.
    existing = storage.find_paper_by_hash(digest)
    if existing:
        dest_path.unlink(missing_ok=True)
        existing_id, fields = existing
        return PaperOut(paper_id=existing_id, **{k: v for k, v in fields.items() if k in PaperOut.model_fields})

    storage.upsert_paper(
        paper_id,
        filename=filename,
        status="uploaded",
        pages=None,
        error=None,
        sha256=digest,
        retryable=False,
    )
    return PaperOut(paper_id=paper_id, filename=filename, status="uploaded")


@router.get("", response_model=list[PaperOut])
def list_papers():
    papers = storage.load_papers()
    chunks = storage.load_chunks()
    chunk_counts = Counter(c["paper_id"] for c in chunks.values())

    return [
        PaperOut(
            paper_id=pid,
            chunk_count=chunk_counts.get(pid, 0),
            **{k: v for k, v in fields.items() if k in PaperOut.model_fields and k != "chunk_count"},
        )
        for pid, fields in papers.items()
    ]


@router.delete("/{paper_id}")
def delete_paper(paper_id: str):
    papers = storage.load_papers()
    if paper_id not in papers:
        raise HTTPException(404, "Paper not found.")
    if papers[paper_id].get("status") == "processing":
        # Deleting mid-processing would let the worker re-insert chunks for a
        # paper that no longer exists.
        raise HTTPException(409, "Paper is currently being processed; try again when it finishes.")

    vector_delete_paper(paper_id)
    storage.remove_paper_chunks_from_store(paper_id)
    storage.delete_paper_record(paper_id)

    pdf_path = UPLOAD_DIR / f"{paper_id}.pdf"
    pdf_path.unlink(missing_ok=True)

    return {"deleted": paper_id}


def _discard_partial(paper_id: str) -> None:
    """Remove any chunks left behind by an interrupted/failed attempt."""
    try:
        vector_delete_paper(paper_id)
    except Exception:
        pass
    storage.remove_paper_chunks_from_store(paper_id)


def _process_one(paper_id: str, filename: str) -> tuple[bool, str | None]:
    """Returns (ok, error_message)."""
    pdf_path = UPLOAD_DIR / f"{paper_id}.pdf"
    try:
        pages = extract_pages(str(pdf_path))
        if not pages:
            raise ValueError("No extractable text found in this PDF.")

        chunks = chunk_pages(pages, paper_id, filename)
        embeddings = get_embeddings_batch([c["text"] for c in chunks])

        # Clear leftovers from any earlier interrupted attempt so the index
        # never holds a mix of old and new chunks for this paper.
        _discard_partial(paper_id)
        add_chunks(chunks, embeddings)
        storage.add_chunks_to_store(chunks)

        # update_paper (not upsert): if the record vanished meanwhile, don't resurrect it.
        if not storage.update_paper(paper_id, status="processed", pages=len(pages),
                                    error=None, retryable=False):
            _discard_partial(paper_id)
            return False, "Paper was deleted during processing."
        return True, None

    except OllamaConnectionError as e:
        _discard_partial(paper_id)
        # Transient: the next /process call retries it automatically.
        storage.update_paper(paper_id, status="failed", error=str(e), retryable=True)
        return False, str(e)
    except Exception as e:
        _discard_partial(paper_id)
        # Permanent (corrupt PDF, no text): don't retry forever.
        storage.update_paper(paper_id, status="failed", error=str(e), retryable=False)
        return False, str(e)


@router.post("/process", response_model=ProcessResult)
def process_papers():
    # Atomic claim: two concurrent calls can never both process the same paper.
    claimed = storage.claim_papers_for_processing()
    processed, failed = [], []

    for paper_id, filename in claimed:
        ok, err = _process_one(paper_id, filename)
        if ok:
            processed.append(paper_id)
        else:
            failed.append({"paper_id": paper_id, "error": err})

    return ProcessResult(processed=processed, failed=failed)
