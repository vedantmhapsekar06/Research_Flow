"""
api/papers.py
-------------
Endpoints for managing the paper library.
"""

import uuid
import shutil
from collections import Counter
from fastapi import APIRouter, UploadFile, File, HTTPException

from core.config import UPLOAD_DIR
from models.schemas import PaperOut, ProcessResult
from services import storage
from services.pdf_processor import extract_pages
from services.chunker import chunk_pages
from services.embeddings import get_embeddings_batch, OllamaConnectionError
from services.vector_store import add_chunks, delete_paper as vector_delete_paper

router = APIRouter(prefix="/api/papers", tags=["papers"])


@router.post("/upload", response_model=PaperOut)
def upload_paper(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    paper_id = uuid.uuid4().hex[:12]
    dest_path = UPLOAD_DIR / f"{paper_id}.pdf"

    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    storage.upsert_paper(
        paper_id,
        filename=file.filename,
        status="uploaded",
        pages=None,
        error=None,
    )

    return PaperOut(paper_id=paper_id, filename=file.filename, status="uploaded")


@router.get("", response_model=list[PaperOut])
def list_papers():
    papers = storage.load_papers()
    chunks = storage.load_chunks()

    chunk_counts = Counter(c["paper_id"] for c in chunks.values())

    return [
        PaperOut(paper_id=pid, chunk_count=chunk_counts.get(pid, 0), **fields)
        for pid, fields in papers.items()
    ]


@router.delete("/{paper_id}")
def delete_paper(paper_id: str):
    papers = storage.load_papers()
    if paper_id not in papers:
        raise HTTPException(404, "Paper not found.")

    pdf_path = UPLOAD_DIR / f"{paper_id}.pdf"
    if pdf_path.exists():
        pdf_path.unlink()

    vector_delete_paper(paper_id)
    storage.remove_paper_chunks_from_store(paper_id)
    storage.delete_paper_record(paper_id)

    return {"deleted": paper_id}


@router.post("/process", response_model=ProcessResult)
def process_papers():
    papers = storage.load_papers()
    processed, failed = [], []

    for paper_id, fields in papers.items():
        if fields.get("status") != "uploaded":
            continue

        storage.upsert_paper(paper_id, status="processing")
        pdf_path = UPLOAD_DIR / f"{paper_id}.pdf"

        try:
            pages = extract_pages(str(pdf_path))
            if not pages:
                raise ValueError("No extractable text found in this PDF.")

            chunks = chunk_pages(pages, paper_id, fields["filename"])
            embeddings = get_embeddings_batch([c["text"] for c in chunks])

            add_chunks(chunks, embeddings)
            storage.add_chunks_to_store(chunks)

            storage.upsert_paper(paper_id, status="processed", pages=len(pages), error=None)
            processed.append(paper_id)

        except OllamaConnectionError as e:
            storage.upsert_paper(paper_id, status="failed", error=str(e))
            failed.append({"paper_id": paper_id, "error": str(e)})
        except Exception as e:
            storage.upsert_paper(paper_id, status="failed", error=str(e))
            failed.append({"paper_id": paper_id, "error": str(e)})

    return ProcessResult(processed=processed, failed=failed)