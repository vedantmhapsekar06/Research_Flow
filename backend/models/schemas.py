"""
Pydantic models describing the shape of data moving in and out of the API.
Keeping these separate makes the API layer easy to read.
"""

from typing import List, Optional
from pydantic import BaseModel


class PaperOut(BaseModel):
    paper_id: str
    filename: str
    status: str            # uploaded | processing | processed | failed
    pages: Optional[int] = None
    chunk_count: Optional[int] = None
    error: Optional[str] = None


class ProcessResult(BaseModel):
    processed: List[str]
    failed: List[dict]


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"


class SourceOut(BaseModel):
    paper_id: str
    paper_name: str
    page: int
    section: Optional[str] = None
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceOut]


class HistoryTurn(BaseModel):
    role: str
    content: str