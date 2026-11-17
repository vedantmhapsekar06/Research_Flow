"""
Pydantic models describing the shape of data moving in and out of the API.
Keeping these separate makes the API layer easy to read.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


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
    question: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(default="default", max_length=100)


class SourceOut(BaseModel):
    paper_id: str
    paper_name: str
    page: int
    section: Optional[str] = None
    snippet: str
    # 0-1 relevance of this chunk to the query. The UI reads `score` to draw
    # the relevance bar; it was previously never sent, so every bar showed 50%.
    score: Optional[float] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceOut]


class HistoryTurn(BaseModel):
    role: str
    content: str
