"""
api/chat.py
-----------
Endpoints for the chat interface itself.
"""

from fastapi import APIRouter
from models.schemas import ChatRequest, ChatResponse, SourceOut
from services.chat_service import answer_question, get_history

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = answer_question(request.question, request.session_id)
    return ChatResponse(
        answer=result["answer"],
        sources=[SourceOut(**s) for s in result["sources"]],
    )


@router.get("/history")
def history(session_id: str = "default"):
    return {"history": get_history(session_id)}