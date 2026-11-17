"""
api/chat.py
-----------
Endpoints for the chat interface itself.
"""

from fastapi import APIRouter, HTTPException
from models.schemas import ChatRequest, ChatResponse, SourceOut
from services.chat_service import answer_question, get_history
from services.errors import OllamaConnectionError

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(400, "Question must not be empty.")

    try:
        result = answer_question(question, request.session_id)
    except OllamaConnectionError as e:
        # Ollama being down/slow is a service-availability problem, not a bug.
        raise HTTPException(503, str(e))

    return ChatResponse(
        answer=result["answer"],
        sources=[SourceOut(**s) for s in result["sources"]],
    )


@router.get("/history")
def history(session_id: str = "default"):
    return {"history": get_history(session_id)}
