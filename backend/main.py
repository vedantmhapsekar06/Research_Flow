from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api import papers, chat

app = FastAPI(title="ResearchFlow API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(papers.router)
app.include_router(chat.router)


@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "ResearchFlow backend is running"}


# Serves frontend/index.html at http://localhost:8000/
# IMPORTANT: this must be added AFTER the routers above, so /api/* routes
# are matched first — otherwise the static mount would swallow every request.
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")