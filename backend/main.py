from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api import papers, chat
from core.config import ALLOWED_ORIGINS
from services import storage


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Papers left in "processing" by a crash/restart would otherwise be stuck forever.
    storage.reset_stale_processing()
    yield


app = FastAPI(title="ResearchFlow API", lifespan=lifespan)

# The frontend is served by this same app, so its requests are same-origin and
# need no CORS. Only explicitly listed origins (config.ALLOWED_ORIGINS) may
# call the API cross-origin — never "*", which would expose your papers to any
# website open in your browser.
if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )

app.include_router(papers.router)
app.include_router(chat.router)


@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "ResearchFlow backend is running"}


# Serves frontend/index.html at http://localhost:8500/ (the port you pass to uvicorn).
# IMPORTANT: this must be added AFTER the routers above, so /api/* routes
# are matched first — otherwise the static mount would swallow every request.
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
