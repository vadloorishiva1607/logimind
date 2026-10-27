from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.models.db import get_db
from src.agent.executor import ask
from src.agent.memory import close_checkpointer
from src.tools.rag import get_chroma_collection
from src.api.schemas import ChatRequest

app = FastAPI(title="LogiMind")

# Let the Angular UI call this API from a browser. Browsers block calls to a
# different address (origin) unless the API lists that origin here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",                 # Angular dev server (ng serve)
        "https://logimind-blush.vercel.app/",   # # the deployed UI
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "ngrok-skip-browser-warning"],
)


@app.on_event("startup")
def startup():
    # Pre-create the ChromaDB client on the main thread/event loop at
    # app startup, rather than lazily on the first RAG request. See
    # rag.py's get_chroma_collection() docstring/comment for why.
    get_chroma_collection()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/db")
def health_db(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    result = ask(db, req.question, thread_id=req.thread_id, role=req.role)
    return result


@app.on_event("shutdown")
def shutdown():
    close_checkpointer()