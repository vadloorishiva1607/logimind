from fastapi import FastAPI, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.models.db import get_db
from src.agent.executor import ask
from src.agent.memory import close_checkpointer
from src.tools.rag import get_chroma_collection
from src.api.schemas import ChatRequest

app = FastAPI(title="LogiMind")


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