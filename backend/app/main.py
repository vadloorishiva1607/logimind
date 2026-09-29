from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db import get_db
from app.graph import ask, close_checkpointer

app = FastAPI(title="LogiMind")


class ChatRequest(BaseModel):
    question: str
    # Identifies the conversation for cross-turn memory. In the demo UI this
    # is the browser session ID; falls back to "default" for quick testing.
    thread_id: str = "default"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/db")
def health_db(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    result = ask(db, req.question, thread_id=req.thread_id)
    return result


@app.on_event("shutdown")
def shutdown():
    close_checkpointer()
