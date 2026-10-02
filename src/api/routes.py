from fastapi import FastAPI, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.models.db import get_db
from src.agent.executor import ask
from src.agent.memory import close_checkpointer
from src.api.schemas import ChatRequest

app = FastAPI(title="LogiMind")


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