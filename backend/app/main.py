"""
LogiMind FastAPI entrypoint.
Day 1 goal: prove the container, the API, and the database can all
talk to each other. Agents get wired in starting Day 4.
"""
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db import get_db
from app.graph import ask

app = FastAPI(title="LogiMind API")


class ChatRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "service": "logimind-backend"}


@app.get("/health/db")
def health_db(db: Session = Depends(get_db)):
    """Confirms the backend can actually reach Postgres, not just itself."""
    result = db.execute(text("SELECT 1")).scalar()
    return {"status": "ok", "db_check": result}


@app.post("/chat")
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    """First working agent: routes settlement questions through the
    Supervisor -> Settlement Agent -> LLM pipeline, end to end."""
    result = ask(db, req.question)
    return {"answer": result["answer"], "agents_used": result["agents_used"]}

