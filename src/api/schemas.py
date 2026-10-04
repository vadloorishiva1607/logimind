from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    question: str
    thread_id: str = "default"
    role: Optional[str] = "dispatcher"