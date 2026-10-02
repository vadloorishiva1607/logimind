from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    thread_id: str = "default"