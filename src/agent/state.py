from typing import TypedDict, Optional
import re

DRIVER_ID_RE = re.compile(r"\bD\d{3}\b")


class GraphState(TypedDict, total=False):
    question: str
    driver_id: Optional[str]
    answer: Optional[str]
    agents_used: list
    memory_summary: Optional[str]