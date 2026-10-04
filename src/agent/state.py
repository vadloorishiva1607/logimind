from typing import TypedDict, Optional
import re

DRIVER_ID_RE = re.compile(r"\bD\d{3}\b")


class GraphState(TypedDict, total=False):
    question: str
    driver_id: Optional[str]
    # The asking user's role (dispatcher / finance / compliance / ...), used
    # by the RAG agent for role-filtered retrieval (Design Doc Section 12 --
    # RBAC is enforced by backend/tool logic, never by the LLM). There's no
    # auth system built yet, so this comes straight from the request body
    # (see src/api/schemas.py's ChatRequest) rather than a session -- an
    # honest demo-scope simplification, not a security claim.
    role: Optional[str]
    # Which node the Supervisor decided to route to this turn: "settlement",
    # "rag", or "clarify". Mainly useful for debugging/logging; the
    # conditional edge in agent.py reads this to pick the next node.
    route: Optional[str]
    answer: Optional[str]
    agents_used: list
    # Cross-turn memory: a short running summary, not the full transcript.
    # Kept deliberately small so per-call token cost doesn't grow with
    # conversation length (see LogiMind_Advanced_Techniques.md, Section 4:
    # "limit context sent per call").
    memory_summary: Optional[str]