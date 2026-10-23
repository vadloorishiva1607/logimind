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
    #Why the Supervisor picked `route` this turn: which rule fired (e.g#
    # "matched performance keyword 'safety'"). Logged, and shown by
    # tests/test_routing.py on a misroute, so routing is explainable.
    route_reason: Optional[str]
    # Set by the input guardrail when it blocks a question ("action_request",
    # "sensitive_data_request", "sensitive_data_in_message"); None otherwise.
    blocked_reason: Optional[str]
    # The last driver-data agent that answered in this conversation
    # ("settlement" or "performance"). Persisted by the checkpointer, so a
    # follow-up with no topic words ("what about the period before that?")
    # stays with the same agent instead of falling back to Settlement.
    # last_agent so that follow-up questions without topic words go back to the agent that answered last,
    last_agent: Optional[str]
    answer: Optional[str]
    agents_used: list
    # Cross-turn memory: a short running summary, not the full transcript.
    # Kept deliberately small so per-call token cost doesn't grow with
    # conversation length (see LogiMind_Advanced_Techniques.md, Section 4:
    # "limit context sent per call").
    memory_summary: Optional[str]