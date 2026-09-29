import os
import re
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver

from app.agents.settlement import run_settlement_agent

DRIVER_ID_RE = re.compile(r"\bD\d{3}\b")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://logimind_app:CHANGE_ME_APP_PASSWORD@localhost:5432/logimind",
)


class GraphState(TypedDict, total=False):
    question: str
    driver_id: Optional[str]
    answer: Optional[str]
    agents_used: list
    # Cross-turn memory: a short running summary, not the full transcript.
    # Kept deliberately small so per-call token cost doesn't grow with
    # conversation length (see LogiMind_Advanced_Techniques.md, Section 4:
    # "limit context sent per call").
    memory_summary: Optional[str]


def supervisor_node(state: GraphState) -> GraphState:
    question = state["question"]
    match = DRIVER_ID_RE.search(question)
    driver_id = match.group(0) if match else state.get("driver_id")
    return {**state, "driver_id": driver_id}


def settlement_node(state: GraphState, db) -> GraphState:
    driver_id = state.get("driver_id")
    question = state["question"]

    if not driver_id:
        # Negative scenario: the Supervisor couldn't resolve which driver
        # this question is about (no ID in this turn, none carried over
        # from memory either).
        answer = (
            "I need a driver ID (e.g. D102) to look up settlement details. "
            "Could you confirm which driver you're asking about?"
        )
    else:
        answer = run_settlement_agent(db, driver_id, question)

    agents_used = state.get("agents_used", []) + ["settlement"]

    # Update the running memory summary with a compact record of this turn,
    # rather than appending the full question/answer text every time.
    prior_summary = state.get("memory_summary", "")
    new_line = f"Asked about {driver_id or 'unknown driver'}: \"{question}\""
    memory_summary = (prior_summary + "\n" + new_line).strip() if prior_summary else new_line

    return {
        **state,
        "answer": answer,
        "agents_used": agents_used,
        "memory_summary": memory_summary,
    }


def build_graph(db, checkpointer=None):
    graph = StateGraph(GraphState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("settlement_agent", lambda state: settlement_node(state, db))

    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "settlement_agent")
    graph.add_edge("settlement_agent", END)

    return graph.compile(checkpointer=checkpointer)


# --- Persistent checkpointer (cross-turn conversation memory) ---
#
# Closes the "Memory" gap flagged when LogiMind was compared against the
# Multi-Agentic System Orchestration reference poster: without this,
# every call to ask() started from a blank GraphState, so a follow-up like
# "what about last month?" had no idea which driver the previous turn was
# about. PostgresSaver persists each thread's state (including
# `driver_id` and `memory_summary`) in the SAME RDS instance the app
# already uses for structured data -- LangGraph creates and manages its
# own checkpoint tables there, separate from the app's own schema
# (drivers, settlements, etc.), so no changes to schema.sql are needed.
#
# `checkpointer.setup()` is idempotent -- safe to call on every process
# start. It creates LangGraph's internal tables on first run and is a
# no-op afterward.
_checkpointer_cm = None
_checkpointer = None


def get_checkpointer():
    """Lazily open a PostgresSaver against DATABASE_URL and run setup().

    Returns the checkpointer instance. Call close_checkpointer() on
    application shutdown to release the underlying connection pool.
    """
    global _checkpointer_cm, _checkpointer
    if _checkpointer is None:
        _checkpointer_cm = PostgresSaver.from_conn_string(DATABASE_URL)
        _checkpointer = _checkpointer_cm.__enter__()
        _checkpointer.setup()
    return _checkpointer


def close_checkpointer():
    global _checkpointer_cm, _checkpointer
    if _checkpointer_cm is not None:
        _checkpointer_cm.__exit__(None, None, None)
        _checkpointer_cm = None
        _checkpointer = None


def ask(db, question: str, thread_id: str = "default"):
    """Entry point used by the API layer.

    `thread_id` identifies a conversation (e.g. a chat session ID or a
    logged-in user's session). Passing the same thread_id across calls is
    what makes multi-turn follow-ups work: the checkpointer restores
    `driver_id` and `memory_summary` from the prior turn before the
    Supervisor node runs.
    """
    checkpointer = get_checkpointer()
    graph = build_graph(db, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke({"question": question}, config=config)
    return {
        "answer": result.get("answer"),
        "agents_used": result.get("agents_used", []),
    }
