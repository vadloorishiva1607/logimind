"""
LangGraph Supervisor -- currently routes to the Settlement agent only.
Structured so the remaining 6 agents (Driver, Fuel, Escrow, Performance,
Shipment/Risk, RAG/Knowledge) plug into the same pattern: add a node,
add a routing condition, done.
"""
import re
from typing import TypedDict

from langgraph.graph import StateGraph, END
from sqlalchemy.orm import Session

from app.agents.settlement import run_settlement_agent

DRIVER_ID_PATTERN = re.compile(r"\bD\d{3}\b", re.IGNORECASE)


class GraphState(TypedDict):
    question: str
    driver_id: str | None
    answer: str
    agents_used: list[str]


def supervisor_node(state: GraphState) -> GraphState:
    """Extracts the driver ID and decides which agent(s) to call.
    Negative scenario: no agent can confidently handle the question ->
    ask for clarification instead of guessing (Supervisor's own
    negative-scenario row from the agent table).
    """
    match = DRIVER_ID_PATTERN.search(state["question"])
    state["driver_id"] = match.group(0).upper() if match else None
    state["agents_used"] = []
    return state


def settlement_node(state: GraphState, db: Session) -> GraphState:
    if not state["driver_id"]:
        state["answer"] = (
            "I need a driver ID (e.g. D102) to look up settlement details -- "
            "could you clarify which driver you mean?"
        )
        return state
    state["answer"] = run_settlement_agent(db, state["driver_id"])
    state["agents_used"].append("Settlement")
    return state


def build_graph(db: Session):
    """db is bound in via closure so each request gets its own session."""
    graph = StateGraph(GraphState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("settlement_agent", lambda s: settlement_node(s, db))

    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "settlement_agent")
    graph.add_edge("settlement_agent", END)

    return graph.compile()


def ask(db: Session, question: str) -> GraphState:
    app_graph = build_graph(db)
    result = app_graph.invoke(
        {"question": question, "driver_id": None, "answer": "", "agents_used": []}
    )
    return result
