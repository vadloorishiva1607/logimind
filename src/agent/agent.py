from langgraph.graph import StateGraph, END

from src.agent.state import GraphState, DRIVER_ID_RE
from src.tools.settlement import run_settlement_agent


def supervisor_node(state: GraphState) -> GraphState:
    question = state["question"]
    match = DRIVER_ID_RE.search(question)
    driver_id = match.group(0) if match else state.get("driver_id")
    return {**state, "driver_id": driver_id}


def settlement_node(state: GraphState, db) -> GraphState:
    driver_id = state.get("driver_id")
    question = state["question"]

    if not driver_id:
        answer = (
            "I need a driver ID (e.g. D102) to look up settlement details. "
            "Could you confirm which driver you're asking about?"
        )
    else:
        answer = run_settlement_agent(db, driver_id, question)

    agents_used = state.get("agents_used", []) + ["settlement"]

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