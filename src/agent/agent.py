from langgraph.graph import StateGraph, END

from src.agent.state import GraphState, DRIVER_ID_RE
from src.tools.settlement import run_settlement_agent
from src.tools.rag import run_rag_agent

# Rough keyword heuristic for "this sounds like a policy/SOP question,"
# used only when no driver_id is present (a driver_id always wins -- it's
# an unambiguous signal the question is about that driver's data). This is
# deliberately simple rather than an LLM classification call, matching the
# rest of the Supervisor's existing style (plain regex for driver_id); an
# LLM-based intent classifier would be a reasonable upgrade later but isn't
# needed to close the "RAG isn't routed at all" gap.
_POLICY_KEYWORDS = (
    "policy", "sop", "procedure", "guideline", "contract", "rule",
    "terminat", "compliance", "handbook", "regulation",
)


def _looks_like_policy_question(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _POLICY_KEYWORDS)


def supervisor_node(state: GraphState) -> GraphState:
    question = state["question"]
    match = DRIVER_ID_RE.search(question)
    driver_id = match.group(0) if match else state.get("driver_id")

    if driver_id:
        route = "settlement"
    elif _looks_like_policy_question(question):
        route = "rag"
    else:
        # Negative scenario (Design Doc Section 5, Orchestration Layer):
        # no agent can confidently answer -> ask for clarification instead
        # of guessing which one to call.
        route = "clarify"

    return {**state, "driver_id": driver_id, "route": route}


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


def rag_node(state: GraphState) -> GraphState:
    question = state["question"]
    # Default to the lowest-privilege role if the caller didn't send one,
    # rather than defaulting to an elevated role -- fail closed, not open.
    role = state.get("role") or "dispatcher"

    answer = run_rag_agent(question, role)
    agents_used = state.get("agents_used", []) + ["rag"]

    prior_summary = state.get("memory_summary", "")
    new_line = f"Asked a policy question (role={role}): \"{question}\""
    memory_summary = (prior_summary + "\n" + new_line).strip() if prior_summary else new_line

    return {
        **state,
        "answer": answer,
        "agents_used": agents_used,
        "memory_summary": memory_summary,
    }


def clarify_node(state: GraphState) -> GraphState:
    # No agent was confidently selected -- ask rather than guess. Does NOT
    # append to agents_used (no agent actually ran) or memory_summary (no
    # driver/topic was actually established for a future turn to recall).
    answer = (
        "I'm not sure how to help with that yet. If you're asking about a "
        "driver's settlement, include a driver ID (e.g. D102). If you're "
        "asking about a company policy or SOP, try mentioning \"policy\" or "
        "\"procedure\" so I know to search our documents."
    )
    return {**state, "answer": answer, "agents_used": state.get("agents_used", [])}


def build_graph(db, checkpointer=None):
    graph = StateGraph(GraphState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("settlement_agent", lambda state: settlement_node(state, db))
    graph.add_node("rag_agent", rag_node)
    graph.add_node("clarify", clarify_node)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["route"],
        {
            "settlement": "settlement_agent",
            "rag": "rag_agent",
            "clarify": "clarify",
        },
    )
    graph.add_edge("settlement_agent", END)
    graph.add_edge("rag_agent", END)
    graph.add_edge("clarify", END)

    return graph.compile(checkpointer=checkpointer)