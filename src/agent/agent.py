import logging
from langgraph.graph import StateGraph, END

from src.agent.state import GraphState, DRIVER_ID_RE
from src.agent.guardrails import guardrail_node
from src.tools.settlement import run_settlement_agent
from src.tools.rag import run_rag_agent
from src.tools.performance import run_performance_agent

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

logger = logging.getLogger(__name__)

# Which driver-data agent a driver question is about. Checked in this order;
# if neither matches, the conversation's previous agent is reused (follow-ups),
# and Settlement is the default for a fresh conversation.
_PERFORMANCE_KEYWORDS =(
    "safety", "performance", "incident", "accident", "on-time", "on time","ontime", "late deliver", "score",
)

_SETTLEMENT_KEYWORDS = (
     "settlement", "pay", "deduction", "earning", "gross", "net pay",
)

# Driver-data agents the Supervisor can route to: route name -> tool function.
_DRIVER_AGENTS = {
    "settlement": run_settlement_agent,
    "performance": run_performance_agent,
}

def _matched(question: str, keywords) -> bool:
    """Return the first keyword found in the question, or None."""
    q = question.lower()
    return next((kw for kw in keywords if kw in q), None)

def _pick_driver_agent(question: str, last_agent) -> tuple[str, str]:
    """Return (route, reason). The reason says which rule fired."""
    kw = _matched(question, _PERFORMANCE_KEYWORDS)
    if kw:
        return "performance", f"matched performance keyword '{kw}'"
    kw = _matched(question, _SETTLEMENT_KEYWORDS)
    if kw:
        return "settlement", f"matched settlement keyword '{kw}'"
    if last_agent in _DRIVER_AGENTS:
        return last_agent, f"no topic keyword; follow-up reused last_agent '{last_agent}'"
    return "settlement", "no topic keyword and no previous agent; defaulted to settlement"

def supervisor_node(state: GraphState) -> GraphState:
    question = state["question"]
    match = DRIVER_ID_RE.search(question)
    driver_id = match.group(0) if match else state.get("driver_id")
    policy_kw = _matched(question, _POLICY_KEYWORDS)

    if driver_id:
        route, reason = _pick_driver_agent(question, state.get("last_agent"))
        if not match:
            reason += f" (driver {driver_id} taken from memory)"
    elif policy_kw:
        route, reason = "rag", f"no driver ID; matched policy keyword '{policy_kw}'"
    else:
        # Negative scenario (Design Doc Section 5, Orchestration Layer):
        # no agent can confidently answer -> ask for clarification instead
        # of guessing which one to call.
        route = "clarify"
        reason = "no driver ID and no policy keyword"
    logger.info("Supervisor: route=%s driver=%s reason=%s", route, driver_id, reason)
    return {**state, "driver_id": driver_id, "route": route, "route_reason": reason}


def _driver_agent_node(state: GraphState, db, agent_name: str) -> GraphState:
    """Shared node body for the driver-data agents (Settlement, Performance)."""
    driver_id = state.get("driver_id")
    question = state["question"]

    if not driver_id:
        answer = (
            f"I need a driver ID (e.g. D102) to look up {agent_name} details. "
            "Could you confirm which driver you're asking about?"
        )
    else:
        answer = _DRIVER_AGENTS[agent_name](db, driver_id, question)

    agents_used = state.get("agents_used",[])+[agent_name]

    prior_summary = state.get("memory_summary", "")
    new_line = f"Asked about {driver_id or 'unknown driver'} ({agent_name}): \"{question}\""
    memory_summary = (prior_summary + "\n" + new_line).strip() if prior_summary else new_line

    return {
        **state,
        "answer": answer,
        "agents_used": agents_used,
        "last_agent": agent_name,
        "memory_summary": memory_summary,
    }

def settlement_node(state: GraphState, db) -> GraphState:
    return _driver_agent_node(state, db, "settlement")

def performance_node(state: GraphState, db) -> GraphState:
    return _driver_agent_node(state, db, "performance")

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
         "driver's settlement or performance (safety score, on-time rate, "
         "incidents), include a driver ID (e.g. D102). If you're asking about "
         "a company policy or SOP, try mentioning \"policy\" or \"procedure\" "
         "so I know to search our documents."
    )
    return {**state, "answer": answer, "agents_used": state.get("agents_used", [])}


def build_graph(db, checkpointer=None):
    graph = StateGraph(GraphState)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("settlement_agent", lambda state: settlement_node(state, db))
    graph.add_node("performance_agent", lambda state: performance_node(state, db))
    graph.add_node("rag_agent", rag_node)
    graph.add_node("clarify", clarify_node)

    graph.set_entry_point("guardrail")
    graph.add_conditional_edges(
        "guardrail",
        lambda state: "blocked" if state.get("blocked_reason") else "ok",
        {"blocked": END, "ok": "supervisor"},
    )
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["route"],
        {
            "settlement": "settlement_agent",
            "performance": "performance_agent",
            "rag": "rag_agent",
            "clarify": "clarify",
        },
    )
    graph.add_edge("settlement_agent", END)
    graph.add_edge("performance_agent", END)
    graph.add_edge("rag_agent", END)
    graph.add_edge("clarify", END)

    return graph.compile(checkpointer=checkpointer)