import logging
from langgraph.graph import StateGraph, END

from src.agent.state import GraphState, DRIVER_ID_RE, SHIPMENT_ID_RE
from src.agent.guardrails import guardrail_node
from src.tools.settlement import run_settlement_agent
from src.tools.escrow import run_escrow_agent
from src.tools.rag import run_rag_agent
from src.tools.performance import run_performance_agent
from src.tools.shipment import run_shipment_agent
from src.tools.fuel import run_fuel_agent
from src.tools.driver import run_driver_agent

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

_ESCROW_KEYWORDS = ("escrow",)
_SHIPMENT_KEYWORDS = ("shipment",)
_FUEL_KEYWORDS = ("fuel", "mpg", "gallon", "diesel")
_DRIVER_KEYWORDS = ("profile", "status", "onboard", "tell me about", "who is", "overview", "summary")

_SETTLEMENT_KEYWORDS = (
     "settlement", "pay", "deduction", "earning", "gross", "net pay",
)



# Driver-data agents the Supervisor can route to: route name -> tool function.
_DRIVER_AGENTS = {
    "settlement": run_settlement_agent,
    "performance": run_performance_agent,
    "escrow": run_escrow_agent,
    "fuel": run_fuel_agent,
     "driver": run_driver_agent,
}

# Routes a follow-up question (no topic words) may stay with.
_FOLLOW_UP_ROUTES = set(_DRIVER_AGENTS) | {"shipment"}

# Agents that also need the user's role (to hide fields the role may not see).
_ROLE_AWARE_AGENTS = {"driver"}
def _matched(question: str, keywords) -> bool:
    """Return the first keyword found in the question, or None."""
    q = question.lower()
    return next((kw for kw in keywords if kw in q), None)

def _pick_driver_agent(question: str, last_agent) -> tuple[str, str]:
    """Return (route, reason). The reason says which rule fired."""
    kw = _matched(question, _FUEL_KEYWORDS)
    if kw:
        return "fuel", f"matched fuel keyword '{kw}'"
    kw = _matched(question, _SHIPMENT_KEYWORDS)
    if kw:
        return "shipment", f"matched shipment keyword '{kw}'"
    kw = _matched(question, _PERFORMANCE_KEYWORDS)
    if kw:
        return "performance", f"matched performance keyword '{kw}'"
    kw = _matched(question, _ESCROW_KEYWORDS)
    if kw:
        return "escrow", f"matched escrow keyword '{kw}'"
    kw = _matched(question, _SETTLEMENT_KEYWORDS)
    if kw:
        return "settlement", f"matched settlement keyword '{kw}'"
    kw = _matched(question, _DRIVER_KEYWORDS)
    if kw:
        return "driver", f"matched driver keyword '{kw}'"
    if last_agent in _FOLLOW_UP_ROUTES:
        return last_agent, f"no topic keyword; follow-up reused last_agent '{last_agent}'"
    return "settlement", "no topic keyword and no previous agent; defaulted to settlement"

def supervisor_node(state: GraphState) -> GraphState:
    question = state["question"]
    match = DRIVER_ID_RE.search(question)
    driver_id = match.group(0) if match else state.get("driver_id")
    ship_match = SHIPMENT_ID_RE.search(question)
    # A new shipment ID replaces the remembered one; a new driver ID (with
    # no shipment ID) means the user moved on, so forget the old shipment.
    if ship_match:
        shipment_id = ship_match.group(0)
    elif match:
        shipment_id = None  
    else:
        shipment_id = state.get("shipment_id")      
    policy_kw = _matched(question, _POLICY_KEYWORDS)

    if ship_match:
        route, reason = "shipment", f"shipment ID {shipment_id} in the question"
    elif policy_kw and not match:
        route, reason = "rag", f"matched policy keyword '{policy_kw}' and no driver ID typed"    
    elif driver_id:    
        route, reason = _pick_driver_agent(question, state.get("last_agent"))
        if not match:
            reason += f" (driver {driver_id} taken from memory)"
    elif shipment_id and state.get("last_agent") == "shipment":
        route, reason = "shipment", f"follow-up about shipment {shipment_id} (from memory)"        
    else:
        # Negative scenario (Design Doc Section 5, Orchestration Layer):
        # no agent can confidently answer -> ask for clarification instead
        # of guessing which one to call.
        route = "clarify"
        reason = "no driver ID and no policy keyword"
    logger.info("Supervisor: route=%s driver=%s reason=%s", route, driver_id, reason)
    return {**state, "driver_id": driver_id, "shipment_id": shipment_id,  "route": route, "route_reason": reason}


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
        agent = _DRIVER_AGENTS[agent_name]
        if agent_name in _ROLE_AWARE_AGENTS:
            answer = agent(db, driver_id, question, role=state.get("role"))
        else:
            answer = agent(db, driver_id, question)    

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

def escrow_node(state: GraphState, db) -> GraphState:
    return _driver_agent_node(state, db, "escrow")

def fuel_node(state: GraphState, db) -> GraphState:
   return _driver_agent_node(state, db, "fuel")

def driver_node(state: GraphState, db) -> GraphState:
    return _driver_agent_node(state, db, "driver")

def shipment_node(state: GraphState, db) -> GraphState:
    """Shipment questions need a shipment ID or a driver ID (either is enough)."""
    question = state["question"]
    shipment_id = state.get("shipment_id")
    # Use the driver only when there is no shipment: the shipment's own
    # record says which driver it belongs to.
    driver_id = None if shipment_id else state.get("driver_id")

    if not shipment_id and not driver_id:
        answer = ("I need a shipment ID (e.g. S10023) or a driver ID (e.g. D102) "
                  "to look up shipments.")
    else:
        answer = run_shipment_agent(db, question, shipment_id=shipment_id, driver_id=driver_id)

    subject = shipment_id or driver_id or "unknown shipment"
    prior_summary = state.get("memory_summary", "")
    new_line = f"Asked about {subject} (shipment): \"{question}\""
    memory_summary = (prior_summary + "\n" + new_line).strip() if prior_summary else new_line

    return {
        **state,
        "answer": answer,
        "agents_used": state.get("agents_used", []) + ["shipment"],
        "last_agent": "shipment",
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
        "driver's settlement, escrow, fuel or performance (safety score, "
        "on-time rate, incidents), include a driver ID (e.g. D102). For a "
        "shipment, include its ID (e.g. S10023). If you're asking about "
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
    graph.add_node("escrow_agent", lambda state: escrow_node(state, db))
    graph.add_node("shipment_agent", lambda state: shipment_node(state, db))
    graph.add_node("fuel_agent", lambda state: fuel_node(state, db))
    graph.add_node("driver_agent", lambda state: driver_node(state, db))
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
            "escrow": "escrow_agent",
            "shipment": "shipment_agent",
            "fuel": "fuel_agent",
            "driver": "driver_agent",
            "rag": "rag_agent",
            "clarify": "clarify",
        },
    )
    graph.add_edge("settlement_agent", END)
    graph.add_edge("performance_agent", END)
    graph.add_edge("escrow_agent", END)
    graph.add_edge("shipment_agent", END)
    graph.add_edge("fuel_agent", END)
    graph.add_edge("driver_agent", END)
    graph.add_edge("rag_agent", END)
    graph.add_edge("clarify", END)
    return graph.compile(checkpointer=checkpointer)