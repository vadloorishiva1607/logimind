"""
src/tools/settlement.py -- Settlement agent.
Data comes from the MCP server (mcp_server.py, via settlement_mcp_client.py).
Consistency fix (2026-10-16): week-over-week changes are computed in Python,
and the prompt tells the LLM to explain the latest week by default. Before
this, the LLM picked a different drop on different runs (T12/T14/T20).
 """
from src.models.llm_client import get_llm
from src.tools.settlement_mcp_client import get_settlement_history_via_mcp
from src.agent.output_guardrail import check_answer

SETTLEMENT_SYSTEM_PROMPT = """You are LogiMind's Settlement agent. Answer \
using ONLY the settlement data provided below, and quote specific weeks and \
dollar figures from it.

Rules:
1. Unless the question explicitly asks about a different week or period, \
explain the LATEST CHANGE (the most recent week compared with the week \
before it).
2. Use the precomputed changes exactly as given. Do not recompute them.
3. Start with the net pay before and after and the dollar change, then \
explain which components caused it, largest effect first. Higher \
deductions reduce net pay; lower gross pay reduces net pay.
4. If net pay did NOT drop in the latest week, say so plainly, then point \
to the most recent week where it did drop, if any.
5. If the data is insufficient to answer, say so instead of guessing."""


# Money fields returned by the MCP tool, in the order they're described.
_FIELDS = (
    ("net_pay", "net pay"),
    ("gross_pay", "gross pay"),
    ("fuel_deduction", "fuel deduction"),
    ("escrow_deduction", "escrow deduction"),
) 

def _money(x):
    return "n/a" if x is None else f"${x:,.2f}"

def _signed(x):
    return "n/a" if x is None else f"{'+' if x >= 0 else '-'}${abs(x):,.2f}"

def week_over_week(records: list[dict]) -> list[dict]:
    """Return one comparison per pair of consecutive weeks, newest first."""
    rows = sorted(records, key=lambda r: r["week_ending"])  # oldest first
    comparisons = []
    for prev, cur in zip(rows, rows[1:]):
        comp = {"from_week": prev["week_ending"], "to_week": cur["week_ending"]}
        for key, _ in _FIELDS:
            before, after = prev.get(key), cur.get(key)
            change = None if before is None or after is None else round(after - before, 2)
            comp[key] = {"before": before, "after": after, "change": change}
        comparisons.append(comp)
    return list(reversed(comparisons))  # newest first

def _describe(comp: dict) -> str:
    parts = [
        f"{label} {_money(comp[key]['before'])} -> {_money(comp[key]['after'])} "
        f"({_signed(comp[key]['change'])})"
        for key, label in _FIELDS
     ]
    return f"Week ending {comp['from_week']} -> week ending {comp['to_week']}: " + "; ".join(parts)

def build_settlement_context(records: list[dict]) -> str:
    comparisons = week_over_week(records)
    if not comparisons:
        return f"Only one week of data is available: {records}"
    lines = [f"LATEST CHANGE (default focus): {_describe(comparisons[0])}", "",
             "ALL WEEK-OVER-WEEK CHANGES (newest first):"]
    lines += [f"- {_describe(c)}" for c in comparisons]
    return "\n".join(lines)            

def run_settlement_agent(db, driver_id: str, question: str = "") -> str:
    """
    `db` is kept as a parameter for interface compatibility with
    agent.py's settlement_node (and with the other, not-yet-built
    agents, which still take a direct DB session) but is no longer used
    here -- settlement data now comes from the MCP tool call below
    instead of a direct query against `db`.
    """
    try:
        records = get_settlement_history_via_mcp(driver_id, weeks=6)
    except Exception:
        # MCP server subprocess failed to start, DB unreachable from
        # inside it, etc. -- fail closed with a plain message rather
        # than letting the exception bubble up through the graph.
        return (
            f"I couldn't retrieve settlement data for {driver_id} right now. "
            "Please try again in a moment."
        )

    if not records:
        return (
            f"I don't have any settlement records for {driver_id}. "
            "Double-check the driver ID, or this driver may not have a "
            "settlement history yet."
        )

    llm = get_llm()
    context = build_settlement_context(records)
    prompt = (
        f"{SETTLEMENT_SYSTEM_PROMPT}\n\n"
        f"Driver: {driver_id}\n\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    return check_answer(answer, context, "settlement")
