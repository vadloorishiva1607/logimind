"""
src/tools/settlement.py

*** MERGE NOTE — READ BEFORE OVERWRITING YOUR EXISTING FILE ***
I don't have your current settlement.py's exact text in this session
(it lived on your machine/EC2, not here). This is a faithful
reconstruction of what it must do based on agent.py's call signature
(`run_settlement_agent(db, driver_id, question)`), the "insufficient
data" branch mentioned in the Testing & Verification Log, and the
grounded answers already observed in T5/T8 (fuel/escrow deductions,
"I need a driver ID..." fallback). The ONLY functional change from
Section 7's scope is swapping the direct DB query for the MCP client
call. If your real file has different prompt wording, keep YOUR
wording and just make the equivalent swap:
    - replace the direct psycopg2/SQLAlchemy settlement query
    - with: from src.tools.settlement_mcp_client import get_settlement_history_via_mcp
             records = get_settlement_history_via_mcp(driver_id, weeks=6)
That's the whole diff Section 7 asks for -- everything else (prompt,
negative-scenario handling) is supposed to stay the same.
"""
from src.models.llm_client import get_llm
from src.tools.settlement_mcp_client import get_settlement_history_via_mcp

SETTLEMENT_SYSTEM_PROMPT = """You are LogiMind's Settlement agent. Answer \
the driver's question using ONLY the settlement records provided below \
-- reference specific weeks and dollar figures from the data. If the \
data provided is insufficient to answer confidently, say so plainly \
instead of guessing or inventing figures."""


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
    prompt = (
        f"{SETTLEMENT_SYSTEM_PROMPT}\n\n"
        f"Driver: {driver_id}\n"
        f"Settlement records (most recent first): {records}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)
