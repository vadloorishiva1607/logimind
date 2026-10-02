"""
Settlement Agent.

Key Actions: retrieve settlement records; compare current vs. prior period;
correlate with fuel/escrow deductions.
Negative scenario: fewer than 2 settlement records for a driver -> return
"insufficient data" rather than guessing at a trend.
"""
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.models.llm_client import get_llm


def get_driver_settlement_history(db: Session, driver_id: str, limit: int = 2):
    """MCP-style read-only tool: last N settlements for a driver, most recent first."""
    rows = db.execute(
        text(
            """
            SELECT week_ending, gross_pay, fuel_deduction,
                   escrow_deduction, other_deduction, net_pay
            FROM settlements
            WHERE driver_id = :driver_id
            ORDER BY week_ending DESC
            LIMIT :limit
            """
        ),
        {"driver_id": driver_id, "limit": limit},
    ).fetchall()
    return rows


def run_settlement_agent(db: Session, driver_id: str, question: str = "") -> str:
    history = get_driver_settlement_history(db, driver_id, limit=2)

    if len(history) < 2:
        # Negative scenario from the agent table: not enough data to compare.
        return (
            f"I don't have enough settlement history for {driver_id} to "
            "compare periods yet -- insufficient data for analysis."
        )

    current, previous = history[0], history[1]
    evidence = (
        f"Most recent week ({current.week_ending}): "
        f"net_pay=${current.net_pay}, fuel_deduction=${current.fuel_deduction}, "
        f"escrow_deduction=${current.escrow_deduction}, other_deduction=${current.other_deduction}.\n"
        f"Prior week ({previous.week_ending}): "
        f"net_pay=${previous.net_pay}, fuel_deduction=${previous.fuel_deduction}, "
        f"escrow_deduction=${previous.escrow_deduction}, other_deduction=${previous.other_deduction}."
    )

    llm = get_llm()
    prompt = (
        "You are the Settlement Agent in a logistics assistant. Using ONLY the "
        "evidence below, answer the driver's question in 2-3 sentences. "
        "Do not invent numbers not present in the evidence.\n\n"
        f"Driver's question: {question}\n\n"
        f"Evidence:\n{evidence}"
    )
    response = llm.invoke(prompt)
    return response.content
