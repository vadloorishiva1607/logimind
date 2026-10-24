"""
mcp_server.py -- a real MCP server exposing the Settlement agent's one
read-only DB tool over the Model Context Protocol.

Closes the gap flagged in LogiMind_Advanced_Techniques.md Section 7: the
design doc's Technology Stack (Section 11) and Section 5's per-agent
"Key Actions" rows describe every agent retrieving data "via MCP", but
the Settlement agent (the only agent built so far) called the database
directly through a plain Python function. This file is that function
turned into an actual MCP tool server; `src/tools/settlement_mcp_client.py`
is the MCP client that calls it.

Lives at the repo root (next to requirements.txt and Dockerfile) so it
can be spawned as a subprocess via a simple relative/absolute path from
the client, same as the example in Advanced Techniques Section 7.

Run standalone to sanity-check it (Ctrl+C to stop):
    PYTHONPATH=. DATABASE_URL=... python mcp_server.py
The client normally spawns this itself over stdio -- you don't need to
run it by hand in production.
"""
import os

import psycopg2
from mcp.server.fastmcp import FastMCP

# Same env var and same connection string shape as the rest of the app
# (src/models/db.py, src/agent/memory.py) -- read-only usage only. The
# `logimind_app` role's grants (roles.sql) are SELECT-only, so even a
# bug here can't write to the operational tables.
#
# No default value on purpose: if DATABASE_URL is missing, fail right
# away with a clear message instead of trying a placeholder password
# and producing a confusing "authentication failed" error later.
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. Export it (or put it in .env) before "
        "starting mcp_server.py, e.g. "
        "postgresql://logimind_app:<password>@<host>:5432/logimind"
    )

# Upper limit on how much history one tool call can request. An LLM
# chooses the `weeks` argument, so it could ask for something huge; a
# year of weekly settlements is far more than any question needs.
MAX_WEEKS = 52

MAX_TRANSACTIONS = 50

mcp = FastMCP("logimind-settlement-tools")


@mcp.tool()
def get_settlement_history(driver_id: str, weeks: int = 6) -> list[dict]:
    """Return the last N weeks of settlement records for a driver.

    Each record has: week_ending, gross_pay, fuel_deduction,
    escrow_deduction, net_pay. Returns an empty list if the driver has
    no settlement rows (not an error) -- the caller decides how to word
    that to the user.
    """

    # Keep the query small and valid no matter what the LLM passes in:
    # weeks=0 or negative -> 1, weeks=100000 -> 52.
    weeks = max(1, min(weeks, MAX_WEEKS))
    
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT week_ending, gross_pay, fuel_deduction, escrow_deduction, net_pay "
                "FROM settlements WHERE driver_id = %s ORDER BY week_ending DESC LIMIT %s",
                (driver_id, weeks),
            )
            cols = [d.name for d in cur.description]
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()

    # dates/Decimals aren't JSON-serializable as-is; MCP results travel
    # as JSON over the wire, so normalize before returning.
    for row in rows:
        if row.get("week_ending") is not None:
            row["week_ending"] = row["week_ending"].isoformat()
        for money_field in ("gross_pay", "fuel_deduction", "escrow_deduction", "net_pay"):
            if row.get(money_field) is not None:
                row[money_field] = float(row[money_field])
    return rows

@mcp.tool()
def get_performance_history(driver_id: str, periods: int = 6) -> dict:
    """Return a driver's last N performance periods (newest first) and the
    fleet average for the latest period.

    `periods` is clamped to 1-52. Returns {"rows": [...], "fleet": {...} or
    None}. Each row has: period_start, period_end, on_time_pct,
    safety_score, late_deliveries, incidents. An empty "rows" list means
    the driver has no performance history (not an error).
    """
    periods = max(1, min(periods, MAX_WEEKS))

    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT period_start, period_end, on_time_pct, safety_score, "
                "late_deliveries, incidents FROM performance "
                "WHERE driver_id = %s ORDER BY period_end DESC LIMIT %s",
                (driver_id, periods),
            )
            cols = [d.name for d in cur.description]
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
            cur.execute(
                "SELECT round(avg(on_time_pct), 2), round(avg(safety_score), 2), "
                "count(DISTINCT driver_id) FROM performance "
                "WHERE period_end = (SELECT max(period_end) FROM performance)"
            )
            fleet_row = cur.fetchone()
    finally:
        conn.close()

    # Same JSON clean-up as the settlement tool: dates -> ISO text,
    # Decimals -> floats.
    for row in rows:
        for k in ("period_start", "period_end"):
            if row.get(k) is not None:
                row[k] = row[k].isoformat()
        for k in ("on_time_pct", "safety_score"):
            if row.get(k) is not None:
                row[k] = float(row[k])

    fleet = None
    if fleet_row and fleet_row[0] is not None:
        fleet = {"on_time_pct": float(fleet_row[0]),
                 "safety_score": float(fleet_row[1]),
                 "drivers": fleet_row[2]}
    return {"rows": rows, "fleet": fleet}

@mcp.tool()
def get_escrow_balance(driver_id: str) -> dict:
    """Return a driver's CURRENT escrow balance: the balance after their
    most recent escrow transaction, plus that transaction.

    Use this for "what is the balance now" questions. Returns
    {"balance": None, "last_transaction": None} if the driver has no
    escrow transactions (not an error).
    """
    conn=psycopg2.connect(DATABASE_URL);
    try:
           with conn.cursor() as cur:
            cur.execute(
                "SELECT transaction_date, type, amount_usd, balance_after, reason "
                "FROM escrow_transactions WHERE driver_id = %s "
                "ORDER BY transaction_date DESC, id DESC LIMIT 1",
                (driver_id,),
            )
            row = cur.fetchone()
            cols = [d.name for d in cur.description]
    finally:
        conn.close()
    if row is None:
       return {"balance": None, "last_transaction": None}
    last = _clean_escrow_row(dict(zip(cols, row)))
    return {"balance": last["balance_after"], "last_transaction": last}    

@mcp.tool()
def get_escrow_history(driver_id: str, limit: int = 10) -> dict:
    """Return a driver's last N escrow transactions (newest first).

    Use this to explain WHY a balance changed: deposits, withdrawals and
    adjustments with their reasons. `limit` is clamped to 1-50. Returns
    {"rows": [...]}; each row has transaction_date, type, amount_usd,
    balance_after, reason. An empty list means no escrow history.
    """
    limit = max(1, min(limit, MAX_TRANSACTIONS))

    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT transaction_date, type, amount_usd, balance_after, reason "
                "FROM escrow_transactions WHERE driver_id = %s "
                "ORDER BY transaction_date DESC, id DESC LIMIT %s",
                (driver_id, limit),
            )
            cols = [d.name for d in cur.description]
            rows = [_clean_escrow_row(dict(zip(cols, r))) for r in cur.fetchall()]
    finally:
        conn.close()

    return {"rows": rows}

def _clean_escrow_row(row: dict) -> dict:
    """Make one escrow row JSON-safe: date -> ISO text, Decimal -> float."""
    if row.get("transaction_date") is not None:
        row["transaction_date"] = row["transaction_date"].isoformat()
    for k in ("amount_usd", "balance_after"):
        if row.get(k) is not None:
            row[k] = float(row[k])
    return row


if __name__ == "__main__":
    # stdio transport: the client launches this file as a child process
    # and talks to it over stdin/stdout -- no network port to open or
    # secure, which is why stdio (not SSE/HTTP) is the right choice for
    # a tool that only ever has one caller (this app's own backend).
    mcp.run(transport="stdio")
