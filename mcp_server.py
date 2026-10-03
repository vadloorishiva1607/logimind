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
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://logimind_app:CHANGE_ME_APP_PASSWORD@localhost:5432/logimind",
)

mcp = FastMCP("logimind-settlement-tools")


@mcp.tool()
def get_settlement_history(driver_id: str, weeks: int = 6) -> list[dict]:
    """Return the last N weeks of settlement records for a driver.

    Each record has: week_ending, gross_pay, fuel_deduction,
    escrow_deduction, net_pay. Returns an empty list if the driver has
    no settlement rows (not an error) -- the caller decides how to word
    that to the user.
    """
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


if __name__ == "__main__":
    # stdio transport: the client launches this file as a child process
    # and talks to it over stdin/stdout -- no network port to open or
    # secure, which is why stdio (not SSE/HTTP) is the right choice for
    # a tool that only ever has one caller (this app's own backend).
    mcp.run(transport="stdio")
