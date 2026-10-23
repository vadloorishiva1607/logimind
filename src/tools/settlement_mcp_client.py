"""
src/tools/settlement_mcp_client.py

The Settlement agent's MCP client. Spawns `mcp_server.py` (repo root) as
a stdio subprocess and calls its `get_settlement_history` tool, instead
of the agent importing a DB function directly. See
LogiMind_Advanced_Techniques.md Section 7.

Note on placement: destination path is `src/tools/settlement_mcp_client.py`
-- the two `parents[...]` hops below assume that location
(src/tools/<this file> -> src/tools -> src -> repo root). If you place
this file elsewhere, fix `_SERVER_PATH` accordingly.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

_SERVER_PATH = str(Path(__file__).resolve().parents[2] / "mcp_server.py")

_SERVERS = {
    "settlement": {
        "command": sys.executable,
        "args": [_SERVER_PATH],
        "transport": "stdio",
        # The MCP SDK's stdio transport does NOT inherit the parent
        # process's full environment by default -- it spawns the
        # subprocess with a restricted environment unless `env` is given
        # explicitly. Without this, mcp_server.py falls back to its own
        # hardcoded DATABASE_URL default (localhost) regardless of what
        # DATABASE_URL is set to here, which is exactly the bug found
        # live-testing this on 2026-10-04 (see Testing &
        # Verification Log): the main app connected to RDS fine, but the
        # MCP subprocess it spawned silently didn't get the same
        # DATABASE_URL (or OPENAI_API_KEY, if mcp_server.py ever needs
        # it) passed through.
        "env": os.environ.copy(),
    }
}


async def _fetch_async(driver_id: str, weeks: int = 6) -> list[dict]:
    client = MultiServerMCPClient(_SERVERS)
    tools = await client.get_tools()
    try:
        tool = next(t for t in tools if t.name == "get_settlement_history")
    except StopIteration:
        raise RuntimeError(
            "get_settlement_history tool not found on the settlement MCP "
            "server -- check mcp_server.py started correctly"
        )

    result = await tool.ainvoke({"driver_id": driver_id, "weeks": weeks})
    # A list return value arrives as one JSON string per item (MCP sends
    # each list item as its own text content), or as a single JSON string,
    # depending on the adapter version. Normalize to a list of dicts.
    if isinstance(result, str):
        result = json.loads(result)
    if isinstance(result, list):
        result = [json.loads(r) if isinstance(r, str) else r for r in result]
    return result


def get_settlement_history_via_mcp(driver_id: str, weeks: int = 6) -> list[dict]:
    """Sync wrapper -- the rest of the graph (agent.py's settlement_node)
    is synchronous, so this hides the async MCP client behind a plain
    function call. Each call spins up a fresh subprocess + session;
    that's a deliberate simplicity-over-throughput tradeoff appropriate
    for this capstone's traffic, not something to optimize unasked.
    """
    return asyncio.run(_fetch_async(driver_id, weeks))