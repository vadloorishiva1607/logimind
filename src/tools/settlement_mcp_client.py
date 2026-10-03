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
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

_SERVER_PATH = str(Path(__file__).resolve().parents[2] / "mcp_server.py")

_SERVERS = {
    "settlement": {
        "command": sys.executable,
        "args": [_SERVER_PATH],
        "transport": "stdio",
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
    # langchain-mcp-adapters returns the tool's declared return value;
    # depending on adapter version this can arrive as a JSON string
    # rather than an already-parsed list -- handle both.
    if isinstance(result, str):
        result = json.loads(result)
    return result


def get_settlement_history_via_mcp(driver_id: str, weeks: int = 6) -> list[dict]:
    """Sync wrapper -- the rest of the graph (agent.py's settlement_node)
    is synchronous, so this hides the async MCP client behind a plain
    function call. Each call spins up a fresh subprocess + session;
    that's a deliberate simplicity-over-throughput tradeoff appropriate
    for this capstone's traffic, not something to optimize unasked.
    """
    return asyncio.run(_fetch_async(driver_id, weeks))
