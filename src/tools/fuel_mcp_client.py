"""
src/tools/fuel_mcp_client.py

The Fuel agent's MCP client. There is only one fuel tool and its
arguments are fixed, so code calls it directly: no LLM tool choice. An
LLM call here would add time and cost without making any real decision.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

_SERVER_PATH = str(Path(__file__).resolve().parents[2] / "mcp_server.py")

_SERVERS = {
    "fuel": {
        "command": sys.executable,
        "args": [_SERVER_PATH],
        "transport": "stdio",
        # Pass the full environment so the subprocess gets DATABASE_URL.
        "env": os.environ.copy(),
    }
}


async def _fetch_async(driver_id: str, days: int) -> dict:
    client = MultiServerMCPClient(_SERVERS)
    tools = {t.name: t for t in await client.get_tools()}
    if "get_fuel_history" not in tools:
        raise RuntimeError("get_fuel_history tool not found on the MCP server")
    result = await tools["get_fuel_history"].ainvoke({"driver_id": driver_id, "days": days})
    # One dict, which can arrive as a JSON string.
    return json.loads(result) if isinstance(result, str) else result


def get_fuel_history_via_mcp(driver_id: str, days: int = 42):
    """Sync wrapper. Returns (rows, fleet): rows newest first,
    fleet = averages over the same period, or None."""
    data = asyncio.run(_fetch_async(driver_id, days))
    return data.get("rows", []), data.get("fleet")