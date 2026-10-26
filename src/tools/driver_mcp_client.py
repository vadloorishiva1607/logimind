"""
src/tools/driver_mcp_client.py

The Driver agent's MCP client. One tool with one fixed argument, so code
calls it directly: no LLM tool choice (same reasoning as the Fuel client).
"""
import asyncio
import json
import os
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

_SERVER_PATH = str(Path(__file__).resolve().parents[2] / "mcp_server.py")

_SERVERS = {
    "driver": {
        "command": sys.executable,
        "args": [_SERVER_PATH],
        "transport": "stdio",
        # Pass the full environment so the subprocess gets DATABASE_URL.
        "env": os.environ.copy(),
    }
}


async def _fetch_async(driver_id: str) -> dict:
    client = MultiServerMCPClient(_SERVERS)
    tools = {t.name: t for t in await client.get_tools()}
    if "get_driver_overview" not in tools:
        raise RuntimeError("get_driver_overview tool not found on the MCP server")
    result = await tools["get_driver_overview"].ainvoke({"driver_id": driver_id})
    return json.loads(result) if isinstance(result, str) else result


def get_driver_overview_via_mcp(driver_id: str):
    """Sync wrapper. Returns (driver, snapshot); both None if not found."""
    data = asyncio.run(_fetch_async(driver_id))
    return data.get("driver"), data.get("snapshot")