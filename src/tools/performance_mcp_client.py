"""
src/tools/performance_mcp_client.py

The Performance agent's MCP client. Spawns `mcp_server.py` (repo root) as
a stdio subprocess. The LLM sees every tool the server offers and chooses
which one to call, and with what arguments; code then checks that choice
before running it ("the LLM chooses, code checks").
"""
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

from src.models.llm_client import get_llm

log = logging.getLogger(__name__)

_SERVER_PATH = str(Path(__file__).resolve().parents[2] / "mcp_server.py")

_SERVERS = {
    "performance": {
        "command": sys.executable,
        "args": [_SERVER_PATH],
        "transport": "stdio",
        # Pass the full environment so the subprocess gets DATABASE_URL
        # (see the 2026-10-04 bug in the Testing & Verification Log).
        "env": os.environ.copy(),
    }
}

# Tools this agent is allowed to run, even though the server offers more.
_ALLOWED_TOOLS = {"get_performance_history"}
_DEFAULT_TOOL = "get_performance_history"

_TOOL_PROMPT = (
    "You fetch data for LogiMind's Performance agent. Call exactly one tool "
    "to get the data needed to answer the question. The driver is "
    "{driver_id}. Choose how many periods to fetch from the question; if it "
    "doesn't say, use {default}.\n\nQuestion: {question}"
)


def _check_choice(call, driver_id: str, default_periods: int):
    """Validate the LLM's tool choice. Returns (tool_name, args) that are safe to run."""
    if call and call.get("name") in _ALLOWED_TOOLS:
        name, args = call["name"], dict(call.get("args") or {})
    else:
        # No call, or a tool this agent may not use -> fall back to the default.
        log.warning("Performance agent: rejected LLM tool choice %r, using default", call)
        name, args = _DEFAULT_TOOL, {}

    args["driver_id"] = driver_id  # never trust the LLM with the driver ID
    try:
        periods = int(args.get("periods", default_periods))
    except (TypeError, ValueError):
        periods = default_periods
    args["periods"] = max(1, min(periods, 52))
    return name, args


async def _fetch_async(driver_id: str, question: str, default_periods: int) -> dict:
    client = MultiServerMCPClient(_SERVERS)
    tools = await client.get_tools()
    tools_by_name = {t.name: t for t in tools}

    # 1. The LLM sees every tool on the server and chooses one, with arguments.
    llm = get_llm().bind_tools(tools)
    msg = await llm.ainvoke(_TOOL_PROMPT.format(
        driver_id=driver_id, question=question, default=default_periods))
    call = msg.tool_calls[0] if getattr(msg, "tool_calls", None) else None

    # 2. Code checks the choice before running anything.
    name, args = _check_choice(call, driver_id, default_periods)
    if name not in tools_by_name:
        raise RuntimeError(f"{name} tool not found on the MCP server")
    log.info("Performance agent: LLM chose %s(%s)", name, args)

    # 3. Run the checked tool call.
    result = await tools_by_name[name].ainvoke(args)
    # Depending on the adapter version, the result can arrive as a JSON
    # string rather than an already-parsed dict -- handle both.
    if isinstance(result, str):
        result = json.loads(result)
    return result


def get_performance_history_via_mcp(driver_id: str, question: str = "", periods: int = 6):
    """Sync wrapper, like the settlement client. Returns (rows, fleet):
    rows newest first, fleet = latest-period average or None."""
    data = asyncio.run(_fetch_async(driver_id, question, periods))
    return data.get("rows", []), data.get("fleet")