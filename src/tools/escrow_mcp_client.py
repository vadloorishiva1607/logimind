"""
src/tools/escrow_mcp_client.py

The Escrow agent's MCP client. Unlike the Performance client, this agent
has TWO tools to choose from, so the LLM's choice really matters:
  get_escrow_balance  -> "what is the balance now?"
  get_escrow_history  -> "why did the balance change?"
The LLM reads the tools' descriptions and chooses; code checks the choice
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
    "escrow": {
        "command": sys.executable,
        "args": [_SERVER_PATH],
        "transport": "stdio",
        # Pass the full environment so the subprocess gets DATABASE_URL.
        "env": os.environ.copy(),
    }
}

# Tools this agent may run, even though the server offers more.
_ALLOWED_TOOLS = {"get_escrow_balance", "get_escrow_history"}
# If the LLM's choice can't be used, history is the safer default: it also
# contains the current balance (the newest row's balance_after).
_DEFAULT_TOOL = "get_escrow_history"
_DEFAULT_LIMIT = 10

_TOOL_PROMPT = (
    "You fetch data for LogiMind's Escrow agent. Call exactly one tool to "
    "get the data needed to answer the question. The driver is {driver_id}."
    "\n\nQuestion: {question}"
)


def _check_choice(call, driver_id: str):
    """Validate the LLM's tool choice. Returns (tool_name, args) that are safe to run."""
    if call and call.get("name") in _ALLOWED_TOOLS:
        name, args = call["name"], dict(call.get("args") or {})
    else:
        log.warning("Escrow agent: rejected LLM tool choice %r, using default", call)
        name, args = _DEFAULT_TOOL, {}

    if name == "get_escrow_balance":
        # This tool takes only the driver ID; drop anything else the LLM sent.
        return name, {"driver_id": driver_id}

    try:
        limit = int(args.get("limit", _DEFAULT_LIMIT))
    except (TypeError, ValueError):
        limit = _DEFAULT_LIMIT
    # Never trust the LLM with the driver ID; keep limit in 1-50.
    return name, {"driver_id": driver_id, "limit": max(1, min(limit, 50))}


async def _fetch_async(driver_id: str, question: str):
    client = MultiServerMCPClient(_SERVERS)
    tools = await client.get_tools()
    tools_by_name = {t.name: t for t in tools}

    # 1. The LLM sees every tool on the server and chooses one, with arguments.
    llm = get_llm().bind_tools(tools)
    msg = await llm.ainvoke(_TOOL_PROMPT.format(driver_id=driver_id, question=question))
    call = msg.tool_calls[0] if getattr(msg, "tool_calls", None) else None

    # 2. Code checks the choice before running anything.
    name, args = _check_choice(call, driver_id)
    if name not in tools_by_name:
        raise RuntimeError(f"{name} tool not found on the MCP server")
    log.info("Escrow agent: LLM chose %s(%s)", name, args)

    # 3. Run the checked tool call. Both tools return one dict, which can
    # arrive as a JSON string depending on the adapter version.
    result = await tools_by_name[name].ainvoke(args)
    if isinstance(result, str):
        result = json.loads(result)
    return name, result


def get_escrow_data_via_mcp(driver_id: str, question: str = ""):
    """Sync wrapper. Returns (tool_name, data) so the agent knows which
    kind of data it got: a current balance, or a transaction history."""
    return asyncio.run(_fetch_async(driver_id, question))