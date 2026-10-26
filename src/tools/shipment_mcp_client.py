"""
src/tools/shipment_mcp_client.py

The Shipment agent's MCP client. Two tools to choose from:
  get_shipment          -> one shipment by ID ("Why is S10023 delayed?")
  get_driver_shipments  -> a driver's recent shipments ("Is D102 often late?")
The LLM chooses; code checks the choice and supplies the IDs.

When the LLM picks get_shipment, code ALSO fetches that driver's other
shipments, so the agent can check whether the delay is a pattern. That
second lookup is a fixed rule, so it is not left to the LLM.
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
    "shipment": {
        "command": sys.executable,
        "args": [_SERVER_PATH],
        "transport": "stdio",
        # Pass the full environment so the subprocess gets DATABASE_URL.
        "env": os.environ.copy(),
    }
}

_ALLOWED_TOOLS = {"get_shipment", "get_driver_shipments"}
_DEFAULT_LIMIT = 10

_TOOL_PROMPT = (
    "You fetch data for LogiMind's Shipment agent. Call exactly one tool to "
    "get the data needed to answer the question.\n"
    "Shipment ID: {shipment_id}\nDriver ID: {driver_id}\n\nQuestion: {question}"
)


def _check_choice(call, shipment_id, driver_id):
    """Validate the LLM's tool choice. Returns (tool_name, args) that are safe to run.

    Code, not the LLM, supplies the IDs. If the chosen tool needs an ID we
    don't have, code switches to the tool we can actually run.
    """
    name = call.get("name") if call else None
    if name not in _ALLOWED_TOOLS:
        log.warning("Shipment agent: rejected LLM tool choice %r", call)
        name = None

    if name == "get_driver_shipments" and driver_id:
        try:
            limit = int((call.get("args") or {}).get("limit", _DEFAULT_LIMIT))
        except (TypeError, ValueError):
            limit = _DEFAULT_LIMIT
        return name, {"driver_id": driver_id, "limit": max(1, min(limit, 50))}
    if shipment_id:
        return "get_shipment", {"shipment_id": shipment_id}
    if driver_id:
        return "get_driver_shipments", {"driver_id": driver_id, "limit": _DEFAULT_LIMIT}
    raise ValueError("No shipment ID or driver ID to look up")


def _parse(result):
    # Both tools return one dict, which can arrive as a JSON string.
    return json.loads(result) if isinstance(result, str) else result


async def _fetch_async(question: str, shipment_id, driver_id) -> dict:
    client = MultiServerMCPClient(_SERVERS)
    tools = await client.get_tools()
    tools_by_name = {t.name: t for t in tools}

    # 1. The LLM sees every tool on the server and chooses one.
    llm = get_llm().bind_tools(tools)
    msg = await llm.ainvoke(_TOOL_PROMPT.format(
        shipment_id=shipment_id or "not given",
        driver_id=driver_id or "not given",
        question=question,
    ))
    call = msg.tool_calls[0] if getattr(msg, "tool_calls", None) else None

    # 2. Code checks the choice before running anything.
    name, args = _check_choice(call, shipment_id, driver_id)
    for needed in (name, "get_driver_shipments"):
        if needed not in tools_by_name:
            raise RuntimeError(f"{needed} tool not found on the MCP server")
    log.info("Shipment agent: LLM chose %s; running %s(%s)",
             call.get("name") if call else None, name, args)

    # 3. Run the checked call.
    result = _parse(await tools_by_name[name].ainvoke(args))
    data = {"tool": name, "shipment": None, "driver_rows": []}
    if name == "get_shipment":
        data["shipment"] = result.get("shipment")
        # 4. Fixed rule: also fetch this driver's other shipments, to check
        # whether the delay is a pattern.
        if data["shipment"]:
            more = _parse(await tools_by_name["get_driver_shipments"].ainvoke(
                {"driver_id": data["shipment"]["driver_id"], "limit": _DEFAULT_LIMIT}))
            data["driver_rows"] = more.get("rows", [])
    else:
        data["driver_rows"] = result.get("rows", [])
    return data


def get_shipment_data_via_mcp(question: str, shipment_id=None, driver_id=None) -> dict:
    """Sync wrapper. Returns {"tool", "shipment", "driver_rows"}:
    `shipment` is the one shipment asked about (or None), and
    `driver_rows` are the driver's recent shipments, newest first."""
    return asyncio.run(_fetch_async(question, shipment_id, driver_id))