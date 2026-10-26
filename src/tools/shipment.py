"""
src/tools/shipment.py -- Shipment / Risk agent.

Answers questions about a shipment's status, delay and risk, or about a
driver's shipments in general. Data comes from the MCP server via
shipment_mcp_client.py, where the LLM chooses between two tools.

Code computes the facts: the delay in hours, the recorded causes, whether
the driver's delays are a repeated pattern, and the risk level. The LLM
only explains. The output guardrail checks the numbers, and a safety net
guarantees the risk level and any "cause unconfirmed" reach the user.

Design Doc, Shipment/Risk agent negative scenarios:
  - no delay reason recorded  -> "cause unconfirmed", never a guessed cause
  - several causes recorded   -> list them all, don't pick one
"""
import re
from datetime import datetime

from src.agent.output_guardrail import check_answer
from src.models.llm_client import get_llm
from src.tools.shipment_mcp_client import get_shipment_data_via_mcp

# Risk rules. These thresholds are demo assumptions, stated openly in the
# context so the answer can explain them; a real fleet would set its own.
HIGH_DELAY_HOURS = 4.0      # a delay this long or longer is high risk
MIN_HISTORY = 3             # fewer shipments than this: too few to call a pattern
PATTERN_SHARE = 0.5         # half or more of recent shipments delayed = pattern

SHIPMENT_SYSTEM_PROMPT = """You are LogiMind's Shipment and Risk agent. \
Answer using ONLY the shipment data provided below, and quote specific \
shipment IDs, times and figures from it.

Rules:
1. Use the precomputed figures and findings exactly as given. Do not \
recompute them.
2. For one shipment, give the route, the scheduled and actual arrival \
times, the delay in hours, and the status.
3. List every recorded cause of the delay. If there is more than one, say \
the data does not show which contributed more. If the cause is \
unconfirmed, say so and do not guess one.
4. State the risk level and the reason given for it.
5. Describe the driver's shipment history and whether it shows a repeated \
pattern, using the pattern finding exactly as given.
6. You may suggest a next step for a dispatcher, but never state that any \
action has been taken.
7. Write the answer in your own plain sentences. Do not copy the data \
lines or their labels.
8. If the data is insufficient to answer, say so instead of guessing."""


def _time(iso):
    """'2026-09-26T14:45:23.479061' -> '2026-09-26 14:45' (easier to read and check)."""
    return "not recorded" if not iso else datetime.fromisoformat(iso).strftime("%Y-%m-%d %H:%M")


def delay_hours(s: dict):
    """Hours late (negative = early), or None if it hasn't arrived yet."""
    if not s.get("scheduled_eta") or not s.get("actual_eta"):
        return None
    diff = datetime.fromisoformat(s["actual_eta"]) - datetime.fromisoformat(s["scheduled_eta"])
    return round(diff.total_seconds() / 3600, 1)


def is_delayed(s: dict) -> bool:
    hours = delay_hours(s)
    return s.get("status") == "delayed" or (hours is not None and hours > 0)


def causes(reason):
    """'Traffic + driver rest-break delay' -> ['Traffic', 'driver rest-break delay']."""
    if not reason:
        return []
    parts = re.split(r"\s*(?:\+|,|;|\band\b)\s*", reason)
    return [p for p in parts if p]


def pattern_finding(rows: list[dict]):
    """Return (is_pattern, sentence) about the driver's recent shipments."""
    total = len(rows)
    delayed = sum(is_delayed(r) for r in rows)
    if total < MIN_HISTORY:
        return False, (f"{delayed} of {total} recent shipments delayed. Only {total} "
                       f"shipment(s) on record: too few to call a pattern "
                       f"(at least {MIN_HISTORY} are needed).")
    if delayed >= 2 and delayed / total >= PATTERN_SHARE:
        return True, f"Repeated pattern: {delayed} of {total} recent shipments delayed."
    return False, f"No repeated pattern: {delayed} of {total} recent shipments delayed."


def assess_risk(shipment, rows: list[dict], is_pattern: bool):
    """Return (level, reason). Code decides the risk level, not the LLM."""
    if is_pattern:
        return "HIGH", "the driver's delays are a repeated pattern"
    if shipment is None:  # a question about the driver, not one shipment
        if any(is_delayed(r) for r in rows):
            return "MEDIUM", "the driver has delayed shipments, but not a repeated pattern"
        return "LOW", "none of the driver's recent shipments were delayed"
    hours = delay_hours(shipment)
    if hours is not None and hours >= HIGH_DELAY_HOURS:
        return "HIGH", f"a delay of {hours} hours is at or above the {HIGH_DELAY_HOURS:g}-hour threshold"
    if is_delayed(shipment):
        return "MEDIUM", f"the shipment is delayed, below the {HIGH_DELAY_HOURS:g}-hour threshold"
    return "LOW", "the shipment is on time"


def _describe(s: dict) -> str:
    hours = delay_hours(s)
    delay = "not arrived yet" if hours is None else f"delay {hours} hours"
    reason = s.get("delay_reason") or ("cause unconfirmed" if is_delayed(s) else "none")
    return (f"{s['shipment_id']}: {s['origin']} -> {s['destination']}, scheduled "
            f"{_time(s['scheduled_eta'])}, actual {_time(s['actual_eta'])}, "
            f"status {s['status']}, {delay}, reason: {reason}")


def build_shipment_context(data: dict):
    """Turn the MCP data into plain facts. Returns (context, facts) where
    facts holds what the safety net must guarantee in the answer."""
    shipment, rows = data["shipment"], data["driver_rows"]
    is_pattern, pattern_sentence = pattern_finding(rows)
    level, why = assess_risk(shipment, rows, is_pattern)
    unconfirmed = []
    lines = []

    if shipment:
        hours = delay_hours(shipment)
        found = causes(shipment.get("delay_reason"))
        lines += [
            f"SHIPMENT {shipment['shipment_id']} (driver {shipment['driver_id']}): "
            f"{shipment['origin']} -> {shipment['destination']}",
            f"Scheduled arrival {_time(shipment['scheduled_eta'])}, actual arrival "
            f"{_time(shipment['actual_eta'])}, status {shipment['status']}.",
            "Delay: not arrived yet." if hours is None else f"Delay: {hours} hours.",
        ]
        if found:
            lines.append(f"Recorded causes ({len(found)}): " + "; ".join(found) + "."
                         + (" The data does not say which contributed more." if len(found) > 1 else ""))
        elif is_delayed(shipment):
            lines.append("Cause unconfirmed: no delay reason was recorded.")
            unconfirmed.append(shipment["shipment_id"])

    driver = shipment["driver_id"] if shipment else (rows[0]["driver_id"] if rows else "unknown")
    lines += [
        f"Risk level: {level}, because {why}.",
        "",
        f"DRIVER {driver} SHIPMENT HISTORY: {pattern_sentence}",
    ]
    lines += [f"- {_describe(r)}" for r in rows]
    if not shipment:
        unconfirmed += [r["shipment_id"] for r in rows if is_delayed(r) and not r.get("delay_reason")]

    return "\n".join(lines), {"risk": level, "risk_reason": why, "unconfirmed": unconfirmed}


def _ensure_reported(answer: str, facts: dict) -> str:
    """Safety net: the risk level and any unconfirmed cause must reach the
    user, even if the LLM left them out."""
    text = answer.lower()
    notes = []
    if "risk" not in text or not re.search(rf"\b{facts['risk'].lower()}\b", text):
        notes.append(f"Risk level (set by rules): {facts['risk']}, because {facts['risk_reason']}.")
    for sid in facts["unconfirmed"]:
        if "unconfirmed" not in text:
            notes.append(f"Cause unconfirmed for {sid}: no delay reason was recorded.")
    return answer if not notes else answer + "\n\n" + " ".join(notes)


def run_shipment_agent(db, question: str, shipment_id=None, driver_id=None) -> str:
    """`db` is accepted for interface consistency with the other agents; unused."""
    try:
        data = get_shipment_data_via_mcp(question, shipment_id=shipment_id, driver_id=driver_id)
    except Exception:
        return "I couldn't retrieve shipment data right now. Please try again in a moment."

    if data["tool"] == "get_shipment" and not data["shipment"]:
        return (f"I can't find shipment {shipment_id}. "
                "Double-check the shipment ID (for example S10023).")
    if not data["shipment"] and not data["driver_rows"]:
        return f"I don't have any shipments on record for {driver_id}."

    context, facts = build_shipment_context(data)
    llm = get_llm()
    prompt = (
        f"{SHIPMENT_SYSTEM_PROMPT}\n\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    answer = check_answer(answer, context, "shipment")
    return _ensure_reported(answer, facts)