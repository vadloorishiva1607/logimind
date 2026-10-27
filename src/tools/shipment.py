"""
src/tools/shipment.py -- Shipment / Risk agent.

Answers questions about a shipment's status, delay and risk, or about a
driver's shipments in general. Data comes from the MCP server via
shipment_mcp_client.py, where the LLM chooses between two tools.

Code computes the facts: the delay in hours, the recorded causes, whether
the driver's delays are a repeated pattern, and the risk level. The LLM
only explains. The output guardrail checks the numbers, and a safety net
guarantees the risk level and any "cause unconfirmed" reach the user.

Rules follow the Shipment Delay Risk SOP (aligned 2026-10-26):
  - "delayed" = actual ETA more than 60 minutes after the scheduled ETA
  - a single delay with a known, documented cause is routine: no escalation
  - 3 or more delays for the same driver within 2 weeks: escalate as a
    systemic risk (the SOP also covers the same route; route data is not
    fetched yet, so only the driver pattern is checked)
  - no recorded reason: flag "cause unconfirmed", never guess a cause
  - several causes recorded: list them all, don't pick one
"""
import re
from datetime import datetime, timedelta

from src.agent.output_guardrail import check_answer
from src.models.llm_client import get_llm
from src.tools.shipment_mcp_client import get_shipment_data_via_mcp

# Shipment Delay Risk SOP thresholds.
DELAY_MINUTES = 60      # more than this late = "delayed"
PATTERN_COUNT = 3       # this many delays ...
PATTERN_DAYS = 14       # ... within this many days = systemic risk, escalate

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
4. State the escalation level and the reason given for it, naming the \
Shipment Delay Risk SOP.
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
    """SOP: delayed = more than 60 minutes late. Uses the times when both are
    recorded, otherwise the status."""
    hours = delay_hours(s)
    if hours is not None:
        return hours * 60 > DELAY_MINUTES
    return s.get("status") == "delayed"


def causes(reason):
    """'Traffic + driver rest-break delay' -> ['Traffic', 'driver rest-break delay']."""
    if not reason:
        return []
    parts = re.split(r"\s*(?:\+|,|;|\band\b)\s*", reason)
    return [p for p in parts if p]


def pattern_finding(rows: list[dict]):
    """SOP: 3 or more delays for the same driver within 2 weeks is a systemic
    risk. Counts delays in the 14 days up to the driver's latest shipment.
    Returns (is_pattern, sentence)."""
    dated = [r for r in rows if r.get("scheduled_eta")]
    if not dated:
        return False, "No dated shipments to check for a pattern."
    end = max(datetime.fromisoformat(r["scheduled_eta"]) for r in dated)
    start = end - timedelta(days=PATTERN_DAYS)
    recent = [r for r in dated if datetime.fromisoformat(r["scheduled_eta"]) > start]
    delayed = sum(is_delayed(r) for r in recent)
    window = f"the {PATTERN_DAYS} days up to {end.strftime('%Y-%m-%d')}"
    if delayed >= PATTERN_COUNT:
        return True, (f"Systemic risk: {delayed} delays for this driver in {window} "
                      f"(the SOP escalates {PATTERN_COUNT} or more within 2 weeks).")
    return False, (f"{delayed} delay(s) for this driver in {window}, below the SOP's "
                   f"{PATTERN_COUNT}-delay pattern threshold. (Route patterns are not checked.)")


def assess_risk(shipment, rows: list[dict], is_pattern: bool):
    """Return (level, reason) under the Shipment Delay Risk SOP. Code decides, not the LLM."""
    if is_pattern:
        return "ESCALATE", ("the driver has 3 or more delays within 2 weeks, which the "
                            "Shipment Delay Risk SOP treats as a systemic risk")
    targets = [shipment] if shipment else rows
    delayed = [s for s in targets if is_delayed(s)]
    if any(not s.get("delay_reason") for s in delayed):
        return "CAUSE UNCONFIRMED", ("a delay has no recorded reason; the SOP says to flag it "
                                     "as 'cause unconfirmed' rather than guess a cause")
    if delayed:
        return "ROUTINE", ("a delay with a known, documented cause and no repeated pattern is "
                           "routine and needs no escalation under the Shipment Delay Risk SOP")
    return "ROUTINE", "no delay over the SOP's 60-minute rule"


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
        f"Escalation level (Shipment Delay Risk SOP): {level}, because {why}.",
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
    if facts["risk"].lower() not in text:
        notes.append(f"Escalation level (Shipment Delay Risk SOP): {facts['risk']}, "
                     f"because {facts['risk_reason']}.")
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