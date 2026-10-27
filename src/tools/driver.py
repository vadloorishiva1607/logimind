"""
src/tools/driver.py -- Driver agent.

Answers "who is this driver and how are they doing?": profile, status,
onboarding, and a one-line snapshot from each data area (pay, safety,
escrow, shipments). Data comes from the MCP tool get_driver_overview.

Code decides everything that matters:
  - which snapshot areas the user's ROLE may see (hidden areas are never
    sent to the LLM, so they cannot leak);
  - record checks: onboarding taking too long, or a driver still marked
    "onboarding" who is already working, or a suspended driver.
The LLM only explains. A safety net guarantees the record flags and any
"not available for your role" note reach the user.
"""
import logging
from datetime import date

from src.agent.output_guardrail import check_answer
from src.models.llm_client import get_llm
from src.tools.driver_mcp_client import get_driver_overview_via_mcp

log = logging.getLogger(__name__)

# Snapshot areas each role may NOT see. Money (pay, escrow) is for Finance
# and Compliance only. Unknown roles get the most restricted view
# (fail closed), the same default the RAG agent uses.
_HIDDEN_BY_ROLE = {
    "dispatcher": {"settlement", "escrow"},
    "finance": set(),
    "compliance": set(),
}
_MOST_RESTRICTED = "dispatcher"

_AREA_NAMES = {"settlement": "latest pay", "performance": "latest safety and on-time scores",
               "escrow": "escrow balance", "shipments": "shipment counts"}

# Driver Onboarding SOP: "Onboarding is expected to complete within 30 days
# of the start date. Drivers exceeding 45 days in onboarding status should be
# flagged for Driver Manager follow-up".
ONBOARDING_EXPECTED_DAYS = 30
ONBOARDING_FLAG_DAYS = 45

DRIVER_SYSTEM_PROMPT = """You are LogiMind's Driver agent. Answer using \
ONLY the driver data provided below, and quote specific figures and dates \
from it.

Rules:
1. Give the driver's name, status and how long since onboarding, then the \
snapshot figures you were given.
2. Report every record flag, with its figures, and the suggested check.
3. If some information is not available for the user's role, say so \
plainly. Never guess or estimate hidden information.
4. For detail on any one area, suggest asking about it (for example "Why \
did D102's settlement drop?").
5. Never state that any action has been taken.
6. Write the answer in your own plain sentences. Do not copy the data \
lines or their labels.
7. If the data is insufficient to answer, say so instead of guessing."""


def _money(x):
    return "n/a" if x is None else f"${x:,.2f}"


def visible_areas(role):
    """Return (role_used, hidden_areas). Unknown or missing roles are restricted."""
    role_used = role if role in _HIDDEN_BY_ROLE else _MOST_RESTRICTED
    return role_used, _HIDDEN_BY_ROLE[role_used]


def as_of_date(snapshot: dict):
    """The latest date in the data. Day counts use this, not today, so they
    describe the data the user is looking at."""
    dates = [snapshot[k][f] for k, f in (("settlement", "week_ending"),
             ("performance", "period_end"), ("escrow", "as_of")) if snapshot.get(k)]
    return date.fromisoformat(max(dates)) if dates else date.today()


def record_flags(driver: dict, snapshot: dict, as_of: date) -> list[tuple[str, str]]:
    """Checks on the driver record itself, computed by code.
    Returns (kind, sentence) pairs."""
    flags = []
    days = None
    if driver.get("onboarding_date"):
        days = (as_of - date.fromisoformat(driver["onboarding_date"])).days

    if driver["status"] == "onboarding":
        if days is not None and days > ONBOARDING_FLAG_DAYS:
            flags.append(("onboarding_long",
                f"Onboarding taking longer than expected: still 'onboarding' {days} days after "
                f"{driver['onboarding_date']}. Per the Driver Onboarding SOP, onboarding should "
                f"complete within {ONBOARDING_EXPECTED_DAYS} days, and drivers past "
                f"{ONBOARDING_FLAG_DAYS} days are flagged for Driver Manager follow-up "
                "(usually a missing document or a scheduling delay)."))
        if snapshot.get("settlement") or snapshot.get("performance"):
            flags.append(("status_outdated",
                "Status may be out of date: marked 'onboarding' but already has pay or "
                "performance records. Suggest checking the driver record."))
    elif driver["status"] == "suspended":
        flags.append(("suspended", "Driver is suspended: figures are shown for reference only."))
    return flags


def build_driver_context(driver: dict, snapshot: dict, role) -> tuple[str, dict]:
    role_used, hidden = visible_areas(role)
    as_of = as_of_date(snapshot)
    days = (as_of - date.fromisoformat(driver["onboarding_date"])).days if driver.get("onboarding_date") else None
    flags = record_flags(driver, snapshot, as_of)

    lines = [
        f"DRIVER {driver['driver_id']}: {driver['full_name']}, status {driver['status']}, "
        f"driver type {driver.get('role') or 'n/a'}.",
        f"Onboarded {driver.get('onboarding_date') or 'date not recorded'}"
        + ("" if days is None else f": {days} days before {as_of.isoformat()}, the latest data date") + ".",
        "",
        f"SNAPSHOT (as of {as_of.isoformat()}):",
    ]
    s = snapshot
    if "settlement" not in hidden:
        lines.append(f"- Latest pay: net pay {_money(s['settlement']['net_pay'])} for the week ending "
                     f"{s['settlement']['week_ending']}." if s.get("settlement") else "- Latest pay: no settlement data.")
    if "performance" not in hidden:
        lines.append(f"- Latest performance: safety score {s['performance']['safety_score']}, on-time "
                     f"{s['performance']['on_time_pct']}% (period ending {s['performance']['period_end']})."
                     if s.get("performance") else "- Latest performance: no performance data.")
    if "escrow" not in hidden:
        lines.append(f"- Escrow balance: {_money(s['escrow']['balance'])} as of {s['escrow']['as_of']}."
                     if s.get("escrow") else "- Escrow balance: no escrow data.")
    if "shipments" not in hidden:
        sh = s.get("shipments") or {"total": 0, "delayed": 0}
        lines.append(f"- Shipments: {sh['total']} on record, {sh['delayed']} delayed.")

    hidden_names = [_AREA_NAMES[a] for a in ("settlement", "performance", "escrow", "shipments") if a in hidden]
    if hidden_names:
        lines.append(f"- NOT AVAILABLE FOR YOUR ROLE ({role_used}): " + ", ".join(hidden_names) + ".")

    lines += ["", "RECORD FLAGS:"]
    lines += [f"- {text}" for _, text in flags] if flags else ["- None."]
    return "\n".join(lines), {"flags": flags, "hidden": hidden_names, "role": role_used}


# Words that show the answer reported each kind of flag.
_FLAG_WORDS = {
    "onboarding_long": ("longer than expected", "too long", "taking longer"),
    "status_outdated": ("out of date", "outdated", "not up to date"),
    "suspended": ("suspended",),
}


def _ensure_reported(answer: str, facts: dict) -> str:
    """Safety net: record flags and the role note must reach the user."""
    text = answer.lower()
    notes = [msg for kind, msg in facts["flags"]
             if not any(w in text for w in _FLAG_WORDS[kind])]
    if facts["hidden"] and "role" not in text:
        notes.append(f"Not available for your role ({facts['role']}): " + ", ".join(facts["hidden"]) + ".")
    return answer if not notes else answer + "\n\nFlagged by rules: " + " ".join(notes)


def run_driver_agent(db, driver_id: str, question: str = "", role=None) -> str:
    """`db` is accepted for interface consistency with the other agents; unused."""
    try:
        driver, snapshot = get_driver_overview_via_mcp(driver_id)
    except Exception:
        return (f"I couldn't retrieve the driver record for {driver_id} right now. "
                "Please try again in a moment.")

    if driver is None:
        return (f"I can't find driver {driver_id}. Please check the driver ID "
                "(for example D102).")

    context, facts = build_driver_context(driver, snapshot, role)
    if facts["hidden"]:
        # The DB audit_log write is deferred (needs an INSERT-only grant);
        # until then, every role-based hiding is logged here.
        log.warning("Driver agent: hid %s from role=%s for %s",
                    facts["hidden"], facts["role"], driver_id)

    llm = get_llm()
    prompt = (
        f"{DRIVER_SYSTEM_PROMPT}\n\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    answer = check_answer(answer, context, "driver")
    return _ensure_reported(answer, facts)