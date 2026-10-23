"""
src/tools/performance.py -- Performance agent.

Answers questions about a driver's on-time %, safety score, late deliveries
and incidents, compared with their own history and the fleet average.
Code fetches the data and computes the trend; the LLM only explains it.
Data comes from the MCP server (mcp_server.py) via performance_mcp_client.py,
where the LLM chooses the tool and code checks that choice.
"""
from src.models.llm_client import get_llm
from src.tools.performance_mcp_client import get_performance_history_via_mcp
from src.agent.output_guardrail import check_answer
PERFORMANCE_SYSTEM_PROMPT = """You are LogiMind's Performance agent. Answer \
using ONLY the performance data provided below, and quote specific periods \
and figures from it.

Rules:
1. Unless the question asks about a different period, focus on the LATEST \
PERIOD and the TREND facts.
2. Use the precomputed figures exactly as given. Do not recompute them.
3. Lead with the most important finding (for example a falling safety \
score or a new incident), then compare with the driver's previous periods \
and with the fleet average.
4. You may suggest a next step for a manager (for example a coaching \
review), but never state that any action has been taken -- you cannot \
take actions.
5. If the data is insufficient to answer, say so instead of guessing."""


def _fmt(x, suffix=""):
    return "n/a" if x is None else f"{x:g}{suffix}"


def _change(a, b, suffix=""):
    if a is None or b is None:
        return "n/a"
    d = round(b - a, 2)
    return f"{'+' if d >= 0 else ''}{d:g}{suffix}"


def build_performance_context(rows: list[dict], fleet) -> str:
    """Turn rows (newest first) into plain facts for the prompt."""
    latest = rows[0]
    lines = [
        f"LATEST PERIOD ({latest['period_start']} to {latest['period_end']}): "
        f"safety score {_fmt(latest['safety_score'])}, on-time {_fmt(latest['on_time_pct'], '%')}, "
        f"late deliveries {_fmt(latest['late_deliveries'])}, incidents {_fmt(latest['incidents'])}",
    ]

    trend = []
    if len(rows) >= 2:
        prev = rows[1]
        trend.append(
            f"Change vs previous period: safety score {_change(prev['safety_score'], latest['safety_score'])}, "
            f"on-time {_change(prev['on_time_pct'], latest['on_time_pct'], ' pts')}, "
            f"late deliveries {_change(prev['late_deliveries'], latest['late_deliveries'])}, "
            f"incidents {_change(prev['incidents'], latest['incidents'])}"
        )
        # How many periods in a row the safety score has fallen, newest backwards.
        streak = 0
        for newer, older in zip(rows, rows[1:]):
            if None not in (newer["safety_score"], older["safety_score"]) and newer["safety_score"] < older["safety_score"]:
                streak += 1
            else:
                break
        if streak:
            start = rows[streak]["safety_score"]
            trend.append(
                f"Safety score has fallen {streak} period(s) in a row: "
                f"{_fmt(start)} -> {_fmt(latest['safety_score'])} ({_change(start, latest['safety_score'])})"
            )
        else:
            trend.append("Safety score did not fall in the latest period.")
    recent = rows[:3]
    incidents = sum(r["incidents"] or 0 for r in recent)
    trend.append(f"Incidents in the last {len(recent)} period(s): {incidents}")
    lines += ["", "TREND:"] + [f"- {t}" for t in trend]

    if fleet:
        lines += ["", f"FLEET AVERAGE, latest period ({fleet['drivers']} drivers): "
                      f"safety score {_fmt(fleet['safety_score'])}, on-time {_fmt(fleet['on_time_pct'], '%')}"]

    lines += ["", "ALL PERIODS (newest first):"]
    lines += [
        f"- {r['period_start']} to {r['period_end']}: safety {_fmt(r['safety_score'])}, "
        f"on-time {_fmt(r['on_time_pct'], '%')}, late {_fmt(r['late_deliveries'])}, "
        f"incidents {_fmt(r['incidents'])}"
        for r in rows
    ]
    return "\n".join(lines)


def run_performance_agent(db, driver_id: str, question: str = "") -> str:
    """`db` is accepted for interface consistency with the other agents; unused."""
    try:
        rows, fleet = get_performance_history_via_mcp(driver_id, question, periods=6)
    except Exception:
        return (
            f"I couldn't retrieve performance data for {driver_id} right now. "
            "Please try again in a moment."
        )

    if not rows:
        return (
            f"I don't have any performance records for {driver_id}. "
            "Double-check the driver ID, or this driver may be new."
        )

    llm = get_llm()
    # Sample Context
    #LATEST PERIOD (2026-10-05 to 2026-10-11): safety score 75.62, on-time 88.59%, late deliveries 3, incidents 1
    #TREND:
    #- Change vs previous period: safety score -6.48, on-time -2.61 pts, late deliveries +1, incidents 0
    #- Safety score has fallen 2 period(s) in a row: 86.40 -> 75.62 (-10.78)
    #- Incidents in the last 3 period(s): 2
    # FLEET AVERAGE, latest period (10 drivers): safety score 84.30, on-time 91.75%

    #ALL PERIODS (newest first):
    #- 2026-10-05 to 2026-10-11: safety 75.62, on-time 88.59%, late 3, incidents 1
    # - 2026-09-28 to 2026-10-04: safety 82.10, on-time 91.20%, late 2, incidents 1
    # - 2026-09-21 to 2026-09-27: safety 86.40, on-time 93.00%, late 1, incidents 0
    # - 2026-09-14 to 2026-09-20: safety 85.00, on-time 92.50%, late 1, incidents 0
    context = build_performance_context(rows, fleet)
    prompt = (
        f"{PERFORMANCE_SYSTEM_PROMPT}\n\n"
        f"Driver: {driver_id}\n\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    return check_answer(answer, context, "performance")