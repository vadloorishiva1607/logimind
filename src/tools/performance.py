"""
src/tools/performance.py -- Performance agent.

Answers questions about a driver's on-time %, safety score, late deliveries
and incidents, compared with their own history and the fleet average.
Code fetches the data and computes the trend; the LLM only explains it.
Data comes from the MCP server (mcp_server.py) via performance_mcp_client.py,
where the LLM chooses the tool and code checks that choice.
"""
from src.agent.output_guardrail import check_answer
from src.models.llm_client import get_llm
from src.tools.performance_mcp_client import get_performance_history_via_mcp

# Safety and Performance Review Policy: "A safety score decline of more than
# 10 points over a rolling 3-week period should trigger a recommended
# performance review ... A single incident report, combined with any safety
# score decline in the same period, should be treated as a higher-priority
# review candidate". Periods are weekly, so 3 weeks = the latest 3 periods.
REVIEW_DROP_POINTS = 10
REVIEW_WINDOW_PERIODS = 3

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
4. If the data says a performance review is recommended, say so and name \
the Safety and Performance Review Policy, including whether it is a \
higher-priority case. Never state that any action has been taken -- you \
cannot take actions.
5. If the data is insufficient to answer, say so instead of guessing."""


def _fmt(x, suffix=""):
    return "n/a" if x is None else f"{x:g}{suffix}"


def _change(a, b, suffix=""):
    if a is None or b is None:
        return "n/a"
    d = round(b - a, 2)
    return f"{'+' if d >= 0 else ''}{d:g}{suffix}"


def review_finding(rows: list[dict]):
    """Apply the Safety and Performance Review Policy. Returns (level, sentence),
    level is None, "recommended" or "higher priority". `rows` are newest first."""
    window = [r for r in rows[:REVIEW_WINDOW_PERIODS] if r["safety_score"] is not None]
    if len(window) < 2:
        return None, "Not enough periods to apply the 3-week review rule."
    latest, start = window[0], window[-1]
    drop = round(start["safety_score"] - latest["safety_score"], 2)
    span = f"{start['period_start']} to {latest['period_end']}"
    if drop > REVIEW_DROP_POINTS:
        level = "recommended"
        sentence = (f"Performance review recommended: safety score fell {drop:g} points over {span} "
                    f"({_fmt(start['safety_score'])} -> {_fmt(latest['safety_score'])}); the Safety and "
                    f"Performance Review Policy recommends a review for a decline of more than "
                    f"{REVIEW_DROP_POINTS} points over 3 weeks.")
    else:
        level = None
        sentence = (f"No review trigger: safety score change over {span} is {_change(start['safety_score'], latest['safety_score'])} "
                    f"(the policy's trigger is a decline of more than {REVIEW_DROP_POINTS} points).")
    prev = rows[1] if len(rows) > 1 else None
    declined_now = prev is not None and None not in (prev["safety_score"], latest["safety_score"]) \
        and latest["safety_score"] < prev["safety_score"]
    if (latest["incidents"] or 0) > 0 and declined_now:
        level = "higher priority"
        sentence += (f" Higher-priority review candidate: {latest['incidents']} incident(s) in the latest "
                     f"period together with a safety score decline in the same period.")
    return level, sentence


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

    level, sentence = review_finding(rows)
    lines += ["", "REVIEW RULE (Safety and Performance Review Policy):", f"- {sentence}"]

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

    context = build_performance_context(rows, fleet)
    llm = get_llm()
    prompt = (
        f"{PERFORMANCE_SYSTEM_PROMPT}\n\n"
        f"Driver: {driver_id}\n\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    answer = check_answer(answer, context, "performance")
    return _ensure_review_reported(answer, rows)


def _ensure_review_reported(answer: str, rows: list[dict]) -> str:
    """Safety net: a review the policy recommends must reach the user."""
    level, sentence = review_finding(rows)
    text = answer.lower()
    if level is None or ("review" in text and (level != "higher priority" or "priority" in text)):
        return answer
    return answer + "\n\nFlagged by rules: " + sentence