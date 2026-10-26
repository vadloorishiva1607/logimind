"""
src/tools/fuel.py -- Fuel agent.

Answers questions about a driver's fuel cost, price per gallon and fuel
efficiency (MPG). Data comes from the MCP tool get_fuel_history.

Code computes everything: the change since the previous fill-up, how
much of a cost rise came from more gallons versus a higher price, the
comparison with the driver's own earlier average and the fleet, and the
anomaly flags. The LLM only explains. The output guardrail checks the
numbers, and a safety net guarantees every flag reaches the user.
"""
from src.agent.output_guardrail import check_answer
from src.models.llm_client import get_llm
from src.tools.fuel_mcp_client import get_fuel_history_via_mcp

# Anomaly rules. Demo assumptions, stated openly in the context.
UNUSUAL_SHARE = 0.15   # 15% worse than the driver's earlier average = flag
MIN_HISTORY = 3        # fewer fill-ups than this: too few for a trend

FUEL_SYSTEM_PROMPT = """You are LogiMind's Fuel agent. Answer using ONLY \
the fuel data provided below, and quote specific dates and figures from it.

Rules:
1. Use the precomputed figures and findings exactly as given. Do not \
recompute them.
2. For a cost question, start with the cost before and after, then explain \
how much came from more gallons and how much from a higher price per gallon.
3. Report every flag in the data, with its figures, and compare with the \
driver's earlier average and the fleet average.
4. Do not guess why a price or efficiency changed (for example route, \
weather or vehicle); the data does not say. You may suggest what to check.
5. If the history is too short for a trend, say so.
6. Never state that any action has been taken.
7. Write the answer in your own plain sentences. Do not copy the data \
lines or their labels.
8. If the data is insufficient to answer, say so instead of guessing."""


def _money(x, places=2):
    return "n/a" if x is None else f"${x:,.{places}f}"


def _signed(x, places=2, prefix="$"):
    if x is None:
        return "n/a"
    return f"{'+' if x >= 0 else '-'}{prefix}{abs(x):,.{places}f}"


def _pct(new, old):
    if new is None or not old:
        return None
    return round((new - old) / old * 100, 1)


def _pct_text(p):
    return "n/a" if p is None else f"{'+' if p >= 0 else ''}{p}%"


def price_per_gallon(cost, gallons):
    return round(cost / gallons, 3) if cost is not None and gallons else None


def analyse(rows: list[dict], fleet) -> dict:
    """All the fuel facts, computed by code. `rows` are newest first."""
    latest = rows[0]
    result = {"latest": latest, "ppg": price_per_gallon(latest["cost_usd"], latest["gallons"]),
              "enough_history": len(rows) >= MIN_HISTORY, "flags": []}

    if len(rows) >= 2:
        prev = rows[1]
        prev_ppg = price_per_gallon(prev["cost_usd"], prev["gallons"])
        d_gallons = round(latest["gallons"] - prev["gallons"], 2)
        d_ppg = None if prev_ppg is None or result["ppg"] is None else round(result["ppg"] - prev_ppg, 3)
        result["previous"] = {
            "row": prev, "ppg": prev_ppg,
            "d_gallons": d_gallons, "p_gallons": _pct(latest["gallons"], prev["gallons"]),
            "d_cost": round(latest["cost_usd"] - prev["cost_usd"], 2), "p_cost": _pct(latest["cost_usd"], prev["cost_usd"]),
            "d_ppg": d_ppg, "p_ppg": _pct(result["ppg"], prev_ppg),
            "d_mpg": None if latest["mpg"] is None or prev["mpg"] is None else round(latest["mpg"] - prev["mpg"], 2),
            "p_mpg": _pct(latest["mpg"], prev["mpg"]),
        }
        # Split the cost change: (extra gallons x old price) + (price change x new gallons).
        # The two parts add up exactly to the total change.
        if prev_ppg is not None and d_ppg is not None:
            volume = round(d_gallons * prev_ppg, 2)
            result["split"] = {"volume": volume, "price": round(result["previous"]["d_cost"] - volume, 2)}

    earlier = rows[1:]
    if earlier:
        e_gallons = sum(r["gallons"] or 0 for r in earlier)
        e_cost = sum(r["cost_usd"] or 0 for r in earlier)
        mpgs = [r["mpg"] for r in earlier if r["mpg"] is not None]
        base_ppg = price_per_gallon(e_cost, e_gallons)
        base_mpg = round(sum(mpgs) / len(mpgs), 2) if mpgs else None
        result["baseline"] = {"count": len(earlier), "ppg": base_ppg, "mpg": base_mpg,
                              "p_ppg": _pct(result["ppg"], base_ppg), "p_mpg": _pct(latest["mpg"], base_mpg)}
        b = result["baseline"]
        if b["p_ppg"] is not None and b["p_ppg"] >= UNUSUAL_SHARE * 100:
            result["flags"].append(("price",
                f"Unusual price: {_money(result['ppg'], 3)} per gallon on {latest['transaction_date']} is "
                f"{_pct_text(b['p_ppg'])} above the driver's earlier average of {_money(base_ppg, 3)}."))
        if b["p_mpg"] is not None and b["p_mpg"] <= -UNUSUAL_SHARE * 100:
            result["flags"].append(("mpg",
                f"Efficiency drop: {latest['mpg']} MPG on {latest['transaction_date']} is "
                f"{_pct_text(b['p_mpg'])} against the driver's earlier average of {base_mpg} MPG."))
    return result


def build_fuel_context(driver_id: str, rows: list[dict], fleet) -> tuple[str, dict]:
    a = analyse(rows, fleet)
    lt = a["latest"]
    lines = [f"LATEST FILL-UP ({lt['transaction_date']}, {lt['location']}): {lt['gallons']} gallons, "
             f"{_money(lt['cost_usd'])}, {_money(a['ppg'], 3)} per gallon, {lt['mpg']} MPG."]

    if "previous" in a:
        p = a["previous"]
        lines.append(
            f"CHANGE SINCE PREVIOUS FILL-UP ({p['row']['transaction_date']}): cost "
            f"{_money(p['row']['cost_usd'])} -> {_money(lt['cost_usd'])} ({_signed(p['d_cost'])}, {_pct_text(p['p_cost'])}); "
            f"gallons {p['row']['gallons']} -> {lt['gallons']} ({_signed(p['d_gallons'], prefix='')}, {_pct_text(p['p_gallons'])}); "
            f"price per gallon {_money(p['ppg'], 3)} -> {_money(a['ppg'], 3)} ({_signed(p['d_ppg'], 3)}, {_pct_text(p['p_ppg'])}); "
            f"MPG {p['row']['mpg']} -> {lt['mpg']} ({_signed(p['d_mpg'], prefix='')}, {_pct_text(p['p_mpg'])})."
        )
        if "split" in a:
            s = a["split"]
            lines.append(f"WHY THE COST CHANGED: {_signed(s['volume'])} from the change in gallons and "
                         f"{_signed(s['price'])} from the change in price per gallon (total {_signed(p['d_cost'])}).")

    if "baseline" in a:
        b = a["baseline"]
        lines.append(f"DRIVER'S EARLIER AVERAGE ({b['count']} earlier fill-ups): {_money(b['ppg'], 3)} per gallon, "
                     f"{b['mpg']} MPG. Latest vs this: price {_pct_text(b['p_ppg'])}, MPG {_pct_text(b['p_mpg'])}.")
    if fleet:
        lines.append(f"FLEET AVERAGE (same period, {fleet['drivers']} drivers): "
                     f"{_money(fleet['price_per_gallon'], 3)} per gallon, {fleet['mpg']} MPG.")

    lines.append(f"HISTORY: {len(rows)} fill-ups. " + (
        "Enough for a trend." if a["enough_history"] else
        f"Too few for a trend (at least {MIN_HISTORY} are needed)."))

    lines += ["", f"FLAGS (threshold: {UNUSUAL_SHARE * 100:g}% worse than the driver's earlier average):"]
    if a["flags"]:
        lines += [f"- {text}" for _, text in a["flags"]]
        lines.append("- These changes may also explain changes in settlement fuel deductions; "
                     "worth cross-checking with the Settlement agent.")
    else:
        lines.append("- None.")

    lines += ["", f"ALL FILL-UPS FOR {driver_id} (newest first):"]
    lines += [f"- {r['transaction_date']}: {r['gallons']} gallons, {_money(r['cost_usd'])}, "
              f"{_money(price_per_gallon(r['cost_usd'], r['gallons']), 3)} per gallon, {r['mpg']} MPG, {r['location']}"
              for r in rows]
    return "\n".join(lines), a


# Words that show the answer reported each kind of flag.
_FLAG_WORDS = {"price": ("price",), "mpg": ("mpg", "efficien")}


def _ensure_flags_reported(answer: str, analysis: dict) -> str:
    """Safety net: every flag found by code must reach the user."""
    text = answer.lower()
    missing = [msg for kind, msg in analysis["flags"]
               if not any(w in text for w in _FLAG_WORDS[kind])]
    return answer if not missing else answer + "\n\nFlagged by rules: " + " ".join(missing)


def run_fuel_agent(db, driver_id: str, question: str = "") -> str:
    """`db` is accepted for interface consistency with the other agents; unused."""
    try:
        rows, fleet = get_fuel_history_via_mcp(driver_id)
    except Exception:
        return (f"I couldn't retrieve fuel data for {driver_id} right now. "
                "Please try again in a moment.")

    if not rows:
        return f"I don't have any fuel transactions for {driver_id}."

    context, analysis = build_fuel_context(driver_id, rows, fleet)
    llm = get_llm()
    prompt = (
        f"{FUEL_SYSTEM_PROMPT}\n\n"
        f"Driver: {driver_id}\n\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    answer = check_answer(answer, context, "fuel")
    return _ensure_flags_reported(answer, analysis)