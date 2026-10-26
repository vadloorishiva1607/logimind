"""
tests/test_shipment.py -- checks the Shipment agent's facts and risk rules.

Uses S10023 / D102's real data (fetched on EC2, 2026-10-26) plus a few
made-up shipments for cases the real data doesn't cover. Runs offline:
no LLM, no database, no MCP call.

Run from the repo root:
    PYTHONPATH=. python tests/test_shipment.py
"""
from src.agent.output_guardrail import find_unsupported_numbers
from src.tools.shipment import (
    _ensure_reported, build_shipment_context, causes, delay_hours, pattern_finding,
)

S10023 = {"shipment_id": "S10023", "driver_id": "D102", "origin": "Dallas, TX", "destination": "Memphis, TN",
          "scheduled_eta": "2026-09-26T10:45:23.479061", "actual_eta": "2026-09-26T14:45:23.479061",
          "status": "delayed", "delay_reason": "Traffic + driver rest-break delay"}
S10009 = {"shipment_id": "S10009", "driver_id": "D102", "origin": "Thompsonchester", "destination": "South Angela",
          "scheduled_eta": "2026-09-21T10:52:33.978575", "actual_eta": "2026-09-21T10:52:33.978575",
          "status": "delivered", "delay_reason": None}
D102_ROWS = [S10023, S10009]


def made_up(sid, status, hours_late, reason=None):
    """A test shipment for driver D900, `hours_late` after a fixed schedule."""
    actual = None if hours_late is None else f"2026-09-01T{10 + hours_late:02d}:00:00"
    return {"shipment_id": sid, "driver_id": "D900", "origin": "A", "destination": "B",
            "scheduled_eta": "2026-09-01T10:00:00", "actual_eta": actual,
            "status": status, "delay_reason": reason}


def check(label, condition):
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")
    return condition


def run():
    r = []
    # --- Facts computed by code ---
    r.append(check("S10023 delay is 4.0 hours", delay_hours(S10023) == 4.0))
    r.append(check("S10009 delay is 0.0 hours", delay_hours(S10009) == 0.0))
    r.append(check("two causes split from 'Traffic + driver rest-break delay'",
                   causes(S10023["delay_reason"]) == ["Traffic", "driver rest-break delay"]))
    r.append(check("no reason -> no causes", causes(None) == []))

    # --- Pattern rules ---
    is_p, sentence = pattern_finding(D102_ROWS)
    r.append(check("D102: 2 shipments is too few to call a pattern", not is_p and "too few" in sentence))
    three_late = [made_up(f"T{i}", "delayed", 2, "Weather") for i in range(3)]
    r.append(check("3 of 3 delayed is a repeated pattern", pattern_finding(three_late)[0]))
    mostly_ok = [made_up("T1", "delayed", 1, "Weather")] + [made_up(f"T{i}", "delivered", 0) for i in range(2, 5)]
    r.append(check("1 of 4 delayed is not a pattern", not pattern_finding(mostly_ok)[0]))

    # --- Context and risk for the real S10023 question ---
    ctx, facts = build_shipment_context({"tool": "get_shipment", "shipment": S10023, "driver_rows": D102_ROWS})
    r.append(check("S10023 risk is HIGH (4.0 h >= 4 h threshold)", facts["risk"] == "HIGH"))
    r.append(check("context lists both causes and says it can't rank them",
                   "Traffic; driver rest-break delay" in ctx and "does not say which contributed more" in ctx))
    r.append(check("times are readable: 10:45 -> 14:45",
                   "2026-09-26 10:45" in ctx and "2026-09-26 14:45" in ctx))
    good = ("S10023 from Dallas, TX to Memphis, TN was scheduled for 10:45 on 2026-09-26 but arrived at 14:45, "
            "4.0 hours late. Causes: traffic and a driver rest-break. Risk is HIGH.")
    r.append(check("a correct answer passes the number checker", find_unsupported_numbers(good, ctx) == []))

    # --- Negative scenarios from the design ---
    no_reason = made_up("T9", "delayed", 2, None)
    ctx2, facts2 = build_shipment_context({"tool": "get_shipment", "shipment": no_reason, "driver_rows": [no_reason]})
    r.append(check("no delay reason -> 'Cause unconfirmed' in context", "Cause unconfirmed" in ctx2))
    r.append(check("2-hour delay with no pattern -> MEDIUM risk", facts2["risk"] == "MEDIUM"))

    # --- Safety net ---
    added = _ensure_reported("The shipment was late.", facts2)
    r.append(check("safety net adds risk level and 'cause unconfirmed' when missing",
                   "Risk level (set by rules): MEDIUM" in added and "Cause unconfirmed for T9" in added))
    kept = "Risk is MEDIUM. The cause is unconfirmed."
    r.append(check("safety net adds nothing when both are already there", _ensure_reported(kept, facts2) == kept))

    # --- Driver question (no single shipment) ---
    _, facts3 = build_shipment_context({"tool": "get_driver_shipments", "shipment": None, "driver_rows": D102_ROWS})
    r.append(check("D102 driver question -> MEDIUM (a delay, but no pattern)", facts3["risk"] == "MEDIUM"))

    print(f"\nShipment checks: {sum(r)}/{len(r)} passed")
    print("\n--- Context the LLM will see for 'Why is S10023 delayed?' ---\n" + ctx)


if __name__ == "__main__":
    run()