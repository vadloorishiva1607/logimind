"""
tests/test_driver.py -- checks the Driver agent's role filter, record flags
and safety net.

Uses real driver data (fetched on EC2, 2026-10-26) plus a made-up
suspended driver, since the real data has none. Runs offline: no LLM,
no database, no MCP call.

Run from the repo root:
    PYTHONPATH=. python tests/test_driver.py
"""
from src.tools.driver import _ensure_reported, build_driver_context, visible_areas

D102 = ({"driver_id": "D102", "full_name": "Olivia Thompson", "status": "active",
         "onboarding_date": "2025-10-10", "role": "standard"},
        {"settlement": {"week_ending": "2026-09-27", "net_pay": 608.6},
         "performance": {"period_end": "2026-09-27", "safety_score": 96.66, "on_time_pct": 95.2},
         "escrow": {"as_of": "2026-09-27", "balance": 3039.11},
         "shipments": {"total": 2, "delayed": 1}})
D103 = ({"driver_id": "D103", "full_name": "Sarah Ritter", "status": "onboarding",
         "onboarding_date": "2026-01-16", "role": "standard"},
        {"settlement": {"week_ending": "2026-09-27", "net_pay": 1414.74},
         "performance": {"period_end": "2026-09-27", "safety_score": 88.97, "on_time_pct": 88.31},
         "escrow": {"as_of": "2026-09-27", "balance": 2989.2},
         "shipments": {"total": 0, "delayed": 0}})
SUSPENDED = ({"driver_id": "D900", "full_name": "Test Driver", "status": "suspended",
              "onboarding_date": "2026-03-01", "role": "standard"},
             {"settlement": None, "performance": None, "escrow": None,
              "shipments": {"total": 0, "delayed": 0}})


def check(label, condition):
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")
    return condition


def run():
    r = []
    # --- Role filter: hidden data is never in the context ---
    ctx_d, facts_d = build_driver_context(*D102, "dispatcher")
    r.append(check("dispatcher: net pay and escrow balance NOT in context",
                   "608.60" not in ctx_d and "3,039.11" not in ctx_d))
    r.append(check("dispatcher: told which areas are hidden",
                   "NOT AVAILABLE FOR YOUR ROLE (dispatcher): latest pay, escrow balance" in ctx_d))
    r.append(check("dispatcher: safety score and shipments still shown",
                   "96.66" in ctx_d and "2 on record, 1 delayed" in ctx_d))
    ctx_f, facts_f = build_driver_context(*D102, "finance")
    r.append(check("finance: sees net pay and escrow, nothing hidden",
                   "$608.60" in ctx_f and "$3,039.11" in ctx_f and facts_f["hidden"] == []))
    r.append(check("unknown role -> most restricted view (fail closed)",
                   visible_areas("intern") == ("dispatcher", {"settlement", "escrow"})))
    r.append(check("missing role -> most restricted view", visible_areas(None)[0] == "dispatcher"))

    # --- Record flags ---
    r.append(check("D102 (active): no record flags", facts_d["flags"] == []))
    ctx3, facts3 = build_driver_context(*D103, "finance")
    r.append(check("D103: onboarding 254 days -> 'longer than expected'",
                   any("254 days" in text for _, text in facts3["flags"])))
    r.append(check("D103: onboarding but already paid -> 'status may be out of date'",
                   any(kind == "status_outdated" for kind, _ in facts3["flags"])))
    def onboarding(days_before_sep27):
        from datetime import date, timedelta
        start = (date(2026, 9, 27) - timedelta(days=days_before_sep27)).isoformat()
        return ({**D103[0], "onboarding_date": start}, {**D103[1], "settlement": None, "performance": None})
    _, f40 = build_driver_context(*onboarding(40), "finance")
    r.append(check("40 days onboarding: past the expected 30, but not flagged (SOP flags after 45)",
                   not any(k == "onboarding_long" for k, _ in f40["flags"])))
    _, f50 = build_driver_context(*onboarding(50), "finance")
    r.append(check("50 days onboarding: flagged for Driver Manager follow-up (SOP: over 45)",
                   any(k == "onboarding_long" and "Driver Manager" in t for k, t in f50["flags"])))
    _, facts_s = build_driver_context(*SUSPENDED, "finance")
    r.append(check("suspended driver flagged", any(kind == "suspended" for kind, _ in facts_s["flags"])))
    ctx_s, _ = build_driver_context(*SUSPENDED, "finance")
    r.append(check("missing data areas said plainly", "no settlement data" in ctx_s and "no escrow data" in ctx_s))

    # --- Safety net ---
    added = _ensure_reported("Sarah Ritter is a driver.", facts3)
    r.append(check("safety net adds both D103 flags when missing",
                   "254 days" in added and "out of date" in added))
    kept = "She is still onboarding, taking longer than expected, and her status looks out of date."
    r.append(check("safety net adds nothing when the flags are already there", _ensure_reported(kept, facts3) == kept))
    added_role = _ensure_reported("Olivia Thompson is active.", facts_d)
    r.append(check("safety net adds the role note for a dispatcher",
                   "Not available for your role (dispatcher)" in added_role))

    print(f"\nDriver checks: {sum(r)}/{len(r)} passed")
    print("\n--- Context for D103, role=dispatcher ---\n" + build_driver_context(*D103, "dispatcher")[0])


if __name__ == "__main__":
    run()