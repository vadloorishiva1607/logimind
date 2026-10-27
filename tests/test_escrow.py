"""
tests/test_escrow.py -- checks the Escrow agent's reconciliation and context.

Uses D105's real escrow history (fetched on EC2, 2026-10-24). Runs offline:
no LLM, no database, no MCP call.

Run from the repo root:
    PYTHONPATH=. python tests/test_escrow.py
"""
from src.tools.escrow import _ensure_discrepancy_reported, build_escrow_context, policy_checks, reconcile

D105_HISTORY = {"rows": [
    {"transaction_date": "2026-09-27", "type": "withdrawal", "amount_usd": -994.69, "balance_after": 1492.03, "reason": "Adjustment — see settlement deduction detail"},
    {"transaction_date": "2026-09-20", "type": "deposit", "amount_usd": 92.48, "balance_after": 2395.6, "reason": "Weekly escrow contribution"},
    {"transaction_date": "2026-09-13", "type": "deposit", "amount_usd": 67.06, "balance_after": 2303.12, "reason": "Weekly escrow contribution"},
    {"transaction_date": "2026-09-06", "type": "deposit", "amount_usd": 75.3, "balance_after": 2236.06, "reason": "Weekly escrow contribution"},
    {"transaction_date": "2026-08-30", "type": "deposit", "amount_usd": 68.53, "balance_after": 2160.76, "reason": "Weekly escrow contribution"},
    {"transaction_date": "2026-08-23", "type": "deposit", "amount_usd": 71.42, "balance_after": 2092.23, "reason": "Weekly escrow contribution"},
]}

D105_BALANCE = {"balance": 1492.03, "last_transaction": D105_HISTORY["rows"][0]}


def check(label, condition):
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")
    return condition


def run():
    results = []
    rows = D105_HISTORY["rows"]

    problems = reconcile(rows)
    results.append(check("D105: exactly one discrepancy found", len(problems) == 1))
    if problems:
        p = problems[0]
        results.append(check("discrepancy is on 2026-09-27", p["date"] == "2026-09-27"))
        results.append(check("expected balance 1400.91 (2395.60 - 994.69)", p["expected"] == 1400.91))
        results.append(check("difference +91.12", p["difference"] == 91.12))

    results.append(check("older rows (Aug 23 - Sep 20) reconcile cleanly", reconcile(rows[1:]) == []))
    results.append(check("a single row has nothing to compare", reconcile(rows[:1]) == []))

    history_ctx = build_escrow_context("get_escrow_history", D105_HISTORY)
    results.append(check("history context flags the discrepancy",
                         "UNEXPLAINED DISCREPANCY on 2026-09-27" in history_ctx and "+$91.12" in history_ctx))
    results.append(check("history context shows the latest change",
                         "$2,395.60 on 2026-09-20 -> $1,492.03 on 2026-09-27 (-$903.57)" in history_ctx))
    results.append(check("withdrawal share of previous balance is 41.5%", "41.5%" in history_ctx))

    clean_ctx = build_escrow_context("get_escrow_history", {"rows": rows[1:]})
    results.append(check("clean history says all changes match",
                         "All balance changes match" in clean_ctx and "DISCREPANCY" not in clean_ctx))

    balance_ctx = build_escrow_context("get_escrow_balance", D105_BALANCE)
    results.append(check("balance context gives $1,492.03 and the latest withdrawal",
                         "CURRENT BALANCE: $1,492.03" in balance_ctx and "-$994.69" in balance_ctx))

    # --- Escrow Withdrawal Policy checks ---
    checks = policy_checks(rows)
    kinds = [k for k, _ in checks]
    r = results
    r.append(check("D105 Sep 27 withdrawal: 'Adjustment' is not a documented reason -> flagged",
                   kinds.count("undocumented") == 1 and "2026-09-27" in checks[kinds.index("undocumented")][1]))
    r.append(check("D105 Sep 27 withdrawal is 47.1% of the $2,113.30 average balance (> 25%) -> flagged",
                   any(k == "large" and "47.1%" in t and "$2,113.30" in t for k, t in checks)))
    r.append(check("D105 weekly deposits ($67-$92) are under the $150 limit -> no contribution flag",
                   "contribution" not in kinds))
    documented = [{**rows[0], "reason": "Approved damage claim #4411"}] + rows[1:]
    r.append(check("a withdrawal for an approved damage claim is not flagged as undocumented",
                   "undocumented" not in [k for k, _ in policy_checks(documented)]))
    big_deposit = [{**rows[1], "amount_usd": 160.0}] + rows[2:]
    r.append(check("a $160 weekly contribution -> consent check",
                   "contribution" in [k for k, _ in policy_checks(big_deposit)]))
    r.append(check("history context lists the policy checks",
                   "POLICY CHECKS (Escrow Withdrawal Policy)" in history_ctx and "documented reason" in history_ctx))
    added = _ensure_discrepancy_reported("The balance dropped.", "get_escrow_history", D105_HISTORY)
    r.append(check("safety net adds the discrepancy and both policy checks when missing",
                   "Unexplained discrepancy" in added and "Withdrawal without a documented reason" in added
                   and "Large change" in added))
    kept = ("There is an unexplained discrepancy. The withdrawal has no documented reason "
            "and, at over 25% of the average balance, needs a reason code.")
    r.append(check("safety net adds nothing when all are mentioned",
                   _ensure_discrepancy_reported(kept, "get_escrow_history", D105_HISTORY) == kept))

    passed = sum(results)
    print(f"\nEscrow checks: {passed}/{len(results)} passed")
    print("\n--- History context the LLM will see ---\n" + history_ctx)


if __name__ == "__main__":
    run()