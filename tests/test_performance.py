"""
tests/test_performance.py -- checks the Safety and Performance Review Policy
rules in the Performance agent.

D104's safety scores and incidents are real (Testing Log T24-T27); the
on-time and late-delivery figures below are placeholders, since these
tests only check the review rule. Runs offline: no LLM, no database.

Run from the repo root:
    PYTHONPATH=. python tests/test_performance.py
"""
from src.tools.performance import _ensure_review_reported, build_performance_context, review_finding


def period(start, end, safety, incidents, on_time=90.0, late=1):
    return {"period_start": start, "period_end": end, "safety_score": safety,
            "on_time_pct": on_time, "late_deliveries": late, "incidents": incidents}


D104 = [  # newest first
    period("2026-09-21", "2026-09-27", 75.62, 1),
    period("2026-09-14", "2026-09-20", 77.55, 0),
    period("2026-09-07", "2026-09-13", 89.03, 0),
    period("2026-08-31", "2026-09-06", 90.10, 0),
]


def with_scores(scores, latest_incidents=0):
    """A driver with the given safety scores (newest first)."""
    out = []
    for i, sc in enumerate(scores):
        out.append(period(f"2026-09-{21 - 7 * i:02d}", f"2026-09-{27 - 7 * i:02d}", sc,
                          latest_incidents if i == 0 else 0))
    return out


def check(label, condition):
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")
    return condition


def run():
    r = []
    level, sentence = review_finding(D104)
    r.append(check("D104: fell 13.41 points over 3 weeks (89.03 -> 75.62) -> review",
                   level is not None and "13.41" in sentence))
    r.append(check("D104: 1 incident + decline in the same period -> higher priority",
                   level == "higher priority"))
    r.append(check("drop of 12 points, no incident -> recommended (not higher priority)",
                   review_finding(with_scores([80.0, 85.0, 92.0]))[0] == "recommended"))
    r.append(check("drop of exactly 10 points -> no trigger (policy: more than 10)",
                   review_finding(with_scores([80.0, 85.0, 90.0]))[0] is None))
    r.append(check("incident but score went up -> not higher priority",
                   review_finding(with_scores([91.0, 90.0, 89.0], latest_incidents=1))[0] is None))
    r.append(check("only one period -> no rule applied", review_finding(D104[:1])[0] is None))

    ctx = build_performance_context(D104, None)
    r.append(check("context names the policy and the review", "Safety and Performance Review Policy" in ctx
                   and "Performance review recommended" in ctx))

    added = _ensure_review_reported("D104's safety score fell.", D104)
    r.append(check("safety net adds the review when the answer leaves it out", "Flagged by rules" in added))
    kept = "A higher-priority performance review is recommended."
    r.append(check("safety net adds nothing when the review and priority are mentioned",
                   _ensure_review_reported(kept, D104) == kept))

    print(f"\nPerformance checks: {sum(r)}/{len(r)} passed")
    print("\n--- Review section of the D104 context ---")
    print(ctx[ctx.index("REVIEW RULE"):ctx.index("ALL PERIODS")].strip())


if __name__ == "__main__":
    run()