"""
tests/test_output_guardrail.py -- evaluation set for the number checker.

Uses real LogiMind answers (from live tests T23/T24/T33) and the context
the agents gave the LLM. Correct answers must pass; answers with a changed
figure must be caught. Runs offline: no LLM, no database.

SAMPLE_PERFORMANCE_CONTEXT is an illustrative 4-period context in the exact
format build_performance_context() produces, used to show how the checker
treats rounding, IDs and invented figures. KNOWN_LIMITATIONS lists answers
the checker gets wrong today; they are printed but not counted in accuracy.

Run from the repo root:
    PYTHONPATH=. python tests/test_output_guardrail.py
"""
from src.agent.output_guardrail import find_unsupported_numbers

SETTLEMENT_CONTEXT = """LATEST CHANGE (default focus): Week ending 2026-09-20 -> week ending 2026-09-27: net pay $1,253.36 -> $608.60 (-$644.76); gross pay $2,131.94 -> $1,894.89 (-$237.05); fuel deduction $734.25 -> $1,037.25 (+$303.00); escrow deduction $97.12 -> $200.16 (+$103.04)

ALL WEEK-OVER-WEEK CHANGES (newest first):
- Week ending 2026-09-20 -> week ending 2026-09-27: net pay $1,253.36 -> $608.60 (-$644.76); gross pay $2,131.94 -> $1,894.89 (-$237.05); fuel deduction $734.25 -> $1,037.25 (+$303.00); escrow deduction $97.12 -> $200.16 (+$103.04)
- Week ending 2026-09-13 -> week ending 2026-09-20: net pay $1,055.04 -> $1,253.36 (+$198.32); gross pay $1,865.79 -> $2,131.94 (+$266.15); fuel deduction $677.70 -> $734.25 (+$56.55); escrow deduction $81.37 -> $97.12 (+$15.75)"""

PERFORMANCE_CONTEXT = """LATEST PERIOD (2026-09-21 to 2026-09-27): safety score 75.62, on-time 88.1%, late deliveries 2, incidents 1

TREND:
- Change vs previous period: safety score -1.93, on-time -0.49 pts, late deliveries +1, incidents +1
- Safety score has fallen 2 period(s) in a row: 89.03 -> 75.62 (-13.41)
- Incidents in the last 3 period(s): 1

FLEET AVERAGE, latest period (8 drivers): safety score 88.12, on-time 93.4%"""

# Illustrative sample (not live data): what build_performance_context()
# produces for 4 periods, with every change and trend figure pre-computed.
SAMPLE_PERFORMANCE_CONTEXT = """LATEST PERIOD (2026-10-05 to 2026-10-11): safety score 75.62, on-time 88.59%, late deliveries 3, incidents 1

TREND:
- Change vs previous period: safety score -6.48, on-time -2.61 pts, late deliveries +1, incidents 0
- Safety score has fallen 2 period(s) in a row: 86.40 -> 75.62 (-10.78)
- Incidents in the last 3 period(s): 2

FLEET AVERAGE, latest period (10 drivers): safety score 84.30, on-time 91.75%

ALL PERIODS (newest first):
- 2026-10-05 to 2026-10-11: safety 75.62, on-time 88.59%, late 3, incidents 1
- 2026-09-28 to 2026-10-04: safety 82.10, on-time 91.20%, late 2, incidents 1
- 2026-09-21 to 2026-09-27: safety 86.40, on-time 93.00%, late 1, incidents 0
- 2026-09-14 to 2026-09-20: safety 85.00, on-time 92.50%, late 1, incidents 0"""

# (label, answer, context, expected unsupported numbers)
CASES = [
    # --- Real, correct answers: must pass ---
    ("T33 settlement answer",
     "In the latest change from the week ending 2026-09-20 to the week ending 2026-09-27, "
     "D102's net pay dropped from $1,253.36 to $608.60, a decrease of $644.76. "
     "1. Fuel Deduction: Increased from $734.25 to $1,037.25, an increase of $303.00. "
     "2. Escrow Deduction: Increased from $97.12 to $200.16, an increase of $103.04. "
     "3. Gross Pay: Decreased from $2,131.94 to $1,894.89, a reduction of $237.05.",
     SETTLEMENT_CONTEXT, []),
    ("T33 follow-up answer",
     "In the week ending 2026-09-13, the net pay was $1,055.04, which increased to $1,253.36 "
     "in the following week, a change of +$198.32. Gross pay increased from $1,865.79 to "
     "$2,131.94 (+$266.15). Fuel deduction increased from $677.70 to $734.25 (+$56.55).",
     SETTLEMENT_CONTEXT, []),
    ("T24 performance answer",
     "D104's safety score has dropped to 75.62 in the latest period (2026-09-21 to 2026-09-27), "
     "a decrease of 1.93 points. This is the second consecutive decline, a total drop of 13.41 "
     "points from 89.03. D104 recorded 1 incident. The fleet average is 88.12.",
     PERFORMANCE_CONTEXT, []),
    ("rounded figures are fine",
     "Net pay fell by about $645, and the safety score is now 75.6.",
     SETTLEMENT_CONTEXT + "\n" + PERFORMANCE_CONTEXT, []),
    # --- Wrong figures: must be caught ---
    ("invented drop amount",
     "D102's net pay dropped by $700.00 this week.",
     SETTLEMENT_CONTEXT, [700.0]),
    ("miscopied fuel figure",
     "Fuel deduction increased from $734.25 to $1,073.25.",
     SETTLEMENT_CONTEXT, [1073.25]),
    ("LLM did its own (wrong) arithmetic",
     "Fuel and escrow together took an extra $416.04.",
     SETTLEMENT_CONTEXT, [416.04]),
    ("wrong safety score",
     "D104's safety score is 72.5, below the fleet average of 88.12.",
     PERFORMANCE_CONTEXT, [72.5]),
    # --- Illustrative sample: how the checker reads a full context ---
    ("sample: correct answer, rounded",
     "D104's safety score has fallen for 2 periods in a row, from 86.4 to 75.6 "
     "(down 10.8 points), below the fleet average of 84.3. "
     "On-time dropped 2.6 points to 88.6%.",
     SAMPLE_PERFORMANCE_CONTEXT, []),
    ("sample: whole-number rounding (75.62 -> 76)",
     "D104's safety score is about 76.",
     SAMPLE_PERFORMANCE_CONTEXT, []),
    ("sample: driver ID D104 is not a figure",
     "Driver D104 had a safety score of 75.62.",
     SAMPLE_PERFORMANCE_CONTEXT, []),
    ("sample: miscopied (75.26) and invented (90.4) figures",
     "D104's safety score is 75.26 and the on-time rate is 90.4%.",
     SAMPLE_PERFORMANCE_CONTEXT, [75.26, 90.4]),
]

# Answers the checker gets wrong today. Not counted in accuracy; printed so
# the gaps are visible. (label, answer, context, ideal unsupported numbers)
KNOWN_LIMITATIONS = [
    # False negative: whole numbers <= 10 are skipped. Data says 1 incident.
    ("small counts are not checked",
     "D104 had 7 incidents in the latest period.",
     SAMPLE_PERFORMANCE_CONTEXT, [7.0]),
    # False negative: the minus sign is dropped. Data says the score fell 6.48.
    ("direction (rise vs fall) is not checked",
     "D104's safety score rose 6.48 points.",
     SAMPLE_PERFORMANCE_CONTEXT, [6.48]),
    # False negative: 91.2 is in the data, but for the previous period.
    ("number from the wrong period passes",
     "D104's on-time rate in the latest period is 91.2%.",
     SAMPLE_PERFORMANCE_CONTEXT, [91.2]),
    # False positive: the math is right (84.30 - 75.62 = 8.68), but the
    # context doesn't contain the gap to fleet, so a correct figure is flagged.
    ("correct math not in the context is flagged",
     "D104 is 8.68 points below the fleet average.",
     SAMPLE_PERFORMANCE_CONTEXT, []),
]


def run():
    passed, failures = 0, []
    for label, answer, context, expected in CASES:
        got = find_unsupported_numbers(answer, context)
        if got == expected:
            passed += 1
        else:
            failures.append((label, expected, got))

    total = len(CASES)
    print(f"Output guardrail accuracy: {passed}/{total} = {passed / total:.0%}\n")
    for label, expected, got in failures:
        print(f"  - {label}: expected {expected}, got {got}")

    print("\nKnown limitations (not counted in accuracy):")
    for label, answer, context, ideal in KNOWN_LIMITATIONS:
        got = find_unsupported_numbers(answer, context)
        status = "fixed" if got == ideal else "still present"
        print(f"  - {label}: checker returns {got}, ideal {ideal} ({status})")
    return passed, total


if __name__ == "__main__":
    run()