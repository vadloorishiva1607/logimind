"""
tests/test_fuel.py -- checks the Fuel agent's analysis and safety net.

Uses D102's real fuel data (fetched on EC2, 2026-10-26). Runs offline:
no LLM, no database, no MCP call.

Run from the repo root:
    PYTHONPATH=. python tests/test_fuel.py
"""
from src.agent.output_guardrail import find_unsupported_numbers
from src.tools.fuel import _ensure_flags_reported, analyse, build_fuel_context

D102_ROWS = [
    {"transaction_date": "2026-09-27", "gallons": 203.8, "cost_usd": 1037.25, "mpg": 5.24, "location": "Diazborough"},
    {"transaction_date": "2026-09-20", "gallons": 191.71, "cost_usd": 734.25, "mpg": 6.5, "location": "Walshtown"},
    {"transaction_date": "2026-09-13", "gallons": 185.67, "cost_usd": 677.7, "mpg": 6.26, "location": "Lake Johnfort"},
    {"transaction_date": "2026-09-06", "gallons": 159.77, "cost_usd": 634.29, "mpg": 6.85, "location": "West Michelle"},
    {"transaction_date": "2026-08-30", "gallons": 164.77, "cost_usd": 670.61, "mpg": 7.25, "location": "North Marissaport"},
    {"transaction_date": "2026-08-23", "gallons": 198.17, "cost_usd": 735.21, "mpg": 6.59, "location": "Bryanfort"},
]
FLEET = {"mpg": 6.74, "price_per_gallon": 3.892, "drivers": 10}


def check(label, condition):
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")
    return condition


def run():
    r = []
    a = analyse(D102_ROWS, FLEET)
    p = a["previous"]
    r.append(check("cost change +$303.00 (matches the settlement fuel deduction)", p["d_cost"] == 303.0))
    r.append(check("price per gallon $3.830 -> $5.090", p["ppg"] == 3.83 and a["ppg"] == 5.09))
    r.append(check("cost split adds up to the total change",
                   round(a["split"]["volume"] + a["split"]["price"], 2) == p["d_cost"]))
    r.append(check("most of the rise came from price, not gallons",
                   a["split"]["price"] > a["split"]["volume"] > 0))
    r.append(check("trailing 4-week average $679.21 (Sep 20, 13, 6, Aug 30)",
                   a["trailing"]["avg"] == 679.21 and a["trailing"]["count"] == 4))
    r.append(check("cost flag: $358.04 above that average (policy: more than $50)",
                   a["trailing"]["over"] == 358.04 and any(k == "cost" for k, _ in a["flags"])))
    r.append(check("MPG flag: 5.24 is -19.4% against the 6.5 baseline (policy: more than 15% below)",
                   a["vs_baseline"] == -19.4 and any(k == "mpg" for k, _ in a["flags"])))
    r.append(check("6 fill-ups is enough for a trend", a["enough_history"]))

    sep13 = D102_ROWS[2:]  # as if Sep 13 were the latest fill-up
    r.append(check("no flags for Sep 13 ($677.70 vs average $680.04; 6.26 MPG within 15% of 6.5)",
                   analyse(sep13, FLEET)["flags"] == []))
    r.append(check("2 fill-ups is too few for a trend", not analyse(D102_ROWS[:2], FLEET)["enough_history"]))
    r.append(check("a single fill-up works (no comparison, no crash)",
                   "previous" not in analyse(D102_ROWS[:1], FLEET)))

    ctx, a = build_fuel_context("D102", D102_ROWS, FLEET)
    good = ("D102's fuel cost rose from $734.25 to $1,037.25 (+$303.00). About $46.30 came from buying "
            "12.09 more gallons and $256.70 from the price per gallon rising from $3.830 to $5.090. "
            "MPG fell from 6.5 to 5.24, below the fleet average of 6.74.")
    r.append(check("a correct answer passes the number checker", find_unsupported_numbers(good, ctx) == []))

    added = _ensure_flags_reported("The cost went up.", a)
    r.append(check("safety net adds both flags when the answer has neither",
                   "Anomalous fuel deduction" in added and "Efficiency below baseline" in added))
    kept = "The deduction is anomalous and MPG is below the 6.5 baseline."
    r.append(check("safety net adds nothing when both are mentioned", _ensure_flags_reported(kept, a) == kept))

    print(f"\nFuel checks: {sum(r)}/{len(r)} passed")
    print("\n--- Context the LLM will see for D102 ---\n" + ctx)


if __name__ == "__main__":
    run()