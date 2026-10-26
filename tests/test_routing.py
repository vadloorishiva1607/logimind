"""
tests/test_routing.py -- routing evaluation set for the Supervisor.

Each case is a question (plus any remembered conversation state) and the
route the Supervisor SHOULD pick. The Supervisor is plain Python rules, so
this runs offline: no LLM, no database, about a second.

Run from the repo root:
    python tests/test_routing.py
Re-run it after every change to agent.py's keyword lists or routing rules.
"""
from dotenv import load_dotenv

load_dotenv()

from src.agent.agent import supervisor_node  # noqa: E402

# (question, remembered state from earlier turns, expected route, what it tests)
CASES = [
    # --- Settlement: clear wording ---
    ("Why did D102 settlement drop?", {}, "settlement", "settlement keyword"),
    ("Why is D102's pay lower this week?", {}, "settlement", "pay keyword"),
    ("Show D105's deductions", {}, "settlement", "deduction keyword"),
    ("What did D101 earn last week?", {}, "settlement", "earn keyword"),
    ("D103 gross pay?", {}, "settlement", "gross keyword"),
    ("How much was D102 paid?", {}, "settlement", "paid -> pay keyword"),
    # --- Performance: clear wording ---
    ("Why is D104 safety score dropping?", {}, "performance", "safety keyword"),
    ("How was D104's on-time rate?", {}, "performance", "on-time keyword"),
    ("Did D107 have any incidents?", {}, "performance", "incident keyword"),
    ("Show D104 performance", {}, "performance", "performance keyword"),
    ("How many late deliveries for D106?", {}, "performance", "late deliveries"),
    ("Was D104 in an accident?", {}, "performance", "accident keyword"),
    # --- Follow-ups: driver and agent come from memory ---
    ("What about the week before that?", {"driver_id": "D102", "last_agent": "settlement"}, "settlement", "follow-up stays with settlement"),
    ("What about the period before that?", {"driver_id": "D104", "last_agent": "performance"}, "performance", "follow-up stays with performance"),
    ("And his safety score?", {"driver_id": "D102", "last_agent": "settlement"}, "performance", "follow-up switches topic to performance"),
    ("And his pay?", {"driver_id": "D104", "last_agent": "performance"}, "settlement", "follow-up switches topic to settlement"),
    ("Compare with D105", {"driver_id": "D102", "last_agent": "settlement"}, "settlement", "new driver, same topic"),
   
    # --- Escrow ---
    ("What is D105's current escrow balance?", {}, "escrow", "escrow keyword"),
    ("Show D105's escrow withdrawals", {}, "escrow", "escrow keyword, not settlement"),
    ("What about before that?", {"driver_id": "D105", "last_agent": "escrow"}, "escrow", "follow-up stays with escrow"),

   # --- Shipment ---
    ("Why is shipment S10023 delayed?", {}, "shipment", "shipment ID"),
    ("Where is S10023?", {}, "shipment", "shipment ID alone, no topic word"),
    ("Is D102 often late with shipments?", {}, "shipment", "driver ID  shipment keyword"),
    ("When was it supposed to arrive?", {"shipment_id": "S10023", "last_agent": "shipment"}, "shipment", "follow-up about the remembered shipment"),
    ("Why did D102 settlement drop?", {"shipment_id": "S10023", "last_agent": "shipment"}, "settlement", "new driver question moves on from the shipment"),

    # --- Fuel ---
    ("Why did D102's fuel cost go up?", {}, "fuel", "fuel keyword"),
    ("What is D104's MPG?", {}, "fuel", "mpg keyword"),
    ("How many gallons did D102 buy last week?", {}, "fuel", "gallon keyword"),
    ("Why did D102's fuel deduction rise?", {}, "fuel", "fuel wins over the settlement word 'deduction'"),
    ("And the week before?", {"driver_id": "D102", "last_agent": "fuel"}, "fuel", "follow-up stays with fuel"),

    # --- Policy questions (RAG) ---
    ("What is the escrow policy?", {}, "rag", "policy keyword"),
    ("What is the driver termination procedure?", {}, "rag", "procedure keyword"),
    ("Explain the fuel reimbursement SOP", {}, "rag", "SOP keyword"),
    ("What are the rules for shipment delays?", {}, "rag", "rule keyword"),
    ("What does the compliance handbook say about confidential data?", {}, "rag", "handbook keyword"),
    # --- Clarify: nothing to go on ---
    ("What is the weather today?", {}, "clarify", "off-topic"),
    ("Hello", {}, "clarify", "greeting"),
    ("Why did it drop?", {}, "clarify", "no driver, no topic, no memory"),
    ("What are the shipment delay rules?", {}, "rag", "shipment word but no ID: a policy question"),

    # --- Tricky wording the keyword rules may get wrong ---
    ("How's D104 driving lately?", {}, "performance", "no keyword: 'driving' means performance"),
    ("Is D102 being safe on the road?", {}, "performance", "'safe' is not 'safety'"),
    ("Why did D105's escrow balance drop?", {}, "escrow", "escrow keyword -> Escrow agent"),
    ("Why is D102 earning less and having more incidents?", {}, "performance", "multi-intent: both agents needed; today only one runs"),
    ("What is D102's policy on pay?", {}, "settlement", "driver ID wins over policy words"),
]


def run():
    passed, failures = 0, []
    for question, memory, expected, label in CASES:
        state = {"question": question, **memory}
        result = supervisor_node(state)
        route = result["route"]
        if route == expected:
            passed += 1
        else:
            failures.append((question, expected, route, label, result["route_reason"]))

    total = len(CASES)
    print(f"Routing accuracy: {passed}/{total} = {passed / total:.0%}\n")
    if failures:
        print("Misroutes:")
        for question, expected, route, label, reason in failures:
            print(f"  - {question!r}: expected {expected}, got {route}  ({label})")
            print(f"      why: {reason}")
    return passed, total


if __name__ == "__main__":
    run()