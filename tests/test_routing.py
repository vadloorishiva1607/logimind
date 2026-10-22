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
    # --- Tricky wording the keyword rules may get wrong ---
    ("How's D104 driving lately?", {}, "performance", "no keyword: 'driving' means performance"),
    ("Is D102 being safe on the road?", {}, "performance", "'safe' is not 'safety'"),
    ("Why did D105's escrow balance drop?", {}, "settlement", "escrow -> Settlement until the Escrow agent exists"),
    ("Why is D102 earning less and having more incidents?", {}, "performance", "multi-intent: both agents needed; today only one runs"),
    ("What is D102's policy on pay?", {}, "settlement", "driver ID wins over policy words"),
]


def run():
    passed, failures = 0, []
    for question, memory, expected, label in CASES:
        state = {"question": question, **memory}
        route = supervisor_node(state)["route"]
        if route == expected:
            passed += 1
        else:
            failures.append((question, expected, route, label))

    total = len(CASES)
    print(f"Routing accuracy: {passed}/{total} = {passed / total:.0%}\n")
    if failures:
        print("Misroutes:")
        for question, expected, route, label in failures:
            print(f"  - {question!r}: expected {expected}, got {route}  ({label})")
    return passed, total


if __name__ == "__main__":
    run()