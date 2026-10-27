"""
demo/run_demo.py -- end-to-end demo of LogiMind, in story order.

Sends 12 questions to the running app (POST /chat), one at a time, and
prints each answer with the agent that handled it, the time taken, and
what to point out. Covers UC-1 to UC-10. Also works as an end-to-end
check after any change: every step should show the expected agent.

Start the app first (PYTHONPATH=. python main.py), then, from the repo
root in another window:
    python demo/run_demo.py            # pauses after each step (press Enter)
    python demo/run_demo.py --no-pause # runs straight through
"""
import json
import sys
import time
import urllib.request

URL = "http://localhost:8000/chat"
RUN = time.strftime("%H%M%S")   # fresh thread IDs each run, so old memory can't leak in

# (use case, question, role, thread, expected agents, what to point out)
STEPS = [
    ("UC-1 Settlement", "Why did D102 settlement drop?", "finance", "pay", ["settlement"],
     "Python picks the latest week and does the arithmetic; the LLM only explains (-$644.76)."),
    ("UC-8 Fuel (follow-up)", "Why was the fuel cost so high?", "finance", "pay", ["fuel"],
     "No driver ID: memory knows it's D102. Fuel splits +$303 into gallons vs price and cites the Fuel policy."),
    ("UC-6 Shipment", "Why is shipment S10023 delayed?", "dispatcher", "ship", ["shipment"],
     "4.0 h late with a known cause = ROUTINE under the Shipment Delay Risk SOP (not escalated)."),
    ("UC-2 Performance", "Why is D104 safety score dropping?", "driver_manager", "perf", ["performance"],
     "-13.41 points in 3 weeks plus an incident -> higher-priority review under the Safety policy."),
    ("UC-5 Escrow", "Why did D105 escrow balance drop?", "finance", "escrow", ["escrow"],
     "Code reconciles every balance: a $91.12 unexplained discrepancy and two policy checks, guaranteed by the safety net."),
    ("UC-7 Driver (dispatcher)", "Tell me about D103", "dispatcher", "drv-d", ["driver"],
     "Onboarding for 254 days (SOP: flag after 45). Pay and escrow are hidden: removed in code before the LLM."),
    ("UC-7 Driver (finance)", "Tell me about D103", "finance", "drv-f", ["driver"],
     "Same question, finance role: now net pay and escrow balance are shown."),
    ("UC-3 Policy (allowed)", "What is the escrow withdrawal policy?", "finance", "rag-f", ["rag"],
     "RAG: role-filtered search, relevance check, answer cites the Escrow Withdrawal Policy."),
    ("UC-3 Policy (denied)", "What is the escrow withdrawal policy?", "dispatcher", "rag-d", ["rag"],
     "Dispatchers can't see the escrow policy: 'no matching policy', nothing leaked."),
    ("UC-9 Action refused", "Terminate D102", "ops_manager", "act", [],
     "Input guardrail blocks it before the Supervisor: LogiMind is read-only."),
    ("UC-10 Sensitive data", "What is the bank account number of D102?", "finance", "pii", [],
     "Blocked even for finance: sensitive data goes through the HR/payroll system."),
    ("UC-4 Unclear question", "Hello", "dispatcher", "hello", [],
     "No driver, shipment or policy word: the Supervisor asks for clarification instead of guessing."),
]


def ask(question: str, role: str, thread: str) -> tuple[dict, float]:
    body = json.dumps({"question": question, "role": role, "thread_id": f"demo-{RUN}-{thread}"}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    start = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data, time.time() - start


def main(pause: bool):
    passed, times = 0, []
    for n, (uc, question, role, thread, expected, point) in enumerate(STEPS, 1):
        print("=" * 78)
        print(f"Step {n}/{len(STEPS)}  {uc}   (role: {role})")
        print(f"Q: {question}")
        try:
            data, secs = ask(question, role, thread)
        except Exception as e:  # app not running, timeout, server error
            print(f"   ERROR: {e}")
            continue
        times.append(secs)
        agents = data.get("agents_used", [])
        ok = agents == expected
        passed += ok
        print(f"Agent: {agents or 'none (guardrail or clarify)'}   {'OK' if ok else f'EXPECTED {expected}'}"
              f"   ({secs:.1f}s)\n")
        print(data.get("answer", ""))
        print(f"\n>> Point out: {point}")
        if pause and n < len(STEPS):
            input("\n[Enter for the next step] ")

    print("=" * 78)
    print(f"Routing as expected: {passed}/{len(STEPS)}")
    if times:
        print(f"Response time: average {sum(times) / len(times):.1f}s, slowest {max(times):.1f}s")


if __name__ == "__main__":
    main(pause="--no-pause" not in sys.argv)