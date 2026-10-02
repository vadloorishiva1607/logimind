"""
Standalone test for the PostgresSaver checkpointer wiring in app/graph.py.

Runs against a real local Postgres (not RDS) to verify the mechanics:
  1. checkpointer.setup() creates LangGraph's tables without error.
  2. State (driver_id, memory_summary) persists across two separate
     graph.invoke() calls that share a thread_id.
  3. A different thread_id does NOT see that state (no cross-session leak).

The Settlement agent's LLM call is monkeypatched out so this test has no
dependency on an OpenAI API key -- it is testing the checkpointer, not the
agent's answer quality (that was already exercised separately).
"""
import os

os.environ["DATABASE_URL"] = "postgresql://postgres:testpass@localhost:5432/logimind_test"

import src.tools.settlement as settlement_mod

# Avoid a real LLM/DB call inside settlement_node -- we only care about
# graph-level state persistence here.
settlement_mod.run_settlement_agent = lambda db, driver_id, question: f"(mock answer for {driver_id})"

from src.agent.executor import ask   # noqa: E402
from src.agent.memory import close_checkpointer

db = None  # unused by the mocked settlement agent

# Turn 1: mentions D102 explicitly.
r1 = ask(db, "Why did D102's settlement drop last week?", thread_id="session-A")
print("Turn 1 (session-A):", r1)
assert "D102" in r1["answer"], "expected the mocked answer to reference D102"

# Turn 2: SAME thread, no driver ID mentioned -- should still resolve to
# D102 because the checkpointer restored driver_id from turn 1's state.
r2 = ask(db, "What about the week before that?", thread_id="session-A")
print("Turn 2 (session-A, follow-up):", r2)
assert "D102" in r2["answer"], (
    "FAILED: follow-up question lost driver_id -- checkpointer state did not persist"
)

# Turn 3: a DIFFERENT thread with no driver ID -- should NOT inherit D102
# from session-A. This is the negative-scenario / no-cross-session-leak check.
r3 = ask(db, "What about the week before that?", thread_id="session-B")
print("Turn 3 (session-B, new thread):", r3)
assert "need a driver ID" in r3["answer"], (
    "FAILED: session-B incorrectly inherited state from session-A"
)

close_checkpointer()
print("\nAll checkpointer persistence checks passed.")
