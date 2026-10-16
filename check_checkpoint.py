"""
Diagnostic: does the PostgresSaver actually have a persisted checkpoint
for the thread_id you tested with, and what does it contain?

Run from the repo root (new structure):
    PYTHONPATH=. python check_checkpoint.py <thread_id>
"""
import sys
from src.agent.memory import get_checkpointer

thread_id = sys.argv[1] if len(sys.argv) > 1 else "default"

cp = get_checkpointer()
config = {"configurable": {"thread_id": thread_id}}

tuple_ = cp.get_tuple(config)

if tuple_ is None:
    print(f"NO checkpoint found for thread_id={thread_id!r}.")
    print("-> This means the first /chat call never actually got persisted")
    print("   under this thread_id, or you're testing a different thread_id")
    print("   than the one that was used before.")
else:
    print(f"Checkpoint FOUND for thread_id={thread_id!r}")
    print("Channel values restored:")
    values = tuple_.checkpoint.get("channel_values", {})
    for k, v in values.items():
        print(f"  {k!r}: {v!r}")
    print()
    print("Full config echoed back:", tuple_.config)
