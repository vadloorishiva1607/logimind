"""
src/agent/guardrails.py -- input guardrail (runs BEFORE the Supervisor).

Plain Python rules, no LLM: a guardrail must be predictable, cheap, and
impossible to talk out of its decision with a clever prompt.

Blocks two kinds of request:
  1. Action requests (UC-9): LogiMind is read-only. It explains data; it
     never terminates, approves, transfers or changes anything.
  2. Sensitive data (UC-10): asking for a driver's bank account, SSN,
     tax ID or password, or a message that contains such a number.
"""
import logging
import re

from src.agent.state import DRIVER_ID_RE

logger = logging.getLogger(__name__)

# An action request STARTS with an action verb, optionally after "please" /
# "can you" / "could you". Matching only at the start avoids false alarms
# like "Why did D102's pay change?" or "What is the termination procedure?".
_ACTION_RE = re.compile(
    r"^\s*(please\s+)?((can|could|would|will)\s+you\s+)?(please\s+)?"
    r"(terminate|suspend|fire|approve|reject|transfer|delete|remove|update|"
    r"change|modify|cancel|deactivate|reset)\b",
    re.IGNORECASE,
)

# Words that ask for a person's sensitive data.
_SENSITIVE_RE = re.compile(
    r"\b(bank account|account number|routing number|ssn|social security|"
    r"tax id|password)\b",
    re.IGNORECASE,
)

# Sensitive numbers typed into the message itself.
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")   # 123-45-6789
_LONG_NUMBER_RE = re.compile(r"\b\d{9,}\b")      # bank account / card numbers

_MESSAGES = {
    "action_request": (
        "I can't take actions such as terminating, approving, transferring "
        "or changing records. LogiMind is read-only: I can explain a driver's "
        "data or a company policy, and a manager can take the action in the "
        "proper system."
    ),
    "sensitive_data_in_message": (
        "Please don't share account numbers, SSNs or other sensitive numbers "
        "here. Remove the number and ask your question again."
    ),
    "sensitive_data_request": (
        "I can't share sensitive personal data such as bank account numbers, "
        "SSNs, tax IDs or passwords. Please use the HR or payroll system, "
        "which has the right access controls."
    ),
}


def check_input(question: str, remembered_driver=None):
    """Return a block reason, or None if the question may continue."""
    if _ACTION_RE.search(question):
        return "action_request"
    if _SSN_RE.search(question) or _LONG_NUMBER_RE.search(question):
        return "sensitive_data_in_message"
    about_a_driver = DRIVER_ID_RE.search(question) or remembered_driver
    if about_a_driver and _SENSITIVE_RE.search(question):
        return "sensitive_data_request"
    return None


def guardrail_node(state):
    # First node of every turn: start this turn's agents_used list fresh,
    # otherwise the checkpointer carries it over and it grows every turn.
    reason = check_input(state["question"], state.get("driver_id"))
    if reason is None:
        return {**state, "blocked_reason": None, "agents_used": []}

    # Log the reason, never the question: it may contain the sensitive number.
    logger.warning("Guardrail blocked input: reason=%s", reason)
    return {**state, "blocked_reason": reason, "answer": _MESSAGES[reason],
            "agents_used": []}