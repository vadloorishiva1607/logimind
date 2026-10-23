"""
src/agent/output_guardrail.py -- output guardrail: a number checker.

The Settlement and Performance agents give the LLM a block of facts that
code computed (the "context"). This check confirms that every number in
the LLM's answer actually appears in that context, so an invented or
miscopied figure is caught before the user sees it.

Plain Python, no LLM: the checker must not make the same mistake as the
thing it is checking.
"""
import logging
import re

logger = logging.getLogger(__name__)

# A number not glued to a letter, so "D102" (a driver ID) is skipped.
# Matches 644.76, 1,253.36, 88.59 (from 88.59%), 2026 (from a date).
_NUMBER_RE = re.compile(r"(?<![A-Za-z\d.])\d[\d,]*(?:\.\d+)?")

# Small whole numbers are counts and list markers ("1 incident",
# "2 periods", "1."), not data figures, so they are not checked.
_SMALL_COUNT = 10


def _numbers(text: str) -> list[float]:
    return [float(n.replace(",", "")) for n in _NUMBER_RE.findall(text)]


def _supported(value: float, known: list[float]) -> bool:
    """True if `value` is in the data, allowing normal rounding (75.62 -> 75.6 or 76)."""
    for k in known:
        if abs(value - k) < 0.005 or value == round(k, 1) or value == round(k):
            return True
    return False


def find_unsupported_numbers(answer: str, context: str) -> list[float]:
    """Return the numbers in `answer` that do not appear in `context`."""
    known = _numbers(context)
    unsupported = []
    for value in _numbers(answer):
        if value.is_integer() and value <= _SMALL_COUNT:
            continue
        if not _supported(value, known):
            unsupported.append(value)
    return unsupported


def check_answer(answer: str, context: str, agent: str) -> str:
    """Return the answer, with a warning added if any figure can't be verified."""
    unsupported = find_unsupported_numbers(answer, context)
    if not unsupported:
        return answer

    logger.warning("Output guardrail (%s): figures not found in data: %s", agent, unsupported)
    figures = ", ".join(f"{v:g}" for v in unsupported)
    return (
        f"{answer}\n\n"
        f"Note: some figures in this answer ({figures}) could not be verified "
        f"against the source data. Please double-check them before acting."
    )