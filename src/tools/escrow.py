"""
src/tools/escrow.py -- Escrow agent.

Answers questions about a driver's escrow balance and why it changed.
Data comes from the MCP server via escrow_mcp_client.py, where the LLM
chooses between two tools (current balance, or transaction history).

Code reconciles the balance: each balance change must equal the
transaction amount. A gap is flagged as an "unexplained discrepancy"
(Design Doc, Escrow Agent negative scenario). The LLM only explains, and
the output guardrail checks every figure in its answer.
"""
from src.agent.output_guardrail import check_answer
from src.models.llm_client import get_llm
from src.tools.escrow_mcp_client import get_escrow_data_via_mcp

ESCROW_SYSTEM_PROMPT = """You are LogiMind's Escrow agent. Answer using \
ONLY the escrow data provided below, and quote specific dates and dollar \
figures from it.

Rules:
1. Use the precomputed figures exactly as given. Do not recompute them.
2. For a balance question, give the current balance and the date and type \
of the latest transaction.
3. For a question about why the balance changed, lead with the LATEST \
CHANGE: the balance before and after, and the transaction that caused it, \
with its reason.
4. If the data lists an UNEXPLAINED DISCREPANCY, state it plainly with its \
amount, say it needs review by Finance, and do not guess its cause.
5. You cannot change escrow. If asked, say a Finance user must do it in \
the existing application.
6. If the data is insufficient to answer, say so instead of guessing."""


def _money(x):
    return "n/a" if x is None else f"${x:,.2f}"


def _signed(x):
    return "n/a" if x is None else f"{'+' if x >= 0 else '-'}${abs(x):,.2f}"


def _describe(t: dict) -> str:
    return (f"{t['transaction_date']} {t['type']} {_signed(t['amount_usd'])}, "
            f"balance after {_money(t['balance_after'])}, reason: \"{t['reason']}\"")


def reconcile(rows: list[dict]) -> list[dict]:
    """Check that each balance change equals its transaction amount.

    `rows` is the history (any order). Returns one dict per mismatch, with
    the date, the expected and recorded balances, and the difference.
    The oldest row can't be checked: there is no earlier balance to start from.
    """
    ordered = sorted(rows, key=lambda r: r["transaction_date"])  # oldest first
    problems = []
    for prev, cur in zip(ordered, ordered[1:]):
        expected = round(prev["balance_after"] + cur["amount_usd"], 2)
        difference = round(cur["balance_after"] - expected, 2)
        if abs(difference) >= 0.01:
            problems.append({
                "date": cur["transaction_date"],
                "previous_balance": prev["balance_after"],
                "amount": cur["amount_usd"],
                "expected": expected,
                "recorded": cur["balance_after"],
                "difference": difference,
            })
    return problems


def build_escrow_context(tool: str, data: dict) -> str:
    """Turn the MCP tool's output into plain facts for the prompt."""
    if tool == "get_escrow_balance":
        last = data["last_transaction"]
        return (f"CURRENT BALANCE: {_money(data['balance'])} "
                f"(after the transaction on {last['transaction_date']})\n"
                f"LATEST TRANSACTION: {_describe(last)}")

    rows = data["rows"]  # newest first
    latest = rows[0]
    lines = [f"CURRENT BALANCE: {_money(latest['balance_after'])} "
             f"(after the transaction on {latest['transaction_date']})"]

    if len(rows) >= 2:
        prev = rows[1]
        change = round(latest["balance_after"] - prev["balance_after"], 2)
        lines.append(
            f"LATEST CHANGE: balance {_money(prev['balance_after'])} on {prev['transaction_date']} "
            f"-> {_money(latest['balance_after'])} on {latest['transaction_date']} "
            f"({_signed(change)}), caused by: {_describe(latest)}"
        )
        if latest["amount_usd"] < 0 and prev["balance_after"]:
            share = round(abs(latest["amount_usd"]) / prev["balance_after"] * 100, 1)
            lines.append(f"The latest withdrawal was {share}% of the previous balance.")

    problems = reconcile(rows)
    lines += ["", "RECONCILIATION (each balance change checked against its transaction):"]
    if problems:
        for p in problems:
            lines.append(
                f"- UNEXPLAINED DISCREPANCY on {p['date']}: expected balance "
                f"{_money(p['expected'])} ({_money(p['previous_balance'])} {_signed(p['amount'])}), "
                f"recorded {_money(p['recorded'])}, difference {_signed(p['difference'])} "
                f"with no matching transaction."
            )
    else:
        lines.append("- All balance changes match their transactions.")

    lines += ["", "ALL TRANSACTIONS (newest first):"]
    lines += [f"- {_describe(t)}" for t in rows]
    return "\n".join(lines)


def run_escrow_agent(db, driver_id: str, question: str = "") -> str:
    """`db` is accepted for interface consistency with the other agents; unused."""
    try:
        tool, data = get_escrow_data_via_mcp(driver_id, question)
    except Exception:
        return (f"I couldn't retrieve escrow data for {driver_id} right now. "
                "Please try again in a moment.")

    has_data = data.get("last_transaction") if tool == "get_escrow_balance" else data.get("rows")
    if not has_data:
        return (f"I don't have any escrow records for {driver_id}. "
                "Double-check the driver ID, or this driver may have no escrow account.")

    context = build_escrow_context(tool, data)
    llm = get_llm()
    prompt = (
        f"{ESCROW_SYSTEM_PROMPT}\n\n"
        f"Driver: {driver_id}\n\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    return check_answer(answer, context, "escrow")