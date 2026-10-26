"""
eval/eval_rag.py -- retrieval evaluation for the RAG agent.

For each question in rag_eval_set.json, runs the same role-filtered
ChromaDB search the app uses (retrieve_policy_chunks, top 5) and checks
the results against the answer key. No LLM is called: this measures
retrieval only (the embedding API is called once per question).

Question types:
  normal         the role may see the expected document; it should be found
  access_denied  the role may NOT see the document; the right result is
                 "no matching policy" (nothing passes the threshold)
  out_of_scope   not a logistics question; same right result

Run from the repo root (needs OPENAI_API_KEY and the ChromaDB folder):
    PYTHONPATH=. python eval/eval_rag.py
"""
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.tools.rag import DISTANCE_THRESHOLD, retrieve_policy_chunks  # noqa: E402

DATASET = Path(__file__).with_name("rag_eval_set.json")
K_SMALL = 3   # the Recall@3 / Precision@3 cut-off
K_APP = 5     # what run_rag_agent actually retrieves


def search(question: str, role: str) -> list[tuple[str, float]]:
    """(doc_title, distance) for the top K_APP chunks, best first."""
    r = retrieve_policy_chunks(question, role, k=K_APP)
    titles = [m["doc_title"] for m in r.get("metadatas", [[]])[0]]
    distances = r.get("distances", [[]])[0]
    return list(zip(titles, distances))


def score_normal(hits, expected) -> dict:
    """Scores for a question whose answer is in `expected` documents."""
    top = [t for t, _ in hits[:K_SMALL]]
    passed = [t for t, d in hits if d <= DISTANCE_THRESHOLD]   # what the LLM would see
    rank = next((i + 1 for i, (t, _) in enumerate(hits) if t in expected), None)
    return {
        "recall@3": any(t in expected for t in top),
        "recall@5": rank is not None,
        "rr": 1 / rank if rank else 0.0,
        "precision@3": sum(t in expected for t in top) / len(top) if top else 0.0,
        "reaches_llm": any(t in expected for t in passed),
        "context_precision": sum(t in expected for t in passed) / len(passed) if passed else 0.0,
        "best_expected": min((d for t, d in hits if t in expected), default=None),
        "best_other": min((d for t, d in hits if t not in expected), default=None),
    }


def score_refusal(hits, forbidden) -> dict:
    """Scores for a question that should get "no matching policy"."""
    return {
        "refused": not any(d <= DISTANCE_THRESHOLD for _, d in hits),
        "leaked": any(t in forbidden for t, _ in hits),   # role filter broken?
        "best": hits[0][1] if hits else None,
    }


def _avg(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _d(x):
    return "  -  " if x is None else f"{x:.3f}"


def run(search_fn=search, dataset=DATASET):
    cases = json.loads(Path(dataset).read_text())
    normal, refusal = [], []

    print(f"Threshold {DISTANCE_THRESHOLD}, top {K_APP} retrieved\n")
    print("ID   type           role            result")
    for c in cases:
        hits = search_fn(c["question"], c["role"])
        if c["type"] == "normal":
            s = score_normal(hits, c["expected_docs"])
            normal.append((c, s))
            mark = "OK  " if s["reaches_llm"] else "MISS"
            print(f"{c['id']}  normal         {c['role']:<15} {mark} rank {round(1 / s['rr']) if s['rr'] else '-'}"
                  f"  expected {_d(s['best_expected'])}  best other {_d(s['best_other'])}")
        else:
            s = score_refusal(hits, c.get("forbidden_docs", []))
            refusal.append((c, s))
            mark = "OK  " if s["refused"] and not s["leaked"] else "FAIL"
            print(f"{c['id']}  {c['type']:<14} {c['role']:<15} {mark} best {_d(s['best'])}"
                  + ("  LEAKED a forbidden document!" if s["leaked"] else "")
                  + ("" if s["refused"] else f"  (would answer from {hits[0][0]})"))

    n = len(normal)
    print(f"\n=== Retrieval ({n} normal questions) ===")
    print(f"Recall@3            {_avg(s['recall@3'] for _, s in normal):.0%}   right document in the top 3")
    print(f"Recall@5            {_avg(s['recall@5'] for _, s in normal):.0%}   right document in the top 5")
    print(f"MRR                 {_avg(s['rr'] for _, s in normal):.2f}   1.00 = always ranked first")
    print(f"Precision@3         {_avg(s['precision@3'] for _, s in normal):.0%}   share of the top 3 from the right document"
          " (can't reach 100%: each document has only 1-2 chunks)")
    print(f"\n=== After the {DISTANCE_THRESHOLD} threshold (what the LLM sees) ===")
    print(f"Reaches the LLM     {_avg(s['reaches_llm'] for _, s in normal):.0%}   right document passes the threshold")
    print(f"Context precision   {_avg(s['context_precision'] for _, s in normal):.0%}   share of passing chunks from the right document")
    print(f"\n=== Refusals ({len(refusal)} access-denied / out-of-scope questions) ===")
    print(f"Correct 'no match'  {_avg(s['refused'] for _, s in refusal):.0%}")
    print(f"Role-filter leaks   {sum(s['leaked'] for _, s in refusal)}   (must be 0)")

    worst_relevant = max((s["best_expected"] for _, s in normal if s["best_expected"] is not None), default=None)
    closest_refusal = min((s["best"] for _, s in refusal if s["best"] is not None), default=None)
    print(f"\n=== Threshold check ===")
    print(f"Worst distance of a right document   {_d(worst_relevant)}   (must be <= {DISTANCE_THRESHOLD})")
    print(f"Closest distance on a refusal case   {_d(closest_refusal)}   (must be >  {DISTANCE_THRESHOLD})")
    if worst_relevant is not None and closest_refusal is not None:
        gap = closest_refusal - worst_relevant
        print(f"Gap between them                     {gap:.3f}   "
              + ("a threshold fits in the gap" if gap > 0 else "OVERLAP: no single threshold separates them"))


if __name__ == "__main__":
    run()