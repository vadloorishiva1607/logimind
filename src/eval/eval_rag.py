"""
eval/eval_rag.py -- retrieval evaluation for the RAG agent.

For each question in rag_eval_set.json, runs the same role-filtered
ChromaDB search the app uses (retrieve_policy_chunks, top 5) and checks
the results against the answer key, twice: with the distance threshold
only, and after the LLM relevance check (grade_chunks) as the app now
does. The first part needs only the embedding API; the relevance check
adds one short LLM call per question.

Question types:
  normal         the role may see the expected document; it should be found
  access_denied  the role may NOT see the document; the right result is
                 "no matching policy" (nothing passes the threshold)
  out_of_scope   not a logistics question; same right result

Run from the repo root (needs OPENAI_API_KEY and the ChromaDB folder):
    PYTHONPATH=. python src/eval/eval_rag.py
"""
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.tools.rag import DISTANCE_THRESHOLD, grade_chunks, retrieve_policy_chunks  # noqa: E402

DATASET = Path(__file__).with_name("rag_eval_set.json")
K_SMALL = 3   # the Recall@3 / Precision@3 cut-off
K_APP = 5     # what run_rag_agent actually retrieves


def search(question: str, role: str):
    """Returns (hits, kept_titles).

    hits: (doc_title, distance) for the top K_APP chunks, best first.
    kept_titles: the documents left after the threshold AND the relevance
    check, i.e. what the app's answer would actually use.
    """
    r = retrieve_policy_chunks(question, role, k=K_APP)
    docs = r.get("documents", [[]])[0]
    metas = r.get("metadatas", [[]])[0]
    distances = r.get("distances", [[]])[0]
    hits = [(m["doc_title"], d) for m, d in zip(metas, distances)]
    candidates = [(t, m) for t, m, d in zip(docs, metas, distances) if d <= DISTANCE_THRESHOLD]
    kept = [candidates[i][1]["doc_title"] for i in grade_chunks(question, candidates)]
    return hits, kept


def score_normal(hits, expected, kept) -> dict:
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
        "kept_ok": any(t in expected for t in kept),
        "kept_precision": sum(t in expected for t in kept) / len(kept) if kept else 0.0,
    }


def score_refusal(hits, forbidden, kept) -> dict:
    """Scores for a question that should get "no matching policy"."""
    return {
        "refused": not any(d <= DISTANCE_THRESHOLD for _, d in hits),
        "refused_after_check": not kept,
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
    print("ID   type           role            threshold only  after check")
    for c in cases:
        hits, kept = search_fn(c["question"], c["role"])
        if c["type"] == "normal":
            s = score_normal(hits, c["expected_docs"], kept)
            normal.append((c, s))
            print(f"{c['id']}  normal         {c['role']:<15} "
                  f"{'OK  ' if s['reaches_llm'] else 'MISS'}            {'OK  ' if s['kept_ok'] else 'MISS'}"
                  f"   rank {round(1 / s['rr']) if s['rr'] else '-'}, expected {_d(s['best_expected'])},"
                  f" best other {_d(s['best_other'])}, kept {kept or 'nothing'}")
        else:
            s = score_refusal(hits, c.get("forbidden_docs", []), kept)
            refusal.append((c, s))
            print(f"{c['id']}  {c['type']:<14} {c['role']:<15} "
                  f"{'OK  ' if s['refused'] else 'FAIL'}            {'OK  ' if s['refused_after_check'] else 'FAIL'}"
                  f"   best {_d(s['best'])}, kept {kept or 'nothing'}"
                  + ("  LEAKED a forbidden document!" if s["leaked"] else ""))

    n = len(normal)
    print(f"\n=== Retrieval ({n} normal questions) ===")
    print(f"Recall@3            {_avg(s['recall@3'] for _, s in normal):.0%}   right document in the top 3")
    print(f"Recall@5            {_avg(s['recall@5'] for _, s in normal):.0%}   right document in the top 5")
    print(f"MRR                 {_avg(s['rr'] for _, s in normal):.2f}   1.00 = always ranked first")
    print(f"Precision@3         {_avg(s['precision@3'] for _, s in normal):.0%}   share of the top 3 from the right document"
          " (can't reach 100%: each document has only 1-2 chunks)")
    print(f"\n=== What the answer would use ===          threshold only   after relevance check")
    print(f"Right document reaches the LLM             {_avg(s['reaches_llm'] for _, s in normal):>5.0%}"
          f"            {_avg(s['kept_ok'] for _, s in normal):>5.0%}")
    print(f"Context precision                          {_avg(s['context_precision'] for _, s in normal):>5.0%}"
          f"            {_avg(s['kept_precision'] for _, s in normal):>5.0%}")
    print(f"Correct 'no match' ({len(refusal)} refusal questions)    {_avg(s['refused'] for _, s in refusal):>5.0%}"
          f"            {_avg(s['refused_after_check'] for _, s in refusal):>5.0%}")
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