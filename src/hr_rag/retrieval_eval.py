"""Retrieval quality: recall@k and MRR on evals/retrieval_set.jsonl, for several search configurations.

    python main.py eval-retrieval

A hit = a retrieved chunk of an expected document that contains the expected phrase (works for any chunking).
Writes evals/results/retrieval.json and retrieval.md.
"""
import json
import time
from pathlib import Path

from .search import ROOT, Index

SET = ROOT / "evals" / "retrieval_set.jsonl"
RESULTS = ROOT / "evals" / "results"
K = 10


def load_set() -> list[dict]:
    return [json.loads(l) for l in SET.open(encoding="utf-8") if l.strip()]


def is_hit(chunk: dict, expected: list[dict]) -> bool:
    text = " ".join(chunk["text"].split())
    return any(chunk["doc_id"] == e["doc"] and " ".join(e["phrase"].split()) in text for e in expected)


def lint(items: list[dict], idx: Index) -> list[str]:
    """Every expected phrase must exist in some chunk, or the question can never be answered."""
    problems = []
    for q in items:
        if not any(is_hit(c, q["expected"]) for c in idx.chunks):
            problems.append(f"{q['id']}: no chunk in {idx.name} contains any expected phrase")
    return problems


def multi_query(question: str, filters: dict | None) -> list:
    """The RAG v3 retrieval: question + LLM rewrites (uk legal terms, en), reranked against the question."""
    from . import rag
    extra, _ = rag.rewrite(question)
    return rag.index().multi_query(question, extra, K, filters)


def evaluate(idx: Index, items: list[dict], mode: str, filters: dict | None) -> dict:
    ranks, outdated_top1, latency = {}, 0, []
    for q in items:
        t0 = time.perf_counter()
        hits = multi_query(q["question"], filters) if mode == "multi_query" else idx.search(q["question"], mode, K, filters)
        latency.append(time.perf_counter() - t0)
        rank = next((h.rank for h in hits if is_hit(h.chunk, q["expected"])), None)
        ranks[q["id"]] = rank
        outdated_top1 += bool(hits) and hits[0].chunk.get("is_current") is False
    n = len(items)
    found = [r for r in ranks.values() if r]
    return {"recall@1": round(sum(r == 1 for r in found) / n, 3), "recall@5": round(sum(r <= 5 for r in found) / n, 3),
            "recall@10": round(len(found) / n, 3), "mrr@10": round(sum(1 / r for r in found) / n, 3),
            "outdated_top1": outdated_top1, "ms_per_query": round(1000 * sum(latency) / n),
            "missed@5": [qid for qid, r in ranks.items() if not r or r > 5], "ranks": ranks}


def run() -> dict:
    items = load_set()
    structure, fixed = Index("chunks.jsonl", "structure"), Index("chunks_fixed.jsonl", "fixed")
    problems = lint(items, structure) + lint(items, fixed)
    for p in problems:
        print("LINT", p)
    configs = [("fixed chunks, vector", fixed, "vector", None),
               ("vector", structure, "vector", None),
               ("BM25", structure, "bm25", None),
               ("hybrid (RRF)", structure, "hybrid", None),
               ("hybrid + rerank", structure, "hybrid_rerank", None),
               ("hybrid + rerank, current versions only", structure, "hybrid_rerank", {"is_current": True}),
               ("+ query rewrite (uk + en), current only", structure, "multi_query", {"is_current": True})]
    results = {}
    for name, idx, mode, filters in configs:
        results[name] = evaluate(idx, items, mode, filters)
        r = results[name]
        print(f"{name:<42} R@1 {r['recall@1']:.2f}  R@5 {r['recall@5']:.2f}  MRR {r['mrr@10']:.3f}  "
              f"outdated@1 {r['outdated_top1']}  {r['ms_per_query']} ms  missed@5 {r['missed@5']}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "retrieval.json").write_text(json.dumps({"questions": len(items), "lint": problems,
                                                        "results": results}, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    lines = [f"# Retrieval: {len(items)} questions", "",
             "| configuration | recall@1 | recall@5 | recall@10 | MRR@10 | outdated chunk at top-1 | ms / query | missed in top-5 |",
             "|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | {r['recall@1']:.2f} | {r['recall@5']:.2f} | {r['recall@10']:.2f} | {r['mrr@10']:.3f} | "
                     f"{r['outdated_top1']} | {r['ms_per_query']} | {', '.join(r['missed@5']) or '—'} |")
    lines += ["", "Ranks of the first correct chunk per question:", "", "| question | " + " | ".join(results) + " |",
              "|---|" + "---|" * len(results)]
    for q in items:
        lines.append(f"| {q['id']} ({q['kind']}) | " + " | ".join(str(r["ranks"][q["id"]] or "—") for r in results.values()) + " |")
    (RESULTS / "retrieval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results
