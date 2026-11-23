"""End-to-end eval of the RAG answers on evals/testset.jsonl.

    python main.py eval --version v2 [--split dev|test|all] [--no-judge] [--check]

Writes evals/results/answers_<v>.jsonl (answer + sources + scores per question) and metrics_<v>.json.
--check compares the metrics with evals/thresholds.json and exits 1 if any is below its threshold (used in CI,
where HR_RAG_OFFLINE=1 replays the committed LLM / retrieval caches, so CI needs neither an API key nor the models).
"""
import json
import statistics
import sys
from pathlib import Path

from evals import scorers
from src.hr_rag import rag
from src.hr_rag.search import ROOT

TESTSET = ROOT / "evals" / "testset.jsonl"
RESULTS = ROOT / "evals" / "results"
THRESHOLDS = ROOT / "evals" / "thresholds.json"


def load_testset(split: str = "all") -> list[dict]:
    items = [json.loads(l) for l in TESTSET.open(encoding="utf-8") if l.strip()]
    if split == "devtest":                       # model / prompt selection: everything except the holdout
        return [q for q in items if q["split"] in ("dev", "test")]
    return [q for q in items if split == "all" or q["split"] == split]


def mean(values) -> float | None:
    v = [float(x) for x in values if x is not None]
    return round(sum(v) / len(v), 3) if v else None


def aggregate(rows: list[dict]) -> dict:
    S = [r["scores"] for r in rows]
    ans = [r for r in rows if r["answerable"]]
    no = [r for r in rows if not r["answerable"]]
    lat = sorted(r["latency_s"] for r in rows if r["llm_called"])
    cost = sum(r["cost"] for r in rows)
    in_tok = mean(r["input_tokens"] for r in rows if r["llm_called"])
    m = {
        "questions": len(rows),
        "answer_accuracy": mean(s.get("judge_pass") for s in S),              # LLM judge + exact no-answer check
        "no_answer_accuracy": mean(r["scores"]["abstain_ok"] for r in no),    # said "не знаю" when it should
        "false_unknown_rate": mean(r["unknown"] for r in ans),                # said "не знаю" although answerable
        "key_facts": mean(r["scores"].get("key_facts") for r in ans),
        "cited_rate": mean(r["scores"].get("cited") for r in ans if not r["unknown"]),
        "citation_coverage": mean(r["scores"].get("citation_coverage") for r in ans if not r["unknown"]),
        "invalid_citations": sum(r["scores"].get("invalid_citations", 0) for r in rows),
        "source_hit": mean(r["scores"].get("source_hit") for r in ans if not r["unknown"]),
        "outdated_cited": sum(bool(r["scores"].get("outdated_cited")) for r in rows),
        "faithfulness": mean(s.get("faithfulness") for s in S),
        "citation_precision": mean(s.get("citation_precision") for s in S),
        "avg_input_tokens": in_tok,
        "latency_p50_s": round(statistics.median(lat), 2) if lat else None,
        "latency_p95_s": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 2) if lat else None,
        "generation_cost_usd": round(cost, 5),
        "cost_per_1000_questions_usd": round(1000 * cost / len(rows), 3) if rows else None,
    }
    by_cat: dict[str, list] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["scores"].get("judge_pass"))
    m["accuracy_by_category"] = {k: mean(v) for k, v in sorted(by_cat.items())}
    return m


def run(version: str = rag.DEFAULT, split: str = "all", use_judge: bool = True) -> dict:
    items = load_testset(split)
    rows, judge_cost = [], 0.0
    for i, q in enumerate(items, 1):
        a = rag.answer(q["question"], version)
        s, c = scorers.score(q, a, use_judge)
        judge_cost += c
        rows.append({"id": q["id"], "split": q["split"], "category": q["category"], "answerable": q["answerable"],
                     "question": q["question"], "expected_answer": q["expected_answer"], "answer": a.text,
                     "unknown": a.unknown, "llm_called": a.llm_called, "filters": a.filters,
                     "cited_sources": [{"n": n, "chunk_id": a.sources[n - 1]["chunk_id"],
                                        "label": rag.source_label(a.sources[n - 1])} for n in a.cited],
                     "sources": [s_["chunk_id"] for s_ in a.sources], "retrieved": a.retrieved,
                     "input_tokens": a.input_tokens, "output_tokens": a.output_tokens, "cost": a.cost,
                     "latency_s": a.latency_s, "scores": s})
        mark = "ok " if s.get("judge_pass", s["abstain_ok"]) else "BAD"
        print(f"[{i:>2}/{len(items)}] {mark} {q['id']:<4} {q['category']:<18} {a.text[:90]!r}", flush=True)
    m = aggregate(rows)
    m.update(version=version, split=split, judge=use_judge, judge_cost_usd=round(judge_cost, 4))
    RESULTS.mkdir(parents=True, exist_ok=True)
    suffix = version if split == "all" else f"{version}_{split}"
    with (RESULTS / f"answers_{suffix}.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (RESULTS / f"metrics_{suffix}.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    return m


def check(m: dict) -> bool:
    """Thresholds: {"metric": {"min": x}} or {"metric": {"max": x}}."""
    th = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    ok = True
    for k, rule in th.items():
        if k.startswith("_"):
            continue
        v = m.get(k)
        bad = v is None or ("min" in rule and v < rule["min"]) or ("max" in rule and v > rule["max"])
        print(f"  {'FAIL' if bad else 'pass'}  {k:<22} {v}  (threshold {rule})")
        ok &= not bad
    return ok


def print_metrics(m: dict) -> None:
    skip = {"accuracy_by_category"}
    for k, v in m.items():
        if k not in skip:
            print(f"  {k:<28} {v}")
    print("  accuracy by category:", ", ".join(f"{k} {v}" for k, v in m["accuracy_by_category"].items()))


def main(version: str, split: str, use_judge: bool, do_check: bool) -> None:
    m = run(version, split, use_judge)
    print_metrics(m)
    if do_check and not check(m):
        sys.exit(1)
