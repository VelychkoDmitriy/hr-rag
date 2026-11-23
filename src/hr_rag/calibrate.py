"""Calibration of the LLM correctness judge against human labels.

    python main.py label-sheet   # evals/human_labels.csv: answers to label by hand (judge verdict hidden)
    python main.py calibrate     # agreement, Cohen's kappa, confusion matrix, disagreements

A person fills the column human_pass with 1 (correct) or 0 (wrong). Only labelled rows are used.
Writes evals/results/calibration.json and calibration.md.
"""
import csv
import json

from .search import ROOT

RESULTS = ROOT / "evals" / "results"
SHEET = ROOT / "evals" / "human_labels.csv"
FIELDS = ["key", "question", "expected_answer", "system_answer", "human_pass", "comment"]


def answers() -> dict[str, dict]:
    out = {}
    for path in sorted(RESULTS.glob("answers_v*.jsonl")):
        version = path.stem.split("_")[1]
        if path.stem.count("_") > 1:          # split-specific files duplicate the full run
            continue
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            if r["answerable"] and not r["unknown"] and "judge_reason" in r["scores"]:
                out[f"{version}:{r['id']}"] = r
    return out


LABEL_VERSIONS = ("v1", "v5")       # the first and the final version: weak and strong answers, both verdicts


def write_sheet() -> None:
    old = {}
    if SHEET.exists():
        old = {r["key"]: r for r in csv.DictReader(SHEET.open(encoding="utf-8-sig"))}
    rows = {k: r for k, r in answers().items() if k.split(":")[0] in LABEL_VERSIONS}
    with SHEET.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for key, r in rows.items():
            prev = old.get(key, {})
            w.writerow({"key": key, "question": r["question"], "expected_answer": r["expected_answer"],
                        "system_answer": r["answer"], "human_pass": prev.get("human_pass", ""),
                        "comment": prev.get("comment", "")})
    print(f"{SHEET.relative_to(ROOT)}: {len(rows)} answers; fill human_pass with 1/0, then run: python main.py calibrate")


def kappa(pairs: list[tuple[bool, bool]]) -> float:
    n = len(pairs)
    po = sum(h == j for h, j in pairs) / n
    ph, pj = sum(h for h, _ in pairs) / n, sum(j for _, j in pairs) / n
    pe = ph * pj + (1 - ph) * (1 - pj)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def run() -> dict:
    rows = answers()
    labels = [r for r in csv.DictReader(SHEET.open(encoding="utf-8-sig")) if r["human_pass"].strip() in ("0", "1")]
    pairs, disagreements = [], []
    for lab in labels:
        r = rows.get(lab["key"])
        if not r:
            continue
        h, j = lab["human_pass"].strip() == "1", bool(r["scores"]["judge_pass"])
        pairs.append((h, j))
        if h != j:
            disagreements.append({"key": lab["key"], "human": h, "judge": j, "question": r["question"],
                                  "answer": r["answer"], "judge_reason": r["scores"]["judge_reason"],
                                  "human_comment": lab.get("comment", "")})
    if not pairs:
        print("no labelled rows in", SHEET)
        return {}
    tp = sum(h and j for h, j in pairs)
    tn = sum(not h and not j for h, j in pairs)
    fp = sum(not h and j for h, j in pairs)      # judge passes what a human failed: the dangerous error
    fn = sum(h and not j for h, j in pairs)
    res = {"labelled": len(pairs), "agreement": round((tp + tn) / len(pairs), 3), "cohen_kappa": kappa(pairs),
           "confusion": {"both_pass": tp, "both_fail": tn, "judge_pass_human_fail": fp, "judge_fail_human_pass": fn},
           "disagreements": disagreements}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "calibration.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [f"# Judge calibration: {len(pairs)} human labels", "",
          f"- agreement: **{res['agreement']:.0%}**, Cohen's kappa: **{res['cohen_kappa']}**",
          f"- both pass {tp}, both fail {tn}, judge pass / human fail **{fp}**, judge fail / human pass {fn}", ""]
    for d in disagreements:
        md += [f"## {d['key']}: human {'pass' if d['human'] else 'fail'}, judge {'pass' if d['judge'] else 'fail'}",
               f"- question: {d['question']}", f"- answer: {d['answer']}", f"- judge: {d['judge_reason']}",
               f"- human: {d['human_comment']}", ""]
    (RESULTS / "calibration.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:4]))
    return res
