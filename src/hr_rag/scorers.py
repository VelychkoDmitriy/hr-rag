"""Scorers for one RAG answer: deterministic checks first, then two LLM judges (correctness, faithfulness).

Deterministic (free, exact, run in CI on every commit):
  abstain_ok        answered when the testset has an answer, "не знаю" when it has none
  key_facts         share of must_include groups found in the answer (numbers normalised: "15 000" == "15000")
  cited             an answer (not "не знаю") cites at least one valid source
  citation_coverage share of sentences that carry a [n] citation
  invalid_citations [n] outside the source list
  source_hit        a cited source is an expected document containing the expected phrase
  outdated_cited    the answer cites an outdated version although the question did not ask about one
LLM judges (anthropic/claude-sonnet-5.5, a different and stronger model than the generator):
  judge_pass        correctness against the reference answer (calibrated on human labels, see calibrate.py)
  faithfulness      share of claims supported by the sources; citation_precision: share backed by the cited source
"""
import json
import re

from . import llm
from .rag import Answer, citations, format_sources, load_prompt

SENT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")
OUTDATED_OK = {"outdated_explicit", "false_premise"}       # citing the old version is part of a good answer here


def norm(text: str) -> str:
    t = text.casefold().replace("’", "'").replace("ʼ", "'").replace(" ", " ")
    return re.sub(r"(?<=\d)[ ,.](?=\d{3}\b)", "", t)        # 15 000 / 15,000 -> 15000


def phrase_in(phrase: str, text: str) -> bool:
    return " ".join(phrase.split()) in " ".join(text.split())


def sentences(text: str) -> list[str]:
    parts = [s.strip(" -*•") for s in SENT_RE.split(text)]
    return [s for s in parts if len(re.sub(r"\[\d[\d,; ]*\]", "", s).strip()) > 15]   # skip "[1]"-only fragments


def deterministic(item: dict, a: Answer) -> dict:
    s: dict = {"abstain_ok": a.unknown != item["answerable"]}
    if not item["answerable"]:
        return s
    t = norm(a.text)
    groups = item["must_include"]
    s["key_facts"] = round(sum(any(norm(alt) in t for alt in g) for g in groups) / len(groups), 3) if groups else None
    if a.unknown:
        return s
    sents = sentences(a.text)
    s["cited"] = bool(a.cited)
    s["citation_coverage"] = round(sum(bool(citations(x)) for x in sents) / len(sents), 3) if sents else 0.0
    s["invalid_citations"] = len(a.invalid_citations)
    cited_chunks = [a.sources[n - 1] for n in a.cited]
    if item["expected_sources"]:
        s["source_hit"] = any(c["doc_id"] == e["doc"] and phrase_in(e["phrase"], c["text"])
                              for c in cited_chunks for e in item["expected_sources"])
    s["outdated_cited"] = item["category"] not in OUTDATED_OK and any(c.get("is_current") is False for c in cited_chunks)
    return s


def _judge(prompt: str, **fields) -> tuple[dict, llm.LLMResponse]:
    system, user = load_prompt(prompt)
    r = llm.complete([{"role": "system", "content": system}, {"role": "user", "content": user.format(**fields)}],
                     llm.JUDGE, temperature=0.0, max_tokens=900, json_mode=True)
    m = re.search(r"\{.*\}", r.text, re.S)
    try:
        return json.loads(m.group(0) if m else r.text), r
    except json.JSONDecodeError:
        return {"error": r.text[:300]}, r


def judge_correctness(item: dict, a: Answer) -> tuple[dict, float]:
    d, r = _judge("judge_correctness_v1", question=item["question"], expected=item["expected_answer"], answer=a.text)
    return {"judge_pass": bool(d.get("pass")), "judge_reason": d.get("reason", d.get("error", ""))}, r.cost


def judge_faithfulness(a: Answer) -> tuple[dict, float]:
    d, r = _judge("judge_faithfulness_v1", sources=format_sources(a.sources), answer=a.text)
    claims = d.get("claims") or []
    if not claims:
        return {"faithfulness": None, "citation_precision": None, "claims": []}, r.cost
    return {"faithfulness": round(sum(bool(c.get("supported")) for c in claims) / len(claims), 3),
            "citation_precision": round(sum(bool(c.get("citation_ok")) for c in claims) / len(claims), 3),
            "unsupported": [c.get("claim") for c in claims if not c.get("supported")]}, r.cost


def score(item: dict, a: Answer, use_judge: bool = True) -> tuple[dict, float]:
    s, cost = deterministic(item, a), 0.0
    if not use_judge:
        return s, cost
    if not item["answerable"]:
        s["judge_pass"] = s["abstain_ok"]                       # no-answer: correctness is exact, no LLM needed
        return s, cost
    if a.unknown:
        s.update(judge_pass=False, judge_reason="answered 'не знаю' although the testset has an answer")
        return s, cost
    j, c1 = judge_correctness(item, a)
    f, c2 = judge_faithfulness(a)
    s.update(j)
    s.update(f)
    return s, c1 + c2
