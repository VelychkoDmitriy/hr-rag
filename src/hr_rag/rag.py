"""RAG answer: retrieve -> pack sources into a token budget -> LLM answer with [n] citations -> resolve citations.

    python main.py ask "Скільки днів відпустки в senior developer?" [--prompt v2]

Every claim must carry [n]; [n] is resolved to document + section + URL. "Не знаю" when the sources have no answer.
Prompt versions live in prompts/answer_<v>.md (history in prompts/CHANGELOG.md); pipeline settings per version below.
"""
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache

import tiktoken

from . import llm
from .search import ROOT, Hit, Index, body

PROMPTS = ROOT / "prompts"

PIPELINES = {
    # retrieve: top-k after rerank; budget: max tokens of sources in the prompt; min_score: reranker score below
    # which the best source is considered irrelevant and the answer is "не знаю" without calling the LLM
    # rewrite: an LLM turns the question into a Ukrainian + an English search query (false premises removed), the
    # hits of all queries are merged by the best rerank score; route: a document named in the question
    # ("Handbook V1") restricts the search to it; reasoning: False turns off hidden reasoning of the generator
    "v1": {"prompt": "answer_v1", "retrieve": 8, "budget": 2500, "min_score": None, "max_tokens": 600,
           "rewrite": False, "route": False, "reasoning": None},
    "v2": {"prompt": "answer_v2", "retrieve": 8, "budget": 2500, "min_score": None, "max_tokens": 1000,
           "rewrite": True, "route": True, "reasoning": False, "merge": "max_score"},
    # v3: rewrites only add candidates, the cross-encoder ranks them against the original question
    "v3": {"prompt": "answer_v3", "retrieve": 8, "budget": 2500, "min_score": None, "max_tokens": 1000,
           "rewrite": True, "route": True, "reasoning": False, "merge": "rerank_original"},
    # v4: + the best hit of each rewrite keeps a slot; the prompt makes the "не знаю" phrase a strict contract
    "v4": {"prompt": "answer_v4", "retrieve": 8, "budget": 2500, "min_score": None, "max_tokens": 1000,
           "rewrite": True, "route": True, "reasoning": False, "merge": "rerank_original", "reserve": 1},
}
DEFAULT = "v4"

UNKNOWN_RE = re.compile(r"^\W*(не знаю|i don'?t know)", re.I)
CITE_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
# the question asks about an old edition explicitly -> do not filter out outdated versions
OLD_RE = re.compile(r"\b(v1|стар\w*|попередн\w*|раніше|колишн\w*|до змін|редакці\w* 20\d\d|20[01]\d|2020|"
                    r"old|previous|earlier|former)\b", re.I)


@lru_cache(maxsize=1)
def index() -> Index:
    return Index()


@lru_cache(maxsize=1)
def enc():
    return tiktoken.get_encoding("cl100k_base")


def n_tokens(text: str) -> int:
    return len(enc().encode(text))


@lru_cache(maxsize=8)
def load_prompt(name: str) -> tuple[str, str]:
    system, user = (PROMPTS / f"{name}.md").read_text(encoding="utf-8").split("=== USER ===")
    return system.strip(), user.strip()


@dataclass
class Answer:
    question: str
    text: str
    unknown: bool
    sources: list[dict]               # sources shown to the model, numbered from 1
    cited: list[int]                  # source numbers the answer cites
    invalid_citations: list[int]      # [n] outside the source list
    filters: dict | None
    version: str
    queries: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    latency_s: float = 0.0
    llm_called: bool = True
    retrieved: list[dict] = field(default_factory=list)   # everything retrieved, incl. what did not fit the budget

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def source_label(c: dict) -> str:
    status = "чинна" if c.get("is_current", True) else "НЕЧИННА, застаріла версія"
    section = f" — {c['section']}" if c.get("section") else ""
    return f"{c['title']}{section} (версія {c.get('version', '?')}, {status})"


def pack(hits: list[Hit], budget: int) -> list[dict]:
    """Best-first: take sources in rerank order while they fit into the token budget."""
    out, used = [], 0
    for h in hits:
        cost = n_tokens(body(h.chunk["text"])) + 30
        if used + cost > budget:
            continue
        out.append({**h.chunk, "score": round(h.score, 4)})
        used += cost
    return out


def format_sources(sources: list[dict]) -> str:
    return "\n\n".join(f"[{i}] {source_label(s)}\n{body(s['text'])}" for i, s in enumerate(sources, 1))


def citations(text: str) -> list[int]:
    nums = []
    for group in CITE_RE.findall(text):
        nums += [int(n) for n in re.split(r"[,;]", group)]
    return nums


@lru_cache(maxsize=1)
def title_aliases() -> dict[str, set[str]]:
    """'employee handbook v1' and 'handbook v1' -> {'employee_handbook_v1'}: a document named in the question."""
    out: dict[str, set[str]] = {}
    for c in index().chunks:
        t = c["title"].casefold()
        names = {t}
        if re.search(r"v\d+$", t) and len(t.split()) > 2:
            names.add(" ".join(t.split()[-2:]))
        for n in names:
            out.setdefault(n, set()).add(c["doc_id"])
    return out


def route(question: str) -> list[str]:
    q = question.casefold()
    return sorted({d for name, docs in title_aliases().items() if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", q)
                   for d in docs})


def rewrite(question: str) -> tuple[list[str], llm.LLMResponse | None]:
    system, user = load_prompt("query_rewrite_v1")
    r = llm.complete([{"role": "system", "content": system}, {"role": "user", "content": user.format(question=question)}],
                     llm.GENERATOR, temperature=0.0, max_tokens=200, json_mode=True, reasoning=False)
    try:
        m = re.search(r"\{.*\}", r.text, re.S)
        queries = [q for q in json.loads(m.group(0))["queries"] if isinstance(q, str) and q.strip()][:2]
    except (AttributeError, KeyError, json.JSONDecodeError):
        queries = []
    return queries, r


def retrieve(question: str, cfg: dict, filters: dict | None) -> tuple[list[Hit], list[str], llm.LLMResponse | None, dict | None]:
    queries, r = [question], None
    if cfg.get("route") and (docs := route(question)):
        filters = {**(filters or {}), "doc_id": docs}    # old versions stay filtered out unless OLD_RE matched
    if cfg.get("rewrite"):
        extra, r = rewrite(question)
        queries += [q for q in extra if q.casefold() != question.casefold()]
    if cfg.get("merge") == "rerank_original":
        hits = index().multi_query(question, queries[1:], cfg["retrieve"], filters, cfg.get("reserve", 0))
        return hits, queries, r, filters
    best: dict[str, Hit] = {}
    for q in queries:
        for h in index().search(q, "hybrid_rerank", cfg["retrieve"], filters):
            if h.chunk["chunk_id"] not in best or h.score > best[h.chunk["chunk_id"]].score:
                best[h.chunk["chunk_id"]] = h
    hits = sorted(best.values(), key=lambda h: h.score, reverse=True)[:cfg["retrieve"]]
    return [Hit(h.chunk, h.score, i) for i, h in enumerate(hits, 1)], queries, r, filters


def answer(question: str, version: str = DEFAULT, filters: dict | None = "auto") -> Answer:
    cfg = PIPELINES[version]
    if filters == "auto":
        filters = None if OLD_RE.search(question) else {"is_current": True}
    hits, queries, rw, filters = retrieve(question, cfg, filters)
    retrieved = [{"chunk_id": h.chunk["chunk_id"], "score": round(h.score, 4)} for h in hits]
    sources = pack(hits, cfg["budget"])
    rw_in, rw_out, rw_cost = (rw.input_tokens, rw.output_tokens, rw.cost) if rw else (0, 0, 0.0)

    if not sources or (cfg["min_score"] is not None and sources[0]["score"] < cfg["min_score"]):
        return Answer(question, "Не знаю: у документах немає відповіді на це питання.", True, sources, [], [],
                      filters, version, queries, rw_in, rw_out, rw_cost, llm_called=False, retrieved=retrieved)

    system, user = load_prompt(cfg["prompt"])
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": user.format(sources=format_sources(sources), question=question)}]
    r = llm.complete(msgs, llm.GENERATOR, temperature=0.0, max_tokens=cfg["max_tokens"], reasoning=cfg["reasoning"])
    nums = citations(r.text)
    return Answer(question, r.text.strip(), bool(UNKNOWN_RE.match(r.text)), sources,
                  sorted({n for n in nums if 1 <= n <= len(sources)}),
                  sorted({n for n in nums if not 1 <= n <= len(sources)}),
                  filters, version, queries, r.input_tokens + rw_in, r.output_tokens + rw_out, r.cost + rw_cost,
                  r.latency_s + (rw.latency_s if rw else 0.0), retrieved=retrieved)


def render(a: Answer) -> str:
    lines = [a.text, ""]
    for n in a.cited:
        s = a.sources[n - 1]
        lines.append(f"[{n}] {source_label(s)}\n    {s['source']}")
    return "\n".join(lines)
