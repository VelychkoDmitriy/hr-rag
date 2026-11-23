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
    # v5: premise check - the rewrite also extracts the user's assumptions and a neutral question without them;
    # the neutral question is searched too (with a reserved slot), the assumptions go into the prompt to be checked
    "v5": {"prompt": "answer_v5", "retrieve": 8, "budget": 2500, "min_score": None, "max_tokens": 1000,
           "rewrite": True, "rewrite_prompt": "query_rewrite_v2", "route": True, "reasoning": False,
           "merge": "rerank_original", "reserve": 1, "model": "deepseek/deepseek-v4.1-flash"},
    # experiments: v4 with other generators (v4-<model>), see MODEL_CANDIDATES below
    # experiment: v4 with a stronger generator (the rewrite stays on Haiku) - is the remaining error the model?
    "v4-sonnet": {"prompt": "answer_v4", "retrieve": 8, "budget": 2500, "min_score": None, "max_tokens": 4000,
                  "rewrite": True, "route": True, "reasoning": None, "merge": "rerank_original", "reserve": 1,
                  "model": "anthropic/claude-sonnet-5.5"},
}
# generator comparison on dev+test with the v4 prompt (REPORT.md, "Вибір моделі"); DeepSeek won and became default
for _short, _model in {"gpt6luna": "openai/gpt-6-luna", "deepseek": "deepseek/deepseek-v4.1-flash",
                       "mistral": "mistralai/mistral-large-4-0"}.items():
    PIPELINES[f"v4-{_short}"] = {**PIPELINES["v4"], "model": _model}
PIPELINES["v4-mistral"].update(reasoning=None, max_tokens=4000)    # rejects reasoning=false, reasons by default
DEFAULT = "v5"

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


def rewrite(question: str, prompt: str = "query_rewrite_v1") -> tuple[list[str], llm.LLMResponse | None]:
    queries, _, r = rewrite_full(question, prompt)
    return queries, r


def rewrite_full(question: str, prompt: str = "query_rewrite_v1") -> tuple[list[str], list[str], llm.LLMResponse | None]:
    """-> (search queries, user's assumptions, response). query_rewrite_v2 adds assumptions + a neutral question."""
    system, user = load_prompt(prompt)
    r = llm.complete([{"role": "system", "content": system}, {"role": "user", "content": user.format(question=question)}],
                     llm.GENERATOR, temperature=0.0, max_tokens=200 if prompt == "query_rewrite_v1" else 400,
                     json_mode=True, reasoning=False)
    try:
        d = json.loads(re.search(r"\{.*\}", r.text, re.S).group(0))
        queries = [q for q in d.get("queries", []) if isinstance(q, str) and q.strip()][:2]
        assumptions = [a for a in d.get("assumptions", []) if isinstance(a, str) and a.strip()][:3]
        if isinstance(d.get("neutral"), str) and d["neutral"].strip():
            queries.append(d["neutral"].strip())
    except (AttributeError, KeyError, TypeError, json.JSONDecodeError):
        queries, assumptions = [], []
    return queries, assumptions, r


def retrieve(question: str, cfg: dict, filters: dict | None):
    """-> (hits, queries, rewrite response, effective filters, user's assumptions)"""
    queries, r = [question], None
    if cfg.get("route") and (docs := route(question)):
        filters = {**(filters or {}), "doc_id": docs}    # old versions stay filtered out unless OLD_RE matched
    assumptions: list[str] = []
    if cfg.get("rewrite"):
        extra, assumptions, r = rewrite_full(question, cfg.get("rewrite_prompt", "query_rewrite_v1"))
        queries += [q for q in extra if q.casefold() != question.casefold()]
    if cfg.get("merge") == "rerank_original":
        hits = index().multi_query(question, queries[1:], cfg["retrieve"], filters, cfg.get("reserve", 0))
        return hits, queries, r, filters, assumptions
    best: dict[str, Hit] = {}
    for q in queries:
        for h in index().search(q, "hybrid_rerank", cfg["retrieve"], filters):
            if h.chunk["chunk_id"] not in best or h.score > best[h.chunk["chunk_id"]].score:
                best[h.chunk["chunk_id"]] = h
    hits = sorted(best.values(), key=lambda h: h.score, reverse=True)[:cfg["retrieve"]]
    return [Hit(h.chunk, h.score, i) for i, h in enumerate(hits, 1)], queries, r, filters, assumptions


def answer(question: str, version: str = DEFAULT, filters: dict | None = "auto") -> Answer:
    cfg = PIPELINES[version]
    if filters == "auto":
        filters = None if OLD_RE.search(question) else {"is_current": True}
    hits, queries, rw, filters, assumptions = retrieve(question, cfg, filters)
    retrieved = [{"chunk_id": h.chunk["chunk_id"], "score": round(h.score, 4)} for h in hits]
    sources = pack(hits, cfg["budget"])
    rw_in, rw_out, rw_cost = (rw.input_tokens, rw.output_tokens, rw.cost) if rw else (0, 0, 0.0)

    if not sources or (cfg["min_score"] is not None and sources[0]["score"] < cfg["min_score"]):
        return Answer(question, "Не знаю: у документах немає відповіді на це питання.", True, sources, [], [],
                      filters, version, queries, rw_in, rw_out, rw_cost, llm_called=False, retrieved=retrieved)

    system, user = load_prompt(cfg["prompt"])
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": user.format(sources=format_sources(sources), question=question,
                                                     assumptions="; ".join(assumptions) or "немає")}]
    r = llm.complete(msgs, cfg.get("model", llm.GENERATOR), temperature=0.0, max_tokens=cfg["max_tokens"],
                     reasoning=cfg["reasoning"])
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
