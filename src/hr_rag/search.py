"""Index and hybrid search: embeddings in Qdrant + BM25 over lemmas, fused with RRF, reranked by a cross-encoder.

    python main.py index                     # embed chunks (cached) and report the index
    python main.py search "скільки днів відпустки" [--mode hybrid_rerank] [--current-only]

Modes: vector | bm25 | hybrid | hybrid_rerank. Metadata filters: is_current, doc_type, lang, doc_id.
Embeddings are cached in data/index/ by chunk content, so the index is rebuilt in seconds and CI does not embed.
"""
import atexit
import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchAny, MatchValue, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
INDEX = ROOT / "data" / "index"

EMBED_MODEL = "intfloat/multilingual-e5-large"     # e5 expects "query: " / "passage: " prefixes
RERANK_MODEL = "BAAI/bge-reranker-v2-m3-int8"
RRF_K = 60
CANDIDATES = 50          # per retriever before fusion
RERANK_TOP = 20          # fused candidates sent to the cross-encoder

STOP = set("і й та а але або чи що як це той ця ці в у на до з із зі за по від для про при під над без між "
           "не ні же ж би б є бути його її їх він вона воно вони ми ви я ти який яка яке які "
           "the a an of to in on for and or is are be by with as at from that this it".split())


# ---------------------------------------------------------------- text for the models

def body(text: str) -> str:
    """Chunk text without its leading heading line (the header already carries it)."""
    return re.sub(r"^#+ [^\n]*\n+", "", text.strip())


def passage(c: dict) -> str:
    return f"passage: {c['header']}\n{body(c['text'])}"           # contextual header + text


@lru_cache(maxsize=1)
def morph():
    import pymorphy3
    return pymorphy3.MorphAnalyzer(lang="uk")


@lru_cache(maxsize=200_000)
def lemma(word: str) -> str:
    if re.search(r"[а-яіїєґ]", word):
        return morph().parse(word)[0].normal_form
    return word[:-1] if len(word) > 4 and word.endswith("s") else word     # crude English plural


def tokens(text: str) -> list[str]:
    words = re.findall(r"[\w'’ʼ]+", text.casefold().replace("’", "'").replace("ʼ", "'"))
    return [lemma(w) for w in words if w not in STOP and len(w) > 1]


# ---------------------------------------------------------------- embeddings with a cache

def _offline_guard(what: str) -> None:
    import os
    if os.environ.get("HR_RAG_OFFLINE") == "1":
        raise RuntimeError(f"offline mode: {what} not in the committed cache (run the eval locally first)")


@lru_cache(maxsize=1)
def embedder():
    _offline_guard("embedding")
    from fastembed import TextEmbedding
    return TextEmbedding(EMBED_MODEL)


@lru_cache(maxsize=1)
def reranker():
    _offline_guard("rerank score")
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(RERANK_MODEL)


def cache_path(name: str) -> Path:
    return INDEX / f"emb_{name}_{EMBED_MODEL.split('/')[-1]}.npz"


def embed_passages(texts: list[str], name: str) -> np.ndarray:
    """Embed only texts not seen before; the cache key is the sha1 of the exact passage text."""
    import hashlib
    INDEX.mkdir(parents=True, exist_ok=True)
    path = cache_path(name)
    cache = dict(np.load(path)) if path.exists() else {}
    keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
    todo = [(k, t) for k, t in zip(keys, texts) if k not in cache]
    if todo:
        print(f"  embedding {len(todo)} new passages ({name}) with {EMBED_MODEL} ...", flush=True)
        for (k, _), vec in zip(todo, embedder().embed([t for _, t in todo], batch_size=16)):
            cache[k] = vec.astype(np.float16)
        np.savez_compressed(path, **cache)
    return np.stack([cache[k].astype(np.float32) for k in keys])


class _Store:
    """Small persistent key -> value cache (query vectors, rerank scores), saved at exit.
    Committed to the repo, so CI replays retrieval without downloading the models."""
    def __init__(self, path: Path):
        self.path, self.dirty = path, False
        self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        atexit.register(self.save)

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        self.data[key], self.dirty = value, True

    def save(self):
        if self.dirty:
            INDEX.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, ensure_ascii=False, sort_keys=True), encoding="utf-8")
            self.dirty = False


def _sha(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()


QUERY_CACHE = _Store(INDEX / "query_vectors.json")
RERANK_CACHE = _Store(INDEX / "rerank_scores.json")


def embed_query(q: str) -> np.ndarray:
    key = _sha(EMBED_MODEL, q)
    vec = QUERY_CACHE.get(key)
    if vec is None:
        vec = [round(float(x), 5) for x in next(iter(embedder().embed([f"query: {q}"])))]
        QUERY_CACHE.put(key, vec)
    return np.array(vec, dtype=np.float32)


def rerank_scores(q: str, docs: list[str]) -> list[float]:
    keys = [_sha(RERANK_MODEL, q, d) for d in docs]
    todo = [i for i, k in enumerate(keys) if RERANK_CACHE.get(k) is None]
    if todo:
        for i, s in zip(todo, reranker().rerank(q, [docs[i] for i in todo])):
            RERANK_CACHE.put(keys[i], round(float(s), 5))
    return [RERANK_CACHE.get(k) for k in keys]


# ---------------------------------------------------------------- index

@dataclass
class Hit:
    chunk: dict
    score: float
    rank: int


class Index:
    def __init__(self, chunks_file: str = "chunks.jsonl", name: str = "structure"):
        self.chunks = [json.loads(l) for l in (PROCESSED / chunks_file).open(encoding="utf-8")]
        for i, c in enumerate(self.chunks):          # the fixed baseline has no metadata: fill what filters need
            c.setdefault("chunk_id", f"{c['doc_id']}~{i}")
            c.setdefault("header", c.get("title", c["doc_id"]))
        self.name = name
        self.vectors = embed_passages([passage(c) for c in self.chunks], name)
        self.bm25 = BM25Okapi([tokens(c["header"] + " " + body(c["text"])) for c in self.chunks])
        self.qdrant = QdrantClient(":memory:")
        self.qdrant.create_collection(name, vectors_config=VectorParams(size=self.vectors.shape[1],
                                                                        distance=Distance.COSINE))
        payload_keys = ("doc_id", "doc_type", "lang", "is_current")
        self.qdrant.upsert(name, points=[
            PointStruct(id=i, vector=v.tolist(), payload={k: c.get(k) for k in payload_keys})
            for i, (c, v) in enumerate(zip(self.chunks, self.vectors))])

    # filters: {"is_current": True, "doc_type": ["law", "company_policy"], "lang": "uk", "doc_id": [...]}
    def _allowed(self, c: dict, filters: dict | None) -> bool:
        for key, value in (filters or {}).items():
            if key not in c:
                continue
            if isinstance(value, (list, tuple, set)) and c[key] not in value:
                return False
            if not isinstance(value, (list, tuple, set)) and c[key] != value:
                return False
        return True

    def _qfilter(self, filters: dict | None):
        conds = []
        for key, value in (filters or {}).items():
            if key not in ("doc_id", "doc_type", "lang", "is_current") or (self.name != "structure" and key != "doc_id"):
                continue
            conds.append(FieldCondition(key=key, match=MatchAny(any=list(value)) if isinstance(value, (list, tuple, set))
                                        else MatchValue(value=value)))
        return Filter(must=conds) if conds else None

    def vector(self, q: str, k: int = CANDIDATES, filters: dict | None = None) -> list[Hit]:
        res = self.qdrant.query_points(self.name, query=embed_query(q).tolist(), limit=k,
                                       query_filter=self._qfilter(filters)).points
        return [Hit(self.chunks[p.id], p.score, r) for r, p in enumerate(res, 1)]

    def lexical(self, q: str, k: int = CANDIDATES, filters: dict | None = None) -> list[Hit]:
        scores = self.bm25.get_scores(tokens(q))
        order = [i for i in np.argsort(-scores) if scores[i] > 0 and self._allowed(self.chunks[i], filters)][:k]
        return [Hit(self.chunks[i], float(scores[i]), r) for r, i in enumerate(order, 1)]

    def hybrid(self, q: str, k: int = CANDIDATES, filters: dict | None = None, bm25_weight: float = 1.0) -> list[Hit]:
        fused: dict[str, float] = {}
        by_id: dict[str, dict] = {}
        for hits, w in ((self.vector(q, CANDIDATES, filters), 1.0), (self.lexical(q, CANDIDATES, filters), bm25_weight)):
            for h in hits:
                fused[h.chunk["chunk_id"]] = fused.get(h.chunk["chunk_id"], 0.0) + w / (RRF_K + h.rank)
                by_id[h.chunk["chunk_id"]] = h.chunk
        order = sorted(fused, key=fused.get, reverse=True)[:k]
        return [Hit(by_id[cid], fused[cid], r) for r, cid in enumerate(order, 1)]

    def rerank(self, q: str, cands: list[Hit], k: int) -> list[Hit]:
        if not cands:
            return []
        scores = rerank_scores(q, [f"{h.chunk['header']}\n{body(h.chunk['text'])}" for h in cands])
        order = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)[:k]
        return [Hit(cands[i].chunk, float(scores[i]), r) for r, i in enumerate(order, 1)]

    def hybrid_rerank(self, q: str, k: int = 10, filters: dict | None = None) -> list[Hit]:
        return self.rerank(q, self.hybrid(q, RERANK_TOP, filters), k)

    def multi_query(self, q: str, extra: list[str], k: int = 10, filters: dict | None = None,
                    reserve: int = 0) -> list[Hit]:
        """Candidates from the question and its rewrites, reranked against the ORIGINAL question: the rewrites
        widen recall, but relevance is judged by what the user asked. reserve > 0 keeps the best `reserve` hits of
        each rewrite (reranked against the rewrite itself): a question with a false premise ("одноразова
        компенсація 3 000 грн") makes the cross-encoder prefer chunks that echo the premise."""
        cands: dict[str, Hit] = {}
        pools = []
        for query in [q, *extra]:
            pool = self.hybrid(query, RERANK_TOP, filters)
            pools.append((query, pool))
            for h in pool:
                cands.setdefault(h.chunk["chunk_id"], h)
        ranked = self.rerank(q, list(cands.values()), len(cands))
        if not reserve:
            return [Hit(h.chunk, h.score, i) for i, h in enumerate(ranked[:k], 1)]
        keep: list[str] = []
        for query, pool in pools[1:]:
            for h in self.rerank(query, pool, reserve):
                if h.chunk["chunk_id"] not in keep:
                    keep.append(h.chunk["chunk_id"])
        head = [h for h in ranked if h.chunk["chunk_id"] not in keep][:max(k - len(keep), 0)]
        by_id = {h.chunk["chunk_id"]: h for h in ranked}
        out = sorted(head + [by_id[c] for c in keep if c in by_id], key=lambda h: h.score, reverse=True)
        return [Hit(h.chunk, h.score, i) for i, h in enumerate(out[:k], 1)]

    def search(self, q: str, mode: str = "hybrid_rerank", k: int = 10, filters: dict | None = None) -> list[Hit]:
        if mode == "vector":
            return self.vector(q, k, filters)
        if mode == "bm25":
            return self.lexical(q, k, filters)
        if mode == "hybrid":
            return self.hybrid(q, k, filters)
        return self.hybrid_rerank(q, k, filters)
