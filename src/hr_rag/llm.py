"""LLM calls through OpenRouter (OpenAI-compatible API) with a response cache.

Every response is stored in evals/cache/llm/ under the hash of (model, messages, params). A repeated call with the
same input is free and deterministic, and CI replays the committed cache without an API key
(HR_RAG_OFFLINE=1 turns a cache miss into an error instead of a network call).
"""
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "evals" / "cache" / "llm"
BASE_URL = "https://openrouter.ai/api/v1"

# USD per 1M tokens, openrouter.ai/api/v1/models on 2026-11-23
MODELS = {
    "anthropic/claude-haiku-5.5": {"input": 0.10, "output": 0.50},     # answers
    "anthropic/claude-sonnet-5.5": {"input": 2.00, "output": 10.00},   # judge: a different, stronger model
    # generator candidates compared on the eval (REPORT.md, "Вибір моделі")
    "openai/gpt-6-luna": {"input": 0.10, "output": 0.50},
    "deepseek/deepseek-v4.1-flash": {"input": 0.05, "output": 0.199},
    "mistralai/mistral-large-4-0": {"input": 0.70, "output": 2.10},
}
GENERATOR = "anthropic/claude-haiku-5.5"     # query rewrite; the answer model is set per pipeline in rag.py
JUDGE = "anthropic/claude-sonnet-5.5"


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cached: bool

    @property
    def cost(self) -> float:
        p = MODELS.get(self.model, {"input": 0, "output": 0})
        return (self.input_tokens * p["input"] + self.output_tokens * p["output"]) / 1_000_000


_client = None


def client():
    global _client
    if _client is None:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv(ROOT / ".env")
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set (see .env.example)")
        _client = OpenAI(api_key=key, base_url=BASE_URL, timeout=120, max_retries=0)
    return _client


def complete(messages: list[dict], model: str = GENERATOR, temperature: float = 0.0, max_tokens: int = 800,
             json_mode: bool = False, reasoning: bool | None = None) -> LLMResponse:
    """reasoning=False turns off the provider's hidden reasoning: by default Haiku 5.5 may spend the whole
    max_tokens on it and return an empty answer (found in the v1 eval, see REPORT.md)."""
    params = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
              "json_mode": json_mode}
    if reasoning is not None:
        params["reasoning"] = reasoning
    key = hashlib.sha1(json.dumps(params, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    path = CACHE / f"{key}.json"
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        return LLMResponse(d["text"], d["model"], d["input_tokens"], d["output_tokens"], d["latency_s"], True)
    if os.environ.get("HR_RAG_OFFLINE") == "1":
        raise RuntimeError(f"offline mode: no cached response for {key} (run the eval locally with a key first)")

    import openai
    kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if reasoning is not None:
        kwargs["extra_body"] = {"reasoning": {"enabled": reasoning}}
    t0 = time.perf_counter()
    for attempt in range(5):
        try:
            resp = client().chat.completions.create(**kwargs)
            break
        except openai.BadRequestError as e:
            if "reasoning" in str(e).lower() and "extra_body" in kwargs:   # model cannot turn reasoning off
                kwargs.pop("extra_body")
                kwargs["max_tokens"] = max(max_tokens, 4000)
                continue
            raise
        except (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    u = resp.usage
    in_tok, out_tok = (u.prompt_tokens, u.completion_tokens) if u else (0, 0)
    finish = resp.choices[0].finish_reason
    text = resp.choices[0].message.content or ""
    retried = False
    if not text.strip() and finish == "length":
        # the whole budget went into hidden reasoning (Haiku 5.5 in v1, DeepSeek in v5): one retry with 3x tokens,
        # both calls are paid and counted
        kwargs["max_tokens"] = kwargs["max_tokens"] * 3
        resp = client().chat.completions.create(**kwargs)
        u = resp.usage
        in_tok, out_tok = in_tok + (u.prompt_tokens if u else 0), out_tok + (u.completion_tokens if u else 0)
        finish, text, retried = resp.choices[0].finish_reason, resp.choices[0].message.content or "", True
    out = LLMResponse(text, model, in_tok, out_tok, round(time.perf_counter() - t0, 2), False)
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"text": out.text, "model": model, "input_tokens": out.input_tokens,
                                "output_tokens": out.output_tokens, "latency_s": out.latency_s, "finish_reason": finish,
                                "retried_empty": retried, "request": params}, ensure_ascii=False, indent=1), encoding="utf-8")
    return out
