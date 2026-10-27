from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import openai
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_CACHE = Path(".cache/llm_cache.sqlite")

RETRYABLE = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.InternalServerError,
)

PROVIDER_SPECS = [
    ("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "GROQ_MODEL"),
    (
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
    ),
]


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    api_key: str
    model: str


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cached: bool = False


class AllProvidersFailed(RuntimeError):
    pass


def default_providers(require_model: bool = True) -> list[Provider]:
    providers = []
    for name, base_url, key_env, model_env in PROVIDER_SPECS:
        key, model = os.getenv(key_env), os.getenv(model_env, "")
        if key and (model or not require_model):
            providers.append(Provider(name, base_url, key, model))
    if not providers:
        raise RuntimeError(
            "No LLM provider configured. Set <PROVIDER>_API_KEY and <PROVIDER>_MODEL in .env"
        )
    return providers


class LLM:
    def __init__(
        self,
        providers: list[Provider] | None = None,
        cache_path: Path | str | None = DEFAULT_CACHE,
        max_attempts: int = 2,
        backoff_s: float = 2.0,
        client_factory: Callable[[Provider], Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        judge_model: str | None = None,
    ) -> None:
        self.providers = providers or default_providers()
        self.max_attempts = max_attempts
        self.backoff_s = backoff_s
        self.judge_model = judge_model or os.getenv("JUDGE_MODEL")
        self._make_client = client_factory or self._openai_client
        self._clients: dict[str, Any] = {}
        self._sleep = sleep
        self._db: sqlite3.Connection | None = None
        if cache_path is not None:
            if str(cache_path) != ":memory:":
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(cache_path))
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )

    @staticmethod
    def _openai_client(p: Provider) -> OpenAI:
        return OpenAI(base_url=p.base_url, api_key=p.api_key, timeout=60, max_retries=0)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        provider: str | None = None,
        fallback: bool = True,
        use_cache: bool = True,
        model: str | None = None,
    ) -> LLMResponse:
        order = self._order(provider, fallback, model)
        key = self._key(messages, temperature, max_tokens, order)
        if use_cache:
            hit = self._get(key)
            if hit is not None:
                return replace(hit, cached=True)

        errors: list[str] = []
        for p in order:
            for attempt in range(self.max_attempts):
                try:
                    resp = self._call(p, messages, temperature, max_tokens)
                except RETRYABLE as exc:
                    errors.append(f"{p.name}: {type(exc).__name__}: {str(exc)[:200]}")
                    if attempt + 1 < self.max_attempts:
                        self._sleep(self.backoff_s * 2**attempt)
                    continue
                if use_cache:
                    self._put(key, resp)
                return resp
        raise AllProvidersFailed("; ".join(errors))

    def judge(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        if not self.judge_model:
            raise RuntimeError("JUDGE_MODEL is not set in .env")
        return self.chat(messages, model=self.judge_model, **kwargs)

    def _order(
        self, provider: str | None, fallback: bool, model: str | None = None
    ) -> list[Provider]:
        if model is not None:
            name = provider or self.providers[0].name
            base = next((p for p in self.providers if p.name == name), None)
            if base is None:
                raise ValueError(f"Unknown or unconfigured provider: {name}")
            return [replace(base, model=model)]
        if provider is None:
            return list(self.providers)
        first = [p for p in self.providers if p.name == provider]
        if not first:
            raise ValueError(f"Unknown or unconfigured provider: {provider}")
        rest = [p for p in self.providers if p.name != provider] if fallback else []
        return first + rest

    def _call(
        self,
        p: Provider,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int | None,
    ) -> LLMResponse:
        if p.name not in self._clients:
            self._clients[p.name] = self._make_client(p)
        kwargs: dict[str, Any] = {
            "model": p.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        start = time.perf_counter()
        r = self._clients[p.name].chat.completions.create(**kwargs)
        latency = time.perf_counter() - start
        usage = getattr(r, "usage", None)
        return LLMResponse(
            text=r.choices[0].message.content or "",
            provider=p.name,
            model=p.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_s=latency,
        )

    @staticmethod
    def _key(
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int | None,
        order: list[Provider],
    ) -> str:
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "providers": [(p.name, p.model) for p in order],
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _get(self, key: str) -> LLMResponse | None:
        if self._db is None:
            return None
        row = self._db.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return LLMResponse(**json.loads(row[0])) if row else None

    def _put(self, key: str, resp: LLMResponse) -> None:
        if self._db is None:
            return
        self._db.execute(
            "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
            (key, json.dumps(asdict(resp), ensure_ascii=False)),
        )
        self._db.commit()
