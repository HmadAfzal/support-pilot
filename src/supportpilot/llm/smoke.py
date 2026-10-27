from __future__ import annotations

import sys

from openai import OpenAI, OpenAIError

from supportpilot.llm.client import LLM, LLMResponse, default_providers


def list_models() -> None:
    for p in default_providers(require_model=False):
        client = OpenAI(base_url=p.base_url, api_key=p.api_key, timeout=60)
        ids = sorted(m.id for m in client.models.list())
        print(f"\n== {p.name} ({len(ids)} models) ==")
        print("\n".join(ids))


def show(label: str, r: LLMResponse) -> None:
    print(
        f"{label:<8} {r.model:<36} {r.latency_s:5.2f}s "
        f"in={r.prompt_tokens} out={r.completion_tokens} -> {r.text.strip()[:40]!r}"
    )


def run_smoke() -> None:
    llm = LLM(cache_path=":memory:")
    messages = [{"role": "user", "content": "Reply with exactly one word: pong"}]
    for p in llm.providers:
        try:
            show(p.name, llm.chat(messages, provider=p.name, fallback=False, use_cache=False))
        except (OpenAIError, RuntimeError) as exc:
            print(f"{p.name:<8} FAILED: {exc}")
    if llm.judge_model:
        try:
            show("judge", llm.judge(messages, use_cache=False))
        except (OpenAIError, RuntimeError) as exc:
            print(f"judge    FAILED: {exc}")
    else:
        print("judge    skipped (JUDGE_MODEL not set)")
    try:
        first = llm.chat(messages)
        second = llm.chat(messages)
        print(f"cache: first cached={first.cached}, second cached={second.cached}")
    except (OpenAIError, RuntimeError) as exc:
        print(f"cache check FAILED: {exc}")


if __name__ == "__main__":
    list_models() if "--list" in sys.argv else run_smoke()
