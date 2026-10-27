from types import SimpleNamespace

import httpx
import openai
import pytest

from supportpilot.llm.client import LLM, AllProvidersFailed, Provider, default_providers

REQ = httpx.Request("POST", "http://test")


def rate_limit() -> openai.RateLimitError:
    return openai.RateLimitError("slow down", response=httpx.Response(429, request=REQ), body=None)


def bad_key() -> openai.AuthenticationError:
    return openai.AuthenticationError(
        "bad key", response=httpx.Response(401, request=REQ), body=None
    )


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        message = SimpleNamespace(content=outcome)
        usage = SimpleNamespace(prompt_tokens=3, completion_tokens=2)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


PROVIDERS = [Provider("a", "http://a", "k", "model-a"), Provider("b", "http://b", "k", "model-b")]
MSG = [{"role": "user", "content": "hi"}]


def make_llm(a_outcomes, b_outcomes):
    clients = {"a": FakeClient(a_outcomes), "b": FakeClient(b_outcomes)}
    llm = LLM(
        providers=PROVIDERS,
        cache_path=":memory:",
        client_factory=lambda p: clients[p.name],
        sleep=lambda s: None,
    )
    return llm, clients


def test_falls_back_when_first_provider_rate_limited():
    llm, clients = make_llm([rate_limit(), rate_limit()], ["from b"])
    resp = llm.chat(MSG)
    assert resp.provider == "b"
    assert resp.text == "from b"
    assert clients["a"].calls == 2


def test_retry_succeeds_on_second_attempt():
    llm, clients = make_llm([rate_limit(), "from a"], [])
    resp = llm.chat(MSG)
    assert resp.provider == "a"
    assert clients["b"].calls == 0


def test_cache_hit_skips_the_call():
    llm, clients = make_llm(["first"], [])
    first = llm.chat(MSG)
    second = llm.chat(MSG)
    assert not first.cached
    assert second.cached
    assert second.text == "first"
    assert clients["a"].calls == 1


def test_use_cache_false_calls_again():
    llm, clients = make_llm(["one", "two"], [])
    llm.chat(MSG, use_cache=False)
    assert llm.chat(MSG, use_cache=False).text == "two"
    assert clients["a"].calls == 2


def test_all_providers_failed():
    llm, _ = make_llm([rate_limit(), rate_limit()], [rate_limit(), rate_limit()])
    with pytest.raises(AllProvidersFailed):
        llm.chat(MSG)


def test_auth_error_is_not_swallowed():
    llm, clients = make_llm([bad_key()], ["from b"])
    with pytest.raises(openai.AuthenticationError):
        llm.chat(MSG)
    assert clients["b"].calls == 0


def test_provider_only_mode():
    llm, clients = make_llm([], ["from b"])
    resp = llm.chat(MSG, provider="b", fallback=False)
    assert resp.provider == "b"
    assert clients["a"].calls == 0


def test_default_providers_reads_env(monkeypatch):
    for var in ("GROQ_API_KEY", "GROQ_MODEL", "GEMINI_API_KEY", "GEMINI_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("GROQ_MODEL", "some-model")
    providers = default_providers()
    assert [p.name for p in providers] == ["groq"]
    assert providers[0].model == "some-model"


def test_model_override_uses_that_model_without_fallback():
    seen = []
    clients = {"a": FakeClient(["ok"]), "b": FakeClient([])}
    orig = clients["a"]._create

    def spy(**kwargs):
        seen.append(kwargs["model"])
        return orig(**kwargs)

    clients["a"].chat.completions.create = spy
    llm = LLM(
        providers=PROVIDERS,
        cache_path=":memory:",
        client_factory=lambda p: clients[p.name],
        sleep=lambda s: None,
    )
    resp = llm.chat(MSG, model="judge-model")
    assert seen == ["judge-model"]
    assert resp.model == "judge-model"
    assert clients["b"].calls == 0


def test_judge_uses_judge_model_and_separate_cache_entry():
    clients = {"a": FakeClient(["gen", "judged"]), "b": FakeClient([])}
    llm = LLM(
        providers=PROVIDERS,
        cache_path=":memory:",
        client_factory=lambda p: clients[p.name],
        sleep=lambda s: None,
        judge_model="judge-model",
    )
    assert llm.chat(MSG).text == "gen"
    judged = llm.judge(MSG)
    assert judged.text == "judged"
    assert judged.model == "judge-model"


def test_judge_requires_model(monkeypatch):
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    llm, _ = make_llm([], [])
    with pytest.raises(RuntimeError):
        llm.judge(MSG)
