import pytest

from supportpilot.llm.client import LLMResponse
from supportpilot.rag.judge import (
    applicable_metrics,
    average_precision,
    parse_json,
    score_context_precision,
    score_faithfulness,
    score_relevancy,
)

PASSAGES = [
    {"title": "A", "heading_path": "H", "text": "text a"},
    {"title": "B", "heading_path": "", "text": "text b"},
]


class FakeJudge:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[tuple[list[dict[str, str]], dict]] = []

    def judge(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return LLMResponse(self.text, "fake", "m", 10, 5, 0.1)


def test_parse_json_strips_thinking_fences_and_prose():
    assert parse_json('<think>maybe {x}</think>\n{"a": 1}') == {"a": 1}
    assert parse_json('reasoning...</think>{"a": 2}') == {"a": 2}
    assert parse_json('Here:\n```json\n{"a": 3}\n```') == {"a": 3}
    assert parse_json("no json") is None


def test_average_precision():
    assert average_precision([True, False, True]) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision([False, False]) == 0.0
    assert average_precision([True, True]) == 1.0


def test_applicable_metrics():
    all_m = ["faithfulness", "answer_relevancy", "context_precision"]
    assert applicable_metrics(True, False, all_m) == all_m
    assert applicable_metrics(True, True, all_m) == ["context_precision"]
    assert applicable_metrics(False, False, all_m) == ["faithfulness"]
    assert applicable_metrics(False, True, all_m) == []


def test_faithfulness_score_and_unsupported_claims():
    text = (
        '{"claims": [{"claim": "a", "supported": true}, {"claim": "b", "supported": false},'
        ' {"claim": "c", "supported": true}]}'
    )
    judge = FakeJudge(text)
    res, _ = score_faithfulness(judge, "answer [1]", PASSAGES, 500)
    assert res["score"] == pytest.approx(2 / 3)
    assert res["unsupported"] == ["b"]
    assert judge.calls[0][1] == {"temperature": 0.0, "max_tokens": 500}


def test_faithfulness_errors():
    assert score_faithfulness(FakeJudge("nope"), "a", PASSAGES, 1)[0]["error"] == "bad_json"
    assert (
        score_faithfulness(FakeJudge('{"claims": []}'), "a", PASSAGES, 1)[0]["error"] == "no_claims"
    )


def test_relevancy_rescales_and_validates():
    res, _ = score_relevancy(FakeJudge('{"score": 4, "reason": "x"}'), "q", "a", 1)
    assert res["score"] == 0.75
    assert score_relevancy(FakeJudge('{"score": 9}'), "q", "a", 1)[0]["error"] == "bad_json"
    assert score_relevancy(FakeJudge('{"score": "5"}'), "q", "a", 1)[0]["error"] == "bad_json"


def test_context_precision_ok_and_bad_length():
    res, _ = score_context_precision(
        FakeJudge('{"relevant": [false, true]}'), "q", "ref", PASSAGES, 1
    )
    assert res["score"] == 0.5 and res["relevant"] == [False, True]
    bad, _ = score_context_precision(FakeJudge('{"relevant": [true]}'), "q", "ref", PASSAGES, 1)
    assert bad["score"] is None and bad["error"] == "bad_flags"
