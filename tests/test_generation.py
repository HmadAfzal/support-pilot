from supportpilot.llm.client import LLMResponse
from supportpilot.rag.generation import (
    SENTINEL,
    answer_question,
    build_messages,
    is_refusal,
    parse_citations,
)
from supportpilot.rag.retrieval import Hit


def make_hits(n: int) -> list[Hit]:
    return [
        Hit(f"c{i}", 0.0, f"text {i}", {"title": f"Doc {i}", "heading_path": f"H{i}"})
        for i in range(1, n + 1)
    ]


class FakeLLM:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[tuple[list[dict[str, str]], dict]] = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return LLMResponse(self.text, "fake", "m", 10, 5, 0.1)


class StubRetriever:
    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits

    def search(self, query, k):
        return self.hits[:k]


def test_build_messages_numbers_passages_and_includes_question():
    msgs = build_messages("How?", make_hits(2))
    assert msgs[0]["role"] == "system" and SENTINEL in msgs[0]["content"]
    user = msgs[1]["content"]
    assert "[1] Doc 1 | H1\ntext 1" in user
    assert "[2] Doc 2 | H2\ntext 2" in user
    assert user.endswith("Question: How?")


def test_parse_citations_valid_invalid_and_dedupe():
    valid, invalid = parse_citations("A [1]. B [2][1]. C [7]. D [0].", 3)
    assert valid == [1, 2]
    assert invalid == [7, 0]


def test_parse_citations_comma_form():
    valid, invalid = parse_citations("Both [1, 3] and [9, 2].", 3)
    assert valid == [1, 3, 2]
    assert invalid == [9]


def test_parse_citations_ignores_non_citation_brackets():
    assert parse_citations("Use [a] and [] and [1x].", 3) == ([], [])


def test_is_refusal_variants():
    assert is_refusal(SENTINEL)
    assert is_refusal(f"  {SENTINEL}\n")
    assert is_refusal("I don\u2019t know based on the provided documentation.")
    assert not is_refusal("You can do it by clicking Settings [1].")


def test_answer_maps_citations_to_chunk_ids():
    llm = FakeLLM("Do X [2]. Then Y [1][5].")
    ans = answer_question(
        "q", StubRetriever(make_hits(3)), llm, k=3, temperature=0.0, max_tokens=99
    )
    assert ans.cited_chunk_ids == ["c2", "c1"]
    assert ans.invalid_citations == [5]
    assert not ans.refused
    assert llm.calls[0][1] == {"temperature": 0.0, "max_tokens": 99}
    assert ans.provider == "fake" and ans.completion_tokens == 5


def test_answer_detects_refusal():
    ans = answer_question("q", StubRetriever(make_hits(3)), FakeLLM(SENTINEL), k=3)
    assert ans.refused
    assert ans.cited_chunk_ids == []


def test_no_hits_refuses_without_calling_llm():
    llm = FakeLLM("should not be used")
    ans = answer_question("q", StubRetriever([]), llm)
    assert ans.refused and ans.text == SENTINEL
    assert llm.calls == []


def test_k_limits_passages_sent():
    llm = FakeLLM("ok [1]")
    answer_question("q", StubRetriever(make_hits(5)), llm, k=2)
    user = llm.calls[0][0][1]["content"]
    assert "[2]" in user and "[3]" not in user


def test_foreign_citation_format_is_counted():
    llm = FakeLLM("Do X【1†L13-L22】.")
    ans = answer_question("q", StubRetriever(make_hits(2)), llm, k=2)
    assert ans.malformed_citations == 1
    assert ans.cited_chunk_ids == ["c1"]


def test_parse_citations_mixed_formats_keep_order():
    assert parse_citations("A [2]. B【1†L3-L4】. C [9].", 3) == ([2, 1], [9])
