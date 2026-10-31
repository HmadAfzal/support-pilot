from types import SimpleNamespace

from supportpilot.rag.generation_eval import Budget, common_ids, pending_pairs, summarize


def gen(qid, answerable, refused, cited=("c",), gold=True, config="c1"):
    return {
        "config": config,
        "id": qid,
        "answerable": answerable,
        "refused": refused,
        "cited_chunk_ids": list(cited),
        "invalid_citations": [],
        "malformed_citations": 0,
        "gold_in_context": gold,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "latency_s": 1.0,
    }


def test_budget_spends_only_uncached_and_throttles():
    slept: list[float] = []
    b = Budget(6000, 250, slept.append)
    b.spend(SimpleNamespace(cached=True, prompt_tokens=100, completion_tokens=100))
    assert b.used == 0 and slept == [] and not b.exhausted
    b.spend(SimpleNamespace(cached=False, prompt_tokens=200, completion_tokens=100))
    assert b.used == 300 and slept == [3.0] and b.exhausted


def test_common_ids():
    gens = [
        gen("a", True, False, config="c1"),
        gen("b", True, False, config="c1"),
        gen("a", True, False, config="c2"),
    ]
    assert common_ids(gens, ["c1", "c2"]) == {"a"}


def test_pending_pairs_flags_ids_a_config_still_lacks():
    gens = [gen("a", True, False, config="c1"), gen("a", True, False, config="c2")]
    judges = [{"config": "c1", "id": "a", "metric": "faithfulness", "score": 1.0}]
    pend = pending_pairs(gens, judges, ["faithfulness"])
    assert pend == {("faithfulness", "a")}
    judges.append({"config": "c2", "id": "a", "metric": "faithfulness", "score": 0.5})
    assert pending_pairs(gens, judges, ["faithfulness"]) == set()


def test_summarize_counts_and_judge_means():
    gens = [
        gen("a", True, False),
        gen("b", True, True, cited=(), gold=False),
        gen("u1", False, True, cited=()),
        gen("u2", False, False),
    ]
    judges = [
        {"id": "a", "metric": "faithfulness", "score": 1.0},
        {"id": "u2", "metric": "faithfulness", "score": 0.5},
        {"id": "a", "metric": "answer_relevancy", "score": None},
    ]
    s = summarize("c1", gens, judges, 50, 1)
    assert (s["n_answerable"], s["n_unanswerable"]) == (2, 2)
    assert (s["correct_refusals"], s["false_refusals"]) == (1, 1)
    assert s["answered"] == 2 and s["with_valid_citation"] == 2
    assert s["gold_in_context"] == 1 and s["avg_tokens"] == 150
    assert s["metrics"]["faithfulness"]["mean"] == 0.75 and s["metrics"]["faithfulness"]["n"] == 2
    assert "answer_relevancy" not in s["metrics"]
    assert s["judge_errors"] == {"answer_relevancy": 1}
