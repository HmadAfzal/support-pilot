import pytest

from supportpilot.rag.judge_spotcheck import agreement, draw_sample, judged_faithfulness


def gen(qid, config="c1", refused=False, text="answer"):
    return {"id": qid, "config": config, "refused": refused, "text": text}


def judge(qid, score, config="c1", metric="faithfulness"):
    return {"id": qid, "config": config, "metric": metric, "score": score}


def test_judged_faithfulness_filters_refused_missing_and_other_metrics():
    gens = [gen("a"), gen("b", refused=True), gen("c"), gen("d")]
    judges = [
        judge("a", 1.0),
        judge("b", 1.0),
        judge("c", None),
        judge("d", 1.0, metric="answer_relevancy"),
    ]
    rows = judged_faithfulness(gens, judges)
    assert [r["id"] for r in rows] == ["a"]
    assert rows[0]["judge_score"] == 1.0


def _rows():
    rows = []
    for i in range(20):
        for cfg in ("c1", "c2"):
            score = 0.5 if i < 6 else 1.0
            text = "```code```" if i % 5 == 0 else "plain"
            rows.append({**gen(f"q{i}", cfg, text=text), "judge_score": score})
    return rows


def test_draw_sample_counts_unique_ids_and_determinism():
    a = draw_sample(_rows(), 4, 6, 2, seed=1)
    b = draw_sample(_rows(), 4, 6, 2, seed=1)
    assert [(r["id"], r["config"]) for r in a] == [(r["id"], r["config"]) for r in b]
    assert len(a) == 10
    assert len({r["id"] for r in a}) == 10
    assert sum(1 for r in a if r["judge_score"] < 1.0) == 4
    assert sum(1 for r in a if r["judge_score"] >= 1.0 and "```" in r["text"]) >= 2


def test_agreement_perfect_and_chance():
    perfect = agreement([(True, True), (True, True), (False, False), (False, False)])
    assert perfect["agreement"] == 1.0 and perfect["kappa"] == 1.0
    chance = agreement([(True, True), (True, False), (False, False), (False, True)])
    assert chance["agreement"] == 0.5 and chance["kappa"] == pytest.approx(0.0)
    assert chance["judge_lenient"] == 1 and chance["judge_strict"] == 1


def test_agreement_confirmation_rates_and_empty():
    s = agreement([(True, True), (True, False), (False, False)])
    assert s["judge_faithful_confirmed"] == 0.5
    assert s["judge_flagged_confirmed"] == 1.0
    assert agreement([]) == {"n": 0}
