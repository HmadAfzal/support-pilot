from supportpilot.rag.retrieval_eval import (
    aggregate,
    bootstrap_ci,
    gold_chunk_ids,
    index_chunks_by_doc,
    score_question,
)

CHUNKS = [
    {"chunk_id": "a", "doc_id": "d1", "text": "foo evidence here bar"},
    {"chunk_id": "b", "doc_id": "d1", "text": "evidence here baz"},
    {"chunk_id": "c", "doc_id": "d1", "text": "nothing relevant"},
    {"chunk_id": "x", "doc_id": "d2", "text": "evidence here"},
]


def test_gold_chunks_need_doc_and_evidence():
    by_doc = index_chunks_by_doc(CHUNKS)
    item = {"doc_id": "d1", "evidence": "evidence here"}
    assert gold_chunk_ids(item, by_doc) == {"a", "b"}


def test_gold_chunks_empty_when_evidence_split_or_doc_missing():
    by_doc = index_chunks_by_doc(CHUNKS)
    assert gold_chunk_ids({"doc_id": "d1", "evidence": "bar evidence"}, by_doc) == set()
    assert gold_chunk_ids({"doc_id": "zzz", "evidence": "evidence"}, by_doc) == set()


def test_score_question_hit_recall_rr():
    sc = score_question(["c", "a", "z"], {"a", "b"}, [1, 3])
    assert sc["hit@1"] == 0.0 and sc["hit@3"] == 1.0
    assert sc["recall@3"] == 0.5
    assert sc["rank"] == 2 and sc["rr"] == 0.5


def test_score_question_miss_and_no_gold():
    sc = score_question(["c", "z"], {"a"}, [1, 3])
    assert sc["rank"] is None and sc["rr"] == 0.0 and sc["hit@3"] == 0.0
    assert score_question(["a"], set(), [1])["recall@1"] == 0.0


def test_aggregate_means():
    rows = [score_question(["a"], {"a"}, [1]), score_question(["z"], {"a"}, [1])]
    agg = aggregate(rows, [1])
    assert agg["hit@1"] == 0.5 and agg["mrr"] == 0.5


def test_bootstrap_ci_deterministic_and_bounded():
    vals = [1.0, 0.0, 1.0, 1.0, 0.0, 0.5]
    lo, hi = bootstrap_ci(vals, n=200, seed=1)
    assert (lo, hi) == bootstrap_ci(vals, n=200, seed=1)
    assert 0.0 <= lo <= hi <= 1.0
    assert bootstrap_ci([], n=10) == (0.0, 0.0)
