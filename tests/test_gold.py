import random

import pytest

from supportpilot.rag.gold import (
    allocate,
    candidate_sections,
    evidence_in_text,
    is_gold_hit,
    longest_common_run,
    parse_json_object,
    select_sections,
    split_dev_test,
)


def test_evidence_ignores_whitespace_and_smart_quotes():
    text = "Use the \u201cplay\u201d button.\n\nThen   wait."
    assert evidence_in_text('Use the "play" button. Then wait.', text)
    assert not evidence_in_text("Use the stop button.", text)
    assert not evidence_in_text("", text)


def test_is_gold_hit_needs_doc_and_evidence():
    assert is_gold_hit("d1", "a b c evidence here d", "d1", "evidence here")
    assert not is_gold_hit("d2", "a b c evidence here d", "d1", "evidence here")
    assert not is_gold_hit("d1", "a b c", "d1", "evidence here")


def test_longest_common_run():
    assert longest_common_run("how do I set up a runner", "you set up a runner with docker") == 4
    assert longest_common_run("alpha beta", "gamma delta") == 0


def test_allocate_floor_and_total():
    out = allocate(10, {"a": 9, "b": 1}, floor=2)
    assert out == {"a": 7, "b": 3}
    assert sum(out.values()) == 10


def test_allocate_rejects_impossible_floor():
    with pytest.raises(ValueError):
        allocate(3, {"a": 1, "b": 1}, floor=2)


def test_candidate_sections_filters_size_and_headings():
    text = "# T\n\n## Setup\n" + "word " * 150 + "\n\n## Related topics\n" + "x " * 400
    doc = {"doc_id": "d1", "area": "ci_cd", "title": "T", "url": "u", "text": text}
    secs = candidate_sections(doc, 100, 2000, ["related topics"])
    assert [s["heading_path"] for s in secs] == ["T > Setup"]


def _cands():
    return [
        {"doc_id": f"d{i // 5}", "heading_path": f"h{i}", "text": "x", "area": "a"}
        for i in range(10)
    ]


def test_select_sections_caps_per_doc_and_is_deterministic():
    a = select_sections(_cands(), 4, 2, random.Random(1))
    b = select_sections(_cands(), 4, 2, random.Random(1))
    assert a == b and len(a) == 4
    assert sorted(c["doc_id"] for c in a) == ["d0", "d0", "d1", "d1"]


def test_select_sections_cap_grows_for_single_doc_area():
    cands = [{"doc_id": "d0", "heading_path": f"h{i}", "text": "x"} for i in range(10)]
    assert len(select_sections(cands, 6, 2, random.Random(1))) == 6


def test_parse_json_object():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object("no json here") is None
    assert parse_json_object("{broken") is None


def _items():
    items = []
    for area in ("a", "b"):
        for answerable in (True, False):
            for i in range(6):
                items.append(
                    {"id": f"{area}{int(answerable)}{i}", "area": area, "answerable": answerable}
                )
    return items


def test_split_is_stratified_disjoint_and_deterministic():
    dev, test = split_dev_test(_items(), 1 / 3, seed=42)
    assert len(dev) + len(test) == 24
    assert not {x["id"] for x in dev} & {x["id"] for x in test}
    for area in ("a", "b"):
        for answerable in (True, False):
            n = sum(1 for x in test if x["area"] == area and x["answerable"] == answerable)
            assert n == 2
    _, test2 = split_dev_test(_items(), 1 / 3, seed=42)
    assert [x["id"] for x in test] == [x["id"] for x in test2]


def test_evidence_ignores_markdown_formatting():
    text = "Select **Run pipeline** or see [the docs](https://x.y/z) for `rules`."
    assert evidence_in_text("Select Run pipeline or see the docs for rules.", text)
