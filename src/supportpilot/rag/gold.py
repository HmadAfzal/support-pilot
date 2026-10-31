from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from typing import Any

from supportpilot.rag.chunkers import split_sections

_WS = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9]+")
_CHARS = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2011": "-",
        "\u2013": "-",
        "\u2014": "-",
    }
)


def normalize(text: str) -> str:
    return _WS.sub(" ", text.translate(_CHARS)).strip()


_MD = re.compile(r"\[([^\]]*)\]\([^)]*\)|[`*]")


def strip_md(text: str) -> str:
    return _MD.sub(lambda m: m.group(1) or "", text)


def evidence_in_text(evidence: str, text: str) -> bool:
    ev = normalize(strip_md(evidence))
    return bool(ev) and ev in normalize(strip_md(text))


def is_gold_hit(hit_doc_id: str, hit_text: str, gold_doc_id: str, evidence: str) -> bool:
    return hit_doc_id == gold_doc_id and evidence_in_text(evidence, hit_text)


def longest_common_run(a: str, b: str) -> int:
    aw, bw = _WORD.findall(a.lower()), _WORD.findall(b.lower())
    for n in range(min(len(aw), len(bw)), 0, -1):
        grams = {tuple(bw[i : i + n]) for i in range(len(bw) - n + 1)}
        if any(tuple(aw[i : i + n]) in grams for i in range(len(aw) - n + 1)):
            return n
    return 0


def parse_json_object(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def allocate(total: int, weights: dict[str, float], floor: int = 0) -> dict[str, int]:
    keys = list(weights)
    remaining = total - floor * len(keys)
    if remaining < 0:
        raise ValueError("total is smaller than floor * number of keys")
    wsum = sum(weights.values())
    raw = {k: remaining * weights[k] / wsum for k in keys}
    out = {k: floor + int(raw[k]) for k in keys}
    left = total - sum(out.values())
    for k in sorted(keys, key=lambda k: -(raw[k] - int(raw[k])))[:left]:
        out[k] += 1
    return out


def candidate_sections(
    doc: dict[str, Any], min_chars: int, max_chars: int, skip_headings: list[str]
) -> list[dict[str, Any]]:
    skip = {s.lower() for s in skip_headings}
    out = []
    for path, body in split_sections(doc["text"], doc.get("title", ""), max_level=3):
        if path.split(" > ")[-1].strip().lower() in skip:
            continue
        if not min_chars <= len(body) <= max_chars or body.count("```") > 6:
            continue
        out.append(
            {
                "doc_id": doc["doc_id"],
                "area": doc.get("area", ""),
                "title": doc.get("title", ""),
                "heading_path": path,
                "url": doc.get("url", ""),
                "text": body,
            }
        )
    return out


def select_sections(
    cands: list[dict[str, Any]], quota: int, max_per_doc: int, rng: random.Random
) -> list[dict[str, Any]]:
    n_docs = max(len({c["doc_id"] for c in cands}), 1)
    cap = max(max_per_doc, math.ceil(quota / n_docs))
    pool = list(cands)
    rng.shuffle(pool)
    per_doc: dict[str, int] = defaultdict(int)
    out = []
    for c in pool:
        if len(out) >= quota:
            break
        if per_doc[c["doc_id"]] >= cap:
            continue
        per_doc[c["doc_id"]] += 1
        out.append(c)
    return out


def split_dev_test(
    items: list[dict[str, Any]], test_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    strata: dict[tuple[str, bool], list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        strata[(it["area"], bool(it["answerable"]))].append(it)
    dev: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda x: x["id"])
        random.Random(f"{seed}:{key}").shuffle(group)
        n_test = round(len(group) * test_fraction)
        test += group[:n_test]
        dev += group[n_test:]
    return dev, test
