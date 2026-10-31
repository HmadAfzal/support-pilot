from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from supportpilot.rag.gold import evidence_in_text


def index_chunks_by_doc(chunks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        by_doc[c["doc_id"]].append(c)
    return by_doc


def gold_chunk_ids(item: dict[str, Any], by_doc: dict[str, list[dict[str, Any]]]) -> set[str]:
    return {
        c["chunk_id"]
        for c in by_doc.get(item["doc_id"], [])
        if evidence_in_text(item["evidence"], c["text"])
    }


def score_question(ranked_ids: list[str], gold_ids: set[str], ks: list[int]) -> dict[str, Any]:
    rank = next((i for i, cid in enumerate(ranked_ids, start=1) if cid in gold_ids), None)
    out: dict[str, Any] = {"rank": rank, "rr": 1.0 / rank if rank else 0.0}
    for k in ks:
        found = set(ranked_ids[:k]) & gold_ids
        out[f"hit@{k}"] = float(bool(found))
        out[f"recall@{k}"] = len(found) / len(gold_ids) if gold_ids else 0.0
    return out


def aggregate(rows: list[dict[str, Any]], ks: list[int]) -> dict[str, float]:
    n = max(len(rows), 1)
    keys = [f"hit@{k}" for k in ks] + [f"recall@{k}" for k in ks]
    out = {k: sum(r[k] for r in rows) / n for k in keys}
    out["mrr"] = sum(r["rr"] for r in rows) / n
    return out


def bootstrap_ci(
    values: list[float], n: int = 1000, seed: int = 42, alpha: float = 0.05
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    m = len(values)
    means = sorted(sum(values[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return means[int(alpha / 2 * n)], means[min(n - 1, int((1 - alpha / 2) * n))]
