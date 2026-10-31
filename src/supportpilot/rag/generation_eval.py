from __future__ import annotations

import time
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any

from supportpilot.rag.judge import applicable_metrics
from supportpilot.rag.retrieval_eval import bootstrap_ci


class Budget:
    def __init__(self, tokens_per_min: int, cap: int, sleep: Callable[[float], None] = time.sleep):
        self.tokens_per_min = tokens_per_min
        self.cap = cap
        self.used = 0
        self._sleep = sleep

    @property
    def exhausted(self) -> bool:
        return self.used >= self.cap

    def spend(self, resp: Any) -> None:
        if getattr(resp, "cached", False):
            return
        tokens = resp.prompt_tokens + resp.completion_tokens
        self.used += tokens
        if tokens:
            self._sleep(tokens * 60 / self.tokens_per_min)


def common_ids(gens: list[dict[str, Any]], configs: list[str]) -> set[str]:
    per = [{g["id"] for g in gens if g["config"] == c} for c in configs]
    return set.intersection(*per) if per else set()


def pending_pairs(
    gens: list[dict[str, Any]], judges: list[dict[str, Any]], metrics: list[str]
) -> set[tuple[str, str]]:
    done = {(j["config"], j["id"], j["metric"]) for j in judges}
    return {
        (m, g["id"])
        for g in gens
        for m in applicable_metrics(g["answerable"], g["refused"], metrics)
        if (g["config"], g["id"], m) not in done
    }


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def mean_ci(values: list[float], n_boot: int, seed: int) -> dict[str, Any] | None:
    if not values:
        return None
    lo, hi = bootstrap_ci(values, n_boot, seed)
    return {"mean": _avg(values), "lo": lo, "hi": hi, "n": len(values)}


def summarize(
    config: str,
    gens: list[dict[str, Any]],
    judges: list[dict[str, Any]],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    ans = [g for g in gens if g["answerable"]]
    unans = [g for g in gens if not g["answerable"]]
    answered = [g for g in gens if not g["refused"]]
    scores: dict[str, list[float]] = defaultdict(list)
    errors: Counter[str] = Counter()
    for j in judges:
        if j["score"] is None:
            errors[j["metric"]] += 1
        else:
            scores[j["metric"]].append(j["score"])
    return {
        "config": config,
        "n_answerable": len(ans),
        "n_unanswerable": len(unans),
        "correct_refusals": sum(1 for g in unans if g["refused"]),
        "false_refusals": sum(1 for g in ans if g["refused"]),
        "answered": len(answered),
        "with_valid_citation": sum(1 for g in answered if g["cited_chunk_ids"]),
        "invalid_citation_answers": sum(1 for g in answered if g["invalid_citations"]),
        "malformed_citation_answers": sum(1 for g in answered if g["malformed_citations"]),
        "gold_in_context": sum(1 for g in ans if g["gold_in_context"]),
        "avg_tokens": _avg([g["prompt_tokens"] + g["completion_tokens"] for g in gens]),
        "avg_latency_s": _avg([g["latency_s"] for g in gens]),
        "metrics": {m: mean_ci(v, n_boot, seed) for m, v in scores.items()},
        "judge_errors": dict(errors),
    }


def _fmt(m: dict[str, Any] | None) -> str:
    if not m:
        return "n/a"
    return f"{m['mean']:.3f} ({m['lo']:.2f}-{m['hi']:.2f}, n={m['n']})"


def render_table(rows: list[dict[str, Any]]) -> str:
    head = [
        "config",
        "gold in ctx",
        "correct refusal",
        "false refusal",
        "cited",
        "faithfulness",
        "answer relevancy",
        "context precision",
        "tokens/q",
        "s/q",
    ]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        m = r["metrics"]
        cells = [
            r["config"],
            f"{r['gold_in_context']}/{r['n_answerable']}",
            f"{r['correct_refusals']}/{r['n_unanswerable']}",
            f"{r['false_refusals']}/{r['n_answerable']}",
            f"{r['with_valid_citation']}/{r['answered']}",
            _fmt(m.get("faithfulness")),
            _fmt(m.get("answer_relevancy")),
            _fmt(m.get("context_precision")),
            f"{r['avg_tokens']:.0f}",
            f"{r['avg_latency_s']:.1f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
