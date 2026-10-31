from __future__ import annotations

import json
import re
from typing import Any, Protocol

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

FAITHFULNESS_SYSTEM = (
    "You check whether an answer is supported by context passages. Break the answer into its "
    "individual factual claims. Count every step, setting name, command and code example as a "
    "claim. Ignore citation markers of any style (like [1]) and filler. For each claim decide "
    "whether the passages support it. A claim is supported only if a passage states it or "
    "directly restates it. Added explanations, requirements, roles, permissions, locations, "
    "reasons and examples that no passage states are unsupported, even if true or plausible. "
    "Ignore generic interface steps with no specific content, such as 'Save the changes' or "
    "'Select OK'. Use no outside knowledge. "
    'Reply with JSON only: {"claims": [{"claim": "...", "supported": true}]}'
)

RELEVANCY_SYSTEM = (
    "Rate how directly the answer addresses the question, ignoring whether it is correct. "
    "1 = off-topic or refuses to answer, 2 = barely related, 3 = partly addresses it, "
    "4 = mostly addresses it, 5 = fully and directly addresses it. "
    'Reply with JSON only: {"score": 1, "reason": "one short sentence"}'
)

PRECISION_SYSTEM = (
    "For each numbered passage, decide whether it contains information that helps answer the "
    "question. Use the reference answer as a guide to what a good answer contains. "
    'Reply with JSON only: {"relevant": [true, false]} with exactly one entry per passage, '
    "in passage order."
)


class JudgeLLM(Protocol):
    def judge(self, messages: list[dict[str, str]], **kwargs: Any) -> Any: ...


def strip_thinking(text: str) -> str:
    text = _THINK.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def parse_json(text: str) -> dict[str, Any] | None:
    text = strip_thinking(text)
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[m.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def format_passages(passages: list[dict[str, Any]]) -> str:
    blocks = []
    for i, p in enumerate(passages, start=1):
        path = p.get("heading_path", "")
        head = f"[{i}] {p.get('title', '')} | {path}" if path else f"[{i}] {p.get('title', '')}"
        blocks.append(f"{head}\n{p['text']}")
    return "\n\n".join(blocks)


def average_precision(flags: list[bool]) -> float:
    hits, total = 0, 0.0
    for i, relevant in enumerate(flags, start=1):
        if relevant:
            hits += 1
            total += hits / i
    return total / hits if hits else 0.0


def applicable_metrics(answerable: bool, refused: bool, metrics: list[str]) -> list[str]:
    out = []
    if "faithfulness" in metrics and not refused:
        out.append("faithfulness")
    if "answer_relevancy" in metrics and answerable and not refused:
        out.append("answer_relevancy")
    if "context_precision" in metrics and answerable:
        out.append("context_precision")
    return out


def _ask(llm: JudgeLLM, system: str, user: str, max_tokens: int) -> tuple[dict | None, Any]:
    resp = llm.judge(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return parse_json(resp.text), resp


def score_faithfulness(
    llm: JudgeLLM, answer: str, passages: list[dict[str, Any]], max_tokens: int
) -> tuple[dict[str, Any], Any]:
    user = f"Passages:\n\n{format_passages(passages)}\n\nAnswer:\n{answer}"
    obj, resp = _ask(llm, FAITHFULNESS_SYSTEM, user, max_tokens)
    claims = obj.get("claims") if obj else None
    if not isinstance(claims, list):
        return {"score": None, "error": "bad_json"}, resp
    if not claims:
        return {"score": None, "error": "no_claims"}, resp
    unsupported = [
        str(c.get("claim", "")) if isinstance(c, dict) else str(c)
        for c in claims
        if not (isinstance(c, dict) and c.get("supported") is True)
    ]
    n = len(claims)
    result = {"score": (n - len(unsupported)) / n, "n_claims": n, "unsupported": unsupported}
    return {**result, "error": None}, resp


def score_relevancy(
    llm: JudgeLLM, question: str, answer: str, max_tokens: int
) -> tuple[dict[str, Any], Any]:
    obj, resp = _ask(
        llm, RELEVANCY_SYSTEM, f"Question:\n{question}\n\nAnswer:\n{answer}", max_tokens
    )
    raw = obj.get("score") if obj else None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not 1 <= raw <= 5:
        return {"score": None, "error": "bad_json"}, resp
    return {"score": (raw - 1) / 4, "raw": raw, "error": None}, resp


def score_context_precision(
    llm: JudgeLLM, question: str, reference: str, passages: list[dict[str, Any]], max_tokens: int
) -> tuple[dict[str, Any], Any]:
    user = (
        f"Question:\n{question}\n\nReference answer:\n{reference}\n\n"
        f"Passages ({len(passages)}):\n\n{format_passages(passages)}"
    )
    obj, resp = _ask(llm, PRECISION_SYSTEM, user, max_tokens)
    flags = obj.get("relevant") if obj else None
    if (
        not isinstance(flags, list)
        or len(flags) != len(passages)
        or not all(isinstance(f, bool) for f in flags)
    ):
        return {"score": None, "error": "bad_flags"}, resp
    return {"score": average_precision(flags), "relevant": flags, "error": None}, resp


def score_metric(
    llm: JudgeLLM,
    metric: str,
    *,
    question: str,
    reference: str,
    answer: str,
    passages: list[dict[str, Any]],
    max_tokens: int,
) -> tuple[dict[str, Any], Any]:
    if metric == "faithfulness":
        return score_faithfulness(llm, answer, passages, max_tokens)
    if metric == "answer_relevancy":
        return score_relevancy(llm, question, answer, max_tokens)
    if metric == "context_precision":
        return score_context_precision(llm, question, reference, passages, max_tokens)
    raise ValueError(f"unknown metric: {metric!r}")
