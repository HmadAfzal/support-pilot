from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from supportpilot.rag.retrieval import Hit, Retriever

SENTINEL = "I don't know based on the provided documentation."

SYSTEM_PROMPT = (
    "You are a support assistant for GitLab. Answer the question using ONLY the numbered "
    "context passages. Cite every claim with the passage number in plain square brackets, "
    "like [1] or [2][3], placed at the end of the sentence it supports. Use only this exact "
    "format; never use any other citation style. Do not use outside knowledge. If the "
    "passages do not contain enough "
    f"information to answer, reply with exactly this sentence and nothing else: {SENTINEL} "
    "Be concise."
)

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_FOREIGN_CITATION = re.compile(r"【\s*(\d+)[^】]*】")


class ChatLLM(Protocol):
    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> Any: ...


@dataclass
class Answer:
    question: str
    text: str
    refused: bool
    cited_chunk_ids: list[str]
    invalid_citations: list[int]
    malformed_citations: int = 0
    hits: list[Hit] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False


def build_messages(question: str, hits: list[Hit]) -> list[dict[str, str]]:
    blocks = []
    for i, h in enumerate(hits, start=1):
        title = h.metadata.get("title", "")
        path = h.metadata.get("heading_path", "")
        header = f"[{i}] {title} | {path}" if path else f"[{i}] {title}"
        blocks.append(f"{header}\n{h.text}")
    user = "Context:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_citations(text: str, n_passages: int) -> tuple[list[int], list[int]]:
    valid: list[int] = []
    invalid: list[int] = []

    found: list[tuple[int, int]] = []
    for match in _CITATION.finditer(text):
        found.extend((match.start(), int(p.strip())) for p in match.group(1).split(","))
    for match in _FOREIGN_CITATION.finditer(text):
        found.append((match.start(), int(match.group(1))))
    for _, idx in sorted(found):
        target = valid if 1 <= idx <= n_passages else invalid
        if idx not in target:
            target.append(idx)
    return valid, invalid


def is_refusal(text: str) -> bool:
    normalized = text.strip().strip("\"'").replace("\u2019", "'")
    return normalized.startswith(SENTINEL)


def answer_question(
    question: str,
    retriever: Retriever,
    llm: ChatLLM,
    *,
    k: int = 5,
    temperature: float = 0.0,
    max_tokens: int = 1500,
) -> Answer:
    hits = retriever.search(question, k)
    if not hits:
        return Answer(question, SENTINEL, True, [], [], hits=[])
    resp = llm.chat(build_messages(question, hits), temperature=temperature, max_tokens=max_tokens)
    valid, invalid = parse_citations(resp.text, len(hits))
    malformed = len(_FOREIGN_CITATION.findall(resp.text))
    return Answer(
        question=question,
        text=resp.text.strip(),
        refused=is_refusal(resp.text),
        cited_chunk_ids=[hits[i - 1].chunk_id for i in valid],
        invalid_citations=invalid,
        malformed_citations=malformed,
        hits=hits,
        provider=resp.provider,
        model=resp.model,
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        latency_s=resp.latency_s,
        cached=resp.cached,
    )
