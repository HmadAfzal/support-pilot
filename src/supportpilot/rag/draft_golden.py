from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import time
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from supportpilot.llm.client import LLM, AllProvidersFailed
from supportpilot.rag.chunkers import split_sections
from supportpilot.rag.gold import (
    allocate,
    candidate_sections,
    evidence_in_text,
    longest_common_run,
    normalize,
    parse_json_object,
    select_sections,
)

ANSWERABLE_SYSTEM = (
    "You write evaluation questions for a GitLab support-assistant benchmark. Given one "
    "documentation section, write ONE question a real GitLab user might type into a support "
    "chat that this section answers. Rules: phrase it like a user (short, informal, may "
    "describe a goal or problem); do NOT copy phrases of three or more words from the "
    "section; do not mention 'the documentation', 'the section' or 'the passage'; it must "
    "make sense on its own and be answerable from this section alone. Also write a "
    "reference_answer (1-3 sentences, only facts from the section) and an evidence field: one "
    "excerpt copied character for character from the section (one or two sentences, 40-250 "
    "characters) that directly supports the answer. "
    'Reply with JSON only: {"question": "...", "reference_answer": "...", "evidence": "..."}'
)

UNANSWERABLE_SYSTEM = (
    "You write evaluation questions for a GitLab support-assistant benchmark. The assistant "
    "only has GitLab documentation on the topics listed. Write the requested number of "
    "questions a real user in this area might plausibly ask but that this documentation would "
    "NOT answer. Good kinds: features that do not exist, details about the user's own account "
    "or data, GitLab company internals, unreleased plans, comparisons with other products, "
    "exact numbers or limits unlikely to be documented. Do not ask about anything the topic "
    "list suggests is covered. Phrase them like a user typing into support chat. "
    'Reply with JSON only: {"questions": ["...", "..."]}'
)

_META = re.compile(
    r"\b(this|the) (section|documentation|docs|passage|text)\b|according to", re.IGNORECASE
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def topic_lines(docs: list[dict[str, Any]], rng: random.Random, n: int = 40) -> list[str]:
    topics = sorted({d["title"] for d in docs if d.get("title")})
    if len(topics) < 10:
        for d in docs:
            topics.extend(p for p, _ in split_sections(d["text"], d.get("title", ""), 2))
        topics = sorted(set(topics))
    rng.shuffle(topics)
    return topics[:n]


def draft_answerable(
    llm: LLM, sec: dict[str, Any], llm_cfg: dict[str, Any], max_copy_run: int
) -> tuple[dict[str, Any] | None, str, bool]:
    user = (
        f"Area: {sec['area']}\nPage: {sec['title']}\nSection: {sec['heading_path']}\n\n"
        f'Section text:\n"""\n{sec["text"]}\n"""'
    )
    resp = llm.chat(
        [{"role": "system", "content": ANSWERABLE_SYSTEM}, {"role": "user", "content": user}],
        temperature=llm_cfg["temperature"],
        max_tokens=llm_cfg["max_tokens"],
    )
    obj = parse_json_object(resp.text)
    if obj is None:
        return None, "bad_json", resp.cached
    q, ref, ev = (str(obj.get(k, "")).strip() for k in ("question", "reference_answer", "evidence"))
    if not (q and ref and ev):
        return None, "missing_fields", resp.cached
    if _META.search(q):
        return None, "meta_wording", resp.cached
    if not evidence_in_text(ev, sec["text"]):
        return None, "evidence_not_found", resp.cached
    if not 30 <= len(normalize(ev)) <= 300:
        return None, "evidence_length", resp.cached
    run = longest_common_run(q, sec["text"])
    if run >= max_copy_run:
        return None, "copied_wording", resp.cached
    rec = {
        "area": sec["area"],
        "answerable": True,
        "question": q,
        "reference_answer": ref,
        "doc_id": sec["doc_id"],
        "evidence": ev,
        "title": sec["title"],
        "heading_path": sec["heading_path"],
        "url": sec["url"],
        "source_key": f"{sec['doc_id']}|{sec['heading_path']}",
        "copy_run": run,
    }
    return rec, "ok", resp.cached


def draft_unanswerable(
    llm: LLM, area: str, topics: list[str], n: int, llm_cfg: dict[str, Any]
) -> tuple[list[str], str, bool]:
    user = (
        f"Area: {area.replace('_', ' ')}\nTopics covered by the documentation:\n"
        + "\n".join(f"- {t}" for t in topics)
        + f"\n\nWrite {n} questions."
    )
    resp = llm.chat(
        [{"role": "system", "content": UNANSWERABLE_SYSTEM}, {"role": "user", "content": user}],
        temperature=llm_cfg["temperature"],
        max_tokens=llm_cfg["max_tokens"],
    )
    obj = parse_json_object(resp.text)
    qs = obj.get("questions") if obj else None
    if not isinstance(qs, list):
        return [], "bad_json", resp.cached
    return [str(q).strip() for q in qs if str(q).strip()], "ok", resp.cached


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/golden.toml")
    args = parser.parse_args()
    cfg = tomllib.loads(Path(args.config).read_text())
    s, llm_cfg, seed = cfg["sampling"], cfg["llm"], cfg["seed"]

    docs = read_jsonl(Path(cfg["documents"]))
    by_area: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for d in docs:
        by_area[d["area"]].append(d)
    weights = {a: float(len(v)) for a, v in by_area.items()}
    ans_quota = allocate(s["answerable_drafts"], weights, s["min_per_area"])
    un_quota = allocate(s["unanswerable_drafts"], weights, s["min_unanswerable_per_area"])
    print("answerable draft quotas:", ans_quota)
    print("unanswerable draft quotas:", un_quota)

    draft_path = Path(cfg["draft_path"])
    draft_path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(draft_path)
    seen = {r["source_key"] for r in existing}
    ids = itertools.count(max((int(r["id"][1:]) for r in existing), default=0) + 1)

    answerable_tasks = []
    for area in sorted(by_area):
        cands = [
            c
            for d in by_area[area]
            for c in candidate_sections(d, s["min_chars"], s["max_chars"], s["skip_headings"])
        ]
        rng = random.Random(f"{seed}:{area}")
        answerable_tasks += select_sections(cands, ans_quota[area], s["max_per_doc"], rng)

    llm = LLM()
    dropped: Counter[str] = Counter()
    written = Counter()

    def emit(f: Any, rec: dict[str, Any]) -> None:
        rec = {"id": f"g{next(ids):04d}", **rec}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        written["answerable" if rec["answerable"] else "unanswerable"] += 1

    with draft_path.open("a") as f:
        for i, sec in enumerate(answerable_tasks, start=1):
            if f"{sec['doc_id']}|{sec['heading_path']}" in seen:
                continue
            try:
                rec, reason, cached = draft_answerable(llm, sec, llm_cfg, s["max_copy_run"])
            except AllProvidersFailed as exc:
                dropped["llm_failed"] += 1
                print(f"  LLM failed ({str(exc)[:120]}); rerun later to fill gaps")
                time.sleep(5)
                continue
            if rec is None:
                dropped[reason] += 1
            else:
                emit(f, rec)
            if i % 10 == 0:
                print(f"  answerable {i}/{len(answerable_tasks)} kept={written['answerable']}")
            if not cached:
                time.sleep(llm_cfg["sleep_s"])

        for area in sorted(by_area):
            calls = -(-un_quota[area] // 3)
            for j in range(calls):
                key = f"un|{area}|{j}"
                if key in seen:
                    continue
                n = min(3, un_quota[area] - 3 * j)
                topics = topic_lines(by_area[area], random.Random(f"{seed}:{area}:{j}"))
                try:
                    qs, reason, cached = draft_unanswerable(llm, area, topics, n, llm_cfg)
                except AllProvidersFailed:
                    dropped["llm_failed"] += 1
                    time.sleep(5)
                    continue
                if not qs:
                    dropped[reason] += 1
                for q in qs[:n]:
                    emit(
                        f,
                        {
                            "area": area,
                            "answerable": False,
                            "question": q,
                            "reference_answer": "",
                            "doc_id": "",
                            "evidence": "",
                            "title": "",
                            "heading_path": "",
                            "url": "",
                            "source_key": key,
                            "copy_run": 0,
                        },
                    )
                if not cached:
                    time.sleep(llm_cfg["sleep_s"])

    total = read_jsonl(draft_path)
    print(f"\nnew this run: {dict(written)}   dropped: {dict(dropped)}")
    print(f"draft file now has {len(total)} items: {Counter(r['area'] for r in total)}")


if __name__ == "__main__":
    main()
