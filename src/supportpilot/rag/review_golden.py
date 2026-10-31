from __future__ import annotations

import argparse
import json
import random
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

from supportpilot.rag.gold import evidence_in_text
from supportpilot.rag.retrieval import BM25Retriever

FIELDS = (
    "id",
    "area",
    "answerable",
    "question",
    "reference_answer",
    "doc_id",
    "evidence",
    "title",
    "heading_path",
    "url",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def ask(msg: str, valid: str) -> str:
    while True:
        a = input(msg).strip().lower()
        if a and a[0] in valid:
            return a[0]


def edit(item: dict[str, Any], doc_text: str) -> dict[str, Any]:
    new = dict(item)
    q = input("question (Enter keeps): ").strip()
    if q:
        new["question"] = q
    if item["answerable"]:
        r = input("reference answer (Enter keeps): ").strip()
        if r:
            new["reference_answer"] = r
        while True:
            ev = input("evidence, verbatim from the doc (Enter keeps): ").strip()
            if not ev:
                break
            if evidence_in_text(ev, doc_text):
                new["evidence"] = ev
                break
            print("  not found in the doc, try again")
    return new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/golden.toml")
    args = parser.parse_args()
    cfg = tomllib.loads(Path(args.config).read_text())

    docs = {d["doc_id"]: d["text"] for d in read_jsonl(Path(cfg["documents"]))}
    drafts = read_jsonl(Path(cfg["draft_path"]))
    golden_path, rejected_path = Path(cfg["golden_path"]), Path(cfg["rejected_path"])
    golden = read_jsonl(golden_path)
    rejected = set(rejected_path.read_text().split()) if rejected_path.exists() else set()
    done = {g["id"] for g in golden} | rejected

    pending = sorted((d for d in drafts if d["id"] not in done), key=lambda d: d["id"])
    random.Random(cfg["seed"]).shuffle(pending)

    chunks_path = Path(cfg["review_chunks"])
    bm25 = BM25Retriever(read_jsonl(chunks_path)) if chunks_path.exists() else None
    targets = cfg["targets"]
    counts = Counter(bool(g["answerable"]) for g in golden)

    try:
        for n, item in enumerate(pending, start=1):
            print("\n" + "=" * 78)
            print(
                f"[{n}/{len(pending)}] accepted: answerable {counts[True]}/{targets['answerable']}"
                f"  unanswerable {counts[False]}/{targets['unanswerable']}"
            )
            print(f"area: {item['area']}   id: {item['id']}")
            print(f"\nQUESTION: {item['question']}")
            if item["answerable"]:
                print(f"ANSWER:   {item['reference_answer']}")
                print(f"EVIDENCE: {item['evidence']}")
                print(f"SOURCE:   {item['title']} | {item['heading_path']}")
                print(f"          {item['url']}   (copied-wording run: {item.get('copy_run', 0)})")
            else:
                print("\nUNANSWERABLE candidate. Closest chunks in the docs (BM25):")
                for h in bm25.search(item["question"], 3) if bm25 else []:
                    snippet = " ".join(h.text.split())[:160]
                    print(f"  {h.score:5.1f}  {h.metadata['title']} | {h.metadata['heading_path']}")
                    print(f"         {snippet}")
                print("  Accept only if the docs really do not answer it.")

            action = ask("\n[a]ccept [e]dit [r]eject [s]kip [q]uit > ", "aersq")
            if action == "q":
                break
            if action == "s":
                continue
            if action == "r":
                with rejected_path.open("a") as f:
                    f.write(item["id"] + "\n")
                continue
            if action == "e":
                item = edit(item, docs.get(item["doc_id"], ""))
            rec = {k: item.get(k, "") for k in FIELDS}
            rec["answerable"] = bool(item["answerable"])
            with golden_path.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts[rec["answerable"]] += 1
    except (EOFError, KeyboardInterrupt):
        print()

    print(f"\naccepted so far: answerable {counts[True]}, unanswerable {counts[False]}")


if __name__ == "__main__":
    main()
