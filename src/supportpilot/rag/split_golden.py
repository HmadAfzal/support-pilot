from __future__ import annotations

import argparse
import json
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

from supportpilot.rag.gold import evidence_in_text, normalize, split_dev_test


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda r: r["id"])
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/golden.toml")
    parser.add_argument("--force", action="store_true", help="overwrite an existing test split")
    args = parser.parse_args()
    cfg = tomllib.loads(Path(args.config).read_text())

    golden = read_jsonl(Path(cfg["golden_path"]))
    docs = {d["doc_id"]: d["text"] for d in read_jsonl(Path(cfg["documents"]))}

    problems = []
    for dup in [k for k, v in Counter(g["id"] for g in golden).items() if v > 1]:
        problems.append(f"duplicate id {dup}")
    qcount = Counter(normalize(g["question"]).lower() for g in golden)
    problems += [f"duplicate question: {q[:60]}" for q, v in qcount.items() if v > 1]
    for g in golden:
        if g["answerable"] and not evidence_in_text(g["evidence"], docs.get(g["doc_id"], "")):
            problems.append(f"{g['id']}: evidence not found in doc {g['doc_id']}")
    if problems:
        raise SystemExit("Fix these first:\n  " + "\n  ".join(problems))

    test_path = Path(cfg["test_path"])
    if test_path.exists() and not args.force:
        raise SystemExit(f"{test_path} exists. The test split is frozen; use --force to redo it.")

    dev, test = split_dev_test(golden, cfg["split"]["test_fraction"], cfg["seed"])
    write_jsonl(Path(cfg["dev_path"]), dev)
    write_jsonl(test_path, test)

    dc = Counter((g["area"], g["answerable"]) for g in dev)
    tc = Counter((g["area"], g["answerable"]) for g in test)
    print(f"{'area':<18}{'dev ans':>8}{'dev unans':>10}{'test ans':>9}{'test unans':>11}")
    for area in sorted({g["area"] for g in golden}):
        print(
            f"{area:<18}{dc[(area, True)]:>8}{dc[(area, False)]:>10}"
            f"{tc[(area, True)]:>9}{tc[(area, False)]:>11}"
        )
    print(f"\ntotal {len(golden)}: dev {len(dev)}, test {len(test)}")


if __name__ == "__main__":
    main()
