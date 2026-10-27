from __future__ import annotations

import argparse
import json
import statistics
import tomllib
from pathlib import Path

from supportpilot.rag.chunkers import chunk_document
from supportpilot.rag.tokenization import HFTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/chunking.toml")
    args = parser.parse_args()

    cfg = tomllib.loads(Path(args.config).read_text())
    docs = [
        json.loads(line) for line in Path(cfg["input"]).read_text().splitlines() if line.strip()
    ]
    no_headers = sum(1 for d in docs if "\n#" not in "\n" + d["text"])
    print(f"documents: {len(docs)} (without any markdown header: {no_headers})")

    tok = HFTokenizer(cfg["tokenizer"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, spec in cfg["strategies"].items():
        chunks = []
        for doc in docs:
            chunks.extend(chunk_document(doc, name, spec, tok))
        with (out_dir / f"{name}.jsonl").open("w") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        sizes = [c["n_tokens"] for c in chunks]
        empty_docs = len(docs) - len({c["doc_id"] for c in chunks})
        rows.append(
            (
                name,
                len(chunks),
                statistics.mean(sizes),
                statistics.median(sizes),
                max(sizes),
                empty_docs,
            )
        )

    print(f"\n{'strategy':<12}{'chunks':>8}{'mean':>8}{'median':>8}{'max':>6}{'empty_docs':>12}")
    for name, n, mean, med, mx, empty in rows:
        print(f"{name:<12}{n:>8}{mean:>8.0f}{med:>8.0f}{mx:>6}{empty:>12}")


if __name__ == "__main__":
    main()
