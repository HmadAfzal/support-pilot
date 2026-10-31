from __future__ import annotations

import argparse
import json
import time
import tomllib
from pathlib import Path
from typing import Any

import chromadb

from supportpilot.rag.embedder import BGEEmbedder
from supportpilot.rag.retrieval import CrossEncoderReranker, build_retrievers
from supportpilot.rag.retrieval_eval import (
    aggregate,
    bootstrap_ci,
    gold_chunk_ids,
    index_chunks_by_doc,
    score_question,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def render_table(summary: list[dict[str, Any]], ks: list[int]) -> str:
    head = [
        "strategy",
        "variant",
        *[f"hit@{k}" for k in ks],
        "recall@5",
        "MRR",
        "MRR 95% CI",
        "ms/q",
    ]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for s in summary:
        m = s["metrics"]
        cells = [s["strategy"], s["variant"], *[f"{m[f'hit@{k}']:.3f}" for k in ks]]
        cells += [
            f"{m['recall@5']:.3f}",
            f"{m['mrr']:.3f}",
            f"{s['mrr_ci'][0]:.3f}-{s['mrr_ci'][1]:.3f}",
        ]
        cells.append(f"{s['ms_per_query']:.0f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval.toml")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--only", help="strategy:variant, required for the test split")
    parser.add_argument("--force", action="store_true", help="overwrite an existing test report")
    args = parser.parse_args()

    cfg = tomllib.loads(Path(args.config).read_text())
    gold_cfg = tomllib.loads(Path(cfg["golden_config"]).read_text())
    ret_cfg = tomllib.loads(Path(cfg["retrieval_config"]).read_text())
    idx = tomllib.loads(Path(ret_cfg["index_config"]).read_text())
    chunk_cfg = tomllib.loads(Path(idx["chunking_config"]).read_text())
    ks, max_k = cfg["ks"], max(cfg["ks"])

    report_dir, out_dir = Path(cfg["report_dir"]), Path(cfg["out_dir"])
    report_json = report_dir / f"retrieval_{args.split}.json"
    if args.split == "test":
        if not args.only:
            raise SystemExit("Test split: pass --only strategy:variant (the one final config).")
        if report_json.exists() and not args.force:
            raise SystemExit(
                f"{report_json} exists. The test run happens once; use --force to redo."
            )

    strategies = list(chunk_cfg["strategies"])
    variants = list(cfg["variants"])
    if args.only:
        strategy, variant = args.only.split(":")
        strategies, variants = [strategy], [variant]

    items = [x for x in read_jsonl(Path(gold_cfg[f"{args.split}_path"])) if x["answerable"]]
    print(
        f"{args.split}: {len(items)} answerable questions, {len(strategies)} strategies x {len(variants)} variants"
    )

    e, r = idx["embedding"], ret_cfg["rerank"]
    embedder = BGEEmbedder(e["model"], e["query_prefix"], e["batch_size"], e["device"])
    reranker = CrossEncoderReranker(r["model"], r["batch_size"], r["device"])
    client = chromadb.PersistentClient(path=idx["chroma_dir"])

    summary: list[dict[str, Any]] = []
    unreachable: dict[str, int] = {}
    perq: list[dict[str, Any]] = []
    for strategy in strategies:
        chunks = read_jsonl(Path(chunk_cfg["output_dir"]) / f"{strategy}.jsonl")
        by_doc = index_chunks_by_doc(chunks)
        golds = {it["id"]: gold_chunk_ids(it, by_doc) for it in items}
        unreachable[strategy] = sum(1 for g in golds.values() if not g)
        retrievers = build_retrievers(
            chunks=chunks,
            collection=client.get_collection(strategy),
            embedder=embedder,
            reranker=reranker,
            hybrid_candidates=ret_cfg["hybrid"]["candidates"],
            rrf_k=ret_cfg["hybrid"]["rrf_k"],
            rerank_candidates=r["candidates"],
        )
        for variant in variants:
            retriever = retrievers[variant]
            rows, times = [], []
            for it in items:
                t0 = time.perf_counter()
                hits = retriever.search(it["question"], max_k)
                times.append((time.perf_counter() - t0) * 1000)
                ids = [h.chunk_id for h in hits]
                sc = score_question(ids, golds[it["id"]], ks)
                rows.append(sc)
                perq.append(
                    {
                        "id": it["id"],
                        "strategy": strategy,
                        "variant": variant,
                        "question": it["question"],
                        "rank": sc["rank"],
                        "n_gold": len(golds[it["id"]]),
                        "top_ids": ids[:5],
                    }
                )
            metrics = aggregate(rows, ks)
            ci = bootstrap_ci([x["rr"] for x in rows], cfg["bootstrap_samples"], cfg["seed"])
            summary.append(
                {
                    "strategy": strategy,
                    "variant": variant,
                    "metrics": metrics,
                    "mrr_ci": ci,
                    "ms_per_query": sum(times) / len(times),
                    "n": len(rows),
                }
            )
            print(
                f"  {strategy:<10} {variant:<14} hit@5={metrics['hit@5']:.3f} mrr={metrics['mrr']:.3f}"
            )

    table = render_table(summary, ks)
    unreach = ", ".join(f"{s}: {n}/{len(items)}" for s, n in unreachable.items())
    print("\n" + table)
    print(f"\nUnreachable (no chunk contains the full evidence): {unreach}")

    report_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps({"summary": summary, "unreachable": unreachable}, indent=2))
    (report_dir / f"retrieval_{args.split}.md").write_text(
        table + f"\n\nUnreachable (no chunk contains the full evidence): {unreach}\n"
    )
    with (out_dir / f"retrieval_{args.split}_perq.jsonl").open("w") as f:
        for row in perq:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {report_json}, its .md twin, and {out_dir}/retrieval_{args.split}_perq.jsonl")


if __name__ == "__main__":
    main()
