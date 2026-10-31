from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any

import chromadb

from supportpilot.llm.client import LLM, AllProvidersFailed
from supportpilot.rag.embedder import BGEEmbedder
from supportpilot.rag.generation import answer_question
from supportpilot.rag.generation_eval import (
    Budget,
    common_ids,
    pending_pairs,
    render_table,
    summarize,
)
from supportpilot.rag.gold import is_gold_hit
from supportpilot.rag.judge import applicable_metrics, score_metric
from supportpilot.rag.retrieval import CrossEncoderReranker, build_retrievers


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def build_all(strategies: set[str], ret_cfg: dict, idx: dict, chunk_cfg: dict) -> dict:
    e, r = idx["embedding"], ret_cfg["rerank"]
    embedder = BGEEmbedder(e["model"], e["query_prefix"], e["batch_size"], e["device"])
    reranker = CrossEncoderReranker(r["model"], r["batch_size"], r["device"])
    client = chromadb.PersistentClient(path=idx["chroma_dir"])
    built = {}
    for s in sorted(strategies):
        chunks = read_jsonl(Path(chunk_cfg["output_dir"]) / f"{s}.jsonl")
        built[s] = build_retrievers(
            chunks=chunks,
            collection=client.get_collection(s),
            embedder=embedder,
            reranker=reranker,
            hybrid_candidates=ret_cfg["hybrid"]["candidates"],
            rrf_k=ret_cfg["hybrid"]["rrf_k"],
            rerank_candidates=r["candidates"],
        )
    return built


def run_gen(items, names, retrievers, llm, g, budget, path) -> None:
    done = {(r["config"], r["id"]) for r in read_jsonl(path)}
    with path.open("a") as f:
        for it in items:
            for name in names:
                if (name, it["id"]) in done:
                    continue
                if budget.exhausted:
                    print(f"gen: token cap reached ({budget.used}). Rerun later to continue.")
                    return
                try:
                    ans = answer_question(
                        it["question"],
                        retrievers[name],
                        llm,
                        k=g["k"],
                        temperature=g["temperature"],
                        max_tokens=g["max_tokens"],
                    )
                except AllProvidersFailed as exc:
                    print(f"gen: LLM failed ({str(exc)[:200]}). Rerun later to continue.")
                    return
                gold = None
                if it["answerable"]:
                    gold = any(
                        is_gold_hit(
                            h.metadata.get("doc_id", ""), h.text, it["doc_id"], it["evidence"]
                        )
                        for h in ans.hits
                    )
                rec = {
                    "config": name,
                    "id": it["id"],
                    "answerable": it["answerable"],
                    "question": it["question"],
                    "text": ans.text,
                    "refused": ans.refused,
                    "cited_chunk_ids": ans.cited_chunk_ids,
                    "invalid_citations": ans.invalid_citations,
                    "malformed_citations": ans.malformed_citations,
                    "gold_in_context": gold,
                    "passages": [
                        {
                            "chunk_id": h.chunk_id,
                            "title": h.metadata.get("title", ""),
                            "heading_path": h.metadata.get("heading_path", ""),
                            "text": h.text,
                        }
                        for h in ans.hits
                    ],
                    "provider": ans.provider,
                    "model": ans.model,
                    "prompt_tokens": ans.prompt_tokens,
                    "completion_tokens": ans.completion_tokens,
                    "latency_s": ans.latency_s,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                budget.spend(ans)
    print(f"gen: done ({budget.used} tokens used this run)")


def run_judge(gens, items_by_id, metrics, llm, budget, max_tokens, path) -> None:
    done = {(r["config"], r["id"], r["metric"]) for r in read_jsonl(path)}
    with path.open("a") as f:
        for g in sorted(gens, key=lambda r: (r["id"], r["config"])):
            ref = items_by_id[g["id"]].get("reference_answer", "")
            for metric in applicable_metrics(g["answerable"], g["refused"], metrics):
                if (g["config"], g["id"], metric) in done:
                    continue
                if budget.exhausted:
                    print(f"judge: token cap reached ({budget.used}). Rerun later to continue.")
                    return
                try:
                    result, resp = score_metric(
                        llm,
                        metric,
                        question=g["question"],
                        reference=ref,
                        answer=g["text"],
                        passages=g["passages"],
                        max_tokens=max_tokens,
                    )
                except AllProvidersFailed as exc:
                    print(f"judge: LLM failed ({str(exc)[:200]}). Rerun later to continue.")
                    return
                rec = {"config": g["config"], "id": g["id"], "metric": metric, **result}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                budget.spend(resp)
    print(f"judge: done ({budget.used} tokens used this run)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generation_eval.toml")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--only", help="strategy:variant, required for the test split")
    parser.add_argument("--stage", choices=["all", "gen", "judge", "report"], default="all")
    parser.add_argument("--limit", type=int, help="only the first N questions (dev probe)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing test report")
    args = parser.parse_args()

    cfg = tomllib.loads(Path(args.config).read_text())
    gold_cfg = tomllib.loads(Path(cfg["golden_config"]).read_text())
    gen_cfg = tomllib.loads(Path(cfg["generation_config"]).read_text())
    ret_cfg = tomllib.loads(Path(gen_cfg["retrieval_config"]).read_text())
    idx = tomllib.loads(Path(ret_cfg["index_config"]).read_text())
    chunk_cfg = tomllib.loads(Path(idx["chunking_config"]).read_text())

    names = [args.only] if args.only else list(cfg["configs"])
    out_dir, report_dir = Path(cfg["out_dir"]), Path(cfg["report_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    gen_path = out_dir / f"generation_{args.split}_answers.jsonl"
    judge_path = out_dir / f"generation_{args.split}_judge.jsonl"
    report_json = report_dir / f"generation_{args.split}.json"

    if args.split == "test":
        if not args.only:
            raise SystemExit("Test split: pass --only strategy:variant (the one final config).")
        if args.limit:
            raise SystemExit("Test split: --limit is not allowed.")
        if report_json.exists() and not args.force:
            raise SystemExit(f"{report_json} exists. The test run happens once; use --force.")

    items = read_jsonl(Path(gold_cfg[f"{args.split}_path"]))
    if args.limit:
        items = items[: args.limit]
    items_by_id = {it["id"]: it for it in items}
    metrics = cfg["judge_metrics"]
    print(f"{args.split}: {len(items)} questions, configs: {names}")

    if args.stage in ("all", "gen", "judge"):
        llm = LLM()
        if llm.judge_model == llm.providers[0].model:
            raise SystemExit("Judge model must differ from the generator model.")

    if args.stage in ("all", "gen"):
        built = build_all({n.split(":")[0] for n in names}, ret_cfg, idx, chunk_cfg)
        retrievers = {n: built[n.split(":")[0]][n.split(":")[1]] for n in names}
        lim = cfg["gen_limits"]
        run_gen(
            items,
            names,
            retrievers,
            llm,
            gen_cfg["generation"],
            Budget(lim["tokens_per_min"], lim["tokens_per_run"]),
            gen_path,
        )

    if args.stage in ("all", "judge"):
        gens = [g for g in read_jsonl(gen_path) if g["id"] in items_by_id and g["config"] in names]
        lim = cfg["judge_limits"]
        run_judge(
            gens,
            items_by_id,
            metrics,
            llm,
            Budget(lim["tokens_per_min"], lim["tokens_per_run"]),
            cfg["judge"]["max_tokens"],
            judge_path,
        )

    if args.stage in ("all", "report"):
        gens = [g for g in read_jsonl(gen_path) if g["id"] in items_by_id and g["config"] in names]
        common = common_ids(gens, names)
        gens = [g for g in gens if g["id"] in common]
        judges = [j for j in read_jsonl(judge_path) if j["id"] in common and j["config"] in names]
        pend = pending_pairs(gens, judges, metrics)
        judges = [j for j in judges if (j["metric"], j["id"]) not in pend]
        rows = [
            summarize(
                n,
                [g for g in gens if g["config"] == n],
                [j for j in judges if j["config"] == n],
                cfg["bootstrap_samples"],
                cfg["seed"],
            )
            for n in names
        ]
        table = render_table(rows)
        note = f"Questions generated by all configs: {len(common)}/{len(items)}"
        print("\n" + table + "\n\n" + note)
        for r in rows:
            if r["judge_errors"]:
                print(f"judge errors {r['config']}: {r['judge_errors']}")
        report_json.write_text(json.dumps({"rows": rows, "coverage": note}, indent=2))
        (report_dir / f"generation_{args.split}.md").write_text(table + "\n\n" + note + "\n")


if __name__ == "__main__":
    main()
