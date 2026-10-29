from __future__ import annotations

import argparse
import json
import time
import tomllib
from pathlib import Path

import chromadb

from supportpilot.rag.embedder import BGEEmbedder
from supportpilot.rag.retrieval import CrossEncoderReranker, build_retrievers

SAMPLES = [
    "How do I create a CI/CD pipeline?",
    "runner stuck in pending state",
    "what permissions does a Reporter have",
    "how to cancel my subscription",
    "CI_PIPELINE_SOURCE merge_request_event",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retrieval.toml")
    parser.add_argument("--strategy", default="header")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    cfg = tomllib.loads(Path(args.config).read_text())
    idx = tomllib.loads(Path(cfg["index_config"]).read_text())
    chunk_cfg = tomllib.loads(Path(idx["chunking_config"]).read_text())

    chunks_path = Path(chunk_cfg["output_dir"]) / f"{args.strategy}.jsonl"
    chunks = [json.loads(x) for x in chunks_path.read_text().splitlines() if x.strip()]

    e = idx["embedding"]
    embedder = BGEEmbedder(e["model"], e["query_prefix"], e["batch_size"], e["device"])
    r = cfg["rerank"]
    reranker = CrossEncoderReranker(r["model"], r["batch_size"], r["device"])
    client = chromadb.PersistentClient(path=idx["chroma_dir"])

    retrievers = build_retrievers(
        chunks=chunks,
        collection=client.get_collection(args.strategy),
        embedder=embedder,
        reranker=reranker,
        hybrid_candidates=cfg["hybrid"]["candidates"],
        rrf_k=cfg["hybrid"]["rrf_k"],
        rerank_candidates=r["candidates"],
    )

    for q in SAMPLES:
        print(f"\n=== {q}")
        for name, ret in retrievers.items():
            t0 = time.perf_counter()
            hits = ret.search(q, args.k)
            ms = (time.perf_counter() - t0) * 1000
            print(f"[{name}] {ms:.0f} ms")
            for h in hits:
                print(f"  {h.score:7.3f}  {h.metadata['title']} | {h.metadata['heading_path']}")


if __name__ == "__main__":
    main()
