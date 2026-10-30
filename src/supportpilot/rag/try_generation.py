from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import chromadb

from supportpilot.llm.client import LLM
from supportpilot.rag.embedder import BGEEmbedder
from supportpilot.rag.generation import answer_question
from supportpilot.rag.retrieval import CrossEncoderReranker, build_retrievers

SAMPLES = [
    "How do I create a CI/CD pipeline?",
    "What permissions does a Reporter have?",
    "How do I cancel my subscription?",
    "What is the weather in Paris today?",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generation.toml")
    parser.add_argument("--strategy", default="header")
    parser.add_argument("--variant", default="hybrid")
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()

    gen = tomllib.loads(Path(args.config).read_text())
    ret_cfg = tomllib.loads(Path(gen["retrieval_config"]).read_text())
    idx = tomllib.loads(Path(ret_cfg["index_config"]).read_text())
    chunk_cfg = tomllib.loads(Path(idx["chunking_config"]).read_text())

    chunks_path = Path(chunk_cfg["output_dir"]) / f"{args.strategy}.jsonl"
    chunks = [json.loads(x) for x in chunks_path.read_text().splitlines() if x.strip()]
    e, r = idx["embedding"], ret_cfg["rerank"]
    retrievers = build_retrievers(
        chunks=chunks,
        collection=chromadb.PersistentClient(path=idx["chroma_dir"]).get_collection(args.strategy),
        embedder=BGEEmbedder(e["model"], e["query_prefix"], e["batch_size"], e["device"]),
        reranker=CrossEncoderReranker(r["model"], r["batch_size"], r["device"]),
        hybrid_candidates=ret_cfg["hybrid"]["candidates"],
        rrf_k=ret_cfg["hybrid"]["rrf_k"],
        rerank_candidates=r["candidates"],
    )
    retriever = retrievers[args.variant]
    llm = LLM()
    g = gen["generation"]
    k = args.k or g["k"]

    for q in SAMPLES:
        a = answer_question(
            q, retriever, llm, k=k, temperature=g["temperature"], max_tokens=g["max_tokens"]
        )
        print(f"\n=== {q}")
        print(a.text)
        titles = {h.chunk_id: h.metadata.get("title", "") for h in a.hits}
        print(
            f"\nrefused={a.refused} invalid_citations={a.invalid_citations} "
            f"tokens={a.prompt_tokens}+{a.completion_tokens} {a.latency_s:.1f}s cached={a.cached}"
        )
        for cid in a.cited_chunk_ids:
            print(f"  cited: {titles[cid]} ({cid})")


if __name__ == "__main__":
    main()
