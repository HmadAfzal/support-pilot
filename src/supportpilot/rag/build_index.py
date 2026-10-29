from __future__ import annotations

import argparse
import json
import time
import tomllib
from pathlib import Path
from typing import Any

import chromadb

from supportpilot.rag.embedder import BGEEmbedder, Embedder

META_KEYS = (
    "doc_id",
    "strategy",
    "index",
    "n_tokens",
    "heading_path",
    "title",
    "area",
    "url",
    "path",
    "commit",
)


def index_strategy(
    client: Any,
    name: str,
    chunks: list[dict[str, Any]],
    embedder: Embedder,
    add_batch_size: int = 256,
) -> int:
    existing = {c if isinstance(c, str) else c.name for c in client.list_collections()}
    if name in existing:
        client.delete_collection(name)
    col = client.create_collection(name, metadata={"hnsw:space": "cosine"})
    for start in range(0, len(chunks), add_batch_size):
        batch = chunks[start : start + add_batch_size]
        col.add(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=embedder.embed_documents([c["text"] for c in batch]),
            metadatas=[{k: c.get(k, "") for k in META_KEYS} for c in batch],
        )
        print(
            f"\r  {name}: {min(start + add_batch_size, len(chunks))}/{len(chunks)}",
            end="",
            flush=True,
        )
    print()
    return col.count()


def load_chunks(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/index.toml")
    args = parser.parse_args()

    cfg = tomllib.loads(Path(args.config).read_text())
    chunk_cfg = tomllib.loads(Path(cfg["chunking_config"]).read_text())
    chunks_dir = Path(chunk_cfg["output_dir"])

    emb_cfg = cfg["embedding"]
    embedder = BGEEmbedder(
        emb_cfg["model"], emb_cfg["query_prefix"], emb_cfg["batch_size"], emb_cfg["device"]
    )
    Path(cfg["chroma_dir"]).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=cfg["chroma_dir"])

    rows = []
    for name in chunk_cfg["strategies"]:
        chunks = load_chunks(chunks_dir / f"{name}.jsonl")
        t0 = time.perf_counter()
        stored = index_strategy(client, name, chunks, embedder, cfg["build"]["add_batch_size"])
        rows.append((name, len(chunks), stored, time.perf_counter() - t0))

    print(f"\n{'strategy':<12}{'chunks':>8}{'stored':>8}{'seconds':>9}")
    for name, n, stored, secs in rows:
        flag = "" if n == stored else "  MISMATCH"
        print(f"{name:<12}{n:>8}{stored:>8}{secs:>9.0f}{flag}")

    query = "How do I create a CI/CD pipeline?"
    qvec = embedder.embed_queries([query])
    print(f"\nSanity query: {query!r}")
    for name, *_ in rows:
        res = client.get_collection(name).query(query_embeddings=qvec, n_results=3)
        print(f"[{name}]")
        for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
            print(f"  {dist:.3f}  {meta['title']} | {meta['heading_path']}")


if __name__ == "__main__":
    main()
