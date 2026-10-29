import hashlib

import chromadb

from supportpilot.rag.build_index import index_strategy

NAME = "strat"


class FakeEmbedder:
    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255 for b in digest[:8]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


def make_chunks(n: int) -> list[dict]:
    return [
        {
            "chunk_id": f"d1::{NAME}::{i:04d}",
            "doc_id": "d1",
            "strategy": NAME,
            "index": i,
            "text": f"chunk text number {i}",
            "n_tokens": 4,
            "heading_path": "Doc > A",
            "title": "Doc",
            "area": "ci_cd",
            "url": "u",
            "path": "p",
            "commit": "c",
        }
        for i in range(n)
    ]


def test_index_counts_and_metadata(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    stored = index_strategy(client, NAME, make_chunks(7), FakeEmbedder(), add_batch_size=3)
    assert stored == 7
    got = client.get_collection(NAME).get(ids=[f"d1::{NAME}::0002"])
    assert got["documents"] == ["chunk text number 2"]
    assert got["metadatas"][0]["area"] == "ci_cd"
    assert got["metadatas"][0]["heading_path"] == "Doc > A"


def test_rebuild_is_idempotent(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    index_strategy(client, NAME, make_chunks(5), FakeEmbedder())
    assert index_strategy(client, NAME, make_chunks(5), FakeEmbedder()) == 5


def test_query_returns_exact_match_first(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    emb = FakeEmbedder()
    index_strategy(client, NAME, make_chunks(6), emb)
    res = client.get_collection(NAME).query(
        query_embeddings=emb.embed_queries(["chunk text number 4"]), n_results=1
    )
    assert res["ids"][0][0] == f"d1::{NAME}::0004"
