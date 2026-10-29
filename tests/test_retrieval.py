import hashlib

import chromadb

from supportpilot.rag.build_index import index_strategy
from supportpilot.rag.retrieval import (
    BM25Retriever,
    DenseRetriever,
    Hit,
    HybridRetriever,
    RerankedRetriever,
    rrf_fuse,
)

NAME = "strat"

TEXTS = [
    "runner executor docker configuration register token",
    "merge request approval rules and code owners",
    "billing subscription seats and renewal dates",
    "pipeline stages jobs artifacts cache keywords",
    "project permissions roles guest reporter developer",
]


def make_chunks() -> list[dict]:
    return [
        {
            "chunk_id": f"d{i}::{NAME}::0000",
            "doc_id": f"d{i}",
            "strategy": NAME,
            "index": 0,
            "text": t,
            "n_tokens": len(t.split()),
            "heading_path": "H",
            "title": f"Doc {i}",
            "area": "ci_cd",
            "url": "u",
            "path": "p",
            "commit": "c",
        }
        for i, t in enumerate(TEXTS)
    ]


class FakeEmbedder:
    def _vec(self, text: str) -> list[float]:
        return [b / 255 for b in hashlib.sha256(text.encode()).digest()[:8]]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_queries(self, texts):
        return [self._vec(t) for t in texts]


class WordOverlapReranker:
    def score(self, query, texts):
        q = set(query.split())
        return [float(len(q & set(t.split()))) for t in texts]


class StubRetriever:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, k):
        return self._hits[:k]


def hit(cid: str, text: str = "") -> Hit:
    return Hit(cid, 0.0, text, {})


def test_rrf_prefers_docs_ranked_high_in_both():
    a = [hit("x"), hit("y"), hit("z")]
    b = [hit("y"), hit("w"), hit("x")]
    fused = rrf_fuse([a, b], k=60)
    assert [h.chunk_id for h in fused][:2] == ["y", "x"]
    assert {h.chunk_id for h in fused} == {"x", "y", "z", "w"}


def test_bm25_finds_keyword_match():
    ret = BM25Retriever(make_chunks())
    hits = ret.search("how to register a docker runner", 3)
    assert hits[0].chunk_id == f"d0::{NAME}::0000"
    assert hits[0].metadata["title"] == "Doc 0"


def test_bm25_no_match_returns_empty():
    assert BM25Retriever(make_chunks()).search("zzzz qqqq", 3) == []
    assert BM25Retriever(make_chunks()).search("!!!", 3) == []


def test_dense_exact_text_first(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    emb = FakeEmbedder()
    index_strategy(client, NAME, make_chunks(), emb)
    ret = DenseRetriever(client.get_collection(NAME), emb)
    hits = ret.search(TEXTS[2], 2)
    assert hits[0].chunk_id == f"d2::{NAME}::0000"
    assert len(hits) == 2


def test_hybrid_returns_k_unique_hits(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    emb = FakeEmbedder()
    chunks = make_chunks()
    index_strategy(client, NAME, chunks, emb)
    hybrid = HybridRetriever(
        DenseRetriever(client.get_collection(NAME), emb), BM25Retriever(chunks), candidates=5
    )
    hits = hybrid.search("billing subscription seats", 3)
    ids = [h.chunk_id for h in hits]
    assert len(ids) == 3 and len(set(ids)) == 3
    assert ids[0] == f"d2::{NAME}::0000"


def test_reranker_reorders_and_truncates():
    base = StubRetriever(
        [hit("a", "nothing here"), hit("b", "docker runner setup"), hit("c", "docker")]
    )
    ret = RerankedRetriever(base, WordOverlapReranker(), candidates=3)
    hits = ret.search("docker runner", 2)
    assert [h.chunk_id for h in hits] == ["b", "c"]
    assert hits[0].score == 2.0


def test_reranker_empty_base():
    assert RerankedRetriever(StubRetriever([]), WordOverlapReranker()).search("q", 3) == []
