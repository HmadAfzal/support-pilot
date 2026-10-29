from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from supportpilot.rag.build_index import META_KEYS
from supportpilot.rag.embedder import Embedder

_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[Hit]: ...


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class DenseRetriever:
    def __init__(self, collection: Any, embedder: Embedder) -> None:
        self._col = collection
        self._embedder = embedder

    def search(self, query: str, k: int) -> list[Hit]:
        qvec = self._embedder.embed_queries([query])
        res = self._col.query(query_embeddings=qvec, n_results=k)
        return [
            Hit(cid, 1.0 - dist, doc, meta)
            for cid, dist, doc, meta in zip(
                res["ids"][0], res["distances"][0], res["documents"][0], res["metadatas"][0]
            )
        ]


class BM25Retriever:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def search(self, query: str, k: int) -> list[Hit]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = np.asarray(self._bm25.get_scores(tokens))
        top = np.argsort(-scores)[: min(k, len(self._chunks))]
        hits = []
        for i in top:
            if scores[i] <= 0:
                break
            c = self._chunks[int(i)]
            hits.append(
                Hit(
                    c["chunk_id"], float(scores[i]), c["text"], {m: c.get(m, "") for m in META_KEYS}
                )
            )
        return hits


def rrf_fuse(rankings: list[list[Hit]], k: int = 60) -> list[Hit]:
    scores: dict[str, float] = {}
    first_seen: dict[str, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(hit.chunk_id, hit)
    ordered = sorted(scores, key=lambda cid: -scores[cid])
    return [replace(first_seen[cid], score=scores[cid]) for cid in ordered]


class HybridRetriever:
    def __init__(
        self, dense: Retriever, bm25: Retriever, candidates: int = 50, rrf_k: int = 60
    ) -> None:
        self._dense = dense
        self._bm25 = bm25
        self._candidates = candidates
        self._rrf_k = rrf_k

    def search(self, query: str, k: int) -> list[Hit]:
        n = max(self._candidates, k)
        fused = rrf_fuse([self._dense.search(query, n), self._bm25.search(query, n)], self._rrf_k)
        return fused[:k]


class RerankedRetriever:
    def __init__(self, base: Retriever, reranker: Reranker, candidates: int = 30) -> None:
        self._base = base
        self._reranker = reranker
        self._candidates = candidates

    def search(self, query: str, k: int) -> list[Hit]:
        cands = self._base.search(query, max(self._candidates, k))
        if not cands:
            return []
        scores = self._reranker.score(query, [h.text for h in cands])
        ranked = sorted(zip(cands, scores), key=lambda pair: -pair[1])
        return [replace(h, score=float(s)) for h, s in ranked[:k]]


class CrossEncoderReranker:
    def __init__(self, name: str, batch_size: int = 16, device: str = "cpu") -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(name, max_length=512, device=device)
        self._batch_size = batch_size

    def score(self, query: str, texts: list[str]) -> list[float]:
        pairs = [(query, t) for t in texts]
        out = self._model.predict(pairs, batch_size=self._batch_size, show_progress_bar=False)
        return [float(x) for x in out]


def build_retrievers(
    *,
    chunks: list[dict[str, Any]],
    collection: Any,
    embedder: Embedder,
    reranker: Reranker,
    hybrid_candidates: int = 50,
    rrf_k: int = 60,
    rerank_candidates: int = 30,
) -> dict[str, Retriever]:
    dense = DenseRetriever(collection, embedder)
    bm25 = BM25Retriever(chunks)
    hybrid = HybridRetriever(dense, bm25, hybrid_candidates, rrf_k)
    return {
        "dense": dense,
        "bm25": bm25,
        "hybrid": hybrid,
        "hybrid_rerank": RerankedRetriever(hybrid, reranker, rerank_candidates),
    }
