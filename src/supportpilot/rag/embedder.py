from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_queries(self, texts: list[str]) -> list[list[float]]: ...


class BGEEmbedder:
    def __init__(
        self,
        name: str,
        query_prefix: str,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(name, device=device)
        self._model.max_seq_length = 512
        self._prefix = query_prefix
        self._batch_size = batch_size

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._encode([self._prefix + t for t in texts])
