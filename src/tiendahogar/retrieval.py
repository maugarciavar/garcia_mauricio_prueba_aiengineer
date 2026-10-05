"""Retrieval over the policy documents: local embeddings and cosine similarity.

Each policy is embedded whole (title + text). They are short enough that
chunking would only split sentences that belong together, and five vectors
need no index: a NumPy matrix product is the whole search.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from fastembed import TextEmbedding

from tiendahogar.config import load_settings
from tiendahogar.documents import Document, load_documents


@dataclass(frozen=True)
class RetrievedDocument:
    document: Document
    score: float


class Retriever:
    def __init__(
        self,
        documents: Sequence[Document],
        model_name: str,
        top_k: int,
        min_score: float,
    ) -> None:
        self.top_k = top_k
        self.min_score = min_score
        self._documents = tuple(documents)
        self._model = TextEmbedding(model_name=model_name)
        self._matrix = self._embed([f"{doc.title}\n{doc.text}" for doc in self._documents])

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.array(list(self._model.embed(texts)))
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def score_all(self, query: str) -> list[RetrievedDocument]:
        """Every document with its cosine similarity to the query, best first."""
        scores = self._matrix @ self._embed([query])[0]
        ranked = sorted(zip(self._documents, scores), key=lambda pair: pair[1], reverse=True)
        return [RetrievedDocument(document=doc, score=float(score)) for doc, score in ranked]

    def search(self, query: str) -> list[RetrievedDocument]:
        """The top-k documents scoring at least min_score; may be empty."""
        if not query.strip():
            return []
        return [hit for hit in self.score_all(query)[: self.top_k] if hit.score >= self.min_score]


def build_retriever() -> Retriever:
    settings = load_settings()
    return Retriever(
        documents=load_documents(),
        model_name=settings.embedding_model,
        top_k=settings.rag_top_k,
        min_score=settings.rag_min_score,
    )
