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

    def search(self, query: str, context: Sequence[str] = ()) -> list[RetrievedDocument]:
        """The top-k documents scoring at least min_score; may be empty.

        context: the customer's previous messages. A short follow-up such as
        "¿Y de una licuadora?" does not say what it is about, so the query is
        also searched together with that context and each document keeps its
        best score. The best match for the query on its own is always kept, so
        a change of topic is never crowded out by the earlier one.
        """
        if not query.strip():
            return []
        alone = self.score_all(query)
        best = {hit.document.doc_id: hit for hit in alone}
        if context:
            for hit in self.score_all(" ".join([*context, query])):
                if hit.score > best[hit.document.doc_id].score:
                    best[hit.document.doc_id] = hit

        ranked = sorted(best.values(), key=lambda hit: hit.score, reverse=True)[: self.top_k]
        top_alone = best[alone[0].document.doc_id]
        if top_alone not in ranked and top_alone.score >= self.min_score:
            ranked[-1] = top_alone
        return [hit for hit in ranked if hit.score >= self.min_score]


def build_retriever() -> Retriever:
    settings = load_settings()
    return Retriever(
        documents=load_documents(),
        model_name=settings.embedding_model,
        top_k=settings.rag_top_k,
        min_score=settings.rag_min_score,
    )
