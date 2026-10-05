"""Settings read from environment variables, with documented defaults."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    embedding_model: str
    rag_top_k: int
    rag_min_score: float


def load_settings() -> Settings:
    return Settings(
        embedding_model=os.getenv("EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-base-es"),
        rag_top_k=int(os.getenv("RAG_TOP_K", "3")),
        # A noise floor chosen from observed scores, not a relevance guarantee:
        # see "Decisiones técnicas de RAG" in SUBMISSION.md.
        rag_min_score=float(os.getenv("RAG_MIN_SCORE", "0.20")),
    )
