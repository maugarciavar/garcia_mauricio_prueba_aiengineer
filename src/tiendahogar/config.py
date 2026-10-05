"""Settings read from environment variables, with documented defaults."""

import os
from dataclasses import dataclass

# The human support channel named in the contact policy (Doc 5).
SUPPORT_EMAIL = "soporte@tiendahogar.example"


@dataclass(frozen=True)
class Settings:
    openai_model: str
    max_tool_rounds: int
    embedding_model: str
    rag_top_k: int
    rag_min_score: float


def load_settings() -> Settings:
    return Settings(
        # OPENAI_API_KEY is read from the environment by the OpenAI SDK itself.
        openai_model=os.getenv("OPENAI_MODEL", "gpt-6-luna"),
        max_tool_rounds=int(os.getenv("MAX_TOOL_ROUNDS", "3")),
        embedding_model=os.getenv("EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-base-es"),
        rag_top_k=int(os.getenv("RAG_TOP_K", "3")),
        # A noise floor chosen from observed scores, not a relevance guarantee:
        # see "Decisiones técnicas de RAG" in SUBMISSION.md.
        rag_min_score=float(os.getenv("RAG_MIN_SCORE", "0.20")),
    )
