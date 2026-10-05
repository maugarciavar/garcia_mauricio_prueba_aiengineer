import pytest

from tiendahogar.retrieval import Retriever, build_retriever


@pytest.fixture(scope="session")
def retriever() -> Retriever:
    """The real retriever. Loads the local embedding model once per test run
    (downloaded on first use); no API key or LLM call is involved."""
    return build_retriever()
