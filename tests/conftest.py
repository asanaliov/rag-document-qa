"""Shared fixtures. Everything here runs offline: no model downloads, no Ollama."""
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

# Each sentence is on a distinct topic, so a similarity search has an
# unambiguous best match to assert on.
SAMPLE_TEXT = """The Zorblax protocol was invented in 1987 by Dr. Mira Chen.
It uses a 42-bit handshake to negotiate the session key.
The capital of the Zorblax Republic is Vantorra.
Vantorra has a population of 3.2 million people.
"""


@pytest.fixture
def sample_text_file(tmp_path):
    """A real .txt file on disk containing SAMPLE_TEXT."""
    path = tmp_path / "sample.txt"
    path.write_text(SAMPLE_TEXT, encoding="utf-8")
    return str(path)


@pytest.fixture
def fake_embeddings():
    """Embeddings with stable vectors, so tests never touch sentence-transformers."""
    return DeterministicFakeEmbedding(size=64)


@pytest.fixture
def fake_llm():
    return FakeListChatModel(responses=["Dr. Mira Chen invented it in 1987."])
