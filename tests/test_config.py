import pytest

from rag.config import Settings


def test_defaults_apply_when_env_is_empty(monkeypatch):
    for name in ("OLLAMA_MODEL", "CHUNK_SIZE", "RETRIEVAL_K"):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env()

    assert settings.ollama_model == "llama3.2:3b"
    assert settings.chunk_size == 1000
    assert settings.retrieval_k == 4


def test_env_overrides_defaults(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.1:8b")
    monkeypatch.setenv("RETRIEVAL_K", "8")

    settings = Settings.from_env()

    assert settings.ollama_model == "llama3.1:8b"
    assert settings.retrieval_k == 8


def test_blank_env_value_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "   ")

    assert Settings.from_env().ollama_model == "llama3.2:3b"


def test_non_numeric_env_value_is_rejected(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "big")

    with pytest.raises(ValueError, match="CHUNK_SIZE"):
        Settings.from_env()


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        Settings(chunk_size=500, chunk_overlap=500)


def test_retrieval_k_must_be_positive():
    with pytest.raises(ValueError, match="RETRIEVAL_K"):
        Settings(retrieval_k=0)
