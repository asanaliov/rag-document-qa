"""Runtime configuration, resolved from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip() or default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    ollama_model: str = "llama3.2:3b"
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 4
    max_upload_mb: int = 25
    server_host: str = "127.0.0.1"
    server_port: int = 7860
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.retrieval_k < 1:
            raise ValueError("RETRIEVAL_K must be at least 1")
        if self.max_upload_mb < 1:
            raise ValueError("MAX_UPLOAD_MB must be at least 1")

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            ollama_model=_env_str("OLLAMA_MODEL", cls.ollama_model),
            ollama_base_url=_env_str("OLLAMA_BASE_URL", cls.ollama_base_url),
            embedding_model=_env_str("EMBEDDING_MODEL", cls.embedding_model),
            embedding_device=_env_str("EMBEDDING_DEVICE", cls.embedding_device),
            chunk_size=_env_int("CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_env_int("CHUNK_OVERLAP", cls.chunk_overlap),
            retrieval_k=_env_int("RETRIEVAL_K", cls.retrieval_k),
            max_upload_mb=_env_int("MAX_UPLOAD_MB", cls.max_upload_mb),
            server_host=_env_str("SERVER_HOST", cls.server_host),
            server_port=_env_int("SERVER_PORT", cls.server_port),
            log_level=_env_str("LOG_LEVEL", cls.log_level).upper(),
        )
