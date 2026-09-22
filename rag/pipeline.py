"""Stateful orchestration of the RAG pipeline, independent of any UI."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from rag.config import Settings
from rag.errors import GenerationError
from rag.loader import chunk_documents, load_document
from rag.qa import ask, create_llm, create_qa_chain
from rag.store import build_vector_store, create_embeddings

logger = logging.getLogger(__name__)

# httpx errors are matched by name so this module does not depend on httpx directly.
_CONNECTION_ERROR_NAMES = {"ConnectError", "ConnectTimeout", "ReadTimeout"}


@dataclass(frozen=True)
class IndexResult:
    name: str
    documents: int
    chunks: int


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Document]


class DocumentQA:
    """Indexes one document at a time and answers questions about it.

    Holds a single document's index: uploading a new one replaces the previous
    state. The embedding model and LLM are created once and reused, because
    loading the embedding weights takes seconds.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        embeddings: Embeddings | None = None,
        llm: BaseChatModel | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self._embeddings = embeddings
        self._llm = llm
        self._chain = None
        self._retriever = None

    @property
    def is_ready(self) -> bool:
        return self._chain is not None

    def index(self, file_path: str) -> IndexResult:
        """Load, chunk, embed and index a document, replacing any previous one."""
        self.reset()

        documents = load_document(file_path, max_size_mb=self.settings.max_upload_mb)
        chunks = chunk_documents(
            documents,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

        vector_store = build_vector_store(chunks, self._get_embeddings())
        self._chain, self._retriever = create_qa_chain(
            vector_store, self._get_llm(), k=self.settings.retrieval_k
        )

        source = documents[0].metadata.get("source", file_path)
        return IndexResult(name=source, documents=len(documents), chunks=len(chunks))

    def ask(self, question: str) -> Answer:
        """Answer a question from the indexed document."""
        if not self.is_ready:
            raise RuntimeError("No document has been indexed yet")

        try:
            result = ask(self._chain, self._retriever, question)
        except Exception as exc:
            logger.exception("Generation failed")
            raise GenerationError(self._describe_failure(exc)) from exc

        return Answer(text=result["answer"].strip(), sources=result["sources"])

    def reset(self) -> None:
        self._chain = None
        self._retriever = None

    def _get_embeddings(self) -> Embeddings:
        if self._embeddings is None:
            self._embeddings = create_embeddings(
                self.settings.embedding_model, self.settings.embedding_device
            )
        return self._embeddings

    def _get_llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = create_llm(self.settings.ollama_model, self.settings.ollama_base_url)
        return self._llm

    def _describe_failure(self, exc: Exception) -> str:
        if _is_connection_failure(exc):
            return (
                f"Could not reach Ollama at {self.settings.ollama_base_url}. "
                "Start it with `ollama serve` and pull the model "
                f"with `ollama pull {self.settings.ollama_model}`."
            )
        return f"The model could not answer: {exc}"


def _is_connection_failure(exc: BaseException) -> bool:
    """Walk the exception chain looking for a transport-level failure."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, (ConnectionError, TimeoutError)):
            return True
        if type(exc).__name__ in _CONNECTION_ERROR_NAMES:
            return True
        exc = exc.__cause__ or exc.__context__
    return False
