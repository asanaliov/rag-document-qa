"""Stateful orchestration of the RAG pipeline, independent of any UI."""
from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from rag.config import Settings
from rag.errors import GenerationError
from rag.loader import chunk_documents, load_document
from rag.qa import create_condense_chain, create_llm, create_qa_chain, format_docs, format_history
from rag.store import build_vector_store, create_embeddings

logger = logging.getLogger(__name__)

# httpx errors are matched by name so this module does not depend on httpx directly.
_CONNECTION_ERROR_NAMES = {"ConnectError", "ConnectTimeout", "ReadTimeout"}

# Follow-ups rarely refer further back than this, and a short history keeps the rewrite fast.
_HISTORY_TURNS = 3


@dataclass(frozen=True)
class IndexResult:
    name: str
    documents: int
    chunks: int


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Document]
    # The question actually searched for: differs from the user's when a follow-up was rewritten.
    query: str


class DocumentQA:
    """Indexes one document at a time and answers questions about it.

    Holds a single document's index: uploading a new one replaces the previous
    state. One instance serves one user session; the embedding model behind it
    is cached in `rag.store`, so sessions share the loaded weights.
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
        self._condense_chain = None
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
        self._retriever = vector_store.as_retriever(
            search_kwargs={"k": self.settings.retrieval_k}
        )
        self._condense_chain = create_condense_chain(self._get_llm())
        self._chain = create_qa_chain(self._get_llm())

        source = documents[0].metadata.get("source", file_path)
        return IndexResult(name=source, documents=len(documents), chunks=len(chunks))

    def stream(
        self, question: str, history: Sequence[tuple[str, str]] = ()
    ) -> Iterator[Answer]:
        """Answer a question, yielding the answer so far as each piece of text arrives.

        `history` is the earlier (question, answer) turns. It is only used to
        rewrite a follow-up into a standalone question; the answer itself is
        still generated from the retrieved passages alone.
        """
        if not self.is_ready:
            raise RuntimeError("No document has been indexed yet")

        try:
            query = self._standalone_question(question, history)
            sources = self._retriever.invoke(query)
            logger.debug("Retrieved %d chunk(s) for: %s", len(sources), query)

            text = ""
            for piece in self._chain.stream({"question": query, "context": format_docs(sources)}):
                text += piece
                yield Answer(text=text, sources=sources, query=query)
        except Exception as exc:
            logger.exception("Generation failed")
            raise GenerationError(self._describe_failure(exc)) from exc

    def ask(self, question: str, history: Sequence[tuple[str, str]] = ()) -> Answer:
        """Answer a question from the indexed document, returning only the finished answer."""
        answers = list(self.stream(question, history))
        if not answers:
            return Answer(text="", sources=[], query=question)
        return replace(answers[-1], text=answers[-1].text.strip())

    def reset(self) -> None:
        self._chain = None
        self._condense_chain = None
        self._retriever = None

    def _standalone_question(self, question: str, history: Sequence[tuple[str, str]]) -> str:
        if not history:
            return question
        rewritten = self._condense_chain.invoke({
            "question": question,
            "history": format_history(history[-_HISTORY_TURNS:]),
        })
        return rewritten.strip() or question

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
