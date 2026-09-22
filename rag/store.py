"""Embedding model and vector index."""
from __future__ import annotations

import logging

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def create_embeddings(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str = "cpu",
) -> Embeddings:
    """Load the local sentence-transformer model. First call downloads the weights."""
    # Imported lazily: it pulls in torch, which costs several seconds of startup.
    from langchain_huggingface import HuggingFaceEmbeddings

    logger.info("Loading embedding model %s on %s", model_name, device)
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})


def build_vector_store(chunks: list[Document], embeddings: Embeddings) -> FAISS:
    """Embed the chunks into an in-memory FAISS index.

    In-memory is deliberate: an index is scoped to one uploaded document and is
    discarded when the next one is uploaded, so there is nothing worth persisting.
    """
    if not chunks:
        raise ValueError("Cannot build a vector store from zero chunks")

    store = FAISS.from_documents(chunks, embeddings)
    logger.info("Indexed %d chunk(s)", len(chunks))
    return store
