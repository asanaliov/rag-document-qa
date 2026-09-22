"""Reading documents off disk and splitting them into retrievable chunks."""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.errors import EmptyDocumentError, FileTooLargeError, UnsupportedFileTypeError

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md")

_BYTES_PER_MB = 1024 * 1024


def load_document(file_path: str, max_size_mb: int | None = None) -> list[Document]:
    """Load a PDF, TXT, or MD file into Documents (one per PDF page, else one)."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type: {suffix or path.name}. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    if not path.is_file():
        raise FileNotFoundError(f"No such file: {file_path}")

    size_mb = path.stat().st_size / _BYTES_PER_MB
    if max_size_mb is not None and size_mb > max_size_mb:
        raise FileTooLargeError(
            f"File is {size_mb:.1f} MB, which exceeds the {max_size_mb} MB limit."
        )

    loader = PyPDFLoader(file_path) if suffix == ".pdf" else TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    if not any(doc.page_content.strip() for doc in documents):
        raise EmptyDocumentError(
            "No text could be extracted. Scanned PDFs need OCR before they can be indexed."
        )

    logger.info("Loaded %s (%.2f MB) as %d document(s)", path.name, size_mb, len(documents))
    return documents


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split documents into overlapping chunks so sentences spanning a boundary survive."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split %d document(s) into %d chunk(s)", len(documents), len(chunks))
    return chunks
