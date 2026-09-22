import pytest
from langchain_core.documents import Document

from rag.errors import EmptyDocumentError, FileTooLargeError, UnsupportedFileTypeError
from rag.loader import chunk_documents, load_document


def test_loads_txt_file(sample_text_file):
    docs = load_document(sample_text_file)

    assert len(docs) == 1
    assert "Zorblax" in docs[0].page_content


def test_loads_md_file(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Notes\n\nThe Zorblax protocol uses a 42-bit handshake.", encoding="utf-8")

    docs = load_document(str(path))

    assert len(docs) == 1
    assert "42-bit handshake" in docs[0].page_content


def test_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "report.docx"
    path.write_bytes(b"not really a word document")

    with pytest.raises(UnsupportedFileTypeError) as excinfo:
        load_document(str(path))

    assert ".docx" in str(excinfo.value)


def test_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_document(str(tmp_path / "nope.txt"))


def test_rejects_file_over_size_limit(sample_text_file):
    with pytest.raises(FileTooLargeError, match="exceeds"):
        load_document(sample_text_file, max_size_mb=0)


def test_rejects_document_with_no_extractable_text(tmp_path):
    path = tmp_path / "blank.txt"
    path.write_text("   \n\n  ", encoding="utf-8")

    with pytest.raises(EmptyDocumentError):
        load_document(str(path))


def test_chunks_overlap():
    # Numbered sentences make every position unique, so shared text between
    # consecutive chunks can only come from the overlap.
    text = " ".join(f"Sentence number {i} is unique." for i in range(400))
    chunks = chunk_documents([Document(page_content=text)])

    assert len(chunks) > 1
    assert chunks[0].page_content[-50:] in chunks[1].page_content


def test_chunking_preserves_metadata():
    doc = Document(page_content="x " * 2000, metadata={"source": "guide.pdf", "page": 2})

    chunks = chunk_documents([doc])

    assert all(chunk.metadata["source"] == "guide.pdf" for chunk in chunks)
    assert all(chunk.metadata["page"] == 2 for chunk in chunks)
