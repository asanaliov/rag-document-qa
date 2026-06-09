from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_document(file_path: str):
    """Load a PDF, TXT, or MD file into a list of Document objects."""
    path = Path(file_path)

    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(file_path)
    elif path.suffix.lower() in {".txt", ".md"}:
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(
            f"Unsupported file type: {path.suffix}. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    return loader.load()


def chunk_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Split documents into overlapping chunks for embedding.

    The overlap keeps context that spans a chunk boundary intact.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
