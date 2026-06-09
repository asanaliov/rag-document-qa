from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def create_embeddings(model_name: str = DEFAULT_MODEL):
    """Initialize the local sentence-transformer embedding model."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
    )


def build_vector_store(chunks, embeddings):
    """Embed the chunks and build an in-memory FAISS index for search."""
    return FAISS.from_documents(chunks, embeddings)
