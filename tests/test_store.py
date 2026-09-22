import pytest
from langchain_core.documents import Document

from rag.store import build_vector_store


def test_retrieves_the_most_similar_chunk(fake_embeddings):
    chunks = [
        Document(page_content="The capital of the Zorblax Republic is Vantorra."),
        Document(page_content="Unrelated text about gardening in the autumn."),
    ]
    store = build_vector_store(chunks, fake_embeddings)

    results = store.similarity_search("Vantorra", k=1)

    assert len(results) == 1


def test_rejects_empty_chunk_list(fake_embeddings):
    with pytest.raises(ValueError, match="zero chunks"):
        build_vector_store([], fake_embeddings)
