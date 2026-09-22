from langchain_core.documents import Document

from rag.qa import NO_ANSWER, SYSTEM_PROMPT, ask, create_qa_chain, format_docs
from rag.store import build_vector_store


def _store(fake_embeddings):
    chunks = [
        Document(page_content="The Zorblax protocol was invented in 1987 by Dr. Mira Chen."),
        Document(page_content="The capital of the Zorblax Republic is Vantorra."),
    ]
    return build_vector_store(chunks, fake_embeddings)


def test_format_docs_separates_chunks():
    joined = format_docs([Document(page_content="first"), Document(page_content="second")])

    assert "first" in joined and "second" in joined
    assert joined.count("---") == 1


def test_system_prompt_states_the_fallback_answer():
    assert NO_ANSWER in SYSTEM_PROMPT


def test_retriever_honours_k(fake_embeddings, fake_llm):
    _, retriever = create_qa_chain(_store(fake_embeddings), fake_llm, k=1)

    assert len(retriever.invoke("Who invented the protocol?")) == 1


def test_ask_returns_answer_and_the_sources_behind_it(fake_embeddings, fake_llm):
    chain, retriever = create_qa_chain(_store(fake_embeddings), fake_llm, k=2)

    result = ask(chain, retriever, "Who invented the Zorblax protocol?")

    assert result["answer"] == "Dr. Mira Chen invented it in 1987."
    assert result["num_sources"] == len(result["sources"]) == 2
