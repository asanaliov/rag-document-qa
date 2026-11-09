from langchain_core.documents import Document

from rag.qa import (
    NO_ANSWER,
    SYSTEM_PROMPT,
    create_condense_chain,
    create_qa_chain,
    format_docs,
    format_history,
)


def test_format_docs_numbers_chunks_for_citation():
    joined = format_docs([Document(page_content="first"), Document(page_content="second")])

    assert joined == "[1]\nfirst\n\n[2]\nsecond"


def test_format_history_labels_each_turn():
    assert format_history([("Who?", "Dr. Chen.")]) == "User: Who?\nAssistant: Dr. Chen."


def test_system_prompt_states_the_fallback_answer():
    assert NO_ANSWER in SYSTEM_PROMPT


def test_qa_chain_returns_the_model_text(fake_llm):
    chain = create_qa_chain(fake_llm)

    answer = chain.invoke({"question": "Who invented it?", "context": "[1]\nsome context"})

    assert answer == "Dr. Mira Chen invented it in 1987."


def test_condense_chain_returns_the_rewritten_question(fake_llm):
    chain = create_condense_chain(fake_llm)

    rewritten = chain.invoke({"question": "When?", "history": "User: Who?\nAssistant: Chen."})

    assert rewritten == "Dr. Mira Chen invented it in 1987."
