"""Prompts, language model, and the chains that rewrite and answer questions."""
from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

NO_ANSWER = "I couldn't find that in the document."

SYSTEM_PROMPT = f"""You answer questions using only the provided context.
The context is a list of numbered passages.
Rules:
- Use only facts stated in the context. Do not add outside knowledge.
- If the context does not contain the answer, reply: "{NO_ANSWER}"
- Cite the passages you used by their number in square brackets, like [1] or [2][3].
- Answer directly and concisely. Do not repeat the question or pad the response."""

QA_TEMPLATE = """Context:
{context}

Question: {question}

Answer based only on the context above:"""

CONDENSE_SYSTEM_PROMPT = """Rewrite the follow-up question as a standalone question that can be \
understood without the conversation, replacing pronouns and references with what they refer to.
Keep the original language. Do not answer it. Reply with the rewritten question only."""

CONDENSE_TEMPLATE = """Conversation:
{history}

Follow-up question: {question}

Standalone question:"""


def format_docs(docs: list[Document]) -> str:
    """Join retrieved chunks into one context string, numbered so the model can cite them."""
    return "\n\n".join(f"[{i}]\n{doc.page_content}" for i, doc in enumerate(docs, 1))


def format_history(history: Sequence[tuple[str, str]]) -> str:
    return "\n".join(f"User: {question}\nAssistant: {answer}" for question, answer in history)


def create_llm(model: str, base_url: str, temperature: float = 0.0) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model, base_url=base_url, temperature=temperature)


def create_qa_chain(llm: BaseChatModel) -> Runnable:
    """Build the LCEL chain (prompt -> LLM -> text) that answers from given context.

    Retrieval is deliberately not part of the chain, so the caller can show the
    user the exact passages an answer was drawn from.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", QA_TEMPLATE),
    ])
    return prompt | llm | StrOutputParser()


def create_condense_chain(llm: BaseChatModel) -> Runnable:
    """Build the chain that turns a follow-up ("and when was that?") into a standalone question."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", CONDENSE_SYSTEM_PROMPT),
        ("human", CONDENSE_TEMPLATE),
    ])
    return prompt | llm | StrOutputParser()
