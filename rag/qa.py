"""Prompt, language model, and the retrieval-augmented generation chain."""
from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)

NO_ANSWER = "I couldn't find that in the document."

SYSTEM_PROMPT = f"""You answer questions using only the provided context.
Rules:
- Use only facts stated in the context. Do not add outside knowledge.
- If the context does not contain the answer, reply: "{NO_ANSWER}"
- Answer directly and concisely. Do not repeat the question or pad the response."""

QA_TEMPLATE = """Context:
{context}

Question: {question}

Answer based only on the context above:"""


def format_docs(docs: list[Document]) -> str:
    """Join retrieved chunks into one context string, delimited so the model sees the seams."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def create_llm(model: str, base_url: str, temperature: float = 0.0) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model, base_url=base_url, temperature=temperature)


def create_qa_chain(
    vector_store: VectorStore,
    llm: BaseChatModel,
    k: int = 4,
) -> tuple[Runnable, BaseRetriever]:
    """Build the LCEL chain (prompt -> LLM -> text) and its top-k retriever.

    The retriever is returned separately rather than wired into the chain so the
    caller can show the user the exact passages an answer was drawn from.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", QA_TEMPLATE),
    ])
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return prompt | llm | StrOutputParser(), retriever


def ask(chain: Runnable, retriever: BaseRetriever, question: str) -> dict:
    """Retrieve the relevant chunks, then answer from them."""
    sources = retriever.invoke(question)
    logger.debug("Retrieved %d chunk(s) for question: %s", len(sources), question)
    answer = chain.invoke({"question": question, "context": format_docs(sources)})

    return {"answer": answer, "sources": sources, "num_sources": len(sources)}
