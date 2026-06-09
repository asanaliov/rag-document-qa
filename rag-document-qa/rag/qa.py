import os

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# local model served by Ollama
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")


SYSTEM_PROMPT = """You answer questions using only the provided context.
Rules:
- Use only facts stated in the context. Do not add outside knowledge.
- If the context does not contain the answer, reply: "I couldn't find that in the document."
- Answer directly and concisely. Do not repeat the question or pad the response."""

QA_TEMPLATE = """Context:
{context}

Question: {question}

Answer based only on the context above:"""


def format_docs(docs):
    """Join retrieved documents into a single context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def create_qa_chain(vector_store, model: str = DEFAULT_MODEL):
    """Build the LCEL chain (prompt -> LLM -> string) and a top-k retriever."""
    llm = ChatOllama(model=model, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", QA_TEMPLATE),
    ])

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    chain = prompt | llm | StrOutputParser()

    return chain, retriever


def ask(chain, retriever, question: str):
    """Retrieve relevant chunks once, then generate an answer from them."""
    sources = retriever.invoke(question)
    answer = chain.invoke({"question": question, "context": format_docs(sources)})

    return {
        "answer": answer,
        "sources": sources,
        "num_sources": len(sources),
    }
