import gradio as gr
from dotenv import load_dotenv

from rag.loader import load_document, chunk_documents
from rag.store import create_embeddings, build_vector_store
from rag.qa import create_qa_chain, ask

load_dotenv()


class RAGApp:
    """Stateful wrapper for the RAG pipeline."""

    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.chain = None
        self.retriever = None

    def index_document(self, file):
        """Load, chunk, embed, and index an uploaded document."""
        if file is None:
            return "⚠️ Please upload a document first."

        try:
            # load and chunk
            docs = load_document(file.name)
            chunks = chunk_documents(docs)

            # lazy-init embeddings (heavy model, load once)
            if self.embeddings is None:
                self.embeddings = create_embeddings()

            # build vector store and QA chain
            self.vector_store = build_vector_store(chunks, self.embeddings)
            self.chain, self.retriever = create_qa_chain(self.vector_store)

            return (
                f"✅ Indexed **{len(chunks)} chunks** from "
                f"**{len(docs)} page(s)**. Ask away!"
            )
        except Exception as e:
            return f"❌ Error: {e}"

    def answer_question(self, question, history):
        """Run a question through the RAG pipeline."""
        if not question.strip():
            return history, ""

        if self.chain is None:
            history = history + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": "⚠️ Upload and index a document first."},
            ]
            return history, ""

        try:
            result = ask(self.chain, self.retriever, question)
            answer = result["answer"]
            answer += f"\n\n📄 *Retrieved {result['num_sources']} relevant chunks*"
        except Exception as e:
            answer = f"❌ Error: {e}"

        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        return history, ""


def build_ui(app: RAGApp):
    """Construct the Gradio interface."""
    with gr.Blocks(title="RAG Document Q&A") as ui:
        gr.Markdown("# 📄 RAG Document Q&A")
        gr.Markdown(
            "Upload a document, index it into a vector store, "
            "then ask questions answered by a local LLM using only the document's content."
        )

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Upload Document",
                    file_types=[".pdf", ".txt", ".md"],
                )
                index_btn = gr.Button("📥 Index Document", variant="primary")
                status = gr.Markdown("*Waiting for document...*")

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Chat", height=450)
                question = gr.Textbox(
                    label="Your question",
                    placeholder="What is this document about?",
                )
                ask_btn = gr.Button("🔍 Ask", variant="primary")

        # wire events
        index_btn.click(app.index_document, inputs=[file_input], outputs=[status])
        ask_btn.click(
            app.answer_question,
            inputs=[question, chatbot],
            outputs=[chatbot, question],
        )
        question.submit(
            app.answer_question,
            inputs=[question, chatbot],
            outputs=[chatbot, question],
        )

    return ui


if __name__ == "__main__":
    rag_app = RAGApp()
    ui = build_ui(rag_app)
    ui.launch(theme=gr.themes.Soft())
