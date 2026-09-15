import gradio as gr
from dotenv import load_dotenv

from rag.loader import load_document, chunk_documents
from rag.store import create_embeddings, build_vector_store
from rag.qa import create_qa_chain, ask

load_dotenv()

NO_SOURCES = "*Retrieved passages will appear here after you ask a question.*"

CHAT_PLACEHOLDER = """
<div style="text-align:center; opacity:0.6">
  <div style="font-size:2.5rem">💬</div>
  <p>Upload a document on the left, then ask anything about it.</p>
</div>
"""

HEADER = """
<div class="hero">
  <h1>📄 RAG Document Q&amp;A</h1>
  <p>Ask questions about your own documents, answered from their content only.</p>
  <div class="badges">
    <span>🔒 100% local</span><span>🔑 No API keys</span><span>💸 Free</span>
  </div>
</div>
"""

CSS = """
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
.hero { text-align: center; padding: 1.5rem 0 1rem; }
.hero h1 { font-size: 2.2rem; margin: 0 0 .4rem; }
.hero p { margin: 0; opacity: .7; font-size: 1.05rem; }
.badges { display: flex; gap: .5rem; justify-content: center; margin-top: .9rem; }
.badges span {
  font-size: .8rem; padding: .25rem .7rem; border-radius: 999px;
  background: var(--color-accent-soft); color: var(--color-accent);
}
.sidebar { background: var(--block-background-fill); border: 1px solid var(--border-color-primary);
  border-radius: var(--block-radius); padding: 1rem; }
.status { min-height: 2.2rem; }
footer { display: none !important; }
.app-footer { text-align: center; opacity: .5; font-size: .8rem; padding: 1rem 0 .5rem; }
"""

THEME = gr.themes.Soft(
    primary_hue="indigo",
    neutral_hue="slate",
    radius_size="lg",
    font=gr.themes.GoogleFont("Inter"),
)


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
            self.chain = self.retriever = self.vector_store = None
            return "*Waiting for a document...*", [], NO_SOURCES

        try:
            docs = load_document(file.name)
            chunks = chunk_documents(docs)

            # lazy-init embeddings (heavy model, load once)
            if self.embeddings is None:
                self.embeddings = create_embeddings()

            self.vector_store = build_vector_store(chunks, self.embeddings)
            self.chain, self.retriever = create_qa_chain(self.vector_store)

            status = (
                f"✅ **Ready** — {len(chunks)} chunks from {len(docs)} page(s). "
                "Ask a question below."
            )
        except Exception as e:
            self.chain = self.retriever = self.vector_store = None
            status = f"❌ **Error:** {e}"

        # new document: start a fresh conversation
        return status, [], NO_SOURCES

    def answer_question(self, question, history):
        """Run a question through the RAG pipeline."""
        question = question.strip()
        if not question:
            return history, "", gr.update()

        if self.chain is None:
            answer = "⚠️ Upload a document first."
            sources_md = NO_SOURCES
        else:
            try:
                result = ask(self.chain, self.retriever, question)
                answer = result["answer"]
                sources_md = format_sources(result["sources"])
            except Exception as e:
                answer = f"❌ Error: {e}"
                sources_md = NO_SOURCES

        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        return history, "", sources_md


def format_sources(docs, max_chars: int = 400):
    """Render retrieved chunks as markdown so the user can verify the answer."""
    parts = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page")
        label = f"**Passage {i}**" + (f" — page {page + 1}" if page is not None else "")
        text = doc.page_content.strip().replace("\n", " ")
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "…"
        parts.append(f"{label}\n\n> {text}")
    return "\n\n".join(parts)


def build_ui(app: RAGApp):
    """Construct the Gradio interface."""
    with gr.Blocks(title="RAG Document Q&A") as ui:
        gr.HTML(HEADER)

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=280, elem_classes="sidebar"):
                file_input = gr.File(
                    label="Document (PDF, TXT, MD)",
                    file_types=[".pdf", ".txt", ".md"],
                    height=140,
                )
                status = gr.Markdown("*Waiting for a document...*", elem_classes="status")
                with gr.Accordion("How it works", open=False):
                    gr.Markdown(
                        "1. The document is split into overlapping chunks\n"
                        "2. Each chunk is embedded and stored in a FAISS index\n"
                        "3. Your question retrieves the most similar chunks\n"
                        "4. A local LLM answers using only those chunks"
                    )

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    height=480,
                    show_label=False,
                    placeholder=CHAT_PLACEHOLDER,
                )
                question = gr.Textbox(
                    show_label=False,
                    placeholder="Ask something about the document and press Enter…",
                    submit_btn=True,
                )
                with gr.Accordion("Retrieved passages", open=False):
                    sources = gr.Markdown(NO_SOURCES)
                clear_btn = gr.Button("Clear chat", size="sm")

        file_input.change(
            app.index_document,
            inputs=[file_input],
            outputs=[status, chatbot, sources],
        )
        question.submit(
            app.answer_question,
            inputs=[question, chatbot],
            outputs=[chatbot, question, sources],
        )
        clear_btn.click(lambda: ([], NO_SOURCES), outputs=[chatbot, sources])

        gr.HTML('<div class="app-footer">Embeddings: all-MiniLM-L6-v2 · LLM: Ollama · Vector store: FAISS</div>')

    return ui


if __name__ == "__main__":
    rag_app = RAGApp()
    ui = build_ui(rag_app)
    ui.launch(theme=THEME, css=CSS, footer_links=[])
