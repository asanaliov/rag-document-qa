"""Gradio front end for the document question-answering pipeline."""
from __future__ import annotations

import logging

import gradio as gr
from dotenv import load_dotenv

from rag.config import Settings
from rag.errors import RagError
from rag.loader import SUPPORTED_EXTENSIONS
from rag.pipeline import DocumentQA

logger = logging.getLogger(__name__)

IDLE_STATUS = "*Waiting for a document...*"
NO_SOURCES = "*Retrieved passages will appear here after you ask a question.*"
UNEXPECTED_ERROR = "Something went wrong. Check the server logs for details."

CHAT_PLACEHOLDER = """
<div style="text-align:center; opacity:0.6">
  <p>Upload a document on the left, then ask anything about it.</p>
</div>
"""

HEADER = """
<div class="hero">
  <h1>RAG Document Q&amp;A</h1>
  <p>Ask questions about your own documents, answered from their content only.</p>
  <div class="badges">
    <span>Runs locally</span><span>No API keys</span><span>Answers cite their passages</span>
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


def format_sources(docs, max_chars: int = 400) -> str:
    """Render retrieved chunks as markdown so the answer can be checked against them."""
    if not docs:
        return NO_SOURCES

    parts = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page")
        label = f"**Passage {i}**" + (f" — page {page + 1}" if page is not None else "")
        text = " ".join(doc.page_content.split())
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."
        parts.append(f"{label}\n\n> {text}")
    return "\n\n".join(parts)


def build_ui(qa: DocumentQA) -> gr.Blocks:
    """Wire the pipeline up to the Gradio components."""

    def index_document(file):
        if file is None:
            qa.reset()
            return IDLE_STATUS, [], NO_SOURCES

        try:
            result = qa.index(file.name)
            status = f"**Ready** — {result.chunks} chunks from {result.documents} page(s)."
        except RagError as exc:
            status = f"**Could not index the document:** {exc}"
        except Exception:
            logger.exception("Indexing failed for %s", file.name)
            status = f"**Could not index the document.** {UNEXPECTED_ERROR}"

        # A new document means a new conversation: clear the chat and sources.
        return status, [], NO_SOURCES

    def answer_question(question, history):
        question = question.strip()
        if not question:
            return history, "", gr.update()

        if not qa.is_ready:
            answer, sources_md = "Upload a document first.", NO_SOURCES
        else:
            try:
                result = qa.ask(question)
                answer, sources_md = result.text, format_sources(result.sources)
            except RagError as exc:
                answer, sources_md = str(exc), NO_SOURCES
            except Exception:
                logger.exception("Answering failed")
                answer, sources_md = UNEXPECTED_ERROR, NO_SOURCES

        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        return history, "", sources_md

    with gr.Blocks(title="RAG Document Q&A", analytics_enabled=False) as ui:
        gr.HTML(HEADER)

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=280, elem_classes="sidebar"):
                file_input = gr.File(
                    label=f"Document ({', '.join(SUPPORTED_EXTENSIONS)})",
                    file_types=list(SUPPORTED_EXTENSIONS),
                    height=140,
                )
                status = gr.Markdown(IDLE_STATUS, elem_classes="status")
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
                    placeholder="Ask something about the document and press Enter",
                    submit_btn=True,
                )
                with gr.Accordion("Retrieved passages", open=False):
                    sources = gr.Markdown(NO_SOURCES)
                clear_btn = gr.Button("Clear chat", size="sm")

        file_input.change(index_document, inputs=[file_input], outputs=[status, chatbot, sources])
        question.submit(
            answer_question,
            inputs=[question, chatbot],
            outputs=[chatbot, question, sources],
        )
        clear_btn.click(lambda: ([], NO_SOURCES), outputs=[chatbot, sources])

        gr.HTML(
            '<div class="app-footer">'
            f"Embeddings: {qa.settings.embedding_model} &middot; "
            f"LLM: {qa.settings.ollama_model} via Ollama &middot; Vector store: FAISS"
            "</div>"
        )

    return ui


def main() -> None:
    load_dotenv()
    settings = Settings.from_env()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    # Ollama and Gradio both call through httpx, which logs every request at INFO.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    ui = build_ui(DocumentQA(settings))
    ui.launch(
        theme=THEME,
        css=CSS,
        server_name=settings.server_host,
        server_port=settings.server_port,
        footer_links=[],
    )


if __name__ == "__main__":
    main()
