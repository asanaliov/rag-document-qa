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


def format_sources(docs, query: str | None = None) -> str:
    """Render retrieved chunks as markdown, numbered to match the [n] citations in the answer."""
    if not docs:
        return NO_SOURCES

    parts = [f"*Searched for: {query}*"] if query else []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page")
        label = f"**[{i}]**" + (f" — page {page + 1}" if page is not None else "")
        text = " ".join(doc.page_content.split())
        parts.append(f"{label}\n\n> {text}")
    return "\n\n".join(parts)


def build_ui(settings: Settings) -> gr.Blocks:
    """Wire the pipeline up to the Gradio components."""

    def index_document(file, qa):
        if file is None:
            qa.reset()
            return IDLE_STATUS, [], NO_SOURCES, qa, []

        try:
            result = qa.index(file.name)
            status = f"**Ready** — {result.chunks} chunks from {result.documents} page(s)."
        except RagError as exc:
            status = f"**Could not index the document:** {exc}"
        except Exception:
            logger.exception("Indexing failed for %s", file.name)
            status = f"**Could not index the document.** {UNEXPECTED_ERROR}"

        # A new document means a new conversation: clear the chat, sources and history.
        return status, [], NO_SOURCES, qa, []

    def answer_question(question, chat, qa, turns):
        question = question.strip()
        if not question:
            yield chat, "", gr.update(), turns
            return

        chat = chat + [{"role": "user", "content": question}]

        def reply(text):
            return chat + [{"role": "assistant", "content": text}]

        if not qa.is_ready:
            yield reply("Upload a document first."), "", NO_SOURCES, turns
            return

        # Show the question and empty the box right away; the answer streams in below.
        yield chat, "", gr.update(), turns

        try:
            answer = None
            for answer in qa.stream(question, turns):
                # Only mention the search query when a follow-up was rewritten.
                query = answer.query if answer.query != question else None
                yield reply(answer.text), "", format_sources(answer.sources, query), turns
            if answer is not None:
                # Failed turns are left out, so an error never becomes conversation context.
                yield gr.update(), "", gr.update(), turns + [(question, answer.text.strip())]
        except RagError as exc:
            yield reply(str(exc)), "", NO_SOURCES, turns
        except Exception:
            logger.exception("Answering failed")
            yield reply(UNEXPECTED_ERROR), "", NO_SOURCES, turns

    with gr.Blocks(title="RAG Document Q&A", analytics_enabled=False) as ui:
        # One pipeline and one conversation per browser session, so two tabs
        # never overwrite each other's document.
        qa = gr.State(lambda: DocumentQA(settings))
        turns = gr.State([])

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
                        "4. A local LLM answers using only those chunks, "
                        "citing them as [1], [2], ...\n"
                        "5. Follow-up questions are rewritten to stand alone before the search"
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
                    stop_btn=True,
                )
                with gr.Accordion("Retrieved passages", open=False):
                    sources = gr.Markdown(NO_SOURCES)
                clear_btn = gr.Button("Clear chat", size="sm")

        file_input.change(
            index_document,
            inputs=[file_input, qa],
            outputs=[status, chatbot, sources, qa, turns],
        )
        question.submit(
            answer_question,
            inputs=[question, chatbot, qa, turns],
            outputs=[chatbot, question, sources, turns],
        )
        clear_btn.click(lambda: ([], NO_SOURCES, []), outputs=[chatbot, sources, turns])

        gr.HTML(
            '<div class="app-footer">'
            f"Embeddings: {settings.embedding_model} &middot; "
            f"LLM: {settings.ollama_model} via Ollama &middot; Vector store: FAISS"
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

    ui = build_ui(settings)
    ui.launch(
        theme=THEME,
        css=CSS,
        server_name=settings.server_host,
        server_port=settings.server_port,
        footer_links=[],
    )


if __name__ == "__main__":
    main()
