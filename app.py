"""Gradio front end for the document question-answering pipeline."""
from __future__ import annotations

import logging
import threading

import gradio as gr
from dotenv import load_dotenv

from rag.config import Settings
from rag.errors import RagError
from rag.loader import SUPPORTED_EXTENSIONS
from rag.pipeline import DocumentQA

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

IDLE_STATUS = "Waiting for a document..."
NO_SOURCES = "Retrieved passages will appear here after you ask a question."
UNEXPECTED_ERROR = "Something went wrong. Check the server logs for details."

WARMUP_PENDING = "warming-up"
WARMUP_READY = "ready"
WARMUP_FAILED = "failed"

CSS = """
.gradio-container { max-width: 1440px !important; margin: 0 auto !important; }
footer { display: none !important; }
.app-header {
  display: flex; align-items: end; justify-content: space-between; gap: 1.5rem;
  padding: 2rem 0 1.4rem; border-bottom: 1px solid var(--border-color-primary);
  margin-bottom: 1.5rem;
}
.app-title { font-size: clamp(1.8rem, 3vw, 2.5rem); line-height: 1.05; letter-spacing: -.04em; font-weight: 700; margin: 0 0 .55rem; }
.app-subtitle { font-size: 1rem; line-height: 1.55; opacity: .68; margin: 0; max-width: 42rem; }
.app-badges { display: flex; gap: .5rem; flex-wrap: wrap; justify-content: flex-end; }
.app-badges span { font-size: .75rem; padding: .4rem .7rem; border: 1px solid var(--border-color-primary); border-radius: .55rem; white-space: nowrap; }
.warmup-banner { display: flex; align-items: center; gap: .65rem; padding: .8rem 1rem; border-radius: .75rem; font-size: .9rem; font-weight: 500; margin-bottom: 1.2rem; }
.warmup-banner.pending { background: #fff8e7; color: #805512; border: 1px solid #f1d9a5; }
.warmup-banner.ready { background: #edf7f0; color: #286044; border: 1px solid #c6e2d0; }
.warmup-banner.failed { background: #fff0ed; color: #8c382d; border: 1px solid #efc9c1; }
.warmup-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.warmup-dot.pending { background: #c88b23; animation: pulse-dot 1.4s infinite; }
.warmup-dot.ready { background: #39845b; }
.warmup-dot.failed { background: #bd5142; }
@keyframes pulse-dot { 0%, 100% { opacity: 1; } 50% { opacity: .3; } }
.setup-panel { padding: 1rem 1.15rem; margin-bottom: 1.2rem; border: 1px solid var(--border-color-primary); border-radius: .9rem; background: var(--block-background-fill); }
.section-label { font-size: .78rem; font-weight: 650; letter-spacing: .04em; opacity: .65; margin: 0 0 .65rem; }
.document-status { font-size: .9rem; line-height: 1.5; }
.chat-panel { min-width: 0; }
.chat-panel .bubble-wrap { padding: 1.1rem; }
.chat-panel .sources-header { font-size: .78rem; font-weight: 650; opacity: .65; letter-spacing: .04em; margin: 1.15rem 0 .55rem; }
.app-footer { opacity: .5; font-size: .78rem; padding: 1rem 0 .4rem; border-top: 1px solid var(--border-color-primary); margin-top: 1.5rem; }
button, input, textarea { transition: background-color .18s ease, border-color .18s ease, transform .18s ease; }
button:active { transform: translateY(1px); }
button:focus-visible, input:focus-visible, textarea:focus-visible { outline: 3px solid #8196b8; outline-offset: 2px; }
@media (max-width: 700px) {
  .app-header { align-items: flex-start; flex-direction: column; padding-top: 1.2rem; }
  .app-badges { justify-content: flex-start; }
  .setup-panel { padding: .85rem; }
}
"""

THEME = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
    radius_size="md",
    font=gr.themes.Default(),
)


# --------------------------------------------------------------------------- #
# Warmup
# --------------------------------------------------------------------------- #

def warmup_ollama(qa: DocumentQA, on_done: callable) -> None:
    """Send a tiny dummy request so Ollama loads the model into RAM.

    Runs in a background daemon thread so the Gradio server starts immediately.
    The first real question then gets a fast response instead of waiting several
    seconds for the model to load from disk.
    """
    try:
        from langchain_core.messages import HumanMessage
        from langchain_ollama import ChatOllama

        model = qa.settings.ollama_model
        base_url = qa.settings.ollama_base_url
        logger.info("Warming up model %s at %s ...", model, base_url)

        llm = ChatOllama(model=model, base_url=base_url, temperature=0.0, num_predict=1)
        llm.invoke([HumanMessage(content="hi")])
        logger.info("Model warm-up complete.")
        on_done(WARMUP_READY)
    except Exception:
        logger.warning("Model warm-up failed (Ollama may not be running yet).", exc_info=True)
        on_done(WARMUP_FAILED)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def format_sources(docs, max_chars: int = 450) -> str:
    """Render retrieved chunks as readable markdown."""
    if not docs:
        return NO_SOURCES

    parts = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page")
        label = f"**Passage {i}**" + (f" - page {page + 1}" if page is not None else "")
        text = " ".join(doc.page_content.split())
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."
        parts.append(f"{label}\n\n> {text}")
    return "\n\n".join(parts)


def warmup_html(state: str, model: str = "") -> str:
    """Generate the HTML for the warmup status banner."""
    pending_msg = f"Loading {model} into memory - first response may take a few extra seconds"
    failed_msg = f"Could not reach Ollama. Run: ollama serve &amp;&amp; ollama pull {model}"
    labels = {
        WARMUP_PENDING: ("pending", pending_msg),
        WARMUP_READY:   ("ready",   f"{model} is ready - answers will be fast"),
        WARMUP_FAILED:  ("failed",  failed_msg),
    }
    cls, msg = labels.get(state, labels[WARMUP_PENDING])
    return (
        f'<div class="warmup-banner {cls}">'
        f'<div class="warmup-dot {cls}"></div>'
        f"{msg}"
        f"</div>"
    )


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

def build_ui(qa: DocumentQA, warmup_state: list[str]) -> gr.Blocks:
    """Wire the pipeline up to the Gradio components.

    warmup_state is a mutable single-element list so the background thread
    can write to it and the UI timer can read it without needing global state.
    """

    model_name = qa.settings.ollama_model
    embed_name = qa.settings.embedding_model
    def index_document(file):
        if file is None:
            qa.reset()
            return f"*{IDLE_STATUS}*", [], f"*{NO_SOURCES}*"

        try:
            result = qa.index(file.name)
            status = f"**Ready** - {result.chunks} chunks from {result.documents} page(s)."
        except RagError as exc:
            status = f"**Error:** {exc}"
        except Exception:
            logger.exception("Indexing failed for %s", file.name)
            status = f"**Error:** {UNEXPECTED_ERROR}"

        return status, [], f"*{NO_SOURCES}*"

    def answer_question(question, history):
        question = question.strip()
        if not question:
            return history, "", gr.update()

        if not qa.is_ready:
            answer = "Please upload a document first."
            sources_md = f"*{NO_SOURCES}*"
        else:
            try:
                result = qa.ask(question)
                answer = result.text
                sources_md = format_sources(result.sources)
            except RagError as exc:
                answer = str(exc)
                sources_md = f"*{NO_SOURCES}*"
            except Exception:
                logger.exception("Answering failed")
                answer = UNEXPECTED_ERROR
                sources_md = f"*{NO_SOURCES}*"

        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        return history, "", sources_md

    def poll_warmup():
        return warmup_html(warmup_state[0], model_name)

    with gr.Blocks(
        title="RAG Document Q&A",
        analytics_enabled=False,
        theme=THEME,
        css=CSS,
    ) as ui:

        # ---- header ----
        gr.HTML(
            '<div class="app-header">'
            "<div>"
            '<p class="app-title">Ask your documents.</p>'
            '<p class="app-subtitle">'
            "Upload a file, ask a question, and check the passages behind every answer."
            "</p>"
            "</div>"
            '<div class="app-badges">'
            "<span>Runs locally</span>"
            "<span>No API keys</span>"
            "<span>Cited answers</span>"
            "</div>"
            "</div>"
        )

        # ---- model warmup banner (auto-refreshes every 2 s until ready) ----
        warmup_banner = gr.HTML(warmup_html(warmup_state[0], model_name))
        warmup_timer = gr.Timer(value=2, active=True)

        with gr.Column(elem_classes=["setup-panel"]):
            gr.HTML('<p class="section-label">1 / ADD A DOCUMENT</p>')
            with gr.Row(equal_height=False):
                file_input = gr.File(
                    label=f"Choose a PDF, TXT, or Markdown file · up to {qa.settings.max_upload_mb} MB",
                    file_types=list(SUPPORTED_EXTENSIONS),
                    height=105,
                    scale=3,
                )
                with gr.Column(scale=2, min_width=220):
                    gr.HTML('<p class="section-label">DOCUMENT STATUS</p>')
                    doc_status = gr.Markdown(f"*{IDLE_STATUS}*", elem_classes=["document-status"])
                    gr.Markdown("Your file stays on this machine.")

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, elem_classes=["chat-panel"]):
                gr.HTML('<p class="section-label">2 / ASK A QUESTION</p>')
                chatbot = gr.Chatbot(
                    height=540,
                    show_label=False,
                    type="messages",
                    placeholder=(
                        "<div style='text-align:center;opacity:.5;padding:3rem 1rem'>"
                        "<p style='font-size:2rem'>💬</p>"
                        "<p>Add a document above, then ask a specific question.</p>"
                        "</div>"
                    ),
                )

                with gr.Row():
                    question = gr.Textbox(
                        show_label=False,
                        placeholder="What would you like to find in this document?",
                        scale=5,
                        container=False,
                        autofocus=True,
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)

                clear_btn = gr.Button("Clear conversation", size="sm", variant="secondary")

                # Sources are always visible - no accordion hiding them
                gr.HTML('<p class="sources-header">PASSAGES USED FOR THE ANSWER</p>')
                sources = gr.Markdown(f"*{NO_SOURCES}*")

        # ---- footer ----
        gr.HTML(
            '<div class="app-footer">'
            f"Embeddings: {embed_name} &middot; "
            f"LLM: {model_name} via Ollama &middot; Vector store: FAISS"
            "</div>"
        )

        # ---- event wiring ----
        file_input.change(
            index_document,
            inputs=[file_input],
            outputs=[doc_status, chatbot, sources],
        )

        submit_inputs = [question, chatbot]
        submit_outputs = [chatbot, question, sources]
        question.submit(answer_question, inputs=submit_inputs, outputs=submit_outputs)
        submit_btn.click(answer_question, inputs=submit_inputs, outputs=submit_outputs)

        clear_btn.click(
            lambda: ([], f"*{NO_SOURCES}*"),
            outputs=[chatbot, sources],
        )

        warmup_timer.tick(poll_warmup, outputs=[warmup_banner])

    return ui


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    load_dotenv()
    settings = Settings.from_env()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    # Ollama and Gradio both call through httpx, which logs every request at INFO.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    qa = DocumentQA(settings)

    # warmup_state is a one-element mutable list so the daemon thread can write
    # to it and the Gradio gr.Timer can read it without any shared globals.
    warmup_state = [WARMUP_PENDING]

    def set_warmup(state: str) -> None:
        warmup_state[0] = state

    # Fire-and-forget: sends a tiny request to Ollama so the model is loaded
    # into RAM before the first real question arrives.
    thread = threading.Thread(target=warmup_ollama, args=(qa, set_warmup), daemon=True)
    thread.start()

    ui = build_ui(qa, warmup_state)
    ui.launch(
        server_name=settings.server_host,
        server_port=settings.server_port,
        show_api=False,
    )


if __name__ == "__main__":
    main()
