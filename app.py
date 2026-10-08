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
/* ---- layout ---- */
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }
footer { display: none !important; }

/* ---- header ---- */
.app-header {
  display: flex; align-items: center; gap: 1rem;
  padding: 1.25rem 0 1rem;
  border-bottom: 1px solid var(--border-color-primary);
  margin-bottom: 1.25rem;
}
.app-title { font-size: 1.5rem; font-weight: 700; margin: 0; }
.app-subtitle { font-size: .9rem; opacity: .6; margin: 0; }
.app-badges {
  display: flex; gap: .5rem; margin-left: auto; flex-wrap: wrap; justify-content: flex-end;
}
.app-badges span {
  font-size: .75rem; padding: .2rem .65rem; border-radius: 999px;
  background: var(--color-accent-soft); color: var(--color-accent); white-space: nowrap;
}

/* ---- warmup banner ---- */
.warmup-banner {
  display: flex; align-items: center; gap: .6rem;
  padding: .6rem 1rem; border-radius: var(--block-radius);
  font-size: .875rem; font-weight: 500; margin-bottom: 1rem;
}
.warmup-banner.pending { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
.warmup-banner.ready   { background: #d1fae5; color: #065f46; border: 1px solid #6ee7b7; }
.warmup-banner.failed  { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
.warmup-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.warmup-dot.pending { background: #f59e0b; animation: pulse-dot 1.4s infinite; }
.warmup-dot.ready   { background: #10b981; }
.warmup-dot.failed  { background: #ef4444; }
@keyframes pulse-dot { 0%, 100% { opacity: 1; } 50% { opacity: .3; } }

/* ---- sidebar ---- */
.sidebar-card {
  background: var(--block-background-fill);
  border: 1px solid var(--border-color-primary);
  border-radius: var(--block-radius);
  padding: .9rem 1rem;
}
.sidebar-card h3 {
  font-size: .8rem; font-weight: 600; margin: 0 0 .5rem;
  opacity: .7; text-transform: uppercase; letter-spacing: .05em;
}

/* ---- sources panel ---- */
.sources-header {
  font-size: .8rem; font-weight: 600; opacity: .6;
  text-transform: uppercase; letter-spacing: .05em; margin-bottom: .4rem;
}

/* ---- footer ---- */
.app-footer {
  text-align: center; opacity: .45; font-size: .78rem;
  padding: 1rem 0 .5rem;
  border-top: 1px solid var(--border-color-primary);
  margin-top: 1.25rem;
}
"""

THEME = gr.themes.Soft(
    primary_hue="violet",
    neutral_hue="slate",
    radius_size="md",
    font=gr.themes.GoogleFont("Inter"),
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
    ext_list = ", ".join(sorted(SUPPORTED_EXTENSIONS))

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
            '<p class="app-title">RAG Document Q&amp;A</p>'
            '<p class="app-subtitle">'
            "Ask questions about your documents, answered from their text only"
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

        with gr.Row(equal_height=False):

            # ---- left sidebar ----
            with gr.Column(scale=1, min_width=290):
                gr.HTML('<div class="sidebar-card"><h3>Document</h3></div>')
                file_input = gr.File(
                    label=f"Upload file ({ext_list})",
                    file_types=list(SUPPORTED_EXTENSIONS),
                    height=150,
                )
                doc_status = gr.Markdown(f"*{IDLE_STATUS}*")

                with gr.Accordion("How it works", open=False):
                    gr.Markdown(
                        "1. Document is split into overlapping chunks\n"
                        "2. Each chunk is embedded into a FAISS vector index\n"
                        "3. Your question retrieves the most similar chunks\n"
                        "4. A local LLM answers using only those chunks\n\n"
                        "*Sources are shown below the chat so you can verify every answer.*"
                    )

                with gr.Accordion("Tips", open=False):
                    gr.Markdown(
                        "- Ask specific questions, not vague ones\n"
                        "- If the answer is not in the document, the model says so\n"
                        "- Upload a new file at any time to switch documents\n"
                        f"- Max file size: {qa.settings.max_upload_mb} MB"
                    )

            # ---- right: chat + sources ----
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    height=420,
                    show_label=False,
                    type="messages",
                    placeholder=(
                        "<div style='text-align:center;opacity:.5;padding:3rem 1rem'>"
                        "<p style='font-size:2rem'>💬</p>"
                        "<p>Upload a document on the left, then ask anything about it.</p>"
                        "</div>"
                    ),
                )

                with gr.Row():
                    question = gr.Textbox(
                        show_label=False,
                        placeholder="Ask something about the document...",
                        scale=5,
                        container=False,
                        autofocus=True,
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)

                clear_btn = gr.Button("Clear chat", size="sm", variant="secondary")

                # Sources are always visible - no accordion hiding them
                gr.HTML('<p class="sources-header">Retrieved Passages</p>')
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
