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
.gradio-container {
  max-width: 1480px !important;
  margin: 0 auto !important;
  background: #f5f3ed !important;
  color: #202622 !important;
}
body { background: #f5f3ed !important; }
footer { display: none !important; }
#document-app { --ink: #202622; --muted: #727a73; --line: #deded3; --paper: #fbfaf6; --green: #28614d; }
#document-app .app-topbar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 1.25rem 0; border-bottom: 1px solid var(--line);
}
#document-app .brand { display: flex; align-items: center; gap: .7rem; color: var(--ink); font-size: .95rem; font-weight: 650; letter-spacing: -.02em; }
#document-app .brand-mark { display: grid; place-items: center; width: 2rem; height: 2rem; border-radius: .65rem; color: #f7f5ef; background: var(--green); font-family: Georgia, serif; font-size: 1.2rem; }
#document-app .privacy-note { color: var(--muted); font-size: .76rem; letter-spacing: .08em; text-transform: uppercase; }
#document-app .hero { padding: clamp(2rem, 5vw, 4.5rem) 0 2rem; }
#document-app .eyebrow, #document-app .section-kicker { color: var(--green); font-size: .72rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; }
#document-app .hero h1 { max-width: 850px; margin: .65rem 0 .8rem; color: var(--ink); font-family: Georgia, 'Times New Roman', serif; font-size: clamp(2.65rem, 6vw, 5.25rem); font-weight: 500; letter-spacing: -.055em; line-height: .99; text-wrap: balance; }
#document-app .hero p { max-width: 620px; margin: 0; color: var(--muted); font-size: 1.04rem; line-height: 1.7; }
#document-app .warmup-banner { display: flex; align-items: center; gap: .7rem; padding: .85rem 1rem; border: 1px solid var(--line); border-radius: .8rem; background: rgba(251,250,246,.76); color: var(--muted); font-size: .86rem; margin: 0 0 1.2rem; }
#document-app .warmup-banner.pending { border-color: #e8d4aa; color: #765a28; }
#document-app .warmup-banner.ready { border-color: #c7d9ca; color: #315d46; }
#document-app .warmup-banner.failed { border-color: #e7c4ba; color: #88483a; }
#document-app .warmup-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
#document-app .warmup-dot.pending { background: #bf8a31; animation: pulse-dot 1.4s infinite; }
#document-app .warmup-dot.ready { background: #42805d; }
#document-app .warmup-dot.failed { background: #b65b48; }
@keyframes pulse-dot { 0%, 100% { opacity: 1; } 50% { opacity: .3; } }
#document-app #ingest-card { padding: 1.25rem; margin-bottom: 1.25rem; border: 1px solid var(--line); border-radius: 1rem; background: var(--paper); box-shadow: 0 10px 35px rgba(49, 57, 48, .035); }
#document-app .upload-copy { padding: .25rem .4rem; }
#document-app .upload-copy h2 { margin: .5rem 0 .45rem; font-family: Georgia, 'Times New Roman', serif; font-size: 1.45rem; font-weight: 500; letter-spacing: -.025em; }
#document-app .upload-copy p { max-width: 28rem; margin: 0; color: var(--muted); font-size: .88rem; line-height: 1.6; }
#document-app #ingest-card .document-status { margin-top: .75rem; color: var(--green); font-size: .9rem; }
#document-app #ingest-card .file-preview { border-color: var(--line) !important; border-radius: .75rem !important; background: #f6f5ef !important; }
#document-app #conversation { padding: 1.3rem; border: 1px solid var(--line); border-radius: 1rem; background: var(--paper); box-shadow: 0 12px 42px rgba(49, 57, 48, .045); }
#document-app .conversation-heading { display: flex; align-items: center; justify-content: space-between; gap: 1rem; padding: .1rem .25rem 1rem; }
#document-app .conversation-heading h2 { margin: .35rem 0 0; color: var(--ink); font-family: Georgia, 'Times New Roman', serif; font-size: 1.6rem; font-weight: 500; letter-spacing: -.03em; }
#document-app #chat-window { border: 0 !important; border-radius: .7rem !important; background: #f5f3ed !important; }
#document-app #chat-window .bubble-wrap { padding: 1.1rem; }
#document-app #composer { margin-top: .9rem; padding: .45rem; border: 1px solid #d8d9ce; border-radius: .85rem; background: #fffefa; }
#document-app #composer textarea { border: 0 !important; background: transparent !important; box-shadow: none !important; }
#document-app #send-button { min-height: 2.8rem; border-radius: .65rem !important; background: var(--green) !important; color: white !important; }
#document-app #send-button:hover { background: #204f3e !important; }
#document-app .composer-hint { padding: .35rem .4rem 0; color: var(--muted); font-size: .76rem; }
#document-app .evidence-heading { display: flex; align-items: baseline; gap: .7rem; margin: 1.5rem .25rem .65rem; }
#document-app .evidence-heading h3 { margin: 0; color: var(--ink); font-size: .88rem; font-weight: 650; }
#document-app .evidence-heading span { color: var(--muted); font-size: .78rem; }
#document-app #sources-panel { padding: .75rem 1rem; border-left: 2px solid #9ab5a0; border-radius: 0 .65rem .65rem 0; background: #f5f3ed; color: #4d554f; font-size: .88rem; line-height: 1.7; }
#document-app #sources-panel blockquote { margin: .4rem 0 1rem; padding-left: .8rem; border-left: 1px solid #c9cec3; color: #646b64; }
#document-app .app-footer { display: flex; flex-wrap: wrap; justify-content: space-between; gap: .5rem 1rem; padding: 1.1rem .2rem 2rem; color: var(--muted); font-size: .74rem; }
#document-app button, #document-app input, #document-app textarea { transition: background-color .18s ease, border-color .18s ease, transform .18s ease; }
#document-app button:active { transform: translateY(1px); }
#document-app button:focus-visible, #document-app input:focus-visible, #document-app textarea:focus-visible { outline: 3px solid #74927f; outline-offset: 2px; }
@media (max-width: 700px) {
  #document-app .app-topbar { padding: 1rem 0; }
  #document-app .privacy-note { max-width: 10rem; text-align: right; line-height: 1.5; }
  #document-app .hero { padding: 2.2rem 0 1.5rem; }
  #document-app #ingest-card, #document-app #conversation { padding: .8rem; }
  #document-app .conversation-heading { align-items: flex-start; }
}
"""

THEME = gr.themes.Soft(
    primary_hue="green",
    neutral_hue="stone",
    radius_size="lg",
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
        title="Folio - document questions, with sources",
        analytics_enabled=False,
        theme=THEME,
        css=CSS,
        elem_id="document-app",
    ) as ui:
        gr.HTML(
            '<header class="app-topbar">'
            '<div class="brand"><span class="brand-mark">f</span><span>folio / document studio</span></div>'
            '<span class="privacy-note">Private by default · runs on your machine</span>'
            '</header>'
            '<section class="hero">'
            '<div class="eyebrow">A quieter way to read</div>'
            '<h1>The answer is already in your files.</h1>'
            '<p>Bring in a document, ask a plain question, and follow every answer back to the text that supports it.</p>'
            '</section>'
        )

        warmup_banner = gr.HTML(warmup_html(warmup_state[0], model_name))
        warmup_timer = gr.Timer(value=2, active=True)

        with gr.Column(elem_id="ingest-card"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=2, min_width=230, elem_classes=["upload-copy"]):
                    gr.HTML(
                        '<div class="section-kicker">01 / Start with a file</div>'
                        '<h2>What are we reading?</h2>'
                        '<p>PDF, text, or Markdown. Your document is processed locally and can be replaced at any time.</p>'
                    )
                    doc_status = gr.Markdown(f"*{IDLE_STATUS}*", elem_classes=["document-status"])
                file_input = gr.File(
                    label=f"Drop a file here or browse · up to {qa.settings.max_upload_mb} MB",
                    file_types=list(SUPPORTED_EXTENSIONS),
                    height=135,
                    scale=3,
                )

        with gr.Column(elem_id="conversation"):
            with gr.Row(elem_classes=["conversation-heading"]):
                gr.HTML(
                    '<div><div class="section-kicker">02 / Explore the text</div>'
                    '<h2>Your conversation</h2></div>'
                )
                clear_btn = gr.Button("Clear", size="sm", variant="secondary")

            chatbot = gr.Chatbot(
                height=480,
                show_label=False,
                type="messages",
                elem_id="chat-window",
                placeholder=(
                    "<div style='text-align:center;padding:5rem 1rem;color:#727a73'>"
                    "<p style='font-family:Georgia,serif;font-size:1.45rem;color:#202622'>A good question opens a document.</p>"
                    "<p>Add a file above to begin. Try asking for a summary, a date, or a specific detail.</p>"
                    "</div>"
                ),
            )

            with gr.Row(elem_id="composer"):
                question = gr.Textbox(
                    show_label=False,
                    placeholder="Ask about a detail, a claim, or the whole document...",
                    scale=6,
                    container=False,
                    autofocus=True,
                )
                submit_btn = gr.Button("Ask  ↗", variant="primary", scale=1, min_width=100, elem_id="send-button")
            gr.HTML('<div class="composer-hint">Press Enter to ask · Answers are grounded in the passages below</div>')

            gr.HTML(
                '<div class="evidence-heading"><h3>Text behind the answer</h3>'
                '<span>Retrieved passages, ready to check</span></div>'
            )
            sources = gr.Markdown(f"*{NO_SOURCES}*", elem_id="sources-panel")

        gr.HTML(
            '<footer class="app-footer">'
            '<span>Local document workspace</span>'
            f'<span>{model_name} via Ollama · {embed_name} embeddings · FAISS retrieval</span>'
            '</footer>'
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
