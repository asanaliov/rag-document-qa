"""Web server for the document question-answering pipeline."""
from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from rag.config import Settings
from rag.errors import RagError
from rag.loader import SUPPORTED_EXTENSIONS
from rag.pipeline import DocumentQA

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"
SESSION_COOKIE = "sid"
MAX_SESSIONS = 32
UNEXPECTED_ERROR = "Something went wrong. Check the server logs for details."


@dataclass
class Session:
    qa: DocumentQA
    document: dict | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    # Earlier (question, answer) turns. The browser owns the conversation, so
    # clearing it or reloading the page needs no server call.
    history: list[tuple[str, str]] = Field(default=[], max_length=50)


def _event(**fields) -> str:
    return json.dumps(fields) + "\n"


def answer_events(qa: DocumentQA, question: str, history: list[tuple[str, str]]) -> Iterator[str]:
    """Stream an answer as newline-delimited JSON: the sources once, then text as it arrives."""
    sent = 0
    try:
        for answer in qa.stream(question, history):
            if sent == 0:
                yield _event(
                    type="sources",
                    query=answer.query,
                    sources=[
                        {"text": doc.page_content, "page": doc.metadata.get("page")}
                        for doc in answer.sources
                    ],
                )
            yield _event(type="text", text=answer.text[sent:])
            sent = len(answer.text)
        yield _event(type="done")
    except RagError as exc:
        yield _event(type="error", message=str(exc))
    except Exception:
        logger.exception("Answering failed")
        yield _event(type="error", message=UNEXPECTED_ERROR)


def create_app(
    settings: Settings,
    embeddings: Embeddings | None = None,
    llm: BaseChatModel | None = None,
) -> FastAPI:
    app = FastAPI(title="RAG Document Q&A", docs_url=None, redoc_url=None)
    # One pipeline per browser, so two tabs never overwrite each other's document.
    # The least recently used session is dropped once the limit is reached.
    sessions: OrderedDict[str, Session] = OrderedDict()

    def find_session(request: Request) -> Session | None:
        sid = request.cookies.get(SESSION_COOKIE)
        if sid in sessions:
            sessions.move_to_end(sid)
            return sessions[sid]
        return None

    def open_session(request: Request, response: Response) -> Session:
        session = find_session(request)
        if session is None:
            sid = uuid.uuid4().hex
            session = sessions[sid] = Session(DocumentQA(settings, embeddings, llm))
            response.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax")
            if len(sessions) > MAX_SESSIONS:
                sessions.popitem(last=False)
        return session

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/session")
    async def get_session(request: Request) -> dict:
        session = find_session(request)
        return {
            "document": session.document if session else None,
            "model": settings.ollama_model,
            "max_upload_mb": settings.max_upload_mb,
            "extensions": SUPPORTED_EXTENSIONS,
        }

    @app.post("/api/document")
    async def upload_document(request: Request, response: Response, name: str) -> dict:
        """Index the file sent as the raw request body, replacing the session's document."""
        session = open_session(request, response)
        session.document = None
        session.qa.reset()

        suffix = Path(name).suffix.lower()
        limit = settings.max_upload_mb * 1024 * 1024
        size = 0
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            path = tmp.name
            try:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(
                            413, f"The file exceeds the {settings.max_upload_mb} MB limit."
                        )
                    tmp.write(chunk)
                tmp.flush()
                result = await run_in_threadpool(session.qa.index, path)
            except RagError as exc:
                raise HTTPException(422, str(exc)) from exc
            except HTTPException:
                raise
            except Exception as exc:
                logger.exception("Indexing failed for %s", name)
                raise HTTPException(500, UNEXPECTED_ERROR) from exc
            finally:
                os.unlink(path)

        session.document = {
            "name": name,
            # Only a PDF has real pages; a text file loads as one document.
            "pages": result.documents if suffix == ".pdf" else None,
            "chunks": result.chunks,
        }
        return session.document

    @app.delete("/api/document", status_code=204)
    async def remove_document(request: Request) -> None:
        session = find_session(request)
        if session:
            session.document = None
            session.qa.reset()

    @app.post("/api/ask")
    async def ask(request: Request, body: AskRequest) -> StreamingResponse:
        session = find_session(request)
        if session is None or not session.qa.is_ready:
            raise HTTPException(409, "Add a document first.")
        question = body.question.strip()
        if not question:
            raise HTTPException(422, "The question is empty.")
        return StreamingResponse(
            answer_events(session.qa, question, body.history),
            media_type="application/x-ndjson",
        )

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app


def main() -> None:
    load_dotenv()
    settings = Settings.from_env()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    # The Ollama client calls through httpx, which logs every request at INFO.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    uvicorn.run(
        create_app(settings),
        host=settings.server_host,
        port=settings.server_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
