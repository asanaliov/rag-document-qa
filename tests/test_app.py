import json

import pytest
from fastapi.testclient import TestClient

from app import create_app
from rag.config import Settings
from tests.conftest import SAMPLE_TEXT


@pytest.fixture
def app(fake_embeddings, fake_llm):
    return create_app(Settings(max_upload_mb=1), embeddings=fake_embeddings, llm=fake_llm)


@pytest.fixture
def client(app):
    return TestClient(app)


SAMPLE_BYTES = SAMPLE_TEXT.encode()


def _upload(client, name="sample.txt", content=SAMPLE_BYTES):
    return client.post("/api/document", params={"name": name}, content=content)


def _ask(client, question, history=()):
    response = client.post("/api/ask", json={"question": question, "history": list(history)})
    return response, [json.loads(line) for line in response.text.splitlines()]


def test_serves_the_page(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "Document Q&amp;A" in response.text


def test_new_session_has_no_document(client):
    session = client.get("/api/session").json()

    assert session["document"] is None
    assert session["max_upload_mb"] == 1
    assert ".pdf" in session["extensions"]


def test_upload_indexes_the_document_and_the_session_remembers_it(client):
    response = _upload(client)

    assert response.status_code == 200
    assert response.json() == {"name": "sample.txt", "pages": None, "chunks": 1}
    assert client.get("/api/session").json()["document"]["name"] == "sample.txt"


def test_answer_streams_sources_then_text(client):
    _upload(client)

    response, events = _ask(client, "Who invented the Zorblax protocol?")

    assert response.status_code == 200
    assert [event["type"] for event in events[:2]] == ["sources", "text"]
    assert events[-1] == {"type": "done"}
    assert "Zorblax" in events[0]["sources"][0]["text"]
    text = "".join(event["text"] for event in events if event["type"] == "text")
    assert text == "Dr. Mira Chen invented it in 1987."


def test_asking_without_a_document_is_refused(client):
    response, _ = _ask(client, "Who invented it?")

    assert response.status_code == 409


def test_sessions_do_not_share_documents(app):
    first, second = TestClient(app), TestClient(app)
    _upload(first)

    assert second.get("/api/session").json()["document"] is None
    assert _ask(second, "Who invented it?")[0].status_code == 409


def test_unsupported_file_type_is_rejected_with_a_readable_message(client):
    response = _upload(client, name="report.docx")

    assert response.status_code == 422
    assert ".docx" in response.json()["detail"]


def test_oversized_upload_is_rejected(client):
    response = _upload(client, content=b"x" * (1024 * 1024 + 1))

    assert response.status_code == 413
    assert client.get("/api/session").json()["document"] is None


def test_removing_the_document_resets_the_session(client):
    _upload(client)

    assert client.delete("/api/document").status_code == 204
    assert client.get("/api/session").json()["document"] is None
