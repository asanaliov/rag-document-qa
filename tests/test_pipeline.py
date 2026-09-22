import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag.config import Settings
from rag.errors import GenerationError
from rag.pipeline import DocumentQA, _is_connection_failure


class ConnectError(Exception):
    """Stands in for httpx.ConnectError, which the pipeline matches by name."""


class _ConnectionFailingChatModel(FakeListChatModel):
    def _generate(self, *args, **kwargs):
        raise ConnectError("connection refused")


class _FailingChatModel(FakeListChatModel):
    def _generate(self, *args, **kwargs):
        raise ValueError("model exploded")


def _pipeline(llm, fake_embeddings, **overrides):
    return DocumentQA(Settings(**overrides), embeddings=fake_embeddings, llm=llm)


def test_indexing_then_answering(sample_text_file, fake_embeddings, fake_llm):
    qa = _pipeline(fake_llm, fake_embeddings)
    assert not qa.is_ready

    result = qa.index(sample_text_file)

    assert qa.is_ready
    assert result.documents == 1
    assert result.chunks >= 1

    answer = qa.ask("Who invented the Zorblax protocol?")

    assert answer.text == "Dr. Mira Chen invented it in 1987."
    assert answer.sources


def test_asking_before_indexing_is_a_programming_error(fake_embeddings, fake_llm):
    with pytest.raises(RuntimeError):
        _pipeline(fake_llm, fake_embeddings).ask("anything?")


def test_failed_indexing_leaves_no_stale_chain(
    sample_text_file, tmp_path, fake_embeddings, fake_llm
):
    qa = _pipeline(fake_llm, fake_embeddings)
    qa.index(sample_text_file)

    with pytest.raises(FileNotFoundError):
        qa.index(str(tmp_path / "missing.txt"))

    assert not qa.is_ready


def test_retrieval_respects_configured_k(sample_text_file, fake_embeddings, fake_llm):
    qa = _pipeline(fake_llm, fake_embeddings, chunk_size=60, chunk_overlap=10, retrieval_k=2)
    qa.index(sample_text_file)

    assert len(qa.ask("What is the capital?").sources) == 2


def test_model_failure_is_reported_as_a_generation_error(sample_text_file, fake_embeddings):
    qa = _pipeline(_FailingChatModel(responses=[""]), fake_embeddings)
    qa.index(sample_text_file)

    with pytest.raises(GenerationError, match="model exploded"):
        qa.ask("Who invented it?")


def test_unreachable_ollama_produces_an_actionable_message(sample_text_file, fake_embeddings):
    qa = _pipeline(_ConnectionFailingChatModel(responses=[""]), fake_embeddings)
    qa.index(sample_text_file)

    with pytest.raises(GenerationError, match="ollama serve"):
        qa.ask("Who invented it?")


def test_connection_failure_is_detected_through_the_cause_chain():
    try:
        try:
            raise ConnectError("refused")
        except ConnectError as exc:
            raise RuntimeError("chain invoke failed") from exc
    except RuntimeError as wrapped:
        assert _is_connection_failure(wrapped)


def test_unrelated_failure_is_not_mistaken_for_a_connection_problem():
    assert not _is_connection_failure(ValueError("bad prompt"))
