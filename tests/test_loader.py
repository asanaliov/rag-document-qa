"""Tests for rag/loader.py — document loading and chunking."""
import pytest

from rag.loader import load_document, chunk_documents


# ---------------------------------------------------------------------------
# WORKED EXAMPLE — read this one carefully, then write the three below it.
# ---------------------------------------------------------------------------

def test_loads_txt_file(sample_text_file):
    """A .txt file loads into exactly one Document with the right text."""
    # `sample_text_file` was NOT passed by us. pytest saw the parameter name,
    # found the fixture of that name in conftest.py, ran it, and handed us
    # back its return value: the path to a real .txt file on disk.

    docs = load_document(sample_text_file)

    # TextLoader reads the whole file as a single Document, so: exactly one.
    assert len(docs) == 1

    # A Document is an object with .page_content (the text) and .metadata (a dict).
    # We assert on a substring rather than the whole string, so the test doesn't
    # break every time someone edits an unrelated line of the fixture text.
    assert "Zorblax" in docs[0].page_content


# ---------------------------------------------------------------------------
# YOUR TURN — delete the @pytest.mark.skip line from each one as you write it.
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="you write this one")
def test_loads_md_file(tmp_path):
    """A .md file loads the same way a .txt does.

    There's no `sample_md_file` fixture, so use `tmp_path` directly — it's a
    pytest built-in giving you a fresh empty directory, unique to this test.

    Three steps:
      1. path = tmp_path / "notes.md"          <- `/` joins paths, like os.path.join
      2. path.write_text("...", encoding="utf-8")
      3. load it (load_document wants a str, so wrap: str(path)) and assert on it
    """


@pytest.mark.skip(reason="you write this one")
def test_rejects_unsupported_extension(tmp_path):
    """A .docx file raises ValueError naming the supported types.

    To assert that something raises, you wrap it in a `with` block:

        with pytest.raises(ValueError) as excinfo:
            load_document(str(path))

    If the call raises ValueError, the test passes. If it raises nothing, or
    raises some *other* exception, the test fails. `excinfo.value` then holds
    the exception object, so `str(excinfo.value)` is the message text.

    Assert the message mentions ".docx". Do NOT assert the full message:
    look at SUPPORTED_EXTENSIONS in rag/loader.py:6 and ask yourself what
    type it is and whether ', '.join() over it gives a stable order.
    """


@pytest.mark.skip(reason="you write this one")
def test_chunks_overlap():
    """Consecutive chunks share text — proving chunk_overlap actually works.

    You don't need a file here. chunk_documents takes Documents, and you can
    build one directly:

        from langchain_core.documents import Document
        doc = Document(page_content="...")

    Make the text long enough to force a split (default chunk_size is 1000,
    so aim for several thousand chars) and make it *non-repeating*, or you
    can't tell real overlap from coincidence. A trick: build it from numbered
    sentences so every position is unique, e.g.

        text = " ".join(f"Sentence number {i} is unique." for i in range(400))

    Then chunk it and assert the property. The tail of chunks[0] should appear
    inside chunks[1]. Getting the exact overlap length right is fiddly and
    brittle -- instead take a modest slice from the END of chunks[0]
    (say the last 50 chars) and assert it appears in chunks[1].

    Also assert len(chunks) > 1, or the overlap assertion is vacuous.
    """
