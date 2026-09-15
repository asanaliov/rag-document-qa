"""Shared pytest fixtures.

pytest auto-discovers this file and makes every fixture in it available to
any test in this directory, with no import needed.
"""
import pytest


# Known content: short, and each sentence is on a distinct topic so we can
# assert which chunk a similarity search should rank first.
SAMPLE_TEXT = """The Zorblax protocol was invented in 1987 by Dr. Mira Chen.
It uses a 42-bit handshake to negotiate the session key.
The capital of the Zorblax Republic is Vantorra.
Vantorra has a population of 3.2 million people.
"""


@pytest.fixture
def sample_text_file(tmp_path):
    """Write SAMPLE_TEXT to a real .txt file and yield its path.

    `tmp_path` is a pytest built-in: a fresh empty directory per test,
    cleaned up automatically. Tests get a real file on disk without
    leaving anything behind.
    """
    path = tmp_path / "sample.txt"
    path.write_text(SAMPLE_TEXT, encoding="utf-8")
    return str(path)
