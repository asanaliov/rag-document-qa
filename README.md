# RAG Document Q&A

[![CI](https://github.com/asanaliov/rag-document-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/asanaliov/rag-document-qa/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Upload a document and ask questions about it in plain language. The app chunks the document, embeds it into a vector store, retrieves the most relevant passages for each question, and has a local LLM answer using only those passages — with the retrieved text shown alongside the answer so you can check it.

Retrieval and generation both run on your machine: no API keys, no usage costs, and documents never leave the host.

## How it works

```
Document -> Chunk -> Embed -> FAISS index
                                   |
Question -> Embed -> Similarity search -> Top-k chunks -> LLM -> Answer
```

1. **Load** — reads PDF, TXT, or Markdown, rejecting oversized files and documents with no extractable text
2. **Chunk** — splits into overlapping passages (1000 chars, 200 overlap) so nothing is lost at a boundary
3. **Embed** — turns each chunk into a vector with `all-MiniLM-L6-v2` on CPU
4. **Index** — stores the vectors in an in-memory FAISS index
5. **Retrieve** — finds the k most similar chunks for the question
6. **Generate** — sends those chunks and the question to a local LLM via Ollama, prompted to answer from the context only

## Tech stack

| Component | Technology |
|---|---|
| Framework | LangChain (LCEL) |
| Vector store | FAISS |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| LLM | Ollama (`llama3.2:3b` by default) |
| UI | Gradio |
| Document parsing | pypdf, LangChain TextLoader |
| Tests / lint | pytest, ruff, GitHub Actions |

## Setup

```bash
git clone https://github.com/asanaliov/rag-document-qa.git
cd rag-document-qa

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Ollama provides the local LLM runtime
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b
```

## Run

```bash
ollama serve      # skip if Ollama already runs as a service
python app.py
```

Open `http://localhost:7860`.

1. Upload a PDF, TXT, or MD file — it is chunked and indexed automatically (the first upload also loads the embedding model, so give it a moment)
2. Ask a question and press Enter
3. Expand **Retrieved passages** to see exactly which chunks the answer came from
4. Upload a new file at any time to start over on a different document

## Configuration

Copy `.env.example` to `.env` to override any default. Every setting is read once at startup by `rag/config.py`.

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2:3b` | Generation model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `EMBEDDING_DEVICE` | `cpu` | Set to `cuda` to embed on GPU |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Splitter settings |
| `RETRIEVAL_K` | `4` | Chunks retrieved per question |
| `MAX_UPLOAD_MB` | `25` | Upload size limit |
| `SERVER_HOST` / `SERVER_PORT` | `127.0.0.1` / `7860` | Bind address |
| `LOG_LEVEL` | `INFO` | Root log level |

## Development

```bash
pip install -r requirements-dev.txt
pytest          # 28 tests, all offline: no model downloads, no Ollama
ruff check .
```

The test suite substitutes deterministic fake embeddings and a fake chat model for the real ones, so the full pipeline — loading, chunking, indexing, retrieval, answering and error handling — is covered without network access. CI runs the same two commands on Python 3.11 and 3.12.

## Project structure

```
rag-document-qa/
├── app.py              # Gradio UI and entry point
├── rag/
│   ├── config.py       # Settings resolved from the environment
│   ├── errors.py       # Exceptions carrying user-safe messages
│   ├── loader.py       # Document loading, validation, chunking
│   ├── store.py        # Embedding model and FAISS index
│   ├── qa.py           # Prompt, LLM, LCEL chain
│   └── pipeline.py     # Stateful orchestration, UI-independent
├── tests/
└── .github/workflows/  # Lint and test on every push
```

## Design decisions

- **Pipeline separated from UI** — `DocumentQA` owns all state and knows nothing about Gradio, which is what makes the end-to-end tests possible without a browser or a server
- **Injected models** — embeddings and the LLM are constructor arguments, so tests swap in fakes and a future API-backed model is a one-line change
- **Retrieve once, then generate** — the retriever is kept out of the chain so the exact passages behind an answer can be shown to the user
- **Strict grounding** — the system prompt confines the model to the provided context and gives it an explicit way to say the answer is not there
- **Typed errors** — `RagError` subclasses carry messages safe to display; anything else is logged with a traceback and surfaced as a generic failure, so internals never leak into the UI
- **In-memory index** — an index belongs to one uploaded document and is discarded with it, so there is nothing to persist or invalidate
- **Lazy heavy imports** — torch and the embedding weights load on first use, keeping startup fast

## License

MIT
