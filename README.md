# RAG Document Q&A

[![CI](https://github.com/asanaliov/rag-document-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/asanaliov/rag-document-qa/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Upload a document and ask questions about it.
The app chunks the document, embeds it into a vector store, retrieves the most relevant passages for each question, and answers using only those passages.
The retrieved text is shown alongside every answer so you can verify it.

Everything runs locally - no API keys, no usage costs, documents never leave your machine.

## Quickstart (Docker - recommended)

**Step 1:** Install [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/).

**Step 2:** Clone and start the stack.

```bash
git clone https://github.com/asanaliov/rag-document-qa.git
cd rag-document-qa
docker compose up
```

**Step 3:** Open [http://localhost:7860](http://localhost:7860).

> **First start:** Docker downloads the Ollama image and pulls the model (~2 GB).
> This takes a few minutes once and is cached in a named volume for all future starts.
> After `docker compose up` shows `Uvicorn running`, the UI is ready.
> The model status banner in the UI shows when the LLM is loaded and ready to answer.

```bash
# Subsequent starts are fast - model is already cached
docker compose up

# Rebuild after code changes
docker compose up -d --build

# View live logs
docker compose logs -f app

# Stop
docker compose down          # keeps the model volume
docker compose down -v       # also removes the model volume (re-downloads next time)
```

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

| Component        | Technology                                 |
| ---------------- | ------------------------------------------ |
| Framework        | LangChain (LCEL)                           |
| Vector store     | FAISS                                      |
| Embeddings       | sentence-transformers (`all-MiniLM-L6-v2`) |
| LLM              | Ollama (`llama3.2:3b` by default)          |
| UI               | Gradio                                     |
| Document parsing | pypdf, LangChain TextLoader                |
| Tests / lint     | pytest, ruff, GitHub Actions               |
| Packaging        | Docker Compose (app + Ollama)              |

## Run without Docker

Requires Python 3.10+.

```bash
git clone https://github.com/asanaliov/rag-document-qa.git
cd rag-document-qa

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# CPU-only torch saves ~4 GB vs the default CUDA wheel (skip if you have a GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

**Install and start Ollama** (the local LLM runtime):

```bash
# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download the installer at https://ollama.com/download
```

**Pull the model** (one-time, ~2 GB download):

```bash
ollama pull llama3.2:3b
```

**Start the app:**

```bash
# Terminal 1 - keep this running
ollama serve

# Terminal 2
python app.py
```

Open [http://localhost:7860](http://localhost:7860).

> **Startup note:** When the app starts it sends a warm-up request to Ollama in the background so the model is already loaded into RAM before you ask your first question.
> Watch the status banner at the top of the UI - it turns green once the model is ready.
> The embedding model (~90 MB) downloads automatically on first use and is cached under `~/.cache/huggingface`.



## Usage

1. Upload a PDF, TXT, or MD file - it is chunked and indexed automatically
2. Wait for the model status banner to turn green (warm-up takes a few seconds on first start)
3. Ask a question and press **Enter** or click **Send**
4. The **Retrieved Passages** section below the chat shows which chunks the answer came from
5. Upload a new file at any time to switch documents

## Configuration

Copy `.env.example` to `.env` to override any default. Every setting is read once at startup by `rag/config.py`.

| Variable                       | Default                  | Purpose                       |
| ------------------------------ | ------------------------ | ----------------------------- |
| `OLLAMA_MODEL`                 | `llama3.2:3b`            | Generation model              |
| `OLLAMA_BASE_URL`              | `http://localhost:11434` | Ollama endpoint               |
| `EMBEDDING_MODEL`              | `all-MiniLM-L6-v2`       | Sentence-transformer model    |
| `EMBEDDING_DEVICE`             | `cpu`                    | Set to `cuda` to embed on GPU |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200`           | Splitter settings             |
| `RETRIEVAL_K`                  | `4`                      | Chunks retrieved per question |
| `MAX_UPLOAD_MB`                | `25`                     | Upload size limit             |
| `SERVER_HOST` / `SERVER_PORT`  | `127.0.0.1` / `7860`     | Bind address                  |
| `LOG_LEVEL`                    | `INFO`                   | Root log level                |

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
├── Dockerfile          # CPU-only image with the embedding weights baked in
├── compose.yaml        # App plus Ollama, model pulled on first start
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
- **Ollama as a separate container** — the model server has its own lifecycle and a volume that survives rebuilds, so changing application code never re-downloads several gigabytes of weights

## License

MIT
