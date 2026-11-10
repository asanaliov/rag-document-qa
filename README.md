# RAG Document Q&A

[![CI](https://github.com/asanaliov/rag-document-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/asanaliov/rag-document-qa/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Upload a document and ask questions about it.
The app chunks the document, embeds it into a vector store, retrieves the most relevant passages for each question, and answers using only those passages.
Answers stream in as they are generated and cite the passages they rely on as `[1]`, `[2]`, with the passages shown beside the answer so you can check every claim.
Follow-up questions work the way they do in a conversation.

Everything runs locally - no API keys, no usage costs, documents never leave your machine.

## Quickstart (Docker)

Install [Docker Desktop](https://docs.docker.com/get-docker/) with Compose, then run:

```bash
git clone https://github.com/asanaliov/rag-document-qa.git
cd rag-document-qa
docker compose up -d
```

On the first run, Docker downloads the app image and the default Ollama model (about 2 GB).
The app starts when the model download finishes, and the top bar shows when the model is ready.

On later runs, start the app with `docker compose up -d` and stop it with `docker compose down`.
Both the model and downloaded embedding files stay cached between starts.
To see startup progress, run `docker compose logs -f app` and press Ctrl+C to stop following logs.
To rebuild the app after changing code, run `docker compose up -d --build`.
Avoid `docker compose down -v` unless you want to delete the cached model and download it again.

## How it works

```
Document -> Chunk -> Embed -> FAISS index
                                   |
Question -> (Rewrite if follow-up) -> Embed -> Similarity search -> Top-k chunks -> LLM -> Cited answer
```

1. **Load** — reads PDF, TXT, or Markdown, rejecting oversized files and documents with no extractable text
2. **Chunk** — splits into overlapping passages (1000 chars, 200 overlap) so nothing is lost at a boundary
3. **Embed** — turns each chunk into a vector with `all-MiniLM-L6-v2` on CPU
4. **Index** — stores the vectors in an in-memory FAISS index
5. **Rewrite** — a follow-up such as "when was she born?" is first rewritten by the LLM into a standalone question, using the last few turns; a first question skips this step
6. **Retrieve** — finds the k most similar chunks for the question
7. **Generate** — sends those chunks, numbered, and the question to a local LLM via Ollama, prompted to answer from the context only and cite the passage numbers; the answer is streamed to the page

## Tech stack

| Component        | Technology                                 |
| ---------------- | ------------------------------------------ |
| Framework        | LangChain (LCEL)                           |
| Vector store     | FAISS                                      |
| Embeddings       | sentence-transformers (`all-MiniLM-L6-v2`) |
| LLM              | Ollama (`llama3.2:3b` by default)          |
| Server           | FastAPI, uvicorn                           |
| UI               | Plain HTML, CSS and JavaScript (no build)  |
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
> The top bar says when the model is loading and clears once it is ready.
> The embedding model (~90 MB) downloads automatically on first use and is cached under `~/.cache/huggingface`.

## Usage

1. Choose a PDF, TXT, or MD file, or drop it anywhere on the page - it is chunked and indexed automatically
2. Ask a question and press **Enter** - the answer streams in, and **Stop** cancels it
3. The passages behind the answer sit in the margin beside it, numbered to match the `[n]` citations; point at a citation to highlight its passage, click it to read the passage in full
4. Ask follow-ups naturally; when one is rewritten, the margin shows what was actually searched for
5. Use **Replace document** at any time to switch documents

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
pytest          # 42 tests, all offline: no model downloads, no Ollama
ruff check .
```

The test suite substitutes deterministic fake embeddings and a fake chat model for the real ones, so the API and the full pipeline — loading, chunking, indexing, retrieval, follow-up rewriting, streaming and error handling — is covered without network access. CI runs the same two commands on Python 3.11 and 3.12.

## Project structure

```
rag-document-qa/
├── app.py              # FastAPI server: sessions, upload, streamed answers, model warm-up
├── web/                # The interface: index.html, app.css, app.js, fonts
├── rag/
│   ├── config.py       # Settings resolved from the environment
│   ├── errors.py       # Exceptions carrying user-safe messages
│   ├── loader.py       # Document loading, validation, chunking
│   ├── store.py        # Embedding model and FAISS index
│   ├── qa.py           # Prompts, LLM, LCEL chains
│   └── pipeline.py     # Stateful orchestration, UI-independent
├── tests/
├── Dockerfile          # CPU-only image with the embedding weights baked in
├── compose.yaml        # App plus Ollama, model pulled on first start
└── .github/workflows/  # Lint and test on every push
```

## Design decisions

- **Pipeline separated from UI** — `DocumentQA` owns the document state and knows nothing about HTTP, which is what made it possible to replace the original Gradio front end without touching the pipeline
- **Injected models** — embeddings and the LLM are constructor arguments, so tests swap in fakes and a future API-backed model is a one-line change
- **Retrieve once, then generate** — the retriever is kept out of the chain so the exact passages behind an answer can be shown to the user
- **Numbered context, cited answers** — passages reach the model as `[1]`, `[2]`, ... and the page shows them under the same numbers, so a claim can be traced to its source
- **History rewrites the question, not the answer** — earlier turns are only used to make a follow-up standalone before retrieval; generation still sees nothing but the retrieved passages, so the conversation cannot leak ungrounded facts into an answer
- **One pipeline per browser session** — a cookie maps each browser to its own index, so two users never overwrite each other's document; the embedding weights are loaded once and shared. The browser owns the conversation and sends it with each question, so clearing it needs no server call
- **Answers stream as newline-delimited JSON** — one `fetch` reads the passages first and then the text as it is generated; cancelling the request stops the model
- **Warm-up at startup** — a one-token request in a background thread loads the model into memory, so the first question does not pay the load time
- **No frontend build, nothing fetched from outside** — the interface is three static files, and its two fonts (Literata and Instrument Sans, SIL Open Font License) are served by the app, so the page works offline like everything else
- **Strict grounding** — the system prompt confines the model to the provided context and gives it an explicit way to say the answer is not there
- **Typed errors** — `RagError` subclasses carry messages safe to display; anything else is logged with a traceback and surfaced as a generic failure, so internals never leak into the UI
- **In-memory index** — an index belongs to one uploaded document in one session and is discarded with it, so there is nothing to persist or invalidate
- **Lazy heavy imports** — torch and the embedding weights load on first use, keeping startup fast
- **Ollama as a separate container** — the model server has its own lifecycle and a volume that survives rebuilds, so changing application code never re-downloads several gigabytes of weights

## License

MIT
