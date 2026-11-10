# RAG Document Q&A

[![CI](https://github.com/asanaliov/rag-document-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/asanaliov/rag-document-qa/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Upload a document and ask questions about it in plain language. The app chunks the document, embeds it into a vector store, retrieves the most relevant passages for each question, and has a local LLM answer using only those passages. Answers stream in as they are generated and cite the passages they rely on as `[1]`, `[2]`, with the retrieved text shown alongside so you can check every claim. Follow-up questions work the way they do in a conversation.

Retrieval and generation both run on your machine: no API keys, no usage costs, and documents never leave the host.

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
7. **Generate** — sends those chunks, numbered, and the question to a local LLM via Ollama, prompted to answer from the context only and cite the passage numbers; the answer is streamed to the UI

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

## Run with Docker

The compose stack runs the app and Ollama as two containers, pulls the model on
first start, and needs nothing installed but Docker.

```bash
git clone https://github.com/asanaliov/rag-document-qa.git
cd rag-document-qa
docker compose up
```

Open `http://localhost:7860`.

First start downloads the model (about 2 GB) into a named volume; later starts
reuse it. The embedding weights are baked into the image, so the app container
itself needs no network beyond Ollama. To use a different model, set
`OLLAMA_MODEL` in `.env` — compose reads it for both services.

```bash
docker compose up -d --build   # rebuild after code changes
docker compose logs -f app
docker compose down            # add -v to also drop the model volume
```

The app port is published on `127.0.0.1` only. Change it in `compose.yaml` to
expose the UI on your network. Ollama runs on the CPU by default; `compose.yaml`
has a commented GPU reservation block for machines with an NVIDIA runtime.

## Run without Docker

Needs Python 3.10 or newer.

```bash
git clone https://github.com/asanaliov/rag-document-qa.git
cd rag-document-qa

python3 -m venv .venv
source .venv/bin/activate

# Optional, and worth it on a machine without an NVIDIA GPU: the default torch
# wheel pulls in roughly 4 GB of CUDA libraries this app never uses.
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

# Ollama provides the local LLM runtime.
# On Windows, install it from https://ollama.com/download instead.
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b

ollama serve      # skip if Ollama already runs as a service
python app.py
```

Open `http://localhost:7860`.

The first upload downloads the embedding model (about 90 MB) and caches it under
`~/.cache/huggingface`, so it only happens once.

## Usage

1. Upload a PDF, TXT, or MD file — it is chunked and indexed automatically (the first upload also loads the embedding model, so give it a moment)
2. Ask a question and press Enter — the answer streams in, and **Stop** cancels it
3. The passages behind the answer sit in the margin beside it, numbered to match the `[n]` citations; point at a citation to highlight its passage, click it to read the passage in full
4. Ask follow-ups naturally; when one is rewritten, the margin shows what was actually searched for
5. Use **Replace document** at any time to start over on a different document

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
pytest          # 41 tests, all offline: no model downloads, no Ollama
ruff check .
```

The test suite substitutes deterministic fake embeddings and a fake chat model for the real ones, so the API and the full pipeline — loading, chunking, indexing, retrieval, follow-up rewriting, streaming and error handling — is covered without network access. CI runs the same two commands on Python 3.11 and 3.12.

## Project structure

```
rag-document-qa/
├── app.py              # FastAPI server: sessions, upload, streamed answers
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
- **Numbered context, cited answers** — passages reach the model as `[1]`, `[2]`, ... and the UI shows them under the same numbers, so a claim can be traced to its source
- **History rewrites the question, not the answer** — earlier turns are only used to make a follow-up standalone before retrieval; generation still sees nothing but the retrieved passages, so the conversation cannot leak ungrounded facts into an answer
- **One pipeline per browser session** — a cookie maps each browser to its own index, so two users never overwrite each other's document; the embedding weights are loaded once and shared. The browser owns the conversation and sends it with each question, so clearing it needs no server call
- **Answers stream as newline-delimited JSON** — one `fetch` reads the passages first and then the text as it is generated; cancelling the request stops the model
- **No frontend build, nothing fetched from outside** — the interface is three static files, and its two fonts (Literata and Instrument Sans, SIL Open Font License) are served by the app, so the page works offline like everything else
- **Strict grounding** — the system prompt confines the model to the provided context and gives it an explicit way to say the answer is not there
- **Typed errors** — `RagError` subclasses carry messages safe to display; anything else is logged with a traceback and surfaced as a generic failure, so internals never leak into the UI
- **In-memory index** — an index belongs to one uploaded document in one session and is discarded with it, so there is nothing to persist or invalidate
- **Lazy heavy imports** — torch and the embedding weights load on first use, keeping startup fast
- **Ollama as a separate container** — the model server has its own lifecycle and a volume that survives rebuilds, so changing application code never re-downloads several gigabytes of weights

## License

MIT
