# 📄 RAG Document Q&A

Upload a document and ask questions about it in plain language. The app chunks the document, embeds it into a vector store, retrieves the most relevant passages for each question, and has a local LLM answer using only those passages — with the retrieved text shown alongside the answer so you can check it.

**Runs 100% free and offline.** Embeddings and the language model both run on your machine: no API keys, no usage costs, and your documents never leave your computer.

## How It Works

```
Document → Chunk → Embed → Store (FAISS)
                                  ↓
Question → Embed → Similarity Search → Top-K Chunks → LLM → Answer
```

1. **Load** — reads PDF, TXT, or Markdown files
2. **Chunk** — splits the document into overlapping passages (~1000 chars, 200 overlap) so nothing is lost at boundaries
3. **Embed** — turns each chunk into a vector with `all-MiniLM-L6-v2` (runs locally on CPU)
4. **Index** — stores the vectors in a FAISS index for fast similarity search
5. **Retrieve** — for each question, finds the 4 most similar chunks
6. **Generate** — sends those chunks plus the question to a local LLM via Ollama, which is instructed to answer from the context only

## Tech Stack

| Component | Technology |
|---|---|
| Framework | LangChain (LCEL) |
| Vector Store | FAISS |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| LLM | Ollama (`llama3.2:3b` by default) |
| UI | Gradio |
| Document Parsing | PyPDF, TextLoader |

## Setup

```bash
git clone https://github.com/asanaliov/rag-document-qa.git
cd rag-document-qa

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# install Ollama (local LLM runtime) and pull the default model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b
```

## Run

```bash
ollama serve      # skip if Ollama is already running as a service
python app.py
```

Open `http://localhost:7860`.

Want a sharper model and have the RAM? Pull one (`ollama pull llama3.1:8b`), copy `.env.example` to `.env`, and set `OLLAMA_MODEL=llama3.1:8b`.

## Usage

1. Upload a PDF, TXT, or MD file — it's chunked and indexed automatically (the first upload also loads the embedding model, so give it a moment)
2. Ask a question in the chat box and press Enter
3. Expand **Retrieved passages** under the chat to see exactly which chunks the answer was based on
4. Upload a new file at any time to start over with a different document

## Tests

```bash
pytest
```

## Project Structure

```
rag-document-qa/
├── app.py              # Gradio UI and application entry point
├── rag/
│   ├── loader.py       # Document loading and chunking
│   ├── store.py        # Embedding model and FAISS vector store
│   └── qa.py           # Prompt, LLM, and retrieval chain (LCEL)
├── tests/              # pytest suite
├── requirements.txt
├── .env.example
└── README.md
```

## Design Decisions

- **Fully local** — retrieval and generation both run on your own machine, so there's nothing to configure and nothing to pay for
- **Overlapping chunks** — 200 chars of overlap keeps sentences that straddle a split boundary intact
- **Retrieve once, then generate** — the chain retrieves the chunks a single time and passes them to the LLM, so the same passages that produced the answer can be shown to the user
- **Strict grounding** — the system prompt restricts the model to the provided context and tells it to say when the answer isn't there, which cuts down on hallucination
- **LCEL pipeline** — a small composable `prompt | llm | parser` chain rather than legacy LangChain abstractions

## License

MIT
