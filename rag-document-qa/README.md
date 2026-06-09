# 📄 RAG Document Q&A

A Retrieval-Augmented Generation (RAG) application that lets you upload documents and ask questions about them using natural language. The system chunks the document, embeds it into a vector store, retrieves relevant passages, and generates accurate answers with a locally-run LLM.

**Runs 100% free and offline** — both the embeddings and the language model run locally, so there are no API keys and no usage costs.

## How It Works

```
Document → Chunk → Embed → Store (FAISS)
                                  ↓
Question → Embed → Similarity Search → Top-K Chunks → LLM → Answer
```

**The RAG pipeline in plain terms:**
1. **Load** — reads PDF, TXT, or Markdown files
2. **Chunk** — splits the document into overlapping passages (~1000 chars each) so no context is lost at boundaries
3. **Embed** — converts each chunk into a numerical vector using `all-MiniLM-L6-v2` (runs locally, no API needed)
4. **Index** — stores vectors in a FAISS index for fast similarity search
5. **Retrieve** — when you ask a question, it finds the 4 most relevant chunks by cosine similarity
6. **Generate** — sends the retrieved chunks + your question to a local LLM (via Ollama), which answers using only the provided context

## Tech Stack

| Component | Technology |
|---|---|
| Framework | LangChain (LCEL) |
| Vector Store | FAISS |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| LLM | Ollama (`llama3.2:3b`, runs locally) |
| UI | Gradio |
| Document Parsing | PyPDF, TextLoader |

## Setup

```bash
# Clone
git clone https://github.com/asanaliov/rag-document-qa.git
cd rag-document-qa

# Install dependencies
pip install -r requirements.txt

# Install Ollama (the local LLM runtime) and pull a model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b

# Run
python app.py
```

The app launches at `http://localhost:7860`. No API key required — everything runs locally.

> **Tip:** want a sharper model and have the RAM? Pull a bigger one
> (`ollama pull llama3.1:8b`) and set `OLLAMA_MODEL=llama3.1:8b` in your environment.

## Usage

1. Upload a PDF, TXT, or MD file
2. Click **Index Document** — this chunks and embeds the file
3. Ask any question about the document in the chat
4. The system retrieves relevant passages and generates an answer

## Project Structure

```
rag-document-qa/
├── app.py              # Gradio UI and application entry point
├── rag/
│   ├── loader.py       # Document loading and chunking
│   ├── store.py        # Embedding model and FAISS vector store
│   └── qa.py           # RAG chain with Claude (LCEL pipeline)
├── requirements.txt
├── .env.example
└── README.md
```

## Key Design Decisions

- **Fully local & free** — both retrieval (embeddings) and generation (Ollama) run on your own machine, so there are no API keys, no per-query costs, and your documents never leave your computer
- **Local embeddings** — `all-MiniLM-L6-v2` runs on CPU with no API key, keeping the retrieval step free and fast
- **Overlapping chunks** — 200-char overlap between chunks ensures context isn't lost at split boundaries
- **LCEL pipeline** — uses LangChain Expression Language for a clean, composable chain instead of legacy abstractions
- **Strict grounding** — the system prompt instructs Claude to answer only from the provided context, reducing hallucination

## License

MIT
