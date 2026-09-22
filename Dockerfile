FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface

RUN useradd --create-home --uid 1000 app
WORKDIR /app

# CPU-only torch. The default wheel pulls in ~4 GB of CUDA libraries that this
# image never uses, since embeddings run on CPU.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app rag/ ./rag/
COPY --chown=app:app app.py .

USER app

# Bake the embedding weights into the image so the first upload does not stall
# on a download, and so the container needs no network beyond Ollama.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/')"

CMD ["python", "app.py"]
