FROM python:3.11-slim

WORKDIR /app

# System deps for faiss + pythainlp
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data files
# Pre-download model into image (avoids cold-start download)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

COPY *.py ./
COPY thai_recipes_minilm.index ./
COPY child_texts.pkl ./
COPY child_meta.pkl ./
COPY menus_docstore.jsonl ./

ENV PORT=7860

EXPOSE 7860

CMD ["python", "main.py"]
