# ==============================================================================
# Enterprise Knowledge Navigator (EKN) — Unified Production Container Image
# ==============================================================================
# Packages the complete platform:
# - FastAPI internal REST API (loopback only)
# - Background Ingestion Worker daemon
# - Streamlit Enterprise UI frontend
# ==============================================================================

FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/model_cache \
    TRANSFORMERS_CACHE=/app/model_cache \
    SENTENCE_TRANSFORMERS_HOME=/app/model_cache

# Install system dependencies:
# - build-essential: C extension compilation
# - curl: Container health checks
# - tesseract-ocr, tesseract-ocr-eng: Selective OCR for scanned PDF pages
# - libgl1, libglib2.0-0: PyMuPDF graphics dependencies
# - gosu: Safe privilege step-down from root to appuser after volume mounting
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gosu \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for layer caching.
# Railway runs this workload on CPU: install the official CPU-only PyTorch wheel
# before Sentence Transformers so pip reuses it instead of pulling CUDA/NVIDIA wheels.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    python -c "import torch; assert torch.version.cuda is None, f'Expected CPU-only PyTorch, got CUDA={torch.version.cuda}'; print('Verified CPU-only PyTorch', torch.__version__)"

# Create deterministic model cache directory and bake dense embedding model into image
RUN mkdir -p /app/model_cache && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application modules, configuration, scripts, and Streamlit theme
COPY backend/ ./backend/
COPY config/ ./config/
COPY frontend/ ./frontend/
COPY .streamlit/ ./.streamlit/
COPY scripts/ ./scripts/

# Create internal runtime storage directories
RUN mkdir -p /app/data/storage/uploads \
             /app/data/storage/processed \
             /app/data/storage/reports \
             /app/data/storage/temp \
             /app/data/chroma_db

# Create non-root system user for secure container execution
RUN useradd -u 1000 -m appuser && \
    chown -R appuser:appuser /app && \
    chmod +x /app/scripts/docker_entrypoint.sh /app/scripts/start_production.sh /app/scripts/start_production.py

# Expose ports: 8000 (internal FastAPI), 8501 (Streamlit public UI)
EXPOSE 8000 8501

# Entrypoint fixes volume permissions dynamically and drops privileges to appuser
ENTRYPOINT ["/app/scripts/docker_entrypoint.sh"]

# Default command: Multi-process production supervisor
CMD ["python", "scripts/start_production.py"]
