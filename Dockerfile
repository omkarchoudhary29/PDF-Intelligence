# ==============================================================================
# Enterprise Knowledge Navigator (EKN) — Unified Multi-Service Docker Image
# ==============================================================================
# Serves as the base image for backend, worker, and frontend services.
# ==============================================================================

FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies:
# - build-essential: C extension compilation if needed
# - curl: Container health checks
# - tesseract-ocr, tesseract-ocr-eng: Selective OCR for scanned PDF pages
# - libgl1, libglib2.0-0: PyMuPDF graphics dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application modules and source code
COPY backend/ ./backend/
COPY config/ ./config/
COPY frontend/ ./frontend/

# Create internal data directories
RUN mkdir -p /app/data/storage/uploads \
             /app/data/storage/processed \
             /app/data/storage/reports \
             /app/data/storage/temp \
             /app/data/chroma_db

# Create non-root system user for secure container execution
RUN useradd -u 1000 -m appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose ports: 8000 (FastAPI), 8501 (Streamlit)
EXPOSE 8000 8501

# Default command: FastAPI backend server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
