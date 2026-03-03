# =============================================================
# rag-pipeline Dockerfile
# =============================================================
# Builds a container for the RAG pipeline and Gradio demo UI.
#
# Build:
#   docker build -t rag-pipeline .
#
# Run interactive mode:
#   docker run --env-file .env -v $(pwd)/data:/app/data rag-pipeline
#
# Run single query:
#   docker run --env-file .env -v $(pwd)/data:/app/data \
#     rag-pipeline python src/pipeline.py --query "What is the PTO policy?"
#
# Run Gradio UI:
#   docker run --env-file .env -v $(pwd)/data:/app/data \
#     -p 7860:7860 rag-pipeline python ui/app.py
#
# Notes:
#   - Mount ./data as a volume to persist the Chroma index across runs
#   - Pass API keys via --env-file .env (never bake them into the image)
# =============================================================

# Use official Python slim image for a smaller footprint
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies required by PDF processing libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpoppler-cpp-dev \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY ui/ ./ui/
COPY evaluation/ ./evaluation/
COPY tests/ ./tests/

# Create data directories (mounted as volumes at runtime)
RUN mkdir -p data/raw data/processed data/chroma

# Expose Gradio port
EXPOSE 7860

# Default command — interactive pipeline mode
# Override at runtime for other modes (see usage above)
CMD ["python", "src/pipeline.py"]
