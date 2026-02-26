# ── Build stage ─────────────────────────────────────────────────────────────
# Override CUDA_VERSION at build time to match your host driver.
# E.g.: CUDA_VERSION=12.8.1 TORCH_CUDA_VERSION=cu128 docker compose up --build
ARG CUDA_VERSION=13.0.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu22.04

# System packages required by torch / Pillow / faiss-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        build-essential \
        libglib2.0-0 \
        libgl1 \
    && ln -sf python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/search-server

# Install torch + torchvision together from the PyTorch wheel index.
# TORCH_CUDA_VERSION must match CUDA_VERSION (e.g. cu130 for CUDA 13.0.x).
# Both must come from the same index to guarantee ABI compatibility; installing
# them separately (or letting open-clip-torch pull torchvision from PyPI) causes
# a "operator torchvision::nms does not exist" RuntimeError at import time.
ARG TORCH_CUDA_VERSION=cu130
RUN python3.11 -m pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/${TORCH_CUDA_VERSION}

# Install remaining Python dependencies
COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

EXPOSE 5000

# DATASET_NAME is optional.  When set, CLIP is pre-loaded at startup for that
# dataset.  When empty / unset, CLIP is disabled and only text-based providers
# (BM25, OpenAI) are available.
CMD ["sh", "-c", "python3.11 app.py ${DATASET_NAME}"]
