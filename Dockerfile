FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV TERM=xterm-256color
ENV SHELL=/bin/bash

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy dependency list
COPY requirements.txt .

# NOTE: The base image already provides a CUDA-enabled PyTorch build.
# If `requirements.txt` includes `torch` or related packages, consider
# removing those entries to avoid reinstalling a different wheel.

# Install Python dependencies (do not force reinstalling torch here).
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir --disable-pip-version-check -r requirements.txt || true && \
    python -m pip install --no-cache-dir jupyter jupyterlab ipywidgets

# Copy project files
COPY . .

# Expose Jupyter (8888) and TensorBoard (6006)
EXPOSE 8888 6006

# Ensure exp_result exists and start Jupyter Lab
CMD ["bash", "-lc", "mkdir -p /app/exp_result && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.token='' --ServerApp.password=''"]