# ============================================
# Stage 1: GPU Build (with CUDA support)
# ============================================
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS gpu-stage

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV TERM=xterm-256color
ENV SHELL=/bin/bash

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --timeout=300 --retries=5 \
    torchcontrib \
    numpy \
    soundfile \
    tqdm \
    librosa \
    kagglehub \
    matplotlib \
    tensorboard \
    seaborn \
    pandas \
    jupyter \
    jupyterlab \
    ipywidgets

COPY . .

EXPOSE 8888

# Run as root (simpler for development)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=''", "--ServerApp.password=''"]

# ============================================
# Stage 2: CPU Build
# ============================================
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS cpu-stage

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV TERM=xterm-256color
ENV SHELL=/bin/bash
ENV CUDA_VISIBLE_DEVICES=""
ENV NVIDIA_VISIBLE_DEVICES=void

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --timeout=300 --retries=5 \
    torchcontrib \
    numpy \
    soundfile \
    tqdm \
    librosa \
    kagglehub \
    matplotlib \
    tensorboard \
    seaborn \
    pandas \
    jupyter \
    jupyterlab \
    ipywidgets

COPY . .

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=''", "--ServerApp.password=''"]

FROM gpu-stage AS final