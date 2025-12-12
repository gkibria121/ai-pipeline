# ============================================
# Stage 1: GPU Build (with CUDA support)
# ============================================
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS gpu-stage

# Set environment variables for interactive use
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV TERM=xterm-256color
ENV SHELL=/bin/bash

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (PyTorch already included in base image)
# Using increased timeout and retries for large packages
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

# Copy the rest of the application
COPY . .

# Expose Jupyter port
EXPOSE 8888

# Create a non-root user for security
RUN useradd -m -s /bin/bash jupyter && \
    chown -R jupyter:jupyter /app

USER jupyter

# Default command to start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.token=''", "--ServerApp.password=''"]


# ============================================
# Stage 2: CPU Build (without CUDA)
# ============================================
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS cpu-stage

# Set environment variables for interactive use
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV TERM=xterm-256color
ENV SHELL=/bin/bash
# Explicitly disable GPU
ENV CUDA_VISIBLE_DEVICES=""
ENV NVIDIA_VISIBLE_DEVICES=void

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (PyTorch CPU already included in base image)
# Using increased timeout and retries for large packages
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

# Copy the rest of the application
COPY . .

# Expose Jupyter port
EXPOSE 8888

# Create a non-root user for security
RUN useradd -m -s /bin/bash jupyter && \
    chown -R jupyter:jupyter /app

USER jupyter

# Default command to start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.token=''", "--ServerApp.password=''"]


# ============================================
# Default stage (GPU)
# ============================================
FROM gpu-stage AS final