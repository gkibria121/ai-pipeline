# Use official PyTorch image with CUDA support
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

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
RUN pip install --no-cache-dir \
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
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
