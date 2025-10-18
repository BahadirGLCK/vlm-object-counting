# Dockerfile for Qwen2.5-VL GRPO RL Training
# Optimized for rented GPU servers (RunPod, Vast.ai, Lambda Labs, etc.)

# Base image: NVIDIA CUDA 12.1.0 with cuDNN 8, Ubuntu 22.04
# vLLM works best with CUDA 12.1.x
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Metadata
LABEL maintainer="VLM Object Counting RL"
LABEL description="Docker image for Qwen2.5-VL GRPO reinforcement learning training"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHON_VERSION=3.10 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Paths
ENV APP_DIR=/workspace \
    VENV_PATH=/opt/venv \
    PYTHONPATH="/workspace:${PYTHONPATH}"

# Add virtual environment to PATH
ENV PATH="${VENV_PATH}/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    build-essential \
    git \
    git-lfs \
    wget \
    curl \
    # Python
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    # Video processing (for OpenCV)
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Additional dependencies
    libaio-dev \
    libmpich-dev \
    # Utilities
    vim \
    tmux \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Create Python virtual environment
RUN python${PYTHON_VERSION} -m venv ${VENV_PATH}

# Upgrade pip and install uv (faster package installer)
RUN pip install --upgrade pip setuptools wheel && \
    pip install uv

# Install PyTorch with CUDA 12.1 support
# Using PyTorch 2.4+ for better CUDA 12.1 compatibility
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core ML dependencies
RUN pip install \
    transformers==4.55.4 \
    accelerate \
    bitsandbytes \
    xformers \
    scipy

# Install Unsloth and vLLM for RL training
# Note: vLLM version depends on GPU architecture
RUN pip install unsloth && \
    # Install vLLM - compatible version for CUDA 12.1
    pip install vllm==0.10.2 && \
    pip install triton

# Install TRL for GRPO training
RUN pip install --no-deps trl==0.22.2

# Install vision and data processing libraries
RUN pip install \
    opencv-python \
    "pillow<11.0.0" \
    qwen-vl-utils \
    datasets \
    scikit-learn \
    evaluate

# Install Jupyter and related tools for notebook execution
RUN pip install \
    jupyter \
    jupyterlab \
    ipywidgets \
    jupyter_contrib_nbextensions

# Install monitoring and utilities
RUN pip install \
    tensorboard \
    wandb \
    tqdm \
    pyyaml \
    hydra-core

# Set up workspace
WORKDIR ${APP_DIR}

# Create directories for data and outputs
RUN mkdir -p ${APP_DIR}/data/inputs/vlm_videos \
             ${APP_DIR}/data/inputs \
             ${APP_DIR}/notebooks \
             ${APP_DIR}/outputs_grpo_object_counting

# Copy project files
COPY . ${APP_DIR}

# Expose ports
# Jupyter: 8888
# TensorBoard: 6006
EXPOSE 8888 6006

# Create startup script
RUN echo '#!/bin/bash\n\
echo "=== VLM Object Counting RL Environment ==="\n\
echo ""\n\
echo "Available commands:"\n\
echo "  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"\n\
echo "  tensorboard --logdir=outputs_grpo_object_counting --host=0.0.0.0 --port=6006"\n\
echo ""\n\
echo "GPU Info:"\n\
nvidia-smi\n\
echo ""\n\
exec "$@"\n\
' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

# Default command: Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
