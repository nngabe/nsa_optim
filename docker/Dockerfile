# Dockerfile for NSA + Optimizer Ablation Study
# Ubuntu 24.04 + CUDA 13.1 + PyTorch Nightly + Blackwell GPU Optimization
# Optimized for NVIDIA Blackwell Architecture (sm_100/sm_120)

# ==============================================================================
# Stage 1: Base CUDA image with development tools
# ==============================================================================
FROM nvidia/cuda:13.1.0-cudnn-devel-ubuntu24.04 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python and development tools
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    # Build essentials
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    # NVIDIA tools and libraries
    libnccl2 \
    libnccl-dev \
    libcudnn9-dev \
    # Performance and profiling tools
    linux-tools-generic \
    numactl \
    libnuma-dev \
    # Additional utilities
    htop \
    tmux \
    vim \
    jq \
    unzip \
    ca-certificates \
    # MPI support for distributed training
    libopenmpi-dev \
    openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# ==============================================================================
# Stage 2: Build environment with NVIDIA optimizations for Blackwell
# ==============================================================================
FROM base AS builder

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=${CUDA_HOME}/lib64:${LIBRARY_PATH}

# Blackwell GPU architecture flags (sm_100 for GB100/GB200, sm_120 for future)
ENV TORCH_CUDA_ARCH_LIST="10.0;10.0+PTX;12.0;12.0+PTX"
ENV CUDA_ARCHITECTURES="100;120"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch nightly with CUDA 13.1 support
# Note: PyTorch nightly builds may have varying CUDA support
RUN pip install --no-cache-dir --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu131

# Install core ML/DL dependencies
RUN pip install --no-cache-dir \
    # Transformers and datasets
    transformers>=4.50.0 \
    datasets>=4.0.0 \
    tokenizers>=0.22.0 \
    huggingface-hub>=0.30.0 \
    # Training utilities
    accelerate>=1.0.0 \
    wandb>=0.20.0 \
    peft>=0.18.0 \
    # Distributed training
    fairscale>=0.4.13 \
    # Utilities
    einops>=0.8.0 \
    numpy>=2.0.0 \
    pandas>=2.0.0 \
    tqdm>=4.60.0 \
    PyYAML>=6.0.0 \
    safetensors>=0.5.0

# Install NVIDIA-specific optimization libraries
RUN pip install --no-cache-dir \
    # Triton for custom GPU kernels (critical for Blackwell optimization)
    triton>=3.3.0 \
    # Build tools
    ninja \
    # cuDNN frontend for custom operations
    nvidia-cudnn-frontend

# Install torchao for quantization and sparsity (build from source for latest features)
WORKDIR /tmp/build
RUN git clone https://github.com/pytorch/ao.git && \
    cd ao && \
    pip install --no-cache-dir . && \
    cd .. && rm -rf ao

# Install bitsandbytes with CUDA support
RUN pip install --no-cache-dir bitsandbytes>=0.48.0

# Install Flash Attention 2 (build from source for Blackwell compatibility)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || \
    echo "Flash Attention build failed, will be installed via setup.sh"

# ==============================================================================
# Stage 3: Runtime image
# ==============================================================================
FROM base AS runtime

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set up CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Blackwell GPU optimization environment variables
ENV TORCH_CUDA_ARCH_LIST="10.0;10.0+PTX;12.0;12.0+PTX"
ENV CUDA_ARCHITECTURES="100;120"

# PyTorch performance settings
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
ENV TORCH_NCCL_ASYNC_ERROR_HANDLING=1
ENV NCCL_DEBUG=WARN
ENV NCCL_IB_DISABLE=0
ENV NCCL_P2P_DISABLE=0

# Triton cache for compiled kernels
ENV TRITON_CACHE_DIR=/app/.triton_cache

# General Python settings
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OMP_NUM_THREADS=8
ENV MKL_NUM_THREADS=8
ENV TOKENIZERS_PARALLELISM=false

# Create application directory
WORKDIR /app

# Create directories for external dependencies and cache
RUN mkdir -p /app/external /app/.triton_cache

# Copy application code
COPY . .

# Make setup script executable
RUN chmod +x scripts/setup.sh scripts/smoke_test.sh

# Create non-root user for security
RUN useradd -m -u 1000 trainer && \
    chown -R trainer:trainer /app

# Switch to non-root user
USER trainer

# Default entrypoint
ENTRYPOINT ["/bin/bash"]

# Default command - show help
CMD ["-c", "echo 'NSA + Optimizer Ablation Study Container' && echo '' && echo 'To install dependencies: ./scripts/setup.sh' && echo 'To run training: python train.py --help'"]

# Health check to verify CUDA availability
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'CUDA OK: {torch.cuda.get_device_name(0)}')" || exit 1

# ==============================================================================
# Labels for container metadata
# ==============================================================================
LABEL maintainer="nngabe"
LABEL description="NSA + Optimizer Ablation Study - Blackwell GPU Optimized"
LABEL cuda_version="13.1"
LABEL pytorch="nightly"
LABEL ubuntu_version="24.04"
LABEL gpu_architecture="Blackwell (sm_100/sm_120)"
