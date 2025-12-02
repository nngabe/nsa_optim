# Dockerfile for NSA + Optimizer Ablation Study
# Multi-stage build for optimized image size

# ==============================================================================
# Stage 1: Build dependencies
# ==============================================================================
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone and build native-sparse-attention
RUN git clone https://github.com/fla-org/native-sparse-attention.git && \
    cd native-sparse-attention && \
    git submodule update --init --recursive && \
    pip install . --no-deps

# Clone and build Emerging-Optimizers
RUN git clone https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git && \
    cd Emerging-Optimizers && \
    pip install .

# Try to install flash-attn (may fail on some architectures)
RUN pip install flash-attn --no-build-isolation || echo "Flash attention not available"

# ==============================================================================
# Stage 2: Runtime image
# ==============================================================================
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /opt/conda /opt/conda

WORKDIR /app

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=8 \
    TOKENIZERS_PARALLELISM=false

# Create non-root user
RUN useradd -m -u 1000 trainer && \
    chown -R trainer:trainer /app
USER trainer

# Default command
CMD ["python", "train.py", "--help"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print(torch.cuda.is_available())" || exit 1
