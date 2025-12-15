# Dockerfile for NSA + Optimizer Ablation Study
# Extends the blackwell-cuda131 base image with project-specific dependencies
#
# Build with: docker build -f docker/Dockerfile -t <registry>/nsa-optim:latest .
# Run with: docker run --gpus all -v $(pwd):/workspace <registry>/nsa-optim:latest

FROM nickgabriel/blackwell-cuda131:latest

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Install project dependencies using the setup script
RUN bash /workspace/scripts/setup.sh

# Install additional testing dependencies
RUN pip install pytest pytest-cov

# Create necessary directories
RUN mkdir -p /workspace/outputs /workspace/wandb /workspace/.cache

# Set Python path to include external dependencies
ENV PYTHONPATH="${PYTHONPATH}:/workspace/external/Flash-Sparse-Attention"

# Default command
CMD ["/bin/bash"]
