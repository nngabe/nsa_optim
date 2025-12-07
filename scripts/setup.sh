#!/bin/bash
# ==============================================================================
# NSA + Optimizer Ablation Study Setup Script
# Automatically detects and configures CUDA, Python, and PyTorch versions
# ==============================================================================
set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==============================================================================
# Helper Functions
# ==============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}==========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}==========================================${NC}"
    echo ""
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ==============================================================================
# Version Detection Functions
# ==============================================================================

detect_python_version() {
    # Detect Python version
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python not found. Please install Python 3.10+"
        exit 1
    fi

    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

    print_info "Detected Python version: $PYTHON_VERSION"

    # Validate Python version (minimum 3.10)
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
        print_error "Python 3.10+ is required. Found: $PYTHON_VERSION"
        exit 1
    fi

    export PYTHON_VERSION PYTHON_CMD PYTHON_MAJOR PYTHON_MINOR
}

detect_cuda_version() {
    # Detect CUDA version from nvcc or nvidia-smi
    CUDA_VERSION=""
    CUDA_MAJOR=""
    CUDA_MINOR=""

    # Try nvcc first
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
        print_info "Detected CUDA version from nvcc: $CUDA_VERSION"
    # Fallback to nvidia-smi
    elif command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
        # nvidia-smi shows driver version, need to map to CUDA version
        # For simplicity, try to get from the CUDA version field
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        print_info "Detected CUDA version from nvidia-smi: $CUDA_VERSION"
    # Check environment variable
    elif [ -n "$CUDA_HOME" ]; then
        if [ -f "$CUDA_HOME/version.txt" ]; then
            CUDA_VERSION=$(cat "$CUDA_HOME/version.txt" | head -1 | awk '{print $3}' | cut -d. -f1,2)
        elif [ -f "$CUDA_HOME/version.json" ]; then
            CUDA_VERSION=$(cat "$CUDA_HOME/version.json" | grep -o '"cuda" : "[0-9.]*"' | cut -d'"' -f4)
        fi
        print_info "Detected CUDA version from CUDA_HOME: $CUDA_VERSION"
    fi

    if [ -z "$CUDA_VERSION" ]; then
        print_warn "Could not detect CUDA version. Assuming CUDA 13.1 (latest)"
        CUDA_VERSION="13.1"
    fi

    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)

    # Determine PyTorch CUDA version string (cu118, cu121, cu124, cu130, cu131, etc.)
    if [ "$CUDA_MAJOR" -ge 13 ]; then
        if [ "$CUDA_MINOR" -ge 1 ]; then
            PYTORCH_CUDA_VERSION="cu131"
        else
            PYTORCH_CUDA_VERSION="cu130"
        fi
    elif [ "$CUDA_MAJOR" -eq 12 ]; then
        if [ "$CUDA_MINOR" -ge 4 ]; then
            PYTORCH_CUDA_VERSION="cu124"
        elif [ "$CUDA_MINOR" -ge 1 ]; then
            PYTORCH_CUDA_VERSION="cu121"
        else
            PYTORCH_CUDA_VERSION="cu118"
        fi
    else
        PYTORCH_CUDA_VERSION="cu118"
    fi

    print_info "PyTorch CUDA version string: $PYTORCH_CUDA_VERSION"

    export CUDA_VERSION CUDA_MAJOR CUDA_MINOR PYTORCH_CUDA_VERSION
}

detect_gpu_architecture() {
    # Detect GPU compute capability for optimal kernel compilation
    GPU_ARCH=""

    if command -v nvidia-smi &> /dev/null; then
        # Get GPU name
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
        print_info "Detected GPU: $GPU_NAME"

        # Map GPU to compute capability
        case "$GPU_NAME" in
            *"B200"*|*"B100"*|*"GB200"*|*"GB100"*)
                GPU_ARCH="10.0"
                print_info "Blackwell GPU detected - compute capability 10.0"
                ;;
            *"H100"*|*"H200"*)
                GPU_ARCH="9.0"
                print_info "Hopper GPU detected - compute capability 9.0"
                ;;
            *"A100"*|*"A10"*|*"A30"*|*"A40"*)
                GPU_ARCH="8.0"
                print_info "Ampere GPU detected - compute capability 8.0"
                ;;
            *"4090"*|*"4080"*|*"4070"*|*"L40"*)
                GPU_ARCH="8.9"
                print_info "Ada Lovelace GPU detected - compute capability 8.9"
                ;;
            *"3090"*|*"3080"*|*"3070"*)
                GPU_ARCH="8.6"
                print_info "Ampere consumer GPU detected - compute capability 8.6"
                ;;
            *)
                GPU_ARCH="8.0"
                print_warn "Unknown GPU architecture, defaulting to 8.0"
                ;;
        esac
    else
        print_warn "nvidia-smi not available, using default architecture"
        GPU_ARCH="10.0"  # Default to Blackwell for this project
    fi

    # Set TORCH_CUDA_ARCH_LIST for kernel compilation
    export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"
    export GPU_ARCH
}

detect_pytorch_version() {
    # Check if PyTorch is installed and get version
    PYTORCH_VERSION=""
    PYTORCH_CUDA_BUILT=""

    if $PYTHON_CMD -c "import torch" 2>/dev/null; then
        PYTORCH_VERSION=$($PYTHON_CMD -c "import torch; print(torch.__version__)")
        PYTORCH_CUDA_BUILT=$($PYTHON_CMD -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'None')")
        print_info "Detected PyTorch version: $PYTORCH_VERSION (CUDA: $PYTORCH_CUDA_BUILT)"
    else
        print_info "PyTorch not installed yet"
    fi

    export PYTORCH_VERSION PYTORCH_CUDA_BUILT
}

print_environment_summary() {
    print_header "Environment Summary"
    echo "  Python:        $PYTHON_VERSION ($PYTHON_CMD)"
    echo "  CUDA:          $CUDA_VERSION (PyTorch: $PYTORCH_CUDA_VERSION)"
    echo "  GPU Arch:      $GPU_ARCH"
    if [ -n "$PYTORCH_VERSION" ]; then
        echo "  PyTorch:       $PYTORCH_VERSION (CUDA: $PYTORCH_CUDA_BUILT)"
    fi
    if [ -n "$GPU_NAME" ]; then
        echo "  GPU:           $GPU_NAME"
    fi
    echo ""
}

# ==============================================================================
# Installation Functions
# ==============================================================================

install_pytorch() {
    print_header "Installing PyTorch Nightly"

    # Use nightly for cutting-edge features (Blackwell support)
    if [ "$CUDA_MAJOR" -ge 13 ]; then
        print_info "Installing PyTorch nightly with CUDA $PYTORCH_CUDA_VERSION support..."
        pip install --pre torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/nightly/$PYTORCH_CUDA_VERSION
    else
        print_info "Installing PyTorch stable with CUDA $PYTORCH_CUDA_VERSION support..."
        pip install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/$PYTORCH_CUDA_VERSION
    fi

    # Verify installation
    detect_pytorch_version
    if [ -z "$PYTORCH_VERSION" ]; then
        print_error "PyTorch installation failed"
        exit 1
    fi
}

install_core_dependencies() {
    print_header "Installing Core Dependencies"

    pip install --upgrade pip setuptools wheel

    pip install \
        transformers \
        datasets \
        tokenizers \
        bitsandbytes \
        accelerate \
        wandb \
        einops \
        peft \
        fairscale \
        yacs \
        safetensors
}

install_triton() {
    print_header "Installing Triton"

    # Triton version depends on PyTorch version
    pip install "triton>=3.3.0"

    # Verify Triton installation
    if $PYTHON_CMD -c "import triton" 2>/dev/null; then
        TRITON_VERSION=$($PYTHON_CMD -c "import triton; print(triton.__version__)")
        print_info "Triton installed: $TRITON_VERSION"
    else
        print_warn "Triton installation may have failed"
    fi
}

install_flash_attention() {
    print_header "Installing Flash Attention"

    # Check for prebuilt wheels first
    FLASH_ATTN_INSTALLED=false

    # Try to find compatible prebuilt wheel
    TORCH_VERSION_SHORT=$(echo $PYTORCH_VERSION | cut -d'+' -f1 | sed 's/\.dev.*//')
    PYTHON_TAG="cp${PYTHON_MAJOR}${PYTHON_MINOR}"

    print_info "Looking for Flash Attention wheel for PyTorch $TORCH_VERSION_SHORT, Python $PYTHON_TAG, CUDA $PYTORCH_CUDA_VERSION..."

    # Try installing from pip with no-build-isolation
    export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"
    if pip install flash-attn --no-build-isolation 2>/dev/null; then
        FLASH_ATTN_INSTALLED=true
        print_info "Flash Attention installed successfully"
    else
        print_warn "Flash Attention compilation failed, will use PyTorch SDPA fallback"
    fi
}

install_torchao() {
    print_header "Installing TorchAO from Source"

    if [ ! -d "external/ao" ]; then
        mkdir -p external
        cd external
        git clone https://github.com/pytorch/ao
        cd ao
        $PYTHON_CMD setup.py install
        cd ../..
        print_info "TorchAO installed from source"
    else
        print_info "TorchAO already cloned, reinstalling..."
        cd external/ao
        git pull origin main || true
        $PYTHON_CMD setup.py install
        cd ../..
    fi
}

install_native_sparse_attention() {
    print_header "Installing Native Sparse Attention"

    if [ ! -d "external/native-sparse-attention" ]; then
        mkdir -p external
        cd external
        git clone https://github.com/fla-org/native-sparse-attention.git
        cd native-sparse-attention
        git submodule update --init --recursive
        pip install . --no-deps
        cd ../..
        print_info "Native Sparse Attention installed"
    else
        print_info "Native Sparse Attention already cloned"
    fi
}

install_flash_sparse_attention() {
    print_header "Installing Flash Sparse Attention"

    if [ ! -d "external/Flash-Sparse-Attention" ]; then
        mkdir -p external
        cd external
        git clone https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention.git
        cd ../
        print_info "Flash Sparse Attention cloned"
    else
        print_info "Flash Sparse Attention already cloned"
    fi

    # Install dependencies
    pip install nvidia_cudnn_frontend

    # Add to PYTHONPATH
    export PYTHONPATH="${PYTHONPATH}:$(pwd)/external/Flash-Sparse-Attention"
}

install_emerging_optimizers() {
    print_header "Installing NVIDIA Emerging-Optimizers"

    if [ ! -d "external/Emerging-Optimizers" ]; then
        mkdir -p external
        cd external
        git clone https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git
        cd Emerging-Optimizers
        pip install .
        cd ../..
        print_info "Emerging-Optimizers installed"
    else
        print_info "Emerging-Optimizers already cloned"
    fi
}

install_low_bit_optimizers() {
    print_header "Installing Low-bit Optimizers"

    if [ ! -d "external/low-bit-optimizers" ]; then
        mkdir -p external
        cd external
        git clone https://github.com/thu-ml/low-bit-optimizers.git
        cd low-bit-optimizers
        pip install . --no-build-isolation || print_warn "low-bit-optimizers installation failed"
        cd ../..
    else
        print_info "low-bit-optimizers already cloned, attempting installation..."
        cd external/low-bit-optimizers
        pip install . --no-build-isolation || print_warn "low-bit-optimizers installation failed"
        cd ../..
    fi
}

install_ninja() {
    print_header "Installing Ninja Build System"
    pip install ninja
}

# ==============================================================================
# Main Setup Flow
# ==============================================================================

main() {
    print_header "NSA + Optimizer Ablation Study Setup"

    # Detect environment
    print_header "Detecting Environment"
    detect_python_version
    detect_cuda_version
    detect_gpu_architecture
    detect_pytorch_version
    print_environment_summary

    # Upgrade pip first
    print_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel

    # Install PyTorch if not present or wrong CUDA version
    if [ -z "$PYTORCH_VERSION" ] || [ "$PYTORCH_CUDA_BUILT" != "${CUDA_MAJOR}.${CUDA_MINOR}" ]; then
        install_pytorch
    else
        print_info "PyTorch already installed with correct CUDA version"
    fi

    # Refresh PyTorch version detection
    detect_pytorch_version

    # Install all dependencies
    install_core_dependencies
    install_triton
    install_ninja
    install_flash_attention
    install_torchao
    install_native_sparse_attention
    install_flash_sparse_attention
    install_emerging_optimizers
    install_low_bit_optimizers

    # Final environment summary
    print_header "Setup Complete!"
    detect_pytorch_version
    print_environment_summary

    echo "Installed versions:"
    echo "  Python:    $PYTHON_VERSION"
    echo "  CUDA:      $CUDA_VERSION"
    echo "  PyTorch:   $PYTORCH_VERSION"
    echo "  GPU Arch:  $GPU_ARCH (TORCH_CUDA_ARCH_LIST)"
    echo ""
    echo "To run a single experiment:"
    echo "  python train.py --model_size 0.6B --attention_type dense --optimizer_type adamw"
    echo ""
    echo "To generate all experiments:"
    echo "  python run_experiments.py --mode=generate"
    echo ""
    echo "To run all experiments locally:"
    echo "  python run_experiments.py --mode=run --num_gpus=8"
    echo ""
}

# Run main function
main "$@"
