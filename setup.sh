#!/bin/bash
set -e

echo "=========================================="
echo "NSA + Optimizer Ablation Study Setup"
echo "=========================================="

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install core dependencies
echo "Installing core dependencies..."
# Install PyTorch 2.9.1 with CUDA 13.0 for Blackwell (sm_120) support
pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu130
pip install transformers datasets tokenizers bitsandbytes
pip install accelerate wandb einops peft

# Install Triton
echo "Installing Triton..."
pip install "triton>=3.3.0"

# Install ninja (required for some CUDA extensions)
echo "Installing ninja..."
pip install ninja

# Try to install Flash Attention
echo "Installing Flash Attention..."
pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-2.8.3+cu130torch2.9-cp312-cp312-linux_x86_64.whl"
#export TORCH_CUDA_ARCH_LIST="12.0"
#pip install -v flash-attn --no-build-isolation 2>/dev/null || echo "Warning: Flash Attention installation failed, will use PyTorch SDPA"
#

# INstall torchao from source (update coming soon but broken now: https://github.com/pytorch/ao/issues/2919)
echo "Installing torchao from source"
mkdir -p external
cd external
git clone https://github.com/pytorch/ao
cd ao
python setup.py install
cd ..

# Clone and install Native Sparse Attention
echo "Installing Native Sparse Attention..."
if [ ! -d "external/native-sparse-attention" ]; then
    cd external
    git clone https://github.com/fla-org/native-sparse-attention.git
    cd native-sparse-attention
    git submodule update --init --recursive
    pip install . --no-deps
    cd ../..
else
    echo "Native Sparse Attention already cloned"
fi

# Clone and install Flash Sparse Attention
echo "Installing Flash Sparse Attention..."
if [ ! -d "external/Flash-Sparse-Attention" ]; then
    cd external
    git clone https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention.git
    cd ../
fi

# Install FSA dependencies (skip flash-attn since we already installed it)
echo "Installing Flash Sparse Attention dependencies..."
pip install nvidia_cudnn_frontend

# Add Flash-Sparse-Attention to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/external/Flash-Sparse-Attention"
echo "export PYTHONPATH=\"\${PYTHONPATH}:$(pwd)/external/Flash-Sparse-Attention\"" >> venv/bin/activate

# Clone and install NVIDIA Emerging-Optimizers
echo "Installing NVIDIA Emerging-Optimizers..."
if [ ! -d "external/Emerging-Optimizers" ]; then
    cd external
    git clone https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git
    cd Emerging-Optimizers
    pip install .
    cd ../..
else
    echo "Emerging-Optimizers already cloned"
fi

# Clone low-bit optimizers (optional)
echo "Installing low-bit optimizers..."
if [ ! -d "external/low-bit-optimizers" ]; then
    cd external
    git clone https://github.com/thu-ml/low-bit-optimizers.git
    cd low-bit-optimizers
    pip install . --no-build-isolation || echo "Warning: low-bit-optimizers installation failed"
    cd ../..
else
    echo "low-bit-optimizers already cloned, attempting installation..."
    cd external/low-bit-optimizers
    pip install . --no-build-isolation || echo "Warning: low-bit-optimizers installation failed"
    cd ../..
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
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
echo "To generate SLURM scripts:"
echo "  python run_experiments.py --mode=slurm --partition=gpu --account=myaccount"
