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
pip install torch>=2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.40.0 datasets>=2.18.0 tokenizers>=0.15.0
pip install accelerate>=0.28.0 wandb>=0.16.0 einops>=0.7.0

# Install Triton
echo "Installing Triton..."
pip install triton>=2.2.0

# Try to install Flash Attention
echo "Installing Flash Attention..."
pip install flash-attn --no-build-isolation 2>/dev/null || echo "Warning: Flash Attention installation failed, will use PyTorch SDPA"

# Install torchao and bitsandbytes for low-bit optimizers
echo "Installing low-bit optimizer dependencies..."
pip install torchao>=0.3.0 || echo "Warning: torchao installation failed"
pip install bitsandbytes>=0.43.0 || echo "Warning: bitsandbytes installation failed"

# Clone and install Native Sparse Attention
echo "Installing Native Sparse Attention..."
if [ ! -d "external/native-sparse-attention" ]; then
    mkdir -p external
    cd external
    git clone https://github.com/fla-org/native-sparse-attention.git
    cd native-sparse-attention
    git submodule update --init --recursive
    pip install . --no-deps
    cd ../..
else
    echo "Native Sparse Attention already cloned"
fi

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
    pip install . || echo "Warning: low-bit-optimizers installation failed"
    cd ../..
else
    echo "low-bit-optimizers already cloned"
fi

# Clone PEFT (optional, for comparison)
echo "Installing PEFT..."
pip install peft>=0.10.0

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
