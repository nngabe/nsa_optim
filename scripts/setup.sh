#!/bin/bash
set -e

echo "=========================================="
echo "NSA + Optimizer Ablation Study Setup"
echo "=========================================="

# Create virtual environment if it doesn't exist
#if [ ! -d "venv" ]; then
#    echo "Creating virtual environment..."
#    python -m venv venv
#fi

pip install transformers datasets tokenizers bitsandbytes
pip install accelerate wandb einops peft fairscale yacs


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
#echo "export PYTHONPATH=\"\${PYTHONPATH}:$(pwd)/external/Flash-Sparse-Attention\"" >> venv/bin/activate

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
