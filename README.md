# Mamba + Sparse/Linear Attention + SOAP optimizer Ablation Study

This repository contains experiments for efficient training of foundation models. In particular, we investigate LLM training with the following ablations:
- **Models**: Transformers, Mamba, Jamba/Hybrid
- **Attention mechanisms**: Dense attention vs Native Sparse Attention (NSA)
- **Optimizers**: AdamW, AdamW8bit, SOAP4bit
- **Model sizes**: 100M-4B 
- **Context lengths**: 32K, 64k, 128k


## Overview

We compare efficient architectures, attention mechanisms, and optimizers for training LLMs. Specifically, we investigate usage of the following compute/parameter efficient methods and utilities:

- [Native Sparse Attention](https://github.com/fla-org/native-sparse-attention) - Hardware-aligned sparse attention
- [NVIDIA Emerging-Optimizers](https://github.com/NVIDIA-NeMo/Emerging-Optimizers) - SOAP and Shampoo implementations
- [PyTorch AO](https://github.com/pytorch/ao) - PyTorch native quantization and sparsity for training and inference

## Experiment checkpoint

The latest experiment checkpoint can be viewed on Weights & Biases:

[Latest Checkpoint](https://api.wandb.ai/links/ngabriel-gwu/u1p3w3xx)

## Installation

Run the Docker Container for Blackwell GPUs [Dockerhub](https://hub.docker.com/repository/docker/nickgabriel/blackwell-cuda131/general):
```
docker run --rm -it --gpus all nickgabriel/blackwell-cuda131:latest
```
This image specifically targets archs sm_100 and sm_120 for Blackwell GPUs, with the most up-to-date kernels and support for 4-bit/8-bit states and mixed precision operations. 

If you'd like to run experiments on Ampere/Hopper GPUs, use available images like 
```
docker run -rm -it --gpus all nvcr.io/nvidia/pytorch:24.04-py3
```

To install the remaining python dependencies, use 
```
source scripts/setup.sh
```

## Quickstart
To train hybrid models, you can specify the block structure with a string, e.g. `MDMA` where `M=Mamba`, `A=Attention`, and `D=DeltaNet`. Attention types for the `A` blocks can be set to sparse or dense using the --attn_type flag. For example, to train a 1B model with `AdamW8bit` from `torchao` we can use:
```
python train.py --model_size 1B --block_pattern MDMA --attn_type nsa --optimizer_type adamw8bit
```
or to launch multi-GPU training with FSDP use:
```
torchrun --nproc_per_node=4 train.py --model_size 1B --block_pattern MDMA --attn_type nsa --optimizer_type adamw8bit
```


## Project Structure

```
nsa_optim/
├── config.py              # Configuration classes, enums, and experiment grid
├── optimizers.py          # Optimizer implementations (AdamW, SOAP, Shampoo, low-bit variants)
├── data.py                # Data loading and tokenization
├── train.py               # Main training script with FSDP/DDP support
├── models/                # Model architectures
│   ├── __init__.py        # Package exports
│   ├── transformer.py     # Transformer with Dense/NSA/FSA attention
│   ├── mamba.py           # Mamba2 state space model
│   ├── deltanet.py        # Gated Delta Net linear attention
│   ├── hybrid.py          # Hybrid model (interleaved Mamba/DeltaNet/Attention)
│   └── configs.py         # Model config factory functions
├── profiling/             # Memory and compute profiling utilities
├── analysis/              # Result analysis and evaluation utilities
├── tests/                 # Unit tests
├── scripts/               # Shell scripts (setup.sh, smoke_test.sh)
├── configs/               # Configuration files
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Experiment Grid

The ablation study covers a select subset for the following configurations:

| Dimension | Options |
|-----------|---------|
| Model Size | 0.6B, 4B, 8B, 32B |
| Attention | Dense, NSA |
| Optimizer | AdamW, AdamW8bit, SOAP4bit |
| Context Length | 32K, 64K, 128K |


### Smoke Tests

Run quick smoke tests to verify your setup:

```bash
# Run all smoke tests (unit tests + training tests)
./scripts/smoke_test.sh

# Or run specific tests
source venv/bin/activate

# Run unit tests
pytest -v

# Or just a quick training smoke test
python train.py \
    --model_size 0.6B \
    --attention_type dense \
    --optimizer_type adamw_8bit \
    --context_length 32768 \
    --num_train_steps 2 \
    --log_interval 1 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --dtype bfloat16
```

The smoke test script tests:
- ✓ Unit tests (config, model, optimizers)
- ✓ AdamW with bfloat16 (baseline)
- ✓ AdamW8bit with bfloat16
- ✓ Float16 and Float32 precision modes
- ✓ Gradient checkpointing
- ✓ SOAP optimizer (if available)


    --output_dir ./outputs
```

## Model Architecture

**Dense Attention**: Standard multi-head attention with FlashAttention-2 backend. O(n²) complexity.

**Native Sparse Attention (NSA)**: Hardware-aligned sparse attention combining:
- Block-level top-k attention selection
- Sliding window attention
- Gated combination of selected and window attention
- Near-linear complexity for long sequences

### Optimizers

**AdamW**: Standard adaptive optimizer with decoupled weight decay.

**AdamW8bit**: 8-bit quantized AdamW from torchao. Uses block-wise quantization to reduce optimizer state memory by ~75% with minimal accuracy impact.

**SOAP**: ShampoO with Adam in Preconditioner eigenbasis. Combines:
- Kronecker-factored preconditioning from Shampoo
- Adam-style updates in the eigenbasis
- Periodic eigenbasis updates

**Shampoo**: Full matrix preconditioning with Kronecker factorization.

**SOAP4bit**: SOAP with 4-bit quantized optimizer states for memory efficiency.


## Monitoring

Training logs to wandb by default:

```bash
# Set wandb project
export WANDB_PROJECT=nsa-optimizer-ablation
```

Logged metrics:
- `train/loss`: Training loss
- `train/lr`: Learning rate
- `train/tokens_per_sec`: Throughput
- `train/grad_norm`: Gradient norm
- `eval/loss`: Evaluation loss
- `eval/perplexity`: Evaluation perplexity

## Checkpointing

Checkpoints are saved to `{output_dir}/{run_name}/checkpoint-{step}/`:
- `model.pt`: Model weights
- `optimizer.pt`: Optimizer state
- `scheduler.pt`: LR scheduler state
- `config.json`: Training configuration

Resume training:
```bash
python train.py \
    --resume_from ./outputs/my_run/checkpoint-50000 \
    ...
```

## Citation

If you use this code, please cite the relevant papers:

```bibtex
@inproceedings{yuan2025nsa,
  title={Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention},
  author={Yuan, Jingyang and others},
  year={2025}
}

@inproceedings{vyas2024soap,
  title={SOAP: Improving and Stabilizing Shampoo using Adam},
  author={Vyas, Nikhil and others},
  booktitle={ICLR},
  year={2025}
}

@article{gupta2018shampoo,
  title={Shampoo: Preconditioned Stochastic Tensor Optimization},
  author={Gupta, Vineet and others},
  journal={ICML},
  year={2018}
}
```

## CI/CD with GitLab

The project includes comprehensive GitLab CI/CD integration for automated testing, training, and deployment.

### Pipeline Stages

| Stage | Jobs | Purpose |
|-------|------|---------|
| `lint` | ruff, mypy, black | Code quality checks |
| `test` | unit, config, model, optimizers | Automated testing |
| `build` | docker, package | Build containers and packages |
| `train-smoke` | smoke-dense-adamw, smoke-nsa-soap | Quick validation (100 steps) |
| `train-experiments` | Full experiment matrix | Complete ablation study |
| `analyze` | results, compare-* | Result aggregation |
| `deploy` | model-registry, pages | Upload models, docs |

### Running the Pipeline

```bash
# Trigger smoke tests on every MR/push to main
# Full experiments require manual trigger or variables:

# Run all experiments
git push -o ci.variable="RUN_FULL_EXPERIMENTS=true"

# Run batch experiments (matrix)
git push -o ci.variable="RUN_BATCH_EXPERIMENTS=true"

# Deploy models to HuggingFace
git push -o ci.variable="DEPLOY_MODELS=true" -o ci.variable="HF_TOKEN=xxx"
```

### Required GitLab CI Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `CI_REGISTRY_*` | Container registry credentials | Auto-set |
| `HF_TOKEN` | HuggingFace API token | For deployment |
| `HF_REPO_ID` | Target HuggingFace repo | For deployment |
| `WANDB_API_KEY` | Weights & Biases key | Optional |

### GPU Runners

The pipeline requires GitLab runners with GPU support for training jobs. Tag your runners with:
- `gpu` - Has NVIDIA GPU
- `cuda` - CUDA toolkit installed

### Scheduled Pipelines

Configure in GitLab CI/CD → Schedules:

| Schedule | Variable | Frequency |
|----------|----------|-----------|
| Nightly smoke | `SCHEDULE_TYPE=nightly` | Daily |
| Weekly experiments | `SCHEDULE_TYPE=weekly` | Weekly |

### Local Testing

```bash
# Run tests locally
pytest tests/unit/ -v

# Run linting
ruff check .
black --check .
mypy .

# Build Docker image locally
docker build -t nsa-ablation .
```

### Docker Image

The CI builds a Docker image with all dependencies:

```bash
# Pull from GitLab registry
docker pull $CI_REGISTRY_IMAGE:latest

# Run training in container
docker run --gpus all -v $(pwd)/outputs:/app/outputs \
    $CI_REGISTRY_IMAGE:latest \
    python train.py --model_size 0.6B --attention_type nsa
