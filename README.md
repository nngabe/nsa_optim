# NSA + Optimizer Ablation Study

This repository contains experiments for efficient training of foundation models. In particular, we experiment with LLM training and the following ablations:
- **Attention mechanisms**: Dense attention vs Native Sparse Attention (NSA)
- **Optimizers**: AdamW, AdamW8bit, SOAP, Shampoo, SOAP with low-bit states
- **Model sizes**: 0.6B, 4B, 8B, 32B (Qwen-3 architecture)
- **Context lengths**: 32K, 128K (all), 512K, 1M (NSA only)

## Overview

We compare efficient architectures, attention mechanisms, and optimizers for training LLMs. Specifically, we investigate usage of the following compute/parameter efficient methods:

- [Native Sparse Attention](https://github.com/fla-org/native-sparse-attention) - Hardware-aligned sparse attention
- [NVIDIA Emerging-Optimizers](https://github.com/NVIDIA-NeMo/Emerging-Optimizers) - SOAP and Shampoo implementations
- [HuggingFace PEFT](https://github.com/huggingface/peft) - Parameter-efficient fine-tuning utilities

## Installation

```bash
# Clone this repository
git clone https://github.com/nngabe/nsa_optim.git
cd nsa_optim

# Run setup script
chmod +x setup.sh
./setup.sh

# Activate environment
source venv/bin/activate
```

## Project Structure

```
nsa_optimizer_ablation/
├── config.py           # Configuration classes and experiment grid
├── model.py            # Model architecture (Dense & NSA attention)
├── optimizers.py       # Optimizer implementations (AdamW, AdamW8bit, SOAP, Shampoo)
├── data.py             # Data loading and tokenization
├── train.py            # Main training script
├── run_experiments.py  # Experiment runner and job generation
├── setup.sh            # Installation script
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Experiment Grid

The full ablation study covers:

| Dimension | Options |
|-----------|---------|
| Model Size | 0.6B, 4B, 8B, 32B |
| Attention | Dense, NSA |
| Optimizer | AdamW, AdamW8bit, SOAP, Shampoo, SOAP-LowBit |
| Context Length | 32K, 128K, 512K*, 1M* |

*512K and 1M context lengths are only tested with NSA (native sparse attention is designed for long contexts)

**Total experiments**: 160 (4 sizes × 2 attention × 5 optimizers × 4 contexts, with 512K/1M limited to NSA)

## Quick Start

### Smoke Tests

Run quick smoke tests to verify your setup:

```bash
# Run all smoke tests (unit tests + training tests)
./smoke_test.sh

# Or run specific tests
source venv/bin/activate

# Just unit tests
pytest -v

# Just a quick training smoke test
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

### Single Experiment

```bash
# Train a 0.6B model with dense attention and AdamW
python train.py \
    --model_size 0.6B \
    --attention_type dense \
    --optimizer_type adamw \
    --context_length 32768 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_steps 100000 \
    --output_dir ./outputs

# Train with NSA and SOAP optimizer
python train.py \
    --model_size 0.6B \
    --attention_type native_sparse_attention \
    --optimizer_type soap \
    --context_length 131072 \
    --gradient_checkpointing \
    --output_dir ./outputs
```

### Multi-GPU Training

```bash
# Using torchrun for distributed training
torchrun --nproc_per_node=8 train.py \
    --model_size 4B \
    --attention_type native_sparse_attention \
    --optimizer_type soap \
    --context_length 131072 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --dtype bfloat16 \
    --gradient_checkpointing
```

### Run All Experiments

```bash
# Generate experiment manifest
python run_experiments.py --mode=generate

# Run subset of experiments locally
python run_experiments.py \
    --mode=run \
    --model_sizes 0.6B \
    --attention_types dense native_sparse_attention \
    --optimizer_types adamw soap \
    --num_gpus=8

# Generate SLURM scripts for cluster
python run_experiments.py \
    --mode=slurm \
    --partition=gpu \
    --account=myaccount \
    --time_limit=48:00:00
```

## Model Architecture

Models follow the Qwen-3 architecture:

| Size | Hidden | Layers | Heads | KV Heads | Intermediate |
|------|--------|--------|-------|----------|--------------|
| 0.6B | 1024   | 28     | 16    | 8        | 3072         |
| 4B   | 2560   | 36     | 32    | 8        | 9216         |
| 8B   | 4096   | 36     | 32    | 8        | 12288        |
| 32B  | 5120   | 64     | 40    | 8        | 25600        |

### Attention Mechanisms

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

**SOAP-LowBit**: SOAP with 4-bit quantized optimizer states for memory efficiency.

## Configuration

### Training Config

```python
TrainingConfig(
    model_size=ModelSize.SMALL,       # 0.6B, 4B, 8B, 32B
    attention_type=AttentionType.NSA, # dense, native_sparse_attention
    optimizer_type=OptimizerType.SOAP,
    max_seq_length=131072,            # Context length
    batch_size=1,
    gradient_accumulation_steps=16,
    num_train_steps=100000,
    warmup_steps=2000,
    dtype="bfloat16",
    gradient_checkpointing=True,
)
```

### Optimizer Config

```python
OptimizerConfig(
    optimizer_type=OptimizerType.SOAP,
    learning_rate=1e-4,
    weight_decay=0.1,
    beta1=0.9,
    beta2=0.95,
    precondition_frequency=10,  # SOAP/Shampoo specific
    shampoo_beta=0.95,          # Kronecker factor EMA
    max_precond_dim=8192,       # Max dimension for preconditioning
)
```

## Memory Estimates

### Memory Components (per GPU with DDP)

For a model with `P` parameters:
1. **Model weights**: `P × 2 bytes` (bfloat16)
2. **Gradients**: `P × 2 bytes` (bfloat16)
3. **Optimizer states**:
   - AdamW: `P × 8 bytes` (2 states in fp32)
   - AdamW-8bit: `P × 2 bytes` (~75% reduction)
   - SOAP/Shampoo: `P × 12-16 bytes` (more states, but better convergence)
4. **Activations**: `batch × seq_len × hidden × layers × bytes` (varies significantly)
   - Without gradient checkpointing: Full activations stored
   - With gradient checkpointing: ~50-70% memory reduction, ~20% slower

### Per-GPU Memory Requirements (Single GPU or DDP)

**Batch size 1, bfloat16, with gradient checkpointing:**

| Model | Params | Dense 32K | Dense 128K | NSA 32K | NSA 128K | NSA 512K | NSA 1M |
|-------|--------|-----------|------------|---------|----------|----------|--------|
| 0.6B  | 0.59B  | 12GB | 25GB | 10GB | 20GB | 40GB | 70GB |
| 4B    | 3.92B  | 35GB | 65GB | 30GB | 55GB | 95GB | OOM* |
| 8B    | 7.62B  | 60GB | OOM* | 55GB | 85GB | OOM* | OOM* |
| 32B   | 31.4B  | OOM* | OOM* | OOM* | OOM* | OOM* | OOM* |

**Without gradient checkpointing, add 50-100% to activation memory.**

*Requires multi-GPU with FSDP or pipeline parallelism

### Memory Optimization Tips

1. **Always enable gradient checkpointing** for training (unless you need maximum speed)
2. **Use AdamW-8bit** to reduce optimizer memory by ~75%
3. **Use NSA attention** for long context (512K+) - required for those configs
4. **Use FSDP** instead of DDP for models >8B or when memory is tight
5. **Reduce batch size** or gradient accumulation steps if still OOM
6. **Mixed precision**: bfloat16 is recommended (fp16 requires gradient scaling)

## Monitoring

Training logs to wandb by default:

```bash
# Set wandb project
export WANDB_PROJECT=nsa-optimizer-ablation

# View experiments
wandb login
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

## Results

Based on the literature, expected findings include:

1. **NSA vs Dense**: NSA should show similar quality with 2-4x memory efficiency at long contexts
2. **SOAP vs AdamW**: SOAP typically converges 1.5-2x faster in wall-clock time
3. **Shampoo vs SOAP**: Similar convergence, SOAP slightly more stable
4. **Low-bit SOAP**: ~2x memory reduction with minimal quality loss

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
