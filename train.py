"""
Main training script for NSA + Optimizer ablation study

Supports:
- Multi-GPU training with FSDP/DDP
- Gradient checkpointing
- Mixed precision training
- Logging with wandb/tensorboard
- Hybrid models with --block_pattern (M=Mamba, D=DeltaNet, A=Attention)
"""
import os
import sys
import math
import time
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from contextlib import nullcontext
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from config import (
    TrainingConfig,
    ModelConfig,
    AttentionType,
    MambaType,
    OptimizerType,
    OptimizerConfig,
    get_experiment_grid,
    get_filtered_experiments,
)
from models import (
    TransformerModel, TransformerBlock, create_model,
    Mamba2Model, Mamba2Config, Mamba2Block, create_mamba2,
    GatedDeltaNetModel, GatedDeltaNetConfig, GatedDeltaNetBlock, create_gated_deltanet,
    HybridModel, HybridConfig, MambaBlock, DeltaNetBlock, AttentionBlock,
    create_hybrid_model, get_block_classes_from_pattern,
    get_model_config, get_mamba2_config, get_deltanet_config, get_hybrid_config, get_model_type,
)
from optimizers import create_optimizer, get_lr_scheduler
from data import DataConfig, create_dataloader, get_tokenizer


def setup_distributed():
    """Initialize distributed training"""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def setup_model(
    training_config: TrainingConfig,
    rank: int,
    world_size: int,
    block_pattern: Optional[str] = None,
    block_repeats: int = 1,
    kernel_type: str = "liger",
) -> Tuple[nn.Module, str]:
    """Setup model with optional distributed wrapping.

    Returns:
        model: The initialized model
        model_type: String indicating the model type for logging
    """
    model_type = get_model_type(training_config, block_pattern)

    if model_type == "hybrid":
        hybrid_config = get_hybrid_config(training_config, block_pattern, block_repeats)
        model = create_hybrid_model(hybrid_config)
        block_cls = get_block_classes_from_pattern(block_pattern)
    elif model_type == "mamba2":
        mamba2_config = get_mamba2_config(training_config)
        model = create_mamba2(mamba2_config)
        block_cls = {Mamba2Block}
    elif model_type == "deltanet":
        deltanet_config = get_deltanet_config(training_config)
        model = create_gated_deltanet(deltanet_config)
        block_cls = {GatedDeltaNetBlock}
    else:  # transformer
        model_config = get_model_config(training_config)
        model = create_model(model_config, kernel_type=kernel_type)
        block_cls = {TransformerBlock}

    # Move to device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Setup dtype
    dtype = getattr(torch, training_config.dtype)
    print(f'model dtype: {dtype}')
    model = model.to(dtype)

    # Gradient checkpointing
    if training_config.gradient_checkpointing:
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

    # Distributed setup
    if world_size > 1:
        mixed_precision = MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=torch.float32,
            buffer_dtype=dtype,
        )

        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=block_cls,
        )

        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            auto_wrap_policy=auto_wrap_policy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
        )

    return model, model_type


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Optional[torch.cuda.amp.GradScaler],
    config: TrainingConfig,
    step: int,
    grad_accum_step: int,
    model_type: str = "transformer",
) -> Dict[str, float]:
    """Execute single training step"""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Move batch to device
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Forward pass with autocast
    use_amp = scaler is not None
    autocast_ctx = torch.cuda.amp.autocast(dtype=dtype) if use_amp else nullcontext()

    with autocast_ctx:
        if model_type in ("mamba2", "deltanet"):
            # These models return (logits, loss, cache)
            _, loss, _ = model(input_ids, labels=labels)
        else:
            # Transformer and hybrid return (logits, loss, hidden_states)
            _, loss, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        # Scale loss for gradient accumulation
        loss = loss / config.gradient_accumulation_steps

    # Backward pass
    if use_amp:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    metrics = {"loss": loss.item() * config.gradient_accumulation_steps}

    # Optimizer step after accumulation
    if (grad_accum_step + 1) % config.gradient_accumulation_steps == 0:
        if use_amp:
            scaler.unscale_(optimizer)

        # Gradient clipping
        if config.max_grad_norm > 0:
            if isinstance(model, (DDP, FSDP)):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.module.parameters(), config.max_grad_norm
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm
                )
            metrics["grad_norm"] = grad_norm.item()

        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        metrics["lr"] = scheduler.get_last_lr()[0]

    return metrics


def evaluate(
    model: nn.Module,
    eval_dataloader,
    config: TrainingConfig,
    max_batches: int = 50,
    model_type: str = "transformer",
) -> Dict[str, float]:
    """Run evaluation"""
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            if i >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            with torch.cuda.amp.autocast(dtype=dtype):
                if model_type in ("mamba2", "deltanet"):
                    _, loss, _ = model(input_ids, labels=labels)
                else:
                    _, loss, _ = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

            # Count non-padding tokens
            num_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    model.train()

    avg_loss = total_loss / max(1, total_tokens)
    perplexity = math.exp(avg_loss)

    return {"eval_loss": avg_loss, "eval_perplexity": perplexity}


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    config: TrainingConfig,
    output_dir: str,
    rank: int,
    model_type: str = "transformer",
):
    """Save training checkpoint"""
    if rank != 0:
        return

    checkpoint_dir = Path(output_dir) / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Get model state dict (handle DDP/FSDP)
    if isinstance(model, (DDP, FSDP)):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    torch.save(model_state, checkpoint_dir / "model.pt")
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")

    # Save config
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump({
            "step": step,
            "model_type": model_type,
            "model_size": config.model_size,
            "attention_type": config.attention_type.value,
            "optimizer_type": config.optimizer_type.value,
            "max_seq_length": config.max_seq_length,
        }, f, indent=2)

    print(f"Saved checkpoint to {checkpoint_dir}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    checkpoint_dir: str,
) -> int:
    """Load training checkpoint"""
    checkpoint_dir = Path(checkpoint_dir)

    if isinstance(model, (DDP, FSDP)):
        model.module.load_state_dict(torch.load(checkpoint_dir / "model.pt"))
    else:
        model.load_state_dict(torch.load(checkpoint_dir / "model.pt"))

    optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer.pt"))
    scheduler.load_state_dict(torch.load(checkpoint_dir / "scheduler.pt"))

    with open(checkpoint_dir / "config.json", "r") as f:
        config = json.load(f)

    return config["step"]


def train(
    config: TrainingConfig,
    resume_from: Optional[str] = None,
    block_pattern: Optional[str] = None,
    block_repeats: int = 1,
    kernel_type: str = "liger",
):
    """Main training function"""
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0

    # Determine model type from config
    model_type = get_model_type(config, block_pattern)

    # Logging
    if is_main:
        try:
            import wandb
            wandb.init(
                project="nsa-optimizer-ablation",
                name=config.run_name,
                config={
                    **vars(config),
                    "model_type": model_type,
                    "block_pattern": block_pattern,
                    "block_repeats": block_repeats,
                    "kernel_type": kernel_type,
                },
            )
            use_wandb = True
        except ImportError:
            use_wandb = False
            print("wandb not available, skipping logging")
    else:
        use_wandb = False

    # Load tokenizer - Qwen tokenizer works for all model types (vocab_size=151936)
    tokenizer = get_tokenizer("Qwen/Qwen3-0.6B")

    # Setup model
    if is_main:
        if model_type == "hybrid":
            print(f"Setting up Hybrid model: {config.model_size} "
                  f"(pattern={block_pattern}, repeats={block_repeats})")
        elif model_type in ("mamba2", "deltanet"):
            print(f"Setting up {model_type.upper()} model: {config.model_size}")
        else:
            print(f"Setting up Transformer model: {config.model_size} "
                  f"with {config.attention_type.value} (kernel={kernel_type})")

    model, model_type = setup_model(
        config, rank, world_size, block_pattern, block_repeats, kernel_type
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"Model parameters: {num_params / 1e9:.2f}B")

    # Setup optimizer
    optimizer_config = config.optimizer_config or OptimizerConfig(
        optimizer_type=config.optimizer_type,
        learning_rate=config.optimizer_config.learning_rate if config.optimizer_config else 1e-4,
    )

    # Unwrap model for optimizer
    model_for_opt = model.module if isinstance(model, (DDP, FSDP)) else model
    optimizer = create_optimizer(model_for_opt, optimizer_config, world_size)

    # Setup scheduler
    scheduler = get_lr_scheduler(
        optimizer,
        config.lr_scheduler_type,
        config.num_train_steps,
        config.warmup_steps,
        config.min_lr_ratio,
    )

    # Setup data
    data_config = DataConfig(
        dataset_name=config.dataset_name,
        dataset_subset=config.dataset_subset,
        tokenizer_path="Qwen/Qwen3-0.6B",
        max_seq_length=config.max_seq_length,
        batch_size=config.batch_size,
        streaming=True,
    )

    train_dataloader = create_dataloader(data_config, tokenizer, rank, world_size)

    # Setup AMP
    dtype = getattr(torch, config.dtype)
    use_amp = dtype in (torch.float16, torch.bfloat16)
    scaler = torch.cuda.amp.GradScaler() if use_amp and dtype == torch.float16 else None

    # Resume from checkpoint
    start_step = 0
    if resume_from:
        start_step = load_checkpoint(model, optimizer, scheduler, resume_from)
        if is_main:
            print(f"Resumed from step {start_step}")

    # Training loop
    model.train()
    output_dir = Path(config.output_dir) / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_main:
        print(f"Starting training for {config.num_train_steps} steps")
        print(f"Batch size: {config.batch_size} x {world_size} x {config.gradient_accumulation_steps}")
        print(f"Effective batch size: {config.batch_size * world_size * config.gradient_accumulation_steps}")
        print(f"Context length: {config.max_seq_length}")

    data_iter = iter(train_dataloader)
    step = start_step
    grad_accum_step = 0
    running_loss = 0.0
    start_time = time.time()
    tokens_per_step = config.batch_size * config.max_seq_length * world_size * config.gradient_accumulation_steps
    total_tokens = start_step * tokens_per_step

    while step < config.num_train_steps:
        # Get next batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            batch = next(data_iter)

        # Training step
        metrics = train_step(
            model, batch, optimizer, scheduler, scaler,
            config, step, grad_accum_step, model_type=model_type,
        )

        running_loss += metrics["loss"]
        grad_accum_step += 1

        # Update step counter after full gradient accumulation
        if grad_accum_step % config.gradient_accumulation_steps == 0:
            step += 1
            total_tokens += tokens_per_step

            # Logging
            if step % config.log_interval == 0 and is_main:
                elapsed = time.time() - start_time
                avg_loss = running_loss / config.log_interval
                tokens_per_sec = (
                    config.batch_size * config.max_seq_length *
                    config.log_interval * world_size
                ) / elapsed

                log_str = (
                    f"Step {step}/{config.num_train_steps} | "
                    f"Tokens: {total_tokens / 1e9:.3f}B | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {metrics.get('lr', 0):.2e} | "
                    f"Tok/s: {tokens_per_sec:.0f}"
                )
                if "grad_norm" in metrics:
                    log_str += f" | Grad: {metrics['grad_norm']:.2f}"

                print(log_str)

                if use_wandb:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/lr": metrics.get("lr", 0),
                        "train/tokens": total_tokens,
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/grad_norm": metrics.get("grad_norm", 0),
                    }, step=step)

                running_loss = 0.0
                start_time = time.time()

            # Save checkpoint
            if step % config.save_interval == 0:
                save_checkpoint(
                    model, optimizer, scheduler, step,
                    config, str(output_dir), rank, model_type=model_type,
                )

    # Final save
    save_checkpoint(
        model, optimizer, scheduler, step,
        config, str(output_dir), rank, model_type=model_type,
    )

    if use_wandb and is_main:
        wandb.finish()

    cleanup_distributed()

    if is_main:
        print("Training complete!")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train models for ablation study")

    # Architecture selection
    parser.add_argument("--mamba_type", type=str, default="none",
                       choices=["none", "mamba2", "deltanet"],
                       help="Model type: none (transformer), mamba2, or deltanet")
    parser.add_argument("-attn", "--attn_type", type=str, default="dense",
                       choices=["dense", "native_sparse_attention", "nsa", "flash_sparse_attention", "fsa"],
                       help="Attention type for transformer models")

    # Hybrid model support
    parser.add_argument("--block_pattern", type=str, default=None,
                       help="Block pattern for hybrid models, e.g., 'MMDMMA' where M=Mamba, D=DeltaNet, A=Attention")
    parser.add_argument("--block_repeats", type=int, default=1,
                       help="Number of times to repeat the block pattern")

    # Kernel type
    parser.add_argument("--kernel_type", type=str, default="liger",
                       choices=["baseline", "triton", "liger"],
                       help="Kernel implementation type for transformer")

    # Experiment selection
    parser.add_argument("--model_size", type=str, default="0.6B",
                       help="Model size (e.g., '0.6B', '1B', '2.5B', '500M')")
    parser.add_argument("--optimizer_type", type=str, default="adamw",
                       choices=["adamw", "adamw4bit", "adamw8bit", "soap", "soap4bit", "soap8bit", "shampoo"])
    parser.add_argument("--context_length", type=int, default=8192,
                       choices=[8192, 32768, 65536, 131072, 524288, 1048576])

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_steps", type=int, default=10000)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("-lr","--learning_rate", type=float, default=1e-4)
    parser.add_argument("-wd","--weight_decay", type=float, default=0.1)
    parser.add_argument("-clip","--max_grad_norm", type=float, default=1.0)

    # Precision
    parser.add_argument("--dtype", type=str, default="bfloat16",
                       choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")

    # Data
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset_subset", type=str, default="sample-10BT")

    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--resume_from", type=str, default=None)

    # Logging
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=5000)

    return parser.parse_args()


def main():
    args = parse_args()

    # Normalize attention type aliases
    if args.attn_type == "nsa":
        args.attn_type = "native_sparse_attention"
    if args.attn_type == "fsa":
        args.attn_type = "flash_sparse_attention"

    # Map string args to enums
    model_size = args.model_size  # Now accepts any string like "1B", "2.5B", "500M"
    attention_type = AttentionType(args.attn_type)

    # Map mamba_type to MambaType enum
    if args.mamba_type == "mamba2":
        mamba_type = MambaType.MAMBA2
    elif args.mamba_type == "deltanet":
        mamba_type = MambaType.MAMBA3  # Repurpose MAMBA3 for deltanet
    else:
        mamba_type = MambaType.NONE

    optimizer_type = OptimizerType(args.optimizer_type)

    # Create optimizer config
    optimizer_config = OptimizerConfig(
        optimizer_type=optimizer_type,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Create training config
    config = TrainingConfig(
        model_size=model_size,
        attention_type=attention_type,
        mamba_type=mamba_type,
        optimizer_type=optimizer_type,
        optimizer_config=optimizer_config,
        max_seq_length=args.context_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_steps=args.num_train_steps,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        dtype=args.dtype,
        gradient_checkpointing=args.gradient_checkpointing,
        dataset_name=args.dataset_name,
        dataset_subset=args.dataset_subset,
        output_dir=args.output_dir,
        run_name=args.run_name,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
    )

    # Update run_name if not provided
    if not args.run_name:
        if args.block_pattern:
            arch_name = f"hybrid_{args.block_pattern}x{args.block_repeats}"
        elif args.mamba_type != "none":
            arch_name = args.mamba_type
        elif attention_type == AttentionType.NSA:
            arch_name = "nsa"
        elif attention_type == AttentionType.FSA:
            arch_name = "fsa"
        else:
            arch_name = "dense"

        config.run_name = f"{model_size}_{arch_name}_{optimizer_type.value}_ctx{args.context_length}"

    train(
        config,
        args.resume_from,
        block_pattern=args.block_pattern,
        block_repeats=args.block_repeats,
        kernel_type=args.kernel_type,
    )


if __name__ == "__main__":
    main()
