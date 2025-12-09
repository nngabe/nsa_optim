"""
Memory profiling script to identify memory bottlenecks
"""
import os
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, record_function

from config import TrainingConfig, ModelConfig, ModelSize, AttentionType, OptimizerType, OptimizerConfig, MODEL_CONFIGS
from models import create_model
from data import DataConfig, create_dataloader, get_tokenizer


def format_bytes(bytes):
    """Format bytes to human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"


def print_memory_stats(stage_name):
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            max_allocated = torch.cuda.max_memory_allocated(i)
            print(f"\n{stage_name} - GPU {i}:")
            print(f"  Allocated: {format_bytes(allocated)}")
            print(f"  Reserved:  {format_bytes(reserved)}")
            print(f"  Max Allocated: {format_bytes(max_allocated)}")


def profile_forward_pass(model_size_str="4B", context_length=131072, attention_type="nsa"):
    """Profile a single forward pass"""
    print(f"\n{'='*80}")
    print(f"Profiling {model_size_str} model with {context_length} context length")
    print(f"Attention type: {attention_type}")
    print(f"{'='*80}\n")

    # Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.reset_peak_memory_stats()

    # Create config
    model_size = ModelSize(model_size_str)
    attn_type = AttentionType.NSA if attention_type == "nsa" else AttentionType.DENSE

    base_config = MODEL_CONFIGS[model_size]
    model_config = ModelConfig(
        name=base_config.name,
        hidden_size=base_config.hidden_size,
        num_hidden_layers=base_config.num_hidden_layers,
        num_attention_heads=base_config.num_attention_heads,
        num_key_value_heads=base_config.num_key_value_heads,
        intermediate_size=base_config.intermediate_size,
        vocab_size=base_config.vocab_size,
        max_position_embeddings=context_length,
        attention_type=attn_type,
    )

    print(f"Model config:")
    print(f"  Hidden size: {model_config.hidden_size}")
    print(f"  Layers: {model_config.num_hidden_layers}")
    print(f"  Attention heads: {model_config.num_attention_heads}")
    print(f"  Vocab size: {model_config.vocab_size}")

    # Create model
    print("\nCreating model...")
    model = create_model(model_config)
    model = model.to(device)
    model = model.to(torch.bfloat16)
    model.gradient_checkpointing_enable()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params / 1e9:.2f}B")

    print_memory_stats("After model creation")

    # Create dummy input
    batch_size = 1
    seq_len = context_length

    print(f"\nCreating input tensors (batch={batch_size}, seq_len={seq_len})...")
    input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    print_memory_stats("After input creation")

    # Forward pass breakdown
    model.train()

    with record_function("embedding"):
        hidden_states = model.embed_tokens(input_ids)
        print(f"\nEmbedding output shape: {hidden_states.shape}")
        print(f"Embedding output size: {format_bytes(hidden_states.numel() * hidden_states.element_size())}")
        print_memory_stats("After embedding")

    # First transformer layer
    with record_function("first_layer"):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            hidden_states, _ = model.layers[0](hidden_states)
        print(f"\nFirst layer output shape: {hidden_states.shape}")
        print_memory_stats("After first layer")

    # Try full forward without loss
    print("\n\nRunning full forward pass (no loss)...")
    torch.cuda.reset_peak_memory_stats()

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        with torch.no_grad():
            logits, _, _ = model(input_ids)

    print(f"\nLogits shape: {logits.shape}")
    logits_size = logits.numel() * logits.element_size()
    print(f"Logits tensor size: {format_bytes(logits_size)}")
    print(f"  = batch({batch_size}) × seq_len({seq_len}) × vocab({model_config.vocab_size}) × {logits.element_size()} bytes")

    print_memory_stats("After forward (no loss)")

    # Now try with loss computation
    print("\n\nRunning forward pass WITH loss...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits, loss, _ = model(input_ids, labels=labels)

        print(f"Loss: {loss.item():.4f}")
        print_memory_stats("After forward WITH loss")

    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ OOM ERROR during loss computation!")
        print(f"Error: {e}")
        print_memory_stats("At OOM")

        # Analyze the issue
        print("\n\n" + "="*80)
        print("MEMORY ANALYSIS")
        print("="*80)

        shift_logits_shape = (batch_size, seq_len - 1, model_config.vocab_size)
        shift_logits_flat_shape = ((seq_len - 1) * batch_size, model_config.vocab_size)

        print(f"\nLoss computation creates:")
        print(f"  shift_logits shape: {shift_logits_shape}")
        print(f"  shift_logits.view(-1, vocab) shape: {shift_logits_flat_shape}")

        # In cross_entropy, PyTorch may create intermediate tensors
        intermediate_size = shift_logits_flat_shape[0] * shift_logits_flat_shape[1] * 4  # float32
        print(f"  Estimated intermediate tensor size: {format_bytes(intermediate_size)}")

        print(f"\nTotal memory needed for loss computation:")
        print(f"  Logits: {format_bytes(logits_size)}")
        print(f"  Intermediate: {format_bytes(intermediate_size)}")
        print(f"  Total: {format_bytes(logits_size + intermediate_size)}")

        return False

    return True


def profile_with_pytorch_profiler(model_size_str="4B", context_length=32768):
    """Run PyTorch profiler to get detailed memory trace"""
    print(f"\n{'='*80}")
    print(f"Running PyTorch profiler for {model_size_str} @ {context_length} context")
    print(f"{'='*80}\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create smaller model for profiling
    model_size = ModelSize(model_size_str)
    base_config = MODEL_CONFIGS[model_size]

    model_config = ModelConfig(
        name=base_config.name,
        hidden_size=base_config.hidden_size,
        num_hidden_layers=min(4, base_config.num_hidden_layers),  # Fewer layers for profiling
        num_attention_heads=base_config.num_attention_heads,
        num_key_value_heads=base_config.num_key_value_heads,
        intermediate_size=base_config.intermediate_size,
        vocab_size=base_config.vocab_size,
        max_position_embeddings=context_length,
        attention_type=AttentionType.DENSE,
    )

    model = create_model(model_config)
    model = model.to(device)
    model = model.to(torch.bfloat16)
    model.train()

    input_ids = torch.randint(0, model_config.vocab_size, (1, context_length), device=device)
    labels = input_ids.clone()

    print("Starting profiler...")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            try:
                logits, loss, _ = model(input_ids, labels=labels)
                loss.backward()
            except torch.cuda.OutOfMemoryError:
                print("OOM during profiled run")

    # Print memory summary
    print("\n" + "="*80)
    print("Top memory operations:")
    print("="*80)
    print(prof.key_averages().table(
        sort_by="self_cuda_memory_usage",
        row_limit=20,
    ))

    # Export trace
    prof.export_chrome_trace("memory_profile_trace.json")
    print("\nProfile saved to: memory_profile_trace.json")
    print("View it at: chrome://tracing")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="4B")
    parser.add_argument("--context_length", type=int, default=131072)
    parser.add_argument("--attention_type", type=str, default="fsa")
    parser.add_argument("--detailed", action="store_true", help="Run detailed PyTorch profiler")
    args = parser.parse_args()

    # Run basic profiling
    success = profile_forward_pass(
        args.model_size,
        args.context_length,
        args.attention_type,
    )

    # Run detailed profiling if requested
    if args.detailed and args.context_length <= 32768:
        profile_with_pytorch_profiler(args.model_size, args.context_length)
