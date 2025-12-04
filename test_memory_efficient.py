"""
Test memory-efficient training with smaller model/context
"""
import torch
from config import ModelConfig, AttentionType
from model import create_model

def format_bytes(bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"

def test_small_config():
    """Test with smaller config that fits in memory"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Use smaller model for testing
    model_config = ModelConfig(
        name="test-model",
        hidden_size=1024,
        num_hidden_layers=8,  # Fewer layers
        num_attention_heads=16,
        num_key_value_heads=8,
        intermediate_size=3072,
        vocab_size=151936,
        max_position_embeddings=131072,
        attention_type=AttentionType.DENSE,
    )

    print("Creating model...")
    model = create_model(model_config)
    model = model.to(device)
    model = model.to(torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.train()

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params / 1e9:.3f}B")

    # Test with long sequence
    seq_len = 131072
    batch_size = 1
    vocab_size = model_config.vocab_size

    logits_size_gb = (batch_size * seq_len * vocab_size * 2) / (1024**3)
    print(f"\nSequence length: {seq_len}")
    print(f"Estimated logits size: {logits_size_gb:.2f} GB")
    print(f"Will use chunking: {logits_size_gb > 10.0}")

    torch.cuda.reset_peak_memory_stats()

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    mem_after_input = torch.cuda.memory_allocated() / 1024**3
    print(f"Memory after input: {mem_after_input:.2f} GB")

    print("\nRunning forward pass...")
    try:
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, loss, _ = model(input_ids, labels=labels)

        mem_after_forward = torch.cuda.memory_allocated() / 1024**3
        mem_peak = torch.cuda.max_memory_allocated() / 1024**3

        print(f"✓ Forward pass succeeded!")
        print(f"Loss: {loss.item():.4f}")
        print(f"Logits materialized: {logits is not None}")
        print(f"Memory after forward: {mem_after_forward:.2f} GB")
        print(f"Peak memory: {mem_peak:.2f} GB")

        if logits is None:
            print("✓ Using chunked loss (memory efficient)")
        else:
            print(f"Logits shape: {logits.shape}")

        # Test backward
        print("\nRunning backward pass...")
        loss.backward()

        mem_after_backward = torch.cuda.memory_allocated() / 1024**3
        mem_peak_backward = torch.cuda.max_memory_allocated() / 1024**3

        print(f"✓ Backward pass succeeded!")
        print(f"Memory after backward: {mem_after_backward:.2f} GB")
        print(f"Peak memory (including backward): {mem_peak_backward:.2f} GB")

        return True

    except torch.cuda.OutOfMemoryError as e:
        mem_at_oom = torch.cuda.memory_allocated() / 1024**3
        print(f"✗ OOM during forward/backward")
        print(f"Memory at OOM: {mem_at_oom:.2f} GB")
        print(f"Error: {str(e)[:200]}")
        return False

if __name__ == "__main__":
    success = test_small_config()
    print("\n" + "="*80)
    if success:
        print("SUCCESS: Memory-efficient training works!")
    else:
        print("FAILED: Need more optimization or smaller config")
    print("="*80)
    exit(0 if success else 1)
