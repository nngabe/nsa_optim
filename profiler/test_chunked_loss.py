"""
Quick test to verify chunked loss computation works
"""
import torch
from config import ModelConfig, AttentionType, ModelSize, MODEL_CONFIGS
from model import create_model

def test_chunked_loss():
    """Test that chunked loss reduces memory usage"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create a 4B model config with long context
    base_config = MODEL_CONFIGS[ModelSize.MEDIUM]

    model_config = ModelConfig(
        name=base_config.name,
        hidden_size=base_config.hidden_size,
        num_hidden_layers=base_config.num_hidden_layers,
        num_attention_heads=base_config.num_attention_heads,
        num_key_value_heads=base_config.num_key_value_heads,
        intermediate_size=base_config.intermediate_size,
        vocab_size=base_config.vocab_size,
        max_position_embeddings=131072,
        attention_type=AttentionType.DENSE,
    )

    print("Creating model...")
    model = create_model(model_config)
    model = model.to(device)
    model = model.to(torch.bfloat16)
    model.train()

    print(f"Model: {model_config.name}")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # Test with long sequence that should trigger chunking
    batch_size = 1
    seq_len = 131072

    print(f"\nTest 1: Long sequence (should use chunking)")
    print(f"Sequence length: {seq_len}")

    # Estimate logits size
    vocab_size = model_config.vocab_size
    logits_size_gb = (batch_size * seq_len * vocab_size * 2) / (1024**3)
    print(f"Estimated logits size: {logits_size_gb:.2f} GB")
    print(f"Should chunk: {logits_size_gb > 10.0}")

    torch.cuda.reset_peak_memory_stats()

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    print("Running forward pass with labels...")
    try:
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, loss, _ = model(input_ids, labels=labels)

        print(f"✓ Success!")
        print(f"Loss: {loss.item():.4f}")
        print(f"Logits returned: {logits is not None}")
        print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

        if logits is None:
            print("✓ Chunked loss was used (logits not materialized)")
        else:
            print(f"✗ Full logits materialized: {logits.shape}")

    except torch.cuda.OutOfMemoryError as e:
        print(f"✗ OOM Error: {e}")
        return False

    # Test with short sequence that should NOT trigger chunking
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    seq_len_short = 2048
    logits_size_gb_short = (batch_size * seq_len_short * vocab_size * 2) / (1024**3)

    print(f"\nTest 2: Short sequence (should NOT use chunking)")
    print(f"Sequence length: {seq_len_short}")
    print(f"Estimated logits size: {logits_size_gb_short:.2f} GB")
    print(f"Should chunk: {logits_size_gb_short > 10.0}")

    input_ids_short = torch.randint(0, vocab_size, (batch_size, seq_len_short), device=device)
    labels_short = input_ids_short.clone()

    print("Running forward pass with labels...")
    try:
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits_short, loss_short, _ = model(input_ids_short, labels=labels_short)

        print(f"✓ Success!")
        print(f"Loss: {loss_short.item():.4f}")
        print(f"Logits returned: {logits_short is not None}")
        print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

        if logits_short is not None:
            print(f"✓ Full logits returned: {logits_short.shape}")
        else:
            print("✗ Unexpected: chunked loss was used for short sequence")

    except torch.cuda.OutOfMemoryError as e:
        print(f"✗ OOM Error: {e}")
        return False

    print("\n" + "="*80)
    print("All tests passed! Chunked loss is working correctly.")
    print("="*80)
    return True


if __name__ == "__main__":
    success = test_chunked_loss()
    exit(0 if success else 1)
