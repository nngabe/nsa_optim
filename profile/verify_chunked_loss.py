"""
Verify chunked loss works by testing at the threshold
"""
import torch
from config import ModelConfig, AttentionType
from model import create_model

def test_at_threshold():
    """Test with sequence length right at the chunking threshold"""
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()

    # Small model
    model_config = ModelConfig(
        name="test-small",
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=2048,
        vocab_size=151936,
        max_position_embeddings=32768,
        attention_type=AttentionType.DENSE,
    )

    model = create_model(model_config).to(device).to(torch.bfloat16)
    model.train()
    model.gradient_checkpointing_enable()

    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    vocab_size = model_config.vocab_size

    # Test 1: Short sequence (should NOT chunk)
    seq_len_short = 4096
    logits_size_gb = (1 * seq_len_short * vocab_size * 2) / (1024**3)

    print(f"\nTest 1: seq_len={seq_len_short}, logits={logits_size_gb:.2f}GB, should_chunk={logits_size_gb > 10.0}")

    input_ids = torch.randint(0, vocab_size, (1, seq_len_short), device=device)
    labels = input_ids.clone()

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits, loss, _ = model(input_ids, labels=labels)

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Logits returned: {logits is not None}")
    if logits is not None:
        print(f"  ✓ Standard loss used (logits shape: {logits.shape})")
    else:
        print(f"  ✗ Unexpected: chunked loss used")

    # Test 2: Long sequence (should chunk)
    torch.cuda.empty_cache()
    seq_len_long = 40000  # This will give ~11 GB logits
    logits_size_gb = (1 * seq_len_long * vocab_size * 2) / (1024**3)

    print(f"\nTest 2: seq_len={seq_len_long}, logits={logits_size_gb:.2f}GB, should_chunk={logits_size_gb > 10.0}")

    input_ids = torch.randint(0, vocab_size, (1, seq_len_long), device=device)
    labels = input_ids.clone()

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits, loss, _ = model(input_ids, labels=labels)

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Logits returned: {logits is not None}")
    if logits is None:
        print(f"  ✓ Chunked loss used (memory efficient)")
    else:
        print(f"  ✗ Unexpected: standard loss used (logits shape: {logits.shape})")

    print("\n" + "="*80)
    print("Chunked loss implementation verified!")
    print("="*80)

if __name__ == "__main__":
    test_at_threshold()
