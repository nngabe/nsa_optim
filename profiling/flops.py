"""
FLOPS and Arithmetic Intensity Calculation for Transformer Models

This module provides utilities for calculating:
1. FLOPs (Floating Point Operations) for dense and sparse attention
2. Arithmetic Intensity (FLOPs per byte) for workload characterization
3. Human-readable formatting of these metrics
"""
from config import ModelConfig, AttentionType


def calculate_attention_flops(
    model_config: ModelConfig,
    seq_len: int,
) -> int:
    """
    Calculate FLOPs for attention computation per token.

    Dense attention: O(seq_len * hidden_size) per token
    Sparse attention: O(num_selected_blocks * block_size * hidden_size) per token

    Args:
        model_config: Model configuration
        seq_len: Sequence length

    Returns:
        FLOPs for attention computation per token
    """
    h = model_config.hidden_size
    n_heads = model_config.num_attention_heads
    head_dim = h // n_heads

    if model_config.attention_type == AttentionType.DENSE:
        # Dense attention: Q @ K^T (seq_len x seq_len) + softmax + @ V
        # Per query token: seq_len comparisons * head_dim operations
        # QK^T: seq_len * head_dim * 2 (multiply-add)
        # Softmax: seq_len * 5 (approximate: exp, sum, div)
        # Attention @ V: seq_len * head_dim * 2
        attn_flops_per_head = seq_len * head_dim * 4 + seq_len * 5
        total_attn_flops = attn_flops_per_head * n_heads
    else:
        # Sparse attention (NSA/FSA)
        # Only attends to selected blocks
        block_size = model_config.nsa_block_size
        num_selected = model_config.nsa_num_selected_blocks

        # Effective attention length per query
        eff_attn_len = block_size * num_selected

        # Similar to dense but with reduced sequence length
        attn_flops_per_head = eff_attn_len * head_dim * 4 + eff_attn_len * 5
        total_attn_flops = attn_flops_per_head * n_heads

    return total_attn_flops


def calculate_flops_per_token(model_config: ModelConfig, seq_len: int) -> int:
    """
    Calculate FLOPs per token for a transformer model.

    Forward pass includes:
    - Embedding lookup
    - Attention (dense or sparse)
    - Feed-forward network
    - Layer normalization

    Backward pass is approximately 2x forward.
    Total training FLOPs = 3x forward FLOPs.

    Args:
        model_config: Model configuration
        seq_len: Sequence length

    Returns:
        Total FLOPs per token (forward + backward)
    """
    h = model_config.hidden_size
    n_layers = model_config.num_hidden_layers
    n_heads = model_config.num_attention_heads
    n_kv_heads = model_config.num_key_value_heads
    i_size = model_config.intermediate_size
    vocab_size = model_config.vocab_size

    # === Per-token FLOPs ===

    # 1. Embedding layer (forward)
    # No computation per token (just lookup), but we count the output projection
    embedding_flops = 0

    # 2. Per-layer computation
    per_layer_flops = 0

    # 2a. QKV projection
    # GQA: Q is full, K and V are reduced
    q_flops = 2 * h * h  # Q projection
    kv_flops = 2 * h * h * n_kv_heads // n_heads  # K and V projections (GQA)
    qkv_proj_flops = q_flops + kv_flops

    # 2b. Attention computation (varies by attention type)
    attn_compute_flops = calculate_attention_flops(model_config, seq_len)

    # 2c. Attention output projection
    attn_out_proj_flops = 2 * h * h

    # 2d. FFN (2 linear layers with activation)
    # Up projection: h -> intermediate_size
    # Down projection: intermediate_size -> h
    ffn_flops = 2 * h * i_size + 2 * i_size * h

    # 2e. Layer normalization (2 per layer: pre-attn and pre-ffn)
    # LayerNorm: mean, var, normalize, scale, shift ≈ 10 ops per element
    ln_flops = 2 * 10 * h

    # Total per layer
    per_layer_flops = (
        qkv_proj_flops +
        attn_compute_flops +
        attn_out_proj_flops +
        ffn_flops +
        ln_flops
    )

    # 3. All layers
    all_layers_flops = per_layer_flops * n_layers

    # 4. Final layer norm and output projection (LM head)
    final_ln_flops = 10 * h
    lm_head_flops = 2 * h * vocab_size

    # Total forward FLOPs per token
    forward_flops = embedding_flops + all_layers_flops + final_ln_flops + lm_head_flops

    # Backward pass is ~2x forward (computing gradients)
    # Total training = forward + backward = 3x forward
    total_flops_per_token = 3 * forward_flops

    return total_flops_per_token


def calculate_arithmetic_intensity(
    model_config: ModelConfig,
    batch_size: int,
    seq_len: int,
    num_params: int,
    dtype_bytes: int = 2,  # bfloat16/float16
) -> float:
    """
    Calculate arithmetic intensity (FLOPs per byte of memory traffic).

    Arithmetic Intensity = FLOPs / Memory Access (bytes)

    Memory access includes:
    - Reading model parameters
    - Reading/writing activations
    - Reading/writing gradients

    Higher AI means more compute-bound (better GPU utilization).
    Lower AI means more memory-bound (limited by bandwidth).

    Typical values:
    - < 10 FLOPs/byte: Memory-bound
    - 10-100 FLOPs/byte: Balanced
    - > 100 FLOPs/byte: Compute-bound

    Args:
        model_config: Model configuration
        batch_size: Batch size
        seq_len: Sequence length
        num_params: Number of model parameters
        dtype_bytes: Bytes per parameter (2 for bf16/fp16, 4 for fp32)

    Returns:
        Arithmetic intensity in FLOPs/byte
    """
    flops_per_token = calculate_flops_per_token(model_config, seq_len)
    total_flops = flops_per_token * batch_size * seq_len

    # Memory accesses
    # 1. Parameter reads (forward + backward)
    param_reads = 2 * num_params * dtype_bytes

    # 2. Gradient writes
    grad_writes = num_params * dtype_bytes

    # 3. Activation memory (varies by attention type)
    h = model_config.hidden_size
    n_layers = model_config.num_hidden_layers

    if model_config.attention_type == AttentionType.DENSE:
        # Dense attention: store full attention matrices
        # KV cache + attention weights
        activation_size_per_layer = h * seq_len + seq_len * seq_len
    else:
        # Sparse attention: only store selected blocks
        block_size = model_config.nsa_block_size
        num_selected = model_config.nsa_num_selected_blocks
        # Reduced attention pattern
        activation_size_per_layer = h * seq_len + seq_len * block_size * num_selected

    activation_memory = (
        2 * n_layers * activation_size_per_layer * batch_size * dtype_bytes
    )

    # 4. Optimizer state reads/writes (for Adam: 2 states per param)
    optimizer_memory = 4 * num_params * dtype_bytes

    total_memory_bytes = param_reads + grad_writes + activation_memory + optimizer_memory

    arithmetic_intensity = total_flops / total_memory_bytes

    return arithmetic_intensity


def format_flops(flops: float) -> str:
    """
    Format FLOPs in human-readable units.

    Args:
        flops: FLOPs value

    Returns:
        Formatted string with appropriate units
    """
    if flops >= 1e15:
        return f"{flops/1e15:.2f} PFLOPS"
    elif flops >= 1e12:
        return f"{flops/1e12:.2f} TFLOPS"
    elif flops >= 1e9:
        return f"{flops/1e9:.2f} GFLOPS"
    elif flops >= 1e6:
        return f"{flops/1e6:.2f} MFLOPS"
    else:
        return f"{flops:.2f} FLOPS"


def format_arithmetic_intensity(ai: float) -> str:
    """
    Format arithmetic intensity with appropriate units.

    Args:
        ai: Arithmetic intensity value

    Returns:
        Formatted string
    """
    return f"{ai:.1f} FLOP/B"
