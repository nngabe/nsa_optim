"""
Model configuration functions for NSA + Optimizer Ablation Study

Provides factory functions to create model configurations:
- get_model_config: Transformer model config
- get_mamba2_config: Mamba2 model config
- get_deltanet_config: GatedDeltaNet model config
- get_hybrid_config: Hybrid model config
- get_model_type: Determine model type from training config
"""

from typing import Optional, Tuple

from config import (
    TrainingConfig,
    ModelConfig,
    MambaType,
    get_model_config_for_size,
    parse_model_size,
    compute_model_dimensions,
)

from models.mamba import Mamba2Config
from models.deltanet import GatedDeltaNetConfig
from models.hybrid import HybridConfig


def get_model_config(
    training_config: TrainingConfig,
    nsa_block_size: int = 64,
    nsa_window_size: int = 64,
    nsa_num_selected_blocks: int = 16,
    rope_theta: float = 100000.0,
) -> ModelConfig:
    """Get model configuration based on training config, computing dimensions dynamically"""
    return get_model_config_for_size(
        training_config.model_size,
        attention_type=training_config.attention_type,
        max_position_embeddings=training_config.max_seq_length,
        nsa_block_size=nsa_block_size,
        nsa_window_size=nsa_window_size,
        nsa_num_selected_blocks=nsa_num_selected_blocks,
        rope_theta=rope_theta,
    )


def get_mamba2_config(training_config: TrainingConfig) -> Mamba2Config:
    """Get Mamba2 configuration based on training config, computing dimensions dynamically"""
    target_params = parse_model_size(training_config.model_size)
    vocab_size = 151936
    d_state = 128
    headdim = 64
    expand = 2

    # Compute model dimensions - use similar logic to transformer but adjust for Mamba params
    # Mamba2 per-layer params ≈ d_model * (expand * 2 + d_conv + d_state * 2)
    # Simplified: find d_model and n_layers to match target

    hidden_size, num_layers, _, _, _ = compute_model_dimensions(
        target_params, vocab_size=vocab_size
    )

    return Mamba2Config(
        d_model=hidden_size,
        n_layers=num_layers,
        d_state=d_state,
        headdim=headdim,
        vocab_size=vocab_size,
        expand=expand,
        use_triton=True,
        gradient_checkpointing=training_config.gradient_checkpointing,
    )


def get_deltanet_config(training_config: TrainingConfig) -> GatedDeltaNetConfig:
    """Get GatedDeltaNet configuration based on training config, computing dimensions dynamically"""
    target_params = parse_model_size(training_config.model_size)
    vocab_size = 151936
    d_state = 128
    head_dim = 64

    # Compute model dimensions dynamically
    hidden_size, num_layers, _, _, _ = compute_model_dimensions(
        target_params, vocab_size=vocab_size
    )

    return GatedDeltaNetConfig(
        d_model=hidden_size,
        n_layers=num_layers,
        d_state=d_state,
        head_dim=head_dim,
        vocab_size=vocab_size,
        gradient_checkpointing=training_config.gradient_checkpointing,
    )


def compute_block_repeats(block_pattern: str, model_size: str) -> int:
    """
    Compute appropriate block_repeats based on model size.

    Strategy:
    - Target a reasonable number of layers based on model size
    - Use compute_model_dimensions to get the layer count a transformer would use
    - Divide by pattern length to get block repeats

    Returns:
        Number of block repeats
    """
    target_params = parse_model_size(model_size)
    pattern_len = len(block_pattern)

    # Use compute_model_dimensions to get the standard layer count for this size
    _, computed_layers, _, _, _ = compute_model_dimensions(target_params)

    # Compute repeats, ensuring at least 1
    block_repeats = max(1, computed_layers // pattern_len)

    return block_repeats


def get_hybrid_config(
    training_config: TrainingConfig,
    block_pattern: str,
    block_repeats: int,
    mamba_d_state: int = 128,
    mamba_d_conv: int = 4,
    mamba_expand: int = 2,
    mamba_headdim: int = 64,
    rope_theta: float = 100000.0,
) -> HybridConfig:
    """Get Hybrid model configuration based on training config and pattern, computing dimensions dynamically"""
    # Auto-compute block_repeats if -1
    if block_repeats == -1:
        block_repeats = compute_block_repeats(block_pattern, training_config.model_size)
        print(f"Auto-computed block_repeats={block_repeats} for pattern '{block_pattern}' at size {training_config.model_size}")

    # Compute dimensions for target size
    target_params = parse_model_size(training_config.model_size)
    vocab_size = 151936

    # Adjust target params based on number of blocks in pattern
    # Since hybrid has different block types with different param counts,
    # we use transformer dimensions as base and let the actual param count vary
    hidden_size, _, num_heads, num_kv_heads, intermediate_size = compute_model_dimensions(
        target_params, vocab_size=vocab_size
    )

    return HybridConfig(
        d_model=hidden_size,
        vocab_size=vocab_size,
        block_pattern=block_pattern,
        block_repeats=block_repeats,
        mamba_d_state=mamba_d_state,
        mamba_d_conv=mamba_d_conv,
        mamba_expand=mamba_expand,
        mamba_headdim=mamba_headdim,
        deltanet_head_dim=64,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        max_position_embeddings=training_config.max_seq_length,
        rope_theta=rope_theta,
        intermediate_size=intermediate_size,
        norm_eps=1e-6,
        gradient_checkpointing=training_config.gradient_checkpointing,
    )


def get_model_type(training_config: TrainingConfig, block_pattern: Optional[str] = None) -> str:
    """Determine the model type based on config and block pattern."""
    # If block_pattern is specified, it's a hybrid model
    if block_pattern:
        return "hybrid"

    has_mamba = training_config.mamba_type == MambaType.MAMBA2
    has_deltanet = training_config.mamba_type == MambaType.MAMBA3  # Repurposing MAMBA3 for deltanet

    if has_mamba:
        return "mamba2"
    elif has_deltanet:
        return "deltanet"
    else:
        return "transformer"
