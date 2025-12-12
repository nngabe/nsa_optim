"""
Models package for NSA + Optimizer Ablation Study

Contains:
- TransformerModel: Standard transformer with attention variants (Dense, NSA, FSA)
- Mamba2Model: Mamba-2 state space model
- GatedDeltaNetModel: Gated Delta Net linear attention
- HybridModel: Hybrid model with interleaved blocks
"""

from models.transformer import (
    TransformerModel,
    TransformerBlock,
    DenseAttention,
    NativeSparseAttention,
    FlashSparseAttention,
    RMSNorm,
    RotaryEmbedding,
    MLP,
    create_model,
    create_rms_norm,
    create_mlp,
    apply_rotary_pos_emb,
    rotate_half,
    get_rotary_pos_emb_fn,
    LIGER_AVAILABLE,
    FSA_AVAILABLE,
    TRITON_AVAILABLE,
    KernelType,
)

from models.mamba import (
    Mamba2Model,
    Mamba2Config,
    Mamba2Block,
    create_mamba2,
)

from models.deltanet import (
    GatedDeltaNetModel,
    GatedDeltaNetConfig,
    GatedDeltaNetBlock,
    create_gated_deltanet,
)

from models.hybrid import (
    HybridModel,
    HybridConfig,
    MambaBlock,
    DeltaNetBlock,
    AttentionBlock,
    create_hybrid_model,
    get_block_classes_from_pattern,
    print_hybrid_model_modules,
)

from models.configs import (
    get_model_config,
    get_mamba2_config,
    get_deltanet_config,
    get_hybrid_config,
    get_model_type,
)

__all__ = [
    # Transformer
    "TransformerModel",
    "TransformerBlock",
    "DenseAttention",
    "NativeSparseAttention",
    "FlashSparseAttention",
    "RMSNorm",
    "RotaryEmbedding",
    "MLP",
    "create_model",
    "create_rms_norm",
    "create_mlp",
    "apply_rotary_pos_emb",
    "rotate_half",
    "get_rotary_pos_emb_fn",
    "LIGER_AVAILABLE",
    "FSA_AVAILABLE",
    "TRITON_AVAILABLE",
    "KernelType",
    # Mamba
    "Mamba2Model",
    "Mamba2Config",
    "Mamba2Block",
    "create_mamba2",
    # DeltaNet
    "GatedDeltaNetModel",
    "GatedDeltaNetConfig",
    "GatedDeltaNetBlock",
    "create_gated_deltanet",
    # Hybrid
    "HybridModel",
    "HybridConfig",
    "MambaBlock",
    "DeltaNetBlock",
    "AttentionBlock",
    "create_hybrid_model",
    "get_block_classes_from_pattern",
    "print_hybrid_model_modules",
    # Config functions
    "get_model_config",
    "get_mamba2_config",
    "get_deltanet_config",
    "get_hybrid_config",
    "get_model_type",
]
