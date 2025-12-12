"""
Models package for NSA + Optimizer Ablation Study

Contains:
- TransformerModel: Standard transformer with attention variants (Dense, NSA, FSA)
- Mamba2Model: Mamba-2 state space model
- GatedDeltaNetModel: Gated Delta Net linear attention
- HybridModel: Hybrid model with interleaved blocks

Optimized kernels are centralized in models.kernels for:
- RMSNorm (Liger/Triton/baseline)
- SwiGLU MLP (Liger/Triton/baseline)
- RoPE (Liger/baseline)
- Cross-entropy loss (Liger fused/baseline)
"""

# Import kernels from centralized module
from models.kernels import (
    LIGER_AVAILABLE,
    TRITON_AVAILABLE,
    RMSNorm,
    RotaryEmbedding,
    MLP,
    rotate_half,
    apply_rotary_pos_emb,
    KernelType,
    create_rms_norm,
    create_mlp,
    get_rotary_pos_emb_fn,
    create_cross_entropy_loss,
    compute_cross_entropy_loss,
)

from models.transformer import (
    TransformerModel,
    TransformerBlock,
    DenseAttention,
    NativeSparseAttention,
    FlashSparseAttention,
    FSA_AVAILABLE,
    create_model,
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
    # Kernels (from models.kernels)
    "LIGER_AVAILABLE",
    "TRITON_AVAILABLE",
    "RMSNorm",
    "RotaryEmbedding",
    "MLP",
    "rotate_half",
    "apply_rotary_pos_emb",
    "KernelType",
    "create_rms_norm",
    "create_mlp",
    "get_rotary_pos_emb_fn",
    "create_cross_entropy_loss",
    "compute_cross_entropy_loss",
    # Transformer
    "TransformerModel",
    "TransformerBlock",
    "DenseAttention",
    "NativeSparseAttention",
    "FlashSparseAttention",
    "FSA_AVAILABLE",
    "create_model",
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
