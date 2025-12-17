"""
Hybrid model supporting interleaved blocks of Mamba, Attention, and Gated Delta Nets.

This module enables flexible hybrid architectures with patterns like:
- 'MMDMMA' - 2 Mamba, 1 DeltaNet, 2 Mamba, 1 Attention
- Patterns can be repeated N times with --block_repeats

Block types:
- M: Mamba2 block (state space model)
- D: Gated Delta Net block (linear attention with delta rule)
- A: Attention block (standard transformer attention)

Optimizations (per CLAUDE.md):
- Uses Liger kernels for RMSNorm, SwiGLU, and RoPE
- Uses flash_attn for attention
- Hidden sizes are power of 2 for Triton optimization

Kernels are imported from models.kernels for centralized optimization.
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mamba_ssm import Mamba2

# Import optimized kernels from centralized module
from models.kernels import (
    LIGER_AVAILABLE,
    RotaryEmbedding,
    rotate_half,
    apply_rotary_pos_emb,
    create_rms_norm,
    create_mlp,
    compute_cross_entropy_loss,
    create_cross_entropy_loss,
)

# Workaround for fla library conflict with transformers
# The fla library tries to register 'bitnet' with AutoConfig/AutoModel/AutoModelForCausalLM
# but transformers already has these registered
def _import_fla_with_patch():
    """Import fla.layers while patching transformers auto-registration to allow overwrites."""
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        from transformers.models.auto.modeling_auto import (
            MODEL_MAPPING,
            MODEL_FOR_CAUSAL_LM_MAPPING,
        )

        # Save original register methods
        originals = {
            'config': CONFIG_MAPPING.register,
            'model': MODEL_MAPPING.register,
            'causal_lm': MODEL_FOR_CAUSAL_LM_MAPPING.register,
        }

        # Patch to allow exist_ok=True
        CONFIG_MAPPING.register = lambda k, v, exist_ok=False: originals['config'](k, v, exist_ok=True)
        MODEL_MAPPING.register = lambda k, v, exist_ok=False: originals['model'](k, v, exist_ok=True)
        MODEL_FOR_CAUSAL_LM_MAPPING.register = lambda k, v, exist_ok=False: originals['causal_lm'](k, v, exist_ok=True)

        from fla.layers import GatedDeltaNet

        # Restore original methods
        CONFIG_MAPPING.register = originals['config']
        MODEL_MAPPING.register = originals['model']
        MODEL_FOR_CAUSAL_LM_MAPPING.register = originals['causal_lm']

        return GatedDeltaNet
    except (ImportError, AttributeError):
        # Fallback if transformers structure changes
        from fla.layers import GatedDeltaNet
        return GatedDeltaNet

GatedDeltaNet = _import_fla_with_patch()

# Flash attention for optimized attention
try:
    from flash_attn import flash_attn_func
    from flash_attn.bert_padding import unpad_input, pad_input
    from flash_attn.layers.rotary import apply_rotary_emb as flash_rotary_emb
    from flash_attn.layers.rotary import RotaryEmbedding as FlashRotaryEmbedding
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    flash_rotary_emb = None
    FlashRotaryEmbedding = None


@dataclass
class HybridConfig:
    """Configuration for Hybrid model"""
    d_model: int = 1024
    vocab_size: int = 151936
    block_pattern: str = "MMDMMA"  # M=Mamba, D=DeltaNet, A=Attention
    block_repeats: int = 5  # Number of times to repeat the pattern

    # Mamba2 config
    mamba_d_state: int = 128
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_headdim: int = 64

    # DeltaNet config
    deltanet_head_dim: int = 64
    deltanet_expand_v: int = 2  # Must be int for nn.Linear compatibility
    deltanet_use_gate: bool = True
    deltanet_use_short_conv: bool = True
    deltanet_conv_size: int = 4

    # Attention config
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    max_position_embeddings: int = 131072
    rope_theta: float = 100000.0

    # Common config
    intermediate_size: int = None  # Computed from d_model if None
    norm_eps: float = 1e-6
    gradient_checkpointing: bool = False

    def __post_init__(self):
        if self.intermediate_size is None:
            # SwiGLU uses 8/3 expansion, rounded to nearest 64
            self.intermediate_size = int(self.d_model * 8 / 3)
            self.intermediate_size = ((self.intermediate_size + 63) // 64) * 64

    @property
    def n_layers(self) -> int:
        """Total number of layers"""
        return len(self.block_pattern) * self.block_repeats


class HybridRotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for hybrid model.

    Returns cos/sin in flash_attn compatible format: (seq_len, dim/2)
    This wraps the kernels.RotaryEmbedding but adapts output for flash_attn.
    """
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # shape: (seq_len, dim/2)
        # Store non-duplicated for flash_attn compatibility
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: Tensor, position_ids: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        seq_len = x.shape[1]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)

        if position_ids is not None:
            if position_ids.dim() == 2:
                position_ids = position_ids.squeeze(0)
            cos = self.cos_cached[position_ids]
            sin = self.sin_cached[position_ids]
        else:
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]

        return cos, sin


def apply_rotary_pos_emb_hybrid(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
    """Apply rotary positional embeddings for hybrid model (flash_attn compatible format).

    Args:
        q, k: (batch, seq, heads, head_dim)
        cos, sin: (seq, head_dim/2) - flash_attn compatible format

    Returns:
        q_embed, k_embed with RoPE applied
    """
    # Duplicate cos/sin for non-interleaved RoPE
    cos = torch.cat([cos, cos], dim=-1)  # (seq, head_dim)
    sin = torch.cat([sin, sin], dim=-1)  # (seq, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(2)  # (1, seq, 1, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(2)  # (1, seq, 1, head_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MambaBlock(nn.Module):
    """Mamba2 block for hybrid model (uses Liger RMSNorm + SwiGLU)"""
    def __init__(self, config: HybridConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.block_type = 'M'

        self.norm = create_rms_norm(config.d_model, eps=config.norm_eps)
        self.mixer = Mamba2(
            d_model=config.d_model,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            expand=config.mamba_expand,
            headdim=config.mamba_headdim,
            layer_idx=layer_idx,
        )

        self.post_mixer_norm = create_rms_norm(config.d_model, eps=config.norm_eps)
        self.mlp = create_mlp(config.d_model, config.intermediate_size)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, None]:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.mixer(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_mixer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, None


class DeltaNetBlock(nn.Module):
    """Gated Delta Net block for hybrid model (uses Liger RMSNorm + SwiGLU)"""
    def __init__(self, config: HybridConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.block_type = 'D'

        num_heads = config.d_model // config.deltanet_head_dim

        self.norm = create_rms_norm(config.d_model, eps=config.norm_eps)
        self.mixer = GatedDeltaNet(
            hidden_size=config.d_model,
            expand_v=config.deltanet_expand_v,
            head_dim=config.deltanet_head_dim,
            num_heads=num_heads,
            mode='chunk',
            use_gate=config.deltanet_use_gate,
            use_short_conv=config.deltanet_use_short_conv,
            conv_size=config.deltanet_conv_size,
            layer_idx=layer_idx,
            norm_eps=config.norm_eps,
        )

        self.post_mixer_norm = create_rms_norm(config.d_model, eps=config.norm_eps)
        self.mlp = create_mlp(config.d_model, config.intermediate_size)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, None]:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        mixer_output = self.mixer(hidden_states)
        if isinstance(mixer_output, tuple):
            hidden_states = mixer_output[0]
        else:
            hidden_states = mixer_output
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_mixer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, None


class AttentionBlock(nn.Module):
    """Attention block for hybrid model (uses Liger RMSNorm, SwiGLU, RoPE + flash_attn)"""
    def __init__(self, config: HybridConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.block_type = 'A'

        self.hidden_size = config.d_model
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.norm = create_rms_norm(config.d_model, eps=config.norm_eps)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = HybridRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        self.post_attn_norm = create_rms_norm(config.d_model, eps=config.norm_eps)
        self.mlp = create_mlp(config.d_model, config.intermediate_size)

    def _apply_rope(self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply RoPE using flash_attn's triton kernel if available, else fallback"""
        if FLASH_ATTN_AVAILABLE and flash_rotary_emb is not None:
            # flash_attn rotary expects (x, cos, sin) and works on each tensor separately
            # Input shape: (batch, seq, heads, head_dim)
            q = flash_rotary_emb(q, cos, sin)
            k = flash_rotary_emb(k, cos, sin)
            return q, k
        else:
            # Fallback uses (batch, seq, heads, head_dim) format
            return apply_rotary_pos_emb_hybrid(q, k, cos, sin)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, None]:
        batch_size, seq_len, _ = hidden_states.shape

        residual = hidden_states
        hidden_states = self.norm(hidden_states)

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for attention: (batch, seq, heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply RoPE
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        q, k = self._apply_rope(q, k, cos, sin)

        # Use flash_attn if available (expects batch, seq, heads, head_dim)
        if FLASH_ATTN_AVAILABLE:
            # flash_attn_func handles GQA internally with different num_heads
            attn_output = flash_attn_func(
                q, k, v,
                causal=True,
                softmax_scale=1.0 / (self.head_dim ** 0.5),
            )
        else:
            # Fallback to PyTorch SDPA (needs batch, heads, seq, head_dim)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            if self.num_kv_groups > 1:
                k = k.repeat_interleave(self.num_kv_groups, dim=1)
                v = v.repeat_interleave(self.num_kv_groups, dim=1)

            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                is_causal=attention_mask is None,
                dropout_p=0.0,
            )
            attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.contiguous().view(batch_size, seq_len, -1)
        hidden_states = self.o_proj(attn_output)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attn_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, None


def create_block(config: HybridConfig, block_type: str, layer_idx: int) -> nn.Module:
    """Create a block based on type"""
    if block_type == 'M':
        return MambaBlock(config, layer_idx)
    elif block_type == 'D':
        return DeltaNetBlock(config, layer_idx)
    elif block_type == 'A':
        return AttentionBlock(config, layer_idx)
    else:
        raise ValueError(f"Unknown block type: {block_type}. Use M (Mamba), D (DeltaNet), or A (Attention)")


class HybridModel(nn.Module):
    """
    Hybrid model with interleaved Mamba, DeltaNet, and Attention blocks.

    Example patterns:
    - 'MMDMMA' with block_repeats=5 creates 30 layers
    - 'MAD' with block_repeats=10 creates 30 layers
    """
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = config.gradient_checkpointing

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        # Build layers based on pattern
        self.layers = nn.ModuleList()
        full_pattern = config.block_pattern * config.block_repeats

        for layer_idx, block_type in enumerate(full_pattern):
            self.layers.append(create_block(config, block_type.upper(), layer_idx))

        # Final norm (Liger optimized)
        self.norm = create_rms_norm(config.d_model, eps=config.norm_eps)

        # LM head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Fused cross-entropy loss (Liger if available)
        self.loss_fn = create_cross_entropy_loss(self.lm_head)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling to prevent variance explosion.

        Uses GPT-2/LLaMA style initialization:
        - Base std of 0.02 for most weights
        - Output projections scaled by 1/sqrt(2*n_layers) to prevent variance accumulation
        - LM head initialized with smaller std
        """
        std = 0.02
        n_layers = len(self.layers)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Scale output projections (o_proj, down_proj) by 1/sqrt(2*n_layers)
                if any(proj in name for proj in ['o_proj', 'down_proj']):
                    nn.init.normal_(module.weight, mean=0.0, std=std / math.sqrt(2 * n_layers))
                elif 'lm_head' in name:
                    # LM head gets smaller init
                    nn.init.normal_(module.weight, mean=0.0, std=std / math.sqrt(n_layers))
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=std)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing"""
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], None]:
        batch_size, seq_len = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                hidden_states, _ = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    use_reentrant=False,
                )
            else:
                hidden_states, _ = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )

        hidden_states = self.norm(hidden_states)

        # Use fused linear + cross-entropy if available (Liger)
        if labels is not None:
            logits, loss = compute_cross_entropy_loss(
                hidden_states, self.lm_head, labels, self.loss_fn
            )
        else:
            logits = self.lm_head(hidden_states)
            loss = None

        return logits, loss, None


def create_hybrid_model(config: HybridConfig) -> HybridModel:
    """Factory function to create Hybrid model from config"""
    return HybridModel(config)


def print_hybrid_model_modules(model: HybridModel, block_pattern: str, block_repeats: int):
    """Print the module structure of a hybrid model for debugging.

    Shows each layer's type and the weight dimensions of its submodules.
    Useful for verifying hybrid model architecture with patterns like MMM, DDD, AAA, MMDMMA.
    """
    print("=" * 80)
    print(f"Hybrid Model Structure: pattern='{block_pattern}' x {block_repeats}")
    print(f"Total layers: {len(model.layers)}")
    print("=" * 80)

    for i, layer in enumerate(model.layers):
        block_type = getattr(layer, 'block_type', '?')
        block_name = layer.__class__.__name__
        print(f"\nLayer {i} ({block_type}): {block_name}")
        for name, child in layer.named_children():
            shape_info = ""
            if hasattr(child, 'weight') and child.weight is not None:
                shape_info = f" weight={tuple(child.weight.shape)}"
            elif hasattr(child, 'in_features'):
                shape_info = f" in={child.in_features}, out={child.out_features}"
            print(f"  └─ {name}: {child.__class__.__name__}{shape_info}")
    print("=" * 80)


def get_block_classes_from_pattern(pattern: str) -> set:
    """Get the set of block classes used in a pattern for FSDP wrapping"""
    classes = set()
    for char in pattern.upper():
        if char == 'M':
            classes.add(MambaBlock)
        elif char == 'D':
            classes.add(DeltaNetBlock)
        elif char == 'A':
            classes.add(AttentionBlock)
    return classes
