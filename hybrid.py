"""
Hybrid model supporting interleaved blocks of Mamba, Attention, and Gated Delta Nets.

This module enables flexible hybrid architectures with patterns like:
- 'MMDMMA' - 2 Mamba, 1 DeltaNet, 2 Mamba, 1 Attention
- Patterns can be repeated N times with --block_repeats

Block types:
- M: Mamba2 block (state space model)
- D: Gated Delta Net block (linear attention with delta rule)
- A: Attention block (standard transformer attention)
"""
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mamba_ssm import Mamba2
from fla.layers import GatedDeltaNet


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
    deltanet_expand_v: float = 2.0
    deltanet_use_gate: bool = True
    deltanet_use_short_conv: bool = True
    deltanet_conv_size: int = 4

    # Attention config
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    max_position_embeddings: int = 131072
    rope_theta: float = 10000.0

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


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
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
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

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


def rotate_half(x: Tensor) -> Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
    """Apply rotary positional embeddings to query and key tensors."""
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MambaBlock(nn.Module):
    """Mamba2 block for hybrid model"""
    def __init__(self, config: HybridConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.block_type = 'M'

        self.norm = nn.RMSNorm(config.d_model, eps=config.norm_eps)
        self.mixer = Mamba2(
            d_model=config.d_model,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            expand=config.mamba_expand,
            headdim=config.mamba_headdim,
            layer_idx=layer_idx,
        )

        self.post_mixer_norm = nn.RMSNorm(config.d_model, eps=config.norm_eps)
        self.gate_proj = nn.Linear(config.d_model, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.d_model, bias=False)

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
        hidden_states = self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        hidden_states = residual + hidden_states

        return hidden_states, None


class DeltaNetBlock(nn.Module):
    """Gated Delta Net block for hybrid model"""
    def __init__(self, config: HybridConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.block_type = 'D'

        num_heads = config.d_model // config.deltanet_head_dim

        self.norm = nn.RMSNorm(config.d_model, eps=config.norm_eps)
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

        self.post_mixer_norm = nn.RMSNorm(config.d_model, eps=config.norm_eps)
        self.gate_proj = nn.Linear(config.d_model, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.d_model, bias=False)

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
        hidden_states = self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        hidden_states = residual + hidden_states

        return hidden_states, None


class AttentionBlock(nn.Module):
    """Standard attention block for hybrid model"""
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

        self.norm = nn.RMSNorm(config.d_model, eps=config.norm_eps)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        self.post_attn_norm = nn.RMSNorm(config.d_model, eps=config.norm_eps)
        self.gate_proj = nn.Linear(config.d_model, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.d_model, bias=False)

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

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        q, k = apply_rotary_pos_emb(q.transpose(1, 2), k.transpose(1, 2), cos, sin)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=attention_mask is None,
            dropout_p=0.0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        hidden_states = self.o_proj(attn_output)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attn_norm(hidden_states)
        hidden_states = self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
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

        # Final norm
        self.norm = nn.RMSNorm(config.d_model, eps=config.norm_eps)

        # LM head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
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
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss, None


def create_hybrid_model(config: HybridConfig) -> HybridModel:
    """Factory function to create Hybrid model from config"""
    return HybridModel(config)


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
