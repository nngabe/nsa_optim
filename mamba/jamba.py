"""
Jamba: Hybrid Mamba-Transformer Architecture

Jamba interleaves Mamba blocks (SSM-based) with Transformer attention blocks
at a configurable ratio. This allows combining the strengths of both architectures:
- Mamba: Linear complexity, efficient long-range modeling, fast inference
- Transformer: Strong local attention, well-understood optimization

Reference: Based on the Jamba architecture from AI21 Labs
"""

import math
from typing import Optional, Tuple, List, Union, Literal
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Import attention modules from model.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import (
    DenseAttention,
    NativeSparseAttention,
    FlashSparseAttention,
    MLP,
    RMSNorm,
    RotaryEmbedding,
    apply_rotary_pos_emb,
)
from config import ModelConfig, AttentionType

# Import Mamba modules
from .mamba2 import Mamba2Mixer, Mamba2Config
from .mamba3_triton import Mamba3Mixer, Mamba3Config


@dataclass
class JambaConfig:
    """Configuration for Jamba hybrid model."""
    # Model dimensions
    d_model: int = 2048
    n_layers: int = 32
    vocab_size: int = 151936

    # Attention configuration
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # For GQA
    attn_type: Literal["dense", "nsa", "fsa"] = "dense"

    # NSA specific
    nsa_block_size: int = 64
    nsa_window_size: int = 64
    nsa_num_selected_blocks: int = 16

    # Mamba configuration
    mamba_type: Literal["mamba2", "mamba3"] = "mamba2"
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64

    # Jamba-specific: ratio of mamba blocks to attention blocks
    # jamba_ratio = 7 means 7 mamba blocks per 1 attention block
    jamba_ratio: int = 7

    # MLP
    intermediate_size: int = None  # Auto-computed if None

    # Other
    max_position_embeddings: int = 32768
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    bias: bool = False
    tie_word_embeddings: bool = True
    gradient_checkpointing: bool = False

    def __post_init__(self):
        if self.intermediate_size is None:
            # Default to ~8/3 * d_model, rounded to multiple of 256
            self.intermediate_size = ((int(self.d_model * 8 / 3) + 255) // 256) * 256


def get_block_pattern(n_layers: int, jamba_ratio: int) -> List[str]:
    """
    Generate the pattern of block types (mamba vs attention).

    With jamba_ratio=7, the pattern is:
    [M, M, M, M, M, M, M, A, M, M, M, M, M, M, M, A, ...]

    Args:
        n_layers: Total number of layers
        jamba_ratio: Number of mamba blocks per attention block

    Returns:
        List of 'mamba' or 'attention' for each layer
    """
    pattern = []
    cycle_length = jamba_ratio + 1  # e.g., 7 mamba + 1 attention = 8

    for i in range(n_layers):
        pos_in_cycle = i % cycle_length
        if pos_in_cycle == jamba_ratio:  # Last position in cycle is attention
            pattern.append('attention')
        else:
            pattern.append('mamba')

    return pattern


class JambaMambaBlock(nn.Module):
    """Mamba block for Jamba architecture."""

    def __init__(
        self,
        config: JambaConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.norm1 = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # Create appropriate Mamba mixer
        if config.mamba_type == "mamba2":
            self.mixer = Mamba2Mixer(
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
                headdim=config.headdim,
                bias=config.bias,
                rms_norm_eps=config.rms_norm_eps,
                layer_idx=layer_idx,
            )
        else:  # mamba3
            mamba3_config = Mamba3Config(
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
                head_dim=config.headdim,
                bias=config.bias,
                rms_norm_eps=config.rms_norm_eps,
            )
            self.mixer = Mamba3Mixer(mamba3_config, layer_idx=layer_idx)

        self.norm2 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.mlp = MLP(self._get_model_config())

    def _get_model_config(self) -> ModelConfig:
        """Create a ModelConfig for the MLP."""
        return ModelConfig(
            name="jamba",
            hidden_size=self.config.d_model,
            num_hidden_layers=self.config.n_layers,
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_heads=self.config.num_key_value_heads,
            intermediate_size=self.config.intermediate_size,
            vocab_size=self.config.vocab_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        # Mamba mixer with residual
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states, cache = self.mixer(hidden_states, cache=past_key_value)
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, cache


class JambaAttentionBlock(nn.Module):
    """Attention block for Jamba architecture."""

    def __init__(
        self,
        config: JambaConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        model_config = self._get_model_config()

        self.norm1 = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # Create appropriate attention
        if config.attn_type == "nsa":
            model_config.attention_type = AttentionType.NSA
            self.self_attn = NativeSparseAttention(model_config, layer_idx)
        elif config.attn_type == "fsa":
            model_config.attention_type = AttentionType.FSA
            self.self_attn = FlashSparseAttention(model_config, layer_idx)
        else:  # dense
            model_config.attention_type = AttentionType.DENSE
            self.self_attn = DenseAttention(model_config, layer_idx)

        self.norm2 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.mlp = MLP(model_config)

    def _get_model_config(self) -> ModelConfig:
        """Create a ModelConfig for attention and MLP."""
        return ModelConfig(
            name="jamba",
            hidden_size=self.config.d_model,
            num_hidden_layers=self.config.n_layers,
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_heads=self.config.num_key_value_heads,
            intermediate_size=self.config.intermediate_size,
            vocab_size=self.config.vocab_size,
            max_position_embeddings=self.config.max_position_embeddings,
            rope_theta=self.config.rope_theta,
            rms_norm_eps=self.config.rms_norm_eps,
            nsa_block_size=self.config.nsa_block_size,
            nsa_window_size=self.config.nsa_window_size,
            nsa_num_selected_blocks=self.config.nsa_num_selected_blocks,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Attention with residual
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states, cache = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, cache


class JambaModel(nn.Module):
    """
    Jamba: Hybrid Mamba-Transformer Model

    Interleaves Mamba blocks and Attention blocks according to the jamba_ratio.
    Default ratio of 7:1 means 7 Mamba blocks followed by 1 Attention block.
    """

    def __init__(self, config: JambaConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = config.gradient_checkpointing

        # Embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        # Get block pattern
        self.block_pattern = get_block_pattern(config.n_layers, config.jamba_ratio)

        # Create layers based on pattern
        self.layers = nn.ModuleList()
        for layer_idx, block_type in enumerate(self.block_pattern):
            if block_type == 'mamba':
                self.layers.append(JambaMambaBlock(config, layer_idx))
            else:
                self.layers.append(JambaAttentionBlock(config, layer_idx))

        # Final norm
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # LM head
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing."""
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple]] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List]]:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len)
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            past_key_values: Optional list of cached key/values
            use_cache: Whether to return cache
            labels: Optional labels for computing loss

        Returns:
            logits: Output logits
            loss: Loss if labels provided
            new_cache: Updated cache if use_cache=True
        """
        batch_size, seq_len = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        hidden_states = self.embed_tokens(input_ids)

        # Gradient checkpointing incompatible with caching
        if self.gradient_checkpointing and use_cache:
            use_cache = False

        new_cache = []
        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if past_key_values else None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                hidden_states, cache = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past,
                    use_cache,
                    use_reentrant=False,
                )
            else:
                hidden_states, cache = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past,
                    use_cache=use_cache,
                )

            if use_cache:
                new_cache.append(cache)

        hidden_states = self.norm(hidden_states)

        # Compute logits
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            logits = F.linear(hidden_states, self.embed_tokens.weight)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss, new_cache if use_cache else None

    def get_block_info(self) -> str:
        """Get a string describing the block pattern."""
        mamba_count = sum(1 for b in self.block_pattern if b == 'mamba')
        attn_count = sum(1 for b in self.block_pattern if b == 'attention')
        return (f"Jamba Model: {self.config.n_layers} layers "
                f"({mamba_count} Mamba, {attn_count} Attention, "
                f"ratio {self.config.jamba_ratio}:1)")


def create_jamba(config: JambaConfig) -> JambaModel:
    """Factory function to create Jamba model from config."""
    return JambaModel(config)


def create_jamba_from_model_config(
    model_config: ModelConfig,
    mamba_type: str = "mamba2",
    jamba_ratio: int = 7,
    d_state: int = 128,
    gradient_checkpointing: bool = False,
) -> JambaModel:
    """
    Create a Jamba model from an existing ModelConfig.

    This is useful for creating Jamba variants that match
    existing transformer configurations.
    """
    # Map attention type
    if model_config.attention_type == AttentionType.NSA:
        attn_type = "nsa"
    elif model_config.attention_type == AttentionType.FSA:
        attn_type = "fsa"
    else:
        attn_type = "dense"

    jamba_config = JambaConfig(
        d_model=model_config.hidden_size,
        n_layers=model_config.num_hidden_layers,
        vocab_size=model_config.vocab_size,
        num_attention_heads=model_config.num_attention_heads,
        num_key_value_heads=model_config.num_key_value_heads,
        attn_type=attn_type,
        nsa_block_size=model_config.nsa_block_size,
        nsa_window_size=model_config.nsa_window_size,
        nsa_num_selected_blocks=model_config.nsa_num_selected_blocks,
        mamba_type=mamba_type,
        d_state=d_state,
        jamba_ratio=jamba_ratio,
        intermediate_size=model_config.intermediate_size,
        max_position_embeddings=model_config.max_position_embeddings,
        rope_theta=model_config.rope_theta,
        rms_norm_eps=model_config.rms_norm_eps,
        gradient_checkpointing=gradient_checkpointing,
    )

    return JambaModel(jamba_config)


# ============================================================================
# Test / Demo
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Jamba Implementation Test")
    print("=" * 60)

    # Test configuration
    config = JambaConfig(
        d_model=256,
        n_layers=16,
        vocab_size=1000,
        num_attention_heads=8,
        num_key_value_heads=4,
        mamba_type="mamba2",
        jamba_ratio=7,
    )

    print(f"\nConfig: d_model={config.d_model}, n_layers={config.n_layers}")
    print(f"Mamba type: {config.mamba_type}, Attention type: {config.attn_type}")
    print(f"Jamba ratio: {config.jamba_ratio}:1 (mamba:attention)")

    # Create model
    model = JambaModel(config)
    print(f"\n{model.get_block_info()}")

    # Show block pattern
    print("\nBlock pattern (first 16 layers):")
    pattern = get_block_pattern(16, config.jamba_ratio)
    for i, p in enumerate(pattern):
        marker = "M" if p == "mamba" else "A"
        print(f"  Layer {i}: {marker} ({p})")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")

    # Test forward pass
    batch_size = 2
    seq_len = 128
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    print(f"\nInput shape: {x.shape}")

    # Forward pass
    logits, loss, _ = model(x, labels=labels)
    print(f"Output logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
