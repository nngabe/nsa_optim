"""
Mamba wrapper using the official mamba-ssm library.

This module provides a unified interface for Mamba2 models that matches
the interface expected by the training script.

Kernels are imported from models.kernels for centralized optimization.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mamba_ssm import Mamba2
from mamba_ssm.modules.block import Block
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from models.kernels import (
    create_rms_norm,
    create_mlp,
    compute_cross_entropy_loss,
    create_cross_entropy_loss,
)


@dataclass
class Mamba2Config:
    """Configuration for Mamba2 model wrapper"""
    d_model: int = 1024
    n_layers: int = 24
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    vocab_size: int = 151936
    use_triton: bool = True
    gradient_checkpointing: bool = False
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True


class Mamba2Block(nn.Module):
    """
    Single Mamba2 block with MLP.

    Uses the official mamba_ssm.Mamba2 module.
    Kernels are auto-selected: Liger -> Triton -> baseline.
    """
    def __init__(self, config: Mamba2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Pre-norm (uses Liger/Triton/baseline auto-selection)
        self.norm = create_rms_norm(config.d_model, eps=1e-6)

        # Mamba2 mixer from official library
        self.mixer = Mamba2(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            headdim=config.headdim,
            layer_idx=layer_idx,
        )

        # Post-mixer norm (uses Liger/Triton/baseline auto-selection)
        self.post_mixer_norm = create_rms_norm(config.d_model, eps=1e-6)

        # MLP with SwiGLU (uses Liger/Triton/baseline auto-selection)
        self.intermediate_size = int(config.d_model * 8 / 3)
        self.intermediate_size = ((self.intermediate_size + 63) // 64) * 64

        self.mlp = create_mlp(config.d_model, self.intermediate_size)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        inference_params=None,
    ) -> Tuple[Tensor, None]:
        # Pre-norm and Mamba2 mixer
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        hidden_states = residual + hidden_states

        # MLP with SwiGLU (uses optimized kernel)
        residual = hidden_states
        hidden_states = self.post_mixer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, None


class Mamba2Model(nn.Module):
    """
    Full Mamba2 model for causal language modeling.

    Uses the official mamba-ssm library's Mamba2 modules.
    Kernels are auto-selected: Liger -> Triton -> baseline.
    """
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = config.gradient_checkpointing

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        # Stack of Mamba2 blocks
        self.layers = nn.ModuleList([
            Mamba2Block(config, layer_idx)
            for layer_idx in range(config.n_layers)
        ])

        # Final norm (uses Liger/Triton/baseline auto-selection)
        self.norm = create_rms_norm(config.d_model, eps=1e-6)

        # LM head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Fused cross-entropy loss (Liger if available)
        self.loss_fn = create_cross_entropy_loss(self.lm_head)

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
        labels: Optional[Tensor] = None,
        inference_params=None,
    ) -> Tuple[Tensor, Optional[Tensor], None]:
        hidden_states = self.embed_tokens(input_ids)

        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                hidden_states, _ = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask,
                    inference_params,
                    use_reentrant=False,
                )
            else:
                hidden_states, _ = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    inference_params=inference_params,
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


def create_mamba2(config: Mamba2Config) -> Mamba2Model:
    """Factory function to create Mamba2 model from config"""
    return Mamba2Model(config)
