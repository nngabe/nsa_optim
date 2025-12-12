"""
Gated Delta Net implementation using flash-linear-attention library.

Gated Delta Nets are a variant of linear attention with:
- Delta rule update mechanism for recurrent state
- Gating for selective memory updates
- Sub-quadratic complexity O(n) for sequence length
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

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


@dataclass
class GatedDeltaNetConfig:
    """Configuration for Gated Delta Net model"""
    d_model: int = 1024
    n_layers: int = 24
    d_state: int = 128
    head_dim: int = 64
    expand_v: float = 2.0
    vocab_size: int = 151936
    use_gate: bool = True
    use_short_conv: bool = True
    conv_size: int = 4
    norm_eps: float = 1e-6
    gradient_checkpointing: bool = False


class GatedDeltaNetBlock(nn.Module):
    """
    Single Gated Delta Net block with MLP.

    Architecture:
    - Pre-norm with RMSNorm
    - GatedDeltaNet mixer
    - Post-norm with RMSNorm
    - MLP with SwiGLU activation
    """
    def __init__(self, config: GatedDeltaNetConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Calculate num_heads from d_model and head_dim
        num_heads = config.d_model // config.head_dim

        # Pre-norm
        self.input_layernorm = nn.RMSNorm(config.d_model, eps=config.norm_eps)

        # GatedDeltaNet mixer from flash-linear-attention
        self.mixer = GatedDeltaNet(
            hidden_size=config.d_model,
            expand_v=config.expand_v,
            head_dim=config.head_dim,
            num_heads=num_heads,
            mode='chunk',  # Use chunked mode for training efficiency
            use_gate=config.use_gate,
            use_short_conv=config.use_short_conv,
            conv_size=config.conv_size,
            layer_idx=layer_idx,
            norm_eps=config.norm_eps,
        )

        # Post-attention norm
        self.post_attention_layernorm = nn.RMSNorm(config.d_model, eps=config.norm_eps)

        # MLP with SwiGLU
        self.intermediate_size = int(config.d_model * 8 / 3)  # SwiGLU uses 8/3 expansion
        self.intermediate_size = ((self.intermediate_size + 63) // 64) * 64  # Round to 64

        self.gate_proj = nn.Linear(config.d_model, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.d_model, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, config.d_model, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        # Pre-norm and mixer
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # GatedDeltaNet forward
        # The mixer returns (output, new_state) when use_cache=True
        mixer_output = self.mixer(hidden_states)
        if isinstance(mixer_output, tuple):
            hidden_states, new_cache = mixer_output
        else:
            hidden_states = mixer_output
            new_cache = None

        hidden_states = residual + hidden_states

        # MLP with SwiGLU
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        hidden_states = residual + hidden_states

        return hidden_states, new_cache


class GatedDeltaNetModel(nn.Module):
    """
    Full Gated Delta Net model for causal language modeling.

    Uses flash-linear-attention's GatedDeltaNet layers for efficient
    sequence modeling with linear complexity.
    """
    def __init__(self, config: GatedDeltaNetConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = config.gradient_checkpointing

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        # Stack of GatedDeltaNet blocks
        self.layers = nn.ModuleList([
            GatedDeltaNetBlock(config, layer_idx)
            for layer_idx in range(config.n_layers)
        ])

        # Final norm
        self.norm = nn.RMSNorm(config.d_model, eps=config.norm_eps)

        # LM head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small random values"""
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
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[list]]:
        batch_size, seq_len = input_ids.shape

        hidden_states = self.embed_tokens(input_ids)

        new_cache = [] if use_cache else None

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
                    past,
                    use_cache,
                    use_reentrant=False,
                )
            else:
                hidden_states, cache = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past,
                    use_cache=use_cache,
                )

            if use_cache:
                new_cache.append(cache)

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

        return logits, loss, new_cache


def create_gated_deltanet(config: GatedDeltaNetConfig) -> GatedDeltaNetModel:
    """Factory function to create GatedDeltaNet model from config"""
    return GatedDeltaNetModel(config)
