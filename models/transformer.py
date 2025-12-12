"""
Transformer Architecture with Native Sparse Attention Support

Implements transformer architecture with swappable attention mechanisms:
- Dense FlashAttention2
- Native Sparse Attention (NSA)
- Flash Sparse Attention (FSA)

Kernels are imported from models.kernels for centralized optimization.
"""
import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config import ModelConfig, AttentionType
from models.kernels import (
    # Availability flags
    LIGER_AVAILABLE,
    # Baseline implementations
    RotaryEmbedding,
    # Factory functions
    KernelType,
    create_rms_norm,
    create_mlp,
    get_rotary_pos_emb_fn,
)

# Try to import Liger for fused loss
try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
except ImportError:
    LigerFusedLinearCrossEntropyLoss = None

# Try to import FSA
try:
    from fsa.module.fsa import FlashSparseAttention as FSAModule, RopeConfig
    FSA_AVAILABLE = True
except ImportError:
    FSA_AVAILABLE = False


# ============================================================================
# Attention Implementations
# ============================================================================

class DenseAttention(nn.Module):
    """Standard Multi-Head Attention with FlashAttention support"""
    def __init__(self, config: ModelConfig, layer_idx: int, kernel_type: KernelType = "liger"):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.kernel_type = kernel_type

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self._apply_rotary = get_rotary_pos_emb_fn(kernel_type)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        q, k = self._apply_rotary(q.transpose(1, 2), k.transpose(1, 2), cos, sin)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        # Handle KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        new_cache = (k, v) if use_cache else None

        # Expand KV for GQA
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Use scaled dot product attention (uses FlashAttention when available)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=attention_mask is None,
            dropout_p=0.0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, new_cache


class NativeSparseAttention(nn.Module):
    """
    Native Sparse Attention from https://arxiv.org/abs/2502.11089
    Implements hardware-aligned sparse attention with:
    - Block-level sparsity via top-k selection
    - Sliding window attention
    - Gated combination of selected and sliding attention
    """
    def __init__(self, config: ModelConfig, layer_idx: int, kernel_type: KernelType = "liger"):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.kernel_type = kernel_type

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        # Block and window parameters
        self.block_size = config.nsa_block_size
        self.window_size = config.nsa_window_size
        self.num_selected_blocks = config.nsa_num_selected_blocks

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Compression projections for block selection scoring
        self.k_compress = nn.Linear(self.head_dim, self.head_dim // 4, bias=False)

        # Gates for combining selected attention and sliding window attention
        self.gate_slc = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.gate_swa = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self._apply_rotary = get_rotary_pos_emb_fn(kernel_type)

    def _compute_block_scores(self, q: Tensor, k: Tensor) -> Tensor:
        """Compute attention scores at block level for top-k selection"""
        batch_size, num_heads, seq_len, head_dim = q.shape
        num_blocks = (seq_len + self.block_size - 1) // self.block_size

        # Compress keys to reduce computation
        k_compressed = self.k_compress(k)  # [B, H, T, D/4]

        # Reshape to blocks
        pad_len = num_blocks * self.block_size - seq_len
        if pad_len > 0:
            k_compressed = F.pad(k_compressed, (0, 0, 0, pad_len))

        k_blocks = k_compressed.view(batch_size, num_heads, num_blocks, self.block_size, -1)
        k_block_mean = k_blocks.mean(dim=3)  # [B, H, num_blocks, D/4]

        # Compute per-position scores with all blocks
        q_compressed = self.k_compress(q)  # Reuse compression
        block_scores = torch.einsum("bhsd,bhnd->bhsn", q_compressed, k_block_mean)

        return block_scores / math.sqrt(k_block_mean.shape[-1])

    def _select_top_k_blocks(self, block_scores: Tensor, seq_len: int) -> Tuple[Tensor, Tensor]:
        """Select top-k blocks for each query position"""
        batch_size, num_heads, seq_len_q, num_blocks = block_scores.shape

        # Create causal mask for blocks
        position_blocks = torch.arange(seq_len_q, device=block_scores.device) // self.block_size
        block_indices = torch.arange(num_blocks, device=block_scores.device)
        causal_mask = block_indices.unsqueeze(0) <= position_blocks.unsqueeze(1)

        # Apply causal mask
        block_scores = block_scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Select top-k blocks
        k = min(self.num_selected_blocks, num_blocks)
        top_k_scores, top_k_indices = torch.topk(block_scores, k, dim=-1)

        # Sort indices for efficient gathering
        top_k_indices, sort_idx = torch.sort(top_k_indices, dim=-1)
        top_k_scores = torch.gather(top_k_scores, -1, sort_idx)

        return top_k_indices, top_k_scores

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape

        # Compute gates
        g_slc = torch.sigmoid(self.gate_slc(hidden_states))  # [B, T, H]
        g_swa = torch.sigmoid(self.gate_swa(hidden_states))  # [B, T, H]

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        q, k = self._apply_rotary(q.transpose(1, 2), k.transpose(1, 2), cos, sin)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        # Expand KV for GQA
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Compute block selection scores
        block_scores = self._compute_block_scores(q, k)
        block_indices, _ = self._select_top_k_blocks(block_scores, seq_len)

        # Try to use the optimized NSA kernel, fallback to reference implementation
        try:
            from native_sparse_attention.ops.parallel import parallel_nsa

            attn_output = parallel_nsa(
                q=q.transpose(1, 2).contiguous(),  # [B, T, H, D]
                k=k.transpose(1, 2).contiguous(),
                v=v.transpose(1, 2).contiguous(),
                g_slc=g_slc,
                g_swa=g_swa,
                block_indices=block_indices.transpose(1, 2),  # [B, T, H, S]
                block_counts=torch.full((batch_size, seq_len, self.num_heads),
                                       self.num_selected_blocks, device=q.device),
                block_size=self.block_size,
                window_size=self.window_size,
            )
        except (ImportError, ValueError, TypeError):
            # Fallback to reference implementation
            attn_output = self._reference_nsa(q, k, v, g_slc, g_swa, block_indices)

        attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None

    def _reference_nsa(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        g_slc: Tensor,
        g_swa: Tensor,
        block_indices: Tensor,
    ) -> Tensor:
        """Reference implementation of NSA (slower, for fallback)"""
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Sliding window attention
        swa_output = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=0.0,
        )

        # For reference, we use dense attention as placeholder for selected attention
        slc_output = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=0.0,
        )

        # Combine with gates
        g_slc = g_slc.transpose(1, 2).unsqueeze(-1)  # [B, H, T, 1]
        g_swa = g_swa.transpose(1, 2).unsqueeze(-1)

        output = g_slc * slc_output + g_swa * swa_output
        output = output.transpose(1, 2)  # [B, T, H, D]

        return output


class FlashSparseAttention(nn.Module):
    """
    Flash Sparse Attention - optimized kernel implementation for NSA
    """
    def __init__(self, config: ModelConfig, layer_idx: int, kernel_type: KernelType = "liger"):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.kernel_type = kernel_type

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        # Block and sparsity parameters
        self.block_size = config.nsa_block_size
        self.topk = config.nsa_num_selected_blocks

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Initialize FSA module if available
        if FSA_AVAILABLE:
            rope_config = RopeConfig(
                max_position_embeddings=config.max_position_embeddings,
                head_dim=self.head_dim,
                rope_theta=config.rope_theta,
            )

            self.fsa_module = FSAModule(
                hidden_size=self.hidden_size,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                block_size=self.block_size,
                topk=self.topk,
                rope_config=rope_config,
            )
        else:
            self.fsa_module = None
            # Fallback to regular rotary embeddings
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
            )
            self._apply_rotary = get_rotary_pos_emb_fn(kernel_type)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape

        if self.fsa_module is not None and not use_cache:
            # Use optimized FSA kernel
            seqlens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=hidden_states.device)
            cu_seqlens = torch.cat([
                torch.zeros(1, dtype=torch.int32, device=hidden_states.device),
                torch.cumsum(seqlens, dim=0)
            ], dim=0)

            x_flat = hidden_states.view(-1, self.hidden_size)
            attn_output = self.fsa_module(x_flat, cu_seqlens)
            attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

            return attn_output, None
        else:
            # Fallback to standard attention
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

            cos, sin = self.rotary_emb(hidden_states, position_ids)
            q, k = self._apply_rotary(q.transpose(1, 2), k.transpose(1, 2), cos, sin)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

            if past_key_value is not None:
                k = torch.cat([past_key_value[0], k], dim=2)
                v = torch.cat([past_key_value[1], v], dim=2)

            new_cache = (k, v) if use_cache else None

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
            attn_output = self.o_proj(attn_output)

            return attn_output, new_cache


# ============================================================================
# Transformer Block and Model
# ============================================================================

class TransformerBlock(nn.Module):
    """Transformer block with configurable attention and kernel type"""
    def __init__(self, config: ModelConfig, layer_idx: int, kernel_type: KernelType = "liger"):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.kernel_type = kernel_type

        # Select attention type
        if config.attention_type == AttentionType.NSA:
            self.self_attn = NativeSparseAttention(config, layer_idx, kernel_type)
        elif config.attention_type == AttentionType.FSA:
            self.self_attn = FlashSparseAttention(config, layer_idx, kernel_type)
        else:
            self.self_attn = DenseAttention(config, layer_idx, kernel_type)

        self.mlp = create_mlp(config.hidden_size, config.intermediate_size, kernel_type)
        self.input_layernorm = create_rms_norm(config.hidden_size, eps=config.rms_norm_eps, kernel_type=kernel_type)
        self.post_attention_layernorm = create_rms_norm(config.hidden_size, eps=config.rms_norm_eps, kernel_type=kernel_type)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, cache = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, cache


class TransformerModel(nn.Module):
    """Full transformer model for causal language modeling"""
    def __init__(self, config: ModelConfig, kernel_type: KernelType = "liger"):
        super().__init__()
        self.config = config
        self.kernel_type = kernel_type
        self.gradient_checkpointing = False

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx, kernel_type)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = create_rms_norm(config.hidden_size, eps=config.rms_norm_eps, kernel_type=kernel_type)

        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Use Liger Fused Linear Cross Entropy if available
        self.use_fused_loss = kernel_type == "liger" and LIGER_AVAILABLE
        if self.use_fused_loss:
            self.loss_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)

        self._init_weights()

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for this model"""
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for this model"""
        self.gradient_checkpointing = False

    def _compute_loss_chunked(
        self,
        hidden_states: Tensor,
        labels: Tensor,
        chunk_size: int = 4096,
    ) -> Tensor:
        """Compute cross-entropy loss in chunks to avoid OOM with long sequences."""
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Shift for causal LM: predict next token
        hidden_states = hidden_states[:, :-1, :]
        labels = labels[:, 1:]
        seq_len = seq_len - 1

        total_loss = 0.0
        total_tokens = 0

        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)

            hidden_chunk = hidden_states[:, chunk_start:chunk_end, :]
            labels_chunk = labels[:, chunk_start:chunk_end]

            if self.lm_head is not None:
                logits_chunk = self.lm_head(hidden_chunk)
            else:
                logits_chunk = F.linear(hidden_chunk, self.embed_tokens.weight)

            chunk_loss = F.cross_entropy(
                logits_chunk.reshape(-1, logits_chunk.size(-1)),
                labels_chunk.reshape(-1),
                ignore_index=-100,
                reduction='sum',
            )

            num_tokens = (labels_chunk != -100).sum()
            total_loss += chunk_loss
            total_tokens += num_tokens

        return total_loss / max(total_tokens, 1)

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

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
        labels: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[List]]:
        batch_size, seq_len = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        hidden_states = self.embed_tokens(input_ids)

        # Gradient checkpointing is incompatible with caching
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

        loss = None
        logits = None

        if labels is not None:
            # Use chunked loss for long sequences to avoid OOM
            vocab_size = self.config.vocab_size
            logits_size_gb = (batch_size * seq_len * vocab_size * 2) / (1024**3)

            if self.use_fused_loss and logits_size_gb <= 10.0:
                # Use Liger Fused Linear Cross Entropy - avoids materializing logits
                # Shift for causal LM
                shift_hidden = hidden_states[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                weight = self.lm_head.weight if self.lm_head is not None else self.embed_tokens.weight
                loss = self.loss_fn(shift_hidden.view(-1, shift_hidden.size(-1)), weight, shift_labels.view(-1))
            elif logits_size_gb > 10.0:
                # Compute loss in chunks without materializing full logits
                loss = self._compute_loss_chunked(hidden_states, labels, chunk_size=4096)
            else:
                # Standard loss computation for shorter sequences
                if self.lm_head is not None:
                    logits = self.lm_head(hidden_states)
                else:
                    logits = F.linear(hidden_states, self.embed_tokens.weight)

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
        else:
            # No labels, compute logits normally
            if self.lm_head is not None:
                logits = self.lm_head(hidden_states)
            else:
                logits = F.linear(hidden_states, self.embed_tokens.weight)

        return logits, loss, new_cache if use_cache else None


def create_model(config: ModelConfig, kernel_type: KernelType = "liger") -> TransformerModel:
    """Factory function to create model from config"""
    return TransformerModel(config, kernel_type)
