"""
Optimized Kernel Implementations

Centralized kernel implementations with automatic backend selection:
- Liger kernels (preferred): LigerRMSNorm, LigerSwiGLUMLP, liger_rotary_pos_emb, LigerFusedLinearCrossEntropyLoss
- Triton kernels (fallback): Custom Triton implementations
- Baseline PyTorch (last resort): Pure PyTorch implementations

Usage:
    from models.kernels import create_rms_norm, create_mlp, create_cross_entropy_loss
"""
from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# Availability Flags
# ============================================================================

try:
    from liger_kernel.transformers import (
        LigerRMSNorm,
        LigerSwiGLUMLP,
        LigerFusedLinearCrossEntropyLoss,
        liger_rotary_pos_emb,
    )
    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# ============================================================================
# Baseline Implementations
# ============================================================================

class RMSNorm(nn.Module):
    """RMS Normalization - baseline PyTorch implementation"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) - baseline implementation"""
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
    """Apply rotary positional embeddings to query and key tensors - baseline"""
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(2)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MLP(nn.Module):
    """MLP with SwiGLU activation - baseline PyTorch implementation"""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ============================================================================
# Triton Implementations
# ============================================================================

if TRITON_AVAILABLE:
    @triton.jit
    def _rms_norm_fwd_kernel(
        X_ptr, W_ptr, Y_ptr,
        stride_x_row, stride_y_row,
        N, eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel for RMS normalization forward pass"""
        row_idx = tl.program_id(0)
        X_row_ptr = X_ptr + row_idx * stride_x_row
        Y_row_ptr = Y_ptr + row_idx * stride_y_row

        # Compute variance
        _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for col_start in range(0, N, BLOCK_SIZE):
            col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
            mask = col_offsets < N
            x = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            _var += x * x

        var = tl.sum(_var) / N
        rstd = tl.rsqrt(var + eps)

        # Normalize and scale
        for col_start in range(0, N, BLOCK_SIZE):
            col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
            mask = col_offsets < N
            x = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            y = x * rstd * w
            tl.store(Y_row_ptr + col_offsets, y, mask=mask)


    class TritonRMSNorm(nn.Module):
        """RMS Normalization using Triton kernel"""
        def __init__(self, hidden_size: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps
            self.hidden_size = hidden_size

        def forward(self, x: Tensor) -> Tensor:
            if not x.is_cuda or x.shape[-1] > 8192:
                # Fallback to baseline for CPU or very large hidden sizes
                dtype = x.dtype
                x_float = x.float()
                variance = x_float.pow(2).mean(-1, keepdim=True)
                x_norm = x_float * torch.rsqrt(variance + self.eps)
                return self.weight * x_norm.to(dtype)

            # Use Triton kernel
            original_shape = x.shape
            x = x.view(-1, self.hidden_size)
            y = torch.empty_like(x)

            M, N = x.shape
            BLOCK_SIZE = triton.next_power_of_2(N)
            if BLOCK_SIZE > 8192:
                BLOCK_SIZE = 8192

            _rms_norm_fwd_kernel[(M,)](
                x, self.weight, y,
                x.stride(0), y.stride(0),
                N, self.eps,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            return y.view(original_shape)


    @triton.jit
    def _swiglu_fwd_kernel(
        X_ptr, Gate_ptr, Up_ptr, Y_ptr,
        stride_x, stride_g, stride_u, stride_y,
        N,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel for SwiGLU forward pass"""
        row_idx = tl.program_id(0)
        X_row_ptr = X_ptr + row_idx * stride_x
        Gate_row_ptr = Gate_ptr + row_idx * stride_g
        Up_row_ptr = Up_ptr + row_idx * stride_u
        Y_row_ptr = Y_ptr + row_idx * stride_y

        for col_start in range(0, N, BLOCK_SIZE):
            col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
            mask = col_offsets < N

            gate = tl.load(Gate_row_ptr + col_offsets, mask=mask, other=0.0)
            up = tl.load(Up_row_ptr + col_offsets, mask=mask, other=0.0)

            # SiLU activation: x * sigmoid(x)
            gate_silu = gate * tl.sigmoid(gate)
            y = gate_silu * up

            tl.store(Y_row_ptr + col_offsets, y, mask=mask)


    class TritonSwiGLUMLP(nn.Module):
        """MLP with SwiGLU activation using Triton kernel"""
        def __init__(self, hidden_size: int, intermediate_size: int):
            super().__init__()
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size

            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        def forward(self, x: Tensor) -> Tensor:
            gate = self.gate_proj(x)
            up = self.up_proj(x)

            if not x.is_cuda or self.intermediate_size > 65536:
                # Fallback to baseline
                return self.down_proj(F.silu(gate) * up)

            # Use Triton kernel for fused SwiGLU
            original_shape = gate.shape
            gate_flat = gate.view(-1, self.intermediate_size)
            up_flat = up.view(-1, self.intermediate_size)
            y = torch.empty_like(gate_flat)

            M, N = gate_flat.shape
            BLOCK_SIZE = min(triton.next_power_of_2(N), 8192)

            _swiglu_fwd_kernel[(M,)](
                gate_flat, gate_flat, up_flat, y,
                gate_flat.stride(0), gate_flat.stride(0), up_flat.stride(0), y.stride(0),
                N,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            return self.down_proj(y.view(original_shape))

else:
    # Fallback if Triton not available
    TritonRMSNorm = RMSNorm
    TritonSwiGLUMLP = MLP


# ============================================================================
# Liger Wrappers
# ============================================================================

class LigerSwiGLUMLPWrapper(nn.Module):
    """Wrapper for LigerSwiGLUMLP to match our interface"""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        if LIGER_AVAILABLE:
            class MLPConfig:
                def __init__(self, hidden_size, intermediate_size):
                    self.hidden_size = hidden_size
                    self.intermediate_size = intermediate_size
                    self.hidden_act = "silu"

            self.mlp = LigerSwiGLUMLP(MLPConfig(hidden_size, intermediate_size))
        else:
            self.mlp = MLP(hidden_size, intermediate_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


# ============================================================================
# Factory Functions
# ============================================================================

KernelType = Literal["baseline", "triton", "liger"]


def create_rms_norm(hidden_size: int, eps: float = 1e-6, kernel_type: KernelType = "liger") -> nn.Module:
    """
    Factory function to create RMSNorm with specified kernel type.

    Args:
        hidden_size: Hidden dimension
        eps: Epsilon for numerical stability
        kernel_type: "liger" (preferred), "triton", or "baseline"

    Returns:
        RMSNorm module with best available implementation
    """
    if kernel_type == "liger" and LIGER_AVAILABLE:
        return LigerRMSNorm(hidden_size, eps=eps)
    elif kernel_type == "triton" and TRITON_AVAILABLE:
        return TritonRMSNorm(hidden_size, eps=eps)
    else:
        return RMSNorm(hidden_size, eps=eps)


def create_mlp(hidden_size: int, intermediate_size: int, kernel_type: KernelType = "liger") -> nn.Module:
    """
    Factory function to create MLP with SwiGLU activation.

    Args:
        hidden_size: Input/output dimension
        intermediate_size: Hidden dimension of MLP
        kernel_type: "liger" (preferred), "triton", or "baseline"

    Returns:
        MLP module with best available implementation
    """
    if kernel_type == "liger" and LIGER_AVAILABLE:
        return LigerSwiGLUMLPWrapper(hidden_size, intermediate_size)
    elif kernel_type == "triton" and TRITON_AVAILABLE:
        return TritonSwiGLUMLP(hidden_size, intermediate_size)
    else:
        return MLP(hidden_size, intermediate_size)


def get_rotary_pos_emb_fn(kernel_type: KernelType = "liger"):
    """
    Get the appropriate rotary position embedding function.

    Args:
        kernel_type: "liger" (preferred) or "baseline"

    Returns:
        Rotary embedding function
    """
    if kernel_type == "liger" and LIGER_AVAILABLE:
        return liger_rotary_pos_emb
    else:
        return apply_rotary_pos_emb


def create_cross_entropy_loss(lm_head: nn.Linear, kernel_type: KernelType = "liger"):
    """
    Create cross-entropy loss function, optionally fused with linear projection.

    Args:
        lm_head: The language model head linear layer (for fused implementation)
        kernel_type: "liger" (preferred) or "baseline"

    Returns:
        Loss module or function
    """
    if kernel_type == "liger" and LIGER_AVAILABLE:
        return LigerFusedLinearCrossEntropyLoss()
    else:
        return None  # Use F.cross_entropy directly


def compute_cross_entropy_loss(
    hidden_states: Tensor,
    lm_head: nn.Linear,
    labels: Tensor,
    loss_fn=None,
) -> Tuple[Optional[Tensor], Tensor]:
    """
    Compute cross-entropy loss with optional fusion.

    Args:
        hidden_states: Hidden states before lm_head [batch, seq, hidden]
        lm_head: Language model head
        labels: Target labels [batch, seq]
        loss_fn: Optional LigerFusedLinearCrossEntropyLoss

    Returns:
        (logits, loss) tuple - logits may be None when using fused loss
    """
    if loss_fn is not None and LIGER_AVAILABLE:
        # Fused implementation: linear + cross entropy in one kernel
        # This avoids materializing the full logits tensor
        shift_hidden = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        shift_hidden = shift_hidden.view(-1, shift_hidden.size(-1))
        shift_labels = shift_labels.view(-1)

        # LigerFusedLinearCrossEntropyLoss expects (lin_weight, _input, target)
        loss = loss_fn(lm_head.weight, shift_hidden, shift_labels)

        # Return None for logits - fused kernel doesn't produce them
        # This is a performance optimization - caller should not need logits during training
        return None, loss
    else:
        # Standard implementation
        logits = lm_head(hidden_states)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return logits, loss
