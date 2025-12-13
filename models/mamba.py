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
import triton
import triton.language as tl

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


@triton.jit
def _complex_mul(r1, i1, r2, i2):
    """(a+bi)(c+di) = (ac-bd) + (ad+bc)i"""
    real = r1 * r2 - i1 * i2
    imag = r1 * i2 + i1 * r2
    return real, imag


@triton.jit
def _complex_div(r1, i1, r2, i2):
    """(a+bi)/(c+di)"""
    denom = r2 * r2 + i2 * i2
    real = (r1 * r2 + i1 * i2) / denom
    imag = (i1 * r2 - r1 * i2) / denom
    return real, imag


@triton.jit
def _mamba3_scan_kernel(
    x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr, y_ptr,
    batch: tl.constexpr, seq: tl.constexpr, heads: tl.constexpr,
    headdim: tl.constexpr, state: tl.constexpr,
    stride_xb: tl.constexpr, stride_xl: tl.constexpr, stride_xd: tl.constexpr,
    stride_dtb: tl.constexpr, stride_dtl: tl.constexpr, stride_dth: tl.constexpr,
    stride_Bh: tl.constexpr, stride_Bl: tl.constexpr, stride_Bs: tl.constexpr,
    stride_Ch: tl.constexpr, stride_Cl: tl.constexpr, stride_Cs: tl.constexpr,
    BLOCK_STATE: tl.constexpr,
):
    """
    Fused Mamba3 SSM scan kernel.

    Args:
        x_ptr: Input (B, L, H*D)
        dt_ptr: Delta (B, L, H)
        A_ptr: State matrix (H, N, 2) complex
        B_ptr: Input matrix (B, L, H, N, 2) complex
        C_ptr: Output matrix (B, L, H, N, 2) complex
        D_ptr: Skip connection (H,)
        y_ptr: Output (B, L, H*D)
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_state = tl.arange(0, BLOCK_STATE)
    mask_state = offs_state < state

    A_offset = pid_h * state * 2
    a_real = tl.load(A_ptr + A_offset + offs_state * 2, mask=mask_state, other=0.0)
    a_imag = tl.load(A_ptr + A_offset + offs_state * 2 + 1, mask=mask_state, other=0.0)

    d_val = tl.load(D_ptr + pid_h)

    h_real = tl.zeros([BLOCK_STATE], dtype=tl.float32)
    h_imag = tl.zeros([BLOCK_STATE], dtype=tl.float32)

    for t in range(seq):
        offs_dim = tl.arange(0, headdim)
        x_offs = pid_b * stride_xb + t * stride_xl + pid_h * headdim + offs_dim
        x_t = tl.load(x_ptr + x_offs, mask=offs_dim < headdim, other=0.0)

        dt_offs = pid_b * stride_dtb + t * stride_dtl + pid_h * stride_dth
        dt_val = tl.load(dt_ptr + dt_offs)

        B_base = pid_b * stride_Bh + t * stride_Bl + pid_h * stride_Bs
        b_real = tl.load(B_ptr + B_base + offs_state * 2, mask=mask_state, other=0.0)
        b_imag = tl.load(B_ptr + B_base + offs_state * 2 + 1, mask=mask_state, other=0.0)

        C_base = pid_b * stride_Ch + t * stride_Cl + pid_h * stride_Cs
        c_real = tl.load(C_ptr + C_base + offs_state * 2, mask=mask_state, other=0.0)
        c_imag = tl.load(C_ptr + C_base + offs_state * 2 + 1, mask=mask_state, other=0.0)

        z_real = a_real * dt_val * 0.5
        z_imag = a_imag * dt_val * 0.5

        num_r = 1.0 + z_real
        num_i = z_imag
        den_r = 1.0 - z_real
        den_i = -z_imag

        abar_real, abar_imag = _complex_div(num_r, num_i, den_r, den_i)

        scale_r, scale_i = _complex_div(dt_val, 0.0, den_r, den_i)
        bbar_real, bbar_imag = _complex_mul(b_real, b_imag, scale_r, scale_i)

        t1_r, t1_i = _complex_mul(abar_real, abar_imag, h_real, h_imag)

        x_sum = tl.sum(x_t)
        t2_r = bbar_real * x_sum
        t2_i = bbar_imag * x_sum

        h_real = t1_r + t2_r
        h_imag = t1_i + t2_i

        y_r, _ = _complex_mul(c_real, c_imag, h_real, h_imag)
        y_reduced = tl.sum(y_r, axis=0)

        out_val = x_t * (y_reduced + d_val)

        y_offs = pid_b * stride_xb + t * stride_xl + pid_h * headdim + offs_dim
        tl.store(y_ptr + y_offs, out_val, mask=offs_dim < headdim)


def _mamba3_fused_scan(
    x: Tensor,
    dt: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
) -> Tensor:
    """
    Args:
        x: (B, L, H*D)
        dt: (B, L, H)
        A: (H, N, 2)
        B: (B, L, H, N, 2)
        C: (B, L, H, N, 2)
        D: (H,)
    Returns:
        y: (B, L, H*D)
    """
    B_batch, L, D_inner = x.shape
    nheads = dt.shape[2]
    d_state = A.shape[1]
    headdim = D_inner // nheads

    y = torch.empty_like(x)

    BLOCK_STATE = triton.next_power_of_2(d_state)

    grid = (B_batch, nheads)

    _mamba3_scan_kernel[grid](
        x, dt, A, B, C, D, y,
        B_batch, L, nheads, headdim, d_state,
        x.stride(0), x.stride(1), x.stride(2),
        dt.stride(0), dt.stride(1), dt.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        BLOCK_STATE=BLOCK_STATE,
    )
    return y


@dataclass
class Mamba3Config:
    """Configuration for Mamba3 model"""
    d_model: int = 1024
    n_layers: int = 24
    d_state: int = 128
    expand: int = 2
    headdim: int = 64
    vocab_size: int = 151936
    use_conv: bool = False
    use_complex: bool = True
    dt_min: float = 0.001
    dt_max: float = 0.1
    gradient_checkpointing: bool = False
    rms_norm: bool = True
    residual_in_fp32: bool = True


class Mamba3Block(nn.Module):
    """
    Mamba3 SSM block with fused Triton kernels.

    Args:
        d_model: Model dimension
        d_state: SSM state dimension
        expand: Expansion factor
        headdim: Head dimension
        use_conv: Enable conv1d
        use_complex: Use complex SSM parameters
        dt_min: Min delta value
        dt_max: Max delta value
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        use_conv: bool = False,
        use_complex: bool = True,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.headdim = headdim
        self.nheads = self.d_inner // headdim
        self.use_conv = use_conv
        self.use_complex = use_complex

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        if self.use_conv:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=True,
                kernel_size=4,
                groups=self.d_inner,
                padding=3,
            )

        param_factor = 2 if use_complex else 1

        self.x_proj = nn.Linear(
            self.d_inner,
            self.nheads * (1 + 2 * self.d_state * param_factor),
            bias=False,
        )

        dt = torch.exp(
            torch.rand(self.nheads) * (
                torch.log(torch.tensor(dt_max)) - torch.log(torch.tensor(dt_min))
            ) + torch.log(torch.tensor(dt_min))
        )
        self.dt_projs_bias = nn.Parameter(torch.log(dt))
        self.dt_projs_weight = nn.Parameter(torch.randn(self.nheads, self.d_inner))

        if use_complex:
            A_real = -torch.rand(self.nheads, self.d_state)
            A_imag = torch.randn(self.nheads, self.d_state) * 0.1
            self.A = nn.Parameter(torch.stack([A_real, A_imag], dim=-1))
        else:
            self.A = nn.Parameter(-torch.rand(self.nheads, self.d_state))

        self.D = nn.Parameter(torch.ones(self.nheads))

        self.norm_B = create_rms_norm(self.d_state * param_factor, eps=1e-6)
        self.norm_C = create_rms_norm(self.d_state * param_factor, eps=1e-6)

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(
        self,
        u: Tensor,
        inference_params=None,
    ) -> Tensor:
        """
        Args:
            u: (B, L, D)
        Returns:
            y: (B, L, D)
        """
        batch, seq, dim = u.shape

        xz = self.in_proj(u)
        x, z = xz.chunk(2, dim=-1)

        if self.use_conv:
            x = self.conv1d(x.transpose(1, 2))[:, :, :seq].transpose(1, 2)
        x = F.silu(x)

        dt_raw = F.linear(x, self.dt_projs_weight, self.dt_projs_bias)
        dt = F.softplus(dt_raw)

        BC_proj = self.x_proj(x)
        BC_proj = BC_proj.view(batch, seq, self.nheads, -1)

        param_factor = 2 if self.use_complex else 1
        BC_size = 2 * self.d_state * param_factor

        BC = BC_proj[..., :BC_size]
        BC = BC.view(batch, seq, self.nheads, 2, self.d_state, param_factor)

        B_raw = BC[..., 0, :, :]
        C_raw = BC[..., 1, :, :]

        B_norm = self.norm_B(B_raw.reshape(batch * seq * self.nheads, -1))
        C_norm = self.norm_C(C_raw.reshape(batch * seq * self.nheads, -1))

        B = B_norm.view(batch, seq, self.nheads, self.d_state, param_factor)
        C = C_norm.view(batch, seq, self.nheads, self.d_state, param_factor)

        if self.use_complex:
            B = torch.view_as_complex(B.contiguous())
            C = torch.view_as_complex(C.contiguous())
            B = torch.view_as_real(B)
            C = torch.view_as_real(C)

        y = _mamba3_fused_scan(x, dt, self.A, B, C, self.D)

        y = y * F.silu(z)
        return self.out_proj(y)


class Mamba3Model(nn.Module):
    """
    Full Mamba3 model for causal language modeling.

    Args:
        config: Mamba3Config
    """
    def __init__(self, config: Mamba3Config):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = config.gradient_checkpointing

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        self.layers = nn.ModuleList([
            self._create_layer(layer_idx)
            for layer_idx in range(config.n_layers)
        ])

        self.norm = create_rms_norm(config.d_model, eps=1e-6)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.loss_fn = create_cross_entropy_loss(self.lm_head)

        self._init_weights()

    def _create_layer(self, layer_idx: int) -> nn.Module:
        """Creates single layer with pre-norm, Mamba3, post-norm, MLP"""
        class Layer(nn.Module):
            def __init__(self, config: Mamba3Config):
                super().__init__()
                self.pre_norm = create_rms_norm(config.d_model, eps=1e-6)
                self.mixer = Mamba3Block(
                    d_model=config.d_model,
                    d_state=config.d_state,
                    expand=config.expand,
                    headdim=config.headdim,
                    use_conv=config.use_conv,
                    use_complex=config.use_complex,
                    dt_min=config.dt_min,
                    dt_max=config.dt_max,
                )
                self.post_norm = create_rms_norm(config.d_model, eps=1e-6)
                intermediate_size = int(config.d_model * 8 / 3)
                intermediate_size = ((intermediate_size + 63) // 64) * 64
                self.mlp = create_mlp(config.d_model, intermediate_size)

            def forward(self, x: Tensor, inference_params=None) -> Tensor:
                residual = x
                x = self.pre_norm(x)
                x = self.mixer(x, inference_params)
                x = residual + x

                residual = x
                x = self.post_norm(x)
                x = self.mlp(x)
                x = residual + x
                return x

        return Layer(self.config)

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
        """
        Args:
            input_ids: (B, L)
            attention_mask: Optional (B, L)
            labels: Optional (B, L)
        Returns:
            logits: (B, L, V)
            loss: Optional scalar
            None: Compatibility placeholder
        """
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    inference_params,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(hidden_states, inference_params)

        hidden_states = self.norm(hidden_states)

        if labels is not None:
            logits, loss = compute_cross_entropy_loss(
                hidden_states, self.lm_head, labels, self.loss_fn
            )
        else:
            logits = self.lm_head(hidden_states)
            loss = None

        return logits, loss, None


def create_mamba3(config: Mamba3Config) -> Mamba3Model:
    """Factory function to create Mamba3 model from config"""
    return Mamba3Model(config)
