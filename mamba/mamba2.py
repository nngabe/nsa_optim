# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted for nsa_optim project with fallback implementations.

"""
Mamba-2: Simplified State Space Models with SSD (State Space Duality)

This implementation is based on the state-spaces/mamba repository.
It provides a self-contained version that works without the specialized
mamba_ssm kernels, while using them when available for better performance.

Key features:
- SSD (State Space Duality) formulation for efficient parallel training
- Chunked parallel scan for linear-time sequence processing
- Support for multi-head attention-like structure
"""

import math
from typing import Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# Try to import optimized kernels
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    CAUSAL_CONV1D_AVAILABLE = True
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None
    CAUSAL_CONV1D_AVAILABLE = False

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
    SSD_KERNELS_AVAILABLE = True
except ImportError:
    mamba_chunk_scan_combined = None
    mamba_split_conv1d_scan_combined = None
    SSD_KERNELS_AVAILABLE = False

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
    RMSNORM_GATED_AVAILABLE = True
except ImportError:
    RMSNormGated = None
    RMSNORM_GATED_AVAILABLE = False


@dataclass
class Mamba2Config:
    """Configuration for Mamba-2 model."""
    d_model: int = 2048          # Model dimension
    n_layers: int = 24           # Number of layers
    d_state: int = 128           # SSM state dimension
    d_conv: int = 4              # Convolution kernel size
    expand: int = 2              # Expansion factor for inner dimension
    headdim: int = 64            # Dimension per head
    ngroups: int = 1             # Number of groups for state
    vocab_size: int = 151936     # Qwen tokenizer vocab size
    bias: bool = False           # Whether to use bias in linear projections
    conv_bias: bool = True       # Whether to use bias in conv
    dt_min: float = 0.001        # Minimum delta_t
    dt_max: float = 0.1          # Maximum delta_t
    dt_init_floor: float = 1e-4  # Floor for delta_t initialization
    rmsnorm: bool = True         # Whether to use RMSNorm
    rms_norm_eps: float = 1e-5   # RMSNorm epsilon
    chunk_size: int = 256        # Chunk size for parallel scan
    use_triton: bool = True      # Whether to use optimized Triton kernels
    gradient_checkpointing: bool = False


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        result = (norm * self.weight).to(dtype)
        if z is not None:
            # Gated normalization
            result = result * F.silu(z.to(dtype))
        return result


class Mamba2Mixer(nn.Module):
    """
    Mamba-2 Mixer implementing the SSD (State Space Duality) formulation.

    Key innovations over Mamba-1:
    - Structured state space with head dimension (attention-like)
    - More efficient parallel scan via chunking
    - Simpler architecture without selective scan complexity
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
        conv_bias: bool = True,
        rmsnorm: bool = True,
        rms_norm_eps: float = 1e-5,
        chunk_size: int = 256,
        use_triton: bool = True,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.headdim = headdim
        self.ngroups = ngroups
        self.nheads = self.d_inner // headdim
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init_floor = dt_init_floor
        self.rmsnorm = rmsnorm
        self.chunk_size = chunk_size
        self.use_triton = use_triton and SSD_KERNELS_AVAILABLE
        self.layer_idx = layer_idx

        # Input projection: projects to [z, x, B, C, dt]
        # z: d_inner (gate), x: d_inner (input to SSM)
        # B, C: ngroups * d_state each (state matrices)
        # dt: nheads (timestep)
        d_in_proj = 2 * self.d_inner + 2 * ngroups * d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=bias)

        # Convolution for local context
        conv_dim = self.d_inner + 2 * ngroups * d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            bias=conv_bias,
        )

        self.act = nn.SiLU()

        # Initialize dt bias
        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # A parameter (log-space for numerical stability)
        A = torch.empty(self.nheads, dtype=torch.float32).uniform_(1, 16)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True

        # Normalization
        if rmsnorm:
            if RMSNORM_GATED_AVAILABLE:
                self.norm = RMSNormGated(self.d_inner, eps=rms_norm_eps, norm_before_gate=False,
                                         group_size=self.d_inner // ngroups)
            else:
                self.norm = RMSNorm(self.d_inner, eps=rms_norm_eps)
        else:
            self.norm = None

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for Mamba-2 mixer.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            cache: Optional tuple of (conv_state, ssm_state) for recurrent inference

        Returns:
            output: Output tensor of shape (batch, seq_len, d_model)
            new_cache: Updated cache for next step
        """
        batch, seq_len, _ = x.shape

        # Handle single-step inference with cache
        if cache is not None and seq_len == 1:
            return self._step(x, cache)

        # Input projection
        zxbcdt = self.in_proj(x)  # (B, L, d_in_proj)

        # Split projections
        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # Apply convolution
        xBC = xBC.transpose(1, 2)  # (B, D, L)
        xBC = self.conv1d(xBC)[:, :, :seq_len]  # Remove padding
        xBC = xBC.transpose(1, 2)  # (B, L, D)
        xBC = self.act(xBC)

        # Split x, B, C
        x_ssm, B, C = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1
        )

        # SSM computation
        A = -torch.exp(self.A_log.float())  # (nheads,)

        if self.use_triton and mamba_chunk_scan_combined is not None:
            # Use optimized kernel
            y = mamba_chunk_scan_combined(
                rearrange(x_ssm, 'b l (h p) -> b l h p', p=self.headdim),
                dt,
                A,
                rearrange(B, 'b l (g n) -> b l g n', g=self.ngroups),
                rearrange(C, 'b l (g n) -> b l g n', g=self.ngroups),
                chunk_size=self.chunk_size,
                D=self.D,
                z=rearrange(z, 'b l (h p) -> b l h p', p=self.headdim) if not self.rmsnorm else None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
            )
            y = rearrange(y, 'b l h p -> b l (h p)')
        else:
            # Reference implementation
            y = self._ssm_parallel(x_ssm, B, C, dt, A)

        # Apply normalization and gating
        if self.rmsnorm and self.norm is not None:
            y = self.norm(y, z)
        elif not self.rmsnorm:
            y = y * self.act(z)

        # Output projection
        out = self.out_proj(y)

        # Build cache for potential future inference
        new_cache = None
        if cache is not None:
            # Update conv state
            conv_state = xBC[:, -self.d_conv:, :].transpose(1, 2).contiguous()
            # SSM state would need to be computed from the scan
            # For now, return None for cache update in parallel mode
            new_cache = None

        return out, new_cache

    def _ssm_parallel(
        self,
        x: torch.Tensor,      # (B, L, d_inner)
        B: torch.Tensor,      # (B, L, ngroups * d_state)
        C: torch.Tensor,      # (B, L, ngroups * d_state)
        dt: torch.Tensor,     # (B, L, nheads)
        A: torch.Tensor,      # (nheads,)
    ) -> torch.Tensor:
        """Reference SSM implementation for when optimized kernels aren't available."""
        batch, seq_len, d_inner = x.shape

        # Reshape for multi-head computation
        x = rearrange(x, 'b l (h p) -> b l h p', h=self.nheads, p=self.headdim)
        B = rearrange(B, 'b l (g n) -> b l g n', g=self.ngroups, n=self.d_state)
        C = rearrange(C, 'b l (g n) -> b l g n', g=self.ngroups, n=self.d_state)

        # Apply softplus to dt
        dt = F.softplus(dt + self.dt_bias)  # (B, L, h)

        # Compute discretized A
        dA = torch.exp(dt.unsqueeze(-1) * A.view(1, 1, -1, 1))  # (B, L, h, 1)

        # Sequential scan (reference implementation)
        h = torch.zeros(batch, self.nheads, self.headdim, self.d_state,
                       device=x.device, dtype=x.dtype)
        outputs = []

        # Handle ngroups > 1 by expanding B and C
        if self.ngroups < self.nheads:
            heads_per_group = self.nheads // self.ngroups
            B = repeat(B, 'b l g n -> b l (g r) n', r=heads_per_group)
            C = repeat(C, 'b l g n -> b l (g r) n', r=heads_per_group)

        for t in range(seq_len):
            # State update: h = dA * h + dB * x
            dB = dt[:, t, :, None, None] * B[:, t, :, None, :]  # (B, h, 1, n)
            dBx = dB * x[:, t, :, :, None]  # (B, h, p, n)
            h = dA[:, t, :, :, None] * h + dBx

            # Output: y = C^T @ h
            y = torch.einsum('bhn,bhpn->bhp', C[:, t], h)

            # Add skip connection
            y = y + self.D.view(1, -1, 1) * x[:, t]
            outputs.append(y)

        y = torch.stack(outputs, dim=1)  # (B, L, h, p)
        y = rearrange(y, 'b l h p -> b l (h p)')

        return y

    def _step(
        self,
        x: torch.Tensor,
        cache: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Single step for recurrent inference."""
        conv_state, ssm_state = cache

        # Input projection
        zxbcdt = self.in_proj(x.squeeze(1))  # (B, d_in_proj)

        # Split projections
        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # Update conv state
        conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
        conv_state[:, :, -1] = xBC

        # Apply convolution
        xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, 'd 1 w -> d w'), dim=-1)
        if self.conv1d.bias is not None:
            xBC = xBC + self.conv1d.bias
        xBC = self.act(xBC)

        # Split x, B, C
        x_ssm, B, C = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1
        )

        # SSM step
        A = -torch.exp(self.A_log.float())
        dt = F.softplus(dt + self.dt_bias)  # (B, h)

        # Reshape
        x_ssm = rearrange(x_ssm, 'b (h p) -> b h p', h=self.nheads, p=self.headdim)
        B = rearrange(B, 'b (g n) -> b g n', g=self.ngroups, n=self.d_state)
        C = rearrange(C, 'b (g n) -> b g n', g=self.ngroups, n=self.d_state)

        # Expand B and C for ngroups
        if self.ngroups < self.nheads:
            heads_per_group = self.nheads // self.ngroups
            B = repeat(B, 'b g n -> b (g r) n', r=heads_per_group)
            C = repeat(C, 'b g n -> b (g r) n', r=heads_per_group)

        # Discretize
        dA = torch.exp(dt.unsqueeze(-1) * A.view(1, -1, 1))  # (B, h, 1)
        dB = dt[:, :, None, None] * B[:, :, None, :]  # (B, h, 1, n)
        dBx = dB * x_ssm[:, :, :, None]  # (B, h, p, n)

        # State update
        ssm_state = dA.unsqueeze(-1) * ssm_state + dBx

        # Output
        y = torch.einsum('bhn,bhpn->bhp', C, ssm_state)
        y = y + self.D.view(1, -1, 1) * x_ssm
        y = rearrange(y, 'b h p -> b (h p)')

        # Apply normalization and gating
        if self.rmsnorm and self.norm is not None:
            y = self.norm(y, z)
        else:
            y = y * self.act(z)

        # Output projection
        out = self.out_proj(y).unsqueeze(1)

        return out, (conv_state, ssm_state)

    def allocate_inference_cache(self, batch_size: int, dtype: torch.dtype = None, device: torch.device = None):
        """Allocate cache for inference."""
        if device is None:
            device = self.out_proj.weight.device
        if dtype is None:
            dtype = self.out_proj.weight.dtype

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        conv_state = torch.zeros(batch_size, conv_dim, self.d_conv, device=device, dtype=dtype)
        ssm_state = torch.zeros(batch_size, self.nheads, self.headdim, self.d_state,
                               device=device, dtype=dtype)

        return conv_state, ssm_state


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)
        self.w3 = nn.Linear(d_model, d_ff, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Mamba2Block(nn.Module):
    """
    Single Mamba-2 block with pre-normalization.

    Architecture:
        x -> RMSNorm -> Mamba2Mixer -> + -> RMSNorm -> SwiGLU -> + -> output
    """

    def __init__(self, config: Mamba2Config, layer_idx: int = 0):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.mixer = Mamba2Mixer(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            headdim=config.headdim,
            ngroups=config.ngroups,
            dt_min=config.dt_min,
            dt_max=config.dt_max,
            dt_init_floor=config.dt_init_floor,
            bias=config.bias,
            conv_bias=config.conv_bias,
            rmsnorm=config.rmsnorm,
            rms_norm_eps=config.rms_norm_eps,
            chunk_size=config.chunk_size,
            use_triton=config.use_triton,
            layer_idx=layer_idx,
        )
        self.norm2 = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # MLP
        d_ff = int(config.d_model * 8 / 3)
        d_ff = ((d_ff + 255) // 256) * 256  # Round to multiple of 256
        self.mlp = SwiGLU(config.d_model, d_ff, bias=config.bias)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        # Mamba mixer with residual
        h, new_cache = self.mixer(self.norm1(x), cache=cache)
        x = x + h

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x, new_cache


class Mamba2Model(nn.Module):
    """
    Full Mamba-2 language model.
    """

    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = config.gradient_checkpointing

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            Mamba2Block(config, layer_idx=i) for i in range(config.n_layers)
        ])
        self.norm_f = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: Optional[list] = None,
        return_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len)
            cache: Optional list of layer caches for recurrent inference
            return_cache: Whether to return updated cache

        Returns:
            logits: Output logits of shape (batch, seq_len, vocab_size)
            new_cache: Updated cache if return_cache=True
        """
        x = self.embedding(input_ids)

        new_cache = [] if return_cache else None

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                x, layer_new_cache = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    x,
                    layer_cache,
                    use_reentrant=False,
                )
            else:
                x, layer_new_cache = layer(x, cache=layer_cache)

            if return_cache:
                new_cache.append(layer_new_cache)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        return logits, new_cache

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively."""
        # Prefill
        logits, cache = self.forward(input_ids, return_cache=True)

        # Get next token
        next_token_logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')

        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')

        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated = [input_ids, next_token]

        # Decode
        for _ in range(max_new_tokens - 1):
            logits, cache = self.forward(next_token, cache=cache, return_cache=True)
            next_token_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated.append(next_token)

        return torch.cat(generated, dim=1)


def create_mamba2(config: Mamba2Config) -> Mamba2Model:
    """Factory function to create Mamba2 model from config."""
    return Mamba2Model(config)


# ============================================================================
# Test / Demo
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Mamba-2 Implementation Test")
    print("=" * 60)

    # Test configuration
    config = Mamba2Config(
        d_model=256,
        n_layers=4,
        d_state=64,
        headdim=32,
        vocab_size=1000,
    )

    print(f"\nConfig: d_model={config.d_model}, n_layers={config.n_layers}, "
          f"d_state={config.d_state}")

    # Create model
    model = Mamba2Model(config)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    # Test forward pass
    batch_size = 2
    seq_len = 128
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    print(f"\nInput shape: {x.shape}")

    # Forward pass
    logits, _ = model(x)
    print(f"Output logits shape: {logits.shape}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
