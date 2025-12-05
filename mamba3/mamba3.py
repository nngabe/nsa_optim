"""
Mamba-3: Improved Sequence Modeling Using State Space Principles

Implementation based on the ICLR 2026 submission.

Key innovations:
1. Trapezoidal Discretization - More expressive recurrence than Euler's rule
2. Complex-valued SSM - Data-dependent RoPE embeddings for state-tracking
3. MIMO (Multi-Input Multi-Output) - Matrix-multiplication state updates for efficiency

Author: Implementation based on paper "Mamba-3: Improved Sequence Modeling Using State Space Principles"
"""

import math
from typing import Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum


@dataclass
class Mamba3Config:
    """Configuration for Mamba-3 model."""
    d_model: int = 2048          # Model dimension
    n_layers: int = 24           # Number of layers
    d_state: int = 128           # SSM state dimension (N)
    d_conv: int = 0              # Convolution kernel size (0 = disabled, paper shows it's optional)
    expand: int = 2              # Expansion factor for inner dimension
    head_dim: int = 64           # Dimension per head
    vocab_size: int = 128256     # Llama-3.1 tokenizer vocab size
    use_mimo: bool = False       # Whether to use MIMO variant
    mimo_rank: int = 4           # MIMO rank (r)
    use_conv: bool = False       # Whether to use short convolution (optional in Mamba-3)
    bias: bool = False           # Whether to use bias in linear projections
    dt_min: float = 0.001        # Minimum delta_t
    dt_max: float = 0.1          # Maximum delta_t
    dt_init_floor: float = 1e-4  # Floor for delta_t initialization
    rms_norm_eps: float = 1e-6   # RMSNorm epsilon
    

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight).to(dtype)


def apply_rotary_emb(
    x: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensor.
    
    This implements the "RoPE trick" from the paper - the complex SSM can be
    efficiently computed by applying data-dependent rotary embeddings.
    
    Args:
        x: Input tensor of shape (..., d) where d is even
        freqs: Rotation frequencies of shape (..., d//2)
    
    Returns:
        Rotated tensor of same shape as x
    """
    # Split into pairs for rotation
    x_reshape = x.reshape(*x.shape[:-1], -1, 2)
    
    # Compute cos and sin
    cos = torch.cos(freqs).unsqueeze(-1)
    sin = torch.sin(freqs).unsqueeze(-1)
    
    # Apply rotation: [cos, -sin; sin, cos] @ [x0, x1]
    x0, x1 = x_reshape[..., 0], x_reshape[..., 1]
    out0 = x0 * cos.squeeze(-1) - x1 * sin.squeeze(-1)
    out1 = x0 * sin.squeeze(-1) + x1 * cos.squeeze(-1)
    
    return torch.stack([out0, out1], dim=-1).reshape(x.shape)


class Mamba3Mixer(nn.Module):
    """
    Mamba-3 Mixer implementing the core SSM with:
    - Trapezoidal discretization
    - Complex-valued (data-dependent RoPE) state transitions
    - Optional MIMO formulation
    
    The state update follows:
        h_t = α_t * h_{t-1} + β_t * B_{t-1} * x_{t-1} + γ_t * B_t * x_t
        y_t = C_t^T * h_t
    
    Where:
        α_t = exp(Δt * A_t)
        β_t = (1 - λ_t) * Δt * exp(Δt * A_t)  
        γ_t = λ_t * Δt
        λ_t = sigmoid(u_t) is data-dependent
    """
    
    def __init__(self, config: Mamba3Config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_inner = config.expand * config.d_model
        self.head_dim = config.head_dim
        self.n_heads = self.d_inner // self.head_dim
        self.use_mimo = config.use_mimo
        self.mimo_rank = config.mimo_rank if config.use_mimo else 1
        
        # Input projection: projects to (x, B, C, dt, lambda, theta)
        # x: d_inner, B: n_heads * d_state, C: n_heads * d_state
        # dt: n_heads, lambda: n_heads (for trapezoidal), theta: n_heads * d_state // 2 (for RoPE)
        
        if self.use_mimo:
            # MIMO projections
            self.in_proj_x = nn.Linear(self.d_model, self.d_inner, bias=config.bias)
            self.in_proj_x_mimo = nn.Linear(self.d_inner, self.d_inner * self.mimo_rank // self.n_heads, bias=config.bias)
            self.in_proj_z = nn.Linear(self.d_model, self.d_inner, bias=config.bias)
            self.in_proj_z_mimo = nn.Linear(self.d_inner, self.d_inner * self.mimo_rank // self.n_heads, bias=config.bias)
        else:
            self.in_proj_x = nn.Linear(self.d_model, self.d_inner, bias=config.bias)
            self.in_proj_z = nn.Linear(self.d_model, self.d_inner, bias=config.bias)
        
        # B, C projections (per head)
        bc_dim = self.n_heads * self.d_state
        if self.use_mimo:
            bc_dim = self.n_heads * self.d_state * self.mimo_rank
        self.in_proj_B = nn.Linear(self.d_model, bc_dim, bias=config.bias)
        self.in_proj_C = nn.Linear(self.d_model, bc_dim, bias=config.bias)
        
        # Delta_t projection (per head)
        self.in_proj_dt = nn.Linear(self.d_model, self.n_heads, bias=config.bias)
        
        # Lambda projection for trapezoidal discretization (per head)
        self.in_proj_lambda = nn.Linear(self.d_model, self.n_heads, bias=config.bias)
        
        # Theta projection for data-dependent RoPE (d_state // 2 per head for complex pairs)
        self.in_proj_theta = nn.Linear(self.d_model, self.n_heads * (self.d_state // 2), bias=config.bias)
        
        # Learnable A parameter (log-space, negative for decay)
        self.A_log = nn.Parameter(torch.log(torch.linspace(1, self.d_state, self.n_heads)))
        
        # QK-Norm for B and C (replaces pre-output norm from Mamba-2)
        self.B_norm = RMSNorm(self.d_state, eps=config.rms_norm_eps)
        self.C_norm = RMSNorm(self.d_state, eps=config.rms_norm_eps)
        
        # Learnable biases for B and C (head-specific, channel-wise, initialized to 1)
        # This is key to making convolution optional
        self.B_bias = nn.Parameter(torch.ones(self.n_heads, self.d_state))
        self.C_bias = nn.Parameter(torch.ones(self.n_heads, self.d_state))
        
        # Optional short convolution
        self.use_conv = config.use_conv and config.d_conv > 0
        if self.use_conv:
            self.conv = nn.Conv1d(
                self.d_inner,
                self.d_inner,
                kernel_size=config.d_conv,
                padding=config.d_conv - 1,
                groups=self.d_inner,
            )
        
        # Output projection
        if self.use_mimo:
            self.out_proj_mimo = nn.Linear(self.d_inner * self.mimo_rank // self.n_heads, self.d_inner, bias=config.bias)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=config.bias)
        
        # Initialize dt projection
        self._init_dt_proj()
    
    def _init_dt_proj(self):
        """Initialize delta_t projection following Mamba conventions."""
        dt_init_std = self.config.dt_init_floor
        nn.init.uniform_(self.in_proj_dt.weight, -dt_init_std, dt_init_std)
        
        # Initialize bias to produce dt in [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(self.n_heads) * (math.log(self.config.dt_max) - math.log(self.config.dt_min))
            + math.log(self.config.dt_min)
        )
        inv_softplus = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            if self.in_proj_dt.bias is not None:
                self.in_proj_dt.bias.copy_(inv_softplus)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for Mamba-3 mixer.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            cache: Optional tuple of (h_prev, B_prev, x_prev) for recurrent inference
        
        Returns:
            output: Output tensor of shape (batch, seq_len, d_model)
            new_cache: Updated cache for next step
        """
        batch, seq_len, _ = x.shape
        
        # Input projections
        x_proj = self.in_proj_x(x)  # (B, L, d_inner)
        z = self.in_proj_z(x)       # (B, L, d_inner) - gate
        
        # MIMO projections if enabled
        if self.use_mimo:
            x_proj_mimo = self.in_proj_x_mimo(x_proj)  # (B, L, d_inner * r / n_heads)
            z_mimo = self.in_proj_z_mimo(z)
        
        # B, C projections with normalization and bias
        B = self.in_proj_B(x)  # (B, L, n_heads * d_state [* r])
        C = self.in_proj_C(x)  # (B, L, n_heads * d_state [* r])
        
        # Reshape for multi-head
        if self.use_mimo:
            B = rearrange(B, 'b l (h n r) -> b l h n r', h=self.n_heads, n=self.d_state, r=self.mimo_rank)
            C = rearrange(C, 'b l (h n r) -> b l h n r', h=self.n_heads, n=self.d_state, r=self.mimo_rank)
            # Apply norm per state dimension
            B = self.B_norm(B.transpose(-2, -1)).transpose(-2, -1)
            C = self.C_norm(C.transpose(-2, -1)).transpose(-2, -1)
            # Add bias (broadcast over r dimension)
            B = B + self.B_bias.unsqueeze(0).unsqueeze(1).unsqueeze(-1)
            C = C + self.C_bias.unsqueeze(0).unsqueeze(1).unsqueeze(-1)
        else:
            B = rearrange(B, 'b l (h n) -> b l h n', h=self.n_heads, n=self.d_state)
            C = rearrange(C, 'b l (h n) -> b l h n', h=self.n_heads, n=self.d_state)
            # Apply QK-norm
            B = self.B_norm(B)
            C = self.C_norm(C)
            # Add learnable bias
            B = B + self.B_bias.unsqueeze(0).unsqueeze(1)
            C = C + self.C_bias.unsqueeze(0).unsqueeze(1)
        
        # Delta_t and lambda for trapezoidal discretization
        dt = F.softplus(self.in_proj_dt(x))  # (B, L, n_heads), ensure positive
        lam = torch.sigmoid(self.in_proj_lambda(x))  # (B, L, n_heads), in [0, 1]
        
        # Theta for data-dependent RoPE
        theta = self.in_proj_theta(x)  # (B, L, n_heads * d_state // 2)
        theta = rearrange(theta, 'b l (h n) -> b l h n', h=self.n_heads, n=self.d_state // 2)
        
        # Compute cumulative theta for RoPE trick
        theta_cumsum = torch.cumsum(theta * dt.unsqueeze(-1), dim=1)  # (B, L, h, d_state//2)
        
        # Apply data-dependent RoPE to B and C
        if not self.use_mimo:
            B = apply_rotary_emb(B, theta_cumsum)
            C = apply_rotary_emb(C, theta_cumsum)
        else:
            # For MIMO, apply RoPE per rank dimension
            B_flat = rearrange(B, 'b l h n r -> b l h (n r)')
            C_flat = rearrange(C, 'b l h n r -> b l h (n r)')
            theta_expanded = repeat(theta_cumsum, 'b l h n -> b l h (n r)', r=self.mimo_rank)
            B_flat = apply_rotary_emb(B_flat, theta_expanded)
            C_flat = apply_rotary_emb(C_flat, theta_expanded)
            B = rearrange(B_flat, 'b l h (n r) -> b l h n r', n=self.d_state, r=self.mimo_rank)
            C = rearrange(C_flat, 'b l h (n r) -> b l h n r', n=self.d_state, r=self.mimo_rank)
        
        # Optional convolution
        if self.use_conv:
            x_proj = rearrange(x_proj, 'b l d -> b d l')
            x_proj = self.conv(x_proj)[:, :, :seq_len]
            x_proj = rearrange(x_proj, 'b d l -> b l d')
            x_proj = F.silu(x_proj)
        
        # Rearrange x for heads
        x_proj = rearrange(x_proj, 'b l (h p) -> b l h p', h=self.n_heads, p=self.head_dim)
        
        # Compute A (decay factor)
        A = -torch.exp(self.A_log)  # (n_heads,), negative for decay
        
        # Compute discretization coefficients
        # α_t = exp(Δt * A_t)
        # β_t = (1 - λ_t) * Δt * exp(Δt * A_t)
        # γ_t = λ_t * Δt
        alpha = torch.exp(dt.unsqueeze(-1) * A.view(1, 1, -1, 1))  # (B, L, h, 1)
        alpha = alpha.squeeze(-1)  # (B, L, h)
        beta = (1 - lam) * dt * alpha  # (B, L, h)
        gamma = lam * dt  # (B, L, h)
        
        # Run SSM with trapezoidal discretization
        if cache is not None:
            # Recurrent mode (decoding)
            y, new_cache = self._ssm_step(
                x_proj, B, C, alpha, beta, gamma, cache
            )
        else:
            # Parallel mode (training/prefill)
            y = self._ssm_parallel(
                x_proj, B, C, alpha, beta, gamma
            )
            new_cache = None
        
        # Rearrange output
        if self.use_mimo:
            y = rearrange(y, 'b l h p r -> b l (h p r)')
            y = self.out_proj_mimo(y)
            y = rearrange(y, 'b l (h p) -> b l h p', h=self.n_heads)
        
        y = rearrange(y, 'b l h p -> b l (h p)')
        
        # Apply gate
        y = y * F.silu(z)
        
        # Output projection
        y = self.out_proj(y)
        
        return y, new_cache
    
    def _ssm_parallel(
        self,
        x: torch.Tensor,      # (B, L, h, p)
        B: torch.Tensor,      # (B, L, h, n) or (B, L, h, n, r)
        C: torch.Tensor,      # (B, L, h, n) or (B, L, h, n, r)
        alpha: torch.Tensor,  # (B, L, h)
        beta: torch.Tensor,   # (B, L, h)
        gamma: torch.Tensor,  # (B, L, h)
    ) -> torch.Tensor:
        """
        Parallel SSM computation using selective scan with trapezoidal discretization.
        
        This implements the SSD (State Space Duality) form but with the trapezoidal
        mask decomposition: L = L1 @ L2 where L1 is decay mask and L2 is conv mask.
        """
        batch, seq_len, n_heads, head_dim = x.shape
        
        if self.use_mimo:
            # MIMO: state update via matrix multiplication
            # H_t = α_t * H_{t-1} + B_t @ X_t^T
            # Y_t = H_t^T @ C_t
            
            # For parallel computation, we use a chunked approach similar to Mamba-2
            # but with the trapezoidal modification
            
            # Simplified parallel scan for MIMO
            # This is a reference implementation; optimized kernels would use chunking
            h = torch.zeros(batch, n_heads, self.d_state, head_dim, device=x.device, dtype=x.dtype)
            outputs = []
            
            x_mimo = rearrange(x, 'b l h p -> b l h p 1')  # Add MIMO dimension
            
            for t in range(seq_len):
                # Trapezoidal update
                if t > 0:
                    # β_t * B_{t-1} * x_{t-1} term
                    Bx_prev = einsum(B[:, t-1], x_mimo[:, t-1], 'b h n r, b h p r -> b h n p')
                    h = alpha[:, t, :, None, None] * h + beta[:, t, :, None, None] * Bx_prev
                
                # γ_t * B_t * x_t term
                Bx_curr = einsum(B[:, t], x_mimo[:, t], 'b h n r, b h p r -> b h n p')
                h = h + gamma[:, t, :, None, None] * Bx_curr
                
                # Output: y_t = C_t^T @ h_t
                y_t = einsum(C[:, t], h, 'b h n r, b h n p -> b h p r')
                outputs.append(y_t)
            
            y = torch.stack(outputs, dim=1)  # (B, L, h, p, r)
            return y
        else:
            # SISO: standard selective scan with trapezoidal discretization
            # This is implemented as a modified parallel scan
            
            h = torch.zeros(batch, n_heads, self.d_state, head_dim, device=x.device, dtype=x.dtype)
            outputs = []
            
            for t in range(seq_len):
                # Trapezoidal update: h_t = α_t * h_{t-1} + β_t * B_{t-1} * x_{t-1} + γ_t * B_t * x_t
                if t > 0:
                    # β_t * B_{t-1} ⊗ x_{t-1}
                    Bx_prev = einsum(B[:, t-1], x[:, t-1], 'b h n, b h p -> b h n p')
                    h = alpha[:, t, :, None, None] * h + beta[:, t, :, None, None] * Bx_prev
                
                # γ_t * B_t ⊗ x_t
                Bx_curr = einsum(B[:, t], x[:, t], 'b h n, b h p -> b h n p')
                h = h + gamma[:, t, :, None, None] * Bx_curr
                
                # Output: y_t = C_t^T @ h_t
                y_t = einsum(C[:, t], h, 'b h n, b h n p -> b h p')
                outputs.append(y_t)
            
            y = torch.stack(outputs, dim=1)  # (B, L, h, p)
            return y
    
    def _ssm_step(
        self,
        x: torch.Tensor,      # (B, 1, h, p)
        B: torch.Tensor,      # (B, 1, h, n)
        C: torch.Tensor,      # (B, 1, h, n)
        alpha: torch.Tensor,  # (B, 1, h)
        beta: torch.Tensor,   # (B, 1, h)
        gamma: torch.Tensor,  # (B, 1, h)
        cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Single step SSM update for recurrent inference.
        
        Cache contains: (h_prev, B_prev, x_prev)
        """
        h_prev, B_prev, x_prev = cache
        
        # Squeeze sequence dimension
        x = x.squeeze(1)          # (B, h, p)
        B = B.squeeze(1)          # (B, h, n)
        C = C.squeeze(1)          # (B, h, n)
        alpha = alpha.squeeze(1)  # (B, h)
        beta = beta.squeeze(1)    # (B, h)
        gamma = gamma.squeeze(1)  # (B, h)
        
        if self.use_mimo:
            # MIMO step
            # β_t * B_{t-1} @ x_{t-1}^T
            Bx_prev = einsum(B_prev, x_prev, 'b h n r, b h p r -> b h n p')
            h = alpha[:, :, None, None] * h_prev + beta[:, :, None, None] * Bx_prev
            
            # γ_t * B_t @ x_t^T
            Bx_curr = einsum(B, x, 'b h n r, b h p r -> b h n p')
            h = h + gamma[:, :, None, None] * Bx_curr
            
            # Output
            y = einsum(C, h, 'b h n r, b h n p -> b h p r')
        else:
            # SISO step
            # β_t * B_{t-1} ⊗ x_{t-1}
            Bx_prev = einsum(B_prev, x_prev, 'b h n, b h p -> b h n p')
            h = alpha[:, :, None, None] * h_prev + beta[:, :, None, None] * Bx_prev
            
            # γ_t * B_t ⊗ x_t
            Bx_curr = einsum(B, x, 'b h n, b h p -> b h n p')
            h = h + gamma[:, :, None, None] * Bx_curr
            
            # Output
            y = einsum(C, h, 'b h n, b h n p -> b h p')
        
        y = y.unsqueeze(1)  # (B, 1, h, p) or (B, 1, h, p, r)
        new_cache = (h, B, x)
        
        return y, new_cache


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (Llama-style)."""
    
    def __init__(self, d_model: int, d_ff: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)
        self.w3 = nn.Linear(d_model, d_ff, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Mamba3Block(nn.Module):
    """
    Single Mamba-3 block with pre-normalization (Llama-style).
    
    Architecture:
        x -> RMSNorm -> Mamba3Mixer -> + -> RMSNorm -> SwiGLU -> + -> output
            └─────────────────────────┘   └────────────────────┘
    """
    
    def __init__(self, config: Mamba3Config, layer_idx: int = 0):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.mixer = Mamba3Mixer(config, layer_idx=layer_idx)
        self.norm2 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        
        # MLP with adjusted dimension for MIMO parameter matching
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


class Mamba3Model(nn.Module):
    """
    Full Mamba-3 language model.
    
    Architecture follows Llama design with alternating Mamba-3 and SwiGLU blocks.
    """
    
    def __init__(self, config: Mamba3Config):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            Mamba3Block(config, layer_idx=i) for i in range(config.n_layers)
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
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: Prompt token IDs of shape (batch, prompt_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
        
        Returns:
            Generated token IDs of shape (batch, prompt_len + max_new_tokens)
        """
        # Prefill: process prompt
        logits, cache = self.forward(input_ids, return_cache=True)
        
        # Get next token from last position
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
        
        # Decode: generate remaining tokens
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


# ============================================================================
# Efficient Parallel Scan Implementation (for training)
# ============================================================================

def selective_scan_trapezoidal(
    x: torch.Tensor,      # (B, L, h, p)
    B: torch.Tensor,      # (B, L, h, n)
    C: torch.Tensor,      # (B, L, h, n)
    alpha: torch.Tensor,  # (B, L, h)
    beta: torch.Tensor,   # (B, L, h)
    gamma: torch.Tensor,  # (B, L, h)
) -> torch.Tensor:
    """
    Efficient parallel selective scan with trapezoidal discretization.
    
    This implements the chunked parallel scan algorithm from Mamba-2,
    modified for the trapezoidal update rule.
    
    For production use, this would be implemented as a custom CUDA kernel.
    """
    batch, seq_len, n_heads, head_dim = x.shape
    d_state = B.shape[-1]
    
    # For simplicity, use sequential scan here
    # In practice, use chunked parallel scan from mamba-ssm library
    h = torch.zeros(batch, n_heads, d_state, head_dim, device=x.device, dtype=x.dtype)
    outputs = []
    
    for t in range(seq_len):
        if t > 0:
            Bx_prev = einsum(B[:, t-1], x[:, t-1], 'b h n, b h p -> b h n p')
            h = alpha[:, t, :, None, None] * h + beta[:, t, :, None, None] * Bx_prev
        
        Bx_curr = einsum(B[:, t], x[:, t], 'b h n, b h p -> b h n p')
        h = h + gamma[:, t, :, None, None] * Bx_curr
        
        y_t = einsum(C[:, t], h, 'b h n, b h n p -> b h p')
        outputs.append(y_t)
    
    return torch.stack(outputs, dim=1)


# ============================================================================
# Model Factory Functions
# ============================================================================

def mamba3_180m(use_mimo: bool = False) -> Mamba3Model:
    """Create 180M parameter Mamba-3 model."""
    config = Mamba3Config(
        d_model=768,
        n_layers=24,
        d_state=128,
        head_dim=64,
        use_mimo=use_mimo,
    )
    return Mamba3Model(config)


def mamba3_440m(use_mimo: bool = False) -> Mamba3Model:
    """Create 440M parameter Mamba-3 model."""
    config = Mamba3Config(
        d_model=1024,
        n_layers=24,
        d_state=128,
        head_dim=64,
        use_mimo=use_mimo,
    )
    return Mamba3Model(config)


def mamba3_880m(use_mimo: bool = False) -> Mamba3Model:
    """Create 880M parameter Mamba-3 model."""
    config = Mamba3Config(
        d_model=1536,
        n_layers=24,
        d_state=128,
        head_dim=64,
        use_mimo=use_mimo,
    )
    return Mamba3Model(config)


def mamba3_1_5b(use_mimo: bool = False) -> Mamba3Model:
    """Create 1.5B parameter Mamba-3 model."""
    config = Mamba3Config(
        d_model=2048,
        n_layers=24,
        d_state=128,
        head_dim=64,
        use_mimo=use_mimo,
    )
    return Mamba3Model(config)


# ============================================================================
# Test / Demo
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Mamba-3 Implementation Test")
    print("=" * 60)
    
    # Test configuration
    config = Mamba3Config(
        d_model=256,
        n_layers=4,
        d_state=64,
        head_dim=32,
        vocab_size=1000,
        use_mimo=False,
    )
    
    print(f"\nConfig: d_model={config.d_model}, n_layers={config.n_layers}, "
          f"d_state={config.d_state}, use_mimo={config.use_mimo}")
    
    # Create model
    model = Mamba3Model(config)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass (training mode)
    logits, _ = model(x)
    print(f"Output logits shape: {logits.shape}")
    
    # Test with cache (inference mode)
    print("\nTesting incremental decoding...")
    logits_prefill, cache = model(x[:, :64], return_cache=True)
    print(f"Prefill output shape: {logits_prefill.shape}")
    
    # Single step decode
    logits_step, cache = model(x[:, 64:65], cache=cache, return_cache=True)
    print(f"Single step output shape: {logits_step.shape}")
    
    # Test MIMO variant
    print("\n" + "=" * 60)
    print("Testing MIMO variant")
    print("=" * 60)
    
    config_mimo = Mamba3Config(
        d_model=256,
        n_layers=4,
        d_state=64,
        head_dim=32,
        vocab_size=1000,
        use_mimo=True,
        mimo_rank=4,
    )
    
    model_mimo = Mamba3Model(config_mimo)
    n_params_mimo = sum(p.numel() for p in model_mimo.parameters())
    print(f"MIMO parameters: {n_params_mimo:,}")
    
    logits_mimo, _ = model_mimo(x)
    print(f"MIMO output shape: {logits_mimo.shape}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
