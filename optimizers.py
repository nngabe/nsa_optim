"""
Optimizer implementations for ablation study

Provides unified interface for:
- AdamW (baseline)
- AdamW8bit (8-bit quantized states)
- SOAP (from NVIDIA Emerging-Optimizers)
- Shampoo (distributed or single-GPU)
- SOAP with low-bit optimizer states
"""
import math
from typing import Optional, Dict, Any, List, Callable, Tuple, Iterable

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer

from config import OptimizerConfig, OptimizerType


def create_optimizer(
    model: nn.Module,
    config: OptimizerConfig,
    tensor_parallel_size: int = 1,
    orig_param_shapes: Optional[Dict[int, Tuple[str, Tuple[int, ...]]]] = None,
) -> Optimizer:
    """Factory function to create optimizer from config

    Args:
        model: Model to optimize
        config: Optimizer configuration
        tensor_parallel_size: Number of GPUs for distributed training
        orig_param_shapes: Dict mapping param data_ptr to (name, shape) for FSDP preconditioning
    """

    # Separate parameters by weight decay eligibility
    param_groups = get_param_groups(model, config.weight_decay)

    if config.optimizer_type == OptimizerType.ADAMW:
        return create_adamw(param_groups, config)

    elif config.optimizer_type == OptimizerType.ADAMW_4BIT:
        return create_adamw_lowbit(param_groups, config, bits=4)

    elif config.optimizer_type == OptimizerType.ADAMW_8BIT:
        return create_adamw_lowbit(param_groups, config, bits=8)

    elif config.optimizer_type == OptimizerType.SOAP:
        return create_soap(param_groups, config)

    elif config.optimizer_type == OptimizerType.SOAP_4BIT:
        return create_soap_lowbit(param_groups, config, bits=4, orig_param_shapes=orig_param_shapes)

    elif config.optimizer_type == OptimizerType.SOAP_8BIT:
        return create_soap_lowbit(param_groups, config, bits=8, orig_param_shapes=orig_param_shapes)

    elif config.optimizer_type == OptimizerType.SHAMPOO:
        return create_shampoo(param_groups, config, tensor_parallel_size)

    else:
        raise ValueError(f"Unknown optimizer type: {config.optimizer_type}")


def get_param_groups(model: nn.Module, weight_decay: float) -> List[Dict[str, Any]]:
    """Separate parameters into groups with/without weight decay"""
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Don't apply weight decay to bias, layer norm, and embedding
        if param.ndim == 1 or name.endswith(".bias") or "layernorm" in name.lower() or "embed" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def create_adamw(param_groups: List[Dict], config: OptimizerConfig) -> Optimizer:
    """Create AdamW optimizer"""
    return torch.optim.AdamW(
        param_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        fused=True,  # Use fused implementation for speed
    )


def create_adamw_lowbit(param_groups: List[Dict], config: OptimizerConfig, bits: int = 8) -> Optimizer:
    """
    Create low-bit AdamW optimizer (4-bit or 8-bit)
    Uses gemlite for 4-bit or torchao for 8-bit quantized optimizer states

    Args:
        param_groups: Parameter groups with weight decay settings
        config: Optimizer configuration
        bits: Quantization bits (4 or 8)
    """
    # Try gemlite first for 4-bit
    if bits == 4:
        try:
            from gemlite.optim import AdamW as GemLiteAdamW
            print(f"Using gemlite {bits}-bit AdamW")
            return GemLiteAdamW(
                [p for pg in param_groups for p in pg["params"]],
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.eps,
                weight_decay=config.weight_decay,
            )
        except ImportError:
            print("Warning: gemlite not available, cannot use 4-bit AdamW")
            print("Falling back to standard AdamW")
            return create_adamw(param_groups, config)

    # Use torchao for 8-bit
    if bits == 8:
        try:
            # Try new import path first (torchao >= 0.15)
            try:
                from torchao.optim import AdamW8bit
            except ImportError:
                # Fall back to old import path (torchao < 0.15)
                from torchao.prototype.low_bit_optim import AdamW8bit

            print("Using torchao 8-bit AdamW")
            return AdamW8bit(
                [p for pg in param_groups for p in pg["params"]],
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.eps,
                weight_decay=config.weight_decay,
            )
        except (ImportError, AttributeError, ModuleNotFoundError, RuntimeError) as e:
            print(f"Warning: torchao AdamW8bit not available ({e.__class__.__name__}: {e})")
            print("Falling back to standard AdamW")
            return create_adamw(param_groups, config)

    # Unknown bits value
    print(f"Warning: Unsupported bits={bits}, falling back to standard AdamW")
    return create_adamw(param_groups, config)


def create_soap(param_groups: List[Dict], config: OptimizerConfig) -> Optimizer:
    """
    Create SOAP optimizer from NVIDIA Emerging-Optimizers
    Falls back to reference implementation if not available
    """
    try:
        from emerging_optimizers.soap import SOAP
        
        return SOAP(
            [p for pg in param_groups for p in pg["params"]],
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            shampoo_beta=config.shampoo_beta,
            eps=config.eps,
            weight_decay=config.weight_decay,
            precondition_frequency=config.precondition_frequency,
        )
    except ImportError:
        print("Warning: NVIDIA Emerging-Optimizers not installed, using reference SOAP")
        return SOAPReference(
            param_groups,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            shampoo_beta=config.shampoo_beta,
            eps=config.eps,
            precondition_frequency=config.precondition_frequency,
            max_precond_dim=config.max_precond_dim,
        )


def create_shampoo(param_groups: List[Dict], config: OptimizerConfig, tensor_parallel_size: int) -> Optimizer:
    """
    Create Shampoo optimizer
    Uses distributed version for multi-GPU, single-GPU version otherwise
    """
    try:
        if tensor_parallel_size > 1:
            from distributed_shampoo.distributed_shampoo import DistributedShampoo
            from distributed_shampoo.shampoo_types import (
                AdamPreconditionerConfig,
                RootInvShampooPreconditionerConfig,
            )

            # Convert dtype string to torch dtype
            dtype_map = {
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
            }
            state_dtype = dtype_map.get(config.shampoo_state_dtype, torch.float32)

            # Create preconditioner config with specified dtype
            preconditioner_config = RootInvShampooPreconditionerConfig(
                factor_matrix_dtype=state_dtype,
                inv_factor_matrix_dtype=state_dtype,
            )

            return DistributedShampoo(
                [p for pg in param_groups for p in pg["params"]],
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                epsilon=config.eps,
                momentum=0.0,
                weight_decay=config.weight_decay,
                max_preconditioner_dim=config.max_precond_dim,
                precondition_frequency=config.precondition_frequency,
                use_decoupled_weight_decay=True,
                grafting_config=AdamPreconditionerConfig(
                    beta2=config.beta2,
                    epsilon=config.eps,
                ),
                preconditioner_config=preconditioner_config,
            )
        else:
            # Single-GPU Shampoo
            return ShampooReference(
                param_groups,
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.eps,
                precondition_frequency=config.precondition_frequency,
                max_precond_dim=config.max_precond_dim,
            )
    except ImportError:
        print("Warning: Distributed Shampoo not available, using reference implementation")
        return ShampooReference(
            param_groups,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            precondition_frequency=config.precondition_frequency,
            max_precond_dim=config.max_precond_dim,
        )


def create_soap_lowbit(
    param_groups: List[Dict],
    config: OptimizerConfig,
    bits: int = 4,
    use_optimized: bool = True,
    orig_param_shapes: Optional[Dict[int, Tuple[str, Tuple[int, ...]]]] = None,
) -> Optimizer:
    """
    Create SOAP with low-bit (4-bit or 8-bit) optimizer states
    Uses custom SOAPLowBit implementation with quantized preconditioners

    Args:
        param_groups: Parameter groups with weight decay settings
        config: Optimizer configuration
        bits: Quantization bits (4 or 8)
        use_optimized: If True, use torchao/gemlite optimized ops; if False, use fallback
        orig_param_shapes: Dict mapping param data_ptr to (name, shape) for FSDP preconditioning
    """
    # Try using SOAPLowBit implementation (supports quantization)
    try:
        print(f"Using SOAP with {bits}-bit quantized states (optimized={use_optimized})")
        return SOAPLowBit(
            param_groups,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            shampoo_beta=config.shampoo_beta,
            eps=config.eps,
            weight_decay=config.weight_decay,
            precondition_frequency=config.precondition_frequency,
            max_precond_dim=config.max_precond_dim,
            bits=bits,
            use_optimized=use_optimized,
            orig_param_shapes=orig_param_shapes,
        )
    except Exception as e:
        print(f"Warning: SOAPLowBit not available ({e.__class__.__name__}: {e})")
        print("Falling back to standard SOAP")
        return create_soap(param_groups, config)


class SOAPReference(Optimizer):
    """
    Reference implementation of SOAP (ShampoO with Adam in Preconditioner eigenbasis)
    Based on https://arxiv.org/abs/2409.11321
    """
    def __init__(
        self,
        params: Iterable,
        lr: float = 3e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
        shampoo_beta: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 10,
        max_precond_dim: int = 8192,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            shampoo_beta=shampoo_beta,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim,
        )
        super().__init__(params, defaults)
        
        self._step = 0

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self._step += 1
        
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            shampoo_beta = group["shampoo_beta"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            precond_freq = group["precondition_frequency"]
            max_dim = group["max_precond_dim"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                    # Initialize Kronecker factors for 2D params (weight matrices)
                    if grad.ndim == 2 and all(d <= max_dim for d in grad.shape):
                        state["L"] = torch.zeros(grad.shape[0], grad.shape[0], device=grad.device, dtype=torch.float32)
                        state["R"] = torch.zeros(grad.shape[-1], grad.shape[-1], device=grad.device, dtype=torch.float32)
                        state["QL"] = torch.eye(grad.shape[0], device=grad.device, dtype=torch.float32)
                        state["QR"] = torch.eye(grad.shape[-1], device=grad.device, dtype=torch.float32)
                
                state["step"] += 1
                
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                
                # Decoupled weight decay
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)
                
                # Check if we should use preconditioning
                use_precond = (
                    grad.ndim >= 2 
                    and "L" in state 
                    and all(d <= max_dim for d in grad.shape)
                )
                
                if use_precond:
                    L, R = state["L"], state["R"]
                    QL, QR = state["QL"], state["QR"]
                    
                    # Update Kronecker factors
                    grad_float = grad.float()
                    if grad.ndim == 2:
                        L.mul_(shampoo_beta).add_(grad_float @ grad_float.T, alpha=1 - shampoo_beta)
                        R.mul_(shampoo_beta).add_(grad_float.T @ grad_float, alpha=1 - shampoo_beta)
                    
                    # Update eigenbasis periodically
                    if state["step"] % precond_freq == 0:
                        # QR decomposition for eigenbasis update
                        QL_new = torch.linalg.qr(L @ QL)[0]
                        QR_new = torch.linalg.qr(R @ QR)[0]
                        
                        # Project momentum to new basis
                        exp_avg_proj = QL.T @ exp_avg.float() @ QR
                        exp_avg_sq_proj = QL.T @ exp_avg_sq.float() @ QR
                        
                        exp_avg_proj = QL_new.T @ exp_avg_proj @ QR_new.T
                        exp_avg_sq_proj = QL_new.T @ exp_avg_sq_proj @ QR_new.T
                        
                        exp_avg.copy_(exp_avg_proj.to(exp_avg.dtype))
                        exp_avg_sq.copy_(exp_avg_sq_proj.to(exp_avg_sq.dtype))
                        
                        state["QL"] = QL_new
                        state["QR"] = QR_new
                        QL, QR = QL_new, QR_new
                    
                    # Project gradient to eigenbasis
                    grad_proj = QL.T @ grad_float @ QR
                    grad = grad_proj.to(grad.dtype)
                
                # Adam update in eigenbasis
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                
                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                
                update = exp_avg / denom
                
                if use_precond:
                    # Project update back to original space
                    QL, QR = state["QL"], state["QR"]
                    update = QL @ update.float() @ QR.T
                    update = update.to(p.dtype)
                
                p.add_(update, alpha=-step_size)
        
        return loss


class ShampooReference(Optimizer):
    """
    Reference implementation of Shampoo optimizer
    Based on https://arxiv.org/abs/1802.09568
    """
    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        precondition_frequency: int = 100,
        max_precond_dim: int = 8192,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim,
        )
        super().__init__(params, defaults)
        
        self._step = 0

    def _matrix_power(self, matrix: Tensor, power: float) -> Tensor:
        """Compute matrix power via eigendecomposition"""
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
            eigenvalues = torch.clamp(eigenvalues, min=1e-10)
            return eigenvectors @ torch.diag(eigenvalues.pow(power)) @ eigenvectors.T
        except:
            # Fallback for numerical issues
            return torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self._step += 1
        
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            precond_freq = group["precondition_frequency"]
            max_dim = group["max_precond_dim"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                state = self.state[p]
                
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    
                    # Initialize preconditioners for 2D params
                    if grad.ndim == 2 and all(d <= max_dim for d in grad.shape):
                        state["L"] = torch.zeros(grad.shape[0], grad.shape[0], device=grad.device, dtype=torch.float32)
                        state["R"] = torch.zeros(grad.shape[1], grad.shape[1], device=grad.device, dtype=torch.float32)
                        state["L_inv"] = torch.eye(grad.shape[0], device=grad.device, dtype=torch.float32)
                        state["R_inv"] = torch.eye(grad.shape[1], device=grad.device, dtype=torch.float32)
                
                state["step"] += 1
                
                # Weight decay
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)
                
                use_precond = grad.ndim == 2 and "L" in state
                
                if use_precond:
                    L, R = state["L"], state["R"]
                    grad_float = grad.float()
                    
                    # Update statistics
                    L.mul_(beta2).add_(grad_float @ grad_float.T, alpha=1 - beta2)
                    R.mul_(beta2).add_(grad_float.T @ grad_float, alpha=1 - beta2)
                    
                    # Update inverse roots periodically
                    if state["step"] % precond_freq == 0:
                        state["L_inv"] = self._matrix_power(L + eps * torch.eye(L.shape[0], device=L.device), -0.25)
                        state["R_inv"] = self._matrix_power(R + eps * torch.eye(R.shape[0], device=R.device), -0.25)
                    
                    # Precondition gradient
                    L_inv, R_inv = state["L_inv"], state["R_inv"]
                    precond_grad = L_inv @ grad_float @ R_inv
                    grad = precond_grad.to(grad.dtype)
                
                # Momentum
                exp_avg = state["exp_avg"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Bias correction
                bias_correction = 1 - beta1 ** state["step"]
                step_size = lr / bias_correction
                
                p.add_(exp_avg, alpha=-step_size)
        
        return loss


# Check for torchao quantization support
try:
    from torchao.quantization import (
        quantize_affine,
        dequantize_affine,
        choose_qparams_affine,
        MappingType,
    )
    TORCHAO_QUANT_AVAILABLE = True
except ImportError:
    TORCHAO_QUANT_AVAILABLE = False
    MappingType = None

# Check for gemlite packing support
try:
    from gemlite.bitpack import pack_weights_over_cols, unpack_over_cols_torch
    GEMLITE_PACK_AVAILABLE = True
except ImportError:
    GEMLITE_PACK_AVAILABLE = False


class SOAPLowBit(Optimizer):
    """
    SOAP optimizer with low-bit (4-bit or 8-bit) quantized eigenbasis states.
    Uses sub-row block quantization (default 64) for 4-bit stability.
    Supports distributed training with FSDP via all-reduce of Kronecker factors.

    Args:
        params: Parameters to optimize
        lr: Learning rate
        betas: Adam beta coefficients
        shampoo_beta: EMA coefficient for Kronecker factors
        eps: Numerical stability constant
        weight_decay: Decoupled weight decay
        precondition_frequency: Steps between eigenbasis updates
        max_precond_dim: Max dimension for preconditioning
        bits: Quantization bits (4 or 8)
        q_block_size: Sub-row block size for quantization (default 64)
        distributed: Enable distributed mode (auto-detected if None)
    """
    def __init__(
        self,
        params: Iterable,
        lr: float = 3e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
        shampoo_beta: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 10,
        max_precond_dim: int = 8192,
        bits: int = 4,
        q_block_size: int = 32,
        use_optimized: bool = True,
        distributed: Optional[bool] = None,
        orig_param_shapes: Optional[Dict[int, Tuple[str, Tuple[int, ...]]]] = None,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            shampoo_beta=shampoo_beta,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim,
            bits=bits,
        )
        super().__init__(params, defaults)

        self._step = 0
        self.bits = bits
        self.q_block_size = q_block_size
        self.use_torchao = use_optimized and TORCHAO_QUANT_AVAILABLE

        # Store original param shapes for FSDP support (flat params -> 2D)
        self.orig_param_shapes = orig_param_shapes or {}

        # Auto-detect distributed mode
        if distributed is None:
            import torch.distributed as dist
            self.distributed = dist.is_initialized()
        else:
            self.distributed = distributed

        dist_str = " (distributed)" if self.distributed else ""
        if self.use_torchao:
            print(f"SOAPLowBit: {bits}-bit, block_size={q_block_size} (torchao){dist_str}")
        else:
            print(f"SOAPLowBit: {bits}-bit, block_size={q_block_size} (fallback){dist_str}")

    def _get_orig_shape(self, p: Tensor) -> Optional[Tuple[int, ...]]:
        """Get original 2D shape for a parameter if it was flattened by FSDP."""
        if p.data_ptr() in self.orig_param_shapes:
            _, orig_shape = self.orig_param_shapes[p.data_ptr()]
            return orig_shape
        return None

    def _is_preconditionable(self, p: Tensor) -> Tuple[bool, Optional[Tuple[int, int]]]:
        """Check if param can be preconditioned, returns (can_precond, 2d_shape)."""
        # Direct 2D case
        if p.ndim == 2:
            return True, tuple(p.shape)

        # Check if this is a flattened FSDP param with original 2D shape
        orig_shape = self._get_orig_shape(p)
        if orig_shape is not None and len(orig_shape) == 2:
            # Verify the total size matches
            if p.numel() == orig_shape[0] * orig_shape[1]:
                return True, orig_shape

        return False, None

    def _quantize_block(self, tensor: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Sub-row block quantization for stability.

        For a 2D tensor [rows, cols], quantizes in blocks of [1, q_block_size].
        Each block gets its own scale/zero_point.

        Returns:
            (quantized, scales, zero_points)
            scales/zp shape: [rows, num_blocks] where num_blocks = ceil(cols / q_block_size)
        """
        if tensor.ndim != 2:
            return self._quantize_per_tensor(tensor)

        if self.use_torchao:
            return self._quantize_block_torchao(tensor)
        return self._quantize_block_fallback(tensor)

    def _quantize_block_torchao(self, tensor: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Block quantization using torchao."""
        quant_min = 0
        quant_max = 2**self.bits - 1
        rows, cols = tensor.shape

        # Pad to multiple of block size
        padded_cols = ((cols + self.q_block_size - 1) // self.q_block_size) * self.q_block_size
        if padded_cols != cols:
            tensor_padded = torch.zeros(rows, padded_cols, device=tensor.device, dtype=tensor.dtype)
            tensor_padded[:, :cols] = tensor
        else:
            tensor_padded = tensor

        # Block size: (1, q_block_size) means sub-row blocks
        block_size = (1, self.q_block_size)

        scale, zero_point = choose_qparams_affine(
            tensor_padded,
            mapping_type=MappingType.ASYMMETRIC,
            block_size=block_size,
            target_dtype=torch.uint8,
            quant_min=quant_min,
            quant_max=quant_max,
            scale_dtype=tensor.dtype,
            zero_point_dtype=torch.float32,
        )

        quantized = quantize_affine(
            tensor_padded,
            block_size=block_size,
            scale=scale,
            zero_point=zero_point.to(torch.int32),
            output_dtype=torch.uint8,
            quant_min=quant_min,
            quant_max=quant_max,
        )

        # Store original cols for dequantization
        return quantized, scale, zero_point, cols

    def _quantize_block_fallback(self, tensor: Tensor) -> Tuple[Tensor, Tensor, Tensor, int]:
        """Sub-row block quantization fallback (pure PyTorch)."""
        quant_max = 2**self.bits - 1
        rows, cols = tensor.shape
        bs = self.q_block_size

        # Pad to multiple of block size
        padded_cols = ((cols + bs - 1) // bs) * bs
        num_blocks = padded_cols // bs

        if padded_cols != cols:
            tensor_padded = torch.zeros(rows, padded_cols, device=tensor.device, dtype=tensor.dtype)
            tensor_padded[:, :cols] = tensor
        else:
            tensor_padded = tensor

        # Reshape to [rows, num_blocks, block_size]
        tensor_blocks = tensor_padded.view(rows, num_blocks, bs)

        # Per-block min/max
        min_val = tensor_blocks.min(dim=2, keepdim=True)[0]  # [rows, num_blocks, 1]
        max_val = tensor_blocks.max(dim=2, keepdim=True)[0]

        scale = (max_val - min_val) / quant_max
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        zero_point = (-min_val / scale).round()

        quantized_blocks = ((tensor_blocks / scale) + zero_point).round().clamp(0, quant_max).to(torch.uint8)
        quantized = quantized_blocks.view(rows, padded_cols)

        return quantized, scale.squeeze(2), zero_point.squeeze(2), cols

    def _dequantize_block(self, quantized: Tensor, scale: Tensor, zero_point: Tensor, orig_cols: int) -> Tensor:
        """Dequantize block-quantized tensor.

        Args:
            quantized: uint8 tensor [rows, padded_cols]
            scale: Per-block scales [rows, num_blocks]
            zero_point: Per-block zero points [rows, num_blocks]
            orig_cols: Original number of columns before padding
        """
        # Per-tensor case: orig_cols is None
        if orig_cols is None:
            return self._dequantize_per_tensor(quantized, scale, zero_point)

        if self.use_torchao:
            return self._dequantize_block_torchao(quantized, scale, zero_point, orig_cols)
        return self._dequantize_block_fallback(quantized, scale, zero_point, orig_cols)

    def _dequantize_block_torchao(self, quantized: Tensor, scale: Tensor, zero_point: Tensor, orig_cols: int) -> Tensor:
        """Dequantize using torchao."""
        block_size = (1, self.q_block_size)
        dequantized = dequantize_affine(
            quantized,
            block_size=block_size,
            scale=scale,
            zero_point=zero_point.to(torch.int32),
            input_dtype=torch.uint8,
            quant_min=0,
            quant_max=2**self.bits - 1,
            output_dtype=torch.float32,
        )
        return dequantized[:, :orig_cols]

    def _dequantize_block_fallback(self, quantized: Tensor, scale: Tensor, zero_point: Tensor, orig_cols: int) -> Tensor:
        """Dequantize sub-row blocks (fallback)."""
        rows, padded_cols = quantized.shape
        bs = self.q_block_size
        num_blocks = padded_cols // bs

        # Reshape to blocks
        quantized_blocks = quantized.view(rows, num_blocks, bs).float()
        # scale, zero_point: [rows, num_blocks] -> [rows, num_blocks, 1]
        dequant_blocks = (quantized_blocks - zero_point.unsqueeze(2)) * scale.unsqueeze(2)
        dequantized = dequant_blocks.view(rows, padded_cols)
        return dequantized[:, :orig_cols]

    def _quantize_per_tensor(self, tensor: Tensor) -> Tuple[Tensor, Tensor, Tensor, None]:
        """Per-tensor quantization for 1D tensors."""
        quant_max = 2**self.bits - 1
        min_val = tensor.min()
        max_val = tensor.max()
        scale = (max_val - min_val) / quant_max
        if scale == 0:
            scale = torch.tensor(1.0, device=tensor.device, dtype=tensor.dtype)
        zero_point = (-min_val / scale).round()
        quantized = ((tensor / scale) + zero_point).round().clamp(0, quant_max).to(torch.uint8)
        return quantized, scale, zero_point, None

    def _dequantize_per_tensor(self, quantized: Tensor, scale: Tensor, zero_point: Tensor) -> Tensor:
        """Per-tensor dequantization."""
        return (quantized.float() - zero_point) * scale

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step += 1

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            shampoo_beta = group["shampoo_beta"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            precond_freq = group["precondition_frequency"]
            max_dim = group["max_precond_dim"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                    # Check if param can be preconditioned (2D or flattened 2D from FSDP)
                    can_precond, shape_2d = self._is_preconditionable(p)
                    if can_precond and shape_2d is not None:
                        m, n = shape_2d
                        state["orig_2d_shape"] = (m, n)  # Store for reshaping

                        # Compute number of blocks needed for each dimension
                        num_blocks_row = (m + max_dim - 1) // max_dim  # ceil division
                        num_blocks_col = (n + max_dim - 1) // max_dim

                        # Store block structure info
                        state["num_blocks_row"] = num_blocks_row
                        state["num_blocks_col"] = num_blocks_col
                        state["block_size"] = max_dim

                        # Initialize only eigenbases for each block (quantized)
                        # Kronecker factors L/R are accumulated only during eigenbasis updates
                        state["QL_q_blocks"] = []
                        state["QR_q_blocks"] = []
                        state["block_shapes"] = []  # Store block dimensions

                        for i in range(num_blocks_row):
                            row_start = i * max_dim
                            row_end = min((i + 1) * max_dim, m)
                            block_m = row_end - row_start

                            for j in range(num_blocks_col):
                                col_start = j * max_dim
                                col_end = min((j + 1) * max_dim, n)
                                block_n = col_end - col_start

                                # Eigenbasis for this block (identity initially, will be quantized)
                                QL_block = torch.eye(block_m, device=grad.device, dtype=torch.float32)
                                QR_block = torch.eye(block_n, device=grad.device, dtype=torch.float32)

                                # Immediately quantize to save memory
                                state["QL_q_blocks"].append(self._quantize_block(QL_block))
                                state["QR_q_blocks"].append(self._quantize_block(QR_block))
                                state["block_shapes"].append((block_m, block_n))

                state["step"] += 1

                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Check if we should use preconditioning (handles both native 2D and FSDP flattened)
                use_precond = "QL_q_blocks" in state and "orig_2d_shape" in state

                if use_precond:
                    m, n = state["orig_2d_shape"]
                    orig_grad_shape = grad.shape
                    # Reshape to 2D if flattened by FSDP
                    grad_2d = grad.view(m, n) if grad.ndim == 1 else grad
                    grad_float = grad_2d.float()
                    max_dim = state["block_size"]
                    num_blocks_row = state["num_blocks_row"]
                    num_blocks_col = state["num_blocks_col"]

                    # Process each block
                    grad_preconditioned = torch.zeros_like(grad_float)
                    needs_eigenbasis_update = (state["step"] % precond_freq == 0)

                    # Temporary storage for Kronecker factors (only if updating eigenbasis)
                    if needs_eigenbasis_update:
                        L_accum = {}
                        R_accum = {}

                    for i in range(num_blocks_row):
                        row_start = i * max_dim
                        row_end = min((i + 1) * max_dim, m)

                        for j in range(num_blocks_col):
                            col_start = j * max_dim
                            col_end = min((j + 1) * max_dim, n)

                            block_idx = i * num_blocks_col + j
                            grad_block = grad_float[row_start:row_end, col_start:col_end]
                            block_m, block_n = state["block_shapes"][block_idx]

                            # Dequantize eigenbasis
                            QL_block = self._dequantize_block(*state["QL_q_blocks"][block_idx])
                            QR_block = self._dequantize_block(*state["QR_q_blocks"][block_idx])

                            # Accumulate Kronecker factors if we're updating eigenbasis
                            if needs_eigenbasis_update:
                                if block_idx not in L_accum:
                                    L_accum[block_idx] = torch.zeros(
                                        block_m, block_m, device=grad.device, dtype=torch.float32
                                    )
                                    R_accum[block_idx] = torch.zeros(
                                        block_n, block_n, device=grad.device, dtype=torch.float32
                                    )

                                L_accum[block_idx].add_(grad_block @ grad_block.T)
                                R_accum[block_idx].add_(grad_block.T @ grad_block)

                            # Precondition this block
                            grad_block_precond = QL_block.T @ grad_block @ QR_block
                            grad_preconditioned[row_start:row_end, col_start:col_end] = grad_block_precond

                    # Update eigenbases if needed
                    if needs_eigenbasis_update:
                        # Note: With FSDP, each rank computes local preconditioners on its gradient shard.
                        # All-reduce is skipped to avoid deadlock from FSDP's parameter sharding.
                        # Local preconditioners still provide significant benefit.

                        for block_idx in range(num_blocks_row * num_blocks_col):
                            QL_block = self._dequantize_block(*state["QL_q_blocks"][block_idx])
                            QR_block = self._dequantize_block(*state["QR_q_blocks"][block_idx])

                            # Update eigenbasis using accumulated Kronecker factors
                            L_block = L_accum[block_idx]
                            R_block = R_accum[block_idx]

                            QL_new = torch.linalg.qr(L_block @ QL_block)[0]
                            QR_new = torch.linalg.qr(R_block @ QR_block)[0]

                            # Quantize and store
                            state["QL_q_blocks"][block_idx] = self._quantize_block(QL_new)
                            state["QR_q_blocks"][block_idx] = self._quantize_block(QR_new)

                    # Reshape back to original shape if needed (for FSDP flattened params)
                    grad = grad_preconditioned.to(grad_2d.dtype).view(orig_grad_shape)

                # Adam update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                update = exp_avg / denom

                if use_precond:
                    # Apply block diagonal preconditioner to update
                    # Reshape to 2D if FSDP flattened
                    update_2d = update.view(m, n) if update.ndim == 1 else update
                    update_float = update_2d.float()
                    update_preconditioned = torch.zeros_like(update_float)

                    for i in range(num_blocks_row):
                        row_start = i * max_dim
                        row_end = min((i + 1) * max_dim, m)

                        for j in range(num_blocks_col):
                            col_start = j * max_dim
                            col_end = min((j + 1) * max_dim, n)

                            block_idx = i * num_blocks_col + j
                            update_block = update_float[row_start:row_end, col_start:col_end]

                            # Dequantize eigenbasis for this block
                            if state["QL_q_blocks"][block_idx] is not None:
                                QL_block = self._dequantize_block(*state["QL_q_blocks"][block_idx])
                                QR_block = self._dequantize_block(*state["QR_q_blocks"][block_idx])
                            else:
                                QL_block = state["QL_blocks"][block_idx]
                                QR_block = state["QR_blocks"][block_idx]

                            # Apply preconditioner to this block
                            update_block_precond = QL_block @ update_block @ QR_block.T
                            update_preconditioned[row_start:row_end, col_start:col_end] = update_block_precond

                    # Reshape back to original shape for FSDP
                    update = update_preconditioned.to(p.dtype).view(p.shape)

                p.add_(update, alpha=-step_size)
        
        return loss


def get_lr_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    num_training_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create learning rate scheduler"""
    
    if scheduler_type == "cosine":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
            return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    elif scheduler_type == "linear":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
            return max(min_lr_ratio, 1 - progress * (1 - min_lr_ratio))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    elif scheduler_type == "constant_with_warmup":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return 1.0
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def analyze_preconditioning_coverage(
    orig_param_shapes: Dict[int, Tuple[str, Tuple[int, ...]]],
    optimizer_config: OptimizerConfig,
) -> Dict[str, Any]:
    """
    Analyze what percentage of parameters will be preconditioned in SOAP/Shampoo.

    Parameters are preconditioned if:
    - They are 2D (weight matrices)
    - Both dimensions are <= max_precond_dim
    - They require gradients

    Args:
        orig_param_shapes: Dict mapping data_ptr to (name, shape) of original params
        optimizer_config: Optimizer configuration

    Returns:
        Dictionary with statistics about preconditioning coverage
    """
    max_dim = optimizer_config.max_precond_dim
    optimizer_type = optimizer_config.optimizer_type

    # Only analyze for SOAP/Shampoo optimizers
    is_soap_or_shampoo = optimizer_type in [
        OptimizerType.SOAP,
        OptimizerType.SOAP_4BIT,
        OptimizerType.SOAP_8BIT,
        OptimizerType.SHAMPOO,
    ]

    if not is_soap_or_shampoo:
        return {
            "optimizer_type": optimizer_type.value,
            "uses_preconditioning": False,
        }

    total_params = 0
    preconditioned_params = 0

    total_2d_params = 0
    preconditioned_2d_params = 0

    param_details = {
        "preconditioned": [],
        "not_preconditioned_too_large": [],
        "not_preconditioned_1d": [],
    }

    for data_ptr, (name, shape) in orig_param_shapes.items():
        num_params = 1
        for dim in shape:
            num_params *= dim
        total_params += num_params

        # Check if this parameter will be preconditioned
        # With block diagonal, ALL 2D parameters get preconditioned
        if len(shape) == 2:
            total_2d_params += num_params
            preconditioned_params += num_params
            preconditioned_2d_params += num_params

            # Determine number of blocks needed
            m, n = shape
            num_blocks_row = (m + max_dim - 1) // max_dim
            num_blocks_col = (n + max_dim - 1) // max_dim
            num_blocks = num_blocks_row * num_blocks_col

            param_details["preconditioned"].append({
                "name": name,
                "shape": shape,
                "num_params": num_params,
                "num_blocks": num_blocks,
            })
        elif len(shape) > 2:
            # 3D+ parameters not preconditioned (e.g., Conv1d kernels)
            param_details["not_preconditioned_too_large"].append({
                "name": name,
                "shape": shape,
                "num_params": num_params,
            })
        else:
            param_details["not_preconditioned_1d"].append({
                "name": name,
                "shape": shape,
                "num_params": num_params,
            })

    precond_pct = (preconditioned_params / total_params * 100) if total_params > 0 else 0.0
    precond_2d_pct = (preconditioned_2d_params / total_2d_params * 100) if total_2d_params > 0 else 0.0

    return {
        "optimizer_type": optimizer_type.value,
        "uses_preconditioning": True,
        "max_precond_dim": max_dim,
        "total_params": total_params,
        "preconditioned_params": preconditioned_params,
        "preconditioned_percentage": precond_pct,
        "total_2d_params": total_2d_params,
        "preconditioned_2d_params": preconditioned_2d_params,
        "preconditioned_2d_percentage": precond_2d_pct,
        "num_preconditioned": len(param_details["preconditioned"]),
        "num_too_large": len(param_details["not_preconditioned_too_large"]),
        "num_1d": len(param_details["not_preconditioned_1d"]),
        "details": param_details,
    }


def print_preconditioning_stats(
    orig_param_shapes: Dict[int, Tuple[str, Tuple[int, ...]]],
    optimizer_config: OptimizerConfig
):
    """
    Print statistics about which parameters will be preconditioned.

    Args:
        orig_param_shapes: Dict mapping data_ptr to (name, shape) of original params
        optimizer_config: Optimizer configuration
    """
    stats = analyze_preconditioning_coverage(orig_param_shapes, optimizer_config)

    if not stats["uses_preconditioning"]:
        return

    print('\n')
    print(f"Max preconditioner dimension: {stats['max_precond_dim']} (block diagonal)")
    print(f"  Total trainable parameters: {stats['total_params']:,}")
    print(f"  Preconditioned parameters:  {stats['preconditioned_params']:,} ({stats['preconditioned_percentage']:.1f}%)")

    # Count total blocks across all layers
    total_blocks = sum(item.get("num_blocks", 1) for item in stats['details']['preconditioned'])
    print(f"  Total preconditioner blocks: {total_blocks}")

    # Show largest layers that are now preconditioned with blocks
    large_preconditioned = [
        item for item in stats['details']['preconditioned']
        if item.get("num_blocks", 1) > 1
    ]

    if large_preconditioned:
        large_preconditioned_sorted = sorted(
            large_preconditioned,
            key=lambda x: x["num_params"],
            reverse=True
        )[:5]

        print(f"\n  Large layers preconditioned with block diagonal:")
        for item in large_preconditioned_sorted:
            print(f"    {item['name']:<50} {str(item['shape']):<20} {item['num_blocks']:>2} blocks")

    print()
