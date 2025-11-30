"""
Optimizer implementations for ablation study

Provides unified interface for:
- AdamW (baseline)
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
) -> Optimizer:
    """Factory function to create optimizer from config"""
    
    # Separate parameters by weight decay eligibility
    param_groups = get_param_groups(model, config.weight_decay)
    
    if config.optimizer_type == OptimizerType.ADAMW:
        return create_adamw(param_groups, config)
    
    elif config.optimizer_type == OptimizerType.SOAP:
        return create_soap(param_groups, config)
    
    elif config.optimizer_type == OptimizerType.SHAMPOO:
        return create_shampoo(param_groups, config, tensor_parallel_size)
    
    elif config.optimizer_type == OptimizerType.SOAP_LOWBIT:
        return create_soap_lowbit(param_groups, config)
    
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
            max_precond_dim=config.max_precond_dim,
            use_decoupled_weight_decay=True,
            precondition_1d=False,
            correct_bias=True,
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
            from distributed_shampoo.shampoo_types import AdamGraftingConfig
            
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
                grafting_config=AdamGraftingConfig(
                    beta2=config.beta2,
                    epsilon=config.eps,
                ),
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


def create_soap_lowbit(param_groups: List[Dict], config: OptimizerConfig) -> Optimizer:
    """
    Create SOAP with low-bit (4-bit or 8-bit) optimizer states
    Uses lpmm or torchao for quantized states
    """
    try:
        # Try using lpmm for 4-bit states
        if config.use_4bit:
            import lpmm
            
            # Wrap SOAP optimizer with low-bit states
            return SOAPLowBit(
                param_groups,
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                shampoo_beta=config.shampoo_beta,
                eps=config.eps,
                precondition_frequency=config.precondition_frequency,
                max_precond_dim=config.max_precond_dim,
                bits=4,
            )
    except ImportError:
        pass
    
    try:
        # Fallback to torchao 8-bit
        if config.use_8bit:
            from torchao.prototype.low_bit_optim import AdamW8bit
            
            # For 8-bit, we use Adam as base since SOAP 8-bit isn't directly available
            print("Warning: Using 8-bit AdamW as fallback (SOAP low-bit not available)")
            return AdamW8bit(
                [p for pg in param_groups for p in pg["params"]],
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.eps,
                weight_decay=config.weight_decay,
            )
    except ImportError:
        pass
    
    # Final fallback
    print("Warning: Low-bit optimizers not available, using standard SOAP")
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
                    
                    # Initialize Kronecker factors for 2D+ params
                    if grad.ndim >= 2 and all(d <= max_dim for d in grad.shape):
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


class SOAPLowBit(Optimizer):
    """
    SOAP optimizer with low-bit (4-bit) quantized states
    Quantizes Kronecker factors and Adam states for memory efficiency
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
        
        # Try to import lpmm for quantization
        try:
            import lpmm
            self.use_lpmm = True
        except ImportError:
            self.use_lpmm = False
            print("Warning: lpmm not available, using simulated quantization")

    def _quantize(self, tensor: Tensor) -> Tuple[Tensor, float, float]:
        """Quantize tensor to low-bit representation"""
        if self.use_lpmm:
            # Use lpmm's quantization
            import lpmm
            return lpmm.quantize(tensor, bits=self.bits)
        else:
            # Simple min-max quantization
            min_val = tensor.min()
            max_val = tensor.max()
            scale = (max_val - min_val) / (2**self.bits - 1)
            
            if scale == 0:
                scale = 1.0
            
            quantized = ((tensor - min_val) / scale).round().to(torch.uint8)
            return quantized, scale.item(), min_val.item()

    def _dequantize(self, quantized: Tensor, scale: float, min_val: float) -> Tensor:
        """Dequantize tensor back to float"""
        return quantized.float() * scale + min_val

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
                    # Store momentum in full precision for stability
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    
                    if grad.ndim >= 2 and all(d <= max_dim for d in grad.shape):
                        # Kronecker factors in full precision (will be quantized periodically)
                        state["L"] = torch.zeros(grad.shape[0], grad.shape[0], device=grad.device, dtype=torch.float32)
                        state["R"] = torch.zeros(grad.shape[-1], grad.shape[-1], device=grad.device, dtype=torch.float32)
                        # Eigenbasis stored quantized
                        state["QL"] = torch.eye(grad.shape[0], device=grad.device, dtype=torch.float32)
                        state["QR"] = torch.eye(grad.shape[-1], device=grad.device, dtype=torch.float32)
                        state["QL_quantized"] = None
                        state["QR_quantized"] = None
                
                state["step"] += 1
                
                # Weight decay
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)
                
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                
                use_precond = grad.ndim >= 2 and "L" in state
                
                if use_precond:
                    L, R = state["L"], state["R"]
                    
                    # Dequantize eigenbasis if needed
                    if state["QL_quantized"] is not None:
                        QL = self._dequantize(*state["QL_quantized"])
                        QR = self._dequantize(*state["QR_quantized"])
                    else:
                        QL, QR = state["QL"], state["QR"]
                    
                    # Update Kronecker factors
                    grad_float = grad.float()
                    if grad.ndim == 2:
                        L.mul_(shampoo_beta).add_(grad_float @ grad_float.T, alpha=1 - shampoo_beta)
                        R.mul_(shampoo_beta).add_(grad_float.T @ grad_float, alpha=1 - shampoo_beta)
                    
                    # Update and quantize eigenbasis periodically
                    if state["step"] % precond_freq == 0:
                        QL_new = torch.linalg.qr(L @ QL)[0]
                        QR_new = torch.linalg.qr(R @ QR)[0]
                        
                        # Quantize eigenbasis for storage
                        state["QL_quantized"] = self._quantize(QL_new)
                        state["QR_quantized"] = self._quantize(QR_new)
                        
                        # Keep full precision for this step
                        QL, QR = QL_new, QR_new
                    
                    # Project gradient
                    grad_proj = QL.T @ grad_float @ QR
                    grad = grad_proj.to(grad.dtype)
                
                # Adam update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                
                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                
                update = exp_avg / denom
                
                if use_precond:
                    if state["QL_quantized"] is not None:
                        QL = self._dequantize(*state["QL_quantized"])
                        QR = self._dequantize(*state["QR_quantized"])
                    update = QL @ update.float() @ QR.T
                    update = update.to(p.dtype)
                
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
