#!/usr/bin/env python3
import torch
import torch.nn as nn
from typing import Literal
import argparse
from dataclasses import dataclass
import time
import numpy as np

from models.kernels import (
    create_rms_norm,
    create_mlp,
    get_rotary_pos_emb_fn,
    create_cross_entropy_loss,
    LIGER_AVAILABLE,
    TRITON_AVAILABLE,
)


@dataclass
class ProfileResult:
    kernel_type: str
    operation: str
    forward_mean_ms: float
    forward_std_ms: float
    backward_mean_ms: float
    backward_std_ms: float
    total_mean_ms: float
    memory_allocated_mb: float
    memory_reserved_mb: float


def profile_operation(
    module: nn.Module,
    inputs: dict,
    warmup: int = 10,
    iterations: int = 100,
    backward: bool = True,
) -> tuple[float, float, float, float]:
    """Profile forward and backward passes separately."""
    torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        out = module(**inputs)
        if backward and isinstance(out, torch.Tensor):
            out.sum().backward()
            module.zero_grad(set_to_none=True)

    torch.cuda.synchronize()

    # Forward pass timing
    forward_times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = module(**inputs)
        torch.cuda.synchronize()
        forward_times.append((time.perf_counter() - start) * 1000)

    # Backward pass timing
    backward_times = []
    if backward and isinstance(out, torch.Tensor):
        for _ in range(iterations):
            out = module(**inputs)
            torch.cuda.synchronize()
            start = time.perf_counter()
            out.sum().backward()
            torch.cuda.synchronize()
            backward_times.append((time.perf_counter() - start) * 1000)
            module.zero_grad(set_to_none=True)

    forward_mean = float(np.mean(forward_times))
    forward_std = float(np.std(forward_times))
    backward_mean = float(np.mean(backward_times)) if backward_times else 0.0
    backward_std = float(np.std(backward_times)) if backward_times else 0.0

    return forward_mean, forward_std, backward_mean, backward_std


def profile_rmsnorm(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    kernel_type: Literal["baseline", "triton", "liger"],
    warmup: int,
    iterations: int,
) -> ProfileResult:
    """Profile RMSNorm kernel."""
    device = "cuda"

    norm = create_rms_norm(hidden_size, kernel_type=kernel_type).to(device)
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, requires_grad=True)

    # Liger uses 'hidden_states', others use 'x'
    class NormWrapper(nn.Module):
        def __init__(self, norm_module):
            super().__init__()
            self.norm = norm_module

        def forward(self, x):
            # Try Liger API first
            try:
                return self.norm(x)
            except TypeError:
                # Fallback to standard API
                return self.norm(x=x)

    wrapper = NormWrapper(norm).to(device)
    inputs = {"x": x}

    fwd_mean, fwd_std, bwd_mean, bwd_std = profile_operation(
        wrapper, inputs, warmup, iterations
    )

    mem_alloc = torch.cuda.max_memory_allocated() / 1024**2
    mem_reserved = torch.cuda.max_memory_reserved() / 1024**2

    return ProfileResult(
        kernel_type=kernel_type,
        operation="RMSNorm",
        forward_mean_ms=fwd_mean,
        forward_std_ms=fwd_std,
        backward_mean_ms=bwd_mean,
        backward_std_ms=bwd_std,
        total_mean_ms=fwd_mean + bwd_mean,
        memory_allocated_mb=mem_alloc,
        memory_reserved_mb=mem_reserved,
    )


def profile_swiglu(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    kernel_type: Literal["baseline", "triton", "liger"],
    warmup: int,
    iterations: int,
) -> ProfileResult:
    """Profile SwiGLU MLP kernel."""
    device = "cuda"

    mlp = create_mlp(hidden_size, intermediate_size, kernel_type=kernel_type).to(device)
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, requires_grad=True)

    inputs = {"x": x}

    fwd_mean, fwd_std, bwd_mean, bwd_std = profile_operation(
        mlp, inputs, warmup, iterations
    )

    mem_alloc = torch.cuda.max_memory_allocated() / 1024**2
    mem_reserved = torch.cuda.max_memory_reserved() / 1024**2

    return ProfileResult(
        kernel_type=kernel_type,
        operation="SwiGLU_MLP",
        forward_mean_ms=fwd_mean,
        forward_std_ms=fwd_std,
        backward_mean_ms=bwd_mean,
        backward_std_ms=bwd_std,
        total_mean_ms=fwd_mean + bwd_mean,
        memory_allocated_mb=mem_alloc,
        memory_reserved_mb=mem_reserved,
    )


def profile_rope(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    kernel_type: Literal["baseline", "liger"],
    warmup: int,
    iterations: int,
) -> ProfileResult:
    """Profile Rotary Position Embedding."""
    device = "cuda"

    apply_rotary_fn = get_rotary_pos_emb_fn(kernel_type=kernel_type)

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True)
    cos = torch.randn(1, 1, seq_len, head_dim, device=device)
    sin = torch.randn(1, 1, seq_len, head_dim, device=device)

    class RoPEModule(nn.Module):
        def __init__(self, apply_fn):
            super().__init__()
            self.apply_fn = apply_fn

        def forward(self, q, k, cos, sin):
            return self.apply_fn(q, k, cos, sin)

    module = RoPEModule(apply_rotary_fn).to(device)
    inputs = {"q": q, "k": k, "cos": cos, "sin": sin}

    fwd_mean, fwd_std, bwd_mean, bwd_std = profile_operation(
        module, inputs, warmup, iterations
    )

    mem_alloc = torch.cuda.max_memory_allocated() / 1024**2
    mem_reserved = torch.cuda.max_memory_reserved() / 1024**2

    return ProfileResult(
        kernel_type=kernel_type,
        operation="RoPE",
        forward_mean_ms=fwd_mean,
        forward_std_ms=fwd_std,
        backward_mean_ms=bwd_mean,
        backward_std_ms=bwd_std,
        total_mean_ms=fwd_mean + bwd_mean,
        memory_allocated_mb=mem_alloc,
        memory_reserved_mb=mem_reserved,
    )


def profile_fused_ce(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    vocab_size: int,
    kernel_type: Literal["baseline", "liger"],
    warmup: int,
    iterations: int,
) -> ProfileResult:
    """Profile Fused Linear CrossEntropy."""
    device = "cuda"

    lm_head = nn.Linear(hidden_size, vocab_size, bias=False).to(device)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, requires_grad=True)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    loss_fn = create_cross_entropy_loss(lm_head, kernel_type=kernel_type)

    class CEModule(nn.Module):
        def __init__(self, lm_head, loss_fn):
            super().__init__()
            self.lm_head = lm_head
            self.loss_fn = loss_fn

        def forward(self, hidden_states, labels):
            from models.kernels import compute_cross_entropy_loss
            return compute_cross_entropy_loss(hidden_states, self.lm_head, labels, self.loss_fn)

    module = CEModule(lm_head, loss_fn).to(device)
    inputs = {"hidden_states": hidden_states, "labels": labels}

    fwd_mean, fwd_std, bwd_mean, bwd_std = profile_operation(
        module, inputs, warmup, iterations
    )

    mem_alloc = torch.cuda.max_memory_allocated() / 1024**2
    mem_reserved = torch.cuda.max_memory_reserved() / 1024**2

    return ProfileResult(
        kernel_type=kernel_type,
        operation="FusedLinearCE",
        forward_mean_ms=fwd_mean,
        forward_std_ms=fwd_std,
        backward_mean_ms=bwd_mean,
        backward_std_ms=bwd_std,
        total_mean_ms=fwd_mean + bwd_mean,
        memory_allocated_mb=mem_alloc,
        memory_reserved_mb=mem_reserved,
    )


def print_results(results: list[ProfileResult]):
    """Print profiling results in table format."""
    print("\n" + "="*120)
    print(f"{'Operation':<18} {'Kernel':<10} {'Fwd (ms)':<15} {'Bwd (ms)':<15} {'Total (ms)':<15} {'Mem Alloc (MB)':<15}")
    print("="*120)

    for result in results:
        print(
            f"{result.operation:<18} {result.kernel_type:<10} "
            f"{result.forward_mean_ms:>6.3f} ± {result.forward_std_ms:<5.3f} "
            f"{result.backward_mean_ms:>6.3f} ± {result.backward_std_ms:<5.3f} "
            f"{result.total_mean_ms:>6.3f}         "
            f"{result.memory_allocated_mb:>10.2f}"
        )

    print("="*120)

    # Speedup analysis
    print("\n" + "="*120)
    print("SPEEDUP ANALYSIS (vs Baseline)")
    print("="*120)

    operations = set(r.operation for r in results)
    for op in operations:
        op_results = [r for r in results if r.operation == op]
        baseline = next((r for r in op_results if r.kernel_type == "baseline"), None)

        if baseline:
            print(f"\n{op}:")
            for r in op_results:
                if r.kernel_type != "baseline":
                    speedup = baseline.total_mean_ms / r.total_mean_ms
                    print(f"  {r.kernel_type:<10} {speedup:.2f}x speedup")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--hidden_size", type=int, default=2560)
    parser.add_argument("--intermediate_size", type=int, default=6912)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    head_dim = args.hidden_size // args.num_heads

    print(f"\nKernel Profiling Configuration:")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Sequence Length: {args.seq_len}")
    print(f"  Hidden Size: {args.hidden_size}")
    print(f"  Intermediate Size: {args.intermediate_size}")
    print(f"  Vocab Size: {args.vocab_size}")
    print(f"  Warmup: {args.warmup}")
    print(f"  Iterations: {args.iterations}")
    print(f"\nAvailable Kernels:")
    print(f"  Liger: {LIGER_AVAILABLE}")
    print(f"  Triton: {TRITON_AVAILABLE}")

    results = []

    # RMSNorm
    print("\n[1/4] Profiling RMSNorm...")
    for kernel in ["baseline", "triton", "liger"]:
        torch.cuda.reset_peak_memory_stats()
        try:
            result = profile_rmsnorm(
                args.batch_size, args.seq_len, args.hidden_size,
                kernel, args.warmup, args.iterations
            )
            results.append(result)
            print(f"  {kernel}: {result.total_mean_ms:.3f}ms")
        except Exception as e:
            print(f"  {kernel}: FAILED ({e})")

    # SwiGLU
    print("\n[2/4] Profiling SwiGLU MLP...")
    for kernel in ["baseline", "triton", "liger"]:
        torch.cuda.reset_peak_memory_stats()
        try:
            result = profile_swiglu(
                args.batch_size, args.seq_len, args.hidden_size, args.intermediate_size,
                kernel, args.warmup, args.iterations
            )
            results.append(result)
            print(f"  {kernel}: {result.total_mean_ms:.3f}ms")
        except Exception as e:
            print(f"  {kernel}: FAILED ({e})")

    # RoPE (no triton)
    print("\n[3/4] Profiling RoPE...")
    for kernel in ["baseline", "liger"]:
        torch.cuda.reset_peak_memory_stats()
        try:
            result = profile_rope(
                args.batch_size, args.seq_len, args.num_heads, head_dim,
                kernel, args.warmup, args.iterations
            )
            results.append(result)
            print(f"  {kernel}: {result.total_mean_ms:.3f}ms")
        except Exception as e:
            print(f"  {kernel}: FAILED ({e})")

    # Fused CE (no triton)
    print("\n[4/4] Profiling Fused Linear CrossEntropy...")
    for kernel in ["baseline", "liger"]:
        torch.cuda.reset_peak_memory_stats()
        try:
            result = profile_fused_ce(
                args.batch_size, args.seq_len, args.hidden_size, args.vocab_size,
                kernel, args.warmup, args.iterations
            )
            results.append(result)
            print(f"  {kernel}: {result.total_mean_ms:.3f}ms")
        except Exception as e:
            print(f"  {kernel}: FAILED ({e})")

    print_results(results)


if __name__ == "__main__":
    main()
