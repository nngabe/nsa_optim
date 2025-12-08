"""
Benchmark comparing Triton, Liger and Baseline kernel implementations.

Benchmarks:
1. RMSNorm: baseline vs triton vs liger
2. SwiGLU MLP: baseline vs triton vs liger
3. Rotary Position Embedding: baseline vs liger
4. Fused Linear Cross Entropy: baseline vs liger
"""
import time
import argparse
from typing import Dict, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""
    name: str
    kernel_type: str
    batch_size: int
    seq_len: int
    hidden_size: int
    mean_time_ms: float
    std_time_ms: float
    throughput_toks_per_sec: float


def warmup_gpu():
    """Warmup GPU with dummy computation"""
    x = torch.randn(1024, 1024, device='cuda')
    for _ in range(10):
        x = x @ x
    torch.cuda.synchronize()


def benchmark_fn(fn, inputs: Dict, warmup: int = 10, iterations: int = 100) -> Tuple[float, float]:
    """Benchmark a function with warmup and multiple iterations"""
    # Warmup
    for _ in range(warmup):
        fn(**inputs)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn(**inputs)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
    return mean_time, std_time


# ============================================================================
# Baseline Implementations
# ============================================================================

class BaselineRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(dtype)


class BaselineSwiGLUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def baseline_rotary_pos_emb(q, k, cos, sin):
    """Baseline rotary position embedding"""
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def baseline_cross_entropy(hidden_states, weight, labels):
    """Baseline cross entropy loss computation"""
    logits = F.linear(hidden_states, weight)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
    return loss


# ============================================================================
# Import optimized implementations
# ============================================================================

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Warning: Triton not available")

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
    print("Warning: Liger kernels not available")


# Triton RMSNorm
if TRITON_AVAILABLE:
    @triton.jit
    def _rms_norm_fwd_kernel(
        X_ptr, W_ptr, Y_ptr,
        stride_x_row, stride_y_row,
        N, eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        X_row_ptr = X_ptr + row_idx * stride_x_row
        Y_row_ptr = Y_ptr + row_idx * stride_y_row

        _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for col_start in range(0, N, BLOCK_SIZE):
            col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
            mask = col_offsets < N
            x = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            _var += x * x

        var = tl.sum(_var) / N
        rstd = tl.rsqrt(var + eps)

        for col_start in range(0, N, BLOCK_SIZE):
            col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
            mask = col_offsets < N
            x = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            y = x * rstd * w
            tl.store(Y_row_ptr + col_offsets, y, mask=mask)


    class TritonRMSNorm(nn.Module):
        def __init__(self, hidden_size: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps
            self.hidden_size = hidden_size

        def forward(self, x):
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
        Gate_ptr, Up_ptr, Y_ptr,
        stride_g, stride_u, stride_y,
        N,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        Gate_row_ptr = Gate_ptr + row_idx * stride_g
        Up_row_ptr = Up_ptr + row_idx * stride_u
        Y_row_ptr = Y_ptr + row_idx * stride_y

        for col_start in range(0, N, BLOCK_SIZE):
            col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
            mask = col_offsets < N

            gate = tl.load(Gate_row_ptr + col_offsets, mask=mask, other=0.0)
            up = tl.load(Up_row_ptr + col_offsets, mask=mask, other=0.0)

            gate_silu = gate * tl.sigmoid(gate)
            y = gate_silu * up

            tl.store(Y_row_ptr + col_offsets, y, mask=mask)


    class TritonSwiGLUMLP(nn.Module):
        def __init__(self, hidden_size: int, intermediate_size: int):
            super().__init__()
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        def forward(self, x):
            gate = self.gate_proj(x)
            up = self.up_proj(x)

            original_shape = gate.shape
            gate_flat = gate.view(-1, self.intermediate_size)
            up_flat = up.view(-1, self.intermediate_size)
            y = torch.empty_like(gate_flat)

            M, N = gate_flat.shape
            BLOCK_SIZE = min(triton.next_power_of_2(N), 8192)

            _swiglu_fwd_kernel[(M,)](
                gate_flat, up_flat, y,
                gate_flat.stride(0), up_flat.stride(0), y.stride(0),
                N,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            return self.down_proj(y.view(original_shape))


def run_rms_norm_benchmark(
    batch_sizes: List[int],
    seq_lens: List[int],
    hidden_sizes: List[int],
    dtype: torch.dtype = torch.bfloat16,
) -> List[BenchmarkResult]:
    """Benchmark RMSNorm implementations"""
    results = []

    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            for hidden_size in hidden_sizes:
                print(f"\nRMSNorm: batch={batch_size}, seq={seq_len}, hidden={hidden_size}")

                x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=dtype)

                # Baseline
                baseline = BaselineRMSNorm(hidden_size).cuda().to(dtype)
                mean_time, std_time = benchmark_fn(baseline, {'x': x})
                throughput = (batch_size * seq_len) / (mean_time / 1000)
                results.append(BenchmarkResult(
                    'RMSNorm', 'baseline', batch_size, seq_len, hidden_size,
                    mean_time, std_time, throughput
                ))
                print(f"  Baseline: {mean_time:.3f} +/- {std_time:.3f} ms")

                # Triton
                if TRITON_AVAILABLE:
                    triton_norm = TritonRMSNorm(hidden_size).cuda().to(dtype)
                    mean_time, std_time = benchmark_fn(triton_norm, {'x': x})
                    throughput = (batch_size * seq_len) / (mean_time / 1000)
                    results.append(BenchmarkResult(
                        'RMSNorm', 'triton', batch_size, seq_len, hidden_size,
                        mean_time, std_time, throughput
                    ))
                    print(f"  Triton:   {mean_time:.3f} +/- {std_time:.3f} ms")

                # Liger
                if LIGER_AVAILABLE:
                    liger_norm = LigerRMSNorm(hidden_size).cuda().to(dtype)
                    mean_time, std_time = benchmark_fn(liger_norm, {'x': x})
                    throughput = (batch_size * seq_len) / (mean_time / 1000)
                    results.append(BenchmarkResult(
                        'RMSNorm', 'liger', batch_size, seq_len, hidden_size,
                        mean_time, std_time, throughput
                    ))
                    print(f"  Liger:    {mean_time:.3f} +/- {std_time:.3f} ms")

    return results


def run_swiglu_mlp_benchmark(
    batch_sizes: List[int],
    seq_lens: List[int],
    hidden_sizes: List[int],
    dtype: torch.dtype = torch.bfloat16,
) -> List[BenchmarkResult]:
    """Benchmark SwiGLU MLP implementations"""
    results = []

    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            for hidden_size in hidden_sizes:
                intermediate_size = int(hidden_size * 8 / 3)
                intermediate_size = ((intermediate_size + 63) // 64) * 64
                print(f"\nSwiGLU MLP: batch={batch_size}, seq={seq_len}, hidden={hidden_size}")

                x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=dtype)

                # Baseline
                baseline = BaselineSwiGLUMLP(hidden_size, intermediate_size).cuda().to(dtype)
                mean_time, std_time = benchmark_fn(baseline, {'x': x})
                throughput = (batch_size * seq_len) / (mean_time / 1000)
                results.append(BenchmarkResult(
                    'SwiGLU_MLP', 'baseline', batch_size, seq_len, hidden_size,
                    mean_time, std_time, throughput
                ))
                print(f"  Baseline: {mean_time:.3f} +/- {std_time:.3f} ms")

                # Triton
                if TRITON_AVAILABLE:
                    triton_mlp = TritonSwiGLUMLP(hidden_size, intermediate_size).cuda().to(dtype)
                    mean_time, std_time = benchmark_fn(triton_mlp, {'x': x})
                    throughput = (batch_size * seq_len) / (mean_time / 1000)
                    results.append(BenchmarkResult(
                        'SwiGLU_MLP', 'triton', batch_size, seq_len, hidden_size,
                        mean_time, std_time, throughput
                    ))
                    print(f"  Triton:   {mean_time:.3f} +/- {std_time:.3f} ms")

                # Liger
                if LIGER_AVAILABLE:
                    class MLPConfig:
                        def __init__(self, h, i):
                            self.hidden_size = h
                            self.intermediate_size = i
                            self.hidden_act = "silu"

                    liger_mlp = LigerSwiGLUMLP(MLPConfig(hidden_size, intermediate_size)).cuda().to(dtype)
                    mean_time, std_time = benchmark_fn(liger_mlp, {'x': x})
                    throughput = (batch_size * seq_len) / (mean_time / 1000)
                    results.append(BenchmarkResult(
                        'SwiGLU_MLP', 'liger', batch_size, seq_len, hidden_size,
                        mean_time, std_time, throughput
                    ))
                    print(f"  Liger:    {mean_time:.3f} +/- {std_time:.3f} ms")

    return results


def run_rope_benchmark(
    batch_sizes: List[int],
    seq_lens: List[int],
    hidden_sizes: List[int],
    num_heads: int = 16,
    dtype: torch.dtype = torch.bfloat16,
) -> List[BenchmarkResult]:
    """Benchmark Rotary Position Embedding implementations"""
    results = []

    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            for hidden_size in hidden_sizes:
                head_dim = hidden_size // num_heads
                print(f"\nRoPE: batch={batch_size}, seq={seq_len}, hidden={hidden_size}")

                q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=dtype)
                k = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=dtype)

                # Generate cos/sin
                inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device='cuda').float() / head_dim))
                t = torch.arange(seq_len, device='cuda').float()
                freqs = torch.outer(t, inv_freq)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos().to(dtype)
                sin = emb.sin().to(dtype)

                # Baseline
                def baseline_wrapper(q=q, k=k, cos=cos, sin=sin):
                    return baseline_rotary_pos_emb(q, k, cos, sin)

                mean_time, std_time = benchmark_fn(baseline_wrapper, {})
                throughput = (batch_size * seq_len) / (mean_time / 1000)
                results.append(BenchmarkResult(
                    'RoPE', 'baseline', batch_size, seq_len, hidden_size,
                    mean_time, std_time, throughput
                ))
                print(f"  Baseline: {mean_time:.3f} +/- {std_time:.3f} ms")

                # Liger
                if LIGER_AVAILABLE:
                    def liger_wrapper(q=q, k=k, cos=cos, sin=sin):
                        return liger_rotary_pos_emb(q, k, cos, sin)

                    mean_time, std_time = benchmark_fn(liger_wrapper, {})
                    throughput = (batch_size * seq_len) / (mean_time / 1000)
                    results.append(BenchmarkResult(
                        'RoPE', 'liger', batch_size, seq_len, hidden_size,
                        mean_time, std_time, throughput
                    ))
                    print(f"  Liger:    {mean_time:.3f} +/- {std_time:.3f} ms")

    return results


def run_cross_entropy_benchmark(
    batch_sizes: List[int],
    seq_lens: List[int],
    hidden_sizes: List[int],
    vocab_size: int = 151936,
    dtype: torch.dtype = torch.bfloat16,
) -> List[BenchmarkResult]:
    """Benchmark Cross Entropy Loss implementations"""
    results = []

    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            for hidden_size in hidden_sizes:
                # Skip very large combinations to avoid OOM
                if batch_size * seq_len * vocab_size * 2 > 10 * 1024**3:
                    print(f"\nCE Loss: batch={batch_size}, seq={seq_len}, hidden={hidden_size} - SKIPPED (too large)")
                    continue

                print(f"\nCE Loss: batch={batch_size}, seq={seq_len}, hidden={hidden_size}")

                hidden_states = torch.randn(batch_size * seq_len, hidden_size, device='cuda', dtype=dtype, requires_grad=True)
                weight = torch.randn(vocab_size, hidden_size, device='cuda', dtype=dtype)
                labels = torch.randint(0, vocab_size, (batch_size * seq_len,), device='cuda')

                # Baseline
                def baseline_wrapper():
                    h = hidden_states.detach().clone().requires_grad_(True)
                    loss = baseline_cross_entropy(h, weight, labels)
                    loss.backward()
                    return loss

                try:
                    mean_time, std_time = benchmark_fn(baseline_wrapper, {}, warmup=5, iterations=20)
                    throughput = (batch_size * seq_len) / (mean_time / 1000)
                    results.append(BenchmarkResult(
                        'CrossEntropy', 'baseline', batch_size, seq_len, hidden_size,
                        mean_time, std_time, throughput
                    ))
                    print(f"  Baseline: {mean_time:.3f} +/- {std_time:.3f} ms")
                except torch.cuda.OutOfMemoryError:
                    print(f"  Baseline: OOM")

                # Liger Fused
                if LIGER_AVAILABLE:
                    loss_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)

                    def liger_wrapper():
                        h = hidden_states.detach().clone().requires_grad_(True)
                        loss = loss_fn(h, weight, labels)
                        loss.backward()
                        return loss

                    try:
                        mean_time, std_time = benchmark_fn(liger_wrapper, {}, warmup=5, iterations=20)
                        throughput = (batch_size * seq_len) / (mean_time / 1000)
                        results.append(BenchmarkResult(
                            'CrossEntropy', 'liger', batch_size, seq_len, hidden_size,
                            mean_time, std_time, throughput
                        ))
                        print(f"  Liger:    {mean_time:.3f} +/- {std_time:.3f} ms")
                    except torch.cuda.OutOfMemoryError:
                        print(f"  Liger:    OOM")

    return results


def print_results_table(results: List[BenchmarkResult]):
    """Print results in a formatted table"""
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 100)
    print(f"{'Kernel':<15} {'Type':<10} {'Batch':<6} {'Seq':<8} {'Hidden':<8} {'Time (ms)':<15} {'Throughput':<15}")
    print("-" * 100)

    for r in results:
        print(f"{r.name:<15} {r.kernel_type:<10} {r.batch_size:<6} {r.seq_len:<8} {r.hidden_size:<8} "
              f"{r.mean_time_ms:.3f}+/-{r.std_time_ms:.3f}  {r.throughput_toks_per_sec/1e6:.2f}M tok/s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark kernel implementations")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[1024, 2560, 4096])
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--kernel", type=str, default="all", choices=["all", "rmsnorm", "swiglu", "rope", "ce"])
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print(f"Running benchmarks with dtype={args.dtype}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Sequence lengths: {args.seq_lens}")
    print(f"Hidden sizes: {args.hidden_sizes}")
    print(f"Triton available: {TRITON_AVAILABLE}")
    print(f"Liger available: {LIGER_AVAILABLE}")

    warmup_gpu()

    all_results = []

    if args.kernel in ["all", "rmsnorm"]:
        print("\n" + "=" * 50)
        print("RMSNorm Benchmarks")
        print("=" * 50)
        all_results.extend(run_rms_norm_benchmark(
            args.batch_sizes, args.seq_lens, args.hidden_sizes, dtype
        ))

    if args.kernel in ["all", "swiglu"]:
        print("\n" + "=" * 50)
        print("SwiGLU MLP Benchmarks")
        print("=" * 50)
        all_results.extend(run_swiglu_mlp_benchmark(
            args.batch_sizes, args.seq_lens, args.hidden_sizes, dtype
        ))

    if args.kernel in ["all", "rope"]:
        print("\n" + "=" * 50)
        print("Rotary Position Embedding Benchmarks")
        print("=" * 50)
        all_results.extend(run_rope_benchmark(
            args.batch_sizes, args.seq_lens, args.hidden_sizes, dtype=dtype
        ))

    if args.kernel in ["all", "ce"]:
        print("\n" + "=" * 50)
        print("Cross Entropy Loss Benchmarks")
        print("=" * 50)
        all_results.extend(run_cross_entropy_benchmark(
            args.batch_sizes, [512, 2048], args.hidden_sizes, dtype=dtype  # Use smaller seq lens for CE
        ))

    print_results_table(all_results)


if __name__ == "__main__":
    main()
