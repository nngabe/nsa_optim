"""Profile Mamba2Block vs Mamba3Block memory and throughput"""
import time
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn

from models.mamba import Mamba2Block, Mamba2Config, Mamba3Block, Mamba3Config


@dataclass
class ProfileResult:
    """Profiling results"""
    seq_len: int
    memory_allocated_gb: float
    memory_reserved_gb: float
    tokens_per_sec: float
    fwd_time_ms: float
    bwd_time_ms: float


def profile_block(
    block: nn.Module,
    seq_len: int,
    d_model: int = 1024,
    batch_size: int = 1,
    num_iters: int = 10,
    device: str = "cuda",
) -> ProfileResult:
    """
    Args:
        block: Mamba block to profile
        seq_len: Sequence length
        d_model: Model dimension
        batch_size: Batch size
        num_iters: Number of iterations
        device: Device
    Returns:
        ProfileResult
    """
    block = block.to(device)
    block.train()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    target = torch.randn(batch_size, seq_len, d_model, device=device)

    # Warmup
    for _ in range(3):
        output = block(x)
        y = output[0] if isinstance(output, tuple) else output
        loss = ((y - target) ** 2).mean()
        loss.backward()
        block.zero_grad()
        x.grad = None

    torch.cuda.synchronize()
    fwd_times = []
    bwd_times = []

    for _ in range(num_iters):
        start = time.perf_counter()
        output = block(x)
        y = output[0] if isinstance(output, tuple) else output
        torch.cuda.synchronize()
        fwd_time = time.perf_counter() - start
        fwd_times.append(fwd_time)

        loss = ((y - target) ** 2).mean()

        start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        bwd_time = time.perf_counter() - start
        bwd_times.append(bwd_time)

        block.zero_grad()
        x.grad = None

    mem_alloc = torch.cuda.max_memory_allocated() / 1e9
    mem_reserved = torch.cuda.max_memory_reserved() / 1e9

    fwd_mean = sum(fwd_times) / len(fwd_times) * 1000
    bwd_mean = sum(bwd_times) / len(bwd_times) * 1000
    total_time = (fwd_mean + bwd_mean) / 1000

    tokens_per_sec = (batch_size * seq_len) / total_time

    return ProfileResult(
        seq_len=seq_len,
        memory_allocated_gb=mem_alloc,
        memory_reserved_gb=mem_reserved,
        tokens_per_sec=tokens_per_sec,
        fwd_time_ms=fwd_mean,
        bwd_time_ms=bwd_mean,
    )


def run_profiling(
    seq_lengths: List[int] = [8192, 32768, 131072],
    d_model: int = 1024,
    device: str = "cuda",
) -> Dict[str, List[ProfileResult]]:
    """
    Args:
        seq_lengths: Sequence lengths to test
        d_model: Model dimension
        device: Device
    Returns:
        Dict mapping model name to results
    """
    config2 = Mamba2Config(
        d_model=d_model,
        n_layers=1,
        d_state=128,
        expand=2,
        headdim=64,
    )

    config3 = Mamba3Config(
        d_model=d_model,
        n_layers=1,
        d_state=128,
        expand=2,
        headdim=64,
        use_complex=True,
    )

    results = {"Mamba2": [], "Mamba3": []}

    for seq_len in seq_lengths:
        print(f"\n{'='*60}")
        print(f"Profiling seq_len={seq_len}")
        print(f"{'='*60}")

        # Mamba2
        print(f"\nMamba2Block...")
        torch.cuda.empty_cache()
        block2 = Mamba2Block(config2, layer_idx=0)
        result2 = profile_block(block2, seq_len, d_model, device=device)
        results["Mamba2"].append(result2)

        print(f"  Memory: {result2.memory_allocated_gb:.2f}GB allocated, "
              f"{result2.memory_reserved_gb:.2f}GB reserved")
        print(f"  Throughput: {result2.tokens_per_sec:.0f} tokens/sec")
        print(f"  Times: fwd={result2.fwd_time_ms:.1f}ms, bwd={result2.bwd_time_ms:.1f}ms")

        del block2
        torch.cuda.empty_cache()

        # Mamba3
        print(f"\nMamba3Block...")
        torch.cuda.empty_cache()
        block3 = Mamba3Block(
            d_model=d_model,
            d_state=128,
            expand=2,
            headdim=64,
            use_complex=True,
        )
        result3 = profile_block(block3, seq_len, d_model, device=device)
        results["Mamba3"].append(result3)

        print(f"  Memory: {result3.memory_allocated_gb:.2f}GB allocated, "
              f"{result3.memory_reserved_gb:.2f}GB reserved")
        print(f"  Throughput: {result3.tokens_per_sec:.0f} tokens/sec")
        print(f"  Times: fwd={result3.fwd_time_ms:.1f}ms, bwd={result3.bwd_time_ms:.1f}ms")

        speedup = result3.tokens_per_sec / result2.tokens_per_sec
        mem_ratio = result3.memory_allocated_gb / result2.memory_allocated_gb

        print(f"\n  Mamba3 vs Mamba2:")
        print(f"    Throughput: {speedup:.2f}x")
        print(f"    Memory: {mem_ratio:.2f}x")

        del block3
        torch.cuda.empty_cache()

        if result2.memory_allocated_gb > 30 or result3.memory_allocated_gb > 30:
            print(f"\n  WARNING: Memory exceeded 30GB limit!")

    return results


def print_summary(results: Dict[str, List[ProfileResult]]):
    """Print summary table"""
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<12} {'SeqLen':<10} {'Mem(GB)':<10} {'Tok/s':<12} {'Fwd(ms)':<10} {'Bwd(ms)':<10}")
    print(f"{'-'*80}")

    for model_name, model_results in results.items():
        for res in model_results:
            print(f"{model_name:<12} {res.seq_len:<10} "
                  f"{res.memory_allocated_gb:<10.2f} "
                  f"{res.tokens_per_sec:<12.0f} "
                  f"{res.fwd_time_ms:<10.1f} "
                  f"{res.bwd_time_ms:<10.1f}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available")
        exit(1)

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    results = run_profiling()
    print_summary(results)
