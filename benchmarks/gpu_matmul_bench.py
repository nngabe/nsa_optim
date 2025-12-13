#!/usr/bin/env python3
import torch
import time

def benchmark_matmul(M, N, K, dtype=torch.float16, warmup=10, iterations=1000):
    """
    Benchmark matrix multiplication: (M, K) @ (K, N)
    Typical LLM pattern: [batch*seq_len, hidden] @ [hidden, hidden]
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)

    # Warmup
    for _ in range(warmup):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Calculate FLOPs: 2*M*N*K operations (multiply-add = 2 ops)
    flops = 2 * M * N * K * iterations
    tflops = flops / elapsed / 1e12

    return tflops, elapsed / iterations * 1000

if __name__ == "__main__":
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")

    # LLM-like sizes: [batch*seq, hidden] @ [hidden, hidden]
    configs = [
        (2048, 4096, 4096, "Small (Llama-7B linear)"),
        (4096, 8192, 8192, "Medium (Llama-13B linear)"),
        (8192, 8192, 8192, "Large (square 8K)"),
    ]

    for M, N, K, desc in configs:
        tflops_fp16, latency_fp16 = benchmark_matmul(M, N, K, torch.float16)
        print(f"{desc}")
        print(f"  Shape: ({M}, {K}) @ ({K}, {N})")
        print(f"  FP16: {tflops_fp16:.2f} TFLOPS | {latency_fp16:.3f} ms/iter")
        print()
