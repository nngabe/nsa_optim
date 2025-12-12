#!/usr/bin/env python3
import subprocess
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class TrainingMetrics:
    kernel_type: str
    avg_tokens_per_sec: float
    avg_step_time_ms: float
    final_loss: float
    peak_memory_mb: float
    steps_completed: int
    total_time_sec: float


def parse_training_output(output: str) -> TrainingMetrics:
    """Parse training script output to extract metrics."""
    lines = output.split("\n")

    tokens_per_sec = []
    step_times = []
    losses = []
    memory_mbs = []
    kernel_type = "unknown"

    for line in lines:
        # Extract kernel type
        if "kernel_type:" in line.lower() or "kernel type:" in line.lower():
            match = re.search(r"kernel[_ ]type:\s*(\w+)", line, re.IGNORECASE)
            if match:
                kernel_type = match.group(1)

        # Extract tokens/sec
        if "tok/s" in line or "tokens/sec" in line:
            match = re.search(r"([\d.]+)\s*(?:tok/s|tokens/sec)", line)
            if match:
                tokens_per_sec.append(float(match.group(1)))

        # Extract step time
        if "step time" in line.lower() or "ms/step" in line:
            match = re.search(r"([\d.]+)\s*ms", line)
            if match:
                step_times.append(float(match.group(1)))

        # Extract loss
        if "loss:" in line.lower():
            match = re.search(r"loss:\s*([\d.]+)", line, re.IGNORECASE)
            if match:
                losses.append(float(match.group(1)))

        # Extract memory
        if "memory" in line.lower() and ("mb" in line.lower() or "mib" in line.lower()):
            match = re.search(r"([\d.]+)\s*(?:MB|MiB)", line, re.IGNORECASE)
            if match:
                memory_mbs.append(float(match.group(1)))

    avg_tokens = sum(tokens_per_sec) / len(tokens_per_sec) if tokens_per_sec else 0.0
    avg_step_time = sum(step_times) / len(step_times) if step_times else 0.0
    final_loss = losses[-1] if losses else 0.0
    peak_memory = max(memory_mbs) if memory_mbs else 0.0
    steps = len(losses)
    total_time = sum(step_times) / 1000 if step_times else 0.0

    return TrainingMetrics(
        kernel_type=kernel_type,
        avg_tokens_per_sec=avg_tokens,
        avg_step_time_ms=avg_step_time,
        final_loss=final_loss,
        peak_memory_mb=peak_memory,
        steps_completed=steps,
        total_time_sec=total_time,
    )


def run_training(kernel_type: str, steps: int = 50) -> tuple[TrainingMetrics, str]:
    """Run training with specified kernel type."""
    cmd = [
        "python", "train.py",
        "--model_size", "0.5B",
        "--block_pattern", "MDMA",
        "--attn_type", "nsa",
        "--optimizer_type", "adamw8bit",
        "--kernel_type", kernel_type,
        "--num_train_steps", str(steps),
        "--batch_size", "4",
        "--context_length", "8192",
        "--log_interval", "5",
        "--save_interval", "999999",  # Disable saves
    ]

    print(f"\n{'='*80}")
    print(f"Running training with kernel_type={kernel_type}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd="/root/nsa_optim",
        )

        output = result.stdout + result.stderr

        if result.returncode != 0:
            print(f"WARNING: Training exited with code {result.returncode}")
            print("STDERR:", result.stderr[-1000:])

        metrics = parse_training_output(output)
        metrics.kernel_type = kernel_type

        return metrics, output

    except subprocess.TimeoutExpired:
        print(f"ERROR: Training timed out after 600 seconds")
        return None, ""
    except Exception as e:
        print(f"ERROR: {e}")
        return None, ""


def save_results(metrics_list: list[TrainingMetrics], output_dir: Path = Path("./kernel_validation")):
    """Save results to JSON and create summary."""
    output_dir.mkdir(exist_ok=True)

    # Save raw metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump([asdict(m) for m in metrics_list], f, indent=2)

    # Create summary
    with open(output_dir / "summary.txt", "w") as f:
        f.write("="*100 + "\n")
        f.write("KERNEL VALIDATION SUMMARY\n")
        f.write("="*100 + "\n\n")

        f.write(f"{'Kernel':<12} {'Tok/s':<12} {'Step(ms)':<12} {'Loss':<12} {'Mem(MB)':<12} {'Steps':<8}\n")
        f.write("-"*100 + "\n")

        for m in metrics_list:
            f.write(
                f"{m.kernel_type:<12} "
                f"{m.avg_tokens_per_sec:<12.1f} "
                f"{m.avg_step_time_ms:<12.2f} "
                f"{m.final_loss:<12.4f} "
                f"{m.peak_memory_mb:<12.1f} "
                f"{m.steps_completed:<8}\n"
            )

        # Speedup analysis
        baseline = next((m for m in metrics_list if m.kernel_type == "baseline"), None)
        if baseline and baseline.avg_tokens_per_sec > 0:
            f.write("\n" + "="*100 + "\n")
            f.write("SPEEDUP ANALYSIS (vs Baseline)\n")
            f.write("="*100 + "\n\n")

            for m in metrics_list:
                if m.kernel_type != "baseline":
                    speedup = m.avg_tokens_per_sec / baseline.avg_tokens_per_sec if baseline.avg_tokens_per_sec > 0 else 0.0
                    mem_ratio = m.peak_memory_mb / baseline.peak_memory_mb if baseline.peak_memory_mb > 0 else 0.0
                    loss_diff = abs(m.final_loss - baseline.final_loss)

                    f.write(f"{m.kernel_type}:\n")
                    f.write(f"  Throughput: {speedup:.2f}x\n")
                    f.write(f"  Memory: {mem_ratio:.2f}x\n")
                    f.write(f"  Loss diff: {loss_diff:.6f}\n\n")

        # Consistency check
        if len(metrics_list) > 1:
            f.write("\n" + "="*100 + "\n")
            f.write("CONSISTENCY CHECK\n")
            f.write("="*100 + "\n\n")

            losses = [m.final_loss for m in metrics_list]
            loss_std = (sum((l - sum(losses)/len(losses))**2 for l in losses) / len(losses)) ** 0.5

            f.write(f"Loss std dev: {loss_std:.6f}\n")
            if loss_std < 0.01:
                f.write("✓ All kernels converged consistently\n")
            else:
                f.write("⚠ Loss divergence detected - check implementation\n")

    print(f"\nResults saved to {output_dir}")


def main():
    kernel_types = ["baseline", "triton", "liger"]
    steps = 50
    metrics_list = []

    for kernel in kernel_types:
        metrics, output = run_training(kernel, steps)

        if metrics:
            metrics_list.append(metrics)
            print(f"\n✓ {kernel} completed:")
            print(f"  Tokens/sec: {metrics.avg_tokens_per_sec:.1f}")
            print(f"  Step time: {metrics.avg_step_time_ms:.2f}ms")
            print(f"  Final loss: {metrics.final_loss:.4f}")
            print(f"  Peak memory: {metrics.peak_memory_mb:.1f} MB")
            print(f"  Steps: {metrics.steps_completed}")

            # Save individual log
            log_file = Path(f"./kernel_validation/{kernel}_output.log")
            log_file.parent.mkdir(exist_ok=True)
            log_file.write_text(output)
        else:
            print(f"\n✗ {kernel} failed")

    if metrics_list:
        save_results(metrics_list)

        # Print summary to stdout
        print("\n" + "="*100)
        print("SUMMARY")
        print("="*100)
        print(f"{'Kernel':<12} {'Tok/s':<12} {'Step(ms)':<12} {'Loss':<12} {'Mem(MB)':<12}")
        print("-"*100)
        for m in metrics_list:
            print(
                f"{m.kernel_type:<12} "
                f"{m.avg_tokens_per_sec:<12.1f} "
                f"{m.avg_step_time_ms:<12.2f} "
                f"{m.final_loss:<12.4f} "
                f"{m.peak_memory_mb:<12.1f}"
            )
        print("="*100)
    else:
        print("\nNo successful runs to summarize")
        sys.exit(1)


if __name__ == "__main__":
    main()
