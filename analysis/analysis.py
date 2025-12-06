"""
Analysis and evaluation utilities for ablation study results

Provides:
- Result aggregation across experiments
- Statistical comparison of methods
- Visualization of training curves
- Final model evaluation
"""
import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math

import torch
import numpy as np
from collections import defaultdict


@dataclass
class ExperimentResult:
    """Container for experiment results"""
    name: str
    model_size: str
    attention_type: str
    optimizer_type: str
    context_length: int
    final_loss: float
    final_perplexity: float
    tokens_per_second: float
    total_tokens: int
    training_time_hours: float
    peak_memory_gb: float
    convergence_step: Optional[int] = None  # Step where loss plateaued


def load_experiment_results(output_dir: str) -> List[ExperimentResult]:
    """Load results from all experiments in output directory"""
    results = []
    output_path = Path(output_dir)
    
    for exp_dir in output_path.iterdir():
        if not exp_dir.is_dir():
            continue
        
        # Find latest checkpoint
        checkpoints = sorted(exp_dir.glob("checkpoint-*"))
        if not checkpoints:
            continue
        
        latest_checkpoint = checkpoints[-1]
        config_path = latest_checkpoint / "config.json"
        
        if not config_path.exists():
            continue
        
        with open(config_path) as f:
            config = json.load(f)
        
        # Load training metrics if available
        metrics_path = exp_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
        else:
            metrics = {}
        
        result = ExperimentResult(
            name=exp_dir.name,
            model_size=config.get("model_size", "unknown"),
            attention_type=config.get("attention_type", "unknown"),
            optimizer_type=config.get("optimizer_type", "unknown"),
            context_length=config.get("max_seq_length", 0),
            final_loss=metrics.get("final_loss", float("nan")),
            final_perplexity=metrics.get("final_perplexity", float("nan")),
            tokens_per_second=metrics.get("tokens_per_second", 0),
            total_tokens=metrics.get("total_tokens", 0),
            training_time_hours=metrics.get("training_time_hours", 0),
            peak_memory_gb=metrics.get("peak_memory_gb", 0),
        )
        results.append(result)
    
    return results


def aggregate_by_dimension(
    results: List[ExperimentResult],
    dimension: str,
) -> Dict[str, List[ExperimentResult]]:
    """Group results by a specific dimension"""
    grouped = defaultdict(list)
    
    for result in results:
        key = getattr(result, dimension)
        grouped[key].append(result)
    
    return dict(grouped)


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute statistics for a list of values"""
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    
    return {
        "mean": np.mean(values),
        "std": np.std(values),
        "min": np.min(values),
        "max": np.max(values),
    }


def compare_attention_types(results: List[ExperimentResult]) -> Dict:
    """Compare Dense vs NSA attention"""
    by_attention = aggregate_by_dimension(results, "attention_type")
    
    comparison = {}
    for attn_type, attn_results in by_attention.items():
        losses = [r.final_loss for r in attn_results]
        throughputs = [r.tokens_per_second for r in attn_results]
        memories = [r.peak_memory_gb for r in attn_results]
        
        comparison[attn_type] = {
            "count": len(attn_results),
            "loss": compute_statistics(losses),
            "throughput": compute_statistics(throughputs),
            "memory": compute_statistics(memories),
        }
    
    return comparison


def compare_optimizers(results: List[ExperimentResult]) -> Dict:
    """Compare different optimizers"""
    by_optimizer = aggregate_by_dimension(results, "optimizer_type")
    
    comparison = {}
    for opt_type, opt_results in by_optimizer.items():
        losses = [r.final_loss for r in opt_results]
        times = [r.training_time_hours for r in opt_results]
        
        comparison[opt_type] = {
            "count": len(opt_results),
            "loss": compute_statistics(losses),
            "training_time": compute_statistics(times),
        }
    
    return comparison


def compare_context_lengths(results: List[ExperimentResult]) -> Dict:
    """Compare different context lengths"""
    by_context = aggregate_by_dimension(results, "context_length")
    
    comparison = {}
    for ctx_len, ctx_results in sorted(by_context.items()):
        losses = [r.final_loss for r in ctx_results]
        memories = [r.peak_memory_gb for r in ctx_results]
        throughputs = [r.tokens_per_second for r in ctx_results]
        
        comparison[ctx_len] = {
            "count": len(ctx_results),
            "loss": compute_statistics(losses),
            "memory": compute_statistics(memories),
            "throughput": compute_statistics(throughputs),
        }
    
    return comparison


def generate_report(results: List[ExperimentResult], output_path: str):
    """Generate comprehensive comparison report"""
    report = {
        "summary": {
            "total_experiments": len(results),
            "completed": len([r for r in results if not math.isnan(r.final_loss)]),
        },
        "attention_comparison": compare_attention_types(results),
        "optimizer_comparison": compare_optimizers(results),
        "context_length_comparison": compare_context_lengths(results),
        "detailed_results": [
            {
                "name": r.name,
                "model_size": r.model_size,
                "attention_type": r.attention_type,
                "optimizer_type": r.optimizer_type,
                "context_length": r.context_length,
                "final_loss": r.final_loss,
                "final_perplexity": r.final_perplexity,
                "tokens_per_second": r.tokens_per_second,
                "peak_memory_gb": r.peak_memory_gb,
            }
            for r in sorted(results, key=lambda x: x.final_loss if not math.isnan(x.final_loss) else float("inf"))
        ],
    }
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved to {output_path}")
    return report


def print_summary_table(results: List[ExperimentResult]):
    """Print summary table of results"""
    print("\n" + "="*120)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*120)
    
    # Header
    print(f"{'Model':<8} {'Attention':<15} {'Optimizer':<12} {'Context':<10} {'Loss':<10} {'PPL':<12} {'Tok/s':<12} {'Memory':<10}")
    print("-"*120)
    
    # Sort by loss
    sorted_results = sorted(results, key=lambda x: x.final_loss if not math.isnan(x.final_loss) else float("inf"))
    
    for r in sorted_results:
        loss_str = f"{r.final_loss:.4f}" if not math.isnan(r.final_loss) else "N/A"
        ppl_str = f"{r.final_perplexity:.2f}" if not math.isnan(r.final_perplexity) else "N/A"
        
        print(f"{r.model_size:<8} {r.attention_type:<15} {r.optimizer_type:<12} {r.context_length:<10} "
              f"{loss_str:<10} {ppl_str:<12} {r.tokens_per_second:<12.0f} {r.peak_memory_gb:<10.1f}")
    
    print("="*120)
    
    # Best results by category
    print("\nBEST RESULTS BY CATEGORY:")
    print("-"*60)
    
    # Best by attention type
    print("\nBy Attention Type:")
    for attn_type in set(r.attention_type for r in results):
        attn_results = [r for r in results if r.attention_type == attn_type and not math.isnan(r.final_loss)]
        if attn_results:
            best = min(attn_results, key=lambda x: x.final_loss)
            print(f"  {attn_type}: {best.name} (loss={best.final_loss:.4f})")
    
    # Best by optimizer
    print("\nBy Optimizer:")
    for opt_type in set(r.optimizer_type for r in results):
        opt_results = [r for r in results if r.optimizer_type == opt_type and not math.isnan(r.final_loss)]
        if opt_results:
            best = min(opt_results, key=lambda x: x.final_loss)
            print(f"  {opt_type}: {best.name} (loss={best.final_loss:.4f})")
    
    # Best by model size
    print("\nBy Model Size:")
    for model_size in ["0.6B", "4B", "8B", "32B"]:
        size_results = [r for r in results if r.model_size == model_size and not math.isnan(r.final_loss)]
        if size_results:
            best = min(size_results, key=lambda x: x.final_loss)
            print(f"  {model_size}: {best.name} (loss={best.final_loss:.4f})")


def plot_training_curves(
    log_dir: str,
    output_path: str,
    metric: str = "loss",
):
    """Plot training curves for all experiments"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return
    
    log_path = Path(log_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Group by dimensions
    curves_by_attention = defaultdict(list)
    curves_by_optimizer = defaultdict(list)
    curves_by_context = defaultdict(list)
    curves_by_size = defaultdict(list)
    
    for exp_dir in log_path.iterdir():
        if not exp_dir.is_dir():
            continue
        
        # Parse experiment name
        parts = exp_dir.name.split("_")
        if len(parts) < 4:
            continue
        
        model_size = parts[0]
        attention = parts[1]
        optimizer = parts[2]
        context = parts[3].replace("ctx", "")
        
        # Load training log
        log_file = exp_dir / "training_log.jsonl"
        if not log_file.exists():
            continue
        
        steps, values = [], []
        with open(log_file) as f:
            for line in f:
                entry = json.loads(line)
                if metric in entry:
                    steps.append(entry["step"])
                    values.append(entry[metric])
        
        if not steps:
            continue
        
        curve = (steps, values, exp_dir.name)
        curves_by_attention[attention].append(curve)
        curves_by_optimizer[optimizer].append(curve)
        curves_by_context[context].append(curve)
        curves_by_size[model_size].append(curve)
    
    # Plot by attention type
    ax = axes[0, 0]
    for attn_type, curves in curves_by_attention.items():
        for steps, values, name in curves[:5]:  # Limit to 5 curves per type
            ax.plot(steps, values, alpha=0.7, label=name[:20])
    ax.set_xlabel("Step")
    ax.set_ylabel(metric.capitalize())
    ax.set_title("By Attention Type")
    ax.legend(fontsize=6)
    
    # Plot by optimizer
    ax = axes[0, 1]
    for opt_type, curves in curves_by_optimizer.items():
        for steps, values, name in curves[:5]:
            ax.plot(steps, values, alpha=0.7, label=name[:20])
    ax.set_xlabel("Step")
    ax.set_ylabel(metric.capitalize())
    ax.set_title("By Optimizer")
    ax.legend(fontsize=6)
    
    # Plot by context length
    ax = axes[1, 0]
    for ctx_len, curves in sorted(curves_by_context.items()):
        for steps, values, name in curves[:5]:
            ax.plot(steps, values, alpha=0.7, label=name[:20])
    ax.set_xlabel("Step")
    ax.set_ylabel(metric.capitalize())
    ax.set_title("By Context Length")
    ax.legend(fontsize=6)
    
    # Plot by model size
    ax = axes[1, 1]
    for size, curves in curves_by_size.items():
        for steps, values, name in curves[:5]:
            ax.plot(steps, values, alpha=0.7, label=name[:20])
    ax.set_xlabel("Step")
    ax.set_ylabel(metric.capitalize())
    ax.set_title("By Model Size")
    ax.legend(fontsize=6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")


def evaluate_model(
    checkpoint_path: str,
    eval_dataset: str = "wikitext",
    max_samples: int = 1000,
) -> Dict[str, float]:
    """Evaluate a trained model checkpoint"""
    from transformers import AutoTokenizer
    from datasets import load_dataset
    from model import TransformerModel
    from config import ModelConfig, MODEL_CONFIGS
    
    checkpoint_path = Path(checkpoint_path)
    config_path = checkpoint_path / "config.json"
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Load model
    model_size = config["model_size"]
    attention_type = config["attention_type"]
    
    from config import ModelSize, AttentionType
    model_config = MODEL_CONFIGS[ModelSize(model_size)]
    model_config.attention_type = AttentionType(attention_type)
    
    model = TransformerModel(model_config)
    model.load_state_dict(torch.load(checkpoint_path / "model.pt"))
    model.eval()
    model.cuda()
    
    # Load tokenizer and data
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    
    if eval_dataset == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    else:
        dataset = load_dataset(eval_dataset, split="test")
    
    # Compute perplexity
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i, example in enumerate(dataset):
            if i >= max_samples:
                break
            
            text = example.get("text", "")
            if not text:
                continue
            
            tokens = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=2048)
            tokens = tokens.cuda()
            
            if tokens.shape[1] < 2:
                continue
            
            _, loss, _ = model(input_ids=tokens, labels=tokens)
            
            total_loss += loss.item() * (tokens.shape[1] - 1)
            total_tokens += tokens.shape[1] - 1
    
    avg_loss = total_loss / max(1, total_tokens)
    perplexity = math.exp(avg_loss)
    
    return {
        "eval_loss": avg_loss,
        "eval_perplexity": perplexity,
        "num_tokens": total_tokens,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--output_dir", type=str, default="./experiments",
                       help="Directory containing experiment outputs")
    parser.add_argument("--report_path", type=str, default="./analysis_report.json",
                       help="Path to save analysis report")
    parser.add_argument("--plot", action="store_true",
                       help="Generate training curve plots")
    
    args = parser.parse_args()
    
    results = load_experiment_results(args.output_dir)
    
    if results:
        print_summary_table(results)
        generate_report(results, args.report_path)
        
        if args.plot:
            plot_training_curves(args.output_dir, "training_curves.png")
    else:
        print("No results found in", args.output_dir)
