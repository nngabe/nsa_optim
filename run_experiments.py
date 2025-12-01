"""
Experiment runner for NSA + Optimizer ablation study

Generates and runs all experiments across:
- Model sizes: 0.6B, 4B, 8B, 32B
- Attention types: Dense, NSA
- Optimizers: AdamW, SOAP, Shampoo, SOAP-LowBit
- Context lengths: 32k, 128k (all), 512k, 1M (NSA only)
"""
import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from dataclasses import asdict

from config import (
    TrainingConfig,
    ModelSize,
    AttentionType,
    OptimizerType,
    get_experiment_grid,
    get_filtered_experiments,
)


def generate_experiment_name(config: TrainingConfig) -> str:
    """Generate unique experiment name"""
    return (
        f"{config.model_size.value}_"
        f"{config.attention_type.value}_"
        f"{config.optimizer_type.value}_"
        f"ctx{config.max_seq_length}"
    )


def estimate_resources(config: TrainingConfig) -> dict:
    """Estimate GPU resources needed for experiment"""
    # Base memory estimates (in GB per GPU)
    model_memory = {
        ModelSize.SMALL: 2,
        ModelSize.MEDIUM: 10,
        ModelSize.LARGE: 20,
        ModelSize.XLARGE: 80,
    }
    
    # Context length multiplier
    ctx_multiplier = config.max_seq_length / 32768
    
    # Attention type factor (NSA is more memory efficient for long sequences)
    attn_factor = 0.7 if config.attention_type == AttentionType.NSA else 1.0
    
    # Optimizer memory factor
    opt_factor = {
        OptimizerType.ADAMW: 1.0,
        OptimizerType.SOAP: 1.5,
        OptimizerType.SHAMPOO: 2.0,
        OptimizerType.SOAP_LOWBIT: 0.7,
    }[config.optimizer_type]
    
    base_mem = model_memory[config.model_size]
    total_mem = base_mem * (1 + ctx_multiplier * attn_factor * 0.5) * opt_factor
    
    # Estimate number of GPUs needed
    gpu_memory = 80  # A100 80GB
    num_gpus = max(1, int(total_mem / (gpu_memory * 0.8)))  # 80% utilization target
    
    # Round up to power of 2
    num_gpus = 2 ** (num_gpus - 1).bit_length()
    
    return {
        "estimated_memory_gb": total_mem,
        "recommended_num_gpus": num_gpus,
        "recommended_batch_size": config.batch_size,
        "recommended_grad_accum": config.gradient_accumulation_steps,
    }


def generate_slurm_script(
    config: TrainingConfig,
    output_dir: str,
    partition: str = "gpu",
    account: str = "default",
    time_limit: str = "48:00:00",
) -> str:
    """Generate SLURM job script"""
    resources = estimate_resources(config)
    exp_name = generate_experiment_name(config)
    
    script = f"""#!/bin/bash
#SBATCH --job-name={exp_name}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={resources['recommended_num_gpus']}
#SBATCH --gpus-per-node={resources['recommended_num_gpus']}
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time={time_limit}
#SBATCH --output={output_dir}/logs/{exp_name}_%j.out
#SBATCH --error={output_dir}/logs/{exp_name}_%j.err

# Load modules
module load cuda/12.1
module load anaconda/3

# Activate environment
conda activate nsa_ablation

# Set environment variables
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 {resources['recommended_num_gpus'] - 1})

# Run training
torchrun \\
    --nproc_per_node={resources['recommended_num_gpus']} \\
    --master_port=$(shuf -i 10000-65535 -n 1) \\
    train.py \\
    --model_size {config.model_size.value} \\
    --attention_type {config.attention_type.value} \\
    --optimizer_type {config.optimizer_type.value} \\
    --context_length {config.max_seq_length} \\
    --batch_size {config.batch_size} \\
    --gradient_accumulation_steps {config.gradient_accumulation_steps} \\
    --num_train_steps {config.num_train_steps} \\
    --warmup_steps {config.warmup_steps} \\
    --dtype {config.dtype} \\
    --gradient_checkpointing \\
    --output_dir {output_dir} \\
    --run_name {exp_name}
"""
    return script


def generate_bash_script(
    config: TrainingConfig,
    output_dir: str,
    num_gpus: int = 1,
) -> str:
    """Generate bash script for local execution"""
    exp_name = generate_experiment_name(config)
    
    if num_gpus > 1:
        run_cmd = f"""torchrun \\
    --nproc_per_node={num_gpus} \\
    --master_port=$(shuf -i 10000-65535 -n 1) \\
    train.py"""
    else:
        run_cmd = "python train.py"
    
    script = f"""#!/bin/bash
set -e

# Create output directory
mkdir -p {output_dir}/logs

# Experiment: {exp_name}
echo "Starting experiment: {exp_name}"
echo "Model: {config.model_size.value}"
echo "Attention: {config.attention_type.value}"
echo "Optimizer: {config.optimizer_type.value}"
echo "Context: {config.max_seq_length}"

{run_cmd} \\
    --model_size {config.model_size.value} \\
    --attention_type {config.attention_type.value} \\
    --optimizer_type {config.optimizer_type.value} \\
    --context_length {config.max_seq_length} \\
    --batch_size {config.batch_size} \\
    --gradient_accumulation_steps {config.gradient_accumulation_steps} \\
    --num_train_steps {config.num_train_steps} \\
    --warmup_steps {config.warmup_steps} \\
    --dtype {config.dtype} \\
    --gradient_checkpointing \\
    --output_dir {output_dir} \\
    --run_name {exp_name}

echo "Completed: {exp_name}"
"""
    return script


def generate_experiment_manifest(experiments: List[TrainingConfig], output_dir: str):
    """Generate experiment manifest JSON"""
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "total_experiments": len(experiments),
        "experiments": []
    }

    for config in experiments:
        exp_name = generate_experiment_name(config)
        resources = estimate_resources(config)

        manifest["experiments"].append({
            "name": exp_name,
            "model_size": config.model_size.value,
            "attention_type": config.attention_type.value,
            "optimizer_type": config.optimizer_type.value,
            "context_length": config.max_seq_length,
            "batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "num_train_steps": config.num_train_steps,
            "estimated_resources": resources,
        })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "experiment_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Generated manifest: {manifest_path}")
    return manifest


def run_experiments_locally(
    experiments: List[TrainingConfig],
    output_dir: str,
    num_gpus: int = 1,
    dry_run: bool = False,
):
    """Run experiments locally"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "scripts").mkdir(exist_ok=True)
    
    for i, config in enumerate(experiments):
        exp_name = generate_experiment_name(config)
        script = generate_bash_script(config, str(output_dir), num_gpus)
        
        script_path = output_dir / "scripts" / f"{exp_name}.sh"
        with open(script_path, "w") as f:
            f.write(script)
        os.chmod(script_path, 0o755)
        
        print(f"\n[{i+1}/{len(experiments)}] {exp_name}")
        print(f"  Script: {script_path}")
        
        if not dry_run:
            print(f"  Running...")
            result = subprocess.run(
                ["bash", str(script_path)],
                cwd=str(Path(__file__).parent),
            )
            if result.returncode != 0:
                print(f"  FAILED with code {result.returncode}")
            else:
                print(f"  SUCCESS")


def generate_slurm_jobs(
    experiments: List[TrainingConfig],
    output_dir: str,
    partition: str = "gpu",
    account: str = "default",
    time_limit: str = "48:00:00",
):
    """Generate all SLURM job scripts"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "slurm").mkdir(exist_ok=True)
    
    job_scripts = []
    
    for config in experiments:
        exp_name = generate_experiment_name(config)
        script = generate_slurm_script(
            config, str(output_dir), partition, account, time_limit
        )
        
        script_path = output_dir / "slurm" / f"{exp_name}.slurm"
        with open(script_path, "w") as f:
            f.write(script)
        
        job_scripts.append(str(script_path))
        print(f"Generated: {script_path}")
    
    # Generate submission script
    submit_script = f"""#!/bin/bash
# Submit all experiments

cd {output_dir}

for script in slurm/*.slurm; do
    echo "Submitting: $script"
    sbatch "$script"
    sleep 1  # Avoid overwhelming scheduler
done

echo "All jobs submitted!"
"""
    
    submit_path = output_dir / "submit_all.sh"
    with open(submit_path, "w") as f:
        f.write(submit_script)
    os.chmod(submit_path, 0o755)
    
    print(f"\nGenerated {len(job_scripts)} SLURM scripts")
    print(f"Submit all with: {submit_path}")


def print_experiment_summary(experiments: List[TrainingConfig]):
    """Print summary of experiments"""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    # Count by dimension
    by_model = {}
    by_attention = {}
    by_optimizer = {}
    by_context = {}
    
    for exp in experiments:
        by_model[exp.model_size.value] = by_model.get(exp.model_size.value, 0) + 1
        by_attention[exp.attention_type.value] = by_attention.get(exp.attention_type.value, 0) + 1
        by_optimizer[exp.optimizer_type.value] = by_optimizer.get(exp.optimizer_type.value, 0) + 1
        by_context[exp.max_seq_length] = by_context.get(exp.max_seq_length, 0) + 1
    
    print(f"\nTotal experiments: {len(experiments)}")
    
    print("\nBy model size:")
    for k, v in sorted(by_model.items()):
        print(f"  {k}: {v}")
    
    print("\nBy attention type:")
    for k, v in sorted(by_attention.items()):
        print(f"  {k}: {v}")
    
    print("\nBy optimizer:")
    for k, v in sorted(by_optimizer.items()):
        print(f"  {k}: {v}")
    
    print("\nBy context length:")
    for k, v in sorted(by_context.items()):
        print(f"  {k:,}: {v}")
    
    # Estimate total compute
    total_gpu_hours = 0
    for exp in experiments:
        resources = estimate_resources(exp)
        # Rough estimate: 1 hour per 10k steps on single GPU
        hours = (exp.num_train_steps / 10000) * resources["recommended_num_gpus"]
        total_gpu_hours += hours
    
    print(f"\nEstimated total GPU-hours: {total_gpu_hours:,.0f}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    
    parser.add_argument("--output_dir", type=str, default="./experiments",
                       help="Output directory for experiments")
    
    # Filtering options
    parser.add_argument("--model_sizes", nargs="+", type=str,
                       choices=["0.6B", "4B", "8B", "32B"],
                       help="Filter by model sizes")
    parser.add_argument("--attention_types", nargs="+", type=str,
                       choices=["dense", "native_sparse_attention"],
                       help="Filter by attention types")
    parser.add_argument("--optimizer_types", nargs="+", type=str,
                       choices=["adamw", "soap", "shampoo", "soap_lowbit"],
                       help="Filter by optimizer types")
    parser.add_argument("--context_lengths", nargs="+", type=int,
                       choices=[32768, 131072, 524288, 1048576],
                       help="Filter by context lengths")
    
    # Execution mode
    parser.add_argument("--mode", type=str, default="generate",
                       choices=["generate", "run", "slurm"],
                       help="Execution mode")
    parser.add_argument("--num_gpus", type=int, default=1,
                       help="Number of GPUs for local execution")
    parser.add_argument("--dry_run", action="store_true",
                       help="Generate scripts without running")
    
    # SLURM options
    parser.add_argument("--partition", type=str, default="gpu")
    parser.add_argument("--account", type=str, default="default")
    parser.add_argument("--time_limit", type=str, default="48:00:00")
    
    args = parser.parse_args()
    
    # Get experiments
    model_sizes = [ModelSize(s) for s in args.model_sizes] if args.model_sizes else None
    attention_types = [AttentionType(a) for a in args.attention_types] if args.attention_types else None
    optimizer_types = [OptimizerType(o) for o in args.optimizer_types] if args.optimizer_types else None
    
    experiments = get_filtered_experiments(
        model_sizes=model_sizes,
        attention_types=attention_types,
        optimizer_types=optimizer_types,
        context_lengths=args.context_lengths,
    )
    
    print_experiment_summary(experiments)
    
    # Generate manifest
    generate_experiment_manifest(experiments, args.output_dir)
    
    # Execute based on mode
    if args.mode == "generate":
        print("\nGenerated experiment manifest only")
        print("Use --mode=run or --mode=slurm to execute")
    
    elif args.mode == "run":
        run_experiments_locally(
            experiments,
            args.output_dir,
            args.num_gpus,
            args.dry_run,
        )
    
    elif args.mode == "slurm":
        generate_slurm_jobs(
            experiments,
            args.output_dir,
            args.partition,
            args.account,
            args.time_limit,
        )


if __name__ == "__main__":
    main()
