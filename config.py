"""
Configuration for NSA + Optimizer Ablation Study
Training from scratch with different architectures, optimizers, and context lengths

Supports dynamic model sizing - specify any size like "0.5B", "1B", "2.5B", "100M"
"""
import math
import re
from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict, Any, Tuple
from enum import Enum


class AttentionType(str, Enum):
    DENSE = "dense"
    NSA = "native_sparse_attention"
    FSA = "flash_sparse_attention"


class MambaType(str, Enum):
    """Mamba architecture types for hybrid models"""
    NONE = "none"           # Pure transformer (no Mamba)
    MAMBA2 = "mamba2"       # Mamba-2 (SSD formulation)
    MAMBA3 = "mamba3"       # Used for DeltaNet


class OptimizerType(str, Enum):
    ADAMW = "adamw"
    ADAMW_4BIT = "adamw4bit"
    ADAMW_8BIT = "adamw8bit"
    SOAP = "soap"
    SOAP_4BIT = "soap4bit"
    SOAP_8BIT = "soap8bit"
    SHAMPOO = "shampoo"


# Keep ModelSize enum for backward compatibility
class ModelSize(str, Enum):
    """Legacy model sizes - prefer using string sizes like '1B', '2.5B'"""
    SMALL = "0.6B"
    MEDIUM = "4B"
    LARGE = "8B"
    XLARGE = "32B"


@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    name: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int = 151936  # Qwen vocab size
    max_position_embeddings: int = 32768
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    attention_type: AttentionType = AttentionType.DENSE

    # NSA specific configs
    nsa_block_size: int = 64
    nsa_window_size: int = 64
    nsa_num_selected_blocks: int = 16


def parse_model_size(size_str: str) -> int:
    """
    Parse model size string to number of parameters.

    Supports formats:
    - "1B", "1.5B", "0.8B" -> billions
    - "100M", "500M" -> millions
    - "1000000000" -> raw number

    Returns:
        Number of parameters as integer
    """
    size_str = size_str.strip().upper()

    # Try to match patterns like "1.5B", "100M", etc.
    match = re.match(r'^([\d.]+)\s*([BMK]?)$', size_str)
    if match:
        value = float(match.group(1))
        suffix = match.group(2)

        if suffix == 'B':
            return int(value * 1e9)
        elif suffix == 'M':
            return int(value * 1e6)
        elif suffix == 'K':
            return int(value * 1e3)
        else:
            return int(value)

    # Try raw number
    try:
        return int(float(size_str))
    except ValueError:
        raise ValueError(f"Cannot parse model size: {size_str}. Use formats like '1B', '500M', '1.5B'")


def _round_to_multiple(x: int, multiple: int) -> int:
    """Round x to nearest multiple"""
    return ((x + multiple // 2) // multiple) * multiple


def _round_up_to_multiple(x: int, multiple: int) -> int:
    """Round x up to nearest multiple"""
    return ((x + multiple - 1) // multiple) * multiple


def _nearest_power_of_2(x: int) -> int:
    """Find nearest power of 2 to x"""
    if x <= 0:
        return 1
    lower = 1 << (x.bit_length() - 1)
    upper = lower << 1
    if x - lower < upper - x:
        return lower
    return upper


# Valid power-of-2 hidden sizes for model dimensions
VALID_HIDDEN_SIZES = [256, 512, 1024, 2048, 4096, 8192, 16384]


def compute_transformer_params(
    hidden_size: int,
    num_layers: int,
    num_kv_heads: int = 8,
    vocab_size: int = 151936,
    intermediate_multiplier: float = 8/3,
) -> int:
    """
    Compute total parameters for a transformer model.

    Architecture assumptions:
    - GQA with fixed num_kv_heads
    - SwiGLU MLP with intermediate_size = hidden_size * intermediate_multiplier
    - head_dim = 64
    - Untied embeddings (input + output)
    """
    head_dim = 64
    num_heads = hidden_size // head_dim
    intermediate_size = _round_up_to_multiple(int(hidden_size * intermediate_multiplier), 64)

    # Embeddings (input + output, untied)
    embed_params = 2 * vocab_size * hidden_size

    # Per layer params
    # Attention: Q, K, V, O projections
    q_params = hidden_size * hidden_size  # Q: hidden -> num_heads * head_dim
    k_params = hidden_size * (num_kv_heads * head_dim)  # K: hidden -> num_kv_heads * head_dim
    v_params = hidden_size * (num_kv_heads * head_dim)  # V: hidden -> num_kv_heads * head_dim
    o_params = hidden_size * hidden_size  # O: num_heads * head_dim -> hidden

    # MLP: gate, up, down (SwiGLU)
    mlp_params = 3 * hidden_size * intermediate_size

    # Layer norms (2 per layer)
    ln_params = 2 * hidden_size

    layer_params = q_params + k_params + v_params + o_params + mlp_params + ln_params

    # Final layer norm
    final_ln_params = hidden_size

    total = embed_params + (num_layers * layer_params) + final_ln_params
    return total


def compute_model_dimensions(
    target_params: int,
    vocab_size: int = 151936,
    num_kv_heads: int = 8,
    head_dim: int = 64,
    intermediate_multiplier: float = 8/3,
    min_layers: int = 4,
    max_layers: int = 128,
) -> Tuple[int, int, int, int, int]:
    """
    Compute model dimensions to achieve target parameter count.

    Returns:
        (hidden_size, num_layers, num_heads, num_kv_heads, intermediate_size)

    Constraints:
    - hidden_size MUST be a power of 2 (for Triton kernel optimization)
    - num_heads = hidden_size / head_dim
    - intermediate_size = round(hidden_size * 8/3) to nearest power of 2
    """
    best_config = None
    best_diff = float('inf')

    # Determine layer range based on target size
    if target_params < 500_000_000:  # < 500M
        layer_range = range(min_layers, min(32, max_layers + 1), 2)
    elif target_params < 5_000_000_000:  # 500M - 5B
        layer_range = range(12, min(48, max_layers + 1), 2)
    else:  # > 5B
        layer_range = range(24, max_layers + 1, 4)

    # Only use power-of-2 hidden sizes for Triton optimization
    for hidden_size in VALID_HIDDEN_SIZES:
        for num_layers in layer_range:
            params = compute_transformer_params(
                hidden_size, num_layers, num_kv_heads, vocab_size, intermediate_multiplier
            )

            diff = abs(params - target_params)

            if diff < best_diff:
                best_diff = diff
                num_heads = hidden_size // head_dim
                # Round intermediate_size to nearest power of 2 for Triton
                raw_intermediate = int(hidden_size * intermediate_multiplier)
                intermediate_size = _nearest_power_of_2(raw_intermediate)
                best_config = (hidden_size, num_layers, num_heads, num_kv_heads, intermediate_size)

    if best_config is None:
        raise ValueError(f"Could not find valid configuration for {target_params} parameters")

    return best_config


def get_model_config_for_size(
    size_str: str,
    attention_type: AttentionType = AttentionType.DENSE,
    max_position_embeddings: int = 32768,
    vocab_size: int = 151936,
    nsa_block_size: int = 64,
    nsa_window_size: int = 64,
    nsa_num_selected_blocks: int = 16,
) -> ModelConfig:
    """
    Get model configuration for a given size string.

    Args:
        size_str: Model size like "1B", "2.5B", "500M"
        attention_type: Type of attention mechanism
        max_position_embeddings: Maximum sequence length
        vocab_size: Vocabulary size
        nsa_block_size: Block size for NSA
        nsa_window_size: Window size for NSA
        nsa_num_selected_blocks: Number of top-k blocks for NSA

    Returns:
        ModelConfig with computed dimensions
    """
    target_params = parse_model_size(size_str)

    hidden_size, num_layers, num_heads, num_kv_heads, intermediate_size = compute_model_dimensions(
        target_params, vocab_size=vocab_size
    )

    # Compute actual params for the name
    actual_params = compute_transformer_params(
        hidden_size, num_layers, num_kv_heads, vocab_size
    )

    # Format actual params for name
    if actual_params >= 1e9:
        param_str = f"{actual_params/1e9:.2f}B"
    else:
        param_str = f"{actual_params/1e6:.0f}M"

    return ModelConfig(
        name=f"transformer-{param_str}",
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        attention_type=attention_type,
        nsa_block_size=nsa_block_size,
        nsa_window_size=nsa_window_size,
        nsa_num_selected_blocks=nsa_num_selected_blocks,
    )


# Legacy MODEL_CONFIGS for backward compatibility
MODEL_CONFIGS: Dict[ModelSize, ModelConfig] = {
    ModelSize.SMALL: ModelConfig(
        name="transformer-0.6B",
        hidden_size=1024,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        intermediate_size=3072,
    ),
    ModelSize.MEDIUM: ModelConfig(
        name="transformer-4B",
        hidden_size=2560,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=9216,
    ),
    ModelSize.LARGE: ModelConfig(
        name="transformer-8B",
        hidden_size=4096,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=12288,
    ),
    ModelSize.XLARGE: ModelConfig(
        name="transformer-32B",
        hidden_size=5120,
        num_hidden_layers=64,
        num_attention_heads=40,
        num_key_value_heads=8,
        intermediate_size=25600,
    ),
}


@dataclass
class OptimizerConfig:
    """Configuration for optimizer"""
    optimizer_type: OptimizerType
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    # SOAP/Shampoo specific
    precondition_frequency: int = 5
    shampoo_beta: float = 0.95
    max_precond_dim: int = 1024

    # Low-bit optimizer specific
    use_4bit: bool = True
    use_8bit: bool = False

    # Shampoo optimizer state precision
    shampoo_state_dtype: str = "bfloat16"


@dataclass
class TrainingConfig:
    """Configuration for training"""
    # Model - now accepts string like "1B", "2.5B", "500M"
    model_size: str = "0.6B"
    attention_type: AttentionType = AttentionType.DENSE

    # Mamba/Jamba configuration
    mamba_type: MambaType = MambaType.NONE
    jamba_ratio: int = 7

    # Optimizer
    optimizer_type: OptimizerType = OptimizerType.ADAMW
    optimizer_config: Optional[OptimizerConfig] = None

    # Training params
    max_seq_length: int = 32768
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_steps: int = 100000
    warmup_steps: int = 2000
    max_grad_norm: float = 1.0

    # Learning rate schedule
    lr_scheduler_type: str = "cosine"
    min_lr_ratio: float = 0.1

    # Precision
    dtype: str = "bfloat16"
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True

    # Distributed
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1

    # Logging
    log_interval: int = 10
    eval_interval: int = 1000
    save_interval: int = 5000

    # Data
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_subset: str = "sample-10BT"

    # Paths
    output_dir: str = "./outputs"
    run_name: str = ""

    def __post_init__(self):
        # Convert ModelSize enum to string if needed
        if isinstance(self.model_size, ModelSize):
            self.model_size = self.model_size.value

        if self.optimizer_config is None:
            self.optimizer_config = OptimizerConfig(optimizer_type=self.optimizer_type)

        if not self.run_name:
            # Build architecture name based on mamba_type and attention_type
            if self.mamba_type != MambaType.NONE and self.attention_type != AttentionType.DENSE:
                arch_name = f"jamba_{self.mamba_type.value}_{self.attention_type.value}"
            elif self.mamba_type != MambaType.NONE:
                arch_name = self.mamba_type.value
            else:
                arch_name = self.attention_type.value
            self.run_name = f"{self.model_size}_{arch_name}_{self.optimizer_type.value}_ctx{self.max_seq_length}"


# Context length configurations
CONTEXT_LENGTHS = {
    "standard": [32768, 131072],
    "extended_nsa": [524288, 1048576],
}


def get_experiment_grid() -> List[TrainingConfig]:
    """Generate all experiment configurations"""
    experiments = []
    model_sizes = ["0.6B", "4B", "8B", "32B"]

    for model_size in model_sizes:
        for attention_type in AttentionType:
            for optimizer_type in OptimizerType:
                context_lengths = CONTEXT_LENGTHS["standard"].copy()
                if attention_type == AttentionType.NSA:
                    context_lengths.extend(CONTEXT_LENGTHS["extended_nsa"])

                for ctx_len in context_lengths:
                    batch_size, grad_accum = _get_batch_config(model_size, ctx_len)
                    train_steps = _get_training_steps(model_size, ctx_len)

                    config = TrainingConfig(
                        model_size=model_size,
                        attention_type=attention_type,
                        optimizer_type=optimizer_type,
                        max_seq_length=ctx_len,
                        batch_size=batch_size,
                        gradient_accumulation_steps=grad_accum,
                        num_train_steps=train_steps,
                    )
                    experiments.append(config)

    return experiments


def _get_batch_config(model_size: str, ctx_len: int) -> tuple:
    """Get batch size and gradient accumulation based on model and context"""
    params = parse_model_size(model_size)

    # Scale based on parameter count
    if params < 1e9:  # < 1B
        base = {32768: (4, 4), 131072: (1, 16), 524288: (1, 32), 1048576: (1, 64)}
    elif params < 5e9:  # 1B - 5B
        base = {32768: (2, 8), 131072: (1, 16), 524288: (1, 32), 1048576: (1, 64)}
    elif params < 15e9:  # 5B - 15B
        base = {32768: (1, 16), 131072: (1, 32), 524288: (1, 64), 1048576: (1, 128)}
    else:  # > 15B
        base = {32768: (1, 32), 131072: (1, 64), 524288: (1, 128), 1048576: (1, 256)}

    return base.get(ctx_len, (1, 16))


def _get_training_steps(model_size: str, ctx_len: int) -> int:
    """Adjust training steps to keep similar token count"""
    base_steps = 100000
    base_ctx = 32768
    return max(10000, int(base_steps * base_ctx / ctx_len))


def get_filtered_experiments(
    model_sizes: Optional[List[str]] = None,
    attention_types: Optional[List[AttentionType]] = None,
    optimizer_types: Optional[List[OptimizerType]] = None,
    context_lengths: Optional[List[int]] = None,
) -> List[TrainingConfig]:
    """Get filtered subset of experiments"""
    all_experiments = get_experiment_grid()

    filtered = []
    for exp in all_experiments:
        if model_sizes and exp.model_size not in model_sizes:
            continue
        if attention_types and exp.attention_type not in attention_types:
            continue
        if optimizer_types and exp.optimizer_type not in optimizer_types:
            continue
        if context_lengths and exp.max_seq_length not in context_lengths:
            continue
        filtered.append(exp)

    return filtered


def print_model_config(size_str: str):
    """Print model configuration for a given size (useful for debugging)"""
    config = get_model_config_for_size(size_str)
    actual_params = compute_transformer_params(
        config.hidden_size,
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.vocab_size,
    )

    print(f"Model Configuration for '{size_str}':")
    print(f"  Name: {config.name}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_hidden_layers}")
    print(f"  Num attention heads: {config.num_attention_heads}")
    print(f"  Num KV heads: {config.num_key_value_heads}")
    print(f"  Head dim: {config.hidden_size // config.num_attention_heads}")
    print(f"  Intermediate size: {config.intermediate_size}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Actual parameters: {actual_params:,} ({actual_params/1e9:.3f}B)")


if __name__ == "__main__":
    # Test various model sizes
    test_sizes = ["100M", "500M", "0.8B", "1B", "1.5B", "2B", "3B", "7B", "13B", "30B", "70B"]
    for size in test_sizes:
        print_model_config(size)
        print()
