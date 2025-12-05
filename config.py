"""
Configuration for NSA + Optimizer Ablation Study
Training from scratch with different architectures, optimizers, and context lengths
"""
from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict, Any
from enum import Enum


class AttentionType(str, Enum):
    DENSE = "dense"
    NSA = "native_sparse_attention"
    FSA = "flash_sparse_attention"


class OptimizerType(str, Enum):
    ADAMW = "adamw"
    ADAMW_4BIT = "adamw4bit"  # 4-bit AdamW from lpmm
    ADAMW_8BIT = "adamw8bit"   # 8-bit AdamW from torchao/lpmm
    SOAP = "soap"
    SOAP_4BIT = "soap4bit"     # SOAP with 4-bit states
    SOAP_8BIT = "soap8bit"     # SOAP with 8-bit states
    SHAMPOO = "shampoo"


class ModelSize(str, Enum):
    """Model sizes based on Qwen-3 architecture"""
    SMALL = "0.6B"    # Qwen3-0.6B
    MEDIUM = "4B"     # Qwen3-4B
    LARGE = "8B"      # Qwen3-8B
    XLARGE = "32B"    # Qwen3-32B


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


# Qwen-3 architecture configurations
MODEL_CONFIGS: Dict[ModelSize, ModelConfig] = {
    ModelSize.SMALL: ModelConfig(
        name="qwen3-0.6B",
        hidden_size=1024,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,  # GQA
        intermediate_size=3072,
    ),
    ModelSize.MEDIUM: ModelConfig(
        name="qwen3-4B",
        hidden_size=2560,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=9216,
    ),
    ModelSize.LARGE: ModelConfig(
        name="qwen3-8B",
        hidden_size=4096,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=12288,
    ),
    ModelSize.XLARGE: ModelConfig(
        name="qwen3-32B",
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
    precondition_frequency: int = 10
    shampoo_beta: float = 0.95
    max_precond_dim: int = 1024  # Reduced from 8192 to save memory (8x less memory per factor)

    # Low-bit optimizer specific (for SOAP_LOWBIT)
    use_4bit: bool = True
    use_8bit: bool = False

    # Shampoo optimizer state precision (for distributed_shampoo)
    # Options: "float32", "bfloat16", "float16"
    # Using bfloat16 can reduce memory by 2x with minimal accuracy loss
    shampoo_state_dtype: str = "bfloat16"


@dataclass
class TrainingConfig:
    """Configuration for training"""
    # Model
    model_size: ModelSize = ModelSize.SMALL
    attention_type: AttentionType = AttentionType.DENSE
    
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
        if self.optimizer_config is None:
            self.optimizer_config = OptimizerConfig(optimizer_type=self.optimizer_type)
        
        if not self.run_name:
            self.run_name = f"{self.model_size.value}_{self.attention_type.value}_{self.optimizer_type.value}_ctx{self.max_seq_length}"


# Context length configurations
CONTEXT_LENGTHS = {
    "standard": [32768, 131072],  # 32k, 128k for all models
    "extended_nsa": [524288, 1048576],  # 512k, 1M for NSA only
}


def get_experiment_grid() -> List[TrainingConfig]:
    """Generate all experiment configurations"""
    experiments = []
    
    for model_size in ModelSize:
        for attention_type in AttentionType:
            for optimizer_type in OptimizerType:
                # Determine context lengths based on attention type
                context_lengths = CONTEXT_LENGTHS["standard"].copy()
                if attention_type == AttentionType.NSA:
                    context_lengths.extend(CONTEXT_LENGTHS["extended_nsa"])
                
                for ctx_len in context_lengths:
                    # Adjust batch size based on model size and context length
                    batch_size, grad_accum = _get_batch_config(model_size, ctx_len)
                    
                    # Adjust training steps for fair comparison
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


def _get_batch_config(model_size: ModelSize, ctx_len: int) -> tuple:
    """Get batch size and gradient accumulation based on model and context"""
    # Base configurations (assuming 8x80GB GPUs)
    configs = {
        ModelSize.SMALL: {32768: (4, 4), 131072: (1, 16), 524288: (1, 32), 1048576: (1, 64)},
        ModelSize.MEDIUM: {32768: (2, 8), 131072: (1, 16), 524288: (1, 32), 1048576: (1, 64)},
        ModelSize.LARGE: {32768: (1, 16), 131072: (1, 32), 524288: (1, 64), 1048576: (1, 128)},
        ModelSize.XLARGE: {32768: (1, 32), 131072: (1, 64), 524288: (1, 128), 1048576: (1, 256)},
    }
    return configs[model_size].get(ctx_len, (1, 16))


def _get_training_steps(model_size: ModelSize, ctx_len: int) -> int:
    """Adjust training steps to keep similar token count"""
    base_steps = 100000
    base_ctx = 32768
    # Fewer steps for longer context to maintain similar compute
    return max(10000, int(base_steps * base_ctx / ctx_len))


def get_filtered_experiments(
    model_sizes: Optional[List[ModelSize]] = None,
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
