"""
Unit tests for configuration module
"""
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
    AttentionType,
    OptimizerType,
    ModelSize,
    MODEL_CONFIGS,
    CONTEXT_LENGTHS,
    get_experiment_grid,
    get_filtered_experiments,
)


class TestModelConfig:
    """Tests for ModelConfig"""
    
    def test_model_configs_exist(self):
        """All model sizes should have configurations"""
        for size in ModelSize:
            assert size in MODEL_CONFIGS
            config = MODEL_CONFIGS[size]
            assert config.hidden_size > 0
            assert config.num_hidden_layers > 0
            assert config.num_attention_heads > 0
    
    def test_model_config_consistency(self):
        """Model configs should have consistent dimensions"""
        for size, config in MODEL_CONFIGS.items():
            # Hidden size should be divisible by num_attention_heads
            assert config.hidden_size % config.num_attention_heads == 0
            # num_attention_heads should be divisible by num_key_value_heads (for GQA)
            assert config.num_attention_heads % config.num_key_value_heads == 0
    
    def test_model_sizes_ordered(self):
        """Model sizes should be ordered by parameter count"""
        sizes = [
            MODEL_CONFIGS[ModelSize.SMALL],
            MODEL_CONFIGS[ModelSize.MEDIUM],
            MODEL_CONFIGS[ModelSize.LARGE],
            MODEL_CONFIGS[ModelSize.XLARGE],
        ]
        
        for i in range(len(sizes) - 1):
            # Larger models should have more hidden units or layers
            assert (sizes[i].hidden_size <= sizes[i+1].hidden_size or 
                    sizes[i].num_hidden_layers <= sizes[i+1].num_hidden_layers)


class TestOptimizerConfig:
    """Tests for OptimizerConfig"""
    
    def test_default_config(self):
        """Default config should have reasonable values"""
        config = OptimizerConfig(optimizer_type=OptimizerType.ADAMW)
        assert config.learning_rate > 0
        assert 0 < config.beta1 < 1
        assert 0 < config.beta2 < 1
        assert config.eps > 0
    
    def test_all_optimizer_types(self):
        """All optimizer types should be configurable"""
        for opt_type in OptimizerType:
            config = OptimizerConfig(optimizer_type=opt_type)
            assert config.optimizer_type == opt_type


class TestTrainingConfig:
    """Tests for TrainingConfig"""
    
    def test_default_config(self):
        """Default config should be valid"""
        config = TrainingConfig()
        assert config.model_size in ModelSize
        assert config.attention_type in AttentionType
        assert config.optimizer_type in OptimizerType
        assert config.max_seq_length > 0
        assert config.batch_size > 0
    
    def test_run_name_generation(self):
        """Run name should be auto-generated if not provided"""
        config = TrainingConfig(
            model_size=ModelSize.SMALL,
            attention_type=AttentionType.NSA,
            optimizer_type=OptimizerType.SOAP,
            max_seq_length=32768,
        )
        assert "0.6B" in config.run_name
        assert "native_sparse_attention" in config.run_name
        assert "soap" in config.run_name
        assert "32768" in config.run_name


class TestExperimentGrid:
    """Tests for experiment grid generation"""
    
    def test_experiment_grid_non_empty(self):
        """Experiment grid should generate experiments"""
        experiments = get_experiment_grid()
        assert len(experiments) > 0
    
    def test_experiment_grid_coverage(self):
        """Grid should cover all combinations"""
        experiments = get_experiment_grid()
        
        model_sizes = set(e.model_size for e in experiments)
        attention_types = set(e.attention_type for e in experiments)
        optimizer_types = set(e.optimizer_type for e in experiments)
        
        # Should have all model sizes
        assert model_sizes == set(ModelSize)
        # Should have all attention types
        assert attention_types == set(AttentionType)
        # Should have all optimizer types
        assert optimizer_types == set(OptimizerType)
    
    def test_extended_context_only_for_nsa(self):
        """512k and 1M context should only be for NSA"""
        experiments = get_experiment_grid()
        
        for exp in experiments:
            if exp.max_seq_length > 131072:
                assert exp.attention_type == AttentionType.NSA, \
                    f"Extended context {exp.max_seq_length} should only be for NSA"
    
    def test_filtered_experiments(self):
        """Filtering should return subset"""
        all_experiments = get_experiment_grid()
        
        # Filter by model size
        filtered = get_filtered_experiments(model_sizes=[ModelSize.SMALL])
        assert len(filtered) < len(all_experiments)
        assert all(e.model_size == ModelSize.SMALL for e in filtered)
        
        # Filter by attention type
        filtered = get_filtered_experiments(attention_types=[AttentionType.NSA])
        assert all(e.attention_type == AttentionType.NSA for e in filtered)
        
        # Filter by optimizer
        filtered = get_filtered_experiments(optimizer_types=[OptimizerType.ADAMW])
        assert all(e.optimizer_type == OptimizerType.ADAMW for e in filtered)
        
        # Multiple filters
        filtered = get_filtered_experiments(
            model_sizes=[ModelSize.SMALL],
            attention_types=[AttentionType.DENSE],
            optimizer_types=[OptimizerType.ADAMW],
        )
        assert all(
            e.model_size == ModelSize.SMALL and
            e.attention_type == AttentionType.DENSE and
            e.optimizer_type == OptimizerType.ADAMW
            for e in filtered
        )


class TestContextLengths:
    """Tests for context length configurations"""
    
    def test_context_lengths_defined(self):
        """Context lengths should be defined"""
        assert "standard" in CONTEXT_LENGTHS
        assert "extended_nsa" in CONTEXT_LENGTHS
    
    def test_standard_context_lengths(self):
        """Standard context lengths should include 32k and 128k"""
        standard = CONTEXT_LENGTHS["standard"]
        assert 32768 in standard
        assert 131072 in standard
    
    def test_extended_context_lengths(self):
        """Extended context lengths should include 512k and 1M"""
        extended = CONTEXT_LENGTHS["extended_nsa"]
        assert 524288 in extended
        assert 1048576 in extended
