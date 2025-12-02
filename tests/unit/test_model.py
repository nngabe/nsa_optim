"""
Unit tests for model module
"""
import pytest
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import ModelConfig, AttentionType, MODEL_CONFIGS, ModelSize
from model import (
    RMSNorm,
    RotaryEmbedding,
    DenseAttention,
    NativeSparseAttention,
    MLP,
    TransformerBlock,
    TransformerModel,
    create_model,
    apply_rotary_pos_emb,
    rotate_half,
)


@pytest.fixture
def small_config():
    """Small model config for testing"""
    config = MODEL_CONFIGS[ModelSize.SMALL]
    # Reduce for faster tests
    return ModelConfig(
        name="test",
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        vocab_size=1000,
        max_position_embeddings=512,
        attention_type=AttentionType.DENSE,
    )


@pytest.fixture
def nsa_config(small_config):
    """NSA model config for testing"""
    small_config.attention_type = AttentionType.NSA
    small_config.nsa_block_size = 32
    small_config.nsa_window_size = 32
    small_config.nsa_num_selected_blocks = 4
    return small_config


class TestRMSNorm:
    """Tests for RMSNorm"""
    
    def test_output_shape(self):
        """Output shape should match input shape"""
        norm = RMSNorm(256)
        x = torch.randn(2, 10, 256)
        out = norm(x)
        assert out.shape == x.shape
    
    def test_normalization(self):
        """Output should be approximately normalized"""
        norm = RMSNorm(256)
        x = torch.randn(2, 10, 256) * 100  # Large values
        out = norm(x)
        # RMS should be close to 1 after normalization (before weight)
        rms = out.pow(2).mean(-1).sqrt()
        assert rms.mean().item() < 10  # Should be much smaller than input


class TestRotaryEmbedding:
    """Tests for Rotary Position Embedding"""
    
    def test_output_shapes(self):
        """Cos and sin should have correct shapes"""
        rope = RotaryEmbedding(64, max_position_embeddings=512)
        x = torch.randn(2, 128, 4, 64)  # [batch, seq, heads, dim]
        cos, sin = rope(x)
        assert cos.shape == (128, 64)
        assert sin.shape == (128, 64)
    
    def test_cache_extension(self):
        """Cache should extend for longer sequences"""
        rope = RotaryEmbedding(64, max_position_embeddings=128)
        x = torch.randn(1, 256, 1, 64)  # Longer than initial cache
        cos, sin = rope(x)
        assert cos.shape[0] >= 256


class TestApplyRotaryPosEmb:
    """Tests for rotary position embedding application"""
    
    def test_output_shapes(self):
        """Q and K should maintain shapes after rotation"""
        q = torch.randn(2, 128, 4, 64)
        k = torch.randn(2, 128, 4, 64)
        cos = torch.randn(128, 64)
        sin = torch.randn(128, 64)
        
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape


class TestDenseAttention:
    """Tests for Dense Attention"""
    
    def test_output_shape(self, small_config):
        """Output should match input shape"""
        attn = DenseAttention(small_config, layer_idx=0)
        x = torch.randn(2, 64, small_config.hidden_size)
        out, _ = attn(x)
        assert out.shape == x.shape
    
    def test_causal_masking(self, small_config):
        """Attention should be causal by default"""
        attn = DenseAttention(small_config, layer_idx=0)
        x = torch.randn(1, 32, small_config.hidden_size)
        
        # Two forward passes with same input should give same output
        out1, _ = attn(x)
        out2, _ = attn(x)
        assert torch.allclose(out1, out2)
    
    def test_kv_cache(self, small_config):
        """KV cache should work correctly"""
        attn = DenseAttention(small_config, layer_idx=0)
        
        # First pass
        x1 = torch.randn(1, 16, small_config.hidden_size)
        out1, cache = attn(x1, use_cache=True)
        assert cache is not None
        assert len(cache) == 2  # K and V
        
        # Second pass with cache
        x2 = torch.randn(1, 8, small_config.hidden_size)
        out2, cache2 = attn(x2, past_key_value=cache, use_cache=True)
        assert cache2[0].shape[2] == 24  # 16 + 8


class TestNativeSparseAttention:
    """Tests for Native Sparse Attention"""
    
    def test_output_shape(self, nsa_config):
        """Output should match input shape"""
        attn = NativeSparseAttention(nsa_config, layer_idx=0)
        x = torch.randn(2, 64, nsa_config.hidden_size)
        out, _ = attn(x)
        assert out.shape == x.shape
    
    def test_block_score_computation(self, nsa_config):
        """Block scores should be computed correctly"""
        attn = NativeSparseAttention(nsa_config, layer_idx=0)
        
        # Create dummy Q, K
        batch_size, seq_len = 2, 64
        q = torch.randn(batch_size, nsa_config.num_attention_heads, seq_len, 
                       nsa_config.hidden_size // nsa_config.num_attention_heads)
        k = torch.randn(batch_size, nsa_config.num_attention_heads, seq_len,
                       nsa_config.hidden_size // nsa_config.num_attention_heads)
        
        scores = attn._compute_block_scores(q, k)
        
        num_blocks = (seq_len + nsa_config.nsa_block_size - 1) // nsa_config.nsa_block_size
        assert scores.shape == (batch_size, nsa_config.num_attention_heads, seq_len, num_blocks)


class TestMLP:
    """Tests for MLP"""
    
    def test_output_shape(self, small_config):
        """Output should match input shape"""
        mlp = MLP(small_config)
        x = torch.randn(2, 64, small_config.hidden_size)
        out = mlp(x)
        assert out.shape == x.shape
    
    def test_swiglu_activation(self, small_config):
        """MLP should use SwiGLU activation"""
        mlp = MLP(small_config)
        x = torch.randn(2, 64, small_config.hidden_size)
        
        # Check intermediate computations
        gate = mlp.gate_proj(x)
        up = mlp.up_proj(x)
        # SwiGLU: silu(gate) * up
        intermediate = torch.nn.functional.silu(gate) * up
        out = mlp.down_proj(intermediate)
        
        # Should match forward output
        assert torch.allclose(out, mlp(x))


class TestTransformerBlock:
    """Tests for Transformer Block"""
    
    def test_output_shape(self, small_config):
        """Output should match input shape"""
        block = TransformerBlock(small_config, layer_idx=0)
        x = torch.randn(2, 64, small_config.hidden_size)
        out, _ = block(x)
        assert out.shape == x.shape
    
    def test_residual_connection(self, small_config):
        """Block should have residual connections"""
        block = TransformerBlock(small_config, layer_idx=0)
        x = torch.randn(2, 64, small_config.hidden_size)
        out, _ = block(x)
        
        # Output should not be exactly the same as attention/MLP output alone
        # (due to residual connection)
        assert not torch.allclose(out, x)


class TestTransformerModel:
    """Tests for full Transformer Model"""
    
    def test_output_shape(self, small_config):
        """Model output should have correct shape"""
        model = TransformerModel(small_config)
        input_ids = torch.randint(0, small_config.vocab_size, (2, 64))
        logits, _, _ = model(input_ids)
        assert logits.shape == (2, 64, small_config.vocab_size)
    
    def test_loss_computation(self, small_config):
        """Model should compute loss when labels provided"""
        model = TransformerModel(small_config)
        input_ids = torch.randint(0, small_config.vocab_size, (2, 64))
        _, loss, _ = model(input_ids, labels=input_ids)
        assert loss is not None
        assert loss.item() > 0
    
    def test_gradient_flow(self, small_config):
        """Gradients should flow through the model"""
        model = TransformerModel(small_config)
        input_ids = torch.randint(0, small_config.vocab_size, (2, 64))
        _, loss, _ = model(input_ids, labels=input_ids)
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_create_model_factory(self, small_config):
        """Factory function should create valid model"""
        model = create_model(small_config)
        assert isinstance(model, TransformerModel)
    
    def test_different_attention_types(self, small_config):
        """Model should work with different attention types"""
        # Dense attention
        small_config.attention_type = AttentionType.DENSE
        model_dense = create_model(small_config)
        
        # NSA attention
        small_config.attention_type = AttentionType.NSA
        small_config.nsa_block_size = 32
        small_config.nsa_window_size = 32
        small_config.nsa_num_selected_blocks = 4
        model_nsa = create_model(small_config)
        
        input_ids = torch.randint(0, small_config.vocab_size, (1, 64))
        
        out_dense, _, _ = model_dense(input_ids)
        out_nsa, _, _ = model_nsa(input_ids)
        
        assert out_dense.shape == out_nsa.shape


class TestModelMemory:
    """Tests for model memory characteristics"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_scaling_with_context(self, small_config):
        """Memory should scale appropriately with context length"""
        model = create_model(small_config).cuda()
        
        torch.cuda.reset_peak_memory_stats()
        
        # Short sequence
        x_short = torch.randint(0, small_config.vocab_size, (1, 64)).cuda()
        _, loss_short, _ = model(x_short, labels=x_short)
        loss_short.backward()
        mem_short = torch.cuda.max_memory_allocated()
        
        torch.cuda.reset_peak_memory_stats()
        
        # Longer sequence
        x_long = torch.randint(0, small_config.vocab_size, (1, 256)).cuda()
        _, loss_long, _ = model(x_long, labels=x_long)
        loss_long.backward()
        mem_long = torch.cuda.max_memory_allocated()
        
        # Memory should increase with sequence length
        assert mem_long > mem_short
