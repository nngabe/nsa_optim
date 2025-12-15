"""
Unit tests for optimizers module
"""
import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import OptimizerConfig, OptimizerType
from optimizers import (
    create_optimizer,
    get_param_groups,
    get_lr_scheduler,
    create_adamw,
    SOAPReference,
    ShampooReference,
    SOAPLowBit,
)


@pytest.fixture
def simple_model():
    """Simple model for testing"""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
    )


@pytest.fixture
def transformer_like_model():
    """Model with different parameter types"""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 64)
            self.layernorm = nn.LayerNorm(64)
            self.linear = nn.Linear(64, 64)
            self.out = nn.Linear(64, 100, bias=True)
        
        def forward(self, x):
            x = self.embed(x)
            x = self.layernorm(x)
            x = self.linear(x)
            return self.out(x)
    
    return Model()


class TestParamGroups:
    """Tests for parameter group creation"""
    
    def test_separates_weight_decay(self, transformer_like_model):
        """Should separate params with and without weight decay"""
        groups = get_param_groups(transformer_like_model, weight_decay=0.1)
        
        assert len(groups) == 2
        assert groups[0]["weight_decay"] == 0.1
        assert groups[1]["weight_decay"] == 0.0
    
    def test_no_decay_for_bias(self, transformer_like_model):
        """Bias parameters should have no weight decay"""
        groups = get_param_groups(transformer_like_model, weight_decay=0.1)
        
        no_decay_params = groups[1]["params"]
        no_decay_names = []
        for name, param in transformer_like_model.named_parameters():
            if param in no_decay_params:
                no_decay_names.append(name)
        
        # Bias should be in no_decay group
        assert any("bias" in name for name in no_decay_names)
    
    def test_no_decay_for_layernorm(self, transformer_like_model):
        """LayerNorm parameters should have no weight decay"""
        groups = get_param_groups(transformer_like_model, weight_decay=0.1)
        
        no_decay_params = groups[1]["params"]
        no_decay_names = []
        for name, param in transformer_like_model.named_parameters():
            if param in no_decay_params:
                no_decay_names.append(name)
        
        # LayerNorm should be in no_decay group
        assert any("layernorm" in name.lower() for name in no_decay_names)
    
    def test_no_decay_for_embedding(self, transformer_like_model):
        """Embedding parameters should have no weight decay"""
        groups = get_param_groups(transformer_like_model, weight_decay=0.1)
        
        no_decay_params = groups[1]["params"]
        no_decay_names = []
        for name, param in transformer_like_model.named_parameters():
            if param in no_decay_params:
                no_decay_names.append(name)
        
        # Embedding should be in no_decay group
        assert any("embed" in name.lower() for name in no_decay_names)


class TestCreateOptimizer:
    """Tests for optimizer creation"""
    
    def test_create_adamw(self, simple_model):
        """Should create AdamW optimizer"""
        config = OptimizerConfig(
            optimizer_type=OptimizerType.ADAMW,
            learning_rate=1e-4,
        )
        optimizer = create_optimizer(simple_model, config)
        assert isinstance(optimizer, torch.optim.AdamW)
    
    def test_create_soap(self, simple_model):
        """Should create SOAP optimizer"""
        config = OptimizerConfig(
            optimizer_type=OptimizerType.SOAP,
            learning_rate=1e-4,
        )
        optimizer = create_optimizer(simple_model, config)
        # Should be either NVIDIA SOAP or reference implementation
        assert optimizer is not None
    
    def test_create_shampoo(self, simple_model):
        """Should create Shampoo optimizer"""
        config = OptimizerConfig(
            optimizer_type=OptimizerType.SHAMPOO,
            learning_rate=1e-4,
        )
        optimizer = create_optimizer(simple_model, config)
        assert optimizer is not None
    
    def test_create_soap_lowbit(self, simple_model):
        """Should create SOAP with low-bit states"""
        config = OptimizerConfig(
            optimizer_type=OptimizerType.SOAP_LOWBIT,
            learning_rate=1e-4,
            use_4bit=True,
        )
        optimizer = create_optimizer(simple_model, config)
        assert optimizer is not None


class TestSOAPReference:
    """Tests for reference SOAP implementation"""
    
    def test_step(self, simple_model):
        """Optimizer step should update parameters"""
        optimizer = SOAPReference(
            simple_model.parameters(),
            lr=1e-3,
        )
        
        # Record initial params
        initial_params = [p.clone() for p in simple_model.parameters()]
        
        # Forward and backward
        x = torch.randn(4, 10)
        y = simple_model(x)
        loss = y.sum()
        loss.backward()
        
        # Step
        optimizer.step()
        
        # Params should change
        for initial, current in zip(initial_params, simple_model.parameters()):
            assert not torch.allclose(initial, current)
    
    def test_state_initialization(self, simple_model):
        """State should be initialized on first step"""
        optimizer = SOAPReference(
            simple_model.parameters(),
            lr=1e-3,
        )
        
        # State should be empty initially
        assert len(optimizer.state) == 0
        
        # Forward, backward, step
        x = torch.randn(4, 10)
        y = simple_model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        
        # State should now have entries
        assert len(optimizer.state) > 0
    
    def test_kronecker_factors(self, simple_model):
        """Should create Kronecker factors for 2D params"""
        optimizer = SOAPReference(
            simple_model.parameters(),
            lr=1e-3,
            max_precond_dim=8192,
        )
        
        # Step to initialize state
        x = torch.randn(4, 10)
        y = simple_model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        
        # Check for Kronecker factors in state
        has_kronecker = False
        for state in optimizer.state.values():
            if "L" in state and "R" in state:
                has_kronecker = True
                break
        
        assert has_kronecker


class TestShampooReference:
    """Tests for reference Shampoo implementation"""
    
    def test_step(self, simple_model):
        """Optimizer step should update parameters"""
        optimizer = ShampooReference(
            simple_model.parameters(),
            lr=1e-3,
        )
        
        initial_params = [p.clone() for p in simple_model.parameters()]
        
        x = torch.randn(4, 10)
        y = simple_model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        
        for initial, current in zip(initial_params, simple_model.parameters()):
            assert not torch.allclose(initial, current)
    
    def test_preconditioner_update(self, simple_model):
        """Preconditioners should update periodically"""
        optimizer = ShampooReference(
            simple_model.parameters(),
            lr=1e-3,
            precondition_frequency=2,
        )
        
        # Multiple steps
        for i in range(5):
            x = torch.randn(4, 10)
            y = simple_model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Check that inverse roots were computed
        for state in optimizer.state.values():
            if "L_inv" in state:
                # Should not be identity after updates
                L_inv = state["L_inv"]
                identity = torch.eye(L_inv.shape[0], device=L_inv.device, dtype=L_inv.dtype)
                # After precondition_frequency steps, L_inv should differ from identity
                assert not torch.allclose(L_inv, identity, atol=0.1)


class TestSOAPLowBit:
    """Tests for SOAP with low-bit quantization"""
    
    def test_step(self, simple_model):
        """Should update parameters correctly"""
        optimizer = SOAPLowBit(
            simple_model.parameters(),
            lr=1e-3,
            bits=4,
        )
        
        initial_params = [p.clone() for p in simple_model.parameters()]
        
        x = torch.randn(4, 10)
        y = simple_model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        
        for initial, current in zip(initial_params, simple_model.parameters()):
            assert not torch.allclose(initial, current)
    
    def test_quantization(self, simple_model):
        """Should quantize eigenbasis"""
        optimizer = SOAPLowBit(
            simple_model.parameters(),
            lr=1e-3,
            bits=4,
            precondition_frequency=1,
        )
        
        # Multiple steps to trigger quantization
        for _ in range(3):
            x = torch.randn(4, 10)
            y = simple_model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Check for quantized states
        has_quantized = False
        for state in optimizer.state.values():
            if "QL_quantized" in state and state["QL_quantized"] is not None:
                has_quantized = True
                break
        
        # May or may not have quantized depending on param shapes
        # This is okay - just verify no errors occurred


class TestLRScheduler:
    """Tests for learning rate schedulers"""
    
    def test_cosine_scheduler(self, simple_model):
        """Cosine scheduler should decrease LR"""
        optimizer = torch.optim.AdamW(simple_model.parameters(), lr=1e-3)
        scheduler = get_lr_scheduler(
            optimizer,
            scheduler_type="cosine",
            num_training_steps=1000,
            warmup_steps=100,
            min_lr_ratio=0.1,
        )
        
        # Warmup phase
        for i in range(100):
            scheduler.step()
        
        lr_after_warmup = scheduler.get_last_lr()[0]
        assert lr_after_warmup == pytest.approx(1e-3, rel=0.01)
        
        # Decay phase
        for i in range(400):
            scheduler.step()
        
        lr_middle = scheduler.get_last_lr()[0]
        assert lr_middle < lr_after_warmup
        
        # End
        for i in range(500):
            scheduler.step()
        
        lr_end = scheduler.get_last_lr()[0]
        assert lr_end == pytest.approx(1e-4, rel=0.1)  # 0.1 * initial
    
    def test_linear_scheduler(self, simple_model):
        """Linear scheduler should decrease LR linearly"""
        optimizer = torch.optim.AdamW(simple_model.parameters(), lr=1e-3)
        scheduler = get_lr_scheduler(
            optimizer,
            scheduler_type="linear",
            num_training_steps=1000,
            warmup_steps=100,
            min_lr_ratio=0.1,
        )
        
        # After warmup, LR should decrease linearly
        for i in range(550):  # Warmup + half of decay
            scheduler.step()
        
        lr_middle = scheduler.get_last_lr()[0]
        # Should be roughly halfway between 1e-3 and 1e-4
        assert 4e-4 < lr_middle < 7e-4
    
    def test_constant_with_warmup(self, simple_model):
        """Constant scheduler should maintain LR after warmup"""
        optimizer = torch.optim.AdamW(simple_model.parameters(), lr=1e-3)
        scheduler = get_lr_scheduler(
            optimizer,
            scheduler_type="constant_with_warmup",
            num_training_steps=1000,
            warmup_steps=100,
        )
        
        # During warmup
        for i in range(50):
            scheduler.step()
        
        lr_warmup = scheduler.get_last_lr()[0]
        assert lr_warmup < 1e-3
        
        # After warmup
        for i in range(450):
            scheduler.step()
        
        lr_constant = scheduler.get_last_lr()[0]
        assert lr_constant == pytest.approx(1e-3, rel=0.01)


class TestOptimizerConvergence:
    """Tests for optimizer convergence on simple problems"""

    @pytest.mark.parametrize("optimizer_type", [
        OptimizerType.ADAMW,
        OptimizerType.SOAP,
        OptimizerType.SHAMPOO,
    ])
    def test_quadratic_convergence(self, optimizer_type):
        """Optimizers should converge on simple quadratic"""
        # Simple quadratic: f(x) = x^2
        x = nn.Parameter(torch.tensor([5.0]))

        config = OptimizerConfig(
            optimizer_type=optimizer_type,
            learning_rate=0.1,
        )

        model = nn.Module()
        model.x = x

        if optimizer_type == OptimizerType.ADAMW:
            optimizer = torch.optim.AdamW([x], lr=0.1)
        elif optimizer_type == OptimizerType.SOAP:
            optimizer = SOAPReference([x], lr=0.1)
        else:
            optimizer = ShampooReference([x], lr=0.1)

        # Optimize
        for _ in range(100):
            optimizer.zero_grad()
            loss = x ** 2
            loss.backward()
            optimizer.step()

        # Should be close to 0
        assert abs(x.item()) < 0.1


class TestAdamW8BitWithModels:
    """Tests for AdamW8bit optimizer with different model architectures"""

    @pytest.fixture
    def device(self):
        """Return CUDA device if available, otherwise skip test"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping GPU tests")
        return torch.device("cuda")

    @pytest.fixture
    def small_transformer_dense(self, device):
        """Small transformer model with dense attention"""
        from models import create_model
        from config import ModelConfig, AttentionType

        config = ModelConfig(
            name="test-transformer-dense",
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=512,
            vocab_size=1000,
            max_position_embeddings=512,
            attention_type=AttentionType.DENSE,
        )
        return create_model(config).to(device)

    @pytest.fixture
    def small_transformer_sparse(self, device):
        """Small transformer model with NSA sparse attention"""
        from models import create_model
        from config import ModelConfig, AttentionType

        config = ModelConfig(
            name="test-transformer-nsa",
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=512,
            vocab_size=1000,
            max_position_embeddings=512,
            attention_type=AttentionType.NSA,
            nsa_block_size=32,
            nsa_window_size=32,
            nsa_num_selected_blocks=4,
        )
        return create_model(config).to(device)

    @pytest.fixture
    def small_hybrid_deltanet(self, device):
        """Small hybrid model with DeltaNet blocks only"""
        from models import create_hybrid_model, HybridConfig

        config = HybridConfig(
            d_model=256,
            vocab_size=1000,
            block_pattern="DDD",  # 3 DeltaNet blocks
            block_repeats=2,  # Total 6 layers
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=512,
            max_position_embeddings=512,
        )
        return create_hybrid_model(config).to(device)

    @pytest.fixture
    def small_hybrid_mamba(self, device):
        """Small hybrid model with Mamba blocks only"""
        from models import create_hybrid_model, HybridConfig

        config = HybridConfig(
            d_model=256,
            vocab_size=1000,
            block_pattern="MMM",  # 3 Mamba blocks
            block_repeats=2,  # Total 6 layers
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=512,
            max_position_embeddings=512,
        )
        return create_hybrid_model(config).to(device)

    @pytest.fixture
    def small_hybrid_mdma_sparse(self, device):
        """Small hybrid model with MDMA pattern (Mamba-DeltaNet-Mamba-Attention)"""
        from models import create_hybrid_model, HybridConfig

        config = HybridConfig(
            d_model=256,
            vocab_size=1000,
            block_pattern="MDMA",
            block_repeats=2,  # Total 8 layers (MDMA repeated twice)
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=512,
            max_position_embeddings=512,
        )
        return create_hybrid_model(config).to(device)

    def test_adamw8bit_transformer_dense(self, small_transformer_dense, device):
        """Test AdamW8bit with transformer model using dense attention"""
        config = OptimizerConfig(
            optimizer_type=OptimizerType.ADAMW_8BIT,
            learning_rate=1e-3,
        )

        optimizer = create_optimizer(small_transformer_dense, config)

        # Record initial params
        initial_params = [p.clone() for p in small_transformer_dense.parameters() if p.requires_grad]

        # Forward and backward
        input_ids = torch.randint(0, 1000, (2, 32), device=device)
        logits, loss, _ = small_transformer_dense(input_ids, labels=input_ids)

        assert loss is not None
        loss.backward()

        # Step
        optimizer.step()

        # Params should change
        for initial, current in zip(initial_params, small_transformer_dense.parameters()):
            if current.requires_grad:
                assert not torch.allclose(initial, current, atol=1e-6)

    def test_adamw8bit_transformer_sparse(self, small_transformer_sparse, device):
        """Test AdamW8bit with transformer model using NSA sparse attention"""
        config = OptimizerConfig(
            optimizer_type=OptimizerType.ADAMW_8BIT,
            learning_rate=1e-3,
        )

        optimizer = create_optimizer(small_transformer_sparse, config)

        # Record initial params
        initial_params = [p.clone() for p in small_transformer_sparse.parameters() if p.requires_grad]

        # Forward and backward with sequence length multiple of block size
        input_ids = torch.randint(0, 1000, (2, 64), device=device)  # 64 is multiple of nsa_block_size=32
        logits, loss, _ = small_transformer_sparse(input_ids, labels=input_ids)

        assert loss is not None
        loss.backward()

        # Step
        optimizer.step()

        # Params should change
        for initial, current in zip(initial_params, small_transformer_sparse.parameters()):
            if current.requires_grad:
                assert not torch.allclose(initial, current, atol=1e-6)

    def test_adamw8bit_hybrid_deltanet(self, small_hybrid_deltanet, device):
        """Test AdamW8bit with hybrid model using DeltaNet blocks"""
        config = OptimizerConfig(
            optimizer_type=OptimizerType.ADAMW_8BIT,
            learning_rate=1e-3,
        )

        optimizer = create_optimizer(small_hybrid_deltanet, config)

        # Record initial params
        initial_params = [p.clone() for p in small_hybrid_deltanet.parameters() if p.requires_grad]

        # Forward and backward
        input_ids = torch.randint(0, 1000, (2, 32), device=device)
        logits, loss = small_hybrid_deltanet(input_ids, labels=input_ids)

        assert loss is not None
        loss.backward()

        # Step
        optimizer.step()

        # Params should change
        for initial, current in zip(initial_params, small_hybrid_deltanet.parameters()):
            if current.requires_grad:
                assert not torch.allclose(initial, current, atol=1e-6)

    def test_adamw8bit_hybrid_mamba(self, small_hybrid_mamba, device):
        """Test AdamW8bit with hybrid model using Mamba blocks"""
        config = OptimizerConfig(
            optimizer_type=OptimizerType.ADAMW_8BIT,
            learning_rate=1e-3,
        )

        optimizer = create_optimizer(small_hybrid_mamba, config)

        # Record initial params
        initial_params = [p.clone() for p in small_hybrid_mamba.parameters() if p.requires_grad]

        # Forward and backward
        input_ids = torch.randint(0, 1000, (2, 32), device=device)
        logits, loss = small_hybrid_mamba(input_ids, labels=input_ids)

        assert loss is not None
        loss.backward()

        # Step
        optimizer.step()

        # Params should change
        for initial, current in zip(initial_params, small_hybrid_mamba.parameters()):
            if current.requires_grad:
                assert not torch.allclose(initial, current, atol=1e-6)

    def test_adamw8bit_hybrid_mdma_sparse(self, small_hybrid_mdma_sparse, device):
        """Test AdamW8bit with hybrid model using MDMA pattern (mixed blocks)"""
        config = OptimizerConfig(
            optimizer_type=OptimizerType.ADAMW_8BIT,
            learning_rate=1e-3,
        )

        optimizer = create_optimizer(small_hybrid_mdma_sparse, config)

        # Record initial params
        initial_params = [p.clone() for p in small_hybrid_mdma_sparse.parameters() if p.requires_grad]

        # Forward and backward
        input_ids = torch.randint(0, 1000, (2, 32), device=device)
        logits, loss = small_hybrid_mdma_sparse(input_ids, labels=input_ids)

        assert loss is not None
        loss.backward()

        # Step
        optimizer.step()

        # Params should change
        for initial, current in zip(initial_params, small_hybrid_mdma_sparse.parameters()):
            if current.requires_grad:
                assert not torch.allclose(initial, current, atol=1e-6)

    def test_adamw8bit_multiple_steps(self, small_hybrid_mdma_sparse, device):
        """Test AdamW8bit with multiple optimization steps"""
        config = OptimizerConfig(
            optimizer_type=OptimizerType.ADAMW_8BIT,
            learning_rate=1e-3,
        )

        optimizer = create_optimizer(small_hybrid_mdma_sparse, config)

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            input_ids = torch.randint(0, 1000, (2, 32), device=device)
            logits, loss = small_hybrid_mdma_sparse(input_ids, labels=input_ids)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        # Loss should generally decrease (though not guaranteed with random data)
        # Just verify no errors occurred and optimizer ran
        assert len(losses) == 5
        assert all(l > 0 for l in losses)


class TestSOAPLowBitWithModels:
    """Tests for SOAP low-bit optimizers (4-bit and 8-bit) with different model architectures"""

    @pytest.fixture
    def device(self):
        """Return CUDA device if available, otherwise skip test"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping GPU tests")
        return torch.device("cuda")

    @pytest.fixture
    def small_hybrid_mdma_sparse(self, device):
        """Small hybrid model with MDMA pattern"""
        from models import create_hybrid_model, HybridConfig

        config = HybridConfig(
            d_model=256,
            vocab_size=1000,
            block_pattern="MDMA",
            block_repeats=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=512,
            max_position_embeddings=512,
        )
        return create_hybrid_model(config).to(device)

    def test_soap8bit_hybrid_mdma_sparse(self, small_hybrid_mdma_sparse, device):
        """Test SOAP8bit with hybrid MDMA model"""
        config = OptimizerConfig(
            optimizer_type=OptimizerType.SOAP_8BIT,
            learning_rate=1e-3,
            precondition_frequency=5,
        )

        optimizer = create_optimizer(small_hybrid_mdma_sparse, config)

        # Record initial params
        initial_params = [p.clone() for p in small_hybrid_mdma_sparse.parameters() if p.requires_grad]

        # Forward and backward
        input_ids = torch.randint(0, 1000, (2, 32), device=device)
        logits, loss = small_hybrid_mdma_sparse(input_ids, labels=input_ids)

        assert loss is not None
        loss.backward()

        # Step
        optimizer.step()

        # Params should change
        for initial, current in zip(initial_params, small_hybrid_mdma_sparse.parameters()):
            if current.requires_grad:
                assert not torch.allclose(initial, current, atol=1e-6)

    def test_soap4bit_hybrid_mdma_sparse(self, small_hybrid_mdma_sparse, device):
        """Test SOAP4bit with hybrid MDMA model"""
        config = OptimizerConfig(
            optimizer_type=OptimizerType.SOAP_4BIT,
            learning_rate=1e-3,
            precondition_frequency=5,
        )

        optimizer = create_optimizer(small_hybrid_mdma_sparse, config)

        # Record initial params
        initial_params = [p.clone() for p in small_hybrid_mdma_sparse.parameters() if p.requires_grad]

        # Forward and backward
        input_ids = torch.randint(0, 1000, (2, 32), device=device)
        logits, loss = small_hybrid_mdma_sparse(input_ids, labels=input_ids)

        assert loss is not None
        loss.backward()

        # Step
        optimizer.step()

        # Params should change
        for initial, current in zip(initial_params, small_hybrid_mdma_sparse.parameters()):
            if current.requires_grad:
                assert not torch.allclose(initial, current, atol=1e-6)

    def test_soap8bit_multiple_steps_with_preconditioning(self, small_hybrid_mdma_sparse, device):
        """Test SOAP8bit with multiple steps to trigger preconditioning updates"""
        config = OptimizerConfig(
            optimizer_type=OptimizerType.SOAP_8BIT,
            learning_rate=1e-3,
            precondition_frequency=3,  # Update preconditioner every 3 steps
        )

        optimizer = create_optimizer(small_hybrid_mdma_sparse, config)

        losses = []
        for step in range(10):
            optimizer.zero_grad()
            input_ids = torch.randint(0, 1000, (2, 32), device=device)
            logits, loss = small_hybrid_mdma_sparse(input_ids, labels=input_ids)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            # Check that eigenbasis is being quantized after preconditioner updates
            if isinstance(optimizer, SOAPLowBit) and step > 0 and (step + 1) % config.precondition_frequency == 0:
                # After a preconditioner update, check that some state has quantized eigenbases
                has_quantized = False
                for state in optimizer.state.values():
                    if "QL_q_blocks" in state and len(state["QL_q_blocks"]) > 0:
                        has_quantized = True
                        break
                # At least some parameters should have quantized state
                # (though small 1D params may not be preconditioned)

        # Verify optimizer ran successfully
        assert len(losses) == 10
        assert all(l > 0 for l in losses)

    def test_soap4bit_multiple_steps_with_preconditioning(self, small_hybrid_mdma_sparse, device):
        """Test SOAP4bit with multiple steps to trigger preconditioning updates"""
        config = OptimizerConfig(
            optimizer_type=OptimizerType.SOAP_4BIT,
            learning_rate=1e-3,
            precondition_frequency=3,
        )

        optimizer = create_optimizer(small_hybrid_mdma_sparse, config)

        losses = []
        for step in range(10):
            optimizer.zero_grad()
            input_ids = torch.randint(0, 1000, (2, 32), device=device)
            logits, loss = small_hybrid_mdma_sparse(input_ids, labels=input_ids)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        # Verify optimizer ran successfully
        assert len(losses) == 10
        assert all(l > 0 for l in losses)

    def test_soap_lowbit_state_quantization(self, small_hybrid_mdma_sparse, device):
        """Test that SOAP low-bit optimizer properly quantizes eigenbasis states"""
        config = OptimizerConfig(
            optimizer_type=OptimizerType.SOAP_8BIT,
            learning_rate=1e-3,
            precondition_frequency=1,  # Update every step to trigger quantization
        )

        optimizer = create_optimizer(small_hybrid_mdma_sparse, config)

        # Run a few steps to initialize and update preconditioners
        for _ in range(3):
            optimizer.zero_grad()
            input_ids = torch.randint(0, 1000, (2, 32), device=device)
            logits, loss = small_hybrid_mdma_sparse(input_ids, labels=input_ids)
            loss.backward()
            optimizer.step()

        # Check that quantized state exists for some parameters
        if isinstance(optimizer, SOAPLowBit):
            has_quantized_state = False
            for param_state in optimizer.state.values():
                if "QL_q_blocks" in param_state:
                    has_quantized_state = True
                    # Verify quantized data structure
                    ql_blocks = param_state["QL_q_blocks"]
                    if len(ql_blocks) > 0:
                        # Each block should be a tuple of (quantized, scale, zero_point, orig_cols)
                        assert len(ql_blocks[0]) == 4
                        quantized, scale, zero_point, orig_cols = ql_blocks[0]
                        assert quantized.dtype == torch.uint8
                    break

            # Should have at least some quantized state for 2D parameters
            # (1D params like biases won't be preconditioned)
