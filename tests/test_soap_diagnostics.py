"""Diagnostic tests for SOAPLowBit optimizer"""
import torch
import torch.nn as nn
from optimizers import SOAPLowBit


def test_kronecker_factor_updates():
    """Test that Kronecker factors are being updated correctly"""
    torch.manual_seed(42)

    # Simple 2D parameter
    param = nn.Parameter(torch.randn(64, 128))
    optimizer = SOAPLowBit([param], lr=1e-3, bits=4, q_block_size=64, precondition_frequency=5)

    print("\n=== Kronecker Factor Update Test ===")

    for step in range(1, 16):
        # Simulate gradient
        param.grad = torch.randn_like(param)
        grad_norm = param.grad.norm().item()

        optimizer.step()
        optimizer.zero_grad()

        state = optimizer.state[param]
        L = state.get("L")
        R = state.get("R")
        QL_q = state.get("QL_q")

        if L is not None:
            L_norm = L.norm().item()
            R_norm = R.norm().item()
            quantized = QL_q is not None
            print(f"Step {step:2d}: grad_norm={grad_norm:.4f}, L_norm={L_norm:.4f}, R_norm={R_norm:.4f}, quantized={quantized}")


def test_eigenbasis_quality():
    """Test eigenbasis quality before and after quantization"""
    torch.manual_seed(42)

    param = nn.Parameter(torch.randn(64, 128))
    optimizer = SOAPLowBit([param], lr=1e-3, bits=4, q_block_size=64, precondition_frequency=1)

    print("\n=== Eigenbasis Quality Test ===")

    for step in range(1, 6):
        param.grad = torch.randn_like(param)
        optimizer.step()

        state = optimizer.state[param]
        if state.get("QL_q") is not None:
            # Dequantize and check orthogonality
            QL = optimizer._dequantize_block(*state["QL_q"])
            QR = optimizer._dequantize_block(*state["QR_q"])

            # Check orthogonality: Q @ Q.T should be identity
            QL_orth_error = (QL @ QL.T - torch.eye(QL.shape[0], device=QL.device)).norm().item()
            QR_orth_error = (QR @ QR.T - torch.eye(QR.shape[0], device=QR.device)).norm().item()

            print(f"Step {step}: QL orthogonality error={QL_orth_error:.6f}, QR orthogonality error={QR_orth_error:.6f}")

        optimizer.zero_grad()


def test_gradient_projection():
    """Test gradient projection through eigenbasis"""
    torch.manual_seed(42)

    param = nn.Parameter(torch.randn(64, 128))
    optimizer = SOAPLowBit([param], lr=1e-3, bits=4, q_block_size=64, precondition_frequency=1)

    print("\n=== Gradient Projection Test ===")

    # Warm up to get eigenbasis
    for _ in range(3):
        param.grad = torch.randn_like(param)
        optimizer.step()
        optimizer.zero_grad()

    state = optimizer.state[param]
    QL = optimizer._dequantize_block(*state["QL_q"])
    QR = optimizer._dequantize_block(*state["QR_q"])

    # New gradient
    grad = torch.randn_like(param)
    grad_norm = grad.norm().item()

    # Project to eigenbasis
    grad_proj = QL.T @ grad.float() @ QR
    proj_norm = grad_proj.norm().item()

    # Project back
    grad_back = QL @ grad_proj @ QR.T
    reconstruction_error = (grad_back - grad.float()).norm().item()

    print(f"Original grad norm: {grad_norm:.4f}")
    print(f"Projected grad norm: {proj_norm:.4f}")
    print(f"Reconstruction error: {reconstruction_error:.6f}")
    print(f"Relative error: {reconstruction_error / grad_norm * 100:.4f}%")


def test_update_magnitude():
    """Test update magnitudes for SOAPLowBit at different bit widths"""
    torch.manual_seed(42)

    param_4bit = nn.Parameter(torch.randn(64, 128))
    param_8bit = nn.Parameter(param_4bit.data.clone())

    opt_4bit = SOAPLowBit([param_4bit], lr=1e-3, bits=4, q_block_size=64, precondition_frequency=1)
    opt_8bit = SOAPLowBit([param_8bit], lr=1e-3, bits=8, q_block_size=64, precondition_frequency=1)

    print("\n=== Update Magnitude: 4-bit vs 8-bit ===")

    for step in range(1, 11):
        torch.manual_seed(step)
        grad = torch.randn(64, 128)

        param_4bit.grad = grad.clone()
        param_8bit.grad = grad.clone()

        p4_before = param_4bit.data.clone()
        p8_before = param_8bit.data.clone()

        opt_4bit.step()
        opt_8bit.step()

        update_4 = (param_4bit.data - p4_before).norm().item()
        update_8 = (param_8bit.data - p8_before).norm().item()
        param_diff = (param_4bit.data - param_8bit.data).norm().item()

        print(f"Step {step:2d}: 4bit_update={update_4:.6f}, 8bit_update={update_8:.6f}, diff={param_diff:.6f}")

        opt_4bit.zero_grad()
        opt_8bit.zero_grad()


def test_quantization_error():
    """Test quantization error for different block sizes"""
    torch.manual_seed(42)

    print("\n=== Quantization Error vs Block Size ===")

    # Create a realistic eigenbasis matrix (orthogonal)
    Q, _ = torch.linalg.qr(torch.randn(256, 256))

    for block_size in [32, 64, 128, 256]:
        optimizer = SOAPLowBit(
            [nn.Parameter(torch.randn(10, 10))],  # Dummy param
            lr=1e-3, bits=4, q_block_size=block_size
        )

        quantized, scale, zp, orig_cols = optimizer._quantize_block(Q)
        Q_recon = optimizer._dequantize_block(quantized, scale, zp, orig_cols)

        quant_error = (Q_recon - Q).norm().item()
        relative_error = quant_error / Q.norm().item() * 100

        # Check orthogonality preservation
        orth_error = (Q_recon @ Q_recon.T - torch.eye(256)).norm().item()

        print(f"block_size={block_size:3d}: quant_error={quant_error:.6f} ({relative_error:.2f}%), orth_error={orth_error:.6f}")


def test_loss_landscape():
    """Test optimization on a simple quadratic loss: 4-bit vs 8-bit"""
    torch.manual_seed(42)

    print("\n=== Simple Quadratic Optimization: 4-bit vs 8-bit ===")

    target = torch.randn(32, 64)

    param_4bit = nn.Parameter(torch.randn(32, 64))
    param_8bit = nn.Parameter(param_4bit.data.clone())

    opt_4bit = SOAPLowBit([param_4bit], lr=0.01, bits=4, q_block_size=64, precondition_frequency=1)
    opt_8bit = SOAPLowBit([param_8bit], lr=0.01, bits=8, q_block_size=64, precondition_frequency=1)

    print("Step | Loss(4bit) | Loss(8bit) | Diff")
    print("-" * 50)

    for step in range(1, 51):
        loss_4bit = ((param_4bit - target) ** 2).mean()
        loss_8bit = ((param_8bit - target) ** 2).mean()

        loss_4bit.backward()
        loss_8bit.backward()

        opt_4bit.step()
        opt_8bit.step()

        opt_4bit.zero_grad()
        opt_8bit.zero_grad()

        if step % 10 == 0 or step == 1:
            print(f"{step:4d} | {loss_4bit.item():10.6f} | {loss_8bit.item():10.6f} | {abs(loss_4bit.item() - loss_8bit.item()):.6f}")


def test_preconditioner_effect():
    """Test whether preconditioning is actually helping gradient direction"""
    torch.manual_seed(42)

    print("\n=== Preconditioner Effect on Gradient ===")

    # Create a model with known loss landscape
    model = nn.Linear(64, 32, bias=False)
    target = torch.randn(10, 32)

    opt = SOAPLowBit(model.parameters(), lr=0.01, bits=4, q_block_size=32, precondition_frequency=1)

    print("Step | Loss | GradNorm | UpdateNorm | Cosine(grad,update)")
    print("-" * 65)

    for step in range(1, 21):
        x = torch.randn(10, 64)
        pred = model(x)
        loss = ((pred - target) ** 2).mean()
        loss.backward()

        # Get gradient before step
        grad_before = model.weight.grad.clone()
        param_before = model.weight.data.clone()

        opt.step()

        # Compute actual update
        actual_update = model.weight.data - param_before
        update_norm = actual_update.norm().item()
        grad_norm = grad_before.norm().item()

        # Cosine similarity between gradient and update direction
        # (negative because update should be opposite to gradient)
        cosine = -torch.nn.functional.cosine_similarity(
            grad_before.flatten().unsqueeze(0),
            actual_update.flatten().unsqueeze(0)
        ).item()

        opt.zero_grad()

        if step % 5 == 0 or step == 1:
            print(f"{step:4d} | {loss.item():.4f} | {grad_norm:.4f} | {update_norm:.6f} | {cosine:.4f}")


def test_8bit_vs_no_quant():
    """Compare 8-bit quantized vs keeping eigenbasis in fp32"""
    torch.manual_seed(42)

    print("\n=== 8-bit Should Be Nearly Lossless ===")

    Q, _ = torch.linalg.qr(torch.randn(128, 128))

    opt = SOAPLowBit([nn.Parameter(torch.randn(10, 10))], lr=1e-3, bits=8, q_block_size=32)

    quantized, scale, zp, orig_cols = opt._quantize_block(Q)
    Q_recon = opt._dequantize_block(quantized, scale, zp, orig_cols)

    quant_error = (Q_recon - Q).norm().item()
    orth_error = (Q_recon @ Q_recon.T - torch.eye(128)).norm().item()

    print(f"8-bit quantization error: {quant_error:.6f} ({quant_error / Q.norm().item() * 100:.4f}%)")
    print(f"8-bit orthogonality error: {orth_error:.6f}")


def test_8bit_eigenbasis_quality():
    """Test 8-bit eigenbasis quality over steps"""
    torch.manual_seed(42)

    param = nn.Parameter(torch.randn(64, 128))
    optimizer = SOAPLowBit([param], lr=1e-3, bits=8, q_block_size=32, precondition_frequency=1)

    print("\n=== 8-bit Eigenbasis Quality Test ===")

    for step in range(1, 6):
        param.grad = torch.randn_like(param)
        optimizer.step()

        state = optimizer.state[param]
        if state.get("QL_q") is not None:
            QL = optimizer._dequantize_block(*state["QL_q"])
            QR = optimizer._dequantize_block(*state["QR_q"])

            QL_orth_error = (QL @ QL.T - torch.eye(QL.shape[0], device=QL.device)).norm().item()
            QR_orth_error = (QR @ QR.T - torch.eye(QR.shape[0], device=QR.device)).norm().item()

            print(f"Step {step}: QL orthogonality error={QL_orth_error:.6f}, QR orthogonality error={QR_orth_error:.6f}")

        optimizer.zero_grad()


def test_8bit_gradient_projection():
    """Test 8-bit gradient projection through eigenbasis"""
    torch.manual_seed(42)

    param = nn.Parameter(torch.randn(64, 128))
    optimizer = SOAPLowBit([param], lr=1e-3, bits=8, q_block_size=32, precondition_frequency=1)

    print("\n=== 8-bit Gradient Projection Test ===")

    for _ in range(3):
        param.grad = torch.randn_like(param)
        optimizer.step()
        optimizer.zero_grad()

    state = optimizer.state[param]
    QL = optimizer._dequantize_block(*state["QL_q"])
    QR = optimizer._dequantize_block(*state["QR_q"])

    grad = torch.randn_like(param)
    grad_norm = grad.norm().item()

    grad_proj = QL.T @ grad.float() @ QR
    proj_norm = grad_proj.norm().item()

    grad_back = QL @ grad_proj @ QR.T
    reconstruction_error = (grad_back - grad.float()).norm().item()

    print(f"Original grad norm: {grad_norm:.4f}")
    print(f"Projected grad norm: {proj_norm:.4f}")
    print(f"Reconstruction error: {reconstruction_error:.6f}")
    print(f"Relative error: {reconstruction_error / grad_norm * 100:.4f}%")


def test_8bit_preconditioner_effect():
    """Test 8-bit preconditioner effect on gradient direction"""
    torch.manual_seed(42)

    print("\n=== 8-bit Preconditioner Effect on Gradient ===")

    model = nn.Linear(64, 32, bias=False)
    target = torch.randn(10, 32)

    opt = SOAPLowBit(model.parameters(), lr=0.01, bits=8, q_block_size=32, precondition_frequency=1)

    print("Step | Loss | GradNorm | UpdateNorm | Cosine(grad,update)")
    print("-" * 65)

    for step in range(1, 21):
        x = torch.randn(10, 64)
        pred = model(x)
        loss = ((pred - target) ** 2).mean()
        loss.backward()

        grad_before = model.weight.grad.clone()
        param_before = model.weight.data.clone()

        opt.step()

        actual_update = model.weight.data - param_before
        update_norm = actual_update.norm().item()
        grad_norm = grad_before.norm().item()

        cosine = -torch.nn.functional.cosine_similarity(
            grad_before.flatten().unsqueeze(0),
            actual_update.flatten().unsqueeze(0)
        ).item()

        opt.zero_grad()

        if step % 5 == 0 or step == 1:
            print(f"{step:4d} | {loss.item():.4f} | {grad_norm:.4f} | {update_norm:.6f} | {cosine:.4f}")


if __name__ == "__main__":
    print("=" * 70)
    print("4-BIT TESTS")
    print("=" * 70)
    test_kronecker_factor_updates()
    test_eigenbasis_quality()
    test_gradient_projection()
    test_quantization_error()
    test_preconditioner_effect()

    print("\n" + "=" * 70)
    print("8-BIT TESTS")
    print("=" * 70)
    test_8bit_vs_no_quant()
    test_8bit_eigenbasis_quality()
    test_8bit_gradient_projection()
    test_8bit_preconditioner_effect()

    print("\n" + "=" * 70)
    print("COMPARISON TESTS")
    print("=" * 70)
    test_update_magnitude()
    test_loss_landscape()
