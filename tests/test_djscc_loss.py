"""Loss semantics and complete differentiable DJSCC path."""

from __future__ import annotations

import torch

from channels.awgn import keyed_complex_noise
from models.djscc import DJSCCOutput, build_djscc
from training.djscc_loss import DJSCCObjective


def test_lambda_zero_retains_mse_record_but_contributes_zero_gradient(
    run_config_factory,
):
    config = run_config_factory(reconstruction_weight=0.0)
    objective = DJSCCObjective.from_config(config)
    reconstruction = torch.rand(2, 3, 8, 8, requires_grad=True)
    logits = torch.randn(2, 10, requires_grad=True)
    inputs = torch.rand_like(reconstruction)
    symbols = torch.ones(2, 4, dtype=torch.complex64)
    output = DJSCCOutput(symbols, symbols, reconstruction, logits, torch.zeros(2))

    loss = objective(output, torch.tensor([1, 2]), inputs)
    loss.total.backward()

    assert torch.equal(loss.total, loss.cross_entropy)
    assert loss.reconstruction_mse.item() > 0
    assert reconstruction.grad is not None
    assert torch.count_nonzero(reconstruction.grad) == 0
    assert torch.count_nonzero(logits.grad)


def test_complete_encoder_awgn_decoder_backward_has_finite_nonzero_encoder_gradients(
    run_config_factory,
):
    config = run_config_factory("cifar10", "r_1_48")
    model = build_djscc(config)
    objective = DJSCCObjective.from_config(config)
    inputs = torch.rand(2, 3, 32, 32)
    noise = keyed_complex_noise(["gradient-a", "gradient-b"], config.resolved["k"])

    output = model(
        inputs,
        config.resolved["train_snr_db"],
        unit_noise=noise,
    )
    loss = objective(output, torch.tensor([1, 2]), inputs)
    loss.total.backward()

    gradients = [
        parameter.grad
        for parameter in model.encoder.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)
