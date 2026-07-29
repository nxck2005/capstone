"""Per-image complex-symbol power and AWGN convention tests."""

from __future__ import annotations

import pytest
import torch

from channels.awgn import AWGN
from channels.power import normalize_unit_average_power
from config.params import get


def test_unit_average_power_is_per_image_and_preserves_phase_shape_dtype():
    generator = torch.Generator().manual_seed(42)
    real = torch.randn(7, 4096, generator=generator)
    imaginary = torch.randn(7, 4096, generator=generator)
    symbols = torch.complex(real, imaginary)
    phases = torch.angle(symbols)

    normalized = normalize_unit_average_power(symbols)

    assert normalized.shape == symbols.shape
    assert normalized.dtype == symbols.dtype
    torch.testing.assert_close(torch.angle(normalized), phases)
    torch.testing.assert_close(
        normalized.abs().square().mean(dim=1),
        torch.ones(7),
        atol=1e-3,
        rtol=0,
    )


@pytest.mark.parametrize(
    "symbols",
    [
        torch.zeros(2, 8, dtype=torch.complex64),
        torch.tensor([[complex(float("nan"), 0)]], dtype=torch.complex64),
    ],
)
def test_power_normalization_rejects_zero_or_nonfinite_samples(symbols):
    with pytest.raises(ValueError, match="finite|non-zero"):
        normalize_unit_average_power(symbols)


def test_power_normalization_rejects_real_or_wrong_shape():
    with pytest.raises(TypeError, match="complex"):
        normalize_unit_average_power(torch.ones(2, 8))
    with pytest.raises(ValueError, match=r"\[B, k\]"):
        normalize_unit_average_power(torch.ones(2, 2, 8, dtype=torch.complex64))


@pytest.mark.parametrize(
    "snr_db",
    [
        get("channel.test_snr_grid_db")[0],
        get("channel.train_snr_db_fixed"),
        get("channel.test_snr_grid_db")[-1],
    ],
)
def test_awgn_empirical_esn0_matches_requested_snr_stably(snr_db):
    symbols = torch.ones(4, 100_000, dtype=torch.complex64)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        real = torch.randn_like(symbols.real)
        imaginary = torch.randn_like(symbols.real)
    unit_noise = torch.complex(real, imaginary) / (2**0.5)
    channel = AWGN().eval()

    received = channel(symbols, snr_db, unit_noise=unit_noise)
    measured = 10 * torch.log10(
        symbols.abs().square().mean() / (received - symbols).abs().square().mean()
    )

    assert abs(float(measured) - snr_db) < 0.04


def test_awgn_broadcasts_per_sample_snr_and_preserves_gradient_dtype_shape():
    symbols = torch.ones(3, 8192, dtype=torch.complex128, requires_grad=True)
    snr = torch.tensor(
        get("channel.test_snr_grid_db")[:3], dtype=torch.float64
    )
    unit_noise = torch.ones_like(symbols) * complex(1 / 2**0.5, 1 / 2**0.5)

    received = AWGN().eval()(symbols, snr, unit_noise=unit_noise)
    received.abs().square().mean().backward()

    assert received.shape == symbols.shape
    assert received.dtype == symbols.dtype
    assert symbols.grad is not None
    assert torch.isfinite(symbols.grad).all()
    assert torch.count_nonzero(symbols.grad)


def test_awgn_evaluation_requires_deterministic_noise():
    with pytest.raises(RuntimeError, match="evaluation requires"):
        AWGN().eval()(torch.ones(1, 8, dtype=torch.complex64), 0)


@pytest.mark.parametrize(
    ("symbols", "snr", "noise", "message"),
    [
        (torch.ones(1, 8), 0, None, "complex"),
        (
            torch.tensor([[complex(float("inf"), 0)]], dtype=torch.complex64),
            0,
            None,
            "finite",
        ),
        (torch.ones(2, 8, dtype=torch.complex64), torch.zeros(3), None, "shape"),
        (torch.ones(1, 8, dtype=torch.complex64), float("nan"), None, "finite"),
        (
            torch.ones(1, 8, dtype=torch.complex64),
            0,
            torch.ones(1, 7, dtype=torch.complex64),
            "shape",
        ),
    ],
)
def test_awgn_rejects_invalid_inputs(symbols, snr, noise, message):
    channel = AWGN()
    with pytest.raises((TypeError, ValueError), match=message):
        channel(symbols, snr, unit_noise=noise)
