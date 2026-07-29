"""Generic symbol-domain PAPR and capped-power projection."""

from __future__ import annotations

import pytest
import torch

from channels.power import PeakPowerConstraint, symbol_papr_db


def test_constant_modulus_symbols_measure_zero_db():
    phases = torch.linspace(0, 6, 1024)
    symbols = torch.polar(torch.ones_like(phases), phases).unsqueeze(0)

    torch.testing.assert_close(symbol_papr_db(symbols), torch.zeros(1), atol=1e-6, rtol=0)


def test_peak_constraint_projects_peaky_vector_with_unit_power_and_bound():
    symbols = torch.tensor(
        [[8 + 0j, 0.2j, -0.1 + 0j, -0.3j, 0.4 + 0j, 0.1j]],
        dtype=torch.complex64,
        requires_grad=True,
    )
    constraint = PeakPowerConstraint(3.0)

    projected = constraint(symbols)
    projected.abs().sum().backward()

    assert float(symbol_papr_db(projected).detach()) <= 3.0 + constraint.bound_tolerance_db
    torch.testing.assert_close(
        projected.abs().square().mean(dim=1),
        torch.ones(1),
        atol=constraint.power_tolerance,
        rtol=0,
    )
    assert symbols.grad is not None
    assert torch.isfinite(symbols.grad).all()
    assert torch.count_nonzero(symbols.grad)
    phase_delta = torch.angle(projected) - torch.angle(symbols)
    torch.testing.assert_close(phase_delta, torch.zeros_like(phase_delta), atol=1e-6, rtol=0)


def test_peak_constraint_leaves_constant_modulus_input_unchanged():
    symbols = torch.tensor([[1, 1j, -1, -1j]], dtype=torch.complex64)
    projected = PeakPowerConstraint(0)(symbols)

    torch.testing.assert_close(projected, symbols)
    torch.testing.assert_close(symbol_papr_db(projected), torch.zeros(1))


@pytest.mark.parametrize("bound", [-0.01, float("nan"), float("inf")])
def test_peak_constraint_rejects_invalid_bound(bound):
    with pytest.raises((ValueError, TypeError), match="finite|floor"):
        PeakPowerConstraint(bound)


def test_peak_constraint_rejects_infeasible_support():
    symbols = torch.tensor([[4, 0, 0, 0]], dtype=torch.complex64)
    with pytest.raises(ValueError, match="support.*infeasible"):
        PeakPowerConstraint(3.0)(symbols)


@pytest.mark.parametrize(
    "symbols",
    [
        torch.zeros(1, 8, dtype=torch.complex64),
        torch.tensor([[complex(float("nan"), 0)]], dtype=torch.complex64),
    ],
)
def test_papr_rejects_zero_or_nonfinite_symbols(symbols):
    with pytest.raises(ValueError, match="finite|non-zero"):
        symbol_papr_db(symbols)
