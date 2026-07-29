"""Per-image symbol power, symbol-domain PAPR, and peak projection."""

from __future__ import annotations

import math

import torch
from torch import nn

_DB_POWER_BASE = 10.0  # literal-ok: definition of power decibels, not a project choice
_PROJECTION_STEPS = 64  # literal-ok: numerical convergence depth, not an experiment setting
_BOUND_TOLERANCE_DB = 1e-4
_POWER_TOLERANCE = 1e-6


def _symbol_powers(symbols: torch.Tensor) -> torch.Tensor:
    if not isinstance(symbols, torch.Tensor):
        raise TypeError("symbols must be a torch.Tensor")
    if symbols.ndim != 2 or symbols.shape[1] == 0:
        raise ValueError("symbols must have complex-symbol shape [B, k] with k > 0")
    if not symbols.is_complex():
        raise TypeError("symbols must use a native complex PyTorch dtype")
    if not torch.isfinite(symbols).all():
        raise ValueError("symbols must contain only finite values")
    powers = symbols.abs().square()
    mean_power = powers.mean(dim=1)
    if not torch.isfinite(mean_power).all() or torch.any(mean_power <= 0):
        raise ValueError("every sample must have finite, non-zero symbol power")
    return powers


def normalize_unit_average_power(symbols: torch.Tensor) -> torch.Tensor:
    """Normalise every image independently to ``mean_k |x|² = 1``."""

    powers = _symbol_powers(symbols)
    scale = torch.rsqrt(powers.mean(dim=1)).unsqueeze(1)
    return symbols * scale


def symbol_papr_db(symbols: torch.Tensor) -> torch.Tensor:
    """Return symbol-domain PAPR in dB, one value per image.

    This is not oversampled waveform PAPR after pulse shaping.
    """

    powers = _symbol_powers(symbols)
    papr_linear = powers.amax(dim=1) / powers.mean(dim=1)
    return _DB_POWER_BASE * torch.log10(papr_linear)


class PeakPowerConstraint(nn.Module):
    """Project symbols onto unit mean power with a requested PAPR cap.

    The projection preserves phase and solves for a common power scaling under
    a hard cap by bisection. Unlike clip-then-renormalise-once, the resulting
    point remains inside both the unit-power surface and the peak bound.
    Numerical contract: mean power is within ``1e-6`` and PAPR is no more than
    ``max_papr_db + 1e-4 dB`` for float32 or higher precision inputs.
    """

    bound_tolerance_db = _BOUND_TOLERANCE_DB
    power_tolerance = _POWER_TOLERANCE

    def __init__(self, max_papr_db: float) -> None:
        super().__init__()
        if isinstance(max_papr_db, bool) or not isinstance(max_papr_db, int | float):
            raise TypeError("max_papr_db must be a finite numeric scalar")
        value = float(max_papr_db)
        if not math.isfinite(value):
            raise ValueError("max_papr_db must be finite")
        if value < 0:
            raise ValueError("max_papr_db cannot be below the physical 0 dB floor")
        self.max_papr_db = value
        self.max_power = _DB_POWER_BASE ** (value / _DB_POWER_BASE)

    def forward(self, symbols: torch.Tensor) -> torch.Tensor:
        powers = _symbol_powers(symbols)
        k = symbols.shape[1]
        support = powers > 0
        support_count = support.sum(dim=1)
        feasible_capacity = support_count.to(torch.float64) * self.max_power
        if torch.any(feasible_capacity < k):
            raise ValueError(
                "input support makes the unit-power/PAPR intersection infeasible"
            )

        detached = powers.detach().to(torch.float64)
        cap = torch.as_tensor(
            self.max_power, dtype=torch.float64, device=symbols.device
        )
        low = torch.zeros((symbols.shape[0], 1), dtype=torch.float64, device=symbols.device)
        high = torch.ones_like(low)
        for _ in range(_PROJECTION_STEPS):
            achieved = torch.minimum(high * detached, cap).mean(dim=1, keepdim=True)
            high = torch.where(achieved < 1, high * 2, high)
        if torch.any(torch.minimum(high * detached, cap).mean(dim=1) < 1):
            raise ValueError(
                "input support makes the unit-power/PAPR intersection infeasible"
            )
        for _ in range(_PROJECTION_STEPS):
            midpoint = (low + high) / 2
            achieved = torch.minimum(midpoint * detached, cap).mean(
                dim=1, keepdim=True
            )
            low = torch.where(achieved < 1, midpoint, low)
            high = torch.where(achieved >= 1, midpoint, high)

        scale = high.to(powers.dtype)
        projected_powers = torch.minimum(scale * powers, powers.new_tensor(self.max_power))
        amplitude_scale = torch.zeros_like(projected_powers)
        amplitude_scale[support] = torch.sqrt(
            projected_powers[support] / powers[support]
        )
        projected = symbols * amplitude_scale

        mean_power = projected.abs().square().mean(dim=1)
        if torch.any(torch.abs(mean_power - 1) > self.power_tolerance):
            raise RuntimeError("peak-power projection did not reach unit average power")
        measured = symbol_papr_db(projected)
        if torch.any(measured > self.max_papr_db + self.bound_tolerance_db):
            raise RuntimeError("peak-power projection violated its requested PAPR bound")
        return projected
