"""Differentiable complex AWGN under the configured Es/N0 convention (SR-7)."""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt

import numpy as np
import torch
from torch import nn

from artifacts.rng import keyed_standard_normal
from config.params import get

_DB_POWER_BASE = 10.0  # literal-ok: definition of power decibels, not a project choice
_STANDARD_COMPLEX_SCALE = sqrt(0.5)  # literal-ok: CN(0,1) component variance


def _validate_complex_symbols(symbols: torch.Tensor, *, label: str) -> None:
    if not isinstance(symbols, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if symbols.ndim != 2 or symbols.shape[1] == 0:
        raise ValueError(f"{label} must have complex-symbol shape [B, k] with k > 0")
    if not symbols.is_complex():
        raise TypeError(f"{label} must use a native complex PyTorch dtype")
    if not torch.isfinite(symbols).all():
        raise ValueError(f"{label} must contain only finite values")


def keyed_complex_noise(
    noise_ids: str | Sequence[str],
    k: int,
    *,
    dtype: torch.dtype = torch.complex64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Construct unit-standard complex noise as a pure function of ``noise_id``.

    Each row is generated from the declared ``channel_noise`` Philox identity
    independently, so ordering, batching, prior draws, system identity, and
    surrounding control flow cannot affect it.
    """

    if get("artifacts.rng_identity_fields.channel_noise") != ["noise_id"]:
        raise RuntimeError("channel_noise RNG identity is no longer exactly noise_id")
    if isinstance(noise_ids, str):
        ids = (noise_ids,)
    else:
        ids = tuple(noise_ids)
    if not ids or any(not isinstance(noise_id, str) or not noise_id for noise_id in ids):
        raise ValueError("noise_ids must contain one or more non-empty strings")
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")
    if dtype not in (torch.complex64, torch.complex128):
        raise TypeError("keyed complex noise dtype must be complex64 or complex128")

    rows: list[np.ndarray] = []
    for noise_id in ids:
        components = keyed_standard_normal(
            "channel_noise",
            {"noise_id": noise_id},
            size=(2, k),
        )
        rows.append((components[0] + 1j * components[1]) * _STANDARD_COMPLEX_SCALE)
    array = np.stack(rows)
    return torch.as_tensor(array, dtype=dtype, device=device)


class AWGN(nn.Module):
    """Complex AWGN with ``E|n|² = 1/gamma`` after unit-power normalisation."""

    def __init__(self) -> None:
        super().__init__()
        definition = get("channel.snr_definition")
        if definition != (
            "Es/N0 in dB per complex channel use, measured after "
            "unit-average-power normalisation"
        ):
            raise NotImplementedError(f"unsupported SNR convention: {definition}")

    def forward(
        self,
        symbols: torch.Tensor,
        snr_db: float | torch.Tensor,
        *,
        unit_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _validate_complex_symbols(symbols, label="symbols")
        real_dtype = symbols.real.dtype
        try:
            snr = torch.as_tensor(snr_db, dtype=real_dtype, device=symbols.device)
        except (TypeError, ValueError):
            raise TypeError("snr_db must be a numeric scalar or per-sample tensor") from None
        if snr.ndim == 0:
            snr = snr.expand(symbols.shape[0])
        elif snr.ndim != 1 or snr.shape[0] != symbols.shape[0]:
            raise ValueError("snr_db must be scalar or have shape [B]")
        if not torch.isfinite(snr).all():
            raise ValueError("snr_db must contain only finite values")

        if unit_noise is None:
            if not self.training:
                raise RuntimeError(
                    "AWGN evaluation requires externally supplied deterministic unit_noise"
                )
            real = torch.randn_like(symbols.real)
            imaginary = torch.randn_like(symbols.real)
            unit_noise = torch.complex(real, imaginary) * _STANDARD_COMPLEX_SCALE
        else:
            _validate_complex_symbols(unit_noise, label="unit_noise")
            if unit_noise.shape != symbols.shape:
                raise ValueError(
                    "unit_noise shape must exactly match the channel-symbol shape"
                )
            if unit_noise.dtype != symbols.dtype or unit_noise.device != symbols.device:
                raise ValueError(
                    "unit_noise dtype and device must match the channel symbols"
                )

        gamma = torch.pow(
            torch.as_tensor(_DB_POWER_BASE, dtype=real_dtype, device=symbols.device),
            snr / _DB_POWER_BASE,
        )
        noise_scale = torch.rsqrt(gamma).unsqueeze(1)
        return symbols + unit_noise * noise_scale
