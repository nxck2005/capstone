"""Prospective validation-only precision simulation for G-11 H4."""

from __future__ import annotations

import hashlib
from statistics import NormalDist
from typing import Any, Sequence

import numpy as np

from config.params import get
from training.deterministic_core import canonical_sha256


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def simulate_h4_precision(
    discordance: Sequence[int | bool],
    *,
    sample_size: int,
    repetitions: int = 10_000,  # literal-ok: configured prospective simulation replication count
) -> dict[str, Any]:
    """Estimate the paired MDE using validation-derived discordance only.

    The simulation samples the number of discordant image identities at the
    future test size and applies the preregistered one-sided 2.5%/80% power
    calculation.  It never consumes test rows.  ``2`` percentage points is the
    upper edge of the residual range described in the existing H4 risk note;
    the gate decision is therefore fixed as ``p95 MDE <= 2 pp``.
    """

    values = np.asarray([int(bool(value)) for value in discordance], dtype=np.uint8)
    _require(values.size > 0, "H4 precision requires validation discordance")
    _require(sample_size > 0 and repetitions > 0, "H4 precision sizes must be positive")
    _require(np.all((values == 0) | (values == 1)), "H4 discordance must be binary")
    p = float(values.mean())
    alpha = 0.025  # literal-ok: H4 one-sided 2.5% power rule
    power = 0.8  # literal-ok: H4 80% power rule
    z_critical = NormalDist().inv_cdf(1.0 - alpha)  # literal-ok: upper-tail critical value
    z_power = NormalDist().inv_cdf(power)
    key_body = {
        "purpose": "h4_mde_simulation",
        "discordance_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "validation_count": int(values.size),
        "sample_size": int(sample_size),
        "repetitions": int(repetitions),
        "alpha": alpha,
        "power": power,
    }
    key = canonical_sha256(key_body)
    generator = np.random.Generator(
        np.random.Philox(int(key[:16], 16))  # literal-ok: keyed Philox seed truncation
    )
    mdes = np.empty(repetitions, dtype=np.float64)
    for offset in range(0, repetitions, 1000):  # literal-ok: bounded simulation chunk
        count = min(1000, repetitions - offset)  # literal-ok: bounded simulation chunk
        discordant_counts = generator.binomial(sample_size, p, size=count)
        estimated = discordant_counts / sample_size
        variance = np.maximum(estimated * (1.0 - estimated), 0.0) / sample_size  # literal-ok: paired Bernoulli variance
        mdes[offset : offset + count] = (z_critical + z_power) * np.sqrt(variance)
    quantiles = {
        "q50": float(np.quantile(mdes, 0.50)),  # literal-ok: fixed simulation summary quantile
        "q95": float(np.quantile(mdes, 0.95)),  # literal-ok: fixed simulation summary quantile
        "q99": float(np.quantile(mdes, 0.99)),  # literal-ok: fixed simulation summary quantile
    }
    residual_limit = 0.02  # literal-ok: existing H4 risk note's 2 percentage-point upper residual
    decision = "full_strength" if quantiles["q95"] <= residual_limit else "narrowed"
    return {
        "schema_version": 1,  # literal-ok: G-11 H4 simulation schema
        "artifact_role": "G11_H4_VALIDATION_PRECISION_SIMULATION",
        "method": str(get("evaluation.h4_mde_simulation")),
        "gate": str(get("evaluation.h4_mde_gate")),
        "inputs": {
            "validation_discordance_count": int(values.size),
            "validation_discordance_rate": p,
            "future_sample_size": int(sample_size),
            "discordance_sha256": key_body["discordance_sha256"],
        },
        "assumptions": {
            "tail": "one_sided_upper",
            "alpha": alpha,
            "power": power,
            "paired_variance": "p_discordance_times_one_minus_p_over_n",
            "plausible_residual_upper_bound": residual_limit,
            "test_data_read": False,
        },
        "repetitions": int(repetitions),
        "random_key": key,
        "quantiles": quantiles,
        "mde_percentage_points": {
            name: value * 100.0  # literal-ok: percentage-point presentation conversion
            for name, value in quantiles.items()
        },
        "gate_rule": "full_strength_if_validation_derived_p95_mde_at_future_sample_size_is_at_most_2pp",
        "gate_decision": decision,
        "test_access": 0,
    }


__all__ = ["simulate_h4_precision"]
