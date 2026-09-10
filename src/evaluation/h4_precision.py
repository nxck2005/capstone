"""AM-97 pointwise H4 precision diagnostics.

The production estimand is deliberately small and explicit: ordinary learned
W8/G-10 trajectories minus the corresponding final ER-9 trajectories in the
three frozen, zipped seed cells, averaged within stable image identity.  This
module is a validation-only diagnostic.  It is not a calibrated H4 power
simulation and it never reads the test split.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import Any

import numpy as np

from config.params import get
from training.deterministic_core import canonical_sha256


EXPECTED_CELLS: tuple[tuple[int, int], ...] = ((0, 0), (1, 1), (2, 2))
ALPHA = 0.025
POWER = 0.8
Z_SUM = NormalDist().inv_cdf(1.0 - ALPHA) + NormalDist().inv_cdf(POWER)


class H4PrecisionError(ValueError):
    """The H4 trajectories do not satisfy the AM-97 contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise H4PrecisionError(message)


def _cell_key(value: Any) -> tuple[int, int]:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])
    if isinstance(value, str):
        for separator in ("/", ":", ",", "_"):
            parts = value.split(separator)
            if len(parts) == 2 and all(part.strip().lstrip("-").isdigit() for part in parts):
                return int(parts[0]), int(parts[1])
    raise H4PrecisionError(f"invalid H4 seed cell: {value!r}")


def _correct(value: Any, *, label: str) -> int:
    if isinstance(value, Mapping):
        if "correct" not in value:
            raise H4PrecisionError(f"{label} has no correctness field")
        value = value["correct"]
    _require(isinstance(value, (bool, int, np.integer)), f"{label} correctness is not binary")
    value = int(value)
    _require(value in (0, 1), f"{label} correctness is not binary")
    return value


def _snr_map(value: Any, grid: tuple[float, ...], *, label: str) -> dict[float, Any]:
    if isinstance(value, Mapping):
        result: dict[float, Any] = {}
        for key, item in value.items():
            try:
                result[float(key)] = item
            except (TypeError, ValueError):
                raise H4PrecisionError(f"{label} has a non-numeric SNR key") from None
        _require(set(result) == set(grid), f"{label} SNR grid differs")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        _require(len(value) == len(grid), f"{label} SNR trajectory length differs")
        return dict(zip(grid, value, strict=True))
    raise H4PrecisionError(f"{label} has no SNR trajectory")


def _cell_major(arm: Any, *, grid: tuple[float, ...], label: str) -> dict[tuple[int, int], dict[str, dict[float, Any]]]:
    _require(isinstance(arm, Mapping), f"{label} must be a mapping")
    cells: dict[tuple[int, int], dict[str, dict[float, Any]]] = {}
    for raw_cell, image_rows in arm.items():
        cell = _cell_key(raw_cell)
        _require(cell not in cells, f"{label} repeats cell {cell}")
        _require(isinstance(image_rows, Mapping), f"{label} cell {cell} is not a mapping")
        rows: dict[str, dict[float, Any]] = {}
        for raw_id, trajectory in image_rows.items():
            stable_id = str(raw_id)
            _require(stable_id not in rows, f"{label} repeats image {stable_id}")
            rows[stable_id] = _snr_map(trajectory, grid, label=f"{label}/{cell}/{stable_id}")
        cells[cell] = rows
    _require(set(cells) == set(EXPECTED_CELLS), f"{label} must contain exactly the three frozen cells")
    return cells


def aggregate_three_cell_differences(
    learned: Mapping[Any, Any],
    er9: Mapping[Any, Any],
    *,
    snr_grid_db: Sequence[float],
    learned_arm: str = "ordinary_learned_w8_g10",
    comparator: str = "er9_digital",
) -> dict[str, Any]:
    """Return the per-image signed three-cell H4 estimand."""

    _require(learned_arm == "ordinary_learned_w8_g10", "H4 learned arm must be ordinary W8/G-10")
    _require(comparator == "er9_digital", "H4 comparator must be final ER-9")
    _require("random" not in learned_arm.lower() and "er2" not in learned_arm.lower(), "randomized ER-2 cannot be H4 arm")
    grid = tuple(float(item) for item in snr_grid_db)
    _require(grid and len(set(grid)) == len(grid), "H4 SNR grid must be non-empty and unique")
    learned_cells = _cell_major(learned, grid=grid, label="learned")
    er9_cells = _cell_major(er9, grid=grid, label="er9")
    _require(set(learned_cells) == set(er9_cells), "H4 cell sets differ")
    stable_ids = set(learned_cells[EXPECTED_CELLS[0]])
    _require(stable_ids, "H4 trajectories contain no stable images")
    for cell in EXPECTED_CELLS:
        _require(set(learned_cells[cell]) == stable_ids, f"learned cell {cell} stable IDs differ")
        _require(set(er9_cells[cell]) == stable_ids, f"ER-9 cell {cell} stable IDs differ")
    ids = tuple(sorted(stable_ids))
    per_cell: list[np.ndarray] = []
    for cell in EXPECTED_CELLS:
        per_cell.append(
            np.asarray(
                [
                    [
                        _correct(learned_cells[cell][stable_id][snr], label=f"learned/{cell}/{stable_id}/{snr}")
                        - _correct(er9_cells[cell][stable_id][snr], label=f"er9/{cell}/{stable_id}/{snr}")
                        for snr in grid
                    ]
                    for stable_id in ids
                ],
                dtype=np.float64,
            )
        )
    values = np.mean(np.stack(per_cell, axis=1), axis=1)
    return {
        "stable_ids": list(ids),
        "snr_grid_db": list(grid),
        "values": values,
        "cell_values": np.stack(per_cell, axis=1),
        "cells": [list(cell) for cell in EXPECTED_CELLS],
        "learned_arm": learned_arm,
        "comparator": comparator,
    }


aggregate_h4_per_image_differences = aggregate_three_cell_differences


def _reference_runs(flags: Sequence[bool]) -> list[int]:
    runs: list[int] = []
    current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        runs.append(current)
    return runs


def _bootstrap_mde(values: np.ndarray, *, repetitions: int, seed_key: str) -> np.ndarray:
    n_images, n_snr = values.shape
    generator = np.random.Generator(np.random.Philox(int(seed_key[:16], 16)))
    result = np.empty((repetitions, n_snr), dtype=np.float64)
    for offset in range(0, repetitions, 256):
        count = min(256, repetitions - offset)
        indices = generator.integers(0, n_images, size=(count, n_images), endpoint=False)
        sampled = values[indices, :]
        variance = np.var(sampled, axis=1, ddof=1 if n_images > 1 else 0) / n_images
        result[offset : offset + count] = Z_SUM * np.sqrt(np.maximum(variance, 0.0))
    return result


def _pointwise_diagnostic(
    aggregate: Mapping[str, Any],
    *,
    region: Sequence[float],
    repetitions: int,
    reference_pp: float,
) -> dict[str, Any]:
    grid = tuple(float(item) for item in aggregate["snr_grid_db"])
    region_set = {float(item) for item in region}
    _require(region_set <= set(grid), "H4 diagnostic region is outside the supplied grid")
    positions = [index for index, snr in enumerate(grid) if snr in region_set]
    _require(positions, "H4 diagnostic region is empty")
    values = np.asarray(aggregate["values"], dtype=np.float64)[:, positions]
    key_body = {
        "stable_ids": list(aggregate["stable_ids"]),
        "grid": list(grid),
        "region": [grid[index] for index in positions],
        "cells": aggregate["cells"],
        "values_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "repetitions": int(repetitions),
    }
    seed_key = canonical_sha256(key_body)
    bootstrap = _bootstrap_mde(values, repetitions=repetitions, seed_key=seed_key)
    q = {name: np.quantile(bootstrap, probability, axis=0) for name, probability in (("q50", 0.50), ("q95", 0.95), ("q99", 0.99))}
    q_pp = {name: (array * 100.0).tolist() for name, array in q.items()}
    flags = [float(value) <= reference_pp for value in q_pp["q95"]]
    snr_points = []
    for column, position in enumerate(positions):
        snr_points.append(
            {
                "snr_db": grid[position],
                "observed_mean_difference": float(np.mean(values[:, column])),
                "q50_mde_pp": float(q_pp["q50"][column]),
                "q95_mde_pp": float(q_pp["q95"][column]),
                "q99_mde_pp": float(q_pp["q99"][column]),
                "reference_pp": float(reference_pp),
                "reference_met": bool(flags[column]),
            }
        )
    return {
        "schema_version": 2,
        "artifact_role": "G11_H4_POINTWISE_PRECISION_DIAGNOSTIC",
        "diagnostic_role": "pointwise_precision_diagnostic_only",
        "status": "pointwise_precision_diagnostic_complete",
        "method": "stable_image_bootstrap_of_three_cell_within_image_signed_correctness_difference",
        "learned_arm": aggregate["learned_arm"],
        "comparator": aggregate["comparator"],
        "randomized_er2_as_h4_arm": False,
        "cells": aggregate["cells"],
        "seed_aggregation": "mean_signed_difference_over_three_zipped_cells_within_stable_image",
        "bootstrap_unit": "stable_image_complete_three_cell_trajectory",
        "bootstrap_resamples": int(repetitions),
        "bootstrap_seed": seed_key,
        "bootstrap_input_order_sha256": hashlib.sha256("\n".join(aggregate["stable_ids"]).encode()).hexdigest(),
        "evaluation_region_snr_db": [grid[index] for index in positions],
        "alpha": ALPHA,
        "power_reference": POWER,
        "z_sum": Z_SUM,
        "reference_pp": float(reference_pp),
        "points": snr_points,
        "reference_runs": _reference_runs(flags),
        "pointwise_reference_status": "pointwise_reference_met" if all(flags) else "pointwise_reference_not_met",
        "interpretation": {
            "pointwise_only": True,
            "full_h4_decision_procedure_power_certified": False,
            "h4_run_calibration_represented": False,
            "correlated_snr_pointwise_probabilities_may_be_multiplied": False,
            "negative_h4_result_excludes_meaningful_advantage": False,
            "negative_h4_conclusion": "conservative",
        },
        "test_data_read": False,
        "test_access": 0,
    }


def compute_h4_precision_diagnostic(
    learned: Mapping[Any, Any],
    er9: Mapping[Any, Any],
    *,
    snr_grid_db: Sequence[float] | None = None,
    region_snr_db: Sequence[float] | None = None,
    bootstrap_resamples: int | None = None,
    reference_pp: float | None = None,
    learned_arm: str = "ordinary_learned_w8_g10",
    comparator: str = "er9_digital",
) -> dict[str, Any]:
    grid = tuple(float(item) for item in (snr_grid_db or get("evaluation.snr_grid_db")))
    region = tuple(float(item) for item in (region_snr_db or [item for item in grid if item <= float(get("evaluation.train_snr_db_fixed"))]))
    repetitions = int(bootstrap_resamples if bootstrap_resamples is not None else get("evaluation.h4_precision_bootstrap_resamples"))
    reference = float(reference_pp if reference_pp is not None else get("evaluation.h4_precision_reference_pp"))
    aggregate = aggregate_three_cell_differences(learned, er9, snr_grid_db=grid, learned_arm=learned_arm, comparator=comparator)
    return _pointwise_diagnostic(aggregate, region=region, repetitions=repetitions, reference_pp=reference)


def zero_difference_mde(discordance_rate: float, sample_size: int, *, alpha: float = ALPHA, power: float = POWER) -> float:
    """Analytical one-cell signed-difference sanity check, in proportions."""

    _require(0.0 <= discordance_rate <= 1.0, "discordance rate must be in [0, 1]")
    _require(sample_size > 0, "sample size must be positive")
    z = NormalDist().inv_cdf(1.0 - alpha) + NormalDist().inv_cdf(power)
    return z * np.sqrt(discordance_rate / sample_size)


def simulate_h4_precision(
    discordance: Sequence[int | bool],
    *,
    sample_size: int,
    repetitions: int = 10_000,
) -> dict[str, Any]:
    """Retain the old API as a deterministic one-cell analytical fixture."""

    values = np.asarray([int(value) for value in discordance], dtype=np.uint8)
    _require(values.size > 0 and np.all((values == 0) | (values == 1)), "H4 discordance must be binary and non-empty")
    _require(sample_size > 0 and repetitions > 0, "H4 sizes must be positive")
    rate = float(values.mean())
    mde = float(zero_difference_mde(rate, sample_size))
    quantiles = {"q50": mde, "q95": mde, "q99": mde}
    return {
        "schema_version": 2,
        "artifact_role": "G11_H4_POINTWISE_PRECISION_SANITY_FIXTURE",
        "diagnostic_role": "pointwise_precision_diagnostic_only",
        "status": "pointwise_precision_diagnostic_complete",
        "method": "one_cell_zero_difference_analytical_sanity_check",
        "repetitions": int(repetitions),
        "inputs": {"validation_discordance_count": int(values.size), "validation_discordance_rate": rate, "future_sample_size": int(sample_size), "discordance_sha256": hashlib.sha256(values.tobytes()).hexdigest()},
        "assumptions": {"alpha": ALPHA, "power": POWER, "paired_variance": "p_discordance_over_n_for_signed_zero_mean_one_cell_differences", "production_estimator": "three_cell_within_image_aggregation", "test_data_read": False},
        "quantiles": quantiles,
        "mde_percentage_points": {name: value * 100.0 for name, value in quantiles.items()},
        "pointwise_reference_status": "pointwise_reference_met" if mde * 100.0 <= 2.0 else "pointwise_reference_not_met",
        "interpretation": {"pointwise_only": True, "full_h4_decision_procedure_power_certified": False, "h4_run_calibration_represented": False, "negative_h4_result_excludes_meaningful_advantage": False, "negative_h4_conclusion": "conservative"},
        "test_access": 0,
    }


__all__ = [
    "EXPECTED_CELLS",
    "H4PrecisionError",
    "aggregate_three_cell_differences",
    "aggregate_h4_per_image_differences",
    "compute_h4_precision_diagnostic",
    "simulate_h4_precision",
    "zero_difference_mde",
]
