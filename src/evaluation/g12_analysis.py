"""ER-10 paired inference for §2's H1–H4, frozen into the G-12 manifest before test.

Every hypothesis reads one estimand (AM-57): for image ``i``, cell ``j``, SNR
``s`` and comparator ``q``, ``d(i,j,s,q) = learned − q`` on per-image
correctness; ``d̄(i,s,q)`` averages the cells first; ``Δ(s,q)`` averages images.
All intervals come from ONE bootstrap of stable images carrying their complete
system × SNR × cell trajectories (``params.evaluation.bootstrap_resample_scope``).

Choices the specification leaves to the implementation are fixed here, before
any test read, and recorded in ``ANALYSIS_CHOICES``:

* intervals are percentile intervals of the shared image bootstrap;
* a studentized point with ``σ̂(s) = 0`` qualifies only if ``Δ̂(s) > 0``;
* H1/H4's calibration applies Rademacher multipliers to centred cell-averaged
  image trajectories and re-studentizes each resample exactly;
* H3's weighted least squares uses inverse-variance weights ``N / σ̂²(s)`` from
  the observed data, held fixed across resamples (a zero variance takes the
  smallest positive variance on the grid);
* H2's drops are ``accuracy(high endpoint) − accuracy(low endpoint)``;
* the bootstrap and calibration draws use a fixed keyed seed.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from config.params import get

ANALYSIS_IMPLEMENTATION = "g12_analysis_v1"
BOOTSTRAP_SEED_KEY = "g12-er10-image-bootstrap-v1"
CALIBRATION_SEED_KEY = "g12-er10-h1-h4-rademacher-v1"
QUALIFIER_THRESHOLD = 1.96  # literal-ok: §2's studentized qualifier is the two-sided 95% normal quantile
ONE_SIDED_LEVEL = 0.95  # literal-ok: §2's one-sided 95% bounds for H2 and H3
TWO_SIDED_LEVEL = 0.95  # literal-ok: params.evaluation.ci is a paired bootstrap 95% interval
MIN_RUN = 3  # literal-ok: §2's three consecutive qualified points
PERCENT = 100.0  # literal-ok: accuracy fractions reported in percentage points

ANALYSIS_CHOICES = {
    "implementation": ANALYSIS_IMPLEMENTATION,
    "interval_method": "percentile_of_one_shared_stable_image_bootstrap",
    "zero_variance_qualifier": "qualifies_iff_mean_positive",
    "h1_h4_calibration": "rademacher_multipliers_on_centred_cell_averaged_image_trajectories_restudentized",
    "h1_h4_p_value": "(1 + #{R_b >= R_obs}) / (B + 1)",
    "h3_wls_weights": "inverse_variance_N_over_sigma2_observed_fixed_across_resamples",
    "h2_drop_definition": "accuracy_at_high_endpoint_minus_accuracy_at_low_endpoint",
    "bootstrap_seed_key": BOOTSTRAP_SEED_KEY,
    "calibration_seed_key": CALIBRATION_SEED_KEY,
}


def _seed(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")  # literal-ok: 64-bit seed from the digest


@dataclass(frozen=True)
class Outcomes:
    """Per-image correctness, one array per system: shape (images, cells, SNRs)."""

    stable_ids: tuple[str, ...]
    snrs: tuple[float, ...]
    cells: tuple[int, ...]
    correct: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        shape = (len(self.stable_ids), len(self.cells), len(self.snrs))
        for system, values in self.correct.items():
            if values.shape != shape:
                raise ValueError(f"{system} outcomes have shape {values.shape}, expected {shape}")
            if not np.isin(values, (0, 1)).all():
                raise ValueError(f"{system} outcomes are not 0/1 correctness")

    def snr_index(self, snr: float) -> int:
        return self.snrs.index(float(snr))


def outcomes_from_rows(rows: Sequence[Mapping[str, Any]], *, systems: Mapping[str, tuple[str, str]], cells: Sequence[int]) -> Outcomes:
    """Build ``Outcomes`` from SR-18 rows; joins are one-to-one or this raises.

    ``systems`` maps an analysis name to ``(system, bw_ratio)`` row filters, and
    each row must carry ``train_seed`` and the scorer stream it came from
    already selected by the caller.
    """

    grouped: dict[str, dict[tuple[str, int, float], int]] = {name: {} for name in systems}
    lookup = {value: name for name, value in systems.items()}
    for row in rows:
        name = lookup.get((str(row["system"]), str(row["bw_ratio"])))
        if name is None:
            continue
        key = (str(row["stable_sample_id"]), int(row["train_seed"]), float(row["test_snr_db"]))
        if key in grouped[name]:
            raise ValueError(f"duplicated trajectory row for {name}: {key}")
        grouped[name][key] = int(bool(row["correct"]))
    first = next(iter(grouped.values()))
    stable_ids = tuple(sorted({key[0] for key in first}))
    snrs = tuple(sorted({key[2] for key in first}))
    cell_tuple = tuple(int(cell) for cell in cells)
    correct = {}
    for name, values in grouped.items():
        expected = len(stable_ids) * len(cell_tuple) * len(snrs)
        if len(values) != expected:
            raise ValueError(f"{name} has {len(values)} rows, expected a complete {expected}-row trajectory set")
        array = np.empty((len(stable_ids), len(cell_tuple), len(snrs)), dtype=np.int8)
        for i, stable_id in enumerate(stable_ids):
            for j, cell in enumerate(cell_tuple):
                for s, snr in enumerate(snrs):
                    try:
                        array[i, j, s] = values[(stable_id, cell, snr)]
                    except KeyError:
                        raise ValueError(f"{name} is missing {(stable_id, cell, snr)}") from None
        correct[name] = array
    return Outcomes(stable_ids, snrs, cell_tuple, correct)


class SharedBootstrap:
    """One image resampling, reused by every interval (drawn once)."""

    def __init__(self, n_images: int, resamples: int) -> None:
        rng = np.random.default_rng(_seed(BOOTSTRAP_SEED_KEY))
        # Multinomial counts give the same resample as index draws, as a matrix.
        self.counts = rng.multinomial(n_images, np.full(n_images, 1.0 / n_images), size=resamples).astype(np.float64)
        self.n = n_images

    def means(self, per_image: np.ndarray) -> np.ndarray:
        """Resampled means of a per-image statistic, shape (resamples, ...)."""

        flat = per_image.reshape(self.n, -1).astype(np.float64)
        return (self.counts @ flat / self.n).reshape((self.counts.shape[0],) + per_image.shape[1:])


def _interval(samples: np.ndarray, level: float = TWO_SIDED_LEVEL) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0
    return float(np.quantile(samples, tail)), float(np.quantile(samples, 1.0 - tail))


def _studentized(dbar: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = dbar.shape[0]
    mean = dbar.mean(axis=0)
    sd = dbar.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(sd > 0, math.sqrt(n) * mean / np.where(sd > 0, sd, 1.0), np.where(mean > 0, np.inf, 0.0))
    return mean, sd, t


def longest_run(qualified: np.ndarray) -> np.ndarray:
    """Longest run of consecutive True values along the last axis."""

    qualified = np.asarray(qualified, dtype=bool)
    best = np.zeros(qualified.shape[:-1], dtype=np.int64)
    current = np.zeros_like(best)
    for s in range(qualified.shape[-1]):
        current = np.where(qualified[..., s], current + 1, 0)
        best = np.maximum(best, current)
    return best


def run_rule(dbar_region: np.ndarray, *, resamples: int) -> dict[str, Any]:
    """§2's H1 rule on cell-averaged differences restricted to the region."""

    n = dbar_region.shape[0]
    mean, sd, t = _studentized(dbar_region)
    qualified = t > QUALIFIER_THRESHOLD
    r_obs = int(longest_run(qualified[None, :])[0])
    centred = dbar_region - mean
    sum_squares = (centred**2).sum(axis=0)
    rng = np.random.default_rng(_seed(CALIBRATION_SEED_KEY))
    exceed = 0
    batch = 500  # literal-ok: memory chunk for the multiplier draws
    for start in range(0, resamples, batch):
        size = min(batch, resamples - start)
        weights = rng.integers(0, 2, size=(size, n)).astype(np.float64) * 2.0 - 1.0
        star_mean = weights @ centred / n
        # (w c)^2 = c^2, so the resampled variance has this exact closed form.
        star_var = (sum_squares[None, :] - n * star_mean**2) / (n - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            star_t = np.where(star_var > 0, math.sqrt(n) * star_mean / np.sqrt(np.where(star_var > 0, star_var, 1.0)), np.where(star_mean > 0, np.inf, 0.0))
        runs = longest_run(star_t > QUALIFIER_THRESHOLD)
        exceed += int((runs >= r_obs).sum())
    p_value = (1 + exceed) / (resamples + 1)
    return {
        "per_point_mean": mean.tolist(),
        "per_point_sd": sd.tolist(),
        "per_point_t": [float(value) for value in t],
        "qualified": qualified.tolist(),
        "r_obs": r_obs,
        "calibrated_p": float(p_value),
        "supported": bool(r_obs >= MIN_RUN and p_value <= float(get("evaluation.h1_run_calibration_alpha"))),
    }


def _cell_mean(outcomes: Outcomes, learned: str, comparator: str) -> np.ndarray:
    return (outcomes.correct[learned].astype(np.float64) - outcomes.correct[comparator].astype(np.float64)).mean(axis=1)


def h1_like(outcomes: Outcomes, boot: SharedBootstrap, *, learned: str, comparator: str, hypothesis: str) -> dict[str, Any]:
    train_snr = float(get("channel.train_snr_db_fixed"))
    region = [index for index, snr in enumerate(outcomes.snrs) if snr <= train_snr]
    dbar = _cell_mean(outcomes, learned, comparator)
    rule = run_rule(dbar[:, region], resamples=int(get("evaluation.h1_run_permutation_resamples")))
    effect_per_image = dbar[:, region].mean(axis=1)
    effect = float(effect_per_image.mean())
    effect_low, effect_high = _interval(boot.means(effect_per_image))
    per_snr = boot.means(dbar)
    points = []
    for index, snr in enumerate(outcomes.snrs):
        low, high = _interval(per_snr[:, index])
        points.append({"snr_db": snr, "delta": float(dbar[:, index].mean()), "ci_low": low, "ci_high": high})
    return {
        "hypothesis": hypothesis,
        "learned": learned,
        "comparator": comparator,
        "region_snr_db": [outcomes.snrs[index] for index in region],
        **rule,
        "effect_size": effect,
        "effect_ci": [effect_low, effect_high],
        "per_snr": points,
    }


def h2(outcomes: Outcomes, boot: SharedBootstrap, *, learned: str, comparator: str, low_snr: float, high_snr: float) -> dict[str, Any]:
    lo, hi = outcomes.snr_index(low_snr), outcomes.snr_index(high_snr)

    def drop(system: str) -> np.ndarray:
        values = outcomes.correct[system].astype(np.float64).mean(axis=1)
        return values[:, hi] - values[:, lo]

    drop_c, drop_l = drop(comparator), drop(learned)
    star_c, star_l = boot.means(drop_c), boot.means(drop_l)
    lower_c = float(np.quantile(star_c, 1.0 - ONE_SIDED_LEVEL))
    upper_l = float(np.quantile(star_l, ONE_SIDED_LEVEL))
    did = drop_c - drop_l
    did_low, did_high = _interval(boot.means(did))
    cliff = float(get("evaluation.cliff_drop_pp")) / PERCENT
    graceful = float(get("evaluation.graceful_drop_pp")) / PERCENT
    return {
        "hypothesis": "H2",
        "window_snr_db": [low_snr, high_snr],
        "classical_drop": float(drop_c.mean()),
        "classical_drop_one_sided_lower": lower_c,
        "learned_drop": float(drop_l.mean()),
        "learned_drop_one_sided_upper": upper_l,
        "difference_in_differences": float(did.mean()),
        "difference_in_differences_ci": [did_low, did_high],
        "classical_margin_met": lower_c >= cliff,
        "learned_margin_met": upper_l <= graceful,
        "supported": bool(lower_c >= cliff and upper_l <= graceful),
    }


def h3(outcomes: Outcomes, boot: SharedBootstrap, *, learned: str, comparator: str) -> dict[str, Any]:
    dbar = _cell_mean(outcomes, learned, comparator)
    n = dbar.shape[0]
    snr = np.asarray(outcomes.snrs, dtype=np.float64)
    variance = dbar.var(axis=0, ddof=1)
    positive = variance[variance > 0]
    floor = float(positive.min()) if positive.size else 1.0
    weights = n / np.where(variance > 0, variance, floor)
    centre = (weights * snr).sum() / weights.sum()
    lever = weights * (snr - centre) / (weights * (snr - centre) ** 2).sum()

    def slope(delta: np.ndarray) -> np.ndarray:
        return delta @ lever

    observed = dbar.mean(axis=0)
    star = boot.means(dbar)
    star_slope = slope(star)
    lo, hi = outcomes.snr_index(float(get("evaluation.h3_low_snr_point_db"))), outcomes.snr_index(float(get("evaluation.h3_high_snr_point_db")))
    contraction = np.abs(star[:, lo]) - np.abs(star[:, hi])
    slope_upper = float(np.quantile(star_slope, ONE_SIDED_LEVEL))
    low_gap_lower = float(np.quantile(star[:, lo], 1.0 - ONE_SIDED_LEVEL))
    contraction_lower = float(np.quantile(contraction, 1.0 - ONE_SIDED_LEVEL))
    clause_i, clause_ii, clause_iii = slope_upper < 0, low_gap_lower > 0, contraction_lower > 0
    return {
        "hypothesis": "H3",
        "slope_per_db": float(slope(observed)),
        "slope_ci": list(_interval(star_slope)),
        "slope_one_sided_upper": slope_upper,
        "low_gap": float(observed[lo]),
        "low_gap_one_sided_lower": low_gap_lower,
        "high_gap": float(observed[hi]),
        "contraction": float(abs(observed[lo]) - abs(observed[hi])),
        "contraction_one_sided_lower": contraction_lower,
        "clauses": {"negative_slope": bool(clause_i), "positive_low_gap": bool(clause_ii), "magnitude_contraction": bool(clause_iii)},
        "supported": bool(clause_i and clause_ii and clause_iii),
    }


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p-value (descriptive only, within one cell)."""

    total = a_only + b_only
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, k) for k in range(0, min(a_only, b_only) + 1)) / 2.0**total
    return float(min(1.0, 2.0 * tail))


def analyse(outcomes: Outcomes, *, h2_window: tuple[float, float], h2_outcomes: Outcomes | None = None) -> dict[str, Any]:
    """H1–H4 on the ER-1 outcomes (and H2 on its own outcomes when supplied)."""

    boot = SharedBootstrap(len(outcomes.stable_ids), int(get("evaluation.bootstrap_resamples")))
    result = {
        "choices": dict(ANALYSIS_CHOICES),
        "n_images": len(outcomes.stable_ids),
        "cells": list(outcomes.cells),
        "snr_grid_db": list(outcomes.snrs),
        "H1": h1_like(outcomes, boot, learned="learned", comparator="classical_adaptive", hypothesis="H1"),
        "H3": h3(outcomes, boot, learned="learned", comparator="classical_adaptive"),
        "H4": h1_like(outcomes, boot, learned="learned", comparator="er9_digital", hypothesis="H4"),
    }
    h2_source = h2_outcomes or outcomes
    if "classical_fixed_mcs" in h2_source.correct:
        result["H2"] = h2(h2_source, boot, learned="learned", comparator="classical_fixed_mcs", low_snr=h2_window[0], high_snr=h2_window[1])
    return result


__all__ = [
    "ANALYSIS_CHOICES",
    "ANALYSIS_IMPLEMENTATION",
    "Outcomes",
    "SharedBootstrap",
    "analyse",
    "h1_like",
    "h2",
    "h3",
    "longest_run",
    "mcnemar_exact",
    "outcomes_from_rows",
    "run_rule",
]
