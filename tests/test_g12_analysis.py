"""ER-10 paired-inference fixtures for the frozen G-12 analysis."""

from __future__ import annotations

import numpy as np
import pytest

from config.params import get
from evaluation.g12_analysis import (
    Outcomes,
    SharedBootstrap,
    analyse,
    h2,
    longest_run,
    mcnemar_exact,
    outcomes_from_rows,
    run_rule,
)

SNRS = tuple(float(value) for value in get("channel.test_snr_grid_db"))
CELLS = (0, 1, 2)


def _draw(rng: np.random.Generator, accuracy: np.ndarray, n: int) -> np.ndarray:
    return (rng.random((n, len(CELLS), len(SNRS))) < accuracy[None, None, :]).astype(np.int8)


def _outcomes(learned: np.ndarray, classical: np.ndarray, er9: np.ndarray, fixed: np.ndarray | None = None, n: int = 400) -> Outcomes:
    rng = np.random.default_rng(3)
    correct = {"learned": _draw(rng, learned, n), "classical_adaptive": _draw(rng, classical, n), "er9_digital": _draw(rng, er9, n)}
    if fixed is not None:
        correct["classical_fixed_mcs"] = _draw(rng, fixed, n)
    return Outcomes(tuple(f"id{i:04d}" for i in range(n)), SNRS, CELLS, correct)


def test_longest_run_counts_consecutive_points_only() -> None:
    assert longest_run(np.array([[1, 1, 0, 1, 1, 1, 0]])).tolist() == [3]
    assert longest_run(np.array([[0, 0, 0]])).tolist() == [0]


def test_clear_low_snr_advantage_supports_h1_and_is_calibrated() -> None:
    snr = np.asarray(SNRS)
    learned = np.full(len(SNRS), 0.75)
    classical = np.where(snr <= -5, 0.10, 0.85)
    outcomes = _outcomes(learned, classical, classical)
    result = analyse(outcomes, h2_window=(3.0, 7.0))
    assert result["H1"]["r_obs"] >= 3 and result["H1"]["calibrated_p"] <= 0.05
    assert result["H1"]["supported"] is True
    assert result["H1"]["effect_ci"][0] < result["H1"]["effect_size"] < result["H1"]["effect_ci"][1]


def test_identical_systems_do_not_support_h1() -> None:
    accuracy = np.full(len(SNRS), 0.6)
    rng = np.random.default_rng(11)
    shared = _draw(rng, accuracy, 300)
    outcomes = Outcomes(tuple(f"id{i:04d}" for i in range(300)), SNRS, CELLS, {"learned": shared, "classical_adaptive": shared.copy(), "er9_digital": shared.copy()})
    result = analyse(outcomes, h2_window=(3.0, 7.0))
    assert result["H1"]["r_obs"] == 0 and result["H1"]["supported"] is False


def test_null_calibration_rarely_rejects() -> None:
    rng = np.random.default_rng(5)
    dbar = rng.choice([-1.0, 0.0, 1.0], size=(300, 16), p=[0.2, 0.6, 0.2])
    rule = run_rule(dbar, resamples=999)
    assert rule["calibrated_p"] > 0.05 or rule["r_obs"] < 3


def test_h2_intersection_union_needs_both_margins() -> None:
    snr = np.asarray(SNRS)
    learned = np.where(snr < 5, 0.70, 0.78)
    fixed = np.where(snr < 5, 0.10, 0.89)
    outcomes = _outcomes(learned, learned, learned, fixed)
    boot = SharedBootstrap(len(outcomes.stable_ids), 2000)
    result = h2(outcomes, boot, learned="learned", comparator="classical_fixed_mcs", low_snr=3.0, high_snr=7.0)
    assert result["classical_margin_met"] and result["learned_margin_met"] and result["supported"]
    shallow = _outcomes(learned, learned, learned, np.where(snr < 5, 0.70, 0.89))
    result = h2(shallow, SharedBootstrap(len(shallow.stable_ids), 2000), learned="learned", comparator="classical_fixed_mcs", low_snr=3.0, high_snr=7.0)
    assert not result["classical_margin_met"] and not result["supported"]


def test_h3_rejects_divergence_after_a_crossing() -> None:
    snr = np.asarray(SNRS)
    learned = np.full(len(SNRS), 0.60)
    converging = np.clip(0.60 - 0.04 * (-(snr - 18) / 26 * 5), 0.0, 1.0)
    result = analyse(_outcomes(learned, converging, learned), h2_window=(3.0, 7.0))["H3"]
    assert result["clauses"]["negative_slope"] and result["clauses"]["positive_low_gap"] and result["supported"]
    diverging = np.clip(0.55 + (snr + 8) / 26 * 0.35, 0.0, 1.0)
    result = analyse(_outcomes(learned, diverging, learned), h2_window=(3.0, 7.0))["H3"]
    assert result["clauses"]["negative_slope"] and not result["clauses"]["magnitude_contraction"] and not result["supported"]


def test_mcnemar_is_symmetric_and_bounded() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(3, 9) == pytest.approx(mcnemar_exact(9, 3))
    assert 0.0 < mcnemar_exact(0, 10) < 0.01


def test_rows_must_form_complete_one_to_one_trajectories() -> None:
    rows = [
        {"system": "learned", "bw_ratio": "r_1_6", "stable_sample_id": f"s{i}", "train_seed": 0, "test_snr_db": snr, "correct": True}
        for i in range(3) for snr in (-8.0, 0.0)
    ]
    built = outcomes_from_rows(rows, systems={"learned": ("learned", "r_1_6")}, cells=(0,))
    assert built.correct["learned"].shape == (3, 1, 2)
    with pytest.raises(ValueError, match="duplicated"):
        outcomes_from_rows(rows + rows[:1], systems={"learned": ("learned", "r_1_6")}, cells=(0,))
    with pytest.raises(ValueError, match="complete"):
        outcomes_from_rows(rows[:-1], systems={"learned": ("learned", "r_1_6")}, cells=(0,))
