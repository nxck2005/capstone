"""Deterministic, count-only coverage for the pre-science AM-94 rule."""

from __future__ import annotations

from fractions import Fraction
import inspect

import numpy as np
import pytest

from evaluation.g10_crossover import (
    AccuracyCount,
    EXPECTED_DIRECTION,
    HEADLINE_COMPARATOR,
    MeasuredPoint,
    OPPOSITE_DIRECTION,
    decide_g10,
    decide_point,
)


def _accuracy(sign: int) -> AccuracyCount:
    return {-1: AccuracyCount(1, 3), 0: AccuracyCount(1, 2), 1: AccuracyCount(2, 3)}[sign]


def _point(snr: int, sign: int) -> MeasuredPoint:
    return MeasuredPoint(snr, (_accuracy(sign),) * 3, AccuracyCount(1, 2))


def _decision(*signs: int):
    return decide_g10([_point(index, sign) for index, sign in enumerate(signs)])


def test_aggregate_is_exact_mean_of_exactly_three_frozen_cells() -> None:
    point = MeasuredPoint(
        0,
        (AccuracyCount(1, 1), AccuracyCount(0, 1), AccuracyCount(0, 1)),
        AccuracyCount(1, 3),
    )
    decided = decide_point(point)
    assert decided.learned_mean == Fraction(1, 3)
    assert decided.gap == 0
    assert decided.sign == 0
    with pytest.raises(ValueError, match="exactly three"):
        MeasuredPoint(0, (AccuracyCount(1, 1),) * 2, AccuracyCount(1, 2))  # type: ignore[arg-type]


def test_gap_orientation_is_learned_minus_adaptive_classical() -> None:
    assert decide_point(_point(0, 1)).sign == 1
    assert decide_point(_point(0, -1)).sign == -1
    assert decide_g10([_point(0, 1)]).comparator == HEADLINE_COMPARATOR == "classical_adaptive"
    assert "classical_fixed" not in inspect.signature(MeasuredPoint).parameters


def test_strict_expected_and_opposite_transitions() -> None:
    expected = _decision(1, -1)
    assert [(event.direction, event.location_kind, event.first_snr_db, event.last_snr_db) for event in expected.events] == [
        (EXPECTED_DIRECTION, "measured_bracket", 0, 1)
    ]
    assert expected.classification == "expected_crossover_observed"

    opposite = _decision(-1, 1)
    assert [event.direction for event in opposite.events] == [OPPOSITE_DIRECTION]
    assert opposite.headline_expected_event is None
    assert opposite.classification == "opposite_direction_only"


def test_exact_zero_point_and_plateau_rules() -> None:
    point = _decision(1, 0, -1)
    assert [(event.location_kind, event.first_snr_db, event.last_snr_db) for event in point.events] == [
        ("exact_measured_point", 1, 1)
    ]
    plateau = _decision(1, 0, 0, -1)
    assert [(event.location_kind, event.first_snr_db, event.last_snr_db) for event in plateau.events] == [
        ("measured_zero_plateau", 1, 2)
    ]


@pytest.mark.parametrize("signs", [(1, 0, 1), (0, -1), (1, 0), (0,), (-1, 0, -1)])
def test_contacts_without_opposite_bracketing_are_not_crossovers(signs: tuple[int, ...]) -> None:
    decision = _decision(*signs)
    assert decision.events == ()
    assert decision.headline_expected_event is None


def test_exact_equality_has_no_tolerance_and_no_rounding() -> None:
    exact = MeasuredPoint(
        0,
        (AccuracyCount(1, 3), AccuracyCount(2, 6), AccuracyCount(100, 300)),
        AccuracyCount(1, 3),
    )
    assert decide_point(exact).gap == 0

    tiny_positive = MeasuredPoint(
        0,
        (AccuracyCount(1, 3),) * 3,
        AccuracyCount(333_333, 1_000_000),
    )
    decided = decide_point(tiny_positive)
    assert decided.gap == Fraction(1, 3_000_000)
    assert decided.sign == 1


def test_interpolation_cannot_enter_the_canonical_decision() -> None:
    decision = _decision(1, -1)
    assert "interpolation" not in inspect.signature(decide_g10).parameters
    assert decision.interpolation_used is False
    assert decision.events[0].location_kind == "measured_bracket"
    assert (decision.events[0].first_snr_db, decision.events[0].last_snr_db) == (0, 1)


def test_all_events_retained_and_first_expected_is_headline() -> None:
    decision = _decision(1, -1, 1, 0, -1)
    assert [event.direction for event in decision.events] == [
        EXPECTED_DIRECTION,
        OPPOSITE_DIRECTION,
        EXPECTED_DIRECTION,
    ]
    assert decision.headline_expected_event == decision.events[0]
    assert decision.multiple_crossings is True
    assert decision.event_count_class == "multiple_crossing_events"


def test_population_sd_uses_ddof_zero_and_cannot_affect_predicate() -> None:
    point = MeasuredPoint(
        0,
        (AccuracyCount(1, 4), AccuracyCount(2, 4), AccuracyCount(3, 4)),
        AccuracyCount(1, 2),
    )
    decided = decide_point(point)
    assert decided.learned_population_sd == pytest.approx(np.std([0.25, 0.5, 0.75], ddof=0))
    assert decided.learned_population_sd != pytest.approx(np.std([0.25, 0.5, 0.75], ddof=1))
    decision = decide_g10([point])
    assert decision.points[0].sign == 0
    assert decision.uncertainty_affects_predicate is False


@pytest.mark.parametrize(
    ("signs", "classification"),
    [
        ((1, 1), "learned_strict_dominance"),
        ((1, 0, 1), "learned_never_overtaken_with_ties"),
        ((-1, -1), "classical_strict_dominance"),
        ((-1, 0, -1), "classical_never_worse_with_ties"),
        ((0, 0), "all_measured_ties"),
        ((-1, 0, 1), "opposite_direction_only"),
        ((1, 0, -1), "expected_crossover_observed"),
    ],
)
def test_deterministic_full_sign_pattern_classification(signs: tuple[int, ...], classification: str) -> None:
    assert _decision(*signs).classification == classification
