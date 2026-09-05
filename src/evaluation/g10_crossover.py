"""Exact, validation-only G-10 observable-crossover semantics (AM-94).

This module consumes count/denominator summaries only.  It does not load a
checkpoint, dataset, image, or scientific result and has no inference entry
point.  Its narrow purpose is to make the pre-science decision rule executable.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from typing import Literal, Sequence

import numpy as np


EXPECTED_DIRECTION = "positive_to_negative"
OPPOSITE_DIRECTION = "negative_to_positive"
HEADLINE_COMPARATOR = "classical_adaptive"
HEADLINE_RATIO = "r_1_6"
LEARNED_CELL_COUNT = 3

Direction = Literal["positive_to_negative", "negative_to_positive"]
LocationKind = Literal["measured_bracket", "exact_measured_point", "measured_zero_plateau"]


@dataclass(frozen=True)
class AccuracyCount:
    correct: int
    denominator: int

    def __post_init__(self) -> None:
        if isinstance(self.correct, bool) or not isinstance(self.correct, int):
            raise TypeError("correct must be an integer count")
        if isinstance(self.denominator, bool) or not isinstance(self.denominator, int):
            raise TypeError("denominator must be an integer count")
        if self.denominator <= 0:
            raise ValueError("denominator must be positive")
        if not 0 <= self.correct <= self.denominator:
            raise ValueError("correct must be between zero and denominator")

    @property
    def exact(self) -> Fraction:
        return Fraction(self.correct, self.denominator)


@dataclass(frozen=True)
class MeasuredPoint:
    snr_db: int | float
    learned_cells: tuple[AccuracyCount, AccuracyCount, AccuracyCount]
    classical_adaptive: AccuracyCount

    def __post_init__(self) -> None:
        if isinstance(self.snr_db, bool) or not isinstance(self.snr_db, (int, float)):
            raise TypeError("snr_db must be numeric")
        if not isfinite(float(self.snr_db)):
            raise ValueError("snr_db must be finite")
        if not isinstance(self.learned_cells, tuple) or len(self.learned_cells) != LEARNED_CELL_COUNT:
            raise ValueError("G-10 requires exactly three frozen W8 headline cells")
        if not all(isinstance(value, AccuracyCount) for value in self.learned_cells):
            raise TypeError("learned_cells must contain AccuracyCount values")
        if not isinstance(self.classical_adaptive, AccuracyCount):
            raise TypeError("the headline comparator must be classical_adaptive counts")


@dataclass(frozen=True)
class PointDecision:
    snr_db: int | float
    learned_cell_accuracies: tuple[Fraction, Fraction, Fraction]
    learned_mean: Fraction
    classical_adaptive_accuracy: Fraction
    gap: Fraction
    sign: int
    learned_population_sd: float


@dataclass(frozen=True)
class CrossoverEvent:
    direction: Direction
    location_kind: LocationKind
    first_snr_db: int | float
    last_snr_db: int | float


@dataclass(frozen=True)
class G10Decision:
    points: tuple[PointDecision, ...]
    events: tuple[CrossoverEvent, ...]
    headline_expected_event: CrossoverEvent | None
    classification: str
    event_count_class: Literal["zero_crossing_events", "exactly_one_event", "multiple_crossing_events"]
    multiple_crossings: bool
    comparator: Literal["classical_adaptive"] = HEADLINE_COMPARATOR
    ratio: Literal["r_1_6"] = HEADLINE_RATIO
    interpolation_used: bool = False
    uncertainty_affects_predicate: bool = False


def _sign(value: Fraction) -> int:
    return (value > 0) - (value < 0)


def decide_point(point: MeasuredPoint) -> PointDecision:
    """Aggregate exactly three cell fractions, then subtract adaptive classical."""

    cells = tuple(value.exact for value in point.learned_cells)
    learned_mean = sum(cells, start=Fraction(0, 1)) / LEARNED_CELL_COUNT
    classical = point.classical_adaptive.exact
    gap = learned_mean - classical
    # The float conversion is confined to the descriptive SD.  It is never
    # read by the exact sign/event/classification path below.
    population_sd = float(np.std(np.asarray([float(value) for value in cells]), ddof=0))
    return PointDecision(
        snr_db=point.snr_db,
        learned_cell_accuracies=cells,
        learned_mean=learned_mean,
        classical_adaptive_accuracy=classical,
        gap=gap,
        sign=_sign(gap),
        learned_population_sd=population_sd,
    )


def _event(direction: Direction, start: PointDecision, end: PointDecision, *, zero_run: bool) -> CrossoverEvent:
    if not zero_run:
        kind: LocationKind = "measured_bracket"
    elif start.snr_db == end.snr_db:
        kind = "exact_measured_point"
    else:
        kind = "measured_zero_plateau"
    return CrossoverEvent(
        direction=direction,
        location_kind=kind,
        first_snr_db=start.snr_db,
        last_snr_db=end.snr_db,
    )


def _events(points: tuple[PointDecision, ...]) -> tuple[CrossoverEvent, ...]:
    events: list[CrossoverEvent] = []
    index = 0
    while index < len(points) - 1:
        current = points[index]
        following = points[index + 1]
        if current.sign and following.sign:
            if current.sign != following.sign:
                direction: Direction = EXPECTED_DIRECTION if current.sign > 0 else OPPOSITE_DIRECTION
                events.append(_event(direction, current, following, zero_run=False))
            index += 1
            continue
        if following.sign == 0:
            zero_start = index + 1
            zero_end = zero_start
            while zero_end + 1 < len(points) and points[zero_end + 1].sign == 0:
                zero_end += 1
            lower = points[zero_start - 1] if zero_start > 0 else None
            upper = points[zero_end + 1] if zero_end + 1 < len(points) else None
            if lower is not None and upper is not None and lower.sign and upper.sign and lower.sign != upper.sign:
                direction = EXPECTED_DIRECTION if lower.sign > 0 else OPPOSITE_DIRECTION
                events.append(_event(direction, points[zero_start], points[zero_end], zero_run=True))
            index = zero_end + 1
            continue
        index += 1
    return tuple(events)


def _classification(signs: tuple[int, ...], events: tuple[CrossoverEvent, ...]) -> str:
    expected = any(event.direction == EXPECTED_DIRECTION for event in events)
    if expected:
        return "expected_crossover_observed"
    if all(sign > 0 for sign in signs):
        return "learned_strict_dominance"
    if all(sign == 0 for sign in signs):
        return "all_measured_ties"
    if all(sign >= 0 for sign in signs):
        return "learned_never_overtaken_with_ties"
    if all(sign < 0 for sign in signs):
        return "classical_strict_dominance"
    if all(sign <= 0 for sign in signs):
        return "classical_never_worse_with_ties"
    if any(event.direction == OPPOSITE_DIRECTION for event in events):
        return "opposite_direction_only"
    raise ValueError("unclassifiable G-10 sign pattern")


def decide_g10(points: Sequence[MeasuredPoint]) -> G10Decision:
    """Apply AM-94 without interpolation, tolerance, or uncertainty gating."""

    decided = tuple(decide_point(point) for point in points)
    if not decided:
        raise ValueError("at least one measured G-10 SNR point is required")
    if any(left.snr_db >= right.snr_db for left, right in zip(decided, decided[1:], strict=False)):
        raise ValueError("G-10 SNR points must be unique and strictly increasing")
    events = _events(decided)
    headline = next((event for event in events if event.direction == EXPECTED_DIRECTION), None)
    count_class: Literal["zero_crossing_events", "exactly_one_event", "multiple_crossing_events"]
    if not events:
        count_class = "zero_crossing_events"
    elif len(events) == 1:
        count_class = "exactly_one_event"
    else:
        count_class = "multiple_crossing_events"
    return G10Decision(
        points=decided,
        events=events,
        headline_expected_event=headline,
        classification=_classification(tuple(point.sign for point in decided), events),
        event_count_class=count_class,
        multiple_crossings=len(events) > 1,
    )
