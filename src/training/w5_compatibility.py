"""Explicit W5 policy boundary for the additive W7 training core.

This adapter does not replace or rewrite the historical W5 engine.  It exposes
its already-frozen role and constructor as a named compatibility boundary so
W7 code cannot accidentally promote a W5 smoke checkpoint.
"""

from __future__ import annotations

from collections.abc import Mapping

from training.djscc import (
    DJSCCTrainer as HistoricalW5Trainer,
    ELIGIBILITY as W5_ELIGIBILITY,
    W5Hold,
    W5SmokeLimits,
    W5SourceLineage,
)


W5_POLICY_NAME = "W5_NON_SCIENTIFIC_PLUMBING_ONLY"


def validate_historical_w5_artifact(eligibility: Mapping[str, str]) -> None:
    """Require the exact historical non-scientific W5 eligibility map."""

    if dict(eligibility) != dict(W5_ELIGIBILITY):
        raise W5Hold("historical W5 artifact eligibility differs")


def historical_w5_trainer(*args: object, **kwargs: object) -> HistoricalW5Trainer:
    """Construct the unchanged W5 engine; no W7 role can be passed through."""

    trainer = HistoricalW5Trainer(*args, **kwargs)  # type: ignore[arg-type]
    validate_historical_w5_artifact(W5_ELIGIBILITY)
    return trainer


__all__ = [
    "HistoricalW5Trainer",
    "W5_ELIGIBILITY",
    "W5Hold",
    "W5SmokeLimits",
    "W5SourceLineage",
    "W5_POLICY_NAME",
    "historical_w5_trainer",
    "validate_historical_w5_artifact",
]
