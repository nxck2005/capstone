"""AM-97 semantic freeze and pointwise-only interpretation checks."""

from __future__ import annotations

from pathlib import Path

from evaluation.am97_spec_compatibility import ALLOWED_PARAMETER_PATHS, load
from evaluation.h4_precision import simulate_h4_precision


REPO = Path(__file__).resolve().parents[1]


def test_am97_downstream_mode_authenticates_successor() -> None:
    value = load(REPO, allow_downstream=True)
    assert value["amendment"] == "AM-97"
    assert value["scientific_boundary"]["new_er9_v4_training"] == 0
    assert value["scientific_boundary"]["test_split"] == "SEALED"
    assert len(ALLOWED_PARAMETER_PATHS) == 7


def test_am97_strict_mode_accepts_only_pre_science_custody() -> None:
    value = load(REPO)
    assert value["status"] == "SEMANTICS_ONLY_FROZEN"


def test_h4_sanity_is_not_full_strength() -> None:
    value = simulate_h4_precision([0] * 9 + [1], sample_size=3925)
    assert value["status"] == "pointwise_precision_diagnostic_complete"
    assert value["diagnostic_role"] == "pointwise_precision_diagnostic_only"
    assert value["interpretation"]["full_h4_decision_procedure_power_certified"] is False
    assert "full_strength" not in value.values()
