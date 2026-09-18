"""AM-97 semantic freeze and pointwise-only interpretation checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation import am97_spec_compatibility
from evaluation.am97_spec_compatibility import ALLOWED_PARAMETER_PATHS, load
from evaluation.h4_precision import simulate_h4_precision


REPO = Path(__file__).resolve().parents[1]


def test_am97_downstream_mode_authenticates_successor() -> None:
    value = load(REPO, allow_downstream=True)
    assert value["amendment"] == "AM-97"
    assert value["scientific_boundary"]["new_er9_v4_training"] == 0
    assert value["scientific_boundary"]["test_split"] == "SEALED"
    assert len(ALLOWED_PARAMETER_PATHS) == 7


def test_am97_strict_mode_rejects_current_post_science_repository() -> None:
    # Strict AM-97 must reject the live repository; the AM-98 successor epoch
    # advances the parameter/current-view frontier, so the refusal may name
    # either the historical boundary or the AM-98 drift.
    with pytest.raises(am97_spec_compatibility.AM97SpecCompatibilityError, match="strict AM-97|downstream ER-9|AM-97 parameter drift"):
        load(REPO)


def test_am97_strict_mode_accepts_pre_science_reference_inventory() -> None:
    allowed = {"er_execution_source_manifest_v4.json", "er9_stage1_execution_authorization_v4.json"}
    am97_spec_compatibility.verify_science_inventory(
        set(allowed), allowed_pre=allowed, er2_exists=False, g11_exists=False, allow_downstream=False
    )


def test_am97_history_mode_accepts_downstream_inventory() -> None:
    am97_spec_compatibility.verify_science_inventory(
        {"final_validation/train0_channel0.json"}, allowed_pre=set(), er2_exists=True, g11_exists=True, allow_downstream=True
    )


def test_am97_strict_mode_rejects_nonzero_v4_execution_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    original_read_json = am97_spec_compatibility._read_json
    authority_path = REPO / "results/learned/er9/er9_stage1_execution_authorization_v4.json"

    def read_with_executed_work(path: Path, label: str) -> tuple[dict, bytes]:
        value, raw = original_read_json(path, label)
        if path != authority_path:
            return value, raw
        mutated = dict(value)
        counters = dict(mutated["pre_execution_counters"])
        counters["new_er9_v4_training"] = 1
        mutated["pre_execution_counters"] = counters
        body = dict(mutated)
        body.pop("authority_id")
        mutated["authority_id"] = "w9er9stage1v4auth-" + am97_spec_compatibility.sha256_bytes(am97_spec_compatibility.canonical(body))
        return mutated, am97_spec_compatibility.rendered(mutated)

    monkeypatch.setattr(am97_spec_compatibility, "_read_json", read_with_executed_work)
    with pytest.raises(am97_spec_compatibility.AM97SpecCompatibilityError, match="counters"):
        am97_spec_compatibility._authenticate_v4_pre_science_custody(REPO)


def test_h4_sanity_is_not_full_strength() -> None:
    value = simulate_h4_precision([0] * 9 + [1], sample_size=3925)
    assert value["status"] == "pointwise_precision_diagnostic_complete"
    assert value["diagnostic_role"] == "pointwise_precision_diagnostic_only"
    assert value["interpretation"]["full_h4_decision_procedure_power_certified"] is False
    assert "full_strength" not in value.values()
