"""G8_A pre-data contract tests; no scientific runner is imported or called."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

import pytest

import verify_g8_preflight as verifier
from baseline.classical.g8_campaign import (
    CAMPAIGN_MANIFEST,
    build_structural_preflight,
    campaign_identifier,
    canonical_json,
    compare_required_to_g2,
    g2_measured_work_units,
    rendered_json,
)


def _payload() -> dict[str, Any]:
    return json.loads(CAMPAIGN_MANIFEST.read_text(encoding="utf-8"))


def _mutated(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    *,
    refresh_id: bool = True,
) -> Path:
    payload = copy.deepcopy(_payload())
    mutate(payload)
    if refresh_id:
        payload["campaign_id"] = campaign_identifier(payload)
    path = tmp_path / "campaign_manifest.json"
    path.write_bytes(rendered_json(payload))
    return path


def test_committed_campaign_contract_verifies() -> None:
    payload = verifier.verify()
    assert payload["authorization_issued"] is False
    assert payload["campaign_started"] is False
    assert payload["test_split_access"] == 0


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda p: p["w4_adjudication"].__setitem__("sha256", "0" * 64), "bound SHA-256"),
        (lambda p: p["selection_policy"].__setitem__("selection_policy_sha256", "0" * 64), "policy hash"),
        (lambda p: p["selection_sources"][0].__setitem__("sha256", "0" * 64), "bound SHA-256"),
        (lambda p: p["normative_sources"][0].__setitem__("sha256", "0" * 64), "bound SHA-256"),
        (lambda p: p["dataset_split_manifests"][0].__setitem__("sha256", "0" * 64), "bound SHA-256"),
        (lambda p: p.__setitem__("phase_order", list(reversed(p["phase_order"]))), "phase order"),
        (lambda p: p.__setitem__("campaign_started", True), "campaign_started"),
        (lambda p: p.__setitem__("authorization_issued", True), "authorization_issued"),
        (lambda p: p["contract_sources"].pop(), "contract_sources"),
    ],
)
def test_manifest_mutations_fail_closed(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    path = _mutated(tmp_path, mutate)
    with pytest.raises(verifier.G8PreflightError, match=match):
        verifier.verify(path)


def test_stale_campaign_id_is_rejected_before_other_claims(tmp_path: Path) -> None:
    path = _mutated(
        tmp_path,
        lambda p: p.__setitem__("campaign_started", True),
        refresh_id=False,
    )
    with pytest.raises(verifier.G8PreflightError, match="campaign_id"):
        verifier.verify(path)


def test_noncanonical_manifest_bytes_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "campaign_manifest.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    with pytest.raises(verifier.G8PreflightError, match="canonical rendered JSON"):
        verifier.verify(path)


def test_structural_grid_is_complete_deterministic_and_not_feasibility_claim() -> None:
    first = build_structural_preflight()
    second = build_structural_preflight()
    assert canonical_json(first) == canonical_json(second)
    axes = first["axes"]
    expected = sum(
        len(axes["ratios"])
        * len(axes["encode_axis_px"][dataset["name"]])
        * len(axes["modulations"])
        * len(axes["ldpc_rates"])
        * len(axes["snr_grid_db"])
        for dataset in axes["datasets"]
    )
    assert first["grid_kind"] == "structural_not_codec_feasible"
    assert first["counts"]["structural_candidates"] == expected == 12096
    assert first["counts"]["packet_configurations"] == 144


def test_required_work_units_have_complete_actual_bler_identities() -> None:
    payload = build_structural_preflight()
    rows = payload["required_bler_work_units"]
    ids = [row["work_unit_id"] for row in rows]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids)) == 3213
    expected_fields = {
        "k_and_n",
        "base_graph",
        "lifting_size",
        "modulation",
        "decoder_algorithm",
        "decoder_offset",
        "iterations",
        "snr_convention",
        "rate",
    }
    assert all(set(row["identity"]) == expected_fields for row in rows)
    assert all(row["information_length"] == row["identity"]["k_and_n"][0] for row in rows)
    assert all(row["codeword_length"] == row["identity"]["k_and_n"][1] for row in rows)


def test_current_g2_coverage_is_honestly_insufficient_without_extrapolation() -> None:
    payload = build_structural_preflight()
    counts = payload["counts"]
    coverage = payload["g2_comparison"]
    assert counts["g2_exact_coverage"] == 0
    assert counts["missing_required"] == counts["required_unique_bler_work_units"]
    assert counts["g2_present_outside_required"] == 24
    assert coverage["coverage_complete"] is False
    assert coverage["interpolation_used"] is False
    assert coverage["extrapolation_used"] is False


def test_g2_exact_point_positive_control_is_recognized() -> None:
    measured = g2_measured_work_units()
    required = copy.deepcopy(measured[0])
    required["work_unit_id"] = "required-exact-positive-control"
    comparison = compare_required_to_g2([required], measured)
    assert comparison["already_characterized_exact"] == [required["work_unit_id"]]
    assert comparison["coverage_complete"] is True


def test_same_g2_identity_at_unmeasured_snr_is_not_extrapolated() -> None:
    measured = g2_measured_work_units()
    required = copy.deepcopy(measured[0])
    required["work_unit_id"] = "required-outside-support"
    required["snr_db"] = 1000
    comparison = compare_required_to_g2([required], measured)
    assert comparison["already_characterized_exact"] == []
    assert comparison["uncharacterized_snr_support"] == [required["work_unit_id"]]
    assert comparison["extrapolation_used"] is False
