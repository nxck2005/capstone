"""G8_A pre-data contract tests; no scientific runner is imported or called."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

import pytest

import verify_g8_preflight as verifier
from baseline.g8_campaign import (
    CAMPAIGN_MANIFEST,
    G8ContractError,
    build_structural_preflight,
    campaign_identifier,
    canonical_json,
    compare_required_to_g2,
    g2_measured_work_units,
    initial_campaign_state,
    load_campaign_state,
    rendered_json,
    validate_campaign_state,
    validate_state_transition,
    write_campaign_state_atomically,
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
        (lambda p: p["normative_sources"][1].__setitem__("sha256", "0" * 64), "bound SHA-256"),
        (lambda p: p["dataset_split_manifests"][0].__setitem__("sha256", "0" * 64), "bound SHA-256"),
        (lambda p: p["dataset_split_manifests"][1].__setitem__("sha256", "0" * 64), "bound SHA-256"),
        (lambda p: p["dataset_split_manifests"][2].__setitem__("sha256", "0" * 64), "bound SHA-256"),
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


def test_initial_state_is_pre_science_and_all_counters_are_zero() -> None:
    state = initial_campaign_state()
    validated = validate_campaign_state(state)
    identity = validated["identity"]
    assert identity["phase"] == "G8_A"
    assert identity["stage"] == "contract_open"
    assert identity["completed_work_unit_ids"] == []
    assert identity["in_progress_work_unit_id"] is None
    assert set(identity["counters"].values()) == {0}


def test_campaign_state_atomic_write_is_byte_reproducible(tmp_path: Path) -> None:
    path = tmp_path / "campaign_state.json"
    state = initial_campaign_state()
    first_hash = write_campaign_state_atomically(path, state)
    first = path.read_bytes()
    second_hash = write_campaign_state_atomically(path, state)
    assert path.read_bytes() == first
    assert second_hash == first_hash
    assert list(tmp_path.glob("*.partial")) == []


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda s: s["identity"].__setitem__("campaign_id", "g8-wrong"), "another campaign"),
        (lambda s: s["identity"].__setitem__("campaign_manifest_sha256", "0" * 64), "manifest hash mismatch"),
        (lambda s: s["identity"]["completed_work_unit_ids"].extend(["x", "x"]), "duplicated"),
        (lambda s: s["identity"].__setitem__("in_progress_work_unit_id", 7), "in-progress"),
        (lambda s: s["identity"]["counters"].__setitem__("training", -1), "non-negative"),
    ],
)
def test_campaign_state_mutations_fail_closed(
    mutate: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    state = initial_campaign_state()
    mutate(state)
    with pytest.raises(G8ContractError, match=match):
        validate_campaign_state(state)


def test_partial_campaign_state_is_not_accepted(tmp_path: Path) -> None:
    path = tmp_path / "campaign_state.json"
    path.write_text('{"schema_version": 1', encoding="utf-8")
    with pytest.raises(G8ContractError, match="cannot read campaign state"):
        load_campaign_state(path)


def test_state_transition_refuses_skipped_and_reversed_phases() -> None:
    opened = initial_campaign_state()
    skipped = copy.deepcopy(opened)
    skipped["identity"]["phase"] = "G8_C"
    skipped["identity"]["stage"] = "characterization_open"
    with pytest.raises(G8ContractError, match="skips or reverses a phase"):
        validate_state_transition(opened, skipped)

    complete = initial_campaign_state(stage="preflight_complete")
    with pytest.raises(G8ContractError, match="skips or reverses a stage"):
        validate_state_transition(complete, opened)


def test_state_transition_allows_only_adjacent_g8a_stage() -> None:
    opened = initial_campaign_state()
    complete = initial_campaign_state(stage="preflight_complete")
    validate_state_transition(opened, complete)


def test_omitted_required_grid_axis_fails_closed() -> None:
    required = build_structural_preflight()
    required["axes"].pop("ratios")
    with pytest.raises(verifier.G8PreflightError, match="ratio axis"):
        verifier.verify_required_structure(required)


def test_duplicate_candidate_id_fails_closed() -> None:
    required = build_structural_preflight()
    duplicate = copy.deepcopy(required["structural_candidates"][0])
    required["structural_candidates"].insert(1, duplicate)
    with pytest.raises(verifier.G8PreflightError, match="duplicate candidate ID"):
        verifier.verify_required_structure(required)


def test_duplicate_bler_work_unit_id_fails_closed() -> None:
    required = build_structural_preflight()
    duplicate = copy.deepcopy(required["required_bler_work_units"][0])
    required["required_bler_work_units"].insert(1, duplicate)
    with pytest.raises(verifier.G8PreflightError, match="duplicate BLER work-unit ID"):
        verifier.verify_required_structure(required)


def test_nondeterministic_candidate_order_fails_closed() -> None:
    required = build_structural_preflight()
    required["structural_candidates"][0], required["structural_candidates"][1] = (
        required["structural_candidates"][1],
        required["structural_candidates"][0],
    )
    with pytest.raises(verifier.G8PreflightError, match="nondeterministic"):
        verifier.verify_required_structure(required)


def test_false_complete_g2_coverage_claim_fails_closed() -> None:
    required = build_structural_preflight()
    required["g2_comparison"]["coverage_complete"] = True
    with pytest.raises(verifier.G8PreflightError, match="falsely complete"):
        verifier.verify_required_structure(required)


def test_mismatched_g2_identity_cannot_be_treated_as_characterized() -> None:
    required = build_structural_preflight()["required_bler_work_units"][0]
    comparison = compare_required_to_g2([required], g2_measured_work_units())
    assert comparison["already_characterized_exact"] == []
    assert comparison["uncharacterized_identity_mismatch"] == [required["work_unit_id"]]


def test_tracked_non_test_authorization_scan_passes_and_detects_real_calls() -> None:
    verifier.verify_no_tracked_authorization_construction()
    assert verifier.authorization_constructions(
        "value = G8Authorization(campaign='G-8')\n", "synthetic.py"
    ) == ["synthetic.py:1"]
    assert verifier.authorization_constructions(
        "# G8Authorization()\ntext = 'G8Authorization()'\n", "synthetic.py"
    ) == []
