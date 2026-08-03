"""G8_B phase opening tests; all state writes use temporary files."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import open_g8_phase as opener
import register_g8_artifact as registrar
import verify_g8_phase_state as phase_verifier
import verify_g8_preflight as preflight
from baseline.g8_campaign import (
    CAMPAIGN_MANIFEST,
    PB3C_TERMINAL_SHA,
    G8ContractError,
    initial_campaign_state,
    load_campaign_state,
    rendered_json,
    validate_campaign_state,
    write_campaign_state_atomically,
)
from config.params import REPO_ROOT


B1_RESTART = (
    'rg -n "trials_per_point|bler_trials|seed|BlerIdentity|run_ldpc_g2" '
    "spec/SPEC.md spec/params.generated.yaml tools/run_ldpc_g2.py src/baseline"
)


def _state_path(tmp_path: Path) -> Path:
    return tmp_path / "campaign_state.json"


def _opening_state(tmp_path: Path) -> tuple[Path, dict]:
    path = _state_path(tmp_path)
    state = initial_campaign_state(stage="preflight_complete")
    write_campaign_state_atomically(path, state)
    return path, state


def _write_raw(path: Path, state: dict) -> None:
    path.write_bytes(rendered_json(state))


def _g8b_state(tmp_path: Path) -> tuple[Path, dict]:
    path, state = _opening_state(tmp_path)
    state = copy.deepcopy(state)
    state["identity"]["phase"] = "G8_B"
    state["identity"]["stage"] = "tooling_open"
    state["identity"]["restart_command"] = B1_RESTART
    write_campaign_state_atomically(path, state)
    return path, state


def test_valid_g8a_to_g8b_opening(tmp_path: Path) -> None:
    path, _ = _opening_state(tmp_path)
    digest = opener.open_phase("G8_B", B1_RESTART, state_path=path)
    opened = load_campaign_state(path)
    assert digest
    assert opened["identity"]["phase"] == "G8_B"
    assert opened["identity"]["stage"] == "tooling_open"
    assert opened["identity"]["restart_command"] == B1_RESTART


def test_resulting_state_passes_normal_validation(tmp_path: Path) -> None:
    path, _ = _opening_state(tmp_path)
    opener.open_phase("G8_B", B1_RESTART, state_path=path)
    validate_campaign_state(json.loads(path.read_text(encoding="utf-8")))


def test_opening_preserves_campaign_id_and_manifest_hash(tmp_path: Path) -> None:
    path, before = _opening_state(tmp_path)
    opener.open_phase("G8_B", B1_RESTART, state_path=path)
    after = load_campaign_state(path)
    assert after["identity"]["campaign_id"] == before["identity"]["campaign_id"]
    assert after["identity"]["campaign_manifest_sha256"] == before["identity"]["campaign_manifest_sha256"]


def test_opening_preserves_produced_artifact_bindings(tmp_path: Path) -> None:
    path, before = _opening_state(tmp_path)
    opener.open_phase("G8_B", B1_RESTART, state_path=path)
    assert load_campaign_state(path)["identity"]["produced_artifacts"] == before["identity"][
        "produced_artifacts"
    ]


def test_opening_preserves_zero_counters_and_empty_work_units(tmp_path: Path) -> None:
    path, _ = _opening_state(tmp_path)
    opener.open_phase("G8_B", B1_RESTART, state_path=path)
    identity = load_campaign_state(path)["identity"]
    assert identity["completed_work_unit_ids"] == []
    assert identity["in_progress_work_unit_id"] is None
    assert set(identity["counters"].values()) == {0}


def test_wrong_source_stage_is_refused(tmp_path: Path) -> None:
    path, _ = _opening_state(tmp_path)
    state = json.loads(path.read_text(encoding="utf-8"))
    state["identity"]["stage"] = "contract_open"
    _write_raw(path, state)
    with pytest.raises(G8ContractError, match="source stage"):
        opener.open_phase("G8_B", B1_RESTART, state_path=path)


def test_wrong_source_phase_is_refused(tmp_path: Path) -> None:
    path, state = _g8b_state(tmp_path)
    with pytest.raises(G8ContractError, match="source phase"):
        opener.open_phase("G8_C", B1_RESTART, state_path=path)


def test_reopening_g8b_is_refused(tmp_path: Path) -> None:
    path, _ = _g8b_state(tmp_path)
    with pytest.raises(G8ContractError, match="source phase"):
        opener.open_phase("G8_B", B1_RESTART, state_path=path)


def test_skipped_g8c_target_is_refused(tmp_path: Path) -> None:
    path, _ = _opening_state(tmp_path)
    with pytest.raises(G8ContractError, match="exact next phase"):
        opener.open_phase("G8_C", B1_RESTART, state_path=path)


def test_reversed_target_is_refused(tmp_path: Path) -> None:
    path, _ = _opening_state(tmp_path)
    with pytest.raises(G8ContractError, match="exact next phase"):
        opener.open_phase("G8_A", B1_RESTART, state_path=path)


def test_nonzero_counter_is_refused(tmp_path: Path) -> None:
    path, state = _opening_state(tmp_path)
    state["identity"]["counters"]["inference"] = 1
    _write_raw(path, state)
    with pytest.raises(G8ContractError, match="nonzero"):
        opener.open_phase("G8_B", B1_RESTART, state_path=path)


def test_completed_work_unit_is_refused(tmp_path: Path) -> None:
    path, state = _opening_state(tmp_path)
    state["identity"]["completed_work_unit_ids"] = ["bler-claimed"]
    _write_raw(path, state)
    with pytest.raises(G8ContractError):
        opener.open_phase("G8_B", B1_RESTART, state_path=path)


def test_in_progress_work_unit_is_refused(tmp_path: Path) -> None:
    path, state = _opening_state(tmp_path)
    state["identity"]["in_progress_work_unit_id"] = "bler-claimed"
    _write_raw(path, state)
    with pytest.raises(G8ContractError):
        opener.open_phase("G8_B", B1_RESTART, state_path=path)


def test_blank_restart_command_is_refused(tmp_path: Path) -> None:
    path, _ = _opening_state(tmp_path)
    with pytest.raises(G8ContractError, match="nonblank"):
        opener.open_phase("G8_B", "   ", state_path=path)


def test_cross_campaign_manifest_mismatch_is_refused(tmp_path: Path) -> None:
    path, state = _opening_state(tmp_path)
    state["identity"]["campaign_id"] = "g8-other-campaign"
    _write_raw(path, state)
    with pytest.raises(G8ContractError, match="another campaign"):
        opener.open_phase("G8_B", B1_RESTART, state_path=path)


def test_dirty_tracked_state_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path, _ = _opening_state(tmp_path)
    monkeypatch.setattr(opener, "_git_state_file_is_clean", lambda _: False)
    with pytest.raises(G8ContractError, match="dirty"):
        opener.open_phase("G8_B", B1_RESTART, state_path=path)


def test_current_phase_verifier_accepts_valid_g8b_state(tmp_path: Path) -> None:
    path, _ = _g8b_state(tmp_path)
    state = phase_verifier.verify(
        phase="G8_B", stage="tooling_open", require_zero_science=True, state_path=path
    )
    assert state["identity"]["phase"] == "G8_B"


def test_current_phase_verifier_rejects_wrong_expected_phase(tmp_path: Path) -> None:
    path, _ = _g8b_state(tmp_path)
    with pytest.raises(phase_verifier.G8PhaseStateError, match="current phase"):
        phase_verifier.verify(
            phase="G8_A", stage="preflight_complete", require_zero_science=True, state_path=path
        )


def test_current_phase_verifier_rejects_wrong_expected_stage(tmp_path: Path) -> None:
    path, _ = _g8b_state(tmp_path)
    with pytest.raises(phase_verifier.G8PhaseStateError, match="current stage"):
        phase_verifier.verify(
            phase="G8_B", stage="tooling_smoke_complete", require_zero_science=True, state_path=path
        )


def test_current_phase_verifier_rejects_nonzero_science_state(tmp_path: Path) -> None:
    path, state = _g8b_state(tmp_path)
    state["identity"]["counters"]["training"] = 1
    _write_raw(path, state)
    with pytest.raises(phase_verifier.G8PhaseStateError, match="nonzero scientific counters"):
        phase_verifier.verify(
            phase="G8_B", stage="tooling_open", require_zero_science=True, state_path=path
        )


def test_repeated_phase_opening_is_not_idempotently_accepted(tmp_path: Path) -> None:
    path, _ = _opening_state(tmp_path)
    opener.open_phase("G8_B", B1_RESTART, state_path=path)
    with pytest.raises(G8ContractError, match="source phase"):
        opener.open_phase("G8_B", B1_RESTART, state_path=path)


def test_verifier_rejects_unknown_phase_and_stage(tmp_path: Path) -> None:
    path, _ = _g8b_state(tmp_path)
    with pytest.raises(phase_verifier.G8PhaseStateError, match="unknown requested phase"):
        phase_verifier.verify(
            phase="G8_X", stage="tooling_open", require_zero_science=True, state_path=path
        )
    with pytest.raises(phase_verifier.G8PhaseStateError, match="unknown requested stage"):
        phase_verifier.verify(
            phase="G8_B", stage="not-a-stage", require_zero_science=True, state_path=path
        )


def test_verifier_rejects_changed_produced_artifact_binding(tmp_path: Path) -> None:
    path, state = _g8b_state(tmp_path)
    state["identity"]["produced_artifacts"][0]["sha256"] = "0" * 64
    _write_raw(path, state)
    with pytest.raises(phase_verifier.G8PhaseStateError, match="produced artifact binding changed"):
        phase_verifier.verify(
            phase="G8_B", stage="tooling_open", require_zero_science=True, state_path=path
        )


def test_manifest_binding_is_the_committed_manifest_hash() -> None:
    state = load_campaign_state()
    assert state["identity"]["campaign_manifest_sha256"] == phase_verifier.sha256_file(CAMPAIGN_MANIFEST)


# --------------------------------------------------------------------------
# B1 produced-artifact registration and the widened phase verifier
# --------------------------------------------------------------------------

B2_RESTART = (
    'rg -n "BLER_WORK_UNIT|derive_seed|produced_artifacts|completed_work_unit_ids|'
    'in_progress_work_unit_id|write_campaign_state_atomically" src/baseline tools tests'
)
B1_CONTRACT_ARTIFACT = "results/baseline/g8/bler_tooling_contract.json"


def _artifact_binding(relative_path: str) -> dict:
    body = (REPO_ROOT / relative_path).read_bytes()
    return {
        "path": relative_path,
        "sha256": phase_verifier.sha256_file(REPO_ROOT / relative_path),
        "bytes": len(body),
    }


def _g8b_state_with_contract(tmp_path: Path) -> tuple[Path, dict]:
    path, state = _g8b_state(tmp_path)
    state = copy.deepcopy(state)
    artifacts = state["identity"]["produced_artifacts"]
    artifacts.append(_artifact_binding(B1_CONTRACT_ARTIFACT))
    artifacts.sort(key=lambda entry: entry["path"])
    write_campaign_state_atomically(path, state)
    return path, state


def test_original_g8a_artifact_bindings_remain_required(tmp_path: Path) -> None:
    path, state = _g8b_state_with_contract(tmp_path)
    state["identity"]["produced_artifacts"] = [
        entry
        for entry in state["identity"]["produced_artifacts"]
        if entry["path"] != "results/baseline/g8/required_bler_identities.json"
    ]
    _write_raw(path, state)
    with pytest.raises(phase_verifier.G8PhaseStateError, match="base G8_A produced-artifact binding is missing"):
        phase_verifier.verify(
            phase="G8_B", stage="tooling_open", require_zero_science=True, state_path=path
        )


def test_altered_original_binding_is_rejected(tmp_path: Path) -> None:
    path, state = _g8b_state_with_contract(tmp_path)
    for entry in state["identity"]["produced_artifacts"]:
        if entry["path"] == "results/baseline/g8/campaign_manifest.json":
            entry["bytes"] += 1
    _write_raw(path, state)
    with pytest.raises(phase_verifier.G8PhaseStateError):
        phase_verifier.verify(
            phase="G8_B", stage="tooling_open", require_zero_science=True, state_path=path
        )


def test_additional_valid_artifact_binding_is_accepted(tmp_path: Path) -> None:
    path, _ = _g8b_state_with_contract(tmp_path)
    state = phase_verifier.verify(
        phase="G8_B", stage="tooling_open", require_zero_science=True, state_path=path
    )
    paths = [entry["path"] for entry in state["identity"]["produced_artifacts"]]
    assert B1_CONTRACT_ARTIFACT in paths
    assert len(paths) == 3


def test_duplicate_added_path_is_rejected(tmp_path: Path) -> None:
    path, state = _g8b_state_with_contract(tmp_path)
    state["identity"]["produced_artifacts"].append(_artifact_binding(B1_CONTRACT_ARTIFACT))
    _write_raw(path, state)
    # The strict campaign-state loader validates every added binding.
    with pytest.raises(phase_verifier.G8PhaseStateError, match="duplicated"):
        phase_verifier.verify(
            phase="G8_B", stage="tooling_open", require_zero_science=True, state_path=path
        )


def test_unsorted_bindings_are_rejected(tmp_path: Path) -> None:
    path, state = _g8b_state_with_contract(tmp_path)
    state["identity"]["produced_artifacts"].reverse()
    _write_raw(path, state)
    with pytest.raises(phase_verifier.G8PhaseStateError, match="unsorted"):
        phase_verifier.verify(
            phase="G8_B", stage="tooling_open", require_zero_science=True, state_path=path
        )


def test_required_artifact_check_accepts_the_registered_contract(tmp_path: Path) -> None:
    path, _ = _g8b_state_with_contract(tmp_path)
    state = phase_verifier.verify(
        phase="G8_B",
        stage="tooling_open",
        require_zero_science=True,
        state_path=path,
        require_artifacts=(B1_CONTRACT_ARTIFACT,),
    )
    assert state["identity"]["phase"] == "G8_B"


def test_required_artifact_check_rejects_an_absent_path(tmp_path: Path) -> None:
    path, _ = _g8b_state(tmp_path)
    with pytest.raises(phase_verifier.G8PhaseStateError, match="required produced-artifact binding is absent"):
        phase_verifier.verify(
            phase="G8_B",
            stage="tooling_open",
            require_zero_science=True,
            state_path=path,
            require_artifacts=(B1_CONTRACT_ARTIFACT,),
        )


def _append_artifact_path(state: dict, path: str) -> None:
    target = Path(path) if Path(path).is_absolute() else REPO_ROOT / path
    body = target.read_bytes()
    state["identity"]["produced_artifacts"].append(
        {"path": path, "sha256": phase_verifier.sha256_file(target), "bytes": len(body)}
    )
    state["identity"]["produced_artifacts"].sort(key=lambda entry: entry["path"])


@pytest.mark.parametrize(
    "bad_path, message",
    [
        ("/etc/hostname", "absolute"),
        ("results/baseline/g8/../w4/integration_adjudication.json", "contains"),
        ("results/baseline/w4/integration_adjudication.json", "outside"),
    ],
)
def test_phase_verifier_rejects_unsafe_additional_artifact_paths(
    tmp_path: Path, bad_path: str, message: str
) -> None:
    path, state = _g8b_state_with_contract(tmp_path)
    state = copy.deepcopy(state)
    _append_artifact_path(state, bad_path)
    _write_raw(path, state)
    with pytest.raises(phase_verifier.G8PhaseStateError, match=message):
        phase_verifier.verify(
            phase="G8_B", stage="tooling_open", require_zero_science=True, state_path=path
        )


def test_phase_verifier_rejects_normalized_artifact_alias(tmp_path: Path) -> None:
    path, state = _g8b_state_with_contract(tmp_path)
    state = copy.deepcopy(state)
    _append_artifact_path(state, "results/baseline/g8/./campaign_manifest.json")
    _write_raw(path, state)
    with pytest.raises(phase_verifier.G8PhaseStateError, match="aliases"):
        phase_verifier.verify(
            phase="G8_B", stage="tooling_open", require_zero_science=True, state_path=path
        )


def test_phase_verifier_binds_live_seed_identity_to_required_tooling_contract(
    tmp_path: Path,
) -> None:
    path, state = _g8b_state_with_contract(tmp_path)
    state = copy.deepcopy(state)
    state["identity"]["seed_derivation_identity"] = "sha256(changed)-v1"
    _write_raw(path, state)
    with pytest.raises(phase_verifier.G8PhaseStateError, match="seed derivation identity"):
        phase_verifier.verify(
            phase="G8_B",
            stage="tooling_open",
            require_zero_science=True,
            state_path=path,
            require_artifacts=(B1_CONTRACT_ARTIFACT,),
        )


def test_registration_preserves_counters_and_work_unit_state(tmp_path: Path) -> None:
    path, before = _g8b_state(tmp_path)
    digest = registrar.register(B1_CONTRACT_ARTIFACT, B2_RESTART, state_path=path)
    after = load_campaign_state(path)
    identity = after["identity"]
    assert digest
    assert identity["completed_work_unit_ids"] == []
    assert identity["in_progress_work_unit_id"] is None
    assert identity["counters"] == before["identity"]["counters"]
    assert set(identity["counters"].values()) == {0}
    assert identity["phase"] == "G8_B" and identity["stage"] == "tooling_open"
    assert identity["campaign_id"] == before["identity"]["campaign_id"]
    assert identity["restart_command"] == B2_RESTART
    paths = [entry["path"] for entry in identity["produced_artifacts"]]
    assert paths == sorted(paths)
    assert B1_CONTRACT_ARTIFACT in paths
    for entry in before["identity"]["produced_artifacts"]:
        assert entry in identity["produced_artifacts"]


def test_registration_is_idempotent_for_an_identical_binding(tmp_path: Path) -> None:
    path, _ = _g8b_state(tmp_path)
    first = registrar.register(B1_CONTRACT_ARTIFACT, B2_RESTART, state_path=path)
    second = registrar.register(B1_CONTRACT_ARTIFACT, B2_RESTART, state_path=path)
    assert first == second
    assert len(load_campaign_state(path)["identity"]["produced_artifacts"]) == 3


def test_registration_refuses_a_conflicting_binding(tmp_path: Path) -> None:
    path, state = _g8b_state(tmp_path)
    stale = _artifact_binding(B1_CONTRACT_ARTIFACT)
    stale["sha256"] = "0" * 64
    state["identity"]["produced_artifacts"].append(stale)
    state["identity"]["produced_artifacts"].sort(key=lambda entry: entry["path"])
    path.write_bytes(rendered_json(state))
    with pytest.raises(G8ContractError):
        registrar.register(B1_CONTRACT_ARTIFACT, B2_RESTART, state_path=path)


def test_registration_refuses_nonzero_scientific_state(tmp_path: Path) -> None:
    path, state = _g8b_state(tmp_path)
    state["identity"]["counters"]["validation_decoding"] = 1
    _write_raw(path, state)
    with pytest.raises(G8ContractError, match="nonzero scientific counters"):
        registrar.register(B1_CONTRACT_ARTIFACT, B2_RESTART, state_path=path)


def test_registration_refuses_a_claimed_work_unit(tmp_path: Path) -> None:
    path, state = _g8b_state(tmp_path)
    state["identity"]["completed_work_unit_ids"] = ["bler-claimed"]
    _write_raw(path, state)
    with pytest.raises(G8ContractError, match="completed scientific work units"):
        registrar.register(B1_CONTRACT_ARTIFACT, B2_RESTART, state_path=path)


@pytest.mark.parametrize(
    "bad_path",
    [
        "results/baseline/g2/bler_results.csv",
        "spec/SPEC.md",
        "/etc/hostname",
        "results/baseline/g8/../w4/integration_adjudication.json",
    ],
)
def test_registration_refuses_paths_outside_the_g8_artifact_root(
    tmp_path: Path, bad_path: str
) -> None:
    path, _ = _g8b_state(tmp_path)
    with pytest.raises(G8ContractError):
        registrar.register(bad_path, B2_RESTART, state_path=path)


def test_registration_refuses_a_missing_artifact(tmp_path: Path) -> None:
    path, _ = _g8b_state(tmp_path)
    with pytest.raises(G8ContractError, match="cannot read artifact"):
        registrar.register("results/baseline/g8/does_not_exist.json", B2_RESTART, state_path=path)


def test_registration_refuses_a_blank_restart_command(tmp_path: Path) -> None:
    path, _ = _g8b_state(tmp_path)
    with pytest.raises(G8ContractError, match="nonblank"):
        registrar.register(B1_CONTRACT_ARTIFACT, "   ", state_path=path)


def test_registration_refuses_a_wrong_phase_or_stage(tmp_path: Path) -> None:
    path, _ = _opening_state(tmp_path)
    with pytest.raises(G8ContractError, match="requires G8_B/tooling_open"):
        registrar.register(B1_CONTRACT_ARTIFACT, B2_RESTART, state_path=path)


# --------------------------------------------------------------------------
# Phase-boundary regression: the frozen G8_A verifier is G8_A-specific
# --------------------------------------------------------------------------


def test_frozen_g8a_verifier_rejects_the_real_live_later_phase_cursor() -> None:
    live = load_campaign_state()
    assert live["identity"]["phase"] == "G8_C"
    with pytest.raises(preflight.G8PreflightError, match="campaign state exposes a later phase"):
        preflight.verify()


def test_live_manifest_named_completeness_checks_are_enforced() -> None:
    # The B0 gap: these manifest clauses were bound but never named by a verifier.
    manifest, _required = phase_verifier._verify_manifest_and_required_artifact()
    assert manifest["scientific_base"]["commit_sha"] == PB3C_TERMINAL_SHA
    assert manifest["scientific_base"]["source_state_mode"] == "content_hashes_with_pb3c_base"
    assert manifest["scientific_base"]["future_g8a_final_commit_not_part_of_identity"] is True
    assert manifest["interpretation_rules"] == {
        "pre_data_contract_not_authorization": True,
        "later_phases_may_not_silently_reinterpret_earlier_artifacts": True,
        "changed_bound_scientific_policy_invalidates_campaign": True,
    }
    assert (
        manifest["selection_policy"]["changing_bound_policy_after_campaign_start_invalidates_campaign"]
        is True
    )
