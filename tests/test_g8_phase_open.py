"""G8_B phase opening tests; all state writes use temporary files."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import open_g8_phase as opener
import verify_g8_phase_state as phase_verifier
from baseline.g8_campaign import (
    CAMPAIGN_MANIFEST,
    G8ContractError,
    initial_campaign_state,
    load_campaign_state,
    rendered_json,
    validate_campaign_state,
    write_campaign_state_atomically,
)


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
