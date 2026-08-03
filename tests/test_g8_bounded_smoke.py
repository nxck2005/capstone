"""Adversarial checks for the independent schema-2 bounded-smoke verifier."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import verify_g8_bounded_smoke as verifier
import gen_g8_bler_runner_contract as runner_generator
from baseline import g8_bler_runner as runner
from baseline.g8_campaign import CAMPAIGN_STATE, rendered_json, sha256_bytes


REPO = Path(__file__).parents[1]
RECORD = REPO / "results/baseline/g8/bounded_smoke_record.json"


def _candidate(tmp_path: Path, mutation) -> tuple[Path, Path, Path]:
    payload = json.loads(RECORD.read_bytes())
    contract_path = tmp_path / "bler_runner_contract.json"
    contract_path.write_bytes(rendered_json(runner_generator.build()))
    contract_sha = sha256_bytes(contract_path.read_bytes())
    contract_id = json.loads(contract_path.read_bytes())["contract_id"]
    payload["bler_runner_contract_id"] = contract_id
    payload["bler_runner_contract_sha256"] = contract_sha
    mutation(payload)
    record_path = tmp_path / "bounded_smoke_record.json"
    record_bytes = rendered_json(payload)
    record_path.write_bytes(record_bytes)
    state = json.loads(CAMPAIGN_STATE.read_bytes())
    for entry in state["identity"]["produced_artifacts"]:
        if entry["path"] == runner.RUNNER_CONTRACT_REPO_RELATIVE_PATH:
            entry.update(sha256=contract_sha, bytes=len(contract_path.read_bytes()))
        elif entry["path"] == runner.SMOKE_RECORD_REPO_RELATIVE_PATH:
            entry.update(sha256=sha256_bytes(record_bytes), bytes=len(record_bytes))
    state_path = tmp_path / "campaign_state.json"
    state_path.write_bytes(rendered_json(state))
    return record_path, state_path, contract_path


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["selected_work_units"][0].__setitem__("attempt", 2),
        lambda p: p["selected_work_units"][0]["seed_records"]["information_bits"].__setitem__("seed_uint64", 0),
        lambda p: p["selected_work_units"][0]["request"].__setitem__("snr_db", 99.0),
        lambda p: p["selected_work_units"][0].__setitem__("request_sha256", "0" * 64),
        lambda p: p["selected_work_units"][0]["result"]["measurement"].__setitem__("bit_errors", 1),
        lambda p: p["selected_work_units"][0].__setitem__("result_sha256", "0" * 64),
        lambda p: p["selected_work_units"][0]["terminal_state"]["identity"].__setitem__("attempt", 2),
        lambda p: p["selected_work_units"][0].__setitem__("terminal_state_sha256", "0" * 64),
        lambda p: p.__setitem__("bler_runner_contract_id", "g8runner-" + "0" * 64),
        lambda p: p.__setitem__("bler_runner_contract_sha256", "0" * 64),
        lambda p: p.__setitem__("campaign_manifest_sha256", "0" * 64),
        lambda p: p.__setitem__("required_bler_artifact_sha256", "0" * 64),
        lambda p: p.__setitem__("selection_policy_sha256", "0" * 64),
        lambda p: p.__setitem__("test_split_access", 1),
        lambda p: p.__setitem__("merge_eligible", True),
        lambda p: p.__setitem__("required_coverage_contribution", 1),
        lambda p: p.__setitem__("production_root_used", True),
        lambda p: p.__setitem__("temporary_root_removed", False),
        lambda p: p["selected_work_units"].append(copy.deepcopy(p["selected_work_units"][0])),
    ],
)
def test_smoke_chain_mutations_fail_closed(tmp_path, mutation):
    candidate, state, contract = _candidate(tmp_path, mutation)
    with pytest.raises(verifier.SmokeVerificationError):
        verifier.verify(candidate, campaign_state_path=state, runner_contract_path=contract)


def test_registered_artifact_binding_mutation_fails_before_chain(tmp_path):
    candidate, state_path, contract = _candidate(tmp_path, lambda payload: None)
    state = json.loads(state_path.read_bytes())
    for entry in state["identity"]["produced_artifacts"]:
        if entry["path"] == "results/baseline/g8/required_bler_identities.json":
            entry["sha256"] = "0" * 64
    state_path.write_bytes(rendered_json(state))
    with pytest.raises(Exception):
        verifier.verify(candidate, campaign_state_path=state_path, runner_contract_path=contract)


@pytest.mark.parametrize("shape", ["missing", "duplicate", "unknown"])
def test_smoke_verifier_rejects_missing_duplicate_and_unknown_artifact_bindings(tmp_path, shape):
    candidate, state_path, contract = _candidate(tmp_path, lambda payload: None)
    state = json.loads(state_path.read_bytes())
    artifacts = state["identity"]["produced_artifacts"]
    runner_entry = next(entry for entry in artifacts if entry["path"] == runner.RUNNER_CONTRACT_REPO_RELATIVE_PATH)
    if shape == "missing":
        artifacts.remove(runner_entry)
    elif shape == "duplicate":
        artifacts.append(copy.deepcopy(runner_entry))
    else:
        runner_entry["path"] = "results/baseline/g8/unknown.json"
    state_path.write_bytes(rendered_json(state))
    with pytest.raises(verifier.SmokeVerificationError):
        verifier.verify(candidate, campaign_state_path=state_path, runner_contract_path=contract)


def test_verifier_accepts_the_installed_record_and_exact_three_unit_selection():
    # The committed v2 record is retained as the pre-migration fixture; this
    # candidate projection exercises the final v3 verifier chain before the
    # live artifact replacement.
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        candidate, state, contract = _candidate(Path(directory), lambda payload: None)
        record = verifier.verify(candidate, campaign_state_path=state, runner_contract_path=contract)
    assert len(record["selected_work_units"]) == 3
    assert record["official_work_unit_count"] == 3
    assert all(item["classification"] == "terminal_nonmergeable" for item in record["selected_work_units"])
    assert all(item["required_coverage_contribution"] == 0 for item in record["selected_work_units"])
