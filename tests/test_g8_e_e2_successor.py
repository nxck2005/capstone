"""Corrected-v3 worker-successor epoch: lifecycle and boundary tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from baseline import g8_e_corrected_v2 as v2
from baseline import g8_e_corrected_v3 as v3
from baseline import g8_e_corrected_v3s as v3s

import aggregate_g8_e_corrected_v3s as aggregate_cli  # noqa: E402
import merge_g8_e_corrected_v3s as merge_cli  # noqa: E402
import run_g8_e_corrected_v3s as runner_cli  # noqa: E402
from prove_g8_e_v3s_lifecycle import (  # noqa: E402
    _authorization,
    _br11_stub,
    fixture_context,
)


def _error(exc) -> type:  # pragma: no cover - helper typing aid
    return type(exc)


def test_successor_source_paths_are_a_strict_extension_of_v3():
    base = dict(v3._source_paths())
    extended = dict(v3s._successor_source_paths())
    assert set(base) <= set(extended)
    added = set(extended) - set(base)
    assert {
        "src/baseline/g8_e_corrected_v3s.py",
        "tools/freeze_g8_e_v3s.py",
        "tools/run_g8_e_corrected_v3s.py",
        "tools/merge_g8_e_corrected_v3s.py",
        "tools/aggregate_g8_e_corrected_v3s.py",
        "tools/verify_g8_e_corrected_v3s.py",
        "requirements-pascal.lock",
    } <= added


def test_aborted_local_campaign_cannot_execute_again():
    aborted = v3s.aborted_local_campaign_id()
    assert aborted == "g8e-v3-c20d9c4f4638687ad9e4e3e69bf7b9dbdf509a62c2c3a4d95dbbe6771ced57b5"
    with pytest.raises(ValueError):
        v3s.reject_superseded_campaign(aborted)


def test_superseded_campaign_ids_cover_every_prior_epoch():
    ids = v3s.superseded_campaign_ids()
    v2_contract, _ = v2._rendered_object(v2.V2_CONTRACT_PATH, "v2 contract")
    assert v2.ORIGINAL_CAMPAIGN_ID in ids
    assert v2.FIRST_CORRECTED_CAMPAIGN_ID in ids
    assert v2_contract["campaign_id"] in ids
    assert v3s.aborted_local_campaign_id() in ids


def test_relocation_provenance_is_frozen_pre_data_with_zero_successor_coverage():
    stored, raw = v3s._rendered_object(
        v3s.V3S_RELOCATION_PROVENANCE_PATH, "v3s relocation provenance"
    )
    assert stored["classification"] == "PARTIAL_OWNER_ABORTED_PROFILE_RELOCATION"
    assert stored["scientific_defect"] == "NONE"
    assert stored["selection_contribution"] == "ZERO"
    assert stored["superseded_before_data"] is False
    assert stored["accepted_evidence_preserved"] is True
    assert stored["used_by_worker_successor"] is False
    assert stored["used_by_e3_e4_e5"] is False
    assert stored["worker"]["device"].startswith("cuda:")
    assert stored["predecessor"]["state"]["completed_prefix_count"] > 0
    assert stored["predecessor"]["state"]["counters"]["training"] == 0
    assert stored["predecessor"]["state"]["counters"]["test_access"] == 0
    assert stored["relocation_provenance_id"] == v3s._id(
        "g8erelocationv3s-",
        {key: child for key, child in stored.items() if key != "relocation_provenance_id"},
    )


def test_validate_contract_rejects_foreign_epoch_shapes():
    with pytest.raises(ValueError):
        v3s.validate_contract({"schema_version": 2})
    with pytest.raises(ValueError):
        v3s.validate_contract({
            "schema_version": 3,
            "checkpoint": "E1_corrected_v3",
            "status": "FROZEN_PRE_DATA_EXECUTABLE",
            "artifact_role": "g8_e_v3s_executable_pre_data_worker_successor_contract",
            "contract_id": "x",
            "campaign_id": "y",
            "authorization": {},
            "safety": {},
            "execution_profile": {},
        })


def test_worker_profile_block_validation_rejects_foreign_profile_or_device():
    good = {
        "profile_id": "confessor_pascal_cu126",
        "device": "cuda:0",
        "config_hash": "a" * 64,
        "lock_file": "requirements-pascal.lock",
        "lock_file_sha256": "d3561c8e930797d328cf45df1bdc5085842833665f9b5c9e5617d4c891e31a82",
        "opportunistic_host_change_forbidden": True,
        "profile_frozen_before_first_measurement": True,
        "sole_writer": "confessor",
    }
    v3s._validate_worker_profile_block(good)
    wrong_profile = dict(good, profile_id="local_4060_cu130")
    with pytest.raises(ValueError):
        v3s._validate_worker_profile_block(wrong_profile)
    wrong_lock = dict(good, lock_file="requirements.lock")
    with pytest.raises(ValueError):
        v3s._validate_worker_profile_block(wrong_lock)
    vague_device = dict(good, device="gpu")
    with pytest.raises(ValueError):
        v3s._validate_worker_profile_block(vague_device)


def _authorized_fixture(tmp_path: Path):
    fixture = fixture_context()
    authorization = tmp_path / "synthetic-owner-authorization.json"
    _authorization(authorization, fixture)
    return fixture, authorization


def test_authorization_binds_the_exact_successor_identity(tmp_path):
    fixture, authorization = _authorized_fixture(tmp_path)
    value = v3s.authenticate_owner_authorization(
        authorization, fixture["contract"], fixture["data_identity"]
    )
    assert value["status"] == "AUTHORIZED"


def test_authorization_rejects_a_substituted_profile(tmp_path):
    fixture, authorization = _authorized_fixture(tmp_path)
    body = json.loads(authorization.read_text())
    del body["issued_sha256"]
    body["profile_id"] = "local_4060_cu130"
    body["issued_sha256"] = v3.sha256_bytes(v3.canonical_json(body))
    authorization.write_bytes(v3.rendered_json(body))
    with pytest.raises(ValueError):
        v3s.authenticate_owner_authorization(
            authorization, fixture["contract"], fixture["data_identity"]
        )


def test_authorization_digest_catches_any_scope_edit(tmp_path):
    fixture, authorization = _authorized_fixture(tmp_path)
    body = json.loads(authorization.read_text())
    del body["issued_sha256"]
    body["scope"] = dict(body["scope"], pass_one=True)
    body["issued_sha256"] = v3.sha256_bytes(v3.canonical_json(body))
    authorization.write_bytes(v3.rendered_json(body))
    with pytest.raises(ValueError):
        v3s.authenticate_owner_authorization(
            authorization, fixture["contract"], fixture["data_identity"]
        )


def test_synthetic_start_resume_completion_e3_and_e4(tmp_path, monkeypatch):
    import baseline.g8_d as g8d

    monkeypatch.setattr(g8d, "account_br11", _br11_stub)
    fixture, authorization = _authorized_fixture(tmp_path)
    runtime = tmp_path / "runtime"
    common = [
        "--campaign-id", fixture["contract"]["campaign_id"],
        "--runtime-root", str(runtime),
        "--authorization", str(authorization),
    ]
    partial = dict(fixture)
    partial["max_units"] = 2
    assert runner_cli.main(["--start", *common], fixture=partial) == 0
    state = v3._state_for_runtime(runtime)
    assert state["completed_prefix_count"] == 2
    assert runner_cli.main(["--resume", *common], fixture=dict(fixture)) == 0
    completed, completion_sha = v3.verify_e2_completion_artifact(
        runtime_root=runtime,
        contract=fixture["contract"],
        authority=fixture["authority"],
        production=False,
    )
    assert completed["status"] == "E2_COMPLETE"
    assert completion_sha == v3.sha256_file(runtime / "e2_completion.json")
    merge_fixture = {
        "contract": fixture["contract"],
        "authority": fixture["authority"],
        "sample_ids": fixture["sample_ids"],
        "sample_labels": fixture["sample_labels"],
    }
    assert merge_cli.main(["--execute", "--runtime-root", str(runtime)], fixture=merge_fixture) == 0
    e3_path = runtime / "e3_exact_set_closure.json"
    e3_sha = v3.sha256_file(e3_path)
    e3 = v3.verify_e3_artifact(e3_path, contract=fixture["contract"], expected_sha256=e3_sha)
    assert e3["observed_work_unit_count"] == len(fixture["sample_ids"]) * 3
    assert aggregate_cli.main(
        ["--execute", "--runtime-root", str(runtime), "--e3", str(e3_path), "--e3-sha256", e3_sha],
        fixture=merge_fixture,
    ) == 0
    e4 = v3.verify_e4_artifact(
        runtime / "e4_count_derived.json",
        contract=fixture["contract"],
        e3_path=e3_path,
        e3_sha256=e3_sha,
    )
    assert e4["record_traversal_count"] == len(fixture["sample_ids"]) * 3
    for obj in e4["objects"]:
        if obj["status"] == "eligible":
            assert obj["total_count"] == len(fixture["sample_ids"])
            assert obj["correct_count"] <= obj["total_count"]


def test_live_identity_check_rejects_the_contract_summary_block():
    """Regression: the runner must authenticate the FULL data-identity file,
    never the contract's summary block, against the live rebuild."""

    if not v3s.V3S_CONTRACT_PATH.is_file():
        pytest.skip("worker-successor contract is frozen on the worker host")
    contract = json.loads(v3s.V3S_CONTRACT_PATH.read_text())
    with pytest.raises(ValueError):
        v3.verify_live_validation_identity(contract["scientific_data_identity"])
    # The tracked repository path named by the contract is the authenticator.
    data_identity_path = Path(v3s.__file__).resolve().parents[2] / str(
        contract["scientific_data_identity"]["path"]
    )
    assert data_identity_path.is_file()
    data_identity, _ = v3s._rendered_object(data_identity_path, "v3 reused data identity")
    assert data_identity["data_identity_id"] == contract["scientific_data_identity"]["id"]


def test_production_e3_rejects_merge_ineligible_fixture_records(tmp_path, monkeypatch):
    import baseline.g8_d as g8d

    monkeypatch.setattr(g8d, "account_br11", _br11_stub)
    fixture, authorization = _authorized_fixture(tmp_path)
    runtime = tmp_path / "runtime"
    common = [
        "--campaign-id", fixture["contract"]["campaign_id"],
        "--runtime-root", str(runtime),
        "--authorization", str(authorization),
    ]
    assert runner_cli.main(["--start", *common], fixture=dict(fixture)) == 0
    with pytest.raises(ValueError):
        v3.build_e3_artifact(
            authority=fixture["authority"],
            sample_ids=fixture["sample_ids"],
            sample_labels=fixture["sample_labels"],
            runtime_root=runtime,
            contract=fixture["contract"],
            production=True,
        )
