"""Synthetic fail-closed mutation tests for the G8_E pre-data boundary."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from baseline.g8_e import (
    E1_AUTHORITY_PATH,
    E1_CONTRACT_PATH,
    E1_CORPUS_SPEC_PATH,
    G8EContractError,
    _record_id,
    canonical_json,
    sha256_bytes,
    validate_e1_candidate_authority,
    validate_e1_contract,
    validate_e1_corpus_spec,
    validate_e1_scientific_record,
)
from data.manifests import manifest_path, validate_manifest_bytes


REPO = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _reid(value: dict, field: str, prefix: str) -> dict:
    body = copy.deepcopy(value)
    body[field] = prefix + sha256_bytes(canonical_json({key: item for key, item in body.items() if key != field}))
    return body


def _reject(callable_):
    with pytest.raises(G8EContractError):
        callable_()


@pytest.fixture(scope="module")
def contract() -> dict:
    return _read(E1_CONTRACT_PATH)


@pytest.fixture(scope="module")
def authority() -> dict:
    return _read(E1_AUTHORITY_PATH)


@pytest.fixture(scope="module")
def corpus() -> dict:
    return _read(E1_CORPUS_SPEC_PATH)


def _contract_mutation(contract: dict, path: tuple[str, ...], value) -> dict:
    mutated = copy.deepcopy(contract)
    node = mutated
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return _reid(mutated, "contract_id", "g8econtract-")


@pytest.mark.parametrize(
    ("name", "path", "value"),
    [
        ("old predecessor BLER table", ("g8_c_binding", "table_id"), "g8pblertable-old-predecessor"),
        ("wrong portable manifest", ("g8_c_binding", "portable_manifest_id"), "g8pportable-wrong"),
        ("portable provenance mutation", ("g8_c_binding", "portable_provenance_id"), "g8pportableprov-mutated"),
        ("wrong G8_D contract", ("g8_d_binding", "contract_id"), "g8dcontract-wrong"),
        ("old pre-repair G8_D contract", ("g8_d_binding", "contract_id"), "g8dcontract-old-pre-repair"),
        ("wrong classifier", ("g1_classifier_binding", "classifier_config_sha256"), "f" * 64),
        ("wrong validation manifest", ("dataset_boundary", "initial_scientific_dataset", "manifest", "sha256"), "e" * 64),
    ],
)
def test_upstream_and_dataset_mutations_are_rejected(contract, name, path, value):
    del name
    _reject(lambda: validate_e1_contract(_contract_mutation(contract, path, value), verify_live_assets=False))


def _record_fixture(contract: dict, authority: dict) -> dict:
    rows = validate_manifest_bytes(
        "imagenette160",
        manifest_path("imagenette160").read_bytes(),
    )
    image = next(row for row in rows if row.split == "val")
    candidate_ordinal, candidate = next(
        (ordinal, row)
        for ordinal, row in enumerate(authority["candidates"])
        if row["dataset"] == "imagenette160"
    )
    initial = contract["dataset_boundary"]["initial_scientific_dataset"]
    split_identity = {
        "dataset": "imagenette160",
        "split": "val",
        "dataset_version": initial["dataset_version"],
        "manifest_sha256": initial["manifest"]["sha256"],
        "stable_id_set_sha256": initial["manifest"]["validation_id_set_sha256"],
    }
    zero = "0" * 64
    g8c = contract["g8_c_binding"]
    g8c_record = {
        key: g8c[key]
        for key in (
            "table_id",
            "table_sha256",
            "merge_report_id",
            "merge_report_sha256",
            "historical_c6_id",
            "historical_c6_sha256",
            "portable_manifest_id",
            "portable_manifest_sha256",
            "portable_provenance_id",
            "portable_provenance_sha256",
            "portable_scientific_runtime_sha256",
            "portable_verification_epoch",
        )
    }
    record = {
        "schema_version": 1,
        "artifact_role": "g8_e_scientific_measurement_record",
        "record_id": "pending",
        "campaign_id": contract["campaign_id"],
        "contract_id": contract["contract_id"],
        "authority_ordinal": candidate_ordinal,
        "authority_candidate_id": candidate["candidate_id"],
        "work_unit_id": "g8eunit-" + zero,
        "dataset": "imagenette160",
        "dataset_role": "headline",
        "validation_split": "val",
        "validation_split_identity": split_identity,
        "image_identity": {
            "stable_sample_id": image.stable_sample_id,
            "label": image.label,
            "dataset": "imagenette160",
            "split": "val",
            "image_identity_id": "g8dimage-" + zero,
        },
        "source_image": {
            "source_bytes_sha256": zero,
            "canonical_pixels_sha256": zero,
            "canonical_shape": [160, 160, 3],
            "source_payload_rule": "exact_encoded_jpeg_file_bytes",
        },
        "candidate": candidate,
        "budget_identity": {
            "ratio": candidate["ratio"],
            "k_symbols": 1,
            "payload_budget_bytes": 1,
            "packet_config_id": candidate["packet_config_id"],
        },
        "codec_configuration": {
            "identity_type": "jpeg2000_configuration",
            "configuration_hash": contract["codec_and_preprocessing"]["configuration_hash"],
            "snapshot_sha256": contract["codec_and_preprocessing"]["snapshot_sha256"],
            "runtime_version": "2.5.4",
            "encode_axis_px": candidate["encode_axis_px"],
        },
        "emitted_codestream": {
            "emitted_file_identity_id": "g8demitted-" + zero,
            "codestream_sha256": zero,
            "emitted_bytes": 1,
            "payload_budget_bytes": 1,
            "filler_bytes": 0,
            "actual_bytes_authoritative": True,
        },
        "reconstruction": {
            "identity_id": "g8drecon-" + zero,
            "decoded_pixels_sha256": zero,
            "output_shape": [160, 160, 3],
            "codec_configuration_hash": contract["codec_and_preprocessing"]["configuration_hash"],
            "image_identity_id": "g8dimage-" + zero,
        },
        "reconstruction_cache_object_id": "g8dreconobj-" + zero,
        "br11": {"accounting_rule": "AM-81", "verdict": "delivered"},
        "g8_c_table": g8c_record,
        "bler_linkage": {
            "lookup_identity": {"candidate_id": candidate["candidate_id"]},
            "table_id": g8c["table_id"],
            "table_sha256": g8c["table_sha256"],
            "lookup_mode": "exact_frozen_identity_and_exact_snr_or_explicit_uncharacterized",
            "interpolation": False,
            "extrapolation": False,
            "uncharacterized_is_ineligible": True,
        },
        "classifier": {"identity_type": "g1_clean_classifier"},
        "outcome": {"status": "delivered", "selection_eligible": True, "failure_semantics": "measured_delivery"},
        "correct_count": 1,
        "total_count": 1,
        "accuracy_derivation": "sum correct_count / sum total_count; no caller-supplied accuracy field",
        "provenance": {
            "source_manifest_id": contract["source_manifest_binding"]["manifest_id"],
            "source_manifest_sha256": contract["source_manifest_binding"]["sha256"],
            "source_commit": contract["source_manifest_binding"]["source_commit"],
            "execution_profile_id": contract["execution_profile"]["profile_id"],
            "execution_profile_selection_sha256": contract["execution_profile"]["selection"]["selection_sha256"],
            "contract_id": contract["contract_id"],
        },
        "validation_only": True,
        "test_access": 0,
        "training": False,
        "scientific_evidence": True,
        "merge_eligible": True,
    }
    record["record_id"] = _record_id(record)
    validate_e1_scientific_record(record, contract=contract, authority=authority)
    return record


def test_d4_non_scientific_record_cannot_enter_e():
    _reject(lambda: validate_e1_scientific_record({"schema_version": 1, "artifact_role": "g8_d_clean_classifier_measurement"}))


def test_test_split_record_is_rejected(contract, authority):
    record = _record_fixture(contract, authority)
    record["validation_split"] = "test"
    record["record_id"] = _record_id(record)
    _reject(lambda: validate_e1_scientific_record(record, contract=contract, authority=authority))


def test_wrong_dataset_role_record_is_rejected(contract, authority):
    record = _record_fixture(contract, authority)
    record["dataset_role"] = "fallback_headline"
    record["record_id"] = _record_id(record)
    _reject(lambda: validate_e1_scientific_record(record, contract=contract, authority=authority))


def test_codec_configuration_mutation_is_rejected(contract, authority):
    record = _record_fixture(contract, authority)
    record["codec_configuration"]["configuration_hash"] = "f" * 64
    record["record_id"] = _record_id(record)
    _reject(lambda: validate_e1_scientific_record(record, contract=contract, authority=authority))


def test_wrong_classifier_record_is_rejected(contract, authority):
    record = _record_fixture(contract, authority)
    record["classifier"]["identity_type"] = "wrong_classifier"
    record["record_id"] = _record_id(record)
    _reject(lambda: validate_e1_scientific_record(record, contract=contract, authority=authority))


def test_bare_accuracy_float_is_rejected(contract, authority):
    record = _record_fixture(contract, authority)
    record["accuracy"] = 1.0
    record["record_id"] = _record_id(record)
    _reject(lambda: validate_e1_scientific_record(record, contract=contract, authority=authority))


def test_missing_candidate_is_rejected(authority):
    mutated = copy.deepcopy(authority)
    mutated["candidates"].pop()
    _reject(lambda: validate_e1_candidate_authority(mutated))


def test_duplicate_candidate_is_rejected(authority):
    mutated = copy.deepcopy(authority)
    mutated["candidates"][-1] = copy.deepcopy(mutated["candidates"][0])
    _reject(lambda: validate_e1_candidate_authority(mutated))


def test_same_count_wrong_candidate_identity_is_rejected(authority):
    mutated = copy.deepcopy(authority)
    mutated["candidates"][0]["composition_candidate_identity"] = "wrong-identity"
    _reject(lambda: validate_e1_candidate_authority(mutated))


def test_snr_alias_is_rejected(authority):
    mutated = copy.deepcopy(authority)
    mutated["candidates"][0]["snr_db"] = 8
    _reject(lambda: validate_e1_candidate_authority(mutated))


@pytest.mark.parametrize(
    ("field", "new_value"),
    [
        ("pass_one_started", True),
        ("authorization_issued", True),
        ("pass_two_started", True),
    ],
)
def test_pass_state_mutations_are_rejected(contract, field, new_value):
    mutated = copy.deepcopy(contract)
    mutated["pass_one_preconditions"][field] = new_value
    mutated = _reid(mutated, "contract_id", "g8econtract-")
    _reject(lambda: validate_e1_contract(mutated, verify_live_assets=False))


def test_corpus_schema_with_validation_or_test_id_is_rejected(corpus):
    mutated = copy.deepcopy(corpus)
    mutated["train_manifest"]["stable_sample_ids"] = ["validation-or-test-id"]
    mutated = _reid(mutated, "corpus_spec_id", "g8ecorpusspec-")
    _reject(lambda: validate_e1_corpus_spec(mutated))


def test_e2_guard_performs_no_scientific_execution(contract):
    runtime_root = REPO / contract["resume_and_custody"]["runtime_root"]
    assert not runtime_root.exists()
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/run_g8_e.py"),
            "--campaign-id",
            contract["campaign_id"],
            "--contract",
            str(E1_CONTRACT_PATH),
            "--runtime-root",
            str(runtime_root),
            "--profile",
            "local_4060_cu130",
            "--device",
            "cuda:0",
            "--start",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert not runtime_root.exists()
    assert contract["safety"]["measurement_coverage"] == 0
    assert contract["safety"]["validation_decoding"] == 0
    assert contract["safety"]["inference"] == 0
    assert contract["safety"]["training"] == 0
    assert contract["safety"]["test_access"] == 0
