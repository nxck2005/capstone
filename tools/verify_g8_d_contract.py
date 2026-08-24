#!/usr/bin/env python3
"""Independent verifier for the G8_D D7 handoff contract.

The expected schema and safety assertions are intentionally restated here;
the verifier does not import the contract builder.  It does import the final
G8_C loader only to ensure the bound table remains a valid runtime input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline.g8_d import (  # noqa: E402
    CODEC_CACHE_SCHEMA_VERSION,
    G8_D_SCHEMA_VERSION,
    IDENTITY_SCHEMA_VERSION,
    RECORD_SCHEMA_VERSION,
    RESUME_SCHEMA_VERSION,
    rendered_json,
)
from baseline.g8_pascal_merge import load_successor_bler_table  # noqa: E402


HEX = re.compile(r"^[0-9a-f]{64}$")
CONTRACT = REPO / "results/baseline/g8_d/measurement_contract.json"
AM87_SOURCE_COMPATIBILITY = REPO / "results/baseline/g8_f/am87_g8e_source_compatibility.json"
AM88_SOURCE_COMPATIBILITY = REPO / "results/baseline/g8_f/am88_g8e_source_compatibility.json"
_HISTORICAL_VERIFIER_SHA256 = "f3b0fcdd719f5e0b43e684226acf607f9d6dea759236b44246845416e7dbd0d7"
_CURRENT_VERIFIER_PROJECTION_SHA256 = "fd18eaa2ba904f5d926f654c0f1f39d14178be01d7c6b636d8ae4b57b6f57da7"


class G8DContractVerificationError(ValueError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    try:
        return _sha_bytes(path.read_bytes())
    except OSError as exc:
        raise G8DContractVerificationError(f"cannot hash {path}: {exc}") from exc


def _verifier_projection(source: bytes) -> bytes:
    pattern = rb'(?m)^(_CURRENT_VERIFIER_PROJECTION_SHA256\s*=\s*)["\'][^"\']+["\']'
    projected, count = re.subn(pattern, rb'\1"<exact-compatibility-binding>"', source, count=1)
    _require(count == 1, "AM-87 verifier compatibility binding is missing")
    return projected


def _read(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise G8DContractVerificationError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict) or raw != rendered_json(value):
        raise G8DContractVerificationError(f"{label} is not canonical rendered JSON")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G8DContractVerificationError(message)


def _digest(value: Any, label: str) -> None:
    _require(isinstance(value, str) and HEX.fullmatch(value) is not None, f"{label} is not a SHA-256")


def _verify_am87_g8d_source(binding: dict[str, Any]) -> None:
    am87 = _read(AM87_SOURCE_COMPATIBILITY, "AM-87 G8_E source compatibility")
    compatibility = _read(AM88_SOURCE_COMPATIBILITY, "AM-88 G8_E source compatibility")
    am87_body = {key: child for key, child in am87.items() if key != "compatibility_id"}
    body = {key: child for key, child in compatibility.items() if key != "compatibility_id"}
    _require(am87.get("compatibility_id") == "g8esourcecompat-" + _sha_bytes(_canonical(am87_body)), "AM-87 G8_E compatibility ID differs")
    _require(compatibility.get("compatibility_id") == "g8esourcecompat-" + _sha_bytes(_canonical(body)), "AM-88 G8_E compatibility ID differs")
    _require(
        compatibility.get("amendment") == "AM-88"
        and compatibility.get("timing") == "post_am87_pre_f0_execution_zero"
        and compatibility.get("protected_boundary", {}).get("g8_d_changed") is False,
        "AM-88 G8_E source-compatibility boundary differs",
    )
    prior_entries = am87.get("entries")
    entries = compatibility.get("entries")
    _require(isinstance(prior_entries, list) and isinstance(entries, list), "AM-87/AM-88 G8_E source entries differ")
    prior = [entry for entry in prior_entries if isinstance(entry, dict) and entry.get("path") == binding["path"]]
    matches = [entry for entry in entries if isinstance(entry, dict) and entry.get("path") == binding["path"]]
    _require(len(prior) == len(matches) == 1, "AM-87/AM-88 G8_D source entry differs")
    entry = matches[0]
    _require(
        prior[0].get("archived_sha256") == binding["sha256"]
        and entry.get("archived_sha256") == prior[0].get("current_sha256")
        and entry.get("current_sha256") == _sha_file(REPO / binding["path"])
        and entry.get("scientific_execution_reachable") is False,
        "AM-87/AM-88 G8_D source byte chain differs",
    )


def validate(path: Path = CONTRACT) -> dict[str, Any]:
    value = _read(path, "G8_D contract")
    required = {
        "schema_version", "artifact_role", "phase", "checkpoint", "status", "contract_id", "campaign_id",
        "g8_c_binding", "d0_open_binding", "validation_split_bindings", "classifier_binding", "codec_binding",
        "upstream_bindings", "identity_schema", "cache_schema", "record_schema", "resume_schema", "smoke_schema", "handoff_schema",
        "work_unit_ordering", "phase_order", "source_bindings", "safety", "next_gate",
    }
    _require(set(value) == required, "G8_D contract schema differs")
    _require((value["schema_version"], value["artifact_role"], value["phase"], value["checkpoint"], value["status"]) == (1, "g8_d_validation_measurement_contract", "G8_D", "D7", "handoff_ready"), "G8_D contract header differs")
    contract_id = value["contract_id"]
    campaign_id = value["campaign_id"]
    _require(isinstance(contract_id, str) and contract_id.startswith("g8dcontract-"), "contract ID prefix differs")
    _require(isinstance(campaign_id, str) and campaign_id.startswith("g8d-"), "campaign ID prefix differs")
    campaign_basis = dict(value)
    campaign_basis.pop("campaign_id")
    campaign_basis.pop("contract_id")
    expected_campaign = "g8d-" + _sha_bytes(_canonical(campaign_basis))
    _require(campaign_id == expected_campaign, "campaign ID does not reproduce")
    contract_basis = dict(value)
    contract_basis.pop("contract_id")
    expected_contract = "g8dcontract-" + _sha_bytes(_canonical(contract_basis))
    _require(contract_id == expected_contract, "contract ID does not reproduce")

    g8c = value["g8_c_binding"]
    _require(g8c["table_id"] == "g8pblertable-69ecc729f3b7dc3d67c0a3a5d8cf071cab927ad0a1e0cd5b18a6bbe674b9126f", "wrong Pascal successor table ID")
    _require(g8c["curves"] == 153 and g8c["measured_points"] == 3213 and g8c["trials_per_point"] == 5000, "Pascal table coverage differs")
    _require(g8c["predecessor_table_contribution"] == "none", "predecessor evidence is not isolated")
    for field in ("table_sha256", "merge_report_sha256", "closeout_sha256", "production_contract_sha256"):
        _digest(g8c[field], f"G8_C {field}")
    _require(value["d0_open_binding"]["artifact_id"].startswith("g8dopen-"), "D0 binding ID differs")
    _digest(value["d0_open_binding"]["artifact_sha256"], "D0 artifact SHA")
    _require(_sha_file(REPO / "results/baseline/g8_d/d0_open.json") == value["d0_open_binding"]["artifact_sha256"], "D0 artifact bytes changed")

    loaded = load_successor_bler_table()
    _require(len(loaded.identities) == 153, "successor loader exposes the wrong curve count")
    table_path = REPO / "results/baseline/g8_pascal_successor/successor_bler_table.json"
    merge_path = REPO / "results/baseline/g8_pascal_successor/successor_bler_merge_report.json"
    closeout_path = REPO / "results/baseline/g8_pascal_successor/successor_closeout_provenance.json"
    _require(_sha_file(table_path) == g8c["table_sha256"], "successor table bytes changed")
    _require(_sha_file(merge_path) == g8c["merge_report_sha256"], "successor merge bytes changed")
    _require(_sha_file(closeout_path) == g8c["closeout_sha256"], "successor closeout bytes changed")

    splits = value["validation_split_bindings"]
    _require(isinstance(splits, list) and len(splits) == 3, "validation split binding count differs")
    for split in splits:
        _require(set(split) == {"schema_version", "identity_type", "dataset", "split", "dataset_version", "manifest_sha256"}, "split identity schema differs")
        _require(split["split"] == "val", "non-validation split is bound")
        _digest(split["dataset_version"], f"{split['dataset']} dataset version")
        _digest(split["manifest_sha256"], f"{split['dataset']} manifest")

    classifier = value["classifier_binding"]
    _require(classifier["variant"] == "clean" and classifier["dataset"] == "imagenette160" and classifier["split"] == "val", "classifier binding differs")
    for field in ("checkpoint_sha256", "classifier_config_sha256", "dataset_version", "manifest_sha256"):
        _digest(classifier[field], f"classifier {field}")
    _require(set(value["phase_order"]) == {"D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7"}, "phase order differs")
    _require(value["next_gate"] == "G8_E/E0", "next gate is not G8_E E0")
    _require(value["identity_schema"]["schema_version"] == IDENTITY_SCHEMA_VERSION, "identity schema version differs")
    _require(value["cache_schema"]["codec_cache_schema_version"] == CODEC_CACHE_SCHEMA_VERSION, "codec cache schema differs")
    _require(value["record_schema"]["schema_version"] == RECORD_SCHEMA_VERSION and value["record_schema"]["measured_accuracy_requires_counts"] is True, "record schema differs")
    resume = value["resume_schema"]
    _require(resume["schema_version"] == RESUME_SCHEMA_VERSION and resume["completed_must_be_exact_prefix"] is True, "resume schema differs")
    _require(resume["state_fields"] == [
        "schema_version", "artifact_role", "campaign_id", "contract_id", "work_unit_order",
        "completed_work_unit_ids", "in_progress_work_unit_id", "in_progress_record_id",
        "in_progress_cache_object_id", "in_progress_cache_reference_sha256", "record_refs",
        "cache_refs", "aggregate_ref", "state_sha256",
    ], "resume state fields differ")
    _require(resume["campaign_lock_is_exclusive"] is True and resume["state_publication_is_atomic"] is True, "resume locking/publication differs")
    _require(resume["same_directory_fsync_required"] is True and resume["aggregate_history_must_be_exact_prefix"] is True, "resume durability differs")
    _require(resume["complete_output_is_reused"] is True and resume["silent_overwrite_is_forbidden"] is True, "resume reuse/overwrite semantics differ")
    _require(resume["crash_recovery_hooks"] == [
        "before_cache_publication", "after_cache_publication", "before_record_publication",
        "after_record_publication", "before_aggregate_publication", "after_aggregate_publication",
        "before_state_publication", "after_state_publication",
    ], "resume crash boundary schema differs")
    _require(value["cache_schema"]["emitted_bytes_authoritative"] is True, "emitted bytes are not authoritative")
    _require(value["cache_schema"]["structural_infeasibility_is_distinct"] is True and value["cache_schema"]["codec_infeasibility_is_recorded"] is True, "infeasibility semantics differ")
    _require(value["cache_schema"]["source_bytes_bound"] is True and value["cache_schema"]["canonical_pixels_bound"] is True, "image binding is incomplete")
    _require(value["cache_schema"]["reconstruction_cache_binds_emitted_bytes"] is True, "reconstruction cache does not bind emitted bytes")
    _require(value["cache_schema"]["reconstruction_cache_binds_image_identity"] is True, "reconstruction cache does not bind image identity")
    _require(value["cache_schema"]["reconstruction_cache_binds_codec_identity"] is True, "reconstruction cache does not bind codec identity")
    _require(value["cache_schema"]["reconstruction_cache_immutable"] is True, "reconstruction cache is not immutable")
    _require(value["cache_schema"]["br11_accounting_rule"] == "AM-81", "BR-11 accounting rule differs")
    _require(value["cache_schema"]["br11_header_is_structural_codestream_bytes"] is True, "BR-11 header semantics differ")
    _require(value["cache_schema"]["br11_payload_is_all_tile_part_data"] is True, "BR-11 payload semantics differ")
    _require(value["cache_schema"]["br11_filler_is_separate"] is True and value["cache_schema"]["br11_includes_decode_failures"] is True, "BR-11 denominator semantics differ")
    _require(value["record_schema"]["clean_classifier_record_is_count_derived"] is True, "clean classifier records are not count-derived")
    _require(value["record_schema"]["clean_classifier_record_validation_only"] is True, "clean classifier records are not validation-only")
    _require(value["record_schema"]["clean_classifier_record_binds_classifier"] is True, "clean classifier identity is not bound")
    _require(value["record_schema"]["clean_classifier_record_binds_reconstruction_cache"] is True, "reconstruction cache identity is not bound")
    _require(value["record_schema"]["clean_classifier_record_merge_eligible"] is False, "D4 records are incorrectly merge-eligible")
    smoke = value["smoke_schema"]
    _require(smoke["schema_version"] == 1 and smoke["artifact_role"] == "g8_d_bounded_non_scientific_smoke", "smoke schema header differs")
    _require(smoke["label"] == "NON-SCIENTIFIC BOUNDED SMOKE" and smoke["non_scientific"] is True, "smoke label/scope differs")
    _require(smoke["non_selection"] is True and smoke["non_headline"] is True and smoke["merge_eligible"] is False, "smoke merge boundary differs")
    _require(smoke["synthetic_pixels_only"] is True and smoke["synthetic_codec_backend_only"] is True and smoke["synthetic_decoder_only"] is True, "smoke fixture boundary differs")
    _require(smoke["classifier_invocation_forbidden"] is True and smoke["test_access_forbidden"] is True, "smoke protected boundary differs")
    _require(smoke["mutation_case_count"] == 20 and len(smoke["mutation_case_names"]) == 20, "smoke mutation matrix count differs")
    _require(all(smoke[field] is False for field in ("full_validation_campaign_started", "selection_started", "training_started", "test_split_accessed")), "smoke execution flags are nonzero")
    handoff = value["handoff_schema"]
    _require(handoff["schema_version"] == 1 and handoff["artifact_role"] == "g8_d_handoff" and handoff["status"] == "GREEN", "handoff schema differs")
    _require(handoff["g8_c_unchanged"] is True and handoff["full_pytest_required"] is True and handoff["full_pytest_skipped_required"] == 0, "handoff verification requirements differ")
    _require(all(handoff[field] is False for field in ("full_validation_campaign_started", "selection_started", "training_started", "test_split_accessed", "g8_e_started")), "handoff safety flags are nonzero")
    _require(handoff["next_gate"] == "G8_E/E0", "handoff next gate differs")
    safety = value["safety"]
    _require(all(safety[field] is False for field in ("validation_campaign_started", "selection_started", "pass_one_started", "pass_two_started", "training_started", "test_split_accessed", "g8_e_started")), "G8_D safety flags are nonzero")
    _require(all(safety[field] == 0 for field in ("test_access", "inference", "training", "validation_decoding")), "G8_D protected counters are nonzero")

    for binding in value["source_bindings"]:
        _require(set(binding) == {"path", "role", "sha256"}, "source binding schema differs")
        _digest(binding["sha256"], f"source {binding['path']} SHA")
        current_sha = _sha_file(REPO / binding["path"])
        if current_sha != binding["sha256"]:
            if binding["path"] == "src/baseline/g8_d.py":
                _verify_am87_g8d_source(binding)
            elif binding["path"] == "tools/verify_g8_d_contract.py":
                current = (REPO / binding["path"]).read_bytes()
                _require(binding["sha256"] == _HISTORICAL_VERIFIER_SHA256, "historical G8_D verifier binding differs")
                _require(
                    _sha_bytes(_verifier_projection(current)) == _CURRENT_VERIFIER_PROJECTION_SHA256,
                    "current G8_D verifier compatibility bytes differ",
                )
            else:
                raise G8DContractVerificationError(f"source changed: {binding['path']}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=CONTRACT)
    args = parser.parse_args()
    try:
        value = validate(args.path)
    except G8DContractVerificationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "PASS", "contract_id": value["contract_id"], "campaign_id": value["campaign_id"], "next_gate": value["next_gate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
