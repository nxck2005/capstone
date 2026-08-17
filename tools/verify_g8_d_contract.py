#!/usr/bin/env python3
"""Independent verifier for the G8_D D1 contract.

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


def validate(path: Path = CONTRACT) -> dict[str, Any]:
    value = _read(path, "G8_D contract")
    required = {
        "schema_version", "artifact_role", "phase", "checkpoint", "status", "contract_id", "campaign_id",
        "g8_c_binding", "d0_open_binding", "validation_split_bindings", "classifier_binding", "codec_binding",
        "upstream_bindings", "identity_schema", "cache_schema", "record_schema", "resume_schema",
        "work_unit_ordering", "phase_order", "source_bindings", "safety", "next_gate",
    }
    _require(set(value) == required, "G8_D contract schema differs")
    _require((value["schema_version"], value["artifact_role"], value["phase"], value["checkpoint"], value["status"]) == (1, "g8_d_validation_measurement_contract", "G8_D", "D2", "codec_search_ready"), "G8_D contract header differs")
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
    _require(value["next_gate"] == "G8_D/D3", "next gate is not D3")
    _require(value["identity_schema"]["schema_version"] == IDENTITY_SCHEMA_VERSION, "identity schema version differs")
    _require(value["cache_schema"]["codec_cache_schema_version"] == CODEC_CACHE_SCHEMA_VERSION, "codec cache schema differs")
    _require(value["record_schema"]["schema_version"] == RECORD_SCHEMA_VERSION and value["record_schema"]["measured_accuracy_requires_counts"] is True, "record schema differs")
    _require(value["resume_schema"]["schema_version"] == RESUME_SCHEMA_VERSION and value["resume_schema"]["completed_must_be_exact_prefix"] is True, "resume schema differs")
    _require(value["cache_schema"]["emitted_bytes_authoritative"] is True, "emitted bytes are not authoritative")
    _require(value["cache_schema"]["structural_infeasibility_is_distinct"] is True and value["cache_schema"]["codec_infeasibility_is_recorded"] is True, "infeasibility semantics differ")
    _require(value["cache_schema"]["source_bytes_bound"] is True and value["cache_schema"]["canonical_pixels_bound"] is True, "image binding is incomplete")
    safety = value["safety"]
    _require(all(safety[field] is False for field in ("validation_campaign_started", "selection_started", "pass_one_started", "pass_two_started", "training_started", "test_split_accessed", "g8_e_started")), "G8_D safety flags are nonzero")
    _require(all(safety[field] == 0 for field in ("test_access", "inference", "training", "validation_decoding")), "G8_D protected counters are nonzero")

    for binding in value["source_bindings"]:
        _require(set(binding) == {"path", "role", "sha256"}, "source binding schema differs")
        _digest(binding["sha256"], f"source {binding['path']} SHA")
        _require(_sha_file(REPO / binding["path"]) == binding["sha256"], f"source changed: {binding['path']}")
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
