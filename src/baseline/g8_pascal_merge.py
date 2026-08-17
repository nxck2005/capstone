"""Successor-specific G8_C merge and measured BLER-table closure.

The Pascal campaign is a completed, authenticated input.  This module is the
deterministic consumer for C3--C5; it is deliberately separate from the
predecessor G8 characterization modules.  It never executes a measurement,
opens a selection gate, or reads the test split.

The merge report carries the accepted result's raw count fields and the
request/result/state artifact digests.  The table is a projection of that
report only: every point is measured, no point is interpolated, imputed, or
extrapolated during construction.  The independent verifier in
``tools/verify_g8_pascal_closeout.py`` reconstructs the report from the
runtime instead of importing the builders below.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from baseline.classical import composition
from baseline.g8_campaign import canonical_json, rendered_json, sha256_bytes
from baseline.g8_pascal_production import (
    RESULT_ARTIFACT_ROLE,
    REQUIRED_COUNT,
    SUCCESSOR_PROFILE_ID,
    TRIALS_PER_IDENTITY,
    audit_campaign,
    inspect_unit,
    request_path,
    result_path,
    state_path,
    successor_bindings,
    validate_campaign_state,
    validate_runtime_namespace,
)
from baseline.g8_pascal_successor import SUCCESSOR_ROOT
from config.params import REPO_ROOT


class SuccessorMergeError(RuntimeError):
    """Successor evidence is not an eligible isolated table input."""


SUCCESSOR_RUNTIME_ROOT = SUCCESSOR_ROOT / "runtime"
MERGE_REPORT_PATH = SUCCESSOR_ROOT / "successor_bler_merge_report.json"
TABLE_PATH = SUCCESSOR_ROOT / "successor_bler_table.json"
PROVENANCE_PATH = SUCCESSOR_ROOT / "successor_closeout_provenance.json"

MERGE_SCHEMA_VERSION = 1
TABLE_SCHEMA_VERSION = 1
MERGE_ARTIFACT_ROLE = "g8_c_pascal_successor_bler_merge_report"
TABLE_ARTIFACT_ROLE = "g8_c_pascal_successor_bler_table"
MERGE_ID_PREFIX = "g8pmerge-"
TABLE_ID_PREFIX = "g8pblertable-"
PHASE = "G8_C"
MEASUREMENT_SOURCE_COMMIT = "426110b05161e73e4d819bdc01f4857c012d6d59"
PROTECTED_COUNTERS = {
    "inference": 0,
    "test_access": 0,
    "training": 0,
    "validation_decoding": 0,
}
HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")

# The C3--C7 artifacts were frozen before the checkout-portability repair.
# These values are historical compatibility anchors, not claims about the
# bytes currently implementing the closeout consumers.  The portable verifier
# authenticates the scientific runtime separately and uses these anchors only
# to prove that the old frozen artifacts remain the same artifacts.
HISTORICAL_MERGE_REPORT_ID = "g8pmerge-2e861c39d8981af0e2d57dc8ded5828b9ed56a1459491e04929b5e9c3418de89"
HISTORICAL_MERGE_REPORT_SHA256 = "71ec9fe2eef8905e7ab27876881ad6bc295415ebee443a45600873b73a0dc8a8"
HISTORICAL_TABLE_ID = "g8pblertable-69ecc729f3b7dc3d67c0a3a5d8cf071cab927ad0a1e0cd5b18a6bbe674b9126f"
HISTORICAL_TABLE_SHA256 = "2c330c4d68dd5b1274374cde9f1528900074f8ed3b2792467194f27aa0d7e7a5"
HISTORICAL_CLOSEOUT_ID = "g8pcloseout-8cc5be86e6bbb350ca35c1806686e751f4528ad32aa7083e9a754c4849feba70"
HISTORICAL_CLOSEOUT_SHA256 = "efda699748cc176c184e0ef2dbcd8fc6591afd3e2fa41a9af4853fb135e3953f"
HISTORICAL_RUNTIME_TREE_SHA256 = "dde5a45a2c58320b9b28e13afa459a8cbf2db1614939ad8ff790d42edc27f14b"
HISTORICAL_CLOSEOUT_SOURCE_DIGEST = "4b25ea37c3185c489a8db3f92edd65fb5611704eb752f06efb6d58e9aaf0bdde"
HISTORICAL_CLOSEOUT_SOURCE_ENTRIES = (
    {
        "path": "src/baseline/g8_pascal_merge.py",
        "role": "successor_c3_c5_merge_and_table_builder",
        "bytes": 49967,
        "sha256": "1343ca754c5f529cc36fd2676574ef83c064ab9452ea92828257ac2ca4d484f3",
    },
    {
        "path": "tools/closeout_g8_pascal_successor.py",
        "role": "successor_closeout_artifact_freezer",
        "bytes": 7953,
        "sha256": "e024dd76c52407b1fed8846adf0d5ff7d94e1ebb3462409e298fb67bb20a4559",
    },
    {
        "path": "tools/verify_g8_pascal_closeout.py",
        "role": "independent_successor_c3_c7_verifier",
        "bytes": 37313,
        "sha256": "c0cbe8ec3700556e80a781f1471fb453e6b5c8cc348731dc5f45b7058144f036",
    },
)

# These are the closeout consumers, not the production measurement sources.
# Their byte digest is carried by the C3/C5 artifacts and the C6 closure.
CLOSEOUT_SOURCE_PATHS = (
    ("src/baseline/g8_pascal_merge.py", "successor_c3_c5_merge_and_table_builder"),
    ("tools/closeout_g8_pascal_successor.py", "successor_closeout_artifact_freezer"),
    ("tools/verify_g8_pascal_closeout.py", "independent_successor_c3_c7_verifier"),
)

MERGE_FIELDS = (
    "schema_version",
    "artifact_role",
    "phase",
    "checkpoint",
    "report_id",
    "campaign_id",
    "execution_profile_id",
    "campaign_manifest_sha256",
    "production_contract_sha256",
    "production_source_manifest_sha256",
    "production_runner_contract_sha256",
    "required_bler_artifact_sha256",
    "required_authority_identity_set_sha256",
    "observed_authority_identity_set_sha256",
    "closeout_source_digest",
    "measurement_source_commit",
    "runtime_relative_path",
    "runtime_tree_sha256",
    "required_identity_count",
    "required_authority_order",
    "units",
    "accepted_count",
    "completed_count",
    "coverage_contribution_sum",
    "missing_count",
    "duplicate_count",
    "unknown_count",
    "failed_count",
    "terminal_invalid_count",
    "unresolved_count",
    "recoverable_count",
    "request_only_attempt_count",
    "failed_result_attempt_count",
    "retry_history_attempt_count",
    "retry_history_ordinals",
    "total_trials",
    "total_information_bits",
    "total_bit_errors",
    "total_block_errors",
    "coverage_complete",
    "merge_ordering",
    "interpolation_used",
    "imputation_used",
    "extrapolation_used",
    "test_access",
    "protected_counters",
    "old_result_ingest",
    "predecessor_campaign_id",
    "predecessor_table_contribution",
)

MERGE_UNIT_FIELDS = (
    "authority_ordinal",
    "work_unit_id",
    "required_work_unit_record_sha256",
    "identity",
    "snr_db",
    "source_packet_config_ids",
    "information_length",
    "codeword_length",
    "campaign_id",
    "execution_profile_id",
    "campaign_manifest_sha256",
    "production_contract_sha256",
    "production_source_manifest_sha256",
    "production_runner_contract_sha256",
    "measurement_source_commit",
    "attempt",
    "request_sha256",
    "request_content_sha256",
    "result_sha256",
    "result_content_sha256",
    "state_sha256",
    "raw_measurement",
    "trials_completed",
    "information_bits",
    "bit_errors",
    "block_errors",
    "ber",
    "bler",
    "bler_confidence_low",
    "bler_confidence_high",
    "confidence_interval_method",
    "confidence_interval_percent",
    "confidence_interval_role",
    "historical_attempts",
)

TABLE_FIELDS = (
    "schema_version",
    "artifact_role",
    "phase",
    "checkpoint",
    "table_id",
    "campaign_id",
    "execution_profile_id",
    "campaign_manifest_sha256",
    "production_contract_sha256",
    "production_source_manifest_sha256",
    "production_runner_contract_sha256",
    "required_bler_artifact_sha256",
    "required_authority_identity_set_sha256",
    "closeout_source_digest",
    "measurement_source_commit",
    "runtime_relative_path",
    "runtime_tree_sha256",
    "merge_report_id",
    "merge_report_sha256",
    "required_identity_count",
    "complete_identity_count",
    "measured_point_count",
    "trials_per_point",
    "total_trials",
    "total_information_bits",
    "total_bit_errors",
    "total_block_errors",
    "interpolation_used",
    "imputation_used",
    "extrapolation_used",
    "test_access",
    "protected_counters",
    "old_result_ingest",
    "predecessor_campaign_id",
    "predecessor_table_contribution",
    "curves",
)

TABLE_POINT_FIELDS = (
    "authority_ordinal",
    "work_unit_id",
    "snr_db",
    "trials",
    "information_bits",
    "bit_errors",
    "block_errors",
    "ber",
    "bler",
    "bler_confidence_low",
    "bler_confidence_high",
    "confidence_interval_method",
    "confidence_interval_percent",
    "confidence_interval_role",
    "raw_measurement",
    "request_sha256",
    "request_content_sha256",
    "result_sha256",
    "result_content_sha256",
    "state_sha256",
    "measurement_source_commit",
)


def _self_id(value: Mapping[str, Any], field: str, prefix: str) -> str:
    body = dict(value)
    body.pop(field, None)
    return prefix + sha256_bytes(canonical_json(body))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SuccessorMergeError(message)


def _strict_object(value: Mapping[str, Any], fields: Sequence[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise SuccessorMergeError(f"{label} schema differs")
    return dict(value)


def _digest(value: Any, label: str) -> None:
    _require(isinstance(value, str) and HEX_DIGEST.fullmatch(value) is not None, f"{label} is not a SHA-256 digest")


def _read_rendered_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise SuccessorMergeError(f"cannot read {label}: {exc}") from exc
    _require(isinstance(payload, dict), f"{label} is not an object")
    _require(raw == rendered_json(payload), f"{label} is not canonical rendered JSON")
    return payload, raw


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise SuccessorMergeError(f"cannot hash {path}: {exc}") from exc


def _authority_key(unit: Mapping[str, Any], ordinal: int) -> dict[str, Any]:
    identity = composition.BlerIdentity.from_mapping(unit["identity"]).as_key()
    _require(identity == unit["identity"], f"required identity {ordinal} is not canonical")
    required_digest = sha256_bytes(canonical_json(dict(unit)))
    return {
        "authority_ordinal": ordinal,
        "work_unit_id": unit["work_unit_id"],
        "required_work_unit_record_sha256": required_digest,
        "identity": identity,
        "snr_db": unit["snr_db"],
        "source_packet_config_ids": list(unit["source_packet_config_ids"]),
        "information_length": unit["information_length"],
        "codeword_length": unit["codeword_length"],
    }


def _result_authority_key(result_identity: Mapping[str, Any]) -> dict[str, Any]:
    identity = composition.BlerIdentity.from_mapping(result_identity["bler_identity"]).as_key()
    k, n = identity["k_and_n"]
    return {
        "authority_ordinal": result_identity["authority_ordinal"],
        "work_unit_id": result_identity["work_unit_id"],
        "required_work_unit_record_sha256": result_identity["required_work_unit_record_sha256"],
        "identity": identity,
        "snr_db": result_identity["snr_db"],
        "source_packet_config_ids": list(result_identity["source_packet_config_ids"]),
        "information_length": k,
        "codeword_length": n,
    }


def load_required_authority() -> tuple[list[dict[str, Any]], str, str]:
    """Load the exact frozen authority bytes and its ordered identity digest."""

    path = REPO_ROOT / "results/baseline/g8/required_bler_identities.json"
    payload, raw = _read_rendered_json(path, "required BLER authority")
    _require(_file_sha256(path) == successor_bindings()["required_bler_artifact_sha256"], "required BLER authority SHA differs from successor contract")
    units = payload.get("required_bler_work_units")
    _require(isinstance(units, list) and len(units) == REQUIRED_COUNT, "required BLER authority is not the exact 3213-cell grid")
    keys: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ordinal, unit in enumerate(units):
        _require(isinstance(unit, Mapping), f"required BLER authority unit {ordinal} is not an object")
        key = _authority_key(unit, ordinal)
        digest = sha256_bytes(canonical_json(key))
        _require(digest not in seen, f"required BLER authority contains duplicate identity at ordinal {ordinal}")
        seen.add(digest)
        keys.append(key)
    return [dict(unit) for unit in units], sha256_bytes(canonical_json(keys)), sha256_bytes(raw)


def closeout_source_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for relative, role in CLOSEOUT_SOURCE_PATHS:
        path = REPO_ROOT / relative
        _require(path.is_file(), f"successor closeout source is missing: {relative}")
        body = path.read_bytes()
        entries.append({"path": relative, "role": role, "bytes": len(body), "sha256": sha256_bytes(body)})
    return entries


def closeout_source_digest() -> str:
    return sha256_bytes(canonical_json(closeout_source_entries()))


def _historical_closeout_source_matches(value: Mapping[str, Any]) -> bool:
    """Return whether *value* is the exact frozen C3--C7 source epoch."""

    return (
        value.get("report_id") == HISTORICAL_MERGE_REPORT_ID
        and value.get("closeout_source_digest") == HISTORICAL_CLOSEOUT_SOURCE_DIGEST
    )


def normalized_runtime_tree_sha256(root: Path | str) -> str:
    """Hash the runtime using the same normalized tar stream as the handoff."""

    root_path = Path(root)
    _require(root_path.is_absolute(), "successor runtime root must be absolute for normalized hashing")
    try:
        completed = subprocess.run(
            [
                "tar",
                "--sort=name",
                "--mtime=UTC 1970-01-01 00:00:00",
                "--owner=0",
                "--group=0",
                "--numeric-owner",
                "-cf",
                "-",
                "-C",
                str(root_path),
                ".",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SuccessorMergeError(f"cannot normalize successor runtime tree: {exc}") from exc
    return sha256_bytes(completed.stdout)


def _read_artifact(path: Path, label: str) -> tuple[dict[str, Any], bytes, str]:
    payload, raw = _read_rendered_json(path, label)
    return payload, raw, sha256_bytes(raw)


def _read_runtime_json(path: Path, label: str) -> tuple[dict[str, Any], bytes, str]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise SuccessorMergeError(f"cannot read {label}: {exc}") from exc
    _require(isinstance(payload, dict), f"{label} is not an object")
    _require(raw == canonical_json(payload), f"{label} is not canonical runtime JSON")
    return payload, raw, sha256_bytes(raw)


def _runtime_state(root: Path, bindings: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    path = root / "campaign_state.json"
    payload, raw, digest = _read_runtime_json(path, "successor runtime campaign state")
    try:
        validate_campaign_state(payload, bindings=bindings)
    except Exception as exc:
        raise SuccessorMergeError(f"successor runtime campaign state is invalid: {exc}") from exc
    expected = {
        "accepted_authority_ordinals": list(range(REQUIRED_COUNT)),
        "in_progress_authority_ordinals": [],
        "failed_authority_ordinals": [],
        "terminal_invalid_authority_ordinals": [],
    }
    for field, value in expected.items():
        _require(payload[field] == value, f"successor runtime campaign state {field} is not complete")
    _require(payload["scientific_execution_performed"] is True, "successor runtime state does not record completed scientific execution")
    _require(payload["protected_counters"] == PROTECTED_COUNTERS, "successor runtime protected counters are nonzero")
    _require(payload["test_access"] == 0, "successor runtime test access is nonzero")
    _require(payload["old_result_ingest_permitted"] is False, "successor runtime permits old-result ingest")
    return payload, digest


def _raw_json(path: Path, label: str) -> tuple[dict[str, Any], bytes, str]:
    payload, raw, digest = _read_runtime_json(path, label)
    return payload, raw, digest


def _history_for_unit(root: Path, report: Mapping[str, Any], unit: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int, int, int]:
    attempts = sorted(set(report["request_attempts"]) | set(report["result_attempts"]))
    _require(attempts == list(range(1, max(attempts) + 1)), f"authority ordinal {report['ordinal']} has a retry-history gap")
    history: list[dict[str, Any]] = []
    request_only = 0
    failed_results = 0
    retries = max(0, len(attempts) - 1)
    for attempt in attempts:
        request, request_raw, request_sha = _raw_json(
            request_path(root, unit["work_unit_id"], attempt),
            f"successor request ordinal {report['ordinal']} attempt {attempt}",
        )
        result_record = report["validated_results"].get(attempt)
        result = None
        result_sha = None
        result_raw = None
        if result_record is not None:
            result, result_raw, result_sha = _raw_json(
                result_path(root, unit["work_unit_id"], attempt),
                f"successor result ordinal {report['ordinal']} attempt {attempt}",
            )
            _require(result_record["sha256"] == result_sha, f"result hash changed while merging ordinal {report['ordinal']} attempt {attempt}")
        else:
            request_only += 1
        if result is not None:
            measurement = result["measurement"]
            contribution = result["disposition"]["required_coverage_contribution"]
            merge_eligible = result["disposition"]["merge_eligible"]
            status = result["status"]
            _require(status in {"complete", "failed"}, f"unknown result status in ordinal {report['ordinal']} history")
            if status == "failed":
                failed_results += 1
                _require(contribution == 0 and merge_eligible is False, f"failed history for ordinal {report['ordinal']} contributes to coverage")
            if status == "complete":
                _require(contribution == 1 and merge_eligible is True, f"complete history for ordinal {report['ordinal']} is not merge eligible")
        else:
            measurement = None
            contribution = 0
            merge_eligible = False
            status = None
        history.append(
            {
                "attempt": attempt,
                "request_sha256": request_sha,
                "request_content_sha256": None,
                "result_sha256": result_sha,
                "result_content_sha256": None if result is None else result["result_sha256"],
                "result_status": status,
                "merge_eligible": merge_eligible,
                "required_coverage_contribution": contribution,
                "trials_completed": None if measurement is None else measurement["trials_completed"],
                "test_access": 0 if result is None else result["disposition"]["test_access"],
                "protected_counters": PROTECTED_COUNTERS if result is None else result["disposition"]["protected_counters"],
            }
        )
        _ = request_raw, result_raw, request
    return history, request_only, failed_results, retries


def _collect_successor_evidence(root: Path | str) -> list[dict[str, Any]]:
    try:
        summary = audit_campaign(root)
        validate_runtime_namespace(root)
    except Exception as exc:
        raise SuccessorMergeError(f"successor evidence cannot be audited: {exc}") from exc
    root_path = Path(root).resolve()
    bindings = successor_bindings()
    authority_units, _authority_set_digest, _authority_file_digest = load_required_authority()
    _runtime_state(root_path, bindings)
    expected_ordinals = list(range(REQUIRED_COUNT))
    if summary["accepted_authority_ordinals"] != expected_ordinals:
        missing = sorted(set(expected_ordinals) - set(summary["accepted_authority_ordinals"]))
        raise SuccessorMergeError(f"successor coverage is incomplete; missing ordinals {missing[:8]}")  # literal-ok: bounded diagnostic preview only
    for field in ("in_progress_authority_ordinals", "failed_authority_ordinals", "terminal_invalid_authority_ordinals"):
        if summary.get(field) != []:
            raise SuccessorMergeError(f"successor runtime has unresolved {field}")
    records: list[dict[str, Any]] = []
    for ordinal, unit in enumerate(authority_units):
        report = inspect_unit(root_path, ordinal)
        _require(report["accepted"], f"authority ordinal {ordinal} is not terminal accepted")
        complete = [item["result"] for item in report["validated_results"].values() if item["result"]["status"] == "complete"]
        _require(len(complete) == 1, f"authority ordinal {ordinal} does not have exactly one complete result")
        result = complete[0]
        identity = result["identity"]
        _require(result["artifact_role"] == RESULT_ARTIFACT_ROLE, f"authority ordinal {ordinal} has a non-successor result role")
        _require(identity["campaign_id"] == bindings["campaign_id"], f"authority ordinal {ordinal} campaign binding differs")
        _require(identity["execution_profile_id"] == SUCCESSOR_PROFILE_ID, f"authority ordinal {ordinal} execution profile differs")
        _require(identity["production_contract_sha256"] == bindings["production_contract_sha256"], f"authority ordinal {ordinal} production contract differs")
        _require(result["execution_provenance"]["git_commit"] == MEASUREMENT_SOURCE_COMMIT, f"authority ordinal {ordinal} measurement source commit differs")
        expected_key = _authority_key(unit, ordinal)
        _require(_result_authority_key(identity) == expected_key, f"authority ordinal {ordinal} exact physical identity differs")
        measurement = result["measurement"]
        _require(measurement["trials_completed"] == TRIALS_PER_IDENTITY, f"authority ordinal {ordinal} does not contain exactly 5000 trials")
        _require(result["disposition"]["protected_counters"] == PROTECTED_COUNTERS, f"authority ordinal {ordinal} protected counters are nonzero")
        _require(result["disposition"]["test_access"] == 0, f"authority ordinal {ordinal} test access is nonzero")
        history, request_only, failed_results, retries = _history_for_unit(root_path, report, unit)
        final_attempt = result["attempt"]
        _require(final_attempt == report["state"]["identity"]["attempt"], f"authority ordinal {ordinal} state/result attempt differs")
        state_payload, state_raw, state_sha = _raw_json(
            state_path(root_path, ordinal, unit["work_unit_id"]),
            f"successor state ordinal {ordinal}",
        )
        _require(state_payload == report["state"], f"successor state changed while merging ordinal {ordinal}")
        request_payload, request_raw, request_sha = _raw_json(
            request_path(root_path, unit["work_unit_id"], final_attempt),
            f"accepted successor request ordinal {ordinal}",
        )
        result_payload, result_raw, result_sha = _raw_json(
            result_path(root_path, unit["work_unit_id"], final_attempt),
            f"accepted successor result ordinal {ordinal}",
        )
        _require(result_payload == result, f"accepted result changed while merging ordinal {ordinal}")
        _require(state_payload["identity"]["request_sha256"] == request_sha and state_payload["identity"]["result_sha256"] == result_sha, f"state hash binding differs for ordinal {ordinal}")
        raw_measurement = dict(measurement)
        row = {
            **expected_key,
            "identity": identity["bler_identity"],
            "campaign_id": identity["campaign_id"],
            "execution_profile_id": identity["execution_profile_id"],
            "campaign_manifest_sha256": identity["campaign_manifest_sha256"],
            "production_contract_sha256": identity["production_contract_sha256"],
            "production_source_manifest_sha256": identity["source_manifest_sha256"],
            "production_runner_contract_sha256": identity["runner_contract_sha256"],
            "measurement_source_commit": result["execution_provenance"]["git_commit"],
            "attempt": final_attempt,
            "request_sha256": request_sha,
            "request_content_sha256": sha256_bytes(canonical_json(request_payload)),
            "result_sha256": result_sha,
            "result_content_sha256": result["result_sha256"],
            "state_sha256": state_sha,
            "raw_measurement": raw_measurement,
            "trials_completed": raw_measurement["trials_completed"],
            "information_bits": raw_measurement["information_bits"],
            "bit_errors": raw_measurement["bit_errors"],
            "block_errors": raw_measurement["block_errors"],
            "ber": raw_measurement["ber"],
            "bler": raw_measurement["bler"],
            "bler_confidence_low": raw_measurement["bler_confidence_low"],
            "bler_confidence_high": raw_measurement["bler_confidence_high"],
            "confidence_interval_method": raw_measurement["confidence_interval_method"],
            "confidence_interval_percent": raw_measurement["confidence_interval_percent"],
            "confidence_interval_role": raw_measurement["confidence_interval_role"],
            "historical_attempts": history,
        }
        row["_result_payload"] = result
        _require(request_raw == canonical_json(request_payload) and result_raw == canonical_json(result_payload) and state_raw == canonical_json(state_payload), f"successor artifact rendering changed for ordinal {ordinal}")
        _require(request_payload["identity"] == identity, f"accepted request/result identity differs for ordinal {ordinal}")
        row["_request_only_attempts"] = request_only
        row["_failed_result_attempts"] = failed_results
        row["_retry_attempts"] = retries
        records.append(row)
    return records


def collect_successor_results(root: Path | str) -> list[dict[str, Any]]:
    """Return only accepted Pascal result payloads in frozen authority order."""

    return [row["_result_payload"] for row in _collect_successor_evidence(root)]


def _clean_unit(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in MERGE_UNIT_FIELDS}


def build_successor_merge_report(root: Path | str) -> dict[str, Any]:
    """Build the hash-bound C3 report from the completed runtime only."""

    root_path = Path(root).resolve()
    bindings = successor_bindings()
    production_contracts = _production_contract_bindings()
    authority_units, authority_set_digest, authority_file_digest = load_required_authority()
    rows = _collect_successor_evidence(root_path)
    authority_order = [_authority_key(unit, ordinal) for ordinal, unit in enumerate(authority_units)]
    observed_order = [
        {
            key: row[key]
            for key in (
                "authority_ordinal",
                "work_unit_id",
                "required_work_unit_record_sha256",
                "identity",
                "snr_db",
                "source_packet_config_ids",
                "information_length",
                "codeword_length",
            )
        }
        for row in rows
    ]
    _require(observed_order == authority_order, "successor observed authority order is not exact")
    totals = {
        "total_trials": sum(row["trials_completed"] for row in rows),
        "total_information_bits": sum(row["information_bits"] for row in rows),
        "total_bit_errors": sum(row["bit_errors"] for row in rows),
        "total_block_errors": sum(row["block_errors"] for row in rows),
    }
    payload: dict[str, Any] = {
        "schema_version": MERGE_SCHEMA_VERSION,
        "artifact_role": MERGE_ARTIFACT_ROLE,
        "phase": PHASE,
        "checkpoint": "C3-C4",
        "report_id": None,
        "campaign_id": bindings["campaign_id"],
        "execution_profile_id": SUCCESSOR_PROFILE_ID,
        "campaign_manifest_sha256": bindings["campaign_manifest_sha256"],
        "production_contract_sha256": production_contracts["production_contract_sha256"],
        "production_source_manifest_sha256": production_contracts["production_source_manifest_sha256"],
        "production_runner_contract_sha256": production_contracts["production_runner_contract_sha256"],
        "required_bler_artifact_sha256": bindings["required_bler_artifact_sha256"],
        "required_authority_identity_set_sha256": authority_set_digest,
        "observed_authority_identity_set_sha256": sha256_bytes(canonical_json(observed_order)),
        "closeout_source_digest": closeout_source_digest(),
        "measurement_source_commit": MEASUREMENT_SOURCE_COMMIT,
        "runtime_relative_path": str(root_path.relative_to(REPO_ROOT)) if root_path.is_relative_to(REPO_ROOT) else str(root_path),
        "runtime_tree_sha256": normalized_runtime_tree_sha256(root_path),
        "required_identity_count": REQUIRED_COUNT,
        "required_authority_order": authority_order,
        "units": [_clean_unit(row) for row in rows],
        "accepted_count": len(rows),
        "completed_count": len(rows),
        "coverage_contribution_sum": sum(1 for _row in rows),
        "missing_count": 0,
        "duplicate_count": 0,
        "unknown_count": 0,
        "failed_count": 0,
        "terminal_invalid_count": 0,
        "unresolved_count": 0,
        "recoverable_count": 0,
        "request_only_attempt_count": sum(row["_request_only_attempts"] for row in rows),
        "failed_result_attempt_count": sum(row["_failed_result_attempts"] for row in rows),
        "retry_history_attempt_count": sum(row["_retry_attempts"] for row in rows),
        "retry_history_ordinals": [row["authority_ordinal"] for row in rows if row["_retry_attempts"]],
        **totals,
        "coverage_complete": True,
        "merge_ordering": "required_bler_identities.json authority ordinal ascending",
        "interpolation_used": False,
        "imputation_used": False,
        "extrapolation_used": False,
        "test_access": 0,
        "protected_counters": PROTECTED_COUNTERS,
        "old_result_ingest": False,
        "predecessor_campaign_id": "g8-8acd86ad87ef223187b69a2caf6ab8d29de3700dac9d5a60bb421cb228d8900a",
        "predecessor_table_contribution": "none",
    }
    _require(payload["coverage_contribution_sum"] == REQUIRED_COUNT, "successor coverage contribution does not equal 3213")
    _require(authority_file_digest == bindings["required_bler_artifact_sha256"], "required authority file binding changed")
    payload["report_id"] = _self_id(payload, "report_id", MERGE_ID_PREFIX)
    validate_successor_merge_report(payload)
    return payload


def _production_contract_bindings() -> dict[str, str]:
    """Return the authenticated production-source hashes without table reuse."""

    from baseline.g8_pascal_production import validate_production_contracts

    value = validate_production_contracts()
    return {
        "production_contract_sha256": value["production_contract_sha256"],
        "production_source_manifest_sha256": value["production_source_manifest_sha256"],
        "production_runner_contract_sha256": value["production_runner_contract_sha256"],
    }


def _validate_identity_and_bindings(row: Mapping[str, Any], expected: Mapping[str, Any], bindings: Mapping[str, Any]) -> None:
    _require(_strict_object(row, MERGE_UNIT_FIELDS, "successor merge unit") == dict(row), "successor merge unit schema differs")
    for field in ("authority_ordinal", "work_unit_id", "required_work_unit_record_sha256", "identity", "snr_db", "source_packet_config_ids", "information_length", "codeword_length"):
        _require(row[field] == expected[field], f"successor merge unit identity differs: {field}")
    for field, binding in (
        ("campaign_id", bindings["campaign_id"]),
        ("execution_profile_id", SUCCESSOR_PROFILE_ID),
        ("campaign_manifest_sha256", bindings["campaign_manifest_sha256"]),
        ("production_contract_sha256", bindings["production_contract_sha256"]),
        ("production_source_manifest_sha256", bindings["source_manifest_sha256"]),
        ("production_runner_contract_sha256", bindings["runner_contract_sha256"]),
        ("measurement_source_commit", MEASUREMENT_SOURCE_COMMIT),
    ):
        _require(row[field] == binding, f"successor merge unit binding differs: {field}")
    for name in ("request_sha256", "request_content_sha256", "result_sha256", "result_content_sha256", "state_sha256"):
        _digest(row[name], f"merge unit {name}")
    measurement = row["raw_measurement"]
    _require(isinstance(measurement, Mapping), "merge unit raw measurement is not an object")
    _require(dict(measurement) == {key: measurement[key] for key in ("ber", "bit_errors", "bler", "bler_confidence_high", "bler_confidence_low", "block_errors", "confidence_interval_method", "confidence_interval_percent", "confidence_interval_role", "information_bits", "trials_completed")}, "merge unit raw measurement schema differs")
    for key in ("trials_completed", "information_bits", "bit_errors", "block_errors", "ber", "bler", "bler_confidence_low", "bler_confidence_high", "confidence_interval_method", "confidence_interval_percent", "confidence_interval_role"):
        _require(row[key] == measurement[key] if key not in {"trials_completed", "information_bits", "bit_errors", "block_errors"} else row[key] == measurement[key], f"merge unit raw measurement was transformed: {key}")
    _require(row["trials_completed"] == TRIALS_PER_IDENTITY, "merge unit does not contain exactly 5000 trials")
    _require(row["information_bits"] == row["trials_completed"] * row["information_length"], "merge unit information-bit count does not reproduce")
    _require(row["block_errors"] <= row["trials_completed"] and row["bit_errors"] <= row["information_bits"], "merge unit error count exceeds its denominator")
    _require(row["bler"] == row["block_errors"] / row["trials_completed"], "merge unit BLER does not reproduce from raw counts")
    _require(row["ber"] == row["bit_errors"] / row["information_bits"], "merge unit BER does not reproduce from raw counts")
    _require(row["attempt"] >= 1, "merge unit accepted attempt is invalid")
    history = row["historical_attempts"]
    _require(isinstance(history, list) and history, "merge unit retry history is empty")
    _require([item["attempt"] for item in history] == list(range(1, row["attempt"] + 1)), "merge unit retry history is not contiguous")
    complete = [item for item in history if item["result_status"] == "complete"]
    _require(len(complete) == 1 and complete[0]["attempt"] == row["attempt"], "merge unit has more than one accepted complete attempt")
    _require(complete[0]["required_coverage_contribution"] == 1 and complete[0]["merge_eligible"] is True, "accepted retry does not contribute exactly once")
    for item in history:
        _digest(item["request_sha256"], "historical request hash")
        if item["result_sha256"] is not None:
            _digest(item["result_sha256"], "historical result hash")
        if item["result_status"] == "failed":
            _require(item["required_coverage_contribution"] == 0 and item["merge_eligible"] is False, "failed retry contributes to the table")
        _require(item["test_access"] == 0 and item["protected_counters"] == PROTECTED_COUNTERS, "historical retry claims protected activity")


def _validate_merge_shape(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = _strict_object(payload, MERGE_FIELDS, "successor merge report")
    _require(value["schema_version"] == MERGE_SCHEMA_VERSION and value["artifact_role"] == MERGE_ARTIFACT_ROLE, "unsupported successor merge report")
    _require(value["phase"] == PHASE and value["checkpoint"] == "C3-C4", "successor merge report phase/checkpoint differs")
    _require(value["report_id"] == _self_id(value, "report_id", MERGE_ID_PREFIX), "successor merge report ID does not reproduce")
    bindings = successor_bindings()
    for field, expected in (
        ("campaign_id", bindings["campaign_id"]),
        ("execution_profile_id", SUCCESSOR_PROFILE_ID),
        ("campaign_manifest_sha256", bindings["campaign_manifest_sha256"]),
        ("production_contract_sha256", bindings["production_contract_sha256"]),
        ("production_source_manifest_sha256", _file_sha256(REPO_ROOT / "results/baseline/g8_pascal_successor/production_source_manifest.json")),
        ("production_runner_contract_sha256", _file_sha256(REPO_ROOT / "results/baseline/g8_pascal_successor/production_runner_contract.json")),
        ("required_bler_artifact_sha256", bindings["required_bler_artifact_sha256"]),
        ("measurement_source_commit", MEASUREMENT_SOURCE_COMMIT),
    ):
        _require(value[field] == expected, f"successor merge report binding differs: {field}")
    authority_units, authority_set_digest, authority_file_digest = load_required_authority()
    _require(value["required_authority_identity_set_sha256"] == authority_set_digest, "successor merge authority digest differs")
    _require(
        value["closeout_source_digest"] == closeout_source_digest()
        or _historical_closeout_source_matches(value),
        "successor merge closeout source digest differs",
    )
    _require(value["required_identity_count"] == REQUIRED_COUNT and value["accepted_count"] == REQUIRED_COUNT and value["completed_count"] == REQUIRED_COUNT, "successor merge coverage count differs")
    _require(value["coverage_contribution_sum"] == REQUIRED_COUNT, "successor merge coverage contribution differs")
    _require(value["coverage_complete"] is True, "successor merge is not complete")
    _require(value["interpolation_used"] is False and value["imputation_used"] is False and value["extrapolation_used"] is False, "successor merge invented points")
    _require(value["test_access"] == 0 and value["protected_counters"] == PROTECTED_COUNTERS and value["old_result_ingest"] is False, "successor merge claims protected activity")
    _require(value["predecessor_table_contribution"] == "none", "predecessor evidence is eligible for successor table")
    _require(value["required_authority_order"] == [_authority_key(unit, ordinal) for ordinal, unit in enumerate(authority_units)], "successor merge required authority order differs")
    units = value["units"]
    _require(isinstance(units, list) and len(units) == REQUIRED_COUNT, "successor merge unit count differs")
    seen_ordinals: set[int] = set()
    observed_keys: list[dict[str, Any]] = []
    totals = {"trials_completed": 0, "information_bits": 0, "bit_errors": 0, "block_errors": 0}
    request_only = failed_results = retries = 0
    for ordinal, row in enumerate(units):
        expected = _authority_key(authority_units[ordinal], ordinal)
        _validate_identity_and_bindings(row, expected, bindings)
        _require(row["authority_ordinal"] not in seen_ordinals, "successor merge contains a duplicate authority ordinal")
        seen_ordinals.add(row["authority_ordinal"])
        observed_keys.append({key: row[key] for key in ("authority_ordinal", "work_unit_id", "required_work_unit_record_sha256", "identity", "snr_db", "source_packet_config_ids", "information_length", "codeword_length")})
        for key in totals:
            totals[key] += row[key]
        request_only += sum(1 for item in row["historical_attempts"] if item["result_sha256"] is None)
        failed_results += sum(1 for item in row["historical_attempts"] if item["result_status"] == "failed")
        retries += max(0, len(row["historical_attempts"]) - 1)
    _require(observed_keys == value["required_authority_order"], "successor merge has an omission, extra, alias, or reordered authority identity")
    _require(value["observed_authority_identity_set_sha256"] == sha256_bytes(canonical_json(observed_keys)), "successor observed authority digest differs")
    _require(value["request_only_attempt_count"] == request_only and value["failed_result_attempt_count"] == failed_results and value["retry_history_attempt_count"] == retries, "successor retry history totals differ")
    _require(value["retry_history_ordinals"] == [row["authority_ordinal"] for row in units if len(row["historical_attempts"]) > 1], "successor retry ordinal attribution differs")
    for key, total in (("total_trials", totals["trials_completed"]), ("total_information_bits", totals["information_bits"]), ("total_bit_errors", totals["bit_errors"]), ("total_block_errors", totals["block_errors"])):
        _require(value[key] == total, f"successor merge total differs: {key}")
    _require(value["missing_count"] == value["duplicate_count"] == value["unknown_count"] == value["failed_count"] == value["terminal_invalid_count"] == value["unresolved_count"] == value["recoverable_count"] == 0, "successor merge has incomplete or invalid coverage")
    _require(authority_file_digest == bindings["required_bler_artifact_sha256"], "successor required authority file changed")
    return value


def validate_successor_merge_report(payload: Mapping[str, Any], *, runtime_root: Path | str | None = None) -> dict[str, Any]:
    """Validate a merge report, optionally against the immutable runtime."""

    value = _validate_merge_shape(payload)
    if runtime_root is not None:
        expected = build_successor_merge_report(runtime_root)
        _require(canonical_json(value) == canonical_json(expected), "successor merge report does not reproduce from runtime evidence")
    return value


def _table_curves(units: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for row in units:
        identity = dict(row["identity"])
        key = canonical_json(identity).decode("ascii")
        if key not in grouped:
            grouped[key] = (identity, [])
        raw = row["raw_measurement"]
        grouped[key][1].append(
            {
                "authority_ordinal": row["authority_ordinal"],
                "work_unit_id": row["work_unit_id"],
                "snr_db": row["snr_db"],
                "trials": raw["trials_completed"],
                "information_bits": raw["information_bits"],
                "bit_errors": raw["bit_errors"],
                "block_errors": raw["block_errors"],
                "ber": raw["ber"],
                "bler": raw["bler"],
                "bler_confidence_low": raw["bler_confidence_low"],
                "bler_confidence_high": raw["bler_confidence_high"],
                "confidence_interval_method": raw["confidence_interval_method"],
                "confidence_interval_percent": raw["confidence_interval_percent"],
                "confidence_interval_role": raw["confidence_interval_role"],
                "raw_measurement": dict(raw),
                "request_sha256": row["request_sha256"],
                "request_content_sha256": row["request_content_sha256"],
                "result_sha256": row["result_sha256"],
                "result_content_sha256": row["result_content_sha256"],
                "state_sha256": row["state_sha256"],
                "measurement_source_commit": row["measurement_source_commit"],
            }
        )
    curves: list[dict[str, Any]] = []
    for key in sorted(grouped):
        identity, points = grouped[key]
        points.sort(key=lambda point: float(point["snr_db"]))
        _require(all(left["snr_db"] < right["snr_db"] for left, right in zip(points, points[1:], strict=False)), "successor table has duplicate or unordered SNR points")
        curves.append({"identity": identity, "points": points})
    return curves


def build_successor_bler_table(merge_report: Mapping[str, Any] | Sequence[Mapping[str, Any]], *, merge_report_sha256: str | None = None) -> dict[str, Any]:
    """Build the measured-only successor table from a complete C3 report."""

    if not isinstance(merge_report, Mapping):
        raise SuccessorMergeError("successor BlerTable requires a complete C3 merge report; raw or partial records are not admissible")
    merge = _validate_merge_shape(merge_report)
    bindings = successor_bindings()
    production_contracts = _production_contract_bindings()
    curves = _table_curves(merge["units"])
    payload: dict[str, Any] = {
        "schema_version": TABLE_SCHEMA_VERSION,
        "artifact_role": TABLE_ARTIFACT_ROLE,
        "phase": PHASE,
        "checkpoint": "C5",
        "table_id": None,
        "campaign_id": merge["campaign_id"],
        "execution_profile_id": merge["execution_profile_id"],
        "campaign_manifest_sha256": merge["campaign_manifest_sha256"],
        "production_contract_sha256": production_contracts["production_contract_sha256"],
        "production_source_manifest_sha256": production_contracts["production_source_manifest_sha256"],
        "production_runner_contract_sha256": production_contracts["production_runner_contract_sha256"],
        "required_bler_artifact_sha256": bindings["required_bler_artifact_sha256"],
        "required_authority_identity_set_sha256": merge["required_authority_identity_set_sha256"],
        "closeout_source_digest": merge["closeout_source_digest"],
        "measurement_source_commit": merge["measurement_source_commit"],
        "runtime_relative_path": merge["runtime_relative_path"],
        "runtime_tree_sha256": merge["runtime_tree_sha256"],
        "merge_report_id": merge["report_id"],
        "merge_report_sha256": merge_report_sha256 or sha256_bytes(rendered_json(dict(merge))),
        "required_identity_count": REQUIRED_COUNT,
        "complete_identity_count": len(curves),
        "measured_point_count": sum(len(curve["points"]) for curve in curves),
        "trials_per_point": TRIALS_PER_IDENTITY,
        "total_trials": merge["total_trials"],
        "total_information_bits": merge["total_information_bits"],
        "total_bit_errors": merge["total_bit_errors"],
        "total_block_errors": merge["total_block_errors"],
        "interpolation_used": False,
        "imputation_used": False,
        "extrapolation_used": False,
        "test_access": 0,
        "protected_counters": PROTECTED_COUNTERS,
        "old_result_ingest": False,
        "predecessor_campaign_id": merge["predecessor_campaign_id"],
        "predecessor_table_contribution": "none",
        "curves": curves,
    }
    payload["table_id"] = _self_id(payload, "table_id", TABLE_ID_PREFIX)
    validate_successor_bler_table(payload, merge_report=merge, merge_report_sha256=payload["merge_report_sha256"])
    return payload


def _validate_table_shape(payload: Mapping[str, Any], merge: Mapping[str, Any], merge_report_sha256: str) -> dict[str, Any]:
    value = _strict_object(payload, TABLE_FIELDS, "successor BLER table")
    _require(value["schema_version"] == TABLE_SCHEMA_VERSION and value["artifact_role"] == TABLE_ARTIFACT_ROLE, "unsupported successor BLER table")
    _require(value["phase"] == PHASE and value["checkpoint"] == "C5", "successor BLER table phase/checkpoint differs")
    _require(value["table_id"] == _self_id(value, "table_id", TABLE_ID_PREFIX), "successor BLER table ID does not reproduce")
    for field in ("campaign_id", "execution_profile_id", "campaign_manifest_sha256", "production_contract_sha256", "production_source_manifest_sha256", "production_runner_contract_sha256", "required_bler_artifact_sha256", "required_authority_identity_set_sha256", "closeout_source_digest", "measurement_source_commit", "runtime_relative_path", "runtime_tree_sha256"):
        _require(value[field] == merge[field], f"successor BLER table binding differs: {field}")
    _require(value["merge_report_id"] == merge["report_id"] and value["merge_report_sha256"] == merge_report_sha256, "successor BLER table merge binding differs")
    _require(value["required_identity_count"] == REQUIRED_COUNT and value["trials_per_point"] == TRIALS_PER_IDENTITY, "successor BLER table coverage/trial contract differs")
    _require(value["interpolation_used"] is False and value["imputation_used"] is False and value["extrapolation_used"] is False, "successor BLER table contains invented points")
    _require(value["test_access"] == 0 and value["protected_counters"] == PROTECTED_COUNTERS and value["old_result_ingest"] is False, "successor BLER table claims protected activity")
    _require(value["predecessor_table_contribution"] == "none", "predecessor BLER evidence entered successor table")
    expected_curves = _table_curves(merge["units"])
    _require(value["curves"] == expected_curves, "successor BLER table curves do not reproduce from the merge report")
    _require(value["complete_identity_count"] == len(expected_curves), "successor BLER table curve count differs")
    point_count = sum(len(curve["points"]) for curve in expected_curves)
    _require(value["measured_point_count"] == point_count == REQUIRED_COUNT, "successor BLER table point count differs")
    for key in ("total_trials", "total_information_bits", "total_bit_errors", "total_block_errors"):
        _require(value[key] == merge[key], f"successor BLER table total differs: {key}")
    return value


def validate_successor_bler_table(
    payload: Mapping[str, Any],
    *,
    merge_report: Mapping[str, Any],
    merge_report_sha256: str,
) -> dict[str, Any]:
    _validate_merge_shape(merge_report)
    return _validate_table_shape(payload, merge_report, merge_report_sha256)


def load_successor_bler_table(
    path: Path | str = TABLE_PATH,
    *,
    merge_path: Path | str = MERGE_REPORT_PATH,
    runtime_root: Path | str = SUCCESSOR_RUNTIME_ROOT,
    verify_runtime: bool = True,
) -> composition.BlerTable:
    """Load only the content-addressed Pascal successor table.

    The path check is intentional: a caller cannot point this loader at the
    historical predecessor table and have it silently treated as successor
    evidence.  Runtime verification is enabled by default for the later
    selection phases.
    """

    if verify_runtime:
        # Keep this historical entry point stable for D0--D7 and future
        # callers, but move its default runtime authentication to the
        # checkout-portable scientific evidence verifier.  The legacy branch
        # below remains available only for explicit historical artifact reads.
        from baseline.g8_pascal_portable import load_portable_successor_bler_table

        return load_portable_successor_bler_table(
            path,
            merge_path=merge_path,
            runtime_root=runtime_root,
        )

    table_path = Path(path).resolve()
    merge_report_path = Path(merge_path).resolve()
    _require(table_path == TABLE_PATH.resolve(), "successor table loader refuses a non-successor table path")
    _require(merge_report_path == MERGE_REPORT_PATH.resolve(), "successor table loader refuses a non-successor merge path")
    table, table_raw, table_sha = _read_artifact(table_path, "successor BLER table")
    merge, merge_raw, merge_sha = _read_artifact(merge_report_path, "successor merge report")
    _require(table_sha == _file_sha256(table_path) and merge_sha == _file_sha256(merge_report_path), "successor closeout artifact hash read failed")
    validate_successor_merge_report(merge)
    validate_successor_bler_table(table, merge_report=merge, merge_report_sha256=merge_sha)
    curves: dict[composition.BlerIdentity, Any] = {}
    for curve in table["curves"]:
        identity = composition.BlerIdentity.from_mapping(curve["identity"])
        points = curve["points"]
        curves[identity] = composition._Curve(
            snr_db=tuple(float(point["snr_db"]) for point in points),
            bler=tuple(float(point["bler"]) for point in points),
            trials=TRIALS_PER_IDENTITY,
        )
    _ = table_raw, merge_raw
    return composition.BlerTable(curves, provenance=str(TABLE_PATH.relative_to(REPO_ROOT)))


__all__ = [
    "CLOSEOUT_SOURCE_PATHS",
    "HISTORICAL_MERGE_REPORT_ID",
    "HISTORICAL_MERGE_REPORT_SHA256",
    "HISTORICAL_TABLE_ID",
    "HISTORICAL_TABLE_SHA256",
    "HISTORICAL_CLOSEOUT_ID",
    "HISTORICAL_CLOSEOUT_SHA256",
    "HISTORICAL_RUNTIME_TREE_SHA256",
    "HISTORICAL_CLOSEOUT_SOURCE_DIGEST",
    "HISTORICAL_CLOSEOUT_SOURCE_ENTRIES",
    "MERGE_REPORT_PATH",
    "TABLE_PATH",
    "PROVENANCE_PATH",
    "SuccessorMergeError",
    "build_successor_merge_report",
    "build_successor_bler_table",
    "collect_successor_results",
    "load_required_authority",
    "closeout_source_entries",
    "closeout_source_digest",
    "normalized_runtime_tree_sha256",
    "validate_successor_merge_report",
    "validate_successor_bler_table",
    "load_successor_bler_table",
]
