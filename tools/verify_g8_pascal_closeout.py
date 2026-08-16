#!/usr/bin/env python3
"""Independently verify the Pascal successor G8_C C3--C7 closeout.

This verifier intentionally does not import ``baseline.g8_pascal_merge`` or
the closeout writer.  It re-audits the authenticated runtime, reconstructs
the accepted rows and measured-only curves, and compares the committed
content-addressed artifacts byte-for-byte at the canonical JSON level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline.classical import composition  # noqa: E402
from baseline.g8_campaign import canonical_json, rendered_json, sha256_bytes  # noqa: E402
from baseline.g8_pascal_production import (  # noqa: E402
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
    validate_production_contracts,
    validate_runtime_namespace,
)
from baseline.g8_pascal_successor import SUCCESSOR_ROOT  # noqa: E402


class CloseoutVerificationError(RuntimeError):
    """The committed successor closeout does not reproduce."""


MERGE_REPORT_PATH = SUCCESSOR_ROOT / "successor_bler_merge_report.json"
TABLE_PATH = SUCCESSOR_ROOT / "successor_bler_table.json"
PROVENANCE_PATH = SUCCESSOR_ROOT / "successor_closeout_provenance.json"
MEASUREMENT_SOURCE_COMMIT = "426110b05161e73e4d819bdc01f4857c012d6d59"
MERGE_SCHEMA_VERSION = 1
TABLE_SCHEMA_VERSION = 1
MERGE_ARTIFACT_ROLE = "g8_c_pascal_successor_bler_merge_report"
TABLE_ARTIFACT_ROLE = "g8_c_pascal_successor_bler_table"
MERGE_ID_PREFIX = "g8pmerge-"
TABLE_ID_PREFIX = "g8pblertable-"
PROVENANCE_ID_PREFIX = "g8pcloseout-"
PROVENANCE_ARTIFACT_ROLE = "g8_c_pascal_successor_closeout_provenance"
PHASE = "G8_C"
PROTECTED_COUNTERS = {
    "inference": 0,
    "test_access": 0,
    "training": 0,
    "validation_decoding": 0,
}
HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
CLOSEOUT_SOURCE_PATHS = (
    ("src/baseline/g8_pascal_merge.py", "successor_c3_c5_merge_and_table_builder"),
    ("tools/closeout_g8_pascal_successor.py", "successor_closeout_artifact_freezer"),
    ("tools/verify_g8_pascal_closeout.py", "independent_successor_c3_c7_verifier"),
)

MERGE_FIELDS = {
    "schema_version", "artifact_role", "phase", "checkpoint", "report_id", "campaign_id", "execution_profile_id",
    "campaign_manifest_sha256", "production_contract_sha256", "production_source_manifest_sha256", "production_runner_contract_sha256",
    "required_bler_artifact_sha256", "required_authority_identity_set_sha256", "observed_authority_identity_set_sha256", "closeout_source_digest",
    "measurement_source_commit", "runtime_relative_path", "runtime_tree_sha256", "required_identity_count", "required_authority_order", "units",
    "accepted_count", "completed_count", "coverage_contribution_sum", "missing_count", "duplicate_count", "unknown_count", "failed_count",
    "terminal_invalid_count", "unresolved_count", "recoverable_count", "request_only_attempt_count", "failed_result_attempt_count",
    "retry_history_attempt_count", "retry_history_ordinals", "total_trials", "total_information_bits", "total_bit_errors", "total_block_errors",
    "coverage_complete", "merge_ordering", "interpolation_used", "imputation_used", "extrapolation_used", "test_access", "protected_counters",
    "old_result_ingest", "predecessor_campaign_id", "predecessor_table_contribution",
}
MERGE_UNIT_FIELDS = {
    "authority_ordinal", "work_unit_id", "required_work_unit_record_sha256", "identity", "snr_db", "source_packet_config_ids",
    "information_length", "codeword_length", "campaign_id", "execution_profile_id", "campaign_manifest_sha256", "production_contract_sha256",
    "production_source_manifest_sha256", "production_runner_contract_sha256", "measurement_source_commit", "attempt", "request_sha256",
    "request_content_sha256", "result_sha256", "result_content_sha256", "state_sha256", "raw_measurement", "trials_completed",
    "information_bits", "bit_errors", "block_errors", "ber", "bler", "bler_confidence_low", "bler_confidence_high",
    "confidence_interval_method", "confidence_interval_percent", "confidence_interval_role", "historical_attempts",
}
TABLE_FIELDS = {
    "schema_version", "artifact_role", "phase", "checkpoint", "table_id", "campaign_id", "execution_profile_id", "campaign_manifest_sha256",
    "production_contract_sha256", "production_source_manifest_sha256", "production_runner_contract_sha256", "required_bler_artifact_sha256",
    "required_authority_identity_set_sha256", "closeout_source_digest", "measurement_source_commit", "runtime_relative_path", "runtime_tree_sha256",
    "merge_report_id", "merge_report_sha256", "required_identity_count", "complete_identity_count", "measured_point_count", "trials_per_point",
    "total_trials", "total_information_bits", "total_bit_errors", "total_block_errors", "interpolation_used", "imputation_used",
    "extrapolation_used", "test_access", "protected_counters", "old_result_ingest", "predecessor_campaign_id", "predecessor_table_contribution", "curves",
}
TABLE_POINT_FIELDS = {
    "authority_ordinal", "work_unit_id", "snr_db", "trials", "information_bits", "bit_errors", "block_errors", "ber", "bler",
    "bler_confidence_low", "bler_confidence_high", "confidence_interval_method", "confidence_interval_percent", "confidence_interval_role",
    "raw_measurement", "request_sha256", "request_content_sha256", "result_sha256", "result_content_sha256", "state_sha256", "measurement_source_commit",
}


def _fail(message: str) -> None:
    raise CloseoutVerificationError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _self_id(value: Mapping[str, Any], field: str, prefix: str) -> str:
    body = dict(value)
    body.pop(field, None)
    return prefix + sha256_bytes(canonical_json(body))


def _digest(value: Any, label: str) -> None:
    _require(isinstance(value, str) and HEX_DIGEST.fullmatch(value) is not None, f"{label} is not a SHA-256 digest")


def _read(path: Path, label: str) -> tuple[dict[str, Any], bytes, str]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read {label}: {exc}")
    _require(isinstance(payload, dict), f"{label} is not an object")
    _require(raw == rendered_json(payload), f"{label} is not canonical rendered JSON")
    return payload, raw, sha256_bytes(raw)


def _read_runtime(path: Path, label: str) -> tuple[dict[str, Any], bytes, str]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read {label}: {exc}")
    _require(isinstance(payload, dict), f"{label} is not an object")
    _require(raw == canonical_json(payload), f"{label} is not canonical runtime JSON")
    return payload, raw, sha256_bytes(raw)


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        _fail(f"cannot hash {path}: {exc}")


def _authority_key(unit: Mapping[str, Any], ordinal: int) -> dict[str, Any]:
    identity = composition.BlerIdentity.from_mapping(unit["identity"]).as_key()
    _require(identity == unit["identity"], f"required identity {ordinal} is not canonical")
    return {
        "authority_ordinal": ordinal,
        "work_unit_id": unit["work_unit_id"],
        "required_work_unit_record_sha256": sha256_bytes(canonical_json(dict(unit))),
        "identity": identity,
        "snr_db": unit["snr_db"],
        "source_packet_config_ids": list(unit["source_packet_config_ids"]),
        "information_length": unit["information_length"],
        "codeword_length": unit["codeword_length"],
    }


def _load_authority() -> tuple[list[dict[str, Any]], str, str]:
    path = REPO / "results/baseline/g8/required_bler_identities.json"
    payload, raw, digest = _read(path, "required BLER authority")
    bindings = successor_bindings()
    _require(digest == bindings["required_bler_artifact_sha256"], "required BLER authority SHA differs from production evidence")
    units = payload.get("required_bler_work_units")
    _require(isinstance(units, list) and len(units) == REQUIRED_COUNT, "required BLER authority is not the exact 3213-cell grid")
    keys = [_authority_key(unit, ordinal) for ordinal, unit in enumerate(units)]
    _require(len({sha256_bytes(canonical_json(key)) for key in keys}) == REQUIRED_COUNT, "required BLER authority has duplicate aliases")
    return [dict(unit) for unit in units], sha256_bytes(canonical_json(keys)), digest


def _source_entries() -> list[dict[str, Any]]:
    entries = []
    for relative, role in CLOSEOUT_SOURCE_PATHS:
        path = REPO / relative
        _require(path.is_file(), f"closeout source is missing: {relative}")
        body = path.read_bytes()
        entries.append({"path": relative, "role": role, "bytes": len(body), "sha256": sha256_bytes(body)})
    return entries


def _source_digest() -> str:
    return sha256_bytes(canonical_json(_source_entries()))


def _runtime_tree_sha256(root: Path) -> str:
    try:
        result = subprocess.run(
            ["tar", "--sort=name", "--mtime=UTC 1970-01-01 00:00:00", "--owner=0", "--group=0", "--numeric-owner", "-cf", "-", "-C", str(root), "."],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        _fail(f"cannot normalize runtime tree: {exc}")
    return sha256_bytes(result.stdout)


def _production() -> dict[str, str]:
    value = validate_production_contracts()
    return {
        "production_contract_sha256": value["production_contract_sha256"],
        "production_source_manifest_sha256": value["production_source_manifest_sha256"],
        "production_runner_contract_sha256": value["production_runner_contract_sha256"],
    }


def _runtime_state(root: Path, bindings: Mapping[str, Any]) -> str:
    payload, _raw, digest = _read_runtime(root / "campaign_state.json", "runtime campaign state")
    try:
        validate_campaign_state(payload, bindings=bindings)
    except Exception as exc:
        _fail(f"runtime campaign state is invalid: {exc}")
    _require(payload["accepted_authority_ordinals"] == list(range(REQUIRED_COUNT)), "runtime accepted ordinal set is incomplete")
    for field in ("in_progress_authority_ordinals", "failed_authority_ordinals", "terminal_invalid_authority_ordinals"):
        _require(payload[field] == [], f"runtime {field} is nonempty")
    _require(payload["scientific_execution_performed"] is True, "runtime does not record completed measurements")
    _require(payload["protected_counters"] == PROTECTED_COUNTERS and payload["test_access"] == 0, "runtime protected counters are nonzero")
    _require(payload["old_result_ingest_permitted"] is False, "runtime permits old-result ingest")
    return digest


def _result_key(identity: Mapping[str, Any]) -> dict[str, Any]:
    physical = composition.BlerIdentity.from_mapping(identity["bler_identity"]).as_key()
    k, n = physical["k_and_n"]
    return {
        "authority_ordinal": identity["authority_ordinal"],
        "work_unit_id": identity["work_unit_id"],
        "required_work_unit_record_sha256": identity["required_work_unit_record_sha256"],
        "identity": physical,
        "snr_db": identity["snr_db"],
        "source_packet_config_ids": list(identity["source_packet_config_ids"]),
        "information_length": k,
        "codeword_length": n,
    }


def _history(root: Path, report: Mapping[str, Any], unit: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int, int, int]:
    attempts = sorted(set(report["request_attempts"]) | set(report["result_attempts"]))
    _require(attempts == list(range(1, max(attempts) + 1)), f"retry history has a gap at ordinal {report['ordinal']}")
    rows = []
    request_only = failed = 0
    for attempt in attempts:
        request, request_raw, request_sha = _read_runtime(request_path(root, unit["work_unit_id"], attempt), "historical request")
        result = None
        result_sha = None
        if attempt in report["validated_results"]:
            result, result_raw, result_sha = _read_runtime(result_path(root, unit["work_unit_id"], attempt), "historical result")
            _require(report["validated_results"][attempt]["sha256"] == result_sha, f"historical result hash differs at ordinal {report['ordinal']}")
        else:
            request_only += 1
        if result is None:
            status = None
            contribution = 0
            merge_eligible = False
            measurement = None
            counters = PROTECTED_COUNTERS
            test_access = 0
            result_content_sha = None
        else:
            status = result["status"]
            contribution = result["disposition"]["required_coverage_contribution"]
            merge_eligible = result["disposition"]["merge_eligible"]
            measurement = result["measurement"]
            counters = result["disposition"]["protected_counters"]
            test_access = result["disposition"]["test_access"]
            result_content_sha = result["result_sha256"]
            if status == "failed":
                failed += 1
                _require(contribution == 0 and merge_eligible is False, "failed retry contributes to coverage")
        rows.append({
            "attempt": attempt,
            "request_sha256": request_sha,
            "request_content_sha256": None,
            "result_sha256": result_sha,
            "result_content_sha256": result_content_sha,
            "result_status": status,
            "merge_eligible": merge_eligible,
            "required_coverage_contribution": contribution,
            "trials_completed": None if measurement is None else measurement["trials_completed"],
            "test_access": test_access,
            "protected_counters": counters,
        })
        _ = request, request_raw, result_raw if result is not None else None
    return rows, request_only, failed, max(0, len(attempts) - 1)


def _expected_units(root: Path, authority_units: Sequence[Mapping[str, Any]], bindings: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for ordinal, unit in enumerate(authority_units):
        report = inspect_unit(root, ordinal)
        _require(report["accepted"], f"ordinal {ordinal} is not accepted")
        complete = [item["result"] for item in report["validated_results"].values() if item["result"]["status"] == "complete"]
        _require(len(complete) == 1, f"ordinal {ordinal} has zero or multiple complete results")
        result = complete[0]
        identity = result["identity"]
        _require(result["artifact_role"] == RESULT_ARTIFACT_ROLE, f"ordinal {ordinal} result role differs")
        _require(identity["campaign_id"] == bindings["campaign_id"] and identity["execution_profile_id"] == SUCCESSOR_PROFILE_ID, f"ordinal {ordinal} campaign/profile differs")
        _require(identity["production_contract_sha256"] == bindings["production_contract_sha256"], f"ordinal {ordinal} production contract differs")
        _require(result["execution_provenance"]["git_commit"] == MEASUREMENT_SOURCE_COMMIT, f"ordinal {ordinal} measurement source differs")
        expected_key = _authority_key(unit, ordinal)
        _require(_result_key(identity) == expected_key, f"ordinal {ordinal} exact identity/SNR tuple differs")
        measurement = result["measurement"]
        _require(measurement["trials_completed"] == TRIALS_PER_IDENTITY, f"ordinal {ordinal} trial count differs")
        _require(result["disposition"]["protected_counters"] == PROTECTED_COUNTERS and result["disposition"]["test_access"] == 0, f"ordinal {ordinal} protected activity differs")
        historical, request_only, failed, retries = _history(root, report, unit)
        final_attempt = result["attempt"]
        state, state_raw, state_sha = _read_runtime(state_path(root, ordinal, unit["work_unit_id"]), "accepted state")
        request, request_raw, request_sha = _read_runtime(request_path(root, unit["work_unit_id"], final_attempt), "accepted request")
        result_payload, result_raw, result_sha = _read_runtime(result_path(root, unit["work_unit_id"], final_attempt), "accepted result")
        _require(result_payload == result and request["identity"] == identity, f"ordinal {ordinal} accepted binding differs")
        _require(state["identity"]["request_sha256"] == request_sha and state["identity"]["result_sha256"] == result_sha, f"ordinal {ordinal} state hash binding differs")
        raw_measurement = dict(measurement)
        rows.append({
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
            "request_content_sha256": sha256_bytes(canonical_json(request)),
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
            "historical_attempts": historical,
            "_request_only_attempts": request_only,
            "_failed_result_attempts": failed,
            "_retry_attempts": retries,
        })
        _ = state_raw, request_raw, result_raw
    return rows


def _expected_merge(root: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    bindings = successor_bindings()
    production = _production()
    authority_units, authority_set_digest, authority_file_digest = _load_authority()
    runtime_state_sha = _runtime_state(root, bindings)
    summary = audit_campaign(root)
    _require(summary["accepted_authority_ordinals"] == list(range(REQUIRED_COUNT)), "audit does not report exact accepted authority")
    _require(summary["in_progress_authority_ordinals"] == summary["failed_authority_ordinals"] == summary["terminal_invalid_authority_ordinals"] == [], "audit reports unresolved successor units")
    validate_runtime_namespace(root)
    rows = _expected_units(root, authority_units, bindings)
    authority_order = [_authority_key(unit, ordinal) for ordinal, unit in enumerate(authority_units)]
    observed_order = [{key: row[key] for key in ("authority_ordinal", "work_unit_id", "required_work_unit_record_sha256", "identity", "snr_db", "source_packet_config_ids", "information_length", "codeword_length")} for row in rows]
    _require(observed_order == authority_order, "accepted observed authority is not exact")
    payload: dict[str, Any] = {
        "schema_version": MERGE_SCHEMA_VERSION,
        "artifact_role": MERGE_ARTIFACT_ROLE,
        "phase": PHASE,
        "checkpoint": "C3-C4",
        "report_id": None,
        "campaign_id": bindings["campaign_id"],
        "execution_profile_id": SUCCESSOR_PROFILE_ID,
        "campaign_manifest_sha256": bindings["campaign_manifest_sha256"],
        "production_contract_sha256": production["production_contract_sha256"],
        "production_source_manifest_sha256": production["production_source_manifest_sha256"],
        "production_runner_contract_sha256": production["production_runner_contract_sha256"],
        "required_bler_artifact_sha256": bindings["required_bler_artifact_sha256"],
        "required_authority_identity_set_sha256": authority_set_digest,
        "observed_authority_identity_set_sha256": sha256_bytes(canonical_json(observed_order)),
        "closeout_source_digest": _source_digest(),
        "measurement_source_commit": MEASUREMENT_SOURCE_COMMIT,
        "runtime_relative_path": str(root.relative_to(REPO)) if root.is_relative_to(REPO) else str(root),
        "runtime_tree_sha256": _runtime_tree_sha256(root),
        "required_identity_count": REQUIRED_COUNT,
        "required_authority_order": authority_order,
        "units": [{key: row[key] for key in MERGE_UNIT_FIELDS} for row in rows],
        "accepted_count": REQUIRED_COUNT,
        "completed_count": REQUIRED_COUNT,
        "coverage_contribution_sum": REQUIRED_COUNT,
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
        "total_trials": sum(row["trials_completed"] for row in rows),
        "total_information_bits": sum(row["information_bits"] for row in rows),
        "total_bit_errors": sum(row["bit_errors"] for row in rows),
        "total_block_errors": sum(row["block_errors"] for row in rows),
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
    _require(runtime_state_sha == "4e7510e850e59d047b512c1df0e7f5916b4ae6d814505d1bb9e042bc1585655e", "runtime state SHA differs from published completed evidence")
    _require(authority_file_digest == bindings["required_bler_artifact_sha256"], "authority file binding differs")
    payload["report_id"] = _self_id(payload, "report_id", MERGE_ID_PREFIX)
    return payload, runtime_state_sha, {"authority_units": authority_units, "authority_set_digest": authority_set_digest}


def _table_curves(units: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for row in units:
        key = canonical_json(row["identity"]).decode("ascii")
        grouped.setdefault(key, (dict(row["identity"]), []))[1].append({
            "authority_ordinal": row["authority_ordinal"],
            "work_unit_id": row["work_unit_id"],
            "snr_db": row["snr_db"],
            "trials": row["trials_completed"],
            "information_bits": row["information_bits"],
            "bit_errors": row["bit_errors"],
            "block_errors": row["block_errors"],
            "ber": row["ber"],
            "bler": row["bler"],
            "bler_confidence_low": row["bler_confidence_low"],
            "bler_confidence_high": row["bler_confidence_high"],
            "confidence_interval_method": row["confidence_interval_method"],
            "confidence_interval_percent": row["confidence_interval_percent"],
            "confidence_interval_role": row["confidence_interval_role"],
            "raw_measurement": dict(row["raw_measurement"]),
            "request_sha256": row["request_sha256"],
            "request_content_sha256": row["request_content_sha256"],
            "result_sha256": row["result_sha256"],
            "result_content_sha256": row["result_content_sha256"],
            "state_sha256": row["state_sha256"],
            "measurement_source_commit": row["measurement_source_commit"],
        })
    curves = []
    for key in sorted(grouped):
        identity, points = grouped[key]
        points.sort(key=lambda point: float(point["snr_db"]))
        _require(all(left["snr_db"] < right["snr_db"] for left, right in zip(points, points[1:], strict=False)), "curve has duplicate or unordered SNR")
        curves.append({"identity": identity, "points": points})
    return curves


def _expected_table(merge: Mapping[str, Any], merge_raw: bytes) -> dict[str, Any]:
    production = _production()
    bindings = successor_bindings()
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
        "production_contract_sha256": production["production_contract_sha256"],
        "production_source_manifest_sha256": production["production_source_manifest_sha256"],
        "production_runner_contract_sha256": production["production_runner_contract_sha256"],
        "required_bler_artifact_sha256": bindings["required_bler_artifact_sha256"],
        "required_authority_identity_set_sha256": merge["required_authority_identity_set_sha256"],
        "closeout_source_digest": merge["closeout_source_digest"],
        "measurement_source_commit": merge["measurement_source_commit"],
        "runtime_relative_path": merge["runtime_relative_path"],
        "runtime_tree_sha256": merge["runtime_tree_sha256"],
        "merge_report_id": merge["report_id"],
        "merge_report_sha256": sha256_bytes(merge_raw),
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
    return payload


def _verify_provenance(payload: Mapping[str, Any], merge: Mapping[str, Any], table: Mapping[str, Any], merge_raw: bytes, table_raw: bytes, runtime_root: Path) -> None:
    required = {"schema_version", "artifact_role", "phase", "checkpoint", "closure_id", "campaign_id", "execution_profile_id", "measurement_source", "closeout_source", "authority", "artifacts", "retry_history", "safety", "predecessor_isolation", "mutation_checks"}
    _require(set(payload) == required, "closeout provenance schema differs")
    _require(payload["schema_version"] == 1 and payload["artifact_role"] == PROVENANCE_ARTIFACT_ROLE and payload["phase"] == PHASE and payload["checkpoint"] == "C6", "unsupported closeout provenance")
    _require(payload["closure_id"] == _self_id(payload, "closure_id", PROVENANCE_ID_PREFIX), "closeout provenance ID does not reproduce")
    bindings = successor_bindings()
    production = _production()
    _require(payload["campaign_id"] == bindings["campaign_id"] and payload["execution_profile_id"] == SUCCESSOR_PROFILE_ID, "closeout provenance campaign/profile differs")
    measurement = payload["measurement_source"]
    _require(measurement == {
        "role": "authenticated_production_measurement_source",
        "commit": MEASUREMENT_SOURCE_COMMIT,
        "production_contract_sha256": production["production_contract_sha256"],
        "production_source_manifest_sha256": production["production_source_manifest_sha256"],
        "production_runner_contract_sha256": production["production_runner_contract_sha256"],
        "execution_profile_id": SUCCESSOR_PROFILE_ID,
        "runtime_relative_path": merge["runtime_relative_path"],
        "runtime_tree_sha256": merge["runtime_tree_sha256"],
        "scientific_execution_performed": True,
    }, "measurement-source provenance differs")
    closeout = payload["closeout_source"]
    _require(closeout["role"] == "deterministic_post_measurement_merge_and_table_consumer" and closeout["source_digest"] == _source_digest(), "closeout source digest differs")
    _require(closeout["sources"] == _source_entries() and closeout["scientific_execution_performed"] is False, "closeout source bytes/status differ")
    _require(closeout["measurement_source_commit_is_not_closeout_commit"] is True and closeout["closeout_commit_is_resolved_by_git_publication"] is True, "closeout source/measurement distinction is missing")
    authority_units, authority_set_digest, authority_file_digest = _load_authority()
    authority = payload["authority"]
    _require(authority == {"path": "results/baseline/g8/required_bler_identities.json", "bytes": len((REPO / "results/baseline/g8/required_bler_identities.json").read_bytes()), "sha256": authority_file_digest, "identity_count": len(authority_units), "ordered_identity_set_sha256": authority_set_digest}, "closeout authority provenance differs")
    artifacts = payload["artifacts"]
    _require(artifacts["merge_report"] == {"path": "results/baseline/g8_pascal_successor/successor_bler_merge_report.json", "bytes": len(merge_raw), "sha256": sha256_bytes(merge_raw), "report_id": merge["report_id"]}, "merge artifact provenance differs")
    _require(artifacts["bler_table"] == {"path": "results/baseline/g8_pascal_successor/successor_bler_table.json", "bytes": len(table_raw), "sha256": sha256_bytes(table_raw), "table_id": table["table_id"]}, "table artifact provenance differs")
    _require(payload["retry_history"] == {"preserved": True, "historical_failed_attempts_contribute": 0, "accepted_attempts_contribute": 1, "retry_ordinals": merge["retry_history_ordinals"], "request_only_attempt_count": merge["request_only_attempt_count"], "failed_result_attempt_count": merge["failed_result_attempt_count"]}, "retry provenance differs")
    _require(payload["safety"] == {"required_identity_count": REQUIRED_COUNT, "trials_per_identity": TRIALS_PER_IDENTITY, "test_access": 0, "protected_counters": PROTECTED_COUNTERS, "old_result_ingest": False, "interpolation_used": False, "imputation_used": False, "extrapolation_used": False}, "closeout safety provenance differs")
    _require(payload["predecessor_isolation"] == {"predecessor_campaign_id": merge["predecessor_campaign_id"], "predecessor_table_contribution": "none", "successor_campaign_id": merge["campaign_id"], "old_rtx4060_profile_id": "local_4060_cu130"}, "predecessor isolation provenance differs")
    _require(payload["mutation_checks"] == {"source_bytes_rehashed": True, "merge_report_id_recomputed": True, "table_id_recomputed": True, "request_result_state_hashes_rechecked": True, "runtime_tree_hash_rechecked": True}, "mutation checks are incomplete")
    _require(runtime_root == (REPO / merge["runtime_relative_path"]).resolve(), "provenance runtime path differs")


def validate_payloads(merge: Mapping[str, Any], table: Mapping[str, Any], provenance: Mapping[str, Any], expected_merge: Mapping[str, Any], expected_table: Mapping[str, Any], *, merge_raw: bytes | None = None, table_raw: bytes | None = None, runtime_root: Path | None = None) -> None:
    """Pure artifact comparison hook used by focused mutation tests."""

    _require(canonical_json(dict(merge)) == canonical_json(dict(expected_merge)), "successor merge artifact mutation or coverage drift detected")
    _require(canonical_json(dict(table)) == canonical_json(dict(expected_table)), "successor BLER table mutation or curve drift detected")
    if merge_raw is not None and table_raw is not None and runtime_root is not None:
        _verify_provenance(provenance, merge, table, merge_raw, table_raw, runtime_root)


def verify(*, runtime_root: Path | None = None) -> dict[str, Any]:
    root = (SUCCESSOR_ROOT / "runtime" if runtime_root is None else runtime_root).resolve()
    old_verifier = None
    try:
        import verify_g8_pascal_successor as old_successor_verifier

        old_verifier = old_successor_verifier.verify(runtime_root=root)
    except Exception as exc:
        _fail(f"successor production verifier failed: {exc}")
    production = _production()
    audit = audit_campaign(root)
    expected_merge, runtime_state_sha, _authority = _expected_merge(root)
    merge, merge_raw, merge_sha = _read(MERGE_REPORT_PATH, "successor merge report")
    validate_payloads(merge, {}, {}, expected_merge, {}, runtime_root=None)
    expected_table = _expected_table(expected_merge, rendered_json(expected_merge))
    table, table_raw, table_sha = _read(TABLE_PATH, "successor BLER table")
    # The committed merge bytes must be the bytes used by the table's binding.
    _require(merge_raw == rendered_json(expected_merge), "successor merge report rendering differs from reconstructed bytes")
    expected_table = _expected_table(expected_merge, merge_raw)
    _require(canonical_json(table) == canonical_json(expected_table), "successor BLER table does not reproduce from independently reconstructed merge")
    provenance, provenance_raw, provenance_sha = _read(PROVENANCE_PATH, "successor closeout provenance")
    _verify_provenance(provenance, merge, table, merge_raw, table_raw, root)
    _require(merge_sha == provenance["artifacts"]["merge_report"]["sha256"] and table_sha == provenance["artifacts"]["bler_table"]["sha256"], "closeout artifact hashes differ")
    _require(_runtime_tree_sha256(root) == merge["runtime_tree_sha256"], "runtime input tree changed after merge")
    _require(audit["accepted_count"] == REQUIRED_COUNT and audit["failed_authority_ordinals"] == [] and audit["terminal_invalid_authority_ordinals"] == [], "successor audit is not complete")
    _require(table["interpolation_used"] is False and table["imputation_used"] is False and table["extrapolation_used"] is False, "table contains an invented point")
    _require(table["test_access"] == 0 and table["protected_counters"] == PROTECTED_COUNTERS and table["old_result_ingest"] is False, "table claims protected activity")
    _require(table["predecessor_table_contribution"] == "none", "predecessor table isolation failed")
    return {
        "status": "PASS",
        "campaign_id": merge["campaign_id"],
        "execution_profile_id": merge["execution_profile_id"],
        "accepted": merge["accepted_count"],
        "required": merge["required_identity_count"],
        "trials_per_point": table["trials_per_point"],
        "total_trials": table["total_trials"],
        "total_information_bits": table["total_information_bits"],
        "total_bit_errors": table["total_bit_errors"],
        "total_block_errors": table["total_block_errors"],
        "merge_report_id": merge["report_id"],
        "merge_report_sha256": merge_sha,
        "table_id": table["table_id"],
        "table_sha256": table_sha,
        "provenance_id": provenance["closure_id"],
        "provenance_sha256": provenance_sha,
        "curves": table["complete_identity_count"],
        "points": table["measured_point_count"],
        "runtime_state_sha256": runtime_state_sha,
        "runtime_tree_sha256": merge["runtime_tree_sha256"],
        "retry_history_ordinals": merge["retry_history_ordinals"],
        "old_successor_verifier": old_verifier,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, help="completed successor runtime root")
    args = parser.parse_args()
    print(json.dumps(verify(runtime_root=args.root), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CloseoutVerificationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
