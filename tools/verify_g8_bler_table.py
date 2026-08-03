#!/usr/bin/env python3
"""Independently verify the complete G8_C merge report and BLER table.

This verifier reconstructs accepted units directly from the authenticated
required identities, B3 runtime census, raw request/result/state files, and
registered contracts.  It does not import the merge generator or the table
loader and therefore cannot inherit either one's calculations.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from baseline import g8_bler_contract as bler_contract  # noqa: E402
from baseline import g8_bler_resume as resume  # noqa: E402
from baseline import g8_bler_work_units as work_units  # noqa: E402
from baseline.classical import composition  # noqa: E402
from baseline.g8_campaign import (  # noqa: E402
    CAMPAIGN_STATE,
    REPO_ROOT,
    canonical_json,
    load_campaign_state,
    rendered_json,
    sha256_bytes,
    sha256_file,
    validate_state_transition,
    write_campaign_state_atomically,
)
import verify_g8_bler_characterization_manifest as source_verifier  # noqa: E402


MERGE_PATH = REPO_ROOT / "results/baseline/g8/bler_merge_report.json"
TABLE_PATH = REPO_ROOT / "results/baseline/g8/bler_table.json"
MERGE_ID_PREFIX = "g8merge"
TABLE_ID_PREFIX = "g8blertable"


class TableVerificationError(RuntimeError):
    """The independent raw-evidence reconstruction disagrees with an artifact."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TableVerificationError(message)


def _read(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    _require(isinstance(payload, dict), f"{label} is not an object")
    _require(raw == rendered_json(payload), f"{label} is not canonical rendered JSON")
    return payload, raw


def _self_id(payload: Mapping[str, Any], field: str, prefix: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    return f"{prefix}-{sha256_bytes(canonical_json(body))}"


def _state(context: resume.AuthenticatedResumeContext, root: Path, work_unit_id: str) -> tuple[dict[str, Any], str, str]:
    path = resume.state_path(context, work_unit_id, root=root)
    state = work_units.read_unit_state(context.state_context, path, root=root)
    raw = path.read_bytes()
    return state, resume.logical_artifact_path(context, work_unit_id, resume.ARTIFACT_KIND_STATE), sha256_bytes(raw)


def _historical(
    context: resume.AuthenticatedResumeContext,
    root: Path,
    work_unit_id: str,
    final_attempt: int,
    request_attempts: list[int],
    result_attempts: list[int],
) -> tuple[int, int]:
    _require(request_attempts == result_attempts, f"request/result history mismatch: {work_unit_id}")
    failures = 0
    for attempt in request_attempts:
        request_record = resume.validate_request_file(context, work_unit_id, attempt, root=root, require_full_strength=True)
        result_record = resume.validate_result_file(
            context,
            work_unit_id,
            attempt,
            root=root,
            request_record=request_record,
            scan_mode=resume.SCAN_MODE_PRODUCTION_MERGE,
        )
        if attempt == final_attempt:
            _require(result_record["status"] == bler_contract.STATUS_COMPLETE, f"final result is not complete: {work_unit_id}")
        else:
            _require(result_record["status"] == bler_contract.STATUS_FAILED, f"historical result is not failed: {work_unit_id}/{attempt}")
            failures += 1
    return len(request_attempts), failures


def _expected_unit(
    context: resume.AuthenticatedResumeContext,
    root: Path,
    record: Mapping[str, Any],
    census: Mapping[str, Any],
) -> dict[str, Any]:
    work_unit_id = record["work_unit_id"]
    _require(record["classification"] == resume.CLASSIFICATION_COMPLETED_FULL_STRENGTH, f"unit not complete: {work_unit_id}")
    final_attempt = record["attempt"]
    unit = context.work_unit_record(work_unit_id)
    request_record = resume.validate_request_file(context, work_unit_id, final_attempt, root=root, require_full_strength=True)
    result_record = resume.validate_result_file(
        context,
        work_unit_id,
        final_attempt,
        root=root,
        request_record=request_record,
        shard_index=record["shard_index"],
        shard_count=record["shard_count"],
        scan_mode=resume.SCAN_MODE_PRODUCTION_MERGE,
    )
    state, state_path, state_sha = _state(context, root, work_unit_id)
    state_identity = state["identity"]
    result = result_record["result"]
    request = request_record["request"]
    measurement = result["measurement"]
    disposition = result["disposition"]
    _require(state_identity["status"] == work_units.STATUS_RESULT_LINKED, f"state status mismatch: {work_unit_id}")
    _require(state_identity["request_sha256"] == request_record["request_sha256"], f"state/request mismatch: {work_unit_id}")
    _require(state_identity["result_sha256"] == result_record["result_sha256"], f"state/result mismatch: {work_unit_id}")
    _require(state_identity["result_path"] == result_record["logical_path"], f"state path mismatch: {work_unit_id}")
    attempts, failed_attempts = _historical(
        context,
        root,
        work_unit_id,
        final_attempt,
        list(census["request_attempts"].get(work_unit_id, [])),
        list(census["result_attempts"].get(work_unit_id, [])),
    )
    return {
        "authority_ordinal": context.ordinal(work_unit_id),
        "work_unit_id": work_unit_id,
        "bler_identity": dict(unit["identity"]),
        "snr_db": unit["snr_db"],
        "source_packet_config_ids": list(unit["source_packet_config_ids"]),
        "final_attempt": final_attempt,
        "request_path": request_record["logical_path"],
        "request_sha256": request_record["request_sha256"],
        "result_path": result_record["logical_path"],
        "result_sha256": result_record["result_sha256"],
        "state_path": state_path,
        "state_sha256": state_sha,
        "trials_requested": request["trials_requested"],
        "trials_completed": measurement["trials_completed"],
        "information_bits": measurement["information_bits"],
        "bit_errors": measurement["bit_errors"],
        "block_errors": measurement["block_errors"],
        "ber": measurement["ber"],
        "bler": measurement["bler"],
        "bler_confidence_low": measurement["bler_confidence_low"],
        "bler_confidence_high": measurement["bler_confidence_high"],
        "merge_eligible": disposition["merge_eligible"],
        "required_coverage_contribution": disposition["required_coverage_contribution"],
        "test_split_access": disposition["test_split_access"],
        "historical_attempt_count": attempts,
        "historical_failed_attempt_count": failed_attempts,
    }


def _verify_merge(
    merge: dict[str, Any],
    source: dict[str, Any],
    context: resume.AuthenticatedResumeContext,
    root: Path,
    inspection: dict[str, Any],
) -> list[dict[str, Any]]:
    required_ids = list(context.ordered_work_unit_ids)
    records = inspection["classifications"]
    expected_units = [_expected_unit(context, root, record, inspection["census"]) for record in records]
    _require([unit["work_unit_id"] for unit in expected_units] == required_ids, "raw accepted units are not exact authority order")
    _require(merge["units"] == expected_units, "merge report units disagree with independent raw reconstruction")
    _require(merge["required_work_unit_ids"] == required_ids, "merge required IDs disagree")
    _require(merge["required_work_unit_count"] == len(required_ids), "merge required count disagrees")
    _require(merge["completed_count"] == len(required_ids), "merge completed count disagrees")
    for field in ("missing_count", "duplicate_count", "unknown_count", "recoverable_count", "failed_count", "terminal_nonmergeable_count"):
        _require(merge[field] == 0, f"merge {field} is nonzero")
    _require(merge["coverage_contribution_sum"] == len(required_ids), "merge coverage contribution disagrees")
    _require(merge["total_trials"] == sum(unit["trials_completed"] for unit in expected_units), "merge total trials disagrees")
    _require(merge["total_information_bits"] == sum(unit["information_bits"] for unit in expected_units), "merge total information bits disagrees")
    _require(merge["total_bit_errors"] == sum(unit["bit_errors"] for unit in expected_units), "merge total bit errors disagrees")
    _require(merge["total_block_errors"] == sum(unit["block_errors"] for unit in expected_units), "merge total block errors disagrees")
    _require(merge["coverage_complete"] is True, "merge is not complete")
    _require(merge["interpolation_used"] is False and merge["extrapolation_used"] is False, "merge reports invented points")
    _require(merge["test_split_access"] == 0, "merge claims test access")
    _require(merge["source_manifest_id"] == source["manifest_id"], "merge source manifest ID disagrees")
    _require(merge["source_manifest_sha256"] == sha256_file(source_verifier.MANIFEST), "merge source manifest SHA disagrees")
    _require(merge["report_id"] == _self_id(merge, "report_id", MERGE_ID_PREFIX), "merge ID does not reproduce")
    return expected_units


def _register_artifact(relative_path: str, raw: bytes) -> str:
    """Publish one independently verified C3/C5 artifact through campaign state."""

    previous = load_campaign_state(CAMPAIGN_STATE)
    identity = previous["identity"]
    _require(identity["phase"] == "G8_C" and identity["stage"] == "characterization_open", "artifact registration requires G8_C/open")
    _require(identity["in_progress_work_unit_id"] is None, "artifact registration requires no in-progress unit")
    _require(all(value == 0 for value in identity["counters"].values()), "artifact registration requires zero protected counters")
    binding = {"path": relative_path, "sha256": sha256_bytes(raw), "bytes": len(raw)}
    current = copy.deepcopy(previous)
    artifacts = current["identity"]["produced_artifacts"]
    matches = [entry for entry in artifacts if entry["path"] == relative_path]
    if matches:
        _require(matches == [binding], f"registered artifact binding changed: {relative_path}")
    else:
        artifacts.append(binding)
        artifacts.sort(key=lambda entry: entry["path"])
    validate_state_transition(previous, current)
    digest = write_campaign_state_atomically(CAMPAIGN_STATE, current)
    installed = load_campaign_state(CAMPAIGN_STATE)
    _require(any(entry == binding for entry in installed["identity"]["produced_artifacts"]), "artifact registration was not installed")
    return digest


def _table_from_merge(merge: Mapping[str, Any], source: Mapping[str, Any], merge_sha: str) -> dict[str, Any]:
    by_identity: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for unit in merge["units"]:
        identity_key = canonical_json(unit["bler_identity"]).decode("ascii")
        if identity_key not in by_identity:
            by_identity[identity_key] = (dict(unit["bler_identity"]), [])
        by_identity[identity_key][1].append(
            {
                "work_unit_id": unit["work_unit_id"],
                "snr_db": unit["snr_db"],
                "trials": unit["trials_completed"],
                "information_bits": unit["information_bits"],
                "bit_errors": unit["bit_errors"],
                "block_errors": unit["block_errors"],
                "ber": unit["ber"],
                "bler": unit["bler"],
                "bler_confidence_low": unit["bler_confidence_low"],
                "bler_confidence_high": unit["bler_confidence_high"],
                "request_sha256": unit["request_sha256"],
                "result_sha256": unit["result_sha256"],
                "state_sha256": unit["state_sha256"],
            }
        )
    curves: list[dict[str, Any]] = []
    for identity_key in sorted(by_identity):
        identity, points = by_identity[identity_key]
        points.sort(key=lambda point: float(point["snr_db"]))
        curves.append({"identity": identity, "points": points})
    point_count = sum(len(curve["points"]) for curve in curves)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_role": "g8_bler_table",
        "phase": "G8_C",
        "checkpoint": "C5",
        "table_id": None,
        "campaign_id": source["campaign_id"],
        "campaign_manifest_sha256": source["campaign_manifest_sha256"],
        "required_bler_artifact_sha256": source["required_bler_artifact_sha256"],
        "selection_policy_sha256": source["selection_policy_sha256"],
        "bler_tooling_contract_id": source["bler_tooling_contract_id"],
        "bler_tooling_contract_sha256": source["bler_tooling_contract_sha256"],
        "bler_state_contract_id": source["bler_state_contract_id"],
        "bler_state_contract_sha256": source["bler_state_contract_sha256"],
        "bler_resume_contract_id": source["bler_resume_contract_id"],
        "bler_resume_contract_sha256": source["bler_resume_contract_sha256"],
        "bler_runner_contract_id": source["bler_runner_contract_id"],
        "bler_runner_contract_sha256": source["bler_runner_contract_sha256"],
        "source_manifest_id": source["manifest_id"],
        "source_manifest_sha256": sha256_file(source_verifier.MANIFEST),
        "merge_report_id": merge["report_id"],
        "merge_report_sha256": merge_sha,
        "required_work_unit_count": len(merge["required_work_unit_ids"]),
        "complete_identity_count": len(curves),
        "measured_point_count": point_count,
        "trials_per_point": bler_contract.full_strength_trial_count(),
        "total_trials": merge["total_trials"],
        "interpolation_used": False,
        "extrapolation_used": False,
        "test_split_access": 0,
        "curves": curves,
    }
    payload["table_id"] = _self_id(payload, "table_id", TABLE_ID_PREFIX)
    return payload


def verify_coverage() -> dict[str, Any]:
    source = source_verifier.verify(require_registered=True)
    merge, merge_raw = _read(MERGE_PATH, "merge report")
    _require(merge["artifact_role"] == "g8_bler_merge_report" and merge["phase"] == "G8_C", "merge artifact identity drift")
    state = load_campaign_state(CAMPAIGN_STATE)
    _require(state["identity"]["phase"] == "G8_C" and state["identity"]["stage"] in {"characterization_open", "characterization_complete"}, "campaign is outside G8_C")
    _require(all(value == 0 for value in state["identity"]["counters"].values()), "protected counter changed")
    merge_matches = [entry for entry in state["identity"]["produced_artifacts"] if entry["path"] == "results/baseline/g8/bler_merge_report.json"]
    if state["identity"]["stage"] == "characterization_complete":
        _require(len(merge_matches) == 1, "merge report is absent at characterization_complete")
    if merge_matches:
        _require(merge_matches == [{"path": "results/baseline/g8/bler_merge_report.json", "sha256": sha256_bytes(merge_raw), "bytes": len(merge_raw)}], "registered merge binding changed")
    context = resume.AuthenticatedResumeContext(require_resume_contract=True)
    root = REPO_ROOT / "results/baseline/g8/work_units"
    b3 = resume.build_merge_report(context, root=root, scan_mode=resume.SCAN_MODE_PRODUCTION_MERGE)
    _require(b3["coverage_complete"] is True, "fresh B3 coverage is incomplete")
    inspection = resume.inspect_runtime_root(context, root=root, scan_mode=resume.SCAN_MODE_PRODUCTION_MERGE, repair_mode=resume.REPAIR_MODE_READ_ONLY)
    expected_units = _verify_merge(merge, source, context, root, inspection)
    _require(state["identity"]["completed_work_unit_ids"] == sorted([unit["work_unit_id"] for unit in expected_units]), "campaign completed IDs do not equal raw evidence")
    _require(state["identity"]["in_progress_work_unit_id"] is None, "campaign has an in-progress unit at coverage verification")
    return {
        "source": source,
        "merge": merge,
        "merge_raw": merge_raw,
        "expected_units": expected_units,
        "state": state,
    }


def verify(*, require_table_registered: bool = True) -> dict[str, Any]:
    coverage = verify_coverage()
    source = coverage["source"]
    merge = coverage["merge"]
    merge_raw = coverage["merge_raw"]
    state = coverage["state"]
    table, table_raw = _read(TABLE_PATH, "BLER table")
    _require(table["artifact_role"] == "g8_bler_table" and table["phase"] == "G8_C", "table artifact identity drift")
    table_matches = [entry for entry in state["identity"]["produced_artifacts"] if entry["path"] == "results/baseline/g8/bler_table.json"]
    if require_table_registered:
        _require(len(table_matches) == 1, "BLER table is not registered exactly once")
        _require(table_matches == [{"path": "results/baseline/g8/bler_table.json", "sha256": sha256_bytes(table_raw), "bytes": len(table_raw)}], "registered table binding changed")
    elif table_matches:
        _require(table_matches == [{"path": "results/baseline/g8/bler_table.json", "sha256": sha256_bytes(table_raw), "bytes": len(table_raw)}], "registered table binding changed")
    expected_table = _table_from_merge(merge, source, sha256_bytes(merge_raw))
    _require(table == expected_table, "BLER table disagrees with independent merge reconstruction")
    _require(table["table_id"] == _self_id(table, "table_id", TABLE_ID_PREFIX), "table ID does not reproduce")
    _require(table["interpolation_used"] is False and table["extrapolation_used"] is False, "table construction invented points")
    _require(table["total_trials"] == 3213 * bler_contract.full_strength_trial_count(), "table total trials are not 16,065,000")
    _require(table["measured_point_count"] == len(coverage["expected_units"]), "table point count disagrees")
    for curve in table["curves"]:
        identity = composition.BlerIdentity.from_mapping(curve["identity"])
        previous = None
        for point in curve["points"]:
            if previous is not None:
                _require(float(point["snr_db"]) > previous, "table SNR points are not ascending")
            previous = float(point["snr_db"])
            _require(point["bler"] == point["block_errors"] / point["trials"], "table BLER arithmetic drift")
            _require(point["ber"] == point["bit_errors"] / point["information_bits"], "table BER arithmetic drift")
            lookup = composition.BlerTable(
                {
                    identity: composition._Curve(
                        snr_db=(float(point["snr_db"]),),
                        bler=(float(point["bler"]),),
                        trials=point["trials"],
                    )
                },
                provenance="independent-verifier",
            ).lookup(identity, point["snr_db"])
            _require(lookup.characterized and lookup.interpolated is False and lookup.bler == point["bler"], "exact point lookup failed")
    return {"source_manifest_id": source["manifest_id"], "merge_report_id": merge["report_id"], "table_id": table["table_id"], "units": len(coverage["expected_units"]), "points": table["measured_point_count"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-only", action="store_true")
    parser.add_argument("--register-merge", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--register-table", action="store_true")
    args = parser.parse_args()
    if sum(bool(value) for value in (args.coverage_only, args.build, args.register_table)) > 1:
        raise SystemExit("--coverage-only, --build and --register-table are mutually exclusive")
    try:
        if args.build:
            from baseline import g8_bler_characterization as characterization  # noqa: PLC0415
            from baseline.classical.outage import write_json_atomically  # noqa: PLC0415

            payload = characterization.build_bler_table_payload()
            raw = rendered_json(payload)
            digest = write_json_atomically(TABLE_PATH, payload)
            print(f"G8_C BLER table written: table_id={payload['table_id']} sha256={digest} bytes={len(raw)}")
            return 0
        if args.coverage_only:
            result = verify_coverage()
            if args.register_merge:
                digest = _register_artifact("results/baseline/g8/bler_merge_report.json", result["merge_raw"])
                print(f"G8_C merge registration PASS: state_sha256={digest}")
                return 0
            print("G8_C independent coverage verification PASS: " + json.dumps({"units": len(result["expected_units"]), "merge_report_id": result["merge"]["report_id"]}, sort_keys=True))
            return 0
        result = verify(require_table_registered=not args.register_table)
        if args.register_table:
            table_raw = TABLE_PATH.read_bytes()
            digest = _register_artifact("results/baseline/g8/bler_table.json", table_raw)
            print(f"G8_C table registration PASS: state_sha256={digest}")
            return 0
    except Exception as exc:
        raise SystemExit(f"G8_C merge/table HOLD: {exc}") from exc
    print("G8_C independent merge/table verification PASS: " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
