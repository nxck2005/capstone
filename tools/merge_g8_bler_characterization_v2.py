#!/usr/bin/env python3
"""Independently assemble the complete G8_C raw-evidence merge report."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from baseline import g8_bler_contract as bler_contract  # noqa: E402
from baseline import g8_bler_resume as resume  # noqa: E402
from baseline import g8_bler_work_units as work_units  # noqa: E402
from baseline import g8_bler_characterization_v2 as characterization  # noqa: E402
from baseline.classical.outage import write_json_atomically  # noqa: E402
from baseline.g8_campaign import (  # noqa: E402
    REPO_ROOT,
    canonical_json,
    rendered_json,
    sha256_bytes,
    sha256_file,
)


class MergeError(RuntimeError):
    """The raw evidence cannot produce one complete merge report."""


UNIT_FIELDS = (
    "authority_ordinal",
    "work_unit_id",
    "bler_identity",
    "snr_db",
    "source_packet_config_ids",
    "source_epoch",
    "source_epoch_manifest_id",
    "final_attempt",
    "request_path",
    "request_sha256",
    "result_path",
    "result_sha256",
    "state_path",
    "state_sha256",
    "trials_requested",
    "trials_completed",
    "information_bits",
    "bit_errors",
    "block_errors",
    "ber",
    "bler",
    "bler_confidence_low",
    "bler_confidence_high",
    "merge_eligible",
    "required_coverage_contribution",
    "test_split_access",
    "historical_attempt_count",
    "historical_request_only_attempt_count",
    "historical_failed_attempt_count",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MergeError(message)


def _id(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("report_id", None)
    return f"{characterization.MERGE_REPORT_ID_PREFIX}-{sha256_bytes(canonical_json(body))}"


def _read_canonical(path: Path, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    _require(isinstance(payload, dict), f"{label} is not an object")
    _require(canonical_json(payload) == raw, f"{label} is not canonical compact JSON")
    return payload


def _source() -> dict[str, Any]:
    raw = characterization.SOURCE_MANIFEST_PATH.read_bytes()
    payload = json.loads(raw)
    _require(canonical_json(payload) == raw, "source manifest is not canonical JSON")
    characterization.validate_source_manifest(payload, require_registered=True)
    return payload


def _state_record(context: resume.AuthenticatedResumeContext, root: Path, work_unit_id: str) -> tuple[dict[str, Any], str, str]:
    path = resume.state_path(context, work_unit_id, root=root)
    state = work_units.read_unit_state(context.state_context, path, root=root)
    raw = path.read_bytes()
    return state, resume.logical_artifact_path(context, work_unit_id, resume.ARTIFACT_KIND_STATE), sha256_bytes(raw)


def _historical_attempts(
    context: resume.AuthenticatedResumeContext,
    root: Path,
    work_unit_id: str,
    final_attempt: int,
    request_attempts: list[int],
    result_attempts: list[int],
) -> tuple[int, int, int]:
    _require(request_attempts == sorted(set(request_attempts)), f"request attempts are not unique/ordered for {work_unit_id}")
    _require(result_attempts == sorted(set(result_attempts)), f"result attempts are not unique/ordered for {work_unit_id}")
    _require(set(result_attempts).issubset(set(request_attempts)), f"result lacks its request for {work_unit_id}")
    _require(final_attempt in request_attempts, f"final attempt is absent from request history for {work_unit_id}")
    _require(final_attempt in result_attempts, f"final result is absent from result history for {work_unit_id}")
    failed = 0
    request_only = 0
    for attempt in request_attempts:
        request_record = resume.validate_request_file(
            context,
            work_unit_id,
            attempt,
            root=root,
            require_full_strength=True,
        )
        if attempt not in result_attempts:
            _require(attempt != final_attempt, f"final attempt has no result for {work_unit_id}")
            request_only += 1
            continue
        result_record = resume.validate_result_file(
            context,
            work_unit_id,
            attempt,
            root=root,
            request_record=request_record,
            scan_mode=resume.SCAN_MODE_PRODUCTION_MERGE,
        )
        if attempt == final_attempt:
            _require(result_record["status"] == bler_contract.STATUS_COMPLETE, f"final result is not complete for {work_unit_id}")
        else:
            _require(result_record["status"] == bler_contract.STATUS_FAILED, f"historical attempt is not failed for {work_unit_id} attempt {attempt}")
            failed += 1
    return len(request_attempts), request_only, failed


def _unit(
    context: resume.AuthenticatedResumeContext,
    root: Path,
    record: Mapping[str, Any],
    census: Mapping[str, Any],
    source_manifest_id: str,
) -> dict[str, Any]:
    work_unit_id = record["work_unit_id"]
    unit = context.work_unit_record(work_unit_id)
    final_attempt = record["attempt"]
    _require(record["classification"] == resume.CLASSIFICATION_COMPLETED_FULL_STRENGTH, f"unit is not merge-complete: {work_unit_id}")
    _require(isinstance(final_attempt, int) and final_attempt > 0, f"invalid final attempt for {work_unit_id}")
    request_record = resume.validate_request_file(
        context,
        work_unit_id,
        final_attempt,
        root=root,
        require_full_strength=True,
    )
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
    state, state_path, state_sha = _state_record(context, root, work_unit_id)
    request = request_record["request"]
    result = result_record["result"]
    state_identity = state["identity"]
    _require(state_identity["status"] == work_units.STATUS_RESULT_LINKED, f"final state is not result_linked for {work_unit_id}")
    _require(state_identity["request_sha256"] == request_record["request_sha256"], f"state/request chain mismatch for {work_unit_id}")
    _require(state_identity["result_sha256"] == result_record["result_sha256"], f"state/result chain mismatch for {work_unit_id}")
    _require(state_identity["result_path"] == result_record["logical_path"], f"state result path mismatch for {work_unit_id}")
    _require(state_identity["trials_completed"] == result["measurement"]["trials_completed"], f"state trial count mismatch for {work_unit_id}")
    attempts, request_only_attempts, failed_attempts = _historical_attempts(
        context,
        root,
        work_unit_id,
        final_attempt,
        list(census["request_attempts"].get(work_unit_id, [])),
        list(census["result_attempts"].get(work_unit_id, [])),
    )
    measurement = result["measurement"]
    disposition = result["disposition"]
    ordinal = context.ordinal(work_unit_id)
    _require(0 <= ordinal <= characterization.EPOCH_2_END_ORDINAL, f"unit ordinal is outside the source chain: {work_unit_id}")
    source_epoch = 1 if ordinal <= characterization.EPOCH_1_END_ORDINAL else 2
    epoch_manifest_id = characterization.PREDECESSOR_MANIFEST_ID if source_epoch == 1 else source_manifest_id
    point = {
        "authority_ordinal": ordinal,
        "work_unit_id": work_unit_id,
        "bler_identity": dict(unit["identity"]),
        "snr_db": unit["snr_db"],
        "source_packet_config_ids": list(unit["source_packet_config_ids"]),
        "source_epoch": source_epoch,
        "source_epoch_manifest_id": epoch_manifest_id,
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
        "historical_request_only_attempt_count": request_only_attempts,
        "historical_failed_attempt_count": failed_attempts,
    }
    _require(set(point) == set(UNIT_FIELDS), f"merge unit schema drift for {work_unit_id}")
    _require(point["trials_requested"] == bler_contract.full_strength_trial_count(), f"full-strength request trial count changed for {work_unit_id}")
    _require(point["trials_completed"] == point["trials_requested"], f"incomplete full-strength result for {work_unit_id}")
    _require(point["merge_eligible"] is True and point["required_coverage_contribution"] == 1, f"unit is not merge eligible for {work_unit_id}")
    _require(point["test_split_access"] == bler_contract.TEST_SPLIT_ACCESS, f"unit claims test access: {work_unit_id}")
    return point


def build_report(root: Path) -> dict[str, Any]:
    source = _source()
    context = resume.AuthenticatedResumeContext(require_resume_contract=True)
    b3_report = resume.build_merge_report(
        context,
        root=root,
        scan_mode=resume.SCAN_MODE_PRODUCTION_MERGE,
    )
    _require(b3_report["coverage_complete"] is True, "B3 does not report complete coverage")
    inspection = resume.inspect_runtime_root(
        context,
        root=root,
        scan_mode=resume.SCAN_MODE_PRODUCTION_MERGE,
        repair_mode=resume.REPAIR_MODE_READ_ONLY,
    )
    classifications = inspection["classifications"]
    census = inspection["census"]
    _require(len(classifications) == len(context.ordered_work_unit_ids), "B3 classification count drift")
    units = [_unit(context, root, record, census, source["manifest_id"]) for record in classifications]
    required_ids = list(context.ordered_work_unit_ids)
    unit_ids = [unit["work_unit_id"] for unit in units]
    missing = [work_unit_id for work_unit_id in required_ids if work_unit_id not in set(unit_ids)]
    duplicate_count = len(unit_ids) - len(set(unit_ids))
    _require(unit_ids == required_ids, "merge units are not exact required authority order")
    total_trials = sum(unit["trials_completed"] for unit in units)
    total_information_bits = sum(unit["information_bits"] for unit in units)
    total_bit_errors = sum(unit["bit_errors"] for unit in units)
    total_block_errors = sum(unit["block_errors"] for unit in units)
    payload: dict[str, Any] = {
        "schema_version": characterization.MERGE_REPORT_SCHEMA_VERSION,
        "artifact_role": characterization.MERGE_REPORT_ARTIFACT_ROLE,
        "phase": "G8_C",
        "checkpoint": "C3",
        "report_id": None,
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
        "source_manifest_sha256": sha256_file(characterization.SOURCE_MANIFEST_PATH),
        "source_epochs": source["source_epochs"],
        "required_work_unit_count": len(required_ids),
        "required_work_unit_ids": required_ids,
        "units": units,
        "completed_count": len(units),
        "missing_count": len(missing),
        "duplicate_count": duplicate_count,
        "unknown_count": 0,
        "recoverable_count": 0,
        "failed_count": 0,
        "terminal_nonmergeable_count": 0,
        "coverage_contribution_sum": sum(unit["required_coverage_contribution"] for unit in units),
        "total_trials": total_trials,
        "total_information_bits": total_information_bits,
        "total_bit_errors": total_bit_errors,
        "total_block_errors": total_block_errors,
        "coverage_complete": (
            len(units) == len(required_ids)
            and not missing
            and duplicate_count == 0
            and payload_placeholder_coverage(units, len(required_ids))
        ),
        "interpolation_used": False,
        "extrapolation_used": False,
        "test_split_access": bler_contract.TEST_SPLIT_ACCESS,
        "request_only_attempt_count": sum(unit["historical_request_only_attempt_count"] for unit in units),
        "failed_result_attempt_count": sum(unit["historical_failed_attempt_count"] for unit in units),
    }
    payload["report_id"] = _id(payload)
    return payload


def payload_placeholder_coverage(units: list[dict[str, Any]], required_count: int) -> bool:
    """Keep the top-level predicate readable without hiding a scientific rule."""

    return (
        len(units) == required_count
        and all(unit["required_coverage_contribution"] == 1 for unit in units)
        and all(unit["trials_completed"] == bler_contract.full_strength_trial_count() for unit in units)
        and all(unit["test_split_access"] == bler_contract.TEST_SPLIT_ACCESS for unit in units)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(REPO_ROOT / characterization.LOGICAL_ROOT))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        root = characterization.validate_production_root(Path(args.root))
        payload = build_report(root)
        expected = rendered_json(payload)
        if args.check:
            actual = characterization.MERGE_REPORT_PATH.read_bytes()
            if actual != expected:
                raise MergeError("bler_merge_report.json is stale")
            print(f"G8_C merge report check PASS: report_id={payload['report_id']} sha256={sha256_bytes(actual)} bytes={len(actual)}")
            return 0
        digest = write_json_atomically(characterization.MERGE_REPORT_PATH, payload)
        print(f"G8_C merge report written: report_id={payload['report_id']} sha256={digest} bytes={len(expected)}")
        return 0
    except Exception as exc:
        raise SystemExit(f"G8_C merge HOLD: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
