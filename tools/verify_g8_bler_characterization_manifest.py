#!/usr/bin/env python3
"""Independently verify the G8_C pre-data characterization source manifest."""

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
from baseline import g8_bler_work_units as work_units  # noqa: E402
from baseline.g8_campaign import (  # noqa: E402
    CAMPAIGN_STATE,
    REPO_ROOT,
    canonical_json,
    load_campaign_state,
    rendered_json,
    sha256_bytes,
    sha256_file,
)


MANIFEST = REPO_ROOT / "results/baseline/g8/bler_characterization_source_manifest.json"
MANIFEST_ROLE = "g8_bler_characterization_source_manifest"
SOURCE_PATHS = (
    "src/baseline/g8_bler_characterization.py",
    "tools/run_g8_bler_characterization.py",
    "tools/gen_g8_bler_characterization_manifest.py",
    "tools/verify_g8_bler_characterization_manifest.py",
    "tools/merge_g8_bler_characterization.py",
    "tools/verify_g8_bler_table.py",
)
DEPENDENCY_PATHS = (
    "src/baseline/classical/composition.py",
    "results/baseline/g8/bler_tooling_contract.json",
    "results/baseline/g8/bler_state_contract.json",
    "results/baseline/g8/bler_resume_contract.json",
    "results/baseline/g8/bler_runner_contract.json",
)
EXPECTED_G8_D_RESTART = (
    'rg -n "G8_D|measurement_tooling_open|codec|reconstruction|BR-11|clean_classifier|'
    'bler_table|characterization_complete" src/baseline tools tests instructions'
)


class VerificationError(RuntimeError):
    """The independent manifest reconstruction failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _binding(path: str, role: str) -> dict[str, Any]:
    raw = (REPO_ROOT / path).read_bytes()
    return {"path": path, "role": role, "sha256": sha256_bytes(raw), "bytes": len(raw)}


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    _require(isinstance(payload, dict), f"{path} is not an object")
    _require(raw == rendered_json(payload), f"{path} is not canonical rendered JSON")
    return payload, raw


def _contract(path: str) -> tuple[str, str, int, dict[str, Any]]:
    payload, raw = _read(REPO_ROOT / path)
    return payload["contract_id"], sha256_bytes(raw), len(raw), payload


def _id(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("manifest_id", None)
    return f"g8charsrc-{sha256_bytes(canonical_json(body))}"


def _assert_no_absolute_paths(value: Any, path: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _require(key not in {"commit_sha", "hostname", "pid", "timestamp", "gpu_name", "device_count"}, f"forbidden provenance field {path}.{key}")
            _assert_no_absolute_paths(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_absolute_paths(child, f"{path}[{index}]")
    elif isinstance(value, str):
        _require(not value.startswith("/"), f"absolute path bound at {path}")


def verify(*, require_registered: bool = False) -> dict[str, Any]:
    payload, raw = _read(MANIFEST)
    expected_fields = {
        "schema_version", "artifact_role", "phase", "checkpoint", "manifest_id",
        "scientific_execution_performed", "characterization_started", "campaign_id",
        "campaign_manifest_sha256", "required_bler_artifact_sha256", "selection_policy_sha256",
        "required_work_unit_count", "full_strength_trials", "request_schema_version",
        "result_schema_version", "unit_state_schema_version", "bler_tooling_contract_id",
        "bler_tooling_contract_sha256", "bler_state_contract_id", "bler_state_contract_sha256",
        "bler_resume_contract_id", "bler_resume_contract_sha256", "bler_runner_contract_id",
        "bler_runner_contract_sha256", "seed_derivation_identity", "runtime", "execution",
        "retry_policy", "count_semantics", "merge_completeness_predicate", "table_schema",
        "handoff", "sources", "dependencies",
    }
    _require(set(payload) == expected_fields, "source manifest top-level schema drift")
    _require(payload["schema_version"] == 1, "source manifest schema_version is not 1")
    _require(payload["artifact_role"] == MANIFEST_ROLE, "source manifest role drift")
    _require(payload["phase"] == "G8_C" and payload["checkpoint"] == "C1", "source manifest phase/checkpoint drift")
    _require(payload["scientific_execution_performed"] is False, "source manifest claims scientific execution")
    _require(payload["characterization_started"] is False, "source manifest claims characterization started")
    _require(payload["manifest_id"] == _id(payload), "source manifest ID does not reproduce")
    _assert_no_absolute_paths(payload)

    campaign, _campaign_raw = _read(REPO_ROOT / "results/baseline/g8/campaign_manifest.json")
    required, _required_raw = _read(REPO_ROOT / "results/baseline/g8/required_bler_identities.json")
    _require(len(required["required_bler_work_units"]) == 3213, "required identity count drift")
    b1_id, b1_sha, _b1_bytes, b1 = _contract("results/baseline/g8/bler_tooling_contract.json")
    b2_id, b2_sha, _b2_bytes, b2 = _contract("results/baseline/g8/bler_state_contract.json")
    b3_id, b3_sha, _b3_bytes, b3 = _contract("results/baseline/g8/bler_resume_contract.json")
    b4_id, b4_sha, _b4_bytes, b4 = _contract("results/baseline/g8/bler_runner_contract.json")
    expected = {
        "campaign_id": campaign["campaign_id"],
        "campaign_manifest_sha256": sha256_file(REPO_ROOT / "results/baseline/g8/campaign_manifest.json"),
        "required_bler_artifact_sha256": sha256_file(REPO_ROOT / "results/baseline/g8/required_bler_identities.json"),
        "selection_policy_sha256": campaign["selection_policy"]["selection_policy_sha256"],
        "required_work_unit_count": len(required["required_bler_work_units"]),
        "full_strength_trials": bler_contract.full_strength_trial_count(),
        "request_schema_version": b1["request_schema"]["version"],
        "result_schema_version": b1["result_schema"]["version"],
        "unit_state_schema_version": b2["unit_state_schema"]["schema_version"],
        "bler_tooling_contract_id": b1_id,
        "bler_tooling_contract_sha256": b1_sha,
        "bler_state_contract_id": b2_id,
        "bler_state_contract_sha256": b2_sha,
        "bler_resume_contract_id": b3_id,
        "bler_resume_contract_sha256": b3_sha,
        "bler_runner_contract_id": b4_id,
        "bler_runner_contract_sha256": b4_sha,
        "seed_derivation_identity": bler_contract.SEED_DERIVATION_IDENTITY,
    }
    for field, value in expected.items():
        _require(payload[field] == value, f"source manifest authority mismatch: {field}")

    runtime = payload["runtime"]
    _require(runtime == {"logical_root": "results/baseline/g8/work_units", "absolute_paths_bound": False}, "runtime binding drift")
    execution = payload["execution"]
    _require(execution["execution_class"] == "full_strength", "execution class drift")
    _require(execution["sharding_algorithm"] == work_units.SHARDING_ALGORITHM, "sharding algorithm drift")
    _require(execution["shard_formula"] == work_units.SHARD_FORMULA, "shard formula drift")
    _require(execution["request_bytes_are_identity_across_attempts"] is True, "request-byte retry identity drift")
    retry = payload["retry_policy"]
    _require(retry["failed_attempts_preserved"] is True and retry["next_attempt_is_clean"] is True, "retry preservation drift")
    _require(retry["no_mid_unit_resume"] is True and retry["retry_only_frozen_b3_next_attempt"] is True, "mid-unit/retry drift")
    counts = payload["merge_completeness_predicate"]
    _require(counts["required_ids"] == 3213, "merge required count drift")
    _require(counts["trials_completed_per_accepted_result"] == bler_contract.full_strength_trial_count(), "merge trial count drift")
    _require(counts["sum_coverage_contribution"] == 3213 and counts["test_split_access"] == 0, "merge coverage drift")
    _require(counts["interpolation_used"] is False and counts["extrapolation_used"] is False, "merge construction invented points")
    table_schema = payload["table_schema"]
    _require(table_schema["schema_version"] == 1 and table_schema["artifact_role"] == "g8_bler_table", "table schema drift")
    _require(table_schema["interpolation_during_construction"] is False and table_schema["extrapolation"] is False, "table construction policy drift")
    _require(payload["handoff"]["exact_restart_command"] == EXPECTED_G8_D_RESTART, "G8_D handoff command drift")
    _require(payload["handoff"]["next_phase"] == "G8_D" and payload["handoff"]["next_stage"] == "measurement_tooling_open", "G8_D handoff stage drift")
    _require(payload["handoff"]["g8_d_execution"] is False, "source manifest claims G8_D execution")

    for field, paths, role in (("sources", SOURCE_PATHS, "g8_c_characterization_source"), ("dependencies", DEPENDENCY_PATHS, "g8_c_frozen_dependency")):
        entries = payload[field]
        _require([entry.get("path") for entry in entries] == list(paths), f"{field} order drift")
        for entry, path in zip(entries, paths, strict=True):
            _require(entry == _binding(path, role), f"{field} binding changed: {path}")
            _require(path != "results/baseline/g8/bler_characterization_source_manifest.json", "source manifest binds itself")

    state = load_campaign_state(CAMPAIGN_STATE)
    identity = state["identity"]
    _require(identity["phase"] == "G8_C" and identity["stage"] in {"characterization_open", "characterization_complete"}, "campaign is outside G8_C")
    _require(not identity["completed_work_unit_ids"] and identity["in_progress_work_unit_id"] is None, "C1 source manifest verification requires pre-data state") if identity["stage"] == "characterization_open" else None
    _require(all(value == 0 for value in identity["counters"].values()), "protected counter changed")
    if require_registered:
        matches = [entry for entry in identity["produced_artifacts"] if entry["path"] == "results/baseline/g8/bler_characterization_source_manifest.json"]
        _require(len(matches) == 1, "source manifest is not registered exactly once")
        _require(matches[0]["sha256"] == sha256_bytes(raw) and matches[0]["bytes"] == len(raw), "registered source manifest binding changed")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-registered", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload = verify(require_registered=args.require_registered)
    except Exception as exc:
        raise SystemExit(f"G8_C source-manifest HOLD: {exc}") from exc
    print(
        "G8_C source-manifest verification PASS: "
        f"manifest_id={payload['manifest_id']} sha256={sha256_file(MANIFEST)} bytes={MANIFEST.stat().st_size}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
