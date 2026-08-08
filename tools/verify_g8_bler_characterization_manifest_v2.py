#!/usr/bin/env python3
"""Independently verify the post-data G8_C source-epoch chain.

This verifier intentionally does not import the epoch-2 source-manifest
builder, the epoch-1 verifier, the coordinator, or the merge generator.  It
reconstructs the frozen bindings and the live B3 census from their registered
bytes, so it remains usable at the activation boundary, during C2, and after
the final raw evidence exists.
"""

from __future__ import annotations

import argparse
import ast
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
from baseline.g8_campaign import (  # noqa: E402
    CAMPAIGN_STATE,
    COUNTER_FIELDS,
    REPO_ROOT,
    canonical_json,
    load_campaign_state,
    rendered_json,
    sha256_bytes,
    sha256_file,
)


V2_MANIFEST = REPO_ROOT / "results/baseline/g8/bler_characterization_source_manifest_v2.json"
V1_MANIFEST = REPO_ROOT / "results/baseline/g8/bler_characterization_source_manifest.json"
V2_MANIFEST_PATH = "results/baseline/g8/bler_characterization_source_manifest_v2.json"
V1_MANIFEST_PATH = "results/baseline/g8/bler_characterization_source_manifest.json"
V1_MANIFEST_ID = "g8charsrc-6926319673ca1f55b95f8746062518c12cfa499aa827448e67850b5a1f74702a"
V1_MANIFEST_SHA256 = "a917f839f945232e85852d6d27f02de4b5dc272adc72b1966a95e9b5e62a014e"
V1_MANIFEST_BYTES = 6672  # literal-ok: immutable epoch-1 manifest byte count
V2_MANIFEST_ID_PREFIX = "g8charsrc2"
REQUIRED_COUNT = 3213  # literal-ok: authenticated G8_A required-work-unit count
FULL_STRENGTH_TRIALS = 5000  # literal-ok: authenticated G8_A full-strength trial count
EPOCH1_SOURCES = (
    "src/baseline/g8_bler_characterization.py",
    "tools/run_g8_bler_characterization.py",
    "tools/gen_g8_bler_characterization_manifest.py",
    "tools/verify_g8_bler_characterization_manifest.py",
    "tools/merge_g8_bler_characterization.py",
    "tools/verify_g8_bler_table.py",
)
EPOCH2_SOURCES = (
    "src/baseline/g8_bler_characterization_v2.py",
    "tools/run_g8_bler_characterization_v2.py",
    "tools/gen_g8_bler_characterization_manifest_v2.py",
    "tools/verify_g8_bler_characterization_manifest_v2.py",
    "tools/merge_g8_bler_characterization_v2.py",
    "tools/verify_g8_bler_table_v2.py",
)
DEPENDENCIES = (
    "src/baseline/classical/composition.py",
    "results/baseline/g8/bler_tooling_contract.json",
    "results/baseline/g8/bler_state_contract.json",
    "results/baseline/g8/bler_resume_contract.json",
    "results/baseline/g8/bler_runner_contract.json",
)
EXPECTED_ACTIVATION = {
    "epoch_1_accepted_result_start_ordinal": 0,
    "epoch_1_accepted_result_end_ordinal": 178,
    "epoch_1_accepted_result_count": 179,
    "first_possible_epoch_2_accepted_ordinal": 179,
    "first_epoch_2_work_unit_id": "bler-0e7c3102fbf553ba90fc5458",
    "first_legal_epoch_2_attempt": 3,
    "ordinal_179_request_attempts": [1, 2],
    "ordinal_179_result_attempts": [],
    "ordinal_179_request_sha256": "42062b62a4f88e08193b00fee25ed998ccbcd502782a206f231d10aae4c1b1c6",
    "ordinal_179_state_sha256": "e4b90d82fdc5760bd6298e82650e266ad759e2271e807be57a258bc6678b9361",
    "protected_counters": {field: 0 for field in COUNTER_FIELDS},
    "test_split_access": 0,
}


class VerificationError(RuntimeError):
    """The registered source-epoch chain or raw evidence is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _read(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read {label}: {exc}") from exc
    _require(isinstance(payload, dict), f"{label} is not an object")
    _require(raw == rendered_json(payload), f"{label} is not canonical rendered JSON")
    return payload, raw


def _binding(path: str, role: str) -> dict[str, Any]:
    raw = (REPO_ROOT / path).read_bytes()
    return {"path": path, "role": role, "sha256": sha256_bytes(raw), "bytes": len(raw)}


def _self_id(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("manifest_id", None)
    return f"{V2_MANIFEST_ID_PREFIX}-{sha256_bytes(canonical_json(body))}"


def _assert_no_forbidden_provenance(value: Any, path: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _require(
                key not in {"commit_sha", "hostname", "pid", "timestamp", "gpu_name", "device_count"},
                f"forbidden volatile provenance field {path}.{key}",
            )
            _assert_no_forbidden_provenance(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_forbidden_provenance(child, f"{path}[{index}]")
    elif isinstance(value, str):
        _require(not value.startswith("/"), f"absolute path bound at {path}")


def _contract(path: str) -> tuple[str, str, dict[str, Any]]:
    payload, raw = _read(REPO_ROOT / path, path)
    identifier = payload.get("contract_id")
    _require(isinstance(identifier, str) and identifier, f"{path} has no contract ID")
    return identifier, sha256_bytes(raw), payload


def _authority() -> dict[str, Any]:
    campaign, _ = _read(REPO_ROOT / "results/baseline/g8/campaign_manifest.json", "campaign manifest")
    required, _ = _read(REPO_ROOT / "results/baseline/g8/required_bler_identities.json", "required identities")
    b1_id, b1_sha, b1 = _contract("results/baseline/g8/bler_tooling_contract.json")
    b2_id, b2_sha, b2 = _contract("results/baseline/g8/bler_state_contract.json")
    b3_id, b3_sha, _b3 = _contract("results/baseline/g8/bler_resume_contract.json")
    b4_id, b4_sha, _b4 = _contract("results/baseline/g8/bler_runner_contract.json")
    units = required["required_bler_work_units"]
    _require(len(units) == REQUIRED_COUNT, "required work-unit count drift")
    return {
        "campaign_id": campaign["campaign_id"],
        "campaign_manifest_sha256": sha256_file(REPO_ROOT / "results/baseline/g8/campaign_manifest.json"),
        "required_bler_artifact_sha256": sha256_file(REPO_ROOT / "results/baseline/g8/required_bler_identities.json"),
        "selection_policy_sha256": campaign["selection_policy"]["selection_policy_sha256"],
        "required_work_unit_count": len(units),
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


def _verify_epoch1_predecessor(state: Mapping[str, Any]) -> None:
    raw = V1_MANIFEST.read_bytes()
    _require(len(raw) == V1_MANIFEST_BYTES, "epoch-1 manifest byte count drift")
    _require(sha256_bytes(raw) == V1_MANIFEST_SHA256, "epoch-1 manifest SHA drift")
    predecessor, _ = _read(V1_MANIFEST, "epoch-1 source manifest")
    _require(predecessor.get("manifest_id") == V1_MANIFEST_ID, "epoch-1 manifest ID drift")
    _require(predecessor.get("schema_version") == 1, "epoch-1 manifest schema drift")
    for field, paths, role in (("sources", EPOCH1_SOURCES, "g8_c_characterization_source"), ("dependencies", DEPENDENCIES, "g8_c_frozen_dependency")):
        entries = predecessor.get(field)
        _require(isinstance(entries, list), f"epoch-1 {field} are not a list")
        _require([entry.get("path") for entry in entries] == list(paths), f"epoch-1 {field} order drift")
        for entry, path in zip(entries, paths, strict=True):
            _require(entry == _binding(path, role), f"epoch-1 {field} changed: {path}")
    matches = [entry for entry in state["identity"]["produced_artifacts"] if entry["path"] == V1_MANIFEST_PATH]
    _require(matches == [{"path": V1_MANIFEST_PATH, "sha256": V1_MANIFEST_SHA256, "bytes": V1_MANIFEST_BYTES}], "epoch-1 manifest registration changed")


def _verify_source_files(payload: Mapping[str, Any]) -> None:
    for field, paths, role in (("sources", EPOCH2_SOURCES, "g8_c_characterization_source"), ("dependencies", DEPENDENCIES, "g8_c_frozen_dependency")):
        entries = payload[field]
        _require(isinstance(entries, list), f"{field} are not a list")
        _require([entry.get("path") for entry in entries] == list(paths), f"{field} order drift")
        for entry, path in zip(entries, paths, strict=True):
            _require(entry == _binding(path, role), f"epoch-2 {field} changed: {path}")
    _require(not set(EPOCH1_SOURCES) & set(EPOCH2_SOURCES), "source epochs overlap by path")
    prohibited = {"data", "data.manifests", "src.data.test_access", "baseline.classical.codec", "baseline.classifier"}
    for relative in EPOCH2_SOURCES:
        path = REPO_ROOT / relative
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except (OSError, SyntaxError) as exc:
            raise VerificationError(f"cannot parse epoch-2 source {relative}: {exc}") from exc
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            _require(not any(name in prohibited or name.startswith("data.") for name in names), f"forbidden data-boundary import in {relative}")


def _verify_manifest(state: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    payload, raw = _read(V2_MANIFEST, "epoch-2 source manifest")
    expected_fields = {
        "schema_version", "artifact_role", "phase", "checkpoint", "manifest_id", "epoch", "predecessor",
        "activation_boundary", "source_epochs", "scientific_execution_performed", "characterization_started",
        "campaign_id", "campaign_manifest_sha256", "required_bler_artifact_sha256", "selection_policy_sha256",
        "required_work_unit_count", "full_strength_trials", "request_schema_version", "result_schema_version",
        "unit_state_schema_version", "bler_tooling_contract_id", "bler_tooling_contract_sha256", "bler_state_contract_id",
        "bler_state_contract_sha256", "bler_resume_contract_id", "bler_resume_contract_sha256", "bler_runner_contract_id",
        "bler_runner_contract_sha256", "seed_derivation_identity", "runtime", "execution", "retry_policy", "count_semantics",
        "merge_completeness_predicate", "table_schema", "handoff", "merge_attribution", "table_attribution", "sources", "dependencies",
    }
    _require(set(payload) == expected_fields, "epoch-2 source manifest schema drift")
    _require(payload["schema_version"] == 2, "epoch-2 source manifest schema is not 2")
    _require(payload["artifact_role"] == "g8_bler_characterization_source_manifest_v2", "epoch-2 source manifest role drift")
    _require(payload["phase"] == "G8_C" and payload["checkpoint"] == "C1", "epoch-2 source manifest phase/checkpoint drift")
    _require(payload["scientific_execution_performed"] is False and payload["characterization_started"] is False, "epoch-2 manifest claims execution")
    _require(payload["manifest_id"] == _self_id(payload), "epoch-2 manifest self-hash does not reproduce")
    _assert_no_forbidden_provenance(payload)
    authority = _authority()
    for field, expected in authority.items():
        _require(payload[field] == expected, f"epoch-2 authority drift: {field}")
    _require(payload["runtime"] == {"logical_root": "results/baseline/g8/work_units", "absolute_paths_bound": False}, "runtime binding drift")
    _require(payload["execution"]["execution_class"] == "full_strength", "execution class drift")
    _require(payload["execution"]["sharding_algorithm"] == work_units.SHARDING_ALGORITHM, "sharding algorithm drift")
    _require(payload["execution"]["shard_formula"] == work_units.SHARD_FORMULA, "shard formula drift")
    _require(payload["retry_policy"]["failed_attempts_preserved"] is True and payload["retry_policy"]["next_attempt_is_clean"] is True, "retry policy drift")
    _require(payload["retry_policy"]["no_mid_unit_resume"] is True, "mid-unit resume is enabled")
    completeness = payload["merge_completeness_predicate"]
    _require(completeness["required_ids"] == REQUIRED_COUNT and completeness["trials_completed_per_accepted_result"] == FULL_STRENGTH_TRIALS, "merge count policy drift")
    _require(completeness["sum_coverage_contribution"] == REQUIRED_COUNT and completeness["test_split_access"] == 0, "merge coverage policy drift")
    _require(completeness["interpolation_used"] is False and completeness["extrapolation_used"] is False, "construction interpolation/extrapolation enabled")
    _require(payload["table_schema"]["schema_version"] == 2 and payload["table_schema"]["artifact_role"] == "g8_bler_table_v2", "table schema drift")
    _require(payload["table_schema"]["interpolation_during_construction"] is False and payload["table_schema"]["extrapolation"] is False, "table construction policy drift")
    _require(payload["handoff"]["g8_d_execution"] is False, "G8_D execution claimed")
    _require(payload["activation_boundary"] == EXPECTED_ACTIVATION, "activation boundary is not the frozen 179-unit boundary")
    _require(payload["source_epochs"] == [
        {"epoch": 1, "accepted_result_ordinals": [0, 178], "accepted_result_count": 179, "manifest_path": V1_MANIFEST_PATH, "manifest_id": V1_MANIFEST_ID, "manifest_sha256": V1_MANIFEST_SHA256},
        {"epoch": 2, "accepted_result_ordinals": [179, 3212], "accepted_result_count": 3034, "manifest_path": V2_MANIFEST_PATH, "manifest_id": None, "manifest_sha256": None},
    ], "source epoch ranges/chain drift")
    _require(payload["merge_attribution"]["request_only_attempts_are_not_failed_results"] is True, "request-only attempts are misclassified")
    _require(payload["merge_attribution"]["no_overlap"] is True and payload["merge_attribution"]["no_gap"] is True, "source epoch range policy drift")
    _verify_epoch1_predecessor(state)
    _verify_source_files(payload)
    binding = {"path": V2_MANIFEST_PATH, "sha256": sha256_bytes(raw), "bytes": len(raw)}
    matches = [entry for entry in state["identity"]["produced_artifacts"] if entry["path"] == V2_MANIFEST_PATH]
    _require(matches == [binding], "epoch-2 source manifest is not registered with exact bytes")
    return payload, raw


def _verify_unit_history(
    context: resume.AuthenticatedResumeContext,
    root: Path,
    record: Mapping[str, Any],
    census: Mapping[str, Any],
) -> None:
    work_unit_id = record["work_unit_id"]
    request_attempts = list(census["request_attempts"].get(work_unit_id, []))
    result_attempts = list(census["result_attempts"].get(work_unit_id, []))
    final_attempt = record.get("attempt")
    _require(isinstance(final_attempt, int) and final_attempt > 0, f"invalid final attempt for {work_unit_id}")
    _require(final_attempt in result_attempts, f"final result is absent for {work_unit_id}")
    _require(request_attempts == list(range(1, max(request_attempts, default=0) + 1)), f"request attempts are not contiguous for {work_unit_id}")
    for attempt in request_attempts:
        request_record = resume.validate_request_file(context, work_unit_id, attempt, root=root, require_full_strength=True)
        if attempt not in result_attempts:
            _require(attempt != final_attempt, f"final attempt has no result for {work_unit_id}")
            continue
        result_record = resume.validate_result_file(
            context,
            work_unit_id,
            attempt,
            root=root,
            request_record=request_record,
            shard_index=record["shard_index"],
            shard_count=record["shard_count"],
            scan_mode=resume.SCAN_MODE_PRODUCTION_MERGE,
        )
        if attempt == final_attempt:
            _require(result_record["status"] == bler_contract.STATUS_COMPLETE, f"final result is not complete for {work_unit_id}")
            measurement = result_record["result"]["measurement"]
            _require(measurement["trials_completed"] == FULL_STRENGTH_TRIALS, f"accepted result trial count changed for {work_unit_id}")
        else:
            _require(result_record["status"] == bler_contract.STATUS_FAILED, f"historical result is not failed for {work_unit_id}/{attempt}")
    state_path = resume.state_path(context, work_unit_id, root=root)
    state = work_units.read_unit_state(context.state_context, state_path, root=root)
    _require(state["identity"]["status"] == work_units.STATUS_RESULT_LINKED, f"accepted state is not result_linked: {work_unit_id}")
    _require(state["identity"]["result_sha256"] == resume.validate_result_file(
        context, work_unit_id, final_attempt, root=root,
        request_record=resume.validate_request_file(context, work_unit_id, final_attempt, root=root, require_full_strength=True),
        shard_index=record["shard_index"], shard_count=record["shard_count"], scan_mode=resume.SCAN_MODE_PRODUCTION_MERGE,
    )["result_sha256"], f"state/result linkage changed for {work_unit_id}")


def _verify_census(state: Mapping[str, Any]) -> dict[str, Any]:
    identity = state["identity"]
    _require(identity["phase"] == "G8_C" and identity["stage"] in {"characterization_open", "characterization_complete"}, "campaign is outside G8_C")
    _require(identity["in_progress_work_unit_id"] is None, "campaign has an in-progress unit")
    _require(all(value == 0 for value in identity["counters"].values()), "protected counter changed")
    context = resume.AuthenticatedResumeContext(require_resume_contract=True)
    required_ids = list(context.ordered_work_unit_ids)
    completed = list(identity["completed_work_unit_ids"])
    _require(completed == required_ids[: len(completed)], "completed IDs are not an authority prefix")
    root = REPO_ROOT / "results/baseline/g8/work_units"
    inspection = resume.inspect_runtime_root(context, root=root, scan_mode=resume.SCAN_MODE_PRODUCTION_MERGE, repair_mode=resume.REPAIR_MODE_READ_ONLY)
    _require(inspection["test_split_access"] == 0, "resume inspection claims test access")
    records = inspection["classifications"]
    _require(len(records) == REQUIRED_COUNT, "census work-unit count drift")
    complete_ids = [record["work_unit_id"] for record in records if record["classification"] == resume.CLASSIFICATION_COMPLETED_FULL_STRENGTH]
    _require(complete_ids == completed, "campaign completed IDs disagree with authenticated census")
    for record in records:
        ordinal = record["canonical_ordinal"]
        _require(record["classification"] != resume.CLASSIFICATION_CLAIMED_UNBOUND, f"unbound claim exists at ordinal {ordinal}")
        if record["classification"] == resume.CLASSIFICATION_COMPLETED_FULL_STRENGTH:
            _require(ordinal < REQUIRED_COUNT, "completed ordinal is outside authority")
            _verify_unit_history(context, root, record, inspection["census"])
        elif ordinal == 179 and record["classification"] == resume.CLASSIFICATION_CLAIMED_REQUEST_PUBLISHED:
            _require(inspection["census"]["request_attempts"].get(record["work_unit_id"]) == [1, 2], "ordinal 179 request-only history changed")
            _require(inspection["census"]["result_attempts"].get(record["work_unit_id"], []) == [], "ordinal 179 has a pre-epoch-2 result")
    return {"inspection": inspection, "context": context, "root": root}


def verify(*, require_registered: bool = True) -> dict[str, Any]:
    state = load_campaign_state(CAMPAIGN_STATE)
    payload, raw = _verify_manifest(state)
    census = _verify_census(state)
    return {
        "manifest": payload,
        "manifest_sha256": sha256_bytes(raw),
        "manifest_bytes": len(raw),
        "completed_count": len(state["identity"]["completed_work_unit_ids"]),
        "remaining_count": REQUIRED_COUNT - len(state["identity"]["completed_work_unit_ids"]),
        "test_split_access": 0,
        "census": census["inspection"]["census"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-registered", action="store_true", default=True)
    args = parser.parse_args(argv)
    try:
        result = verify(require_registered=args.require_registered)
    except Exception as exc:
        raise SystemExit(f"G8_C epoch-2 source-manifest HOLD: {exc}") from exc
    print(
        "G8_C epoch-2 source-manifest verification PASS: "
        + json.dumps({key: result[key] for key in ("completed_count", "remaining_count", "manifest_sha256", "manifest_bytes")}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
