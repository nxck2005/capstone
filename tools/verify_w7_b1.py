#!/usr/bin/env python3
"""W7-B1 source, authorization, and non-scientific resume evidence verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from training.deterministic_core import canonical_bytes, canonical_sha256  # noqa: E402
from training.w7_protocol import (  # noqa: E402
    W7_CALIBRATION_SNR_DB,
    W7_CHANNEL_SEED,
    W7_DATASET,
    W7_EXECUTION_IMAGE_FAMILY,
    W7_LAMBDA_GRID,
    W7_PHYSICAL_BATCH_SIZE,
    W7_PROFILE_ID,
    W7_PSNR_SNR_DB,
    W7_RATIO,
    W7_SELECTED_GPU_NAME,
    W7_SELECTED_GPU_UUID,
    W7_TRAIN_SEED,
    W7_TRAINING_SNR_DB,
    W7_VALIDATION_BATCH_SIZE,
    eligibility_for_role,
    protocol_descriptor,
)
from gen_w7_test_hardening import (  # noqa: E402
    verify_completion as verify_test_hardening_completion,
    verify_source as verify_test_hardening_source,
)
from verify_w7_a import verify as verify_w7_a, verify_profile_freeze  # noqa: E402


W7_ROOT = REPO / "results/learned/w7"
HISTORICAL_SOURCE_PATH = W7_ROOT / "w7_source_manifest.json"
HARDENING_SOURCE_PATH = W7_ROOT / "w7_source_manifest_v2.json"
HARDENING_COMPLETION_PATH = W7_ROOT / "w7_a_test_hardening_completion.json"
HISTORICAL_COMPLETION_PATH = W7_ROOT / "w7_a_completion.json"
PROFILE_FREEZE_PATH = W7_ROOT / "w7_pascal_profile_freeze.json"
B1_SOURCE_PATH = W7_ROOT / "w7_b1_source_manifest.json"
AUTHORIZATION_PATH = W7_ROOT / "w7_execution_authorization.json"
SMOKE_PATH = W7_ROOT / "w7_b1_cuda_resume_smoke.json"
COMPLETION_PATH = W7_ROOT / "w7_b1_completion.json"

B1_SOURCE_SCHEMA_VERSION = 3  # literal-ok: additive B1 source-authority schema
B1_SOURCE_ROLE = "W7_B1_SCIENTIFIC_SOURCE_MANIFEST"
AUTHORIZATION_SCHEMA_VERSION = 2  # literal-ok: successor execution-authorization schema
AUTHORIZATION_ROLE = "W7_G4_SCIENTIFIC_EXECUTION_AUTHORIZATION"
SMOKE_ROLE = "NON_SCIENTIFIC_W7_B1_CUDA_RESUME_SMOKE"
COMPLETION_ROLE = "W7_B1_PRE_EXECUTION_AUTHORIZATION_COMPLETION"

PROTECTED_COUNTER_KEYS = (
    "w7_scientific_optimizer_steps",
    "w7_lambda_pilot_runs",
    "w7_candidate_results",
    "g4_adjudications",
    "w8_final_training_runs",
    "learned_test_inference",
    "test_model_facing_access",
    "g8_scientific_changes",
    "f1_reruns",
    "f2_optimizer_steps_during_w7",
    "f3_reruns",
    "pass_one_reruns",
    "pass_two_reruns",
    "pass_three",
    "bler_regeneration",
)
ZERO_PROTECTED_COUNTERS = {key: 0 for key in PROTECTED_COUNTER_KEYS}

# The v2 hardening source set is the accepted W7-A scientific predecessor.  B1
# adds only the detached-launch boundary, its verifier/smoke, and their tests.
B1_SOURCE_EXTRA = {
    "tools/gen_w7_test_hardening.py": "w7_a_test_hardening_verifier",
    "tools/verify_w7_a.py": "w7a_verifier",
    "tools/verify_w7_profile.py": "profile_verifier",
    "tools/verify_w7_b1.py": "b1_source_and_authorization_verifier",
    "tools/gen_w7_b1_source_manifest.py": "b1_source_manifest_generator",
    "tools/run_w7_b1_cuda_resume_smoke.py": "b1_non_scientific_cuda_resume_smoke",
    "tests/test_w7_b1_launch_boundary.py": "b1_launch_boundary_regressions",
    "tests/test_w7_b1_authorization.py": "b1_authorization_regressions",
    "tests/test_w7_b1_cuda_resume_smoke.py": "b1_smoke_evidence_regressions",
}


class W7B1Hold(RuntimeError):
    """A W7-B1 source, authorization, or evidence boundary violation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise W7B1Hold(message)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise W7B1Hold(f"invalid or missing W7-B1 JSON: {path}") from None
    _require(isinstance(value, dict), f"W7-B1 JSON is not an object: {path}")
    return value


def _sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        raise W7B1Hold(f"cannot hash W7-B1 artifact: {path}") from None


def _full_sha(value: object, width: int) -> bool:
    return isinstance(value, str) and len(value) == width and all(
        character in "0123456789abcdef" for character in value
    )


def _git(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise W7B1Hold(f"cannot authenticate Git source checkout: {exc}") from None
    return result.stdout.strip()


def _git_entry(repo_root: Path, commit: str, path: str) -> dict[str, Any]:
    try:
        raw = subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=repo_root,
            capture_output=True,
            check=True,
        ).stdout
        blob = subprocess.run(
            ["git", "rev-parse", f"{commit}:{path}"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise W7B1Hold(f"B1 source path is absent at {commit}: {path}: {exc}") from None
    return {
        "path": path,
        "role": "",
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "git_blob": blob,
    }


def _source_paths() -> dict[str, str]:
    historical = _load(HARDENING_SOURCE_PATH)
    _require(
        historical.get("manifest_id") == "w7testsource-1cf7ce96ec6a7134ed900ef7bb2c45bb3d292df007123d522916c9d8784c679b",
        "B1 predecessor hardening source ID differs",
    )
    values = {
        str(entry["path"]): str(entry["role"])
        for entry in historical.get("entries", [])
    }
    values.update(B1_SOURCE_EXTRA)
    return dict(sorted(values.items()))


def _predecessor_refs() -> tuple[dict[str, Any], dict[str, Any]]:
    historical_source = _load(HARDENING_SOURCE_PATH)
    hardening_completion = _load(HARDENING_COMPLETION_PATH)
    source_ref = {
        "path": str(HARDENING_SOURCE_PATH.relative_to(REPO)),
        "manifest_id": historical_source["manifest_id"],
        "source_commit": historical_source["source_commit"],
        "file_sha256": _sha(HARDENING_SOURCE_PATH),
    }
    completion_ref = {
        "path": str(HARDENING_COMPLETION_PATH.relative_to(REPO)),
        "completion_id": hardening_completion["completion_id"],
        "file_sha256": _sha(HARDENING_COMPLETION_PATH),
    }
    return source_ref, completion_ref


def build_source(commit: str, *, repo_root: Path = REPO) -> dict[str, Any]:
    if not _full_sha(commit, 40):  # literal-ok: Git SHA-1 width
        raise W7B1Hold("B1 source commit must be a full Git SHA-1")
    source_ref, completion_ref = _predecessor_refs()
    entries = []
    for path, role in _source_paths().items():
        entry = _git_entry(repo_root, commit, path)
        entry["role"] = role
        entries.append(entry)
    body = {
        "schema_version": B1_SOURCE_SCHEMA_VERSION,
        "artifact_role": B1_SOURCE_ROLE,
        "source_commit": commit,
        "predecessor_test_hardening_source": source_ref,
        "predecessor_test_hardening_completion": completion_ref,
        "entries": entries,
        "production_source_changed": True,
        "production_changed_paths": [
            "tools/run_w7_campaign.py",
            "tools/verify_w7_a.py",
            "tools/verify_w7_b1.py",
        ],
        "change_classification": [
            "detached_launcher_source_authority_only",
            "execution_authorization_schema_and_binding_only",
            "non_scientific_cuda_resume_evidence_only",
        ],
        "scientific_semantics_changed": False,
        "g4_protocol_changed": False,
        "scientific_execution_authorization": "ABSENT",
    }
    body["manifest_id"] = "w7b1source-" + canonical_sha256(body)
    return body


def verify_source_manifest(
    value: dict[str, Any],
    *,
    current: bool,
    repo_root: Path = REPO,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "artifact_role",
        "source_commit",
        "predecessor_test_hardening_source",
        "predecessor_test_hardening_completion",
        "entries",
        "production_source_changed",
        "production_changed_paths",
        "change_classification",
        "scientific_semantics_changed",
        "g4_protocol_changed",
        "scientific_execution_authorization",
        "manifest_id",
    }
    _require(isinstance(value, dict) and set(value) == required, "B1 source manifest schema differs")
    expected = build_source(str(value.get("source_commit")), repo_root=repo_root)
    _require(value == expected, "B1 source manifest differs from bound Git bytes")

    predecessor = _load(HARDENING_SOURCE_PATH)
    try:
        verify_test_hardening_source(predecessor, current=False)
        hardening = verify_test_hardening_completion(predecessor and _load(HARDENING_COMPLETION_PATH), predecessor)
    except (ValueError, RuntimeError, OSError) as exc:
        raise W7B1Hold(f"W7-A test-hardening predecessor is invalid: {exc}") from None
    source_ref, completion_ref = _predecessor_refs()
    _require(value["predecessor_test_hardening_source"] == source_ref, "B1 predecessor source binding differs")
    _require(value["predecessor_test_hardening_completion"] == completion_ref, "B1 predecessor completion binding differs")
    _require(hardening["successor_source_manifest"]["manifest_id"] == predecessor["manifest_id"], "B1 hardening completion source differs")

    if current:
        for entry in value["entries"]:
            path = repo_root / str(entry["path"])
            _require(path.is_file() and not path.is_symlink(), f"B1 current source missing/unsafe: {entry['path']}")
            _require(path.stat().st_size == entry["bytes"] and _sha(path) == entry["sha256"], f"B1 current source byte drift: {entry['path']}")
    return value


def verify_scientific_checkout(source_commit: str, *, repo_root: Path = REPO) -> str:
    """Require the launcher checkout to be the frozen source and clean."""

    head = _git(repo_root, "rev-parse", "HEAD")
    _require(head == source_commit, "W7 scientific checkout HEAD differs from B1 source authority")
    dirty = _git(repo_root, "status", "--porcelain", "--untracked-files=all")
    _require(not dirty, "W7 scientific checkout is dirty")
    return head


def verify_source_path(
    path: Path,
    *,
    current: bool,
    repo_root: Path = REPO,
) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), "B1 source manifest is absent or unsafe")
    value = _load(path)
    return verify_source_manifest(value, current=current, repo_root=repo_root)


def _profile_freeze() -> tuple[dict[str, Any], str]:
    freeze = _load(PROFILE_FREEZE_PATH)
    try:
        verify_profile_freeze(freeze)
    except (ValueError, RuntimeError, OSError) as exc:
        raise W7B1Hold(f"Pascal profile freeze is invalid: {exc}") from None
    return freeze, _sha(PROFILE_FREEZE_PATH)


def _actual_upstream() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        w7a = verify_w7_a(run_upstream=False)
        hardening_source, hardening_completion = (
            verify_test_hardening_source(_load(HARDENING_SOURCE_PATH), current=False),
            verify_test_hardening_completion(_load(HARDENING_COMPLETION_PATH), _load(HARDENING_SOURCE_PATH)),
        )
    except (ValueError, RuntimeError, OSError) as exc:
        raise W7B1Hold(f"W7 upstream authority is invalid: {exc}") from None
    return w7a, hardening_source, hardening_completion


def _authorization_required() -> set[str]:
    return {
        "schema_version",
        "artifact_role",
        "authorization_role",
        "status",
        "authorization_scope",
        "authorization_id",
        "campaign_id",
        "w7_a_completion_path",
        "w7_a_completion_id",
        "w7_a_completion_sha256",
        "w7_test_hardening_completion_path",
        "w7_test_hardening_completion_id",
        "w7_test_hardening_completion_sha256",
        "w7_test_hardening_source_manifest_path",
        "w7_test_hardening_source_manifest_id",
        "w7_test_hardening_source_manifest_sha256",
        "source_manifest_path",
        "source_commit",
        "source_manifest_id",
        "source_manifest_sha256",
        "profile_freeze_path",
        "profile_freeze_id",
        "profile_freeze_sha256",
        "execution_image_family",
        "execution_profile_id",
        "gpu_uuid",
        "gpu_name",
        "lambda_grid",
        "lambda_order",
        "train_seed",
        "channel_seed",
        "training_snr_db",
        "calibration_snr_db",
        "psnr_snr_db",
        "ratio",
        "physical_batch_size",
        "accumulation_factor",
        "effective_batch_size",
        "validation_batch_size",
        "protocol",
        "scientific_execution_authorization",
        "test_access",
        "g4_adjudication",
        "w8",
        "lambda_selection",
        "lambda_core_updated",
        "protected_counters",
    }


def build_authorization(
    *,
    campaign_id: str,
    source_path: Path = B1_SOURCE_PATH,
    profile_freeze_path: Path = PROFILE_FREEZE_PATH,
) -> dict[str, Any]:
    _require(bool(campaign_id), "W7 authorization campaign ID is empty")
    source_path = source_path.resolve()
    profile_freeze_path = profile_freeze_path.resolve()
    source = verify_source_path(source_path, current=False)
    freeze, freeze_sha = _profile_freeze()
    w7a, hardening_source, hardening = _actual_upstream()
    source_sha = _sha(source_path)
    body = {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "artifact_role": AUTHORIZATION_ROLE,
        "authorization_role": AUTHORIZATION_ROLE,
        "status": "AUTHORIZED",
        "authorization_scope": "W7_B2_FIVE_LAMBDA_SCIENTIFIC_CAMPAIGN_ONLY",
        "campaign_id": campaign_id,
        "w7_a_completion_path": str(HISTORICAL_COMPLETION_PATH.relative_to(REPO)),
        "w7_a_completion_id": w7a["completion_id"],
        "w7_a_completion_sha256": _sha(HISTORICAL_COMPLETION_PATH),
        "w7_test_hardening_completion_path": str(HARDENING_COMPLETION_PATH.relative_to(REPO)),
        "w7_test_hardening_completion_id": hardening["completion_id"],
        "w7_test_hardening_completion_sha256": _sha(HARDENING_COMPLETION_PATH),
        "w7_test_hardening_source_manifest_path": str(HARDENING_SOURCE_PATH.relative_to(REPO)),
        "w7_test_hardening_source_manifest_id": hardening_source["manifest_id"],
        "w7_test_hardening_source_manifest_sha256": _sha(HARDENING_SOURCE_PATH),
        "source_manifest_path": str(source_path.relative_to(REPO)) if source_path.is_relative_to(REPO) else source_path.name,
        "source_commit": source["source_commit"],
        "source_manifest_id": source["manifest_id"],
        "source_manifest_sha256": source_sha,
        "profile_freeze_path": str(profile_freeze_path.relative_to(REPO)),
        "profile_freeze_id": freeze["profile_freeze_id"],
        "profile_freeze_sha256": freeze_sha,
        "execution_image_family": W7_EXECUTION_IMAGE_FAMILY,
        "execution_profile_id": W7_PROFILE_ID,
        "gpu_uuid": W7_SELECTED_GPU_UUID,
        "gpu_name": W7_SELECTED_GPU_NAME,
        "lambda_grid": list(W7_LAMBDA_GRID),
        "lambda_order": "exact_configured_lambda_grid_order",
        "train_seed": W7_TRAIN_SEED,
        "channel_seed": W7_CHANNEL_SEED,
        "training_snr_db": W7_TRAINING_SNR_DB,
        "calibration_snr_db": W7_CALIBRATION_SNR_DB,
        "psnr_snr_db": W7_PSNR_SNR_DB,
        "ratio": W7_RATIO,
        "physical_batch_size": freeze["physical_batch_size"],
        "accumulation_factor": freeze["accumulation_factor"],
        "effective_batch_size": freeze["effective_batch_size"],
        "validation_batch_size": freeze["validation_batch_size"],
        "protocol": protocol_descriptor(),
        "scientific_execution_authorization": "PRESENT",
        "test_access": "SEALED",
        "g4_adjudication": "NOT_AUTHORIZED_BY_THIS_ARTIFACT",
        "w8": "NOT_AUTHORIZED",
        "lambda_selection": False,
        "lambda_core_updated": False,
        "protected_counters": dict(ZERO_PROTECTED_COUNTERS),
    }
    value = dict(body)
    value["authorization_id"] = "w7auth-" + canonical_sha256(body)
    return value


def verify_execution_authorization(
    value: dict[str, Any],
    *,
    path: Path | None = None,
    repo_root: Path = REPO,
    verify_source: bool = False,
) -> dict[str, Any]:
    _require(isinstance(value, dict) and set(value) == _authorization_required(), "W7 execution authorization schema differs")
    body = dict(value)
    authorization_id = body.pop("authorization_id")
    _require(authorization_id == "w7auth-" + canonical_sha256(body), "W7 execution authorization digest differs")
    _require(
        value["schema_version"] == AUTHORIZATION_SCHEMA_VERSION
        and value["artifact_role"] == AUTHORIZATION_ROLE
        and value["authorization_role"] == AUTHORIZATION_ROLE
        and value["status"] == "AUTHORIZED"
        and value["authorization_scope"] == "W7_B2_FIVE_LAMBDA_SCIENTIFIC_CAMPAIGN_ONLY",
        "W7 execution authorization role/status differs",
    )
    _require(value["scientific_execution_authorization"] == "PRESENT", "W7 scientific execution authorization is absent")
    _require(value["test_access"] == "SEALED", "W7 execution authorization does not preserve the test seal")
    _require(value["g4_adjudication"] == "NOT_AUTHORIZED_BY_THIS_ARTIFACT", "W7 execution authorization cannot authorize G4")
    _require(value["w8"] == "NOT_AUTHORIZED" and value["lambda_selection"] is False and value["lambda_core_updated"] is False, "W7 execution authorization opens a forbidden downstream gate")
    _require(value["lambda_grid"] == list(W7_LAMBDA_GRID) and value["lambda_order"] == "exact_configured_lambda_grid_order", "W7 authorization lambda grid/order differs")
    _require(value["execution_image_family"] == W7_EXECUTION_IMAGE_FAMILY and value["execution_profile_id"] == W7_PROFILE_ID, "W7 authorization image/profile differs")
    _require(value["gpu_uuid"] == W7_SELECTED_GPU_UUID and value["gpu_name"] == W7_SELECTED_GPU_NAME, "W7 authorization GPU differs")
    _require(
        value["train_seed"] == W7_TRAIN_SEED
        and value["channel_seed"] == W7_CHANNEL_SEED
        and value["training_snr_db"] == W7_TRAINING_SNR_DB
        and value["calibration_snr_db"] == W7_CALIBRATION_SNR_DB
        and value["psnr_snr_db"] == W7_PSNR_SNR_DB
        and value["ratio"] == W7_RATIO,
        "W7 authorization protocol scalar differs",
    )
    _require(value["protocol"] == protocol_descriptor(), "W7 authorization protocol descriptor differs")
    _require(
        value["physical_batch_size"] == W7_PHYSICAL_BATCH_SIZE
        and value["effective_batch_size"] == 32  # literal-ok: owner-frozen effective batch
        and value["validation_batch_size"] == W7_VALIDATION_BATCH_SIZE,
        "W7 authorization batch policy differs",
    )
    _require(value["protected_counters"] == ZERO_PROTECTED_COUNTERS, "W7 authorization protected counter is nonzero")
    for field, width in (
        ("source_commit", 40),
        ("source_manifest_sha256", 64),
        ("profile_freeze_sha256", 64),
        ("w7_a_completion_sha256", 64),
        ("w7_test_hardening_completion_sha256", 64),
        ("w7_test_hardening_source_manifest_sha256", 64),
    ):
        _require(_full_sha(value[field], width), f"W7 authorization {field} is invalid")

    w7a, hardening_source, hardening = _actual_upstream()
    _require(value["w7_a_completion_path"] == str(HISTORICAL_COMPLETION_PATH.relative_to(REPO)), "W7 historical completion path differs")
    _require(value["w7_a_completion_id"] == w7a["completion_id"] and value["w7_a_completion_sha256"] == _sha(HISTORICAL_COMPLETION_PATH), "W7 authorization historical completion binding differs")
    _require(value["w7_test_hardening_completion_path"] == str(HARDENING_COMPLETION_PATH.relative_to(REPO)), "W7 hardening completion path differs")
    _require(value["w7_test_hardening_completion_id"] == hardening["completion_id"] and value["w7_test_hardening_completion_sha256"] == _sha(HARDENING_COMPLETION_PATH), "W7 authorization test-hardening completion binding differs")
    _require(value["w7_test_hardening_source_manifest_path"] == str(HARDENING_SOURCE_PATH.relative_to(REPO)), "W7 hardening source path differs")
    _require(value["w7_test_hardening_source_manifest_id"] == hardening_source["manifest_id"] and value["w7_test_hardening_source_manifest_sha256"] == _sha(HARDENING_SOURCE_PATH), "W7 authorization test-hardening source binding differs")
    freeze, freeze_sha = _profile_freeze()
    _require(value["profile_freeze_path"] == str(PROFILE_FREEZE_PATH.relative_to(REPO)) and value["profile_freeze_id"] == freeze["profile_freeze_id"] and value["profile_freeze_sha256"] == freeze_sha, "W7 authorization profile freeze binding differs")
    _require(
        value["physical_batch_size"] == freeze["physical_batch_size"]
        and value["accumulation_factor"] == freeze["accumulation_factor"]
        and value["effective_batch_size"] == freeze["effective_batch_size"]
        and value["validation_batch_size"] == freeze["validation_batch_size"],
        "W7 authorization freeze batch binding differs",
    )
    _require(value["source_manifest_path"] == "results/learned/w7/w7_b1_source_manifest.json", "W7 authorization source manifest path differs")
    if verify_source:
        source_path = (repo_root / value["source_manifest_path"]) if not Path(value["source_manifest_path"]).is_absolute() else Path(value["source_manifest_path"])
        source = verify_source_path(source_path, current=False, repo_root=repo_root)
        _require(value["source_commit"] == source["source_commit"] and value["source_manifest_id"] == source["manifest_id"] and value["source_manifest_sha256"] == _sha(source_path), "W7 authorization source binding differs")
    if path is not None:
        _require(path.is_file() and not path.is_symlink(), "W7 execution authorization path is absent or unsafe")
    return value


def verify_authorization_path(
    path: Path,
    *,
    repo_root: Path = REPO,
    verify_source: bool = False,
) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), "W7 execution authorization is absent; W7-B is not authorized")
    return verify_execution_authorization(_load(path), path=path, repo_root=repo_root, verify_source=verify_source)


def _verify_digest_record(value: dict[str, Any], *, prefix: str) -> None:
    identity = value.get("evidence_id")
    body = dict(value)
    body.pop("evidence_id", None)
    _require(identity == prefix + canonical_sha256(body), f"{prefix} evidence digest differs")


def verify_smoke(value: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "artifact_role",
        "smoke_id",
        "status",
        "scientific_status",
        "role",
        "eligibility",
        "source_commit",
        "source_manifest_id",
        "source_manifest_sha256",
        "profile_freeze_id",
        "profile_freeze_sha256",
        "execution_image_family",
        "execution_profile_id",
        "gpu_uuid",
        "gpu_name",
        "device",
        "amp_enabled",
        "grad_scaler_present",
        "smoke_config_hash",
        "smoke_batch",
        "process_a",
        "process_b",
        "resume_seam",
        "checkpoint_chain",
        "validation",
        "scientific_boundary",
        "non_scientific_w7_b1_resume_smoke_optimizer_steps",
        "protected_counters",
    }
    _require(isinstance(value, dict) and set(value) == required, "W7-B1 CUDA smoke schema differs")
    body = dict(value)
    smoke_id = body.pop("smoke_id")
    _require(smoke_id == "w7b1smoke-" + canonical_sha256(body), "W7-B1 CUDA smoke digest differs")
    _require(value["schema_version"] == 1 and value["artifact_role"] == SMOKE_ROLE and value["status"] == "PASSED", "W7-B1 CUDA smoke role/status differs")  # literal-ok: smoke evidence schema
    _require(value["scientific_status"] == "NON_SCIENTIFIC_ZERO_G4_COVERAGE" and value["role"] == "NON_SCIENTIFIC_PROFILE", "W7-B1 CUDA smoke is scientific")
    _require(value["eligibility"] == eligibility_for_role("NON_SCIENTIFIC_PROFILE"), "W7-B1 CUDA smoke eligibility differs")
    _require(_full_sha(value["source_commit"], 40) and _full_sha(value["source_manifest_sha256"], 64) and _full_sha(value["profile_freeze_sha256"], 64), "W7-B1 CUDA smoke lineage digest is invalid")
    _require(value["source_manifest_id"].startswith("w7b1source-") and value["profile_freeze_id"].startswith("w7profilefreeze-"), "W7-B1 CUDA smoke lineage ID differs")
    _require(value["execution_image_family"] == W7_EXECUTION_IMAGE_FAMILY and value["execution_profile_id"] == W7_PROFILE_ID, "W7-B1 CUDA smoke image/profile differs")
    _require(value["gpu_uuid"] == W7_SELECTED_GPU_UUID and value["gpu_name"] == W7_SELECTED_GPU_NAME and value["device"] == "cuda:0", "W7-B1 CUDA smoke GPU differs")
    _require(value["amp_enabled"] is True and value["grad_scaler_present"] is True, "W7-B1 CUDA smoke did not use AMP/GradScaler")
    _require(_full_sha(value["smoke_config_hash"], 64), "W7-B1 CUDA smoke config hash is invalid")
    batch = value["smoke_batch"]
    _require(set(batch) == {"dataset", "sample_count", "physical_batch_size", "accumulation_factor", "validation_batch_size"}, "W7-B1 CUDA smoke batch schema differs")
    _require(batch["dataset"] == "synthetic_w7_b1_fixture" and batch["sample_count"] > 0 and batch["validation_batch_size"] == W7_VALIDATION_BATCH_SIZE, "W7-B1 CUDA smoke dataset/batch differs")

    process_a = value["process_a"]
    process_b = value["process_b"]
    for process, prefix in ((process_a, "w7b1processa-"), (process_b, "w7b1processb-")):
        _require(isinstance(process, dict), "W7-B1 CUDA smoke process evidence is not a mapping")
        _verify_digest_record(process, prefix=prefix)
        _require(process["artifact_role"] == "NON_SCIENTIFIC_PROFILE_CHECKPOINT", "W7-B1 CUDA smoke checkpoint role is scientific")
        _require(process["amp_enabled"] is True and process["grad_scaler_present"] is True and process["validation_performed"] is False, "W7-B1 CUDA smoke process boundary differs")
        for key in ("checkpoint_id", "model_state_sha256", "optimizer_state_sha256", "scheduler_state_sha256", "scaler_state_sha256"):
            _require(_full_sha(process[key], 64), f"W7-B1 CUDA smoke {key} is invalid")
        _require(isinstance(process["optimizer_steps"], int) and process["optimizer_steps"] > 0, "W7-B1 CUDA smoke optimizer work is missing")
    _require(process_a["completed_epoch"] == 0 and process_b["completed_epoch"] == 1, "W7-B1 CUDA smoke epoch boundaries differ")  # literal-ok: two-epoch resume proof
    _require(process_a["global_optimizer_step"] > 0 and process_b["global_optimizer_step"] > process_a["global_optimizer_step"], "W7-B1 CUDA smoke global-step boundary differs")

    seam = value["resume_seam"]
    _require(
        seam == {
            "model_state_equal": True,
            "optimizer_state_equal": True,
            "scheduler_state_equal": True,
            "scaler_state_equal": True,
            "completed_epoch_equal": True,
            "global_optimizer_step_equal": True,
            "scaler_state_before_restore_sha256": process_a["scaler_state_sha256"],
            "scaler_state_after_restore_sha256": process_a["scaler_state_sha256"],
            "restored_checkpoint_id": process_a["checkpoint_id"],
        },
        "W7-B1 CUDA smoke resume seam differs",
    )
    _require(
        value["checkpoint_chain"] == {
            "latest_only": True,
            "older_fallback": False,
            "full_chain_authenticated": True,
            "successor_predecessor_checkpoint_id": process_a["checkpoint_id"],
            "successor_checkpoint_id": process_b["checkpoint_id"],
        },
        "W7-B1 CUDA smoke checkpoint chain differs",
    )
    _require(value["validation"] == {"performed": False, "model_facing": False}, "W7-B1 CUDA smoke performed validation")
    _require(value["scientific_boundary"] == {
        "scientific_execution_authorization": "NOT_USED",
        "w7_scientific_optimizer_steps": 0,
        "w7_candidate_results": 0,
        "g4_adjudications": 0,
        "w8_final_training_runs": 0,
        "learned_test_inference": 0,
        "test_model_facing_access": 0,
        "test_access": "SEALED",
    }, "W7-B1 CUDA smoke scientific boundary differs")
    expected_steps = process_a["optimizer_steps"] + process_b["optimizer_steps"]
    _require(value["non_scientific_w7_b1_resume_smoke_optimizer_steps"] == expected_steps > 0, "W7-B1 CUDA smoke step accounting differs")
    _require(value["protected_counters"] == ZERO_PROTECTED_COUNTERS, "W7-B1 CUDA smoke protected counter is nonzero")
    return value


def build_completion(*, authorization_path: Path = AUTHORIZATION_PATH, smoke_path: Path = SMOKE_PATH) -> dict[str, Any]:
    authorization = verify_authorization_path(authorization_path, verify_source=False)
    smoke = verify_smoke(_load(smoke_path))
    body = {
        "schema_version": 1,
        "artifact_role": COMPLETION_ROLE,
        "status": "GREEN_PRE_EXECUTION_AUTHORIZATION",
        "scientific_execution_authorization": "PRESENT",
        "authorization": {
            "path": str(authorization_path.relative_to(REPO)),
            "authorization_id": authorization["authorization_id"],
            "file_sha256": _sha(authorization_path),
            "campaign_id": authorization["campaign_id"],
        },
        "historical_w7_a_completion": {
            "path": str(HISTORICAL_COMPLETION_PATH.relative_to(REPO)),
            "completion_id": authorization["w7_a_completion_id"],
            "file_sha256": authorization["w7_a_completion_sha256"],
        },
        "w7_test_hardening_completion": {
            "path": str(HARDENING_COMPLETION_PATH.relative_to(REPO)),
            "completion_id": authorization["w7_test_hardening_completion_id"],
            "file_sha256": authorization["w7_test_hardening_completion_sha256"],
        },
        "scientific_source": {
            "path": authorization["source_manifest_path"],
            "source_commit": authorization["source_commit"],
            "manifest_id": authorization["source_manifest_id"],
            "file_sha256": authorization["source_manifest_sha256"],
        },
        "pascal_profile": {
            "profile_id": authorization["execution_profile_id"],
            "gpu_uuid": authorization["gpu_uuid"],
            "profile_freeze_id": authorization["profile_freeze_id"],
            "profile_freeze_sha256": authorization["profile_freeze_sha256"],
            "rerun": False,
        },
        "cuda_resume_smoke": {
            "path": str(smoke_path.relative_to(REPO)),
            "smoke_id": smoke["smoke_id"],
            "file_sha256": _sha(smoke_path),
            "role": smoke["role"],
            "status": "PASSED",
            "optimizer_steps": smoke["non_scientific_w7_b1_resume_smoke_optimizer_steps"],
        },
        "scientific_pilots_run": 0,
        "g4_adjudications": 0,
        "lambda_selected": False,
        "w8": "UNOPENED",
        "test": "SEALED",
        "protected_counters": dict(ZERO_PROTECTED_COUNTERS),
        "next_action": "RETURN_FOR_INDEPENDENT_AUDIT_AND_SEPARATE_W7_B2_FIVE_LAMBDA_LAUNCH_AUTHORIZATION",
    }
    value = dict(body)
    value["completion_id"] = "w7b1completion-" + canonical_sha256(body)
    return value


def verify_completion(value: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "artifact_role",
        "completion_id",
        "status",
        "scientific_execution_authorization",
        "authorization",
        "historical_w7_a_completion",
        "w7_test_hardening_completion",
        "scientific_source",
        "pascal_profile",
        "cuda_resume_smoke",
        "scientific_pilots_run",
        "g4_adjudications",
        "lambda_selected",
        "w8",
        "test",
        "protected_counters",
        "next_action",
    }
    _require(isinstance(value, dict) and set(value) == required, "W7-B1 completion schema differs")
    body = dict(value)
    completion_id = body.pop("completion_id")
    _require(completion_id == "w7b1completion-" + canonical_sha256(body), "W7-B1 completion digest differs")
    _require(value["schema_version"] == 1 and value["artifact_role"] == COMPLETION_ROLE and value["status"] == "GREEN_PRE_EXECUTION_AUTHORIZATION", "W7-B1 completion status differs")  # literal-ok: completion evidence schema
    _require(value["scientific_execution_authorization"] == "PRESENT" and value["scientific_pilots_run"] == 0 and value["g4_adjudications"] == 0 and value["lambda_selected"] is False and value["w8"] == "UNOPENED" and value["test"] == "SEALED", "W7-B1 completion opens scientific work")
    _require(value["protected_counters"] == ZERO_PROTECTED_COUNTERS, "W7-B1 completion protected counter is nonzero")
    auth_ref = value["authorization"]
    auth_path = REPO / auth_ref["path"]
    auth = verify_authorization_path(auth_path, verify_source=False)
    _require(auth_ref == {
        "path": str(AUTHORIZATION_PATH.relative_to(REPO)),
        "authorization_id": auth["authorization_id"],
        "file_sha256": _sha(AUTHORIZATION_PATH),
        "campaign_id": auth["campaign_id"],
    }, "W7-B1 authorization completion binding differs")
    smoke_ref = value["cuda_resume_smoke"]
    smoke = verify_smoke(_load(SMOKE_PATH))
    _require(smoke_ref == {
        "path": str(SMOKE_PATH.relative_to(REPO)),
        "smoke_id": smoke["smoke_id"],
        "file_sha256": _sha(SMOKE_PATH),
        "role": smoke["role"],
        "status": "PASSED",
        "optimizer_steps": smoke["non_scientific_w7_b1_resume_smoke_optimizer_steps"],
    }, "W7-B1 smoke completion binding differs")
    _require(value["historical_w7_a_completion"]["completion_id"] == auth["w7_a_completion_id"] and value["historical_w7_a_completion"]["file_sha256"] == auth["w7_a_completion_sha256"], "W7-B1 historical completion binding differs")
    _require(value["w7_test_hardening_completion"]["completion_id"] == auth["w7_test_hardening_completion_id"] and value["w7_test_hardening_completion"]["file_sha256"] == auth["w7_test_hardening_completion_sha256"], "W7-B1 test-hardening completion binding differs")
    _require(value["scientific_source"] == {
        "path": auth["source_manifest_path"],
        "source_commit": auth["source_commit"],
        "manifest_id": auth["source_manifest_id"],
        "file_sha256": auth["source_manifest_sha256"],
    }, "W7-B1 scientific source binding differs")
    _require(value["pascal_profile"] == {
        "profile_id": auth["execution_profile_id"],
        "gpu_uuid": auth["gpu_uuid"],
        "profile_freeze_id": auth["profile_freeze_id"],
        "profile_freeze_sha256": auth["profile_freeze_sha256"],
        "rerun": False,
    }, "W7-B1 Pascal profile binding differs")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    source = sub.add_parser("source")
    source.add_argument("--source-commit", required=True)
    source.add_argument("--output", type=Path, default=B1_SOURCE_PATH)
    authorization = sub.add_parser("authorization")
    authorization.add_argument("--campaign-id", required=True)
    authorization.add_argument("--source-manifest", type=Path, default=B1_SOURCE_PATH)
    authorization.add_argument("--output", type=Path, default=AUTHORIZATION_PATH)
    complete = sub.add_parser("completion")
    complete.add_argument("--authorization", type=Path, default=AUTHORIZATION_PATH)
    complete.add_argument("--smoke", type=Path, default=SMOKE_PATH)
    complete.add_argument("--output", type=Path, default=COMPLETION_PATH)
    verify = sub.add_parser("verify")
    verify.add_argument("--source-manifest", type=Path, default=B1_SOURCE_PATH)
    verify.add_argument("--authorization", type=Path, default=AUTHORIZATION_PATH)
    verify.add_argument("--smoke", type=Path, default=SMOKE_PATH)
    verify.add_argument("--completion", type=Path, default=COMPLETION_PATH)
    args = parser.parse_args(argv)
    if args.command == "source":
        value = build_source(args.source_commit)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_bytes(value))
        print(f"wrote {args.output}: {value['manifest_id']}")
    elif args.command == "authorization":
        value = build_authorization(campaign_id=args.campaign_id, source_path=args.source_manifest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_bytes(value))
        print(f"wrote {args.output}: {value['authorization_id']}")
    elif args.command == "completion":
        value = build_completion(authorization_path=args.authorization, smoke_path=args.smoke)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_bytes(value))
        print(f"wrote {args.output}: {value['completion_id']}")
    else:
        source = verify_source_path(args.source_manifest, current=False)
        authorization = verify_authorization_path(args.authorization, verify_source=True)
        _require(authorization["source_manifest_id"] == source["manifest_id"], "B1 verification source/authorization differs")
        _require(authorization["source_manifest_sha256"] == _sha(args.source_manifest), "B1 verification source SHA differs")
        verify_smoke(_load(args.smoke))
        verify_completion(_load(args.completion))
        print(f"W7-B1 PASS: {authorization['authorization_id']} / {source['manifest_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
