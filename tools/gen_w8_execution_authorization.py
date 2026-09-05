#!/usr/bin/env python3
"""Freeze the W8-A execution boundary on a carrier commit.

This artifact is authority, not a result.  It permits only the exact six-cell
core to be launched later under a separately detached W8-B launch
authorization.  It does not open G-10, ER-2, PAPR, ER-9, or the test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from config.params import get  # noqa: E402
from config.run_config import (  # noqa: E402
    canonical_sha256 as run_config_canonical_sha256,
    config_hash as run_config_hash,
)
from evaluation.g10_spec_compatibility import (  # noqa: E402
    ALLOWED_PARAMETER_PATHS as AM94_ALLOWED_PARAMETER_PATHS,
    load as load_am94_spec_compatibility,
)
from gen_w8_source_manifest import CRITICAL_SOURCES, verify_manifest  # noqa: E402
from training.deterministic_core import canonical_bytes, canonical_sha256  # noqa: E402
from training.w8_protocol import (  # noqa: E402
    W8_ACCUMULATION_FACTOR,
    W8_CHECKPOINT_SELECTION_CHANNEL_SEED_RULE,
    W8_CHECKPOINT_SELECTION_SNR_PARAMETER,
    W8_CHANNEL_SEEDS,
    W8_CORE_ROLE,
    W8_DATASET,
    W8_EPOCHS,
    W8_EFFECTIVE_BATCH_SIZE,
    W8_EXECUTION_IMAGE_FAMILY,
    W8_EXPECTED_K,
    W8_EXPECTED_LAMBDA,
    W8_EXPECTED_RATIOS,
    W8_PHYSICAL_BATCH_SIZE,
    W8_PROFILE_ID,
    W8_SELECTED_GPU_NAME,
    W8_SELECTED_GPU_UUID,
    W8_TRAIN_SAMPLE_COUNT,
    W8_TRAIN_SEEDS,
    W8_VALIDATION_BATCH_SIZE,
    W8_VALIDATION_SAMPLE_COUNT,
    checkpoint_selection_snr_db,
    protocol_config_hash,
    protocol_descriptor,
    run_cells,
    load_w8_config,
)


AUTHORIZATION_ROLE = "W8_EXECUTION_AUTHORIZATION"
AUTHORIZATION_PREFIX = "w8auth-"
DEFAULT_OUTPUT = REPO / "results/learned/w8/w8_execution_authorization.json"
CAMPAIGN_ID = "w8-final-pascal-20260831"
CAMPAIGN_ROOT = "/home/nick/w8-final-pascal-20260831"
HEARTBEAT_PATH = "/home/nick/w8-final-pascal-20260831.heartbeat.json"
STDOUT_LOG_PATH = "/home/nick/w8-final-pascal-20260831.stdout.log"
PASCAL_LOCK_SHA256 = "d3561c8e930797d328cf45df1bdc5085842833665f9b5c9e5617d4c891e31a82"
W7_TERMINAL_ID = "w7completion-fcd91d565ec3c98e1aff6c69a71b86af398971e7f8e898efa0499dc6e5c3dc1f"
W7_TERMINAL_PATH = "results/learned/w7/w7_completion.json"
W7_TERMINAL_SHA256 = "99f5e9c148a511de00fa19478fc47796f4478b606b6c53547e61afc25ae6d38b"
G4_ID = "w7g4adjudication-2136277dbb5e4d3f8a467c6e4137e959e5b1ffc947777c70b487aa3e884e3ec0"
G4_PATH = "results/learned/w7/w7_g4_result.json"
G4_SHA256 = "06f67ce3bcf6c3d2d8facf3bc014c79676b08b7575390f62f6070d1d8b757e3f"
G4_ADJUDICATOR_BLOB = "f1071971bce8dc6a48ddf504e5743e3faea5edfa"
B2R_ID = "w7b2rcompletion-172842c61df0231efd451d3d66b7857b5a67e79af887ff2d2bd8bcd9c801bee3"
B2R_PATH = "results/learned/w7/w7_b2_completion.json"
B2R_SHA256 = "569b53bc41852d20936d11c3f0df3f5089ab783e2e6dcc534a5f8e69a7087395"
W7_SOURCE_COMMIT = "cc704fcacec706719bc2791ae14a6c9d71dd4032"
W7_SOURCE_MANIFEST_ID = "w7b1source-ef005dc427ea83a2ff38904362c6a85612ff133a253e880fc671f2654a4aeb3f"
W7_SOURCE_MANIFEST_SHA256 = "163d83e25e2dbeb2d0dfd77610634fc2c975a218a4d3a6f1cb189fe24352e392"
W7_PASCAL_PROFILE_PATH = "results/learned/w7/w7_pascal_profile.json"
W7_PASCAL_PROFILE_ID = "w7profile-c2e70848dc6857fe4df3868c90af1ccff4d6e0c7d267cbad8b9ad49b228e5d69"
W7_PASCAL_PROFILE_SHA256 = "938e3ad8420a8e543a5f4576aa7d44ef78cd50c71a8c2540910a552672bd6bf0"
SOURCE_MANIFEST_REPO_PATH = "results/learned/w8/w8_source_manifest.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def _require_file(path: str, expected: str) -> None:
    candidate = REPO / path
    if not candidate.is_file() or candidate.is_symlink():
        raise ValueError(f"upstream authority path is missing or unsafe: {path}")
    actual = _sha(candidate)
    if actual != expected:
        raise ValueError(f"upstream authority bytes differ at {path}: {actual} != {expected}")


def _cells() -> list[dict[str, Any]]:
    return [cell.to_dict() for cell in run_cells()]


def _config_bindings() -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for cell in run_cells():
        config = load_w8_config(cell.ratio, cell.train_seed, cell.channel_seed)
        values.append(
            {
                "run_index": cell.run_index,
                "config_hash": run_config_hash(config),
                "protocol_config_hash": protocol_config_hash(config),
            }
        )
    return values


def _am94_predecessor_config_bindings(*, role: str = W8_CORE_ROLE) -> list[dict[str, Any]]:
    """Reconstruct the exact pre-AM-94 hashes for frozen W8 authority."""

    load_am94_spec_compatibility(REPO)
    names = []
    for path in AM94_ALLOWED_PARAMETER_PATHS:
        prefix = "evaluation."
        if not path.startswith(prefix) or "." in path[len(prefix):]:
            raise ValueError("AM-94 compatibility contains a non-evaluation leaf")
        names.append(path[len(prefix):])
    values: list[dict[str, Any]] = []
    for cell in run_cells():
        config = load_w8_config(
            cell.ratio, cell.train_seed, cell.channel_seed, role=role
        )
        historical = config.to_dict()
        evaluation = historical["parameters"]["evaluation"]
        for name in names:
            if name not in evaluation:
                raise ValueError(f"AM-94 parameter is absent from W8 config: {name}")
            del evaluation[name]
        values.append(
            {
                "run_index": cell.run_index,
                "config_hash": run_config_canonical_sha256(
                    {
                        "fingerprint_schema_version": historical["fingerprint_schema_version"],
                        "resolved": historical["resolved"],
                        "parameters": historical["parameters"],
                    }
                ),
                "protocol_config_hash": run_config_canonical_sha256(
                    {"protocol": protocol_descriptor(), "config": historical}
                ),
            }
        )
    return values


def build_authorization(
    *,
    source_commit: str,
    source_manifest_path: Path,
    issued_at_utc: str | None = None,
) -> dict[str, Any]:
    if _git("rev-parse", "--verify", f"{source_commit}^{{commit}}") != source_commit:
        raise ValueError("W8 source commit is not an exact commit object")
    source_manifest = verify_manifest(source_manifest_path, expected_source_commit=source_commit)
    _require_file(W7_TERMINAL_PATH, W7_TERMINAL_SHA256)
    _require_file(G4_PATH, G4_SHA256)
    _require_file(B2R_PATH, B2R_SHA256)
    _require_file(W7_PASCAL_PROFILE_PATH, W7_PASCAL_PROFILE_SHA256)
    if get("learned_system.lambda_core") != W8_EXPECTED_LAMBDA or get("learned_system.lambda_status") != "selected_at_G-4":
        raise ValueError("current G-4 lambda state is not exactly selected_at_G-4 = 3.0")
    if checkpoint_selection_snr_db() != get("channel.train_snr_db_fixed"):
        raise ValueError("W8 checkpoint selection SNR no longer resolves to train_snr_db_fixed")
    protocol = protocol_descriptor()
    cells = _cells()
    body: dict[str, Any] = {
        "schema_version": 1,
        "artifact_role": AUTHORIZATION_ROLE,
        "status": "FROZEN_PRE_EXECUTION",
        "authorization_scope": "W8_SIX_CORE_RUNS_ONLY",
        "authorization_basis": "W8-A final multi-seed pre-execution freeze; detached W8-B launch requires a separate owner authorization",
        "issued_at_utc": issued_at_utc or datetime.now(timezone.utc).isoformat(),
        "scientific_execution_authorized": "SIX_CORE_RUNS_ONLY",
        "w8_b_launch_authorization_required": True,
        "upstream": {
            "w7_terminal": {
                "completion_id": W7_TERMINAL_ID,
                "path": W7_TERMINAL_PATH,
                "file_sha256": W7_TERMINAL_SHA256,
                "status": "W7_GREEN_CLOSED",
            },
            "g4_adjudication": {
                "adjudication_id": G4_ID,
                "path": G4_PATH,
                "file_sha256": G4_SHA256,
                "adjudicator_git_blob_sha1": G4_ADJUDICATOR_BLOB,
                "selected_lambda": W8_EXPECTED_LAMBDA,
                "lambda_status": "selected_at_G-4",
            },
            "b2r_completion": {
                "completion_id": B2R_ID,
                "path": B2R_PATH,
                "file_sha256": B2R_SHA256,
            },
            "w7_scientific_lineage": {
                "source_commit": W7_SOURCE_COMMIT,
                "source_manifest_id": W7_SOURCE_MANIFEST_ID,
                "source_manifest_sha256": W7_SOURCE_MANIFEST_SHA256,
                "g4_adjudicator_blob_sha1": G4_ADJUDICATOR_BLOB,
                "pilot_initialization_forbidden": True,
            },
        },
        "scientific_source": {
            "source_commit": source_commit,
            "source_manifest_id": source_manifest["manifest_id"],
            "source_manifest_path": str(source_manifest_path.relative_to(REPO)),
            "source_manifest_file_sha256": _sha(source_manifest_path),
            "source_manifest_source_commit": source_manifest["source_commit"],
            "source_manifest_entry_count": source_manifest["entry_count"],
        },
        "campaign": {
            "campaign_id": CAMPAIGN_ID,
            "campaign_root": CAMPAIGN_ROOT,
            "heartbeat_path": HEARTBEAT_PATH,
            "stdout_log_path": STDOUT_LOG_PATH,
            "order_rule": "seed_major_then_ratio_minor",
            "run_count": len(cells),
            "run_cells": cells,
            "unique_ratios": list(W8_EXPECTED_RATIOS),
            "k_by_ratio": {ratio: protocol["run_matrix"][index]["k"] for index, ratio in enumerate(W8_EXPECTED_RATIOS)},
            "train_seeds": list(W8_TRAIN_SEEDS),
            "channel_seeds": list(W8_CHANNEL_SEEDS),
            "seed_pairing": "zipped_not_cross_product",
            "config_bindings": _config_bindings(),
        },
        "training": {
            "dataset": W8_DATASET,
            "lambda": W8_EXPECTED_LAMBDA,
            "lambda_parameter": "params.learned_system.lambda_core",
            "lambda_status": "selected_at_G-4",
            "train_snr_parameter": "params.channel.train_snr_db_fixed",
            "train_snr_db": get("channel.train_snr_db_fixed"),
            "epochs_per_run": W8_EPOCHS,
            "optimizer": protocol["optimizer_recipe"],
            "fresh_initialization": protocol["initialization"],
            "w7_checkpoint_transfer_forbidden": True,
            "prior_w8_state_transfer_forbidden": True,
        },
        "checkpoint_selection": {
            "split": "validation",
            "metric": "validation_top1_accuracy",
            "mode": "max",
            "tie_break": "earliest_epoch",
            "snr_parameter": W8_CHECKPOINT_SELECTION_SNR_PARAMETER,
            "snr_resolution": "params.channel.train_snr_db_fixed",
            "snr_db": checkpoint_selection_snr_db(),
            "channel_seed_rule": W8_CHECKPOINT_SELECTION_CHANNEL_SEED_RULE,
            "validation_denominator": W8_VALIDATION_SAMPLE_COUNT,
            "full_validation_every_completed_epoch": True,
            "fixed_noise_across_epochs": True,
            "cross_seed_selection": False,
            "forbidden_inputs": ["psnr", "papr", "reconstruction_loss"],
        },
        "profile": {
            "execution_profile_id": W8_PROFILE_ID,
            "scientific_writer_host": "confessor",
            "execution_image_family": W8_EXECUTION_IMAGE_FAMILY,
            "gpu_name": W8_SELECTED_GPU_NAME,
            "gpu_uuid": W8_SELECTED_GPU_UUID,
            "device": "cuda:0",
            "requirements_lock": "requirements-pascal.lock",
            "requirements_lock_sha256": PASCAL_LOCK_SHA256,
            "physical_batch_size": W8_PHYSICAL_BATCH_SIZE,
            "accumulation_factor": W8_ACCUMULATION_FACTOR,
            "effective_batch_size": W8_EFFECTIVE_BATCH_SIZE,
            "validation_batch_size": W8_VALIDATION_BATCH_SIZE,
            "train_samples": W8_TRAIN_SAMPLE_COUNT,
            "drop_last": False,
            "batch_binding_evidence": {
                "path": W7_PASCAL_PROFILE_PATH,
                "profile_id": W7_PASCAL_PROFILE_ID,
                "file_sha256": W7_PASCAL_PROFILE_SHA256,
                "status": "PASSED",
                "scientific_status": "NON_SCIENTIFIC_ZERO_G4_COVERAGE",
                "physical_batch_size": W8_PHYSICAL_BATCH_SIZE,
                "accumulation_factor": W8_ACCUMULATION_FACTOR,
                "effective_batch_size": W8_EFFECTIVE_BATCH_SIZE,
                "validation_batch_size": W8_VALIDATION_BATCH_SIZE,
            },
        },
        "resume_and_custody": {
            "epoch_record_role": "W8_FINAL_TRAINING_EPOCH_RECORD",
            "epoch_checkpoint_role": "W8_FINAL_TRAINING_CHECKPOINT",
            "epoch_summary_role": "W8_VALIDATION_EPOCH_SUMMARY",
            "selected_checkpoint_role": "W8_SELECTED_CHECKPOINT",
            "resume_rule": "latest_authenticated_completed_epoch_only",
            "corrupt_latest": "HOLD_NO_OLDER_FALLBACK",
            "incomplete_epoch": "REPLAY_FROM_LATEST_AUTHENTICATED_COMPLETED_EPOCH",
            "checkpoint_publication": "atomic_immutable_payload_sidecar_latest_pointer",
            "campaign_lock": "/tmp/capstone-w8-final-global.lock",
            "sequential_runs": True,
            "single_writer": True,
        },
        "boundary": {
            "w8_scientific_execution": "SIX_CORE_RUNS_ONLY",
            "er2_randomized_training": "NOT_AUTHORIZED",
            "papr_constrained_training": "NOT_AUTHORIZED_BOUND_REQUIRES_DOWNSTREAM_PROTOCOL_ITEM",
            "er9_training": "NOT_AUTHORIZED",
            "g10": "NOT_AUTHORIZED",
            "test": "SEALED",
            "test_model_facing_access": 0,
            "learned_test_inference": 0,
            "g10_adjudications": 0,
        },
        "pre_execution_zero_counters": {
            "w8_final_training_runs": 0,
            "w8_scientific_optimizer_steps": 0,
            "w8_completed_runs": 0,
            "w8_scientific_checkpoints": 0,
            "g10_adjudications": 0,
            "er2_randomized_training": 0,
            "papr_constrained_training": 0,
            "er9_training": 0,
            "test_model_facing_access": 0,
            "learned_test_inference": 0,
        },
        "protocol_hash": canonical_sha256(protocol),
        "source_contains_no_w8_results": True,
        "campaign_root_created_at_freeze": False,
    }
    body["authorization_id"] = AUTHORIZATION_PREFIX + canonical_sha256(body)
    return body


def _expect_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} schema differs")


def _expected_batch_binding_evidence() -> dict[str, Any]:
    return {
        "path": W7_PASCAL_PROFILE_PATH,
        "profile_id": W7_PASCAL_PROFILE_ID,
        "file_sha256": W7_PASCAL_PROFILE_SHA256,
        "status": "PASSED",
        "scientific_status": "NON_SCIENTIFIC_ZERO_G4_COVERAGE",
        "physical_batch_size": W8_PHYSICAL_BATCH_SIZE,
        "accumulation_factor": W8_ACCUMULATION_FACTOR,
        "effective_batch_size": W8_EFFECTIVE_BATCH_SIZE,
        "validation_batch_size": W8_VALIDATION_BATCH_SIZE,
    }


def verify_authorization(
    path: Path,
    *,
    expected_source_commit: str | None = None,
    expected_source_manifest_path: Path | None = None,
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"W8 authorization path is missing or unsafe: {path}")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("W8 authorization is not a JSON object")
    _expect_keys(
        value,
        {
            "schema_version", "artifact_role", "status", "authorization_scope",
            "authorization_basis", "issued_at_utc", "scientific_execution_authorized",
            "w8_b_launch_authorization_required", "upstream", "scientific_source",
            "campaign", "training", "checkpoint_selection", "profile",
            "resume_and_custody", "boundary", "pre_execution_zero_counters",
            "protocol_hash", "source_contains_no_w8_results",
            "campaign_root_created_at_freeze", "authorization_id",
        },
        "W8 authorization",
    )
    body = dict(value)
    identifier = body.pop("authorization_id")
    if identifier != AUTHORIZATION_PREFIX + canonical_sha256(body):
        raise ValueError("W8 authorization ID does not authenticate its body")
    if value["schema_version"] != 1 or value["artifact_role"] != AUTHORIZATION_ROLE or value["status"] != "FROZEN_PRE_EXECUTION":
        raise ValueError("W8 authorization role/status differs")
    if value["authorization_scope"] != "W8_SIX_CORE_RUNS_ONLY" or value["scientific_execution_authorized"] != "SIX_CORE_RUNS_ONLY" or value["w8_b_launch_authorization_required"] is not True:
        raise ValueError("W8 authorization scope differs")
    if value["authorization_basis"] != "W8-A final multi-seed pre-execution freeze; detached W8-B launch requires a separate owner authorization" or not isinstance(value["issued_at_utc"], str) or not value["issued_at_utc"]:
        raise ValueError("W8 authorization basis/timestamp differs")

    expected_upstream = {
        "w7_terminal": {
            "completion_id": W7_TERMINAL_ID,
            "path": W7_TERMINAL_PATH,
            "file_sha256": W7_TERMINAL_SHA256,
            "status": "W7_GREEN_CLOSED",
        },
        "g4_adjudication": {
            "adjudication_id": G4_ID,
            "path": G4_PATH,
            "file_sha256": G4_SHA256,
            "adjudicator_git_blob_sha1": G4_ADJUDICATOR_BLOB,
            "selected_lambda": W8_EXPECTED_LAMBDA,
            "lambda_status": "selected_at_G-4",
        },
        "b2r_completion": {
            "completion_id": B2R_ID,
            "path": B2R_PATH,
            "file_sha256": B2R_SHA256,
        },
        "w7_scientific_lineage": {
            "source_commit": W7_SOURCE_COMMIT,
            "source_manifest_id": W7_SOURCE_MANIFEST_ID,
            "source_manifest_sha256": W7_SOURCE_MANIFEST_SHA256,
            "g4_adjudicator_blob_sha1": G4_ADJUDICATOR_BLOB,
            "pilot_initialization_forbidden": True,
        },
    }
    if value["upstream"] != expected_upstream:
        raise ValueError("W8 upstream authority binding differs")
    _require_file(W7_TERMINAL_PATH, W7_TERMINAL_SHA256)
    _require_file(G4_PATH, G4_SHA256)
    _require_file(B2R_PATH, B2R_SHA256)
    _require_file(W7_PASCAL_PROFILE_PATH, W7_PASCAL_PROFILE_SHA256)

    source = value["scientific_source"]
    _expect_keys(
        source,
        {
            "source_commit", "source_manifest_id", "source_manifest_path",
            "source_manifest_file_sha256", "source_manifest_source_commit",
            "source_manifest_entry_count",
        },
        "W8 scientific source",
    )
    source_commit = source["source_commit"]
    if not isinstance(source_commit, str) or len(source_commit) != 40 or any(character not in "0123456789abcdef" for character in source_commit):  # literal-ok: Git SHA-1 width
        raise ValueError("W8 authorization source commit is invalid")
    if expected_source_commit is not None and source_commit != expected_source_commit:
        raise ValueError("W8 authorization source commit differs")
    recorded_manifest_path = Path(str(source["source_manifest_path"]))
    if recorded_manifest_path.is_absolute() or ".." in recorded_manifest_path.parts or str(recorded_manifest_path) != SOURCE_MANIFEST_REPO_PATH:
        raise ValueError("W8 authorization source manifest path is unsafe")
    manifest_path = expected_source_manifest_path if expected_source_manifest_path is not None else REPO / recorded_manifest_path
    manifest = verify_manifest(manifest_path, expected_source_commit=source_commit)
    expected_source = {
        "source_commit": source_commit,
        "source_manifest_id": manifest["manifest_id"],
        "source_manifest_path": SOURCE_MANIFEST_REPO_PATH,
        "source_manifest_file_sha256": _sha(manifest_path),
        "source_manifest_source_commit": source_commit,
        "source_manifest_entry_count": len(CRITICAL_SOURCES),
    }
    if source != expected_source:
        raise ValueError("W8 authorization source manifest binding differs")

    protocol = protocol_descriptor()
    if value["protocol_hash"] != canonical_sha256(protocol):
        raise ValueError("W8 authorization protocol hash differs")
    expected_campaign = {
        "campaign_id": CAMPAIGN_ID,
        "campaign_root": CAMPAIGN_ROOT,
        "heartbeat_path": HEARTBEAT_PATH,
        "stdout_log_path": STDOUT_LOG_PATH,
        "order_rule": "seed_major_then_ratio_minor",
        "run_count": len(run_cells()),
        "run_cells": _cells(),
        "unique_ratios": list(W8_EXPECTED_RATIOS),
        "k_by_ratio": dict(W8_EXPECTED_K),
        "train_seeds": list(W8_TRAIN_SEEDS),
        "channel_seeds": list(W8_CHANNEL_SEEDS),
        "seed_pairing": "zipped_not_cross_product",
        "config_bindings": _config_bindings(),
    }
    if value["campaign"] != expected_campaign:
        try:
            expected_campaign["config_bindings"] = _am94_predecessor_config_bindings()
        except Exception as exc:
            raise ValueError(f"W8 authorization AM-94 compatibility differs: {exc}") from None
        if value["campaign"] != expected_campaign:
            raise ValueError("W8 authorization campaign binding differs")

    expected_training = {
        "dataset": W8_DATASET,
        "lambda": W8_EXPECTED_LAMBDA,
        "lambda_parameter": "params.learned_system.lambda_core",
        "lambda_status": "selected_at_G-4",
        "train_snr_parameter": "params.channel.train_snr_db_fixed",
        "train_snr_db": get("channel.train_snr_db_fixed"),
        "epochs_per_run": W8_EPOCHS,
        "optimizer": protocol["optimizer_recipe"],
        "fresh_initialization": protocol["initialization"],
        "w7_checkpoint_transfer_forbidden": True,
        "prior_w8_state_transfer_forbidden": True,
    }
    if value["training"] != expected_training:
        raise ValueError("W8 authorization training recipe/boundary differs")

    expected_selection = {
        "split": "validation",
        "metric": "validation_top1_accuracy",
        "mode": "max",
        "tie_break": "earliest_epoch",
        "snr_parameter": W8_CHECKPOINT_SELECTION_SNR_PARAMETER,
        "snr_resolution": "params.channel.train_snr_db_fixed",
        "snr_db": checkpoint_selection_snr_db(),
        "channel_seed_rule": W8_CHECKPOINT_SELECTION_CHANNEL_SEED_RULE,
        "validation_denominator": W8_VALIDATION_SAMPLE_COUNT,
        "full_validation_every_completed_epoch": True,
        "fixed_noise_across_epochs": True,
        "cross_seed_selection": False,
        "forbidden_inputs": ["psnr", "papr", "reconstruction_loss"],
    }
    if value["checkpoint_selection"] != expected_selection:
        raise ValueError("W8 authorization checkpoint-selection policy differs")

    expected_profile = {
        "execution_profile_id": W8_PROFILE_ID,
        "scientific_writer_host": "confessor",
        "execution_image_family": W8_EXECUTION_IMAGE_FAMILY,
        "gpu_name": W8_SELECTED_GPU_NAME,
        "gpu_uuid": W8_SELECTED_GPU_UUID,
        "device": "cuda:0",
        "requirements_lock": "requirements-pascal.lock",
        "requirements_lock_sha256": PASCAL_LOCK_SHA256,
        "physical_batch_size": W8_PHYSICAL_BATCH_SIZE,
        "accumulation_factor": W8_ACCUMULATION_FACTOR,
        "effective_batch_size": W8_EFFECTIVE_BATCH_SIZE,
        "validation_batch_size": W8_VALIDATION_BATCH_SIZE,
        "train_samples": W8_TRAIN_SAMPLE_COUNT,
        "drop_last": False,
        "batch_binding_evidence": _expected_batch_binding_evidence(),
    }
    if value["profile"] != expected_profile:
        raise ValueError("W8 authorization execution profile/batch binding differs")

    expected_resume = {
        "epoch_record_role": "W8_FINAL_TRAINING_EPOCH_RECORD",
        "epoch_checkpoint_role": "W8_FINAL_TRAINING_CHECKPOINT",
        "epoch_summary_role": "W8_VALIDATION_EPOCH_SUMMARY",
        "selected_checkpoint_role": "W8_SELECTED_CHECKPOINT",
        "resume_rule": "latest_authenticated_completed_epoch_only",
        "corrupt_latest": "HOLD_NO_OLDER_FALLBACK",
        "incomplete_epoch": "REPLAY_FROM_LATEST_AUTHENTICATED_COMPLETED_EPOCH",
        "checkpoint_publication": "atomic_immutable_payload_sidecar_latest_pointer",
        "campaign_lock": "/tmp/capstone-w8-final-global.lock",
        "sequential_runs": True,
        "single_writer": True,
    }
    if value["resume_and_custody"] != expected_resume:
        raise ValueError("W8 authorization resume/custody policy differs")
    expected_boundary = {
        "w8_scientific_execution": "SIX_CORE_RUNS_ONLY",
        "er2_randomized_training": "NOT_AUTHORIZED",
        "papr_constrained_training": "NOT_AUTHORIZED_BOUND_REQUIRES_DOWNSTREAM_PROTOCOL_ITEM",
        "er9_training": "NOT_AUTHORIZED",
        "g10": "NOT_AUTHORIZED",
        "test": "SEALED",
        "test_model_facing_access": 0,
        "learned_test_inference": 0,
        "g10_adjudications": 0,
    }
    if value["boundary"] != expected_boundary:
        raise ValueError("W8 authorization forbidden boundary differs")
    expected_zero = {
        "w8_final_training_runs": 0,
        "w8_scientific_optimizer_steps": 0,
        "w8_completed_runs": 0,
        "w8_scientific_checkpoints": 0,
        "g10_adjudications": 0,
        "er2_randomized_training": 0,
        "papr_constrained_training": 0,
        "er9_training": 0,
        "test_model_facing_access": 0,
        "learned_test_inference": 0,
    }
    if value["pre_execution_zero_counters"] != expected_zero:
        raise ValueError("W8 authorization pre-execution counters differ")
    if value["source_contains_no_w8_results"] is not True or value["campaign_root_created_at_freeze"] is not False:
        raise ValueError("W8 authorization source/root boundary differs")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_authorization(value: dict[str, Any], path: Path) -> None:
    """Publish an immutable authorization without a replaceable final name."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable W8 authorization already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            raise FileExistsError(f"immutable W8 authorization already exists: {path}") from None
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--issued-at-utc")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    source_manifest = args.source_manifest if args.source_manifest.is_absolute() else REPO / args.source_manifest
    output = args.output if args.output.is_absolute() else REPO / args.output
    if args.check:
        value = verify_authorization(output, expected_source_commit=args.source_commit, expected_source_manifest_path=source_manifest)
        print(f"W8 execution authorization PASS: {value['authorization_id']}")
    else:
        value = build_authorization(source_commit=args.source_commit, source_manifest_path=source_manifest, issued_at_utc=args.issued_at_utc)
        write_authorization(value, output)
        print(f"W8 execution authorization written: {value['authorization_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
