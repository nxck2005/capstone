#!/usr/bin/env python3
"""Fail-closed verifier for the post-campaign W7-B2R reconciliation.

The verifier authenticates compact worker/portable evidence only.  It never
runs a model, opens a checkpoint as a model, performs a validation pass, runs
G-4 adjudication, or changes any scientific state.  ``--worker-metadata-root``
adds an exact read-only comparison with the copied worker JSON metadata;
``--worker-root`` additionally hashes locally available checkpoint bytes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from config.params import get  # noqa: E402
from config.run_config import config_hash as run_config_hash  # noqa: E402
from config.w7_execution import verify_frozen_gpu_binding  # noqa: E402
from data.djscc_validation import validation_noise_id  # noqa: E402
from training.deterministic_core import canonical_bytes, canonical_sha256  # noqa: E402
from training.w7_g4 import W7_CHECKPOINT_ROLE, W7_PROTECTED_COUNTERS, W7_RNG_STATE_POLICY  # noqa: E402
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
    W7_SELECTED_GPU_UUID,
    W7_TRAIN_SEED,
    W7_TRAINING_SNR_DB,
    W7_VALIDATION_BATCH_SIZE,
    W7_VALIDATION_NOISE_POLICY,
    eligibility_for_role,
    load_w7_config,
    protocol_config_hash,
)
from reconcile_w7_b2r import (  # noqa: E402
    AUTHORIZATION_ID,
    AUTHORIZATION_SHA256,
    CAMPAIGN_COMPLETION_ID,
    CAMPAIGN_COMPLETION_SHA256,
    CAMPAIGN_ID,
    CAMPAIGN_MANIFEST_ID,
    CAMPAIGN_MANIFEST_SHA256,
    CANDIDATES,
    DATASET_VERSION,
    EXPECTED_EPOCHS,
    FINAL_PARTIAL_BATCH,
    GPU_NAME,
    MICROBATCHES,
    NOISE_ID_DIGEST,
    NOISE_POLICY_HASH,
    PROFILE_FREEZE_ID,
    PROFILE_FREEZE_SHA256,
    PROTOCOL_HASH,
    SOURCE_COMMIT,
    SOURCE_MANIFEST_ID,
    SOURCE_MANIFEST_SHA256,
    SPLIT_MANIFEST_HASH,
    TARGET_BATCH,
    TRAIN_COUNT,
    VAL_COUNT,
    VAL_STABLE_ID_DIGEST,
    expected_homogeneity,
    expected_lr,
    expected_order,
    expected_validation_noise,
    file_sha256,
    line_digest,
    load_manifest_ids,
    training_noise_digest,
)

SCHEMA_VERSION = 1  # literal-ok: B2R evidence schema version
INDEX_ROLE = "W7_B2R_CANDIDATE_INDEX"
CUSTODY_ROLE = "W7_B2R_CHECKPOINT_CUSTODY_MANIFEST"
NOISE_ROLE = "W7_B2R_COMMON_VALIDATION_NOISE_AUDIT"
RECONCILIATION_ROLE = "W7_B2R_RECONCILIATION"
COMPLETION_ROLE = "W7_B2R_RECONCILIATION_COMPLETE_NOT_YET_ADJUDICATED"
CANDIDATE_ROLE = "W7_G4_LAMBDA_CANDIDATE_COMPLETION"
SIDECAR_ROLE = "W7_G4_SCIENTIFIC_PILOT_CHECKPOINT_SIDECAR"
EPOCH_ROLE = "W7_G4_EPOCH_RECORD"
SUMMARY_ROLE = "W7_VALIDATION_EPOCH_SUMMARY"
SELECTED_RESULT_ROLE = "W7_SELECTED_CHECKPOINT_VALIDATION_RESULT"
AUTH_STATUS = "PASSED"
PROFILE_LOCK_SHA256 = "d3561c8e930797d328cf45df1bdc5085842833665f9b5c9e5617d4c891e31a82"
EVALUATION_CONFIG_HASH = "f1c25277250ec10ec766aac99539d46bc988cd87c05e4b6b7e1b725a4fee2d65"
W7_A_COMPLETION_ID = "w7acompletion-e623063c65348e23833fcec31588ef04ac43f793eeb2be0272af158071b7ba17"
W7_TEST_HARDENING_ID = "w7testhardening-a7011b78b327184bd08bd58c6e85cd04253bc583e617a376f74392f467effab3"
W7_TEST_HARDENING_SHA256 = "d54e5aa7507f7d9e976fdac029ead08c346acab53c17ca1254773df16dad2bf2"
W5_COMPLETION_ID = "w5repaircompletion-8b2fa9178cc0dec943d32f1eebec85f50d152075d29188a32e42f40d6d63fb89"
W6_COMPLETION_ID = "w6completion-f992e38e553dce4075406ef8f08df0d42feb2a141a3b00b0ae29a0490e834515"
PROFILE_VERIFIER_ID = "w7profile-c2e70848dc6857fe4df3868c90af1ccff4d6e0c7d267cbad8b9ad49b228e5d69"
INITIAL_CARRIER_COMMIT = "a3665b854dd1e9065a8082a66680a69ce29a10c1"

INDEX_KEYS = {
    "schema_version", "artifact_role", "campaign_id", "campaign_manifest", "campaign_completion",
    "authorization", "source", "profile", "protocol", "homogeneity", "candidate_order",
    "candidates", "factual_measurements", "worker_scan", "checkpoint_audit_capture", "index_id",
}
CUSTODY_KEYS = {
    "schema_version", "artifact_role", "status", "campaign_id", "candidate_index_id", "worker_hostname",
    "worker_campaign_root", "checkpoint_policy", "expected_checkpoint_count", "observed_checkpoint_count",
    "selected_checkpoint_bytes_remain_on_worker", "candidates", "custody_id",
}
NOISE_KEYS = {
    "schema_version", "artifact_role", "status", "campaign_id", "candidate_index_id", "evaluation_role",
    "snr_db", "channel_seed", "ratio", "dataset", "validation_denominator", "validation_order",
    "noise_policy", "noise_policy_hash", "stable_id_digest", "noise_id_digest", "identity_fields",
    "paired_samples", "epoch_summary_audit", "selected_candidate_audit", "pairwise_comparisons",
    "all_selected_sample_mismatch_count", "test_model_facing_access", "audit_id",
}
RECONCILIATION_KEYS = {
    "schema_version", "artifact_role", "status", "campaign_status", "campaign_id", "scientific_source_commit",
    "reconciliation_tooling_base_commit", "campaign_manifest_id", "campaign_completion_id", "candidate_index_id",
    "custody_id", "common_noise_audit_id", "upstream_reauthentication", "homogeneity", "factual_candidate_measurements",
    "worker_custody", "protected_boundary", "operational_closeout", "no_model_facing_recomputation",
    "decision_boundary", "reconciliation_id",
}
COMPLETION_KEYS = {
    "schema_version", "artifact_role", "status", "campaign_id", "candidate_count", "complete_candidate_count",
    "completed_epoch_cycles", "candidate_index_id", "custody_id", "common_noise_audit_id", "reconciliation_id",
    "worker_campaign_completion_id", "worker_campaign_completion_file_sha256", "candidate_references",
    "g4_adjudication_run", "lambda_decision", "lambda_core_updated", "lambda_status", "w8_final_training_runs",
    "w8_state", "test_model_facing_access", "learned_test_inference", "test_state", "source_commit",
    "execution_profile_id", "gpu_uuid", "no_scientific_execution_performed_by_reconciliation", "completion_id",
}
INDEX_CANDIDATE_KEYS = {
    "campaign_order", "lambda", "candidate_root", "candidate_completion", "profile_binding", "config_hash",
    "protocol_config_hash", "homogeneity", "epochs", "checkpoints", "validation_summaries",
    "checkpoint_payload_audit", "latest", "selected", "training_totals", "worker_candidate_id",
}
INDEX_EPOCH_KEYS = {
    "epoch", "next_epoch", "epoch_record_path", "epoch_record_sha256", "record_id", "samples",
    "expected_samples", "stable_id_count", "stable_id_order_sha256", "stable_id_set_sha256",
    "training_noise_id_count", "training_noise_id_sha256", "microbatches", "optimizer_steps",
    "grad_scaler_skips", "global_optimizer_step", "lr", "finite_loss", "lineage_sha256", "gradient_audit",
}
INDEX_CHECKPOINT_KEYS = {
    "epoch", "path", "checkpoint_id", "checkpoint_bytes", "sidecar_path", "sidecar_sha256", "sidecar",
    "epoch_record_path", "epoch_record_id", "epoch_record_sha256", "predecessor_checkpoint_id",
    "global_optimizer_step", "payload_audit",
}
PAYLOAD_ENTRY_KEYS = {
    "epoch", "checkpoint_id", "lineage_sha256", "execution_profile_sha256", "eligibility_sha256",
    "rng_policy_sha256", "protected_counters_sha256", "epoch_manifest", "predecessor_checkpoint_id",
    "model_state_key_count", "optimizer_state_param_count", "optimizer_param_group_count", "scheduler_state",
    "scaler_state_present",
}
INDEX_SELECTED_KEYS = {
    "selected_epoch", "selected_checkpoint_id", "selected_checkpoint", "selected_checkpoint_result",
    "independent_selection", "selected_validation", "derived_n_correct", "derived_n_total",
    "derived_top1_accuracy", "derived_mean_psnr_db", "derived_mean_papr", "psnr_definition", "papr_definition",
}
VALIDATION_REF_KEYS = {"path", "file_sha256", "summary"}
CANDIDATE_REF_KEYS = {"path", "file_sha256", "value"}
SELECTED_RESULT_REF_KEYS = {"path", "file_sha256", "value"}
CUSTODY_CANDIDATE_KEYS = {
    "campaign_order", "lambda", "candidate_root", "candidate_completion_path", "candidate_completion_id",
    "candidate_completion_file_sha256", "selected_checkpoint_path", "selected_checkpoint_id",
    "selected_checkpoint_file_sha256", "selected_checkpoint_sidecar_path", "selected_checkpoint_sidecar_sha256",
    "selected_checkpoint_result_path", "selected_checkpoint_result_file_sha256", "selected_checkpoint_result_digest",
    "validation_history_digest", "checkpoint_chain_digest", "candidate_result_digest", "checkpoint_count", "checkpoints",
}
CUSTODY_CHECKPOINT_KEYS = {
    "epoch", "path", "checkpoint_id", "checkpoint_bytes", "sidecar_path", "sidecar_sha256", "epoch_record_path",
    "epoch_record_id", "epoch_record_sha256", "predecessor_checkpoint_id", "global_optimizer_step",
}
SIDECAR_KEYS = {
    "schema_version", "artifact_role", "eligibility", "checkpoint_path", "checkpoint_id", "checkpoint_bytes",
    "completed_epoch", "next_epoch", "global_optimizer_step", "accumulation_position", "config_hash",
    "protocol_config_hash", "source_commit", "source_manifest_id", "source_manifest_sha256", "execution_image",
    "execution_profile_id", "gpu_uuid", "dataset", "ratio", "k", "lambda", "train_seed", "channel_seed",
    "train_snr_db", "predecessor_checkpoint_id", "epoch_record_path", "epoch_record_id", "epoch_record_sha256",
    "checkpoint_write_seconds",
}
SUMMARY_KEYS = {
    "schema_version", "artifact_role", "epoch", "checkpoint_id", "n_correct", "n_total", "top1_accuracy",
    "prediction_digest", "evaluation_config_hash", "noise_policy", "noise_policy_hash", "noise_id_digest",
    "row_digest", "summary_id",
}
EPOCH_GRADIENT_KEYS = {
    "all_named_present_gradients_finite", "all_optimizer_gradients_finite", "optimizer_parameter_count",
    "optimizer_gradient_count_min", "optimizer_gradient_count_max",
}
SELECTED_RESULT_KEYS = {
    "artifact_role", "calibration_rows", "calibration_validation", "checkpoint_epoch", "checkpoint_id",
    "protected_counters", "psnr_evaluation", "result_digest", "schema_version", "selection",
}
CALIBRATION_ROW_KEYS = {"stable_sample_id", "label", "prediction", "correct", "noise_id", "mse", "psnr_db", "papr_db"}
PSNR_EVALUATION_KEYS = {
    "checkpoint_id", "snr_db", "denominator", "data_range", "psnr_definition", "psnr_db", "papr_definition",
    "papr_db", "per_image_digest", "per_image",
}
PER_IMAGE_KEYS = {"stable_sample_id", "psnr_db", "papr_db"}
PROFILE_KEYS = {
    "schema_version", "authentication_status", "execution_profile_id", "gpu_uuid", "gpu_name",
    "gpu_compute_capability", "cuda_visible_devices", "device", "profile_environment", "lock_file_sha256",
    "git_commit", "config_hash", "binding_sha256",
}
PROFILE_ENVIRONMENT_KEYS = {
    "amp", "config_hash", "deterministic_backend", "driver_version", "execution_profile_id", "git_commit",
    "git_dirty", "gpu_compute_capability", "gpu_index", "gpu_name", "gpu_uuid", "gpu_vram_mib", "lock_file",
    "lock_file_sha256", "numpy_version", "nvidia_smi_index", "openjpeg_version", "python_version", "sionna_version",
    "torch_cuda_build", "torch_version", "torchvision_version",
}
FORBIDDEN_KEYS = {
    "rank", "winner", "best_lambda", "qualifying_lambda", "accuracy_ok", "primary_eligible",
    "relaxed_eligible", "recommended_lambda", "selected_lambda",
}


class VerificationError(RuntimeError):
    """A consequential B2R invariant failed."""


def fail(message: str) -> None:
    raise VerificationError(message)


def expect_keys(value: Any, keys: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        fail(f"{label} schema differs")


def expect_hex(value: Any, width: int, label: str) -> None:
    if not isinstance(value, str) or len(value) != width or any(char not in "0123456789abcdef" for char in value):
        fail(f"{label} is not a lowercase digest")


def load_artifact(path: Path, keys: set[str], role: str, id_key: str, prefix: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"cannot read {path}: {exc}")
    expect_keys(value, keys, str(path))
    if value["schema_version"] != SCHEMA_VERSION or value["artifact_role"] != role:
        fail(f"{path} has an unknown schema/role")
    identifier = value[id_key]
    if identifier != prefix + canonical_sha256({key: item for key, item in value.items() if key != id_key}):
        fail(f"{path} outer ID does not authenticate its body")
    return value


def scan_for_forbidden_keys(value: Any, path: str = "artifact") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in FORBIDDEN_KEYS:
                fail(f"forbidden decision field {path}.{key}")
            scan_for_forbidden_keys(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            scan_for_forbidden_keys(item, f"{path}[{index}]")


def verify_manifest_ref(ref: dict[str, Any]) -> dict[str, Any]:
    expect_keys(ref, {"manifest_id", "file_sha256", "worker_path", "value"}, "campaign_manifest reference")
    if ref["manifest_id"] != CAMPAIGN_MANIFEST_ID or ref["file_sha256"] != CAMPAIGN_MANIFEST_SHA256:
        fail("campaign manifest reference differs")
    value = ref["value"]
    expected_keys = {
        "accumulation_factor", "artifact_role", "authorization_id", "calibration_snr_db", "campaign_id", "channel_seed",
        "effective_batch_size", "execution_image_family", "execution_profile_id", "execution_source_commit",
        "g4_adjudication_run", "gpu_uuid", "lambda_core_updated", "lambda_grid", "lambda_order", "manifest_id",
        "physical_batch_size", "profile_freeze_id", "profile_freeze_sha256", "ratio", "schema_version",
        "scientific_execution_authorization", "source_commit", "source_manifest_id", "source_manifest_sha256", "status",
        "train_seed", "training_snr_db", "validation_batch_size", "w7_a_completion_id", "w7_test_hardening_completion_id",
        "w7_test_hardening_completion_sha256",
    }
    expect_keys(value, expected_keys, "worker campaign manifest")
    body = {key: item for key, item in value.items() if key != "manifest_id"}
    if value["manifest_id"] != "w7campaignmanifest-" + canonical_sha256(body):
        fail("worker campaign manifest digest differs")
    expected = {
        "schema_version": SCHEMA_VERSION, "artifact_role": "W7_G4_CAMPAIGN_MANIFEST", "status": "FROZEN_BEFORE_FIRST_CANDIDATE",
        "campaign_id": CAMPAIGN_ID, "authorization_id": AUTHORIZATION_ID, "w7_a_completion_id": W7_A_COMPLETION_ID,
        "w7_test_hardening_completion_id": W7_TEST_HARDENING_ID, "w7_test_hardening_completion_sha256": W7_TEST_HARDENING_SHA256,
        "source_commit": SOURCE_COMMIT, "execution_source_commit": SOURCE_COMMIT, "source_manifest_id": SOURCE_MANIFEST_ID,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256, "profile_freeze_id": PROFILE_FREEZE_ID, "profile_freeze_sha256": PROFILE_FREEZE_SHA256,
        "execution_image_family": W7_EXECUTION_IMAGE_FAMILY, "execution_profile_id": W7_PROFILE_ID, "gpu_uuid": W7_SELECTED_GPU_UUID,
        "lambda_grid": list(W7_LAMBDA_GRID), "lambda_order": "exact_configured_lambda_grid_order", "train_seed": W7_TRAIN_SEED,
        "channel_seed": W7_CHANNEL_SEED, "training_snr_db": W7_TRAINING_SNR_DB, "calibration_snr_db": W7_CALIBRATION_SNR_DB,
        "ratio": W7_RATIO, "physical_batch_size": W7_PHYSICAL_BATCH_SIZE, "accumulation_factor": 1, "effective_batch_size": TARGET_BATCH,
        "validation_batch_size": W7_VALIDATION_BATCH_SIZE, "scientific_execution_authorization": "PRESENT", "g4_adjudication_run": 0,
        "lambda_core_updated": False,
    }
    if any(value[key] != expected_value for key, expected_value in expected.items()):
        fail("worker campaign manifest frozen field differs")
    return value


def verify_worker_completion(ref: dict[str, Any]) -> dict[str, Any]:
    expect_keys(ref, {"completion_id", "file_sha256", "worker_path", "value"}, "campaign_completion reference")
    if ref["completion_id"] != CAMPAIGN_COMPLETION_ID or ref["file_sha256"] != CAMPAIGN_COMPLETION_SHA256:
        fail("campaign completion reference differs")
    value = ref["value"]
    expect_keys(value, {
        "artifact_role", "campaign_id", "candidate_lambdas", "candidate_paths", "candidates", "completion_id",
        "g4_adjudication_run", "lambda_core_updated", "schema_version", "scientific_execution_authorization", "status",
    }, "worker campaign completion")
    if value["completion_id"] != "w7campaign-" + canonical_sha256({key: item for key, item in value.items() if key != "completion_id"}):
        fail("worker campaign completion digest differs")
    if value["artifact_role"] != "W7_G4_CAMPAIGN_COMPLETE_NOT_ADJUDICATED" or value["status"] != "COMPLETE_NOT_ADJUDICATED":
        fail("worker campaign completion role/status differs")
    if value["campaign_id"] != CAMPAIGN_ID or value["candidate_lambdas"] != list(W7_LAMBDA_GRID):
        fail("worker campaign completion campaign/grid differs")
    expected_candidate_paths = [
        f"/home/nick/w7-b2-g4-pascal-20260829/{dirname}/candidate_completion.json"
        for _lambda, dirname in CANDIDATES
    ]
    if value["candidate_paths"] != expected_candidate_paths:
        fail("worker campaign completion candidate paths/order differs")
    if value["g4_adjudication_run"] != 0 or value["lambda_core_updated"] is not False or value["scientific_execution_authorization"] != "PRESENT":
        fail("worker campaign completion downstream boundary differs")
    if len(value["candidates"]) != len(CANDIDATES) or [candidate["lambda"] for candidate in value["candidates"]] != list(W7_LAMBDA_GRID):
        fail("worker campaign completion candidate count/order differs")
    return value


def verify_profile(profile: dict[str, Any], expected_config_hash: str) -> None:
    expect_keys(profile, PROFILE_KEYS, "profile binding")
    environment = profile["profile_environment"]
    expect_keys(environment, PROFILE_ENVIRONMENT_KEYS, "profile environment")
    try:
        verify_frozen_gpu_binding(profile, config_hash=expected_config_hash)
    except (RuntimeError, ValueError) as exc:
        fail(f"profile binding validation failed: {exc}")
    if profile["authentication_status"] != AUTH_STATUS or profile["execution_profile_id"] != W7_PROFILE_ID:
        fail("profile authentication status differs")
    if profile["gpu_uuid"] != W7_SELECTED_GPU_UUID or profile["gpu_name"] != GPU_NAME or profile["gpu_compute_capability"] != "6.1":
        fail("physical GPU binding differs")
    if profile["cuda_visible_devices"] != W7_SELECTED_GPU_UUID or profile["device"] != "cuda:0":
        fail("profile UUID/ordinal binding differs")
    if profile["git_commit"] != SOURCE_COMMIT or profile["lock_file_sha256"] != PROFILE_LOCK_SHA256:
        fail("profile source/lock binding differs")
    if environment["git_dirty"] is not False or environment["gpu_uuid"] != W7_SELECTED_GPU_UUID or environment["gpu_name"] != GPU_NAME:
        fail("profile environment GPU/source binding differs")
    if environment["git_commit"] != SOURCE_COMMIT or environment["config_hash"] != expected_config_hash:
        fail("profile environment commit/config differs")
    if environment["execution_profile_id"] != W7_PROFILE_ID or environment["lock_file_sha256"] != PROFILE_LOCK_SHA256:
        fail("profile environment profile/lock differs")


def verify_sidecar(sidecar: dict[str, Any], checkpoint: dict[str, Any], epoch: dict[str, Any], lambda_value: float, config_hash: str) -> None:
    expect_keys(sidecar, SIDECAR_KEYS, "checkpoint sidecar")
    if sidecar["schema_version"] != SCHEMA_VERSION or sidecar["artifact_role"] != SIDECAR_ROLE:
        fail("checkpoint sidecar role/version differs")
    if sidecar["eligibility"] != eligibility_for_role(W7_CHECKPOINT_ROLE):
        fail("checkpoint sidecar eligibility differs")
    if sidecar["completed_epoch"] != epoch["epoch"] or sidecar["next_epoch"] != epoch["epoch"] + 1:
        fail("checkpoint sidecar epoch differs")
    if sidecar["checkpoint_id"] != checkpoint["checkpoint_id"] or sidecar["checkpoint_bytes"] != checkpoint["checkpoint_bytes"]:
        fail("checkpoint sidecar file identity differs")
    if sidecar["checkpoint_path"] != f"checkpoints/epoch-{epoch['epoch']:04d}.pt" or sidecar["epoch_record_path"] != f"epochs/epoch-{epoch['epoch']:04d}.json":
        fail("checkpoint sidecar relative path differs")
    if sidecar["config_hash"] != config_hash or sidecar["protocol_config_hash"] != PROTOCOL_HASH:
        fail("checkpoint sidecar config/protocol differs")
    expected = {
        "source_commit": SOURCE_COMMIT, "source_manifest_id": SOURCE_MANIFEST_ID, "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "execution_image": W7_EXECUTION_IMAGE_FAMILY, "execution_profile_id": W7_PROFILE_ID, "gpu_uuid": W7_SELECTED_GPU_UUID,
        "dataset": W7_DATASET, "ratio": W7_RATIO, "k": int(get("bandwidth.k_symbols.imagenette160.r_1_6")), "lambda": lambda_value,
        "train_seed": W7_TRAIN_SEED, "channel_seed": W7_CHANNEL_SEED, "train_snr_db": W7_TRAINING_SNR_DB,
        "epoch_record_path": f"epochs/epoch-{epoch['epoch']:04d}.json", "epoch_record_id": epoch["record_id"], "epoch_record_sha256": epoch["epoch_record_sha256"],
        "global_optimizer_step": epoch["global_optimizer_step"], "accumulation_position": 0,
    }
    if any(sidecar[key] != expected_value for key, expected_value in expected.items()):
        fail("checkpoint sidecar frozen binding differs")
    if not isinstance(sidecar["checkpoint_write_seconds"], (int, float)) or isinstance(sidecar["checkpoint_write_seconds"], bool) or not math.isfinite(float(sidecar["checkpoint_write_seconds"])) or float(sidecar["checkpoint_write_seconds"]) < 0:
        fail("checkpoint write duration is invalid")
    if hashlib.sha256(canonical_bytes(sidecar)).hexdigest() != checkpoint["sidecar_sha256"]:
        fail("checkpoint sidecar content/file SHA differs")


def verify_validation(summary: dict[str, Any], epoch: int, checkpoint_id: str) -> None:
    expect_keys(summary, SUMMARY_KEYS, "validation summary")
    if summary["schema_version"] != SCHEMA_VERSION or summary["artifact_role"] != SUMMARY_ROLE or summary["epoch"] != epoch:
        fail("validation summary role/epoch differs")
    if summary["checkpoint_id"] != checkpoint_id or summary["n_total"] != VAL_COUNT:
        fail("validation summary checkpoint/denominator differs")
    if not isinstance(summary["n_correct"], int) or isinstance(summary["n_correct"], bool) or not 0 <= summary["n_correct"] <= VAL_COUNT:
        fail("validation correct count is invalid")
    if summary["top1_accuracy"] != summary["n_correct"] / summary["n_total"]:
        fail("validation top1 is not count-derived")
    if summary["noise_policy"] != W7_VALIDATION_NOISE_POLICY or summary["noise_policy_hash"] != NOISE_POLICY_HASH or summary["noise_id_digest"] != NOISE_ID_DIGEST:
        fail("validation noise policy/digest differs")
    for field in ("prediction_digest", "evaluation_config_hash", "noise_policy_hash", "noise_id_digest", "row_digest"):
        expect_hex(summary[field], 64, f"validation {field}")
    if summary["evaluation_config_hash"] != EVALUATION_CONFIG_HASH:
        fail("validation evaluation config differs")
    if summary["summary_id"] != canonical_sha256({key: item for key, item in summary.items() if key != "summary_id"}):
        fail("validation summary content digest differs")
    expect_hex(summary["summary_id"], 64, "validation summary ID")


def verify_epoch(epoch: dict[str, Any], raw_checkpoint: dict[str, Any], raw_summary: dict[str, Any], payload: dict[str, Any], lambda_value: float, config_hash: str, prior_checkpoint: str | None, prior_step: int, train: list[tuple[str, int]]) -> None:
    expect_keys(epoch, INDEX_EPOCH_KEYS, "compact epoch record")
    if epoch["epoch"] < 0 or epoch["epoch"] >= EXPECTED_EPOCHS or epoch["next_epoch"] != epoch["epoch"] + 1:
        fail("compact epoch boundary differs")
    e = epoch["epoch"]
    if epoch["samples"] != TRAIN_COUNT or epoch["expected_samples"] != TRAIN_COUNT or epoch["stable_id_count"] != TRAIN_COUNT:
        fail(f"training denominator differs at epoch {e}")
    order_sha, set_sha = expected_order(train, e)
    if epoch["stable_id_order_sha256"] != order_sha or epoch["stable_id_set_sha256"] != set_sha:
        fail(f"training stable-ID order/set differs at epoch {e}")
    if epoch["training_noise_id_count"] != TRAIN_COUNT or epoch["training_noise_id_sha256"] != training_noise_digest([item[0] for item in train], e):
        fail(f"training noise identity digest differs at epoch {e}")
    for field in ("epoch_record_sha256", "record_id", "stable_id_order_sha256", "stable_id_set_sha256", "training_noise_id_sha256", "lineage_sha256"):
        expect_hex(epoch[field], 64, f"epoch {e} {field}")
    if epoch["microbatches"] != MICROBATCHES or epoch["optimizer_steps"] + epoch["grad_scaler_skips"] != MICROBATCHES:
        fail(f"microbatch/update arithmetic differs at epoch {e}")
    if epoch["global_optimizer_step"] != prior_step + epoch["optimizer_steps"]:
        fail(f"global optimizer-step recurrence differs at epoch {e}")
    if epoch["lr"] != expected_lr(e):
        fail(f"epoch-start cosine LR differs at epoch {e}")
    if epoch["finite_loss"] is not True:
        fail(f"finite loss flag differs at epoch {e}")
    gradient = epoch["gradient_audit"]
    expect_keys(gradient, EPOCH_GRADIENT_KEYS, f"epoch {e} gradient audit")
    if gradient["optimizer_parameter_count"] != 66 or gradient["optimizer_gradient_count_min"] != 66 or gradient["optimizer_gradient_count_max"] != 66:
        fail(f"optimizer-wide gradient count differs at epoch {e}")
    if gradient["all_optimizer_gradients_finite"] != (epoch["grad_scaler_skips"] == 0) or not isinstance(gradient["all_named_present_gradients_finite"], bool):
        fail(f"post-unscale finite classification differs at epoch {e}")
    if epoch["lineage_sha256"] != payload["lineage_sha256"]:
        fail(f"epoch/payload lineage digest differs at epoch {e}")
    for field in ("checkpoint_id", "sidecar_sha256", "epoch_record_id", "epoch_record_sha256"):
        expect_hex(raw_checkpoint[field], 64, f"checkpoint {e} {field}")
    if raw_checkpoint["predecessor_checkpoint_id"] is not None:
        expect_hex(raw_checkpoint["predecessor_checkpoint_id"], 64, f"checkpoint {e} predecessor")
    if raw_checkpoint["predecessor_checkpoint_id"] != prior_checkpoint or raw_checkpoint["global_optimizer_step"] != epoch["global_optimizer_step"]:
        fail(f"checkpoint chain/step differs at epoch {e}")
    verify_validation(raw_summary, e, raw_checkpoint["checkpoint_id"])


def verify_selected_result(result: dict[str, Any], selected: dict[str, Any], summaries: list[dict[str, Any]], validation: list[tuple[str, int]], expected_noise: list[str], lambda_value: float) -> dict[str, Any]:
    expect_keys(result, SELECTED_RESULT_KEYS, "selected checkpoint result")
    if result["schema_version"] != SCHEMA_VERSION or result["artifact_role"] != SELECTED_RESULT_ROLE:
        fail("selected checkpoint result role/version differs")
    if result["result_digest"] != canonical_sha256({key: item for key, item in result.items() if key != "result_digest"}):
        fail("selected checkpoint result digest differs")
    max_accuracy = max(summary["top1_accuracy"] for summary in summaries)
    selected_epoch = min(summary["epoch"] for summary in summaries if summary["top1_accuracy"] == max_accuracy)
    summary = summaries[selected_epoch]
    independent = {
        "metric": "top1_accuracy", "mode": "max", "tie_break": "earliest_epoch", "selected_epoch": selected_epoch,
        "selected_checkpoint_id": summary["checkpoint_id"], "n_correct": summary["n_correct"], "n_total": summary["n_total"],
        "top1_accuracy": summary["top1_accuracy"],
    }
    if selected["selected_epoch"] != selected_epoch or selected["selected_checkpoint_id"] != summary["checkpoint_id"]:
        fail(f"selected epoch/checkpoint differs for lambda {lambda_value}")
    if result["checkpoint_epoch"] != selected_epoch or result["checkpoint_id"] != summary["checkpoint_id"] or result["selection"] != independent:
        fail(f"selected result selection differs for lambda {lambda_value}")
    if result["calibration_validation"] != summary:
        fail(f"selected calibration summary differs for lambda {lambda_value}")
    rows = result["calibration_rows"]
    expect_rows = [item[0] for item in validation]
    if len(rows) != VAL_COUNT or [row["stable_sample_id"] for row in rows] != expect_rows or [row["noise_id"] for row in rows] != expected_noise:
        fail(f"selected calibration identity/noise coverage differs for lambda {lambda_value}")
    for row in rows:
        expect_keys(row, CALIBRATION_ROW_KEYS, "selected calibration row")
        if not isinstance(row["label"], int) or isinstance(row["label"], bool) or not isinstance(row["prediction"], int) or isinstance(row["prediction"], bool) or not isinstance(row["correct"], bool):
            fail(f"selected calibration types differ for lambda {lambda_value}")
        if row["correct"] != (row["prediction"] == row["label"]):
            fail(f"selected calibration correctness differs for lambda {lambda_value}")
        if not isinstance(row["mse"], (int, float)) or isinstance(row["mse"], bool) or not math.isfinite(float(row["mse"])) or float(row["mse"]) < 0:
            fail(f"selected calibration MSE is invalid for lambda {lambda_value}")
        if row["psnr_db"] == "inf":
            if row["mse"] != 0:
                fail(f"selected calibration zero-MSE representation differs for lambda {lambda_value}")
        elif not isinstance(row["psnr_db"], (int, float)) or not math.isfinite(float(row["psnr_db"])) or float(row["mse"]) <= 0 or float(row["psnr_db"]) != 10.0 * math.log10(1.0 / float(row["mse"])):
            fail(f"selected calibration PSNR definition differs for lambda {lambda_value}")
        if not isinstance(row["papr_db"], (int, float)) or isinstance(row["papr_db"], bool) or not math.isfinite(float(row["papr_db"])):
            fail(f"selected calibration PAPR is invalid for lambda {lambda_value}")
    n_correct = sum(int(row["correct"]) for row in rows)
    if n_correct != summary["n_correct"] or canonical_sha256(rows) != summary["row_digest"]:
        fail(f"selected calibration count/row digest differs for lambda {lambda_value}")
    prediction_digest = canonical_sha256([{"stable_sample_id": row["stable_sample_id"], "prediction": row["prediction"], "correct": row["prediction"] == row["label"]} for row in rows])
    if prediction_digest != summary["prediction_digest"]:
        fail(f"selected calibration prediction digest differs for lambda {lambda_value}")
    if result["protected_counters"] != {"w7_candidate_results": 0, "learned_test_inference": 0, "test_model_facing_access": 0}:
        fail(f"selected protected counters differ for lambda {lambda_value}")
    psnr = result["psnr_evaluation"]
    expect_keys(psnr, PSNR_EVALUATION_KEYS, "selected PSNR/PAPR evaluation")
    if psnr["checkpoint_id"] != result["checkpoint_id"] or psnr["snr_db"] != W7_PSNR_SNR_DB or psnr["denominator"] != VAL_COUNT or psnr["data_range"] != 1.0:
        fail(f"selected PSNR binding differs for lambda {lambda_value}")
    if psnr["psnr_definition"] != "per_image_mse_all_RGB_pixels_then_arithmetic_mean" or psnr["papr_definition"] != "symbol_domain_per_image_then_arithmetic_mean":
        fail(f"selected PSNR/PAPR definitions differ for lambda {lambda_value}")
    per_image = psnr["per_image"]
    if len(per_image) != VAL_COUNT or [row["stable_sample_id"] for row in per_image] != expect_rows:
        fail(f"selected 15 dB metric coverage differs for lambda {lambda_value}")
    for row in per_image:
        expect_keys(row, PER_IMAGE_KEYS, "selected per-image metric")
        if row["psnr_db"] != "inf" and (not isinstance(row["psnr_db"], (int, float)) or not math.isfinite(float(row["psnr_db"]))):
            fail(f"selected per-image PSNR is invalid for lambda {lambda_value}")
        if not isinstance(row["papr_db"], (int, float)) or isinstance(row["papr_db"], bool) or not math.isfinite(float(row["papr_db"])):
            fail(f"selected per-image PAPR is invalid for lambda {lambda_value}")
    if psnr["per_image_digest"] != canonical_sha256(per_image):
        fail(f"selected per-image digest differs for lambda {lambda_value}")
    mean_psnr = "inf" if any(row["psnr_db"] == "inf" for row in per_image) else float(np.mean(np.asarray([float(row["psnr_db"]) for row in per_image], dtype=np.float64)))
    mean_papr = float(np.mean(np.asarray([float(row["papr_db"]) for row in per_image], dtype=np.float64)))
    if mean_psnr != psnr["psnr_db"] or mean_papr != psnr["papr_db"]:
        fail(f"selected aggregate PSNR/PAPR differs for lambda {lambda_value}")
    return {"selected_epoch": selected_epoch, "selected_checkpoint_id": summary["checkpoint_id"], "n_correct": n_correct, "n_total": VAL_COUNT, "top1_accuracy": n_correct / VAL_COUNT, "mean_psnr_db": mean_psnr, "mean_papr": mean_papr}


def verify_index(index: dict[str, Any]) -> dict[str, Any]:
    scan_for_forbidden_keys(index)
    if index["campaign_id"] != CAMPAIGN_ID or index["candidate_order"] != list(W7_LAMBDA_GRID):
        fail("candidate index campaign/order differs")
    verify_manifest_ref(index["campaign_manifest"])
    worker_completion = verify_worker_completion(index["campaign_completion"])
    expect_keys(index["authorization"], {"authorization_id", "file_sha256", "path"}, "authorization reference")
    if index["authorization"]["authorization_id"] != AUTHORIZATION_ID or index["authorization"]["file_sha256"] != AUTHORIZATION_SHA256:
        fail("authorization reference differs")
    expect_keys(index["source"], {"source_commit", "source_manifest_id", "source_manifest_sha256"}, "source reference")
    if index["source"] != {"source_commit": SOURCE_COMMIT, "source_manifest_id": SOURCE_MANIFEST_ID, "source_manifest_sha256": SOURCE_MANIFEST_SHA256}:
        fail("source reference differs")
    expect_keys(index["profile"], {"profile_freeze_id", "profile_freeze_sha256", "execution_profile_id", "execution_image", "gpu_name", "gpu_uuid"}, "profile reference")
    if index["profile"] != {"profile_freeze_id": PROFILE_FREEZE_ID, "profile_freeze_sha256": PROFILE_FREEZE_SHA256, "execution_profile_id": W7_PROFILE_ID, "execution_image": W7_EXECUTION_IMAGE_FAMILY, "gpu_name": GPU_NAME, "gpu_uuid": W7_SELECTED_GPU_UUID}:
        fail("profile reference differs")
    expect_keys(index["protocol"], {
        "dataset", "dataset_version", "ratio", "k", "lambda_grid", "lambda_order", "train_seed", "channel_seed", "training_snr_db", "validation_snr_db", "psnr_snr_db", "epochs", "training_denominator", "validation_denominator", "physical_batch_size", "accumulation_factor", "effective_batch_size", "validation_batch_size", "drop_last", "final_partial_batch_samples", "microbatches_per_epoch", "optimizer", "optimizer_implementation", "scheduler", "scheduler_indexing", "scheduler_step_unit", "amp_enabled", "grad_scaler_policy", "scaler_state_checkpointed", "validation_noise_policy", "checkpoint_selection", "psnr_definition", "papr_definition",
    }, "protocol")
    protocol_expected = {
        "dataset": W7_DATASET, "dataset_version": DATASET_VERSION, "ratio": W7_RATIO, "k": int(get("bandwidth.k_symbols.imagenette160.r_1_6")), "lambda_grid": list(W7_LAMBDA_GRID), "lambda_order": "exact_configured_lambda_grid_order", "train_seed": W7_TRAIN_SEED, "channel_seed": W7_CHANNEL_SEED, "training_snr_db": W7_TRAINING_SNR_DB, "validation_snr_db": W7_CALIBRATION_SNR_DB, "psnr_snr_db": W7_PSNR_SNR_DB, "epochs": EXPECTED_EPOCHS, "training_denominator": TRAIN_COUNT, "validation_denominator": VAL_COUNT, "physical_batch_size": W7_PHYSICAL_BATCH_SIZE, "accumulation_factor": 1, "effective_batch_size": TARGET_BATCH, "validation_batch_size": W7_VALIDATION_BATCH_SIZE, "drop_last": False, "final_partial_batch_samples": FINAL_PARTIAL_BATCH, "microbatches_per_epoch": MICROBATCHES, "optimizer": "adam", "optimizer_implementation": "torch.optim.Adam", "scheduler": "cosine", "scheduler_indexing": "zero_based", "scheduler_step_unit": "epoch_start", "amp_enabled": True, "grad_scaler_policy": "optimizer_wide_post_unscale_finite_authoritative_skips_excluded_from_global_step", "scaler_state_checkpointed": True, "validation_noise_policy": W7_VALIDATION_NOISE_POLICY, "checkpoint_selection": {"metric": "top1_accuracy", "mode": "max", "snr_db": W7_CALIBRATION_SNR_DB, "tie_break": "earliest_epoch"}, "psnr_definition": "per_image_mse_all_RGB_pixels_then_arithmetic_mean", "papr_definition": "symbol_domain_per_image_then_arithmetic_mean",
    }
    if index["protocol"] != protocol_expected:
        fail("frozen protocol differs")
    expect_keys(index["homogeneity"], {"fields", "candidate_rows", "unexpected_differences"}, "homogeneity")
    if index["homogeneity"]["unexpected_differences"] != [] or len(index["homogeneity"]["candidate_rows"]) != len(CANDIDATES):
        fail("homogeneity matrix is incomplete")
    if len(index["candidates"]) != len(CANDIDATES) or len(index["factual_measurements"]) != len(CANDIDATES):
        fail("candidate count differs")
    train, validation = load_manifest_ids()
    expected_noise = expected_validation_noise(validation)
    common_fields = None
    candidate_results: list[dict[str, Any]] = []
    for order, ((lambda_value, dirname), candidate) in enumerate(zip(CANDIDATES, index["candidates"], strict=True)):
        expect_keys(candidate, INDEX_CANDIDATE_KEYS, f"candidate {lambda_value}")
        if candidate["campaign_order"] != order or candidate["lambda"] != lambda_value or candidate["candidate_root"] != f"/home/nick/w7-b2-g4-pascal-20260829/{dirname}":
            fail(f"candidate order/root/lambda differs for lambda {lambda_value}")
        config = load_w7_config(lambda_value=lambda_value, role=W7_CHECKPOINT_ROLE, physical_batch_size=W7_PHYSICAL_BATCH_SIZE, accumulation_factor=1, validation_batch_size=W7_VALIDATION_BATCH_SIZE)
        expected_config = run_config_hash(config)
        if candidate["config_hash"] != expected_config or candidate["protocol_config_hash"] != PROTOCOL_HASH or protocol_config_hash(config) != PROTOCOL_HASH:
            fail(f"candidate config/protocol hash differs for lambda {lambda_value}")
        expected_row = expected_homogeneity(lambda_value, config)
        if candidate["homogeneity"] != expected_row or index["homogeneity"]["candidate_rows"][order] != expected_row:
            fail(f"candidate homogeneity differs for lambda {lambda_value}")
        candidate_common = {key: value for key, value in expected_row.items() if key != "lambda"}
        if common_fields is None:
            common_fields = candidate_common
        elif candidate_common != common_fields:
            fail("homogeneity fields differ across candidates")
        verify_profile(candidate["profile_binding"], expected_config)
        candidate_ref = candidate["candidate_completion"]
        expect_keys(candidate_ref, CANDIDATE_REF_KEYS, f"candidate {lambda_value} completion reference")
        raw_candidate = candidate_ref["value"]
        expect_keys(raw_candidate, {"schema_version", "artifact_role", "candidate_id", "status", "authentication_status", "eligibility", "lambda", "lineage", "selected_validation", "psnr_evaluation", "selected_validation_result_digest", "selected_evidence", "test_access"}, f"worker candidate {lambda_value}")
        if candidate_ref["file_sha256"] != candidate["candidate_completion"]["file_sha256"] or raw_candidate["lambda"] != lambda_value or raw_candidate["candidate_id"] != candidate["worker_candidate_id"]:
            fail(f"candidate completion binding differs for lambda {lambda_value}")
        if raw_candidate["candidate_id"] != "w7candidate-" + canonical_sha256({"lambda": lambda_value, "selected": raw_candidate["selected_validation_result_digest"]}) or raw_candidate["status"] != "COMPLETE" or raw_candidate["authentication_status"] != AUTH_STATUS or raw_candidate["test_access"] != 0:
            fail(f"candidate completion identity/status differs for lambda {lambda_value}")
        if raw_candidate["eligibility"] != {"selection_eligibility": "ELIGIBLE_FOR_OWN_G4_ONLY", "w7_g4_eligibility": "ELIGIBLE_FOR_G4_CANDIDATE", "w8_eligibility": "NOT_ELIGIBLE_FOR_W8_INITIALIZATION", "test_eligibility": "NOT_ELIGIBLE_FOR_TEST"}:
            fail(f"candidate eligibility differs for lambda {lambda_value}")
        lineage = raw_candidate["lineage"]
        expected_lineage_fields = {
            "protocol_version": "w7-g4-pre-execution-v1",
            "source_commit": SOURCE_COMMIT,
            "source_manifest_id": SOURCE_MANIFEST_ID,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "protocol_config_hash": PROTOCOL_HASH,
            "execution_image": W7_EXECUTION_IMAGE_FAMILY,
            "execution_profile_id": W7_PROFILE_ID,
            "gpu_uuid": W7_SELECTED_GPU_UUID,
            "dataset": W7_DATASET,
            "split_manifest_hash": SPLIT_MANIFEST_HASH,
            "architecture": expected_row["architecture"],
            "ratio": W7_RATIO,
            "k": expected_row["k"],
            "train_seed": W7_TRAIN_SEED,
            "channel_seed": W7_CHANNEL_SEED,
            "train_snr_db": W7_TRAINING_SNR_DB,
            "epochs": EXPECTED_EPOCHS,
            "optimizer": "adam",
            "scheduler": "cosine",
            "checkpoint_selection": {"metric": "top1_accuracy", "mode": "max", "tie_break": "earliest_epoch", "snr_db": W7_CALIBRATION_SNR_DB},
            "validation_noise_policy": W7_VALIDATION_NOISE_POLICY,
        }
        if lineage != expected_lineage_fields:
            fail(f"candidate lineage differs for lambda {lambda_value}")
        if raw_candidate["selected_evidence"]["file_sha256"] != candidate["selected"]["selected_checkpoint_result"]["file_sha256"] or raw_candidate["selected_evidence"]["result_digest"] != candidate["selected"]["selected_checkpoint_result"]["value"]["result_digest"]:
            fail(f"candidate selected-evidence binding differs for lambda {lambda_value}")
        expect_keys(candidate["checkpoint_payload_audit"], {"schema_version", "checkpoint_payload_count", "payload_summary_sha256", "deep_validation_count", "deep_entries_sha256", "entries", "selected_checkpoint_audit"}, f"candidate {lambda_value} payload audit")
        payload_audit = candidate["checkpoint_payload_audit"]
        if payload_audit["schema_version"] != SCHEMA_VERSION or payload_audit["checkpoint_payload_count"] != EXPECTED_EPOCHS or payload_audit["deep_validation_count"] != EXPECTED_EPOCHS or payload_audit["deep_entries_sha256"] != canonical_sha256(payload_audit["entries"]):
            fail(f"candidate payload audit count/digest differs for lambda {lambda_value}")
        if len(payload_audit["entries"]) != EXPECTED_EPOCHS:
            fail(f"candidate payload audit entries differ for lambda {lambda_value}")
        deep = payload_audit["selected_checkpoint_audit"]
        expect_keys(deep, {"checkpoint_id", "config_hash", "lambda", "model_state_key_count", "optimizer_state_param_count", "payload_validated", "protocol_hash", "scaler_state_present", "scheduler_state", "selected_epoch", "sidecar_validated"}, f"candidate {lambda_value} selected payload audit")
        if deep["payload_validated"] is not True or deep["sidecar_validated"] is not True or deep["lambda"] != lambda_value or deep["config_hash"] != expected_config or deep["protocol_hash"] != PROTOCOL_HASH:
            fail(f"candidate selected payload deep audit differs for lambda {lambda_value}")
        if len(candidate["epochs"]) != EXPECTED_EPOCHS or len(candidate["checkpoints"]) != EXPECTED_EPOCHS or len(candidate["validation_summaries"]) != EXPECTED_EPOCHS:
            fail(f"candidate history count differs for lambda {lambda_value}")
        if [item["epoch"] for item in candidate["epochs"]] != list(range(EXPECTED_EPOCHS)) or [item["epoch"] for item in candidate["checkpoints"]] != list(range(EXPECTED_EPOCHS)) or [item["summary"]["epoch"] for item in candidate["validation_summaries"]] != list(range(EXPECTED_EPOCHS)):
            fail(f"candidate history epoch sequence differs for lambda {lambda_value}")
        prior_checkpoint = None
        prior_step = 0
        applied = 0
        skips = 0
        summaries: list[dict[str, Any]] = []
        for e in range(EXPECTED_EPOCHS):
            epoch = candidate["epochs"][e]
            checkpoint = candidate["checkpoints"][e]
            validation_ref = candidate["validation_summaries"][e]
            expect_keys(checkpoint, INDEX_CHECKPOINT_KEYS, f"candidate {lambda_value} checkpoint {e}")
            if checkpoint["epoch"] != e or checkpoint["path"] != f"{candidate['candidate_root']}/checkpoints/epoch-{e:04d}.pt" or checkpoint["sidecar_path"] != f"{candidate['candidate_root']}/checkpoints/epoch-{e:04d}.sidecar.json":
                fail(f"candidate {lambda_value} checkpoint path/epoch differs")
            if checkpoint["checkpoint_id"] != checkpoint["sidecar"]["checkpoint_id"] or checkpoint["checkpoint_id"] != checkpoint["payload_audit"]["checkpoint_id"]:
                fail(f"candidate {lambda_value} checkpoint identity differs")
            if checkpoint["predecessor_checkpoint_id"] != prior_checkpoint:
                fail(f"candidate {lambda_value} checkpoint predecessor differs at {e}")
            verify_sidecar(checkpoint["sidecar"], checkpoint, epoch, lambda_value, expected_config)
            expect_keys(checkpoint["payload_audit"], PAYLOAD_ENTRY_KEYS, f"candidate {lambda_value} payload {e}")
            payload = checkpoint["payload_audit"]
            if payload["epoch"] != e or payload["epoch_manifest"] != {"path": f"epochs/epoch-{e:04d}.json", "record_id": epoch["record_id"], "record_sha256": epoch["epoch_record_sha256"]} or payload["predecessor_checkpoint_id"] != prior_checkpoint:
                fail(f"candidate {lambda_value} payload lineage differs at {e}")
            if payload["model_state_key_count"] != 68 or payload["optimizer_state_param_count"] != 66 or payload["optimizer_param_group_count"] != 1 or payload["scaler_state_present"] is not True or payload["scheduler_state"] != {"completed_epoch": e}:
                fail(f"candidate {lambda_value} payload state schema differs at {e}")
            if payload["lineage_sha256"] != epoch["lineage_sha256"] or payload["execution_profile_sha256"] != canonical_sha256(candidate["profile_binding"]):
                fail(f"candidate {lambda_value} payload binding digest differs at {e}")
            if payload["eligibility_sha256"] != canonical_sha256(eligibility_for_role(W7_CHECKPOINT_ROLE)) or payload["rng_policy_sha256"] != canonical_sha256(W7_RNG_STATE_POLICY) or payload["protected_counters_sha256"] != canonical_sha256(W7_PROTECTED_COUNTERS):
                fail(f"candidate {lambda_value} payload policy digest differs at {e}")
            verify_epoch(epoch, checkpoint, candidate["validation_summaries"][e]["summary"], payload, lambda_value, expected_config, prior_checkpoint, prior_step, train)
            summary = candidate["validation_summaries"][e]["summary"]
            expect_keys(candidate["validation_summaries"][e], VALIDATION_REF_KEYS, f"candidate {lambda_value} validation reference {e}")
            if candidate["validation_summaries"][e]["path"] != f"{candidate['candidate_root']}/validation/epoch-{e:04d}.json":
                fail(f"candidate {lambda_value} validation path differs at {e}")
            expect_hex(candidate["validation_summaries"][e]["file_sha256"], 64, f"candidate {lambda_value} validation file SHA {e}")
            summaries.append(summary)
            prior_checkpoint = checkpoint["checkpoint_id"]
            prior_step = epoch["global_optimizer_step"]
            applied += epoch["optimizer_steps"]
            skips += epoch["grad_scaler_skips"]
        if prior_step != candidate["training_totals"]["final_global_optimizer_step"] or applied != candidate["training_totals"]["applied_optimizer_steps"] or skips != candidate["training_totals"]["gradscaler_skips"]:
            fail(f"candidate {lambda_value} training totals differ")
        totals = candidate["training_totals"]
        expect_keys(totals, {"completed_epochs", "training_denominator_per_epoch", "microbatches_per_epoch", "final_partial_batch_samples", "applied_optimizer_steps", "gradscaler_skips", "final_global_optimizer_step"}, f"candidate {lambda_value} training totals")
        if totals["completed_epochs"] != EXPECTED_EPOCHS or totals["training_denominator_per_epoch"] != TRAIN_COUNT or totals["microbatches_per_epoch"] != MICROBATCHES or totals["final_partial_batch_samples"] != FINAL_PARTIAL_BATCH:
            fail(f"candidate {lambda_value} training completeness differs")
        latest = candidate["latest"]
        expect_keys(latest, {"path", "file_sha256", "value"}, f"candidate {lambda_value} latest")
        if latest["path"] != f"{candidate['candidate_root']}/latest.json" or latest["value"] != candidate["checkpoints"][-1]["sidecar"] or latest["file_sha256"] != candidate["checkpoints"][-1]["sidecar_sha256"]:
            fail(f"candidate {lambda_value} latest pointer differs")
        selected_result_ref = candidate["selected"]["selected_checkpoint_result"]
        expect_keys(selected_result_ref, SELECTED_RESULT_REF_KEYS, f"candidate {lambda_value} selected result reference")
        if selected_result_ref["path"] != f"{candidate['candidate_root']}/selected_checkpoint_result.json":
            fail(f"candidate {lambda_value} selected result path differs")
        selected = candidate["selected"]
        expect_keys(selected, INDEX_SELECTED_KEYS, f"candidate {lambda_value} selected index")
        if selected["selected_checkpoint"] != candidate["checkpoints"][selected["selected_epoch"]]:
            fail(f"candidate {lambda_value} selected checkpoint reference differs")
        metrics = verify_selected_result(selected_result_ref["value"], selected, summaries, validation, expected_noise, lambda_value)
        expected_selected = {"selected_epoch": metrics["selected_epoch"], "selected_checkpoint_id": metrics["selected_checkpoint_id"], "derived_n_correct": metrics["n_correct"], "derived_n_total": metrics["n_total"], "derived_top1_accuracy": metrics["top1_accuracy"], "derived_mean_psnr_db": metrics["mean_psnr_db"], "derived_mean_papr": metrics["mean_papr"], "psnr_definition": "per_image_mse_all_RGB_pixels_then_arithmetic_mean", "papr_definition": "symbol_domain_per_image_then_arithmetic_mean"}
        for key, value in expected_selected.items():
            if selected[key] != value:
                fail(f"candidate {lambda_value} selected derived field differs: {key}")
        raw_candidate_selected = raw_candidate["selected_validation"]
        if raw_candidate_selected != {"checkpoint_id": metrics["selected_checkpoint_id"], "epoch": metrics["selected_epoch"], "n_correct": metrics["n_correct"], "n_total": VAL_COUNT, "top1_accuracy": metrics["top1_accuracy"]}:
            fail(f"candidate {lambda_value} frozen selected-validation block differs")
        if raw_candidate["selected_validation_result_digest"] != selected_result_ref["value"]["result_digest"] or raw_candidate["psnr_evaluation"]["psnr_db"] != metrics["mean_psnr_db"] or raw_candidate["psnr_evaluation"]["per_image_digest"] != selected_result_ref["value"]["psnr_evaluation"]["per_image_digest"]:
            fail(f"candidate {lambda_value} frozen selected metric block differs")
        factual = index["factual_measurements"][order]
        if factual != {"lambda": lambda_value, "selected_epoch": metrics["selected_epoch"], "selected_checkpoint_id": metrics["selected_checkpoint_id"], "n_correct": metrics["n_correct"], "n_total": VAL_COUNT, "top1_accuracy": metrics["top1_accuracy"], "mean_psnr_db": metrics["mean_psnr_db"], "mean_papr": metrics["mean_papr"], "applied_optimizer_steps": applied, "gradscaler_skips": skips}:
            fail(f"factual measurement differs for lambda {lambda_value}")
        candidate_results.append(metrics | {"lambda": lambda_value, "applied_optimizer_steps": applied, "gradscaler_skips": skips})
        if worker_completion["candidates"][order] != raw_candidate:
            fail(f"worker campaign completion candidate differs for lambda {lambda_value}")
    if index["homogeneity"]["fields"] != common_fields:
        fail("homogeneity common fields differ")
    if index["worker_scan"]["schema_version"] != SCHEMA_VERSION or index["worker_scan"]["worker_hostname"] != "confessor":
        fail("worker scan reference differs")
    expect_keys(index["worker_scan"], {"path", "file_sha256", "schema_version", "worker_hostname"}, "worker scan reference")
    expect_keys(index["checkpoint_audit_capture"], {"path", "file_sha256", "deep_path", "deep_file_sha256"}, "checkpoint audit capture")
    return {"index": index, "worker_completion": worker_completion, "candidates": candidate_results, "train": train, "validation": validation, "expected_noise": expected_noise}


def verify_custody(custody: dict[str, Any], index_result: dict[str, Any]) -> None:
    index = index_result["index"]
    scan_for_forbidden_keys(custody)
    if custody["campaign_id"] != CAMPAIGN_ID or custody["candidate_index_id"] != index["index_id"] or custody["worker_hostname"] != "confessor" or custody["status"] != "FROZEN_WORKER_CUSTODY":
        fail("checkpoint custody header differs")
    if custody["checkpoint_policy"] != "one_completed_epoch_checkpoint_per_epoch_0_through_99" or custody["expected_checkpoint_count"] != EXPECTED_EPOCHS * len(CANDIDATES) or custody["observed_checkpoint_count"] != EXPECTED_EPOCHS * len(CANDIDATES) or custody["selected_checkpoint_bytes_remain_on_worker"] is not True:
        fail("checkpoint custody count/policy differs")
    if len(custody["candidates"]) != len(CANDIDATES):
        fail("checkpoint custody candidate count differs")
    for index_candidate, custody_candidate in zip(index["candidates"], custody["candidates"], strict=True):
        expect_keys(custody_candidate, CUSTODY_CANDIDATE_KEYS, "custody candidate")
        if custody_candidate["lambda"] != index_candidate["lambda"] or custody_candidate["campaign_order"] != index_candidate["campaign_order"] or custody_candidate["candidate_root"] != index_candidate["candidate_root"]:
            fail("custody candidate identity differs")
        if custody_candidate["candidate_completion_path"] != index_candidate["candidate_completion"]["path"] or custody_candidate["candidate_completion_id"] != index_candidate["worker_candidate_id"] or custody_candidate["candidate_completion_file_sha256"] != index_candidate["candidate_completion"]["file_sha256"]:
            fail("custody candidate completion binding differs")
        cp_rows = index_candidate["checkpoints"]
        if custody_candidate["checkpoint_count"] != EXPECTED_EPOCHS or len(custody_candidate["checkpoints"]) != EXPECTED_EPOCHS:
            fail("custody checkpoint count differs")
        if custody_candidate["validation_history_digest"] != canonical_sha256(index_candidate["validation_summaries"]) or custody_candidate["checkpoint_chain_digest"] != canonical_sha256(cp_rows):
            fail("custody history digest differs")
        expected_result_digest = canonical_sha256({"candidate_id": index_candidate["worker_candidate_id"], "selected_result_digest": index_candidate["selected"]["selected_checkpoint_result"]["value"]["result_digest"], "selected_epoch": index_candidate["selected"]["selected_epoch"]})
        if custody_candidate["candidate_result_digest"] != expected_result_digest:
            fail("custody candidate result digest differs")
        selected_cp = cp_rows[index_candidate["selected"]["selected_epoch"]]
        if custody_candidate["selected_checkpoint_path"] != selected_cp["path"] or custody_candidate["selected_checkpoint_id"] != selected_cp["checkpoint_id"] or custody_candidate["selected_checkpoint_file_sha256"] != selected_cp["checkpoint_id"] or custody_candidate["selected_checkpoint_sidecar_path"] != selected_cp["sidecar_path"] or custody_candidate["selected_checkpoint_sidecar_sha256"] != selected_cp["sidecar_sha256"]:
            fail("custody selected checkpoint binding differs")
        for field in ("selected_checkpoint_file_sha256", "selected_checkpoint_sidecar_sha256"):
            expect_hex(custody_candidate[field], 64, f"custody {field}")
        selected_result = index_candidate["selected"]["selected_checkpoint_result"]
        if custody_candidate["selected_checkpoint_result_path"] != selected_result["path"] or custody_candidate["selected_checkpoint_result_file_sha256"] != selected_result["file_sha256"] or custody_candidate["selected_checkpoint_result_digest"] != selected_result["value"]["result_digest"]:
            fail("custody selected result binding differs")
        expect_hex(custody_candidate["selected_checkpoint_result_file_sha256"], 64, "custody selected result file SHA")
        expect_hex(custody_candidate["selected_checkpoint_result_digest"], 64, "custody selected result digest")
        if custody_candidate["checkpoints"] != [{key: item[key] for key in CUSTODY_CHECKPOINT_KEYS} for item in cp_rows]:
            fail("custody checkpoint index differs")
        for e, cp in enumerate(cp_rows):
            custody_cp = custody_candidate["checkpoints"][e]
            expect_keys(custody_cp, CUSTODY_CHECKPOINT_KEYS, "custody checkpoint")
            if cp["checkpoint_id"] != custody_cp["checkpoint_id"] or cp["sidecar_sha256"] != custody_cp["sidecar_sha256"] or cp["epoch_record_sha256"] != custody_cp["epoch_record_sha256"]:
                fail(f"custody checkpoint hash differs at epoch {e}")


def verify_noise(noise: dict[str, Any], index_result: dict[str, Any]) -> None:
    index = index_result["index"]
    train = index_result["train"]
    validation = index_result["validation"]
    expected_noise = index_result["expected_noise"]
    scan_for_forbidden_keys(noise)
    if noise["campaign_id"] != CAMPAIGN_ID or noise["candidate_index_id"] != index["index_id"] or noise["status"] != "AUTHENTICATED_COMMON_NOISE":
        fail("common-noise header differs")
    expected_header = {"evaluation_role": "validation_checkpoint_selection_calibration", "snr_db": W7_CALIBRATION_SNR_DB, "channel_seed": W7_CHANNEL_SEED, "ratio": W7_RATIO, "dataset": W7_DATASET, "validation_denominator": VAL_COUNT, "validation_order": "stable_manifest_order", "noise_policy": W7_VALIDATION_NOISE_POLICY, "noise_policy_hash": NOISE_POLICY_HASH, "stable_id_digest": VAL_STABLE_ID_DIGEST, "noise_id_digest": NOISE_ID_DIGEST, "identity_fields": ["dataset_version", "split_manifest_hash", "stable_sample_id", "channel_seed", "channel", "bw_ratio", "k", "snr_db", "rng_purpose"]}
    if any(noise[key] != value for key, value in expected_header.items()):
        fail("common-noise frozen header differs")
    pairs = noise["paired_samples"]
    if len(pairs) != VAL_COUNT or pairs != [{"stable_sample_id": sid, "noise_id": nid} for (sid, _label), nid in zip(validation, expected_noise, strict=True)]:
        fail("common-noise sample pairing differs")
    if line_digest([item["stable_sample_id"] for item in pairs]) != VAL_STABLE_ID_DIGEST or canonical_sha256([item["noise_id"] for item in pairs]) != NOISE_ID_DIGEST:
        fail("common-noise sample digests differ")
    if len(noise["epoch_summary_audit"]) != EXPECTED_EPOCHS * len(CANDIDATES):
        fail("common-noise epoch audit count differs")
    expected_epoch_rows = []
    for candidate in index["candidates"]:
        for ref in candidate["validation_summaries"]:
            summary = ref["summary"]
            expected_epoch_rows.append({"lambda": candidate["lambda"], "epoch": summary["epoch"], "path": ref["path"], "file_sha256": ref["file_sha256"], "summary_id": summary["summary_id"], "checkpoint_id": summary["checkpoint_id"], "n_total": summary["n_total"], "noise_id_digest": summary["noise_id_digest"], "noise_policy_hash": summary["noise_policy_hash"]})
    if noise["epoch_summary_audit"] != expected_epoch_rows:
        fail("common-noise epoch audit binding differs")
    selected_audit_expected = []
    for candidate in index["candidates"]:
        result = candidate["selected"]["selected_checkpoint_result"]["value"]
        rows = result["calibration_rows"]
        selected_audit_expected.append({"lambda": candidate["lambda"], "selected_epoch": candidate["selected"]["selected_epoch"], "selected_result_path": candidate["selected"]["selected_checkpoint_result"]["path"], "selected_result_file_sha256": candidate["selected"]["selected_checkpoint_result"]["file_sha256"], "selected_result_digest": result["result_digest"], "stable_id_digest": canonical_sha256([row["stable_sample_id"] for row in rows]), "noise_id_digest": canonical_sha256([row["noise_id"] for row in rows]), "row_digest": canonical_sha256(rows), "mismatched_expected_noise_ids": 0})
    if noise["selected_candidate_audit"] != selected_audit_expected:
        fail("common-noise selected audit binding differs")
    expected_pairs = [{"left_lambda": left[0], "right_lambda": right[0], "mismatched_sample_noise_ids": 0, "sample_count": VAL_COUNT} for left_index, left in enumerate(CANDIDATES) for right in CANDIDATES[left_index + 1:]]
    if noise["pairwise_comparisons"] != expected_pairs or noise["all_selected_sample_mismatch_count"] != 0 or noise["test_model_facing_access"] != 0:
        fail("common-noise pairwise result differs")
    for candidate in index["candidates"]:
        selected_rows = candidate["selected"]["selected_checkpoint_result"]["value"]["calibration_rows"]
        if [row["noise_id"] for row in selected_rows] != expected_noise:
            fail("selected candidate noise IDs are not common")


def verify_reconciliation(reconciliation: dict[str, Any], index_result: dict[str, Any], custody: dict[str, Any], noise: dict[str, Any], completion: dict[str, Any]) -> None:
    scan_for_forbidden_keys(reconciliation)
    index = index_result["index"]
    if reconciliation["status"] != "GREEN" or reconciliation["campaign_status"] != "COMPLETE_NOT_YET_ADJUDICATED" or reconciliation["campaign_id"] != CAMPAIGN_ID:
        fail("reconciliation status/campaign differs")
    if reconciliation["scientific_source_commit"] != SOURCE_COMMIT or reconciliation["reconciliation_tooling_base_commit"] != INITIAL_CARRIER_COMMIT:
        fail("reconciliation source/carrier differs")
    if reconciliation["campaign_manifest_id"] != CAMPAIGN_MANIFEST_ID or reconciliation["campaign_completion_id"] != CAMPAIGN_COMPLETION_ID or reconciliation["candidate_index_id"] != index["index_id"] or reconciliation["custody_id"] != custody["custody_id"] or reconciliation["common_noise_audit_id"] != noise["audit_id"]:
        fail("reconciliation evidence references differ")
    if reconciliation["homogeneity"] != {"result": "PASS", "candidate_count": len(CANDIDATES), "only_intended_candidate_field": "lambda", "unexpected_differences": []}:
        fail("reconciliation homogeneity result differs")
    if reconciliation["factual_candidate_measurements"] != index["factual_measurements"]:
        fail("reconciliation factual measurements differ")
    if reconciliation["worker_custody"] != {"hostname": "confessor", "checkpoint_count": EXPECTED_EPOCHS * len(CANDIDATES), "expected_checkpoint_count": EXPECTED_EPOCHS * len(CANDIDATES), "bytes_preserved_on_worker": True}:
        fail("reconciliation custody result differs")
    protected_expected = {"g8_reruns": 0, "f1_reruns": 0, "f2_optimizer_steps": 0, "f3_reruns": 0, "pass_one_reruns": 0, "pass_two_reruns": 0, "pass_three": 0, "bler_regeneration": 0, "g4_adjudications": 0, "lambda_core_updated": False, "lambda_decision": "NOT_PERFORMED", "lambda_status": "provisional_until_G-4", "w8_final_training_runs": 0, "w8_state": "UNOPENED", "test_model_facing_access": 0, "learned_test_inference": 0, "test_state": "SEALED"}
    if reconciliation["protected_boundary"] != protected_expected:
        fail("reconciliation protected boundary differs")
    if reconciliation["no_model_facing_recomputation"] is not True or reconciliation["decision_boundary"] != "return_for_hostile_audit_then_separate_W7_C_authorization":
        fail("reconciliation decision boundary differs")
    operational = reconciliation["operational_closeout"]
    expect_keys(operational, {"heartbeat_path", "heartbeat_file_sha256", "heartbeat", "process_state", "campaign_process_absent", "tmux_w7_g4_absent", "global_campaign_lock", "worker_source_worktree_clean", "monitor_is_operational_only", "worker_runtime_mutated"}, "operational closeout")
    if operational["heartbeat"] != {"artifact_role": "W7_OPERATIONAL_HEARTBEAT", "campaign_id": CAMPAIGN_ID, "checkpoint_id": None, "current_epoch": None, "current_lambda": None, "process_state": "COMPLETE_NOT_ADJUDICATED", "schema_version": 1, "updated_at_utc": "2026-08-30T00:30:58Z"} or operational["heartbeat_file_sha256"] != "bbee63b79a3a6546f1ec0ff14cf7d40320dd96758e0e73908969a78c70fd6512" or operational["process_state"] != "COMPLETE_NOT_ADJUDICATED" or operational["campaign_process_absent"] is not True or operational["tmux_w7_g4_absent"] is not True or operational["global_campaign_lock"] != "FREE" or operational["worker_source_worktree_clean"] is not True or operational["monitor_is_operational_only"] is not True or operational["worker_runtime_mutated"] is not False:
        fail("operational closeout differs")
    upstream = reconciliation["upstream_reauthentication"]
    expected_upstream = {
        "w7_b1": {"status": "PASS", "command": "tools/verify_w7_b1.py verify", "authorization_id": AUTHORIZATION_ID, "source_manifest_id": SOURCE_MANIFEST_ID},
        "w7_a": {"status": "PASS", "command": "tools/verify_w7_a.py --no-upstream", "completion_id": W7_A_COMPLETION_ID, "test_hardening_completion_id": W7_TEST_HARDENING_ID},
        "w5": {"status": "PASS", "command": "tools/verify_w5_training_system.py", "completion_id": W5_COMPLETION_ID},
        "w6": {"status": "PASS", "command": "tools/verify_w6_complete.py", "completion_id": W6_COMPLETION_ID, "test": "SEALED", "pass_one": 1, "pass_two": 1, "pass_three": 0},
        "pascal_profile": {"status": "PASS", "command": "tools/verify_w7_profile.py results/learned/w7/w7_pascal_profile.json", "profile_id": PROFILE_VERIFIER_ID},
    }
    if upstream != expected_upstream:
        fail("upstream reauthentication record differs")


def verify_completion(completion: dict[str, Any], index_result: dict[str, Any], custody: dict[str, Any], noise: dict[str, Any], reconciliation: dict[str, Any]) -> None:
    scan_for_forbidden_keys(completion)
    index = index_result["index"]
    if completion["status"] != "COMPLETE_NOT_YET_ADJUDICATED" or completion["campaign_id"] != CAMPAIGN_ID or completion["candidate_count"] != len(CANDIDATES) or completion["complete_candidate_count"] != len(CANDIDATES) or completion["completed_epoch_cycles"] != EXPECTED_EPOCHS * len(CANDIDATES):
        fail("B2R completion status/count differs")
    if completion["candidate_index_id"] != index["index_id"] or completion["custody_id"] != custody["custody_id"] or completion["common_noise_audit_id"] != noise["audit_id"] or completion["reconciliation_id"] != reconciliation["reconciliation_id"]:
        fail("B2R completion evidence references differ")
    if completion["worker_campaign_completion_id"] != CAMPAIGN_COMPLETION_ID or completion["worker_campaign_completion_file_sha256"] != CAMPAIGN_COMPLETION_SHA256:
        fail("B2R completion worker reference differs")
    refs = [{"lambda": candidate["lambda"], "candidate_id": candidate["worker_candidate_id"], "candidate_completion_file_sha256": candidate["candidate_completion"]["file_sha256"]} for candidate in index["candidates"]]
    if completion["candidate_references"] != refs:
        fail("B2R completion candidate references differ")
    if completion["g4_adjudication_run"] != 0 or completion["lambda_decision"] != "NOT_PERFORMED" or completion["lambda_core_updated"] is not False or completion["lambda_status"] != "provisional_until_G-4" or completion["w8_final_training_runs"] != 0 or completion["w8_state"] != "UNOPENED" or completion["test_model_facing_access"] != 0 or completion["learned_test_inference"] != 0 or completion["test_state"] != "SEALED":
        fail("B2R completion protected boundary differs")
    if completion["source_commit"] != SOURCE_COMMIT or completion["execution_profile_id"] != W7_PROFILE_ID or completion["gpu_uuid"] != W7_SELECTED_GPU_UUID or completion["no_scientific_execution_performed_by_reconciliation"] is not True:
        fail("B2R completion source/boundary differs")


def map_worker_path(path: str, remote_root: Path, metadata_root: Path) -> Path:
    try:
        return metadata_root / Path(path).relative_to(remote_root)
    except ValueError:
        fail(f"worker path escapes root: {path}")
    raise AssertionError("unreachable")


def verify_worker_metadata(index_result: dict[str, Any], metadata_root: Path) -> None:
    if not metadata_root.is_dir():
        fail(f"worker metadata root is not a directory: {metadata_root}")
    index = index_result["index"]
    remote_root = Path(index["candidates"][0]["candidate_root"]).parent
    def compare(path: str, expected_sha: str, expected_value: Any) -> None:
        local = map_worker_path(path, remote_root, metadata_root)
        if file_sha256(local) != expected_sha or json.loads(local.read_bytes()) != expected_value:
            fail(f"worker metadata mismatch: {local}")
    manifest = index["campaign_manifest"]
    compare(manifest["worker_path"], manifest["file_sha256"], manifest["value"])
    campaign = index["campaign_completion"]
    compare(campaign["worker_path"], campaign["file_sha256"], campaign["value"])
    for candidate in index["candidates"]:
        ref = candidate["candidate_completion"]
        compare(ref["path"], ref["file_sha256"], ref["value"])
        compare(candidate["latest"]["path"], candidate["latest"]["file_sha256"], candidate["latest"]["value"])
        result = candidate["selected"]["selected_checkpoint_result"]
        compare(result["path"], result["file_sha256"], result["value"])
        for epoch, checkpoint in enumerate(candidate["checkpoints"]):
            sidecar_local = map_worker_path(checkpoint["sidecar_path"], remote_root, metadata_root)
            if file_sha256(sidecar_local) != checkpoint["sidecar_sha256"] or json.loads(sidecar_local.read_bytes()) != checkpoint["sidecar"]:
                fail(f"worker sidecar mismatch: {sidecar_local}")
            epoch_local = map_worker_path(checkpoint["epoch_record_path"], remote_root, metadata_root)
            if file_sha256(epoch_local) != checkpoint["epoch_record_sha256"]:
                fail(f"worker epoch-record file mismatch: {epoch_local}")
            epoch_value = json.loads(epoch_local.read_bytes())
            record_id = epoch_value.get("record_id")
            if record_id != checkpoint["epoch_record_id"] or record_id != canonical_sha256({key: item for key, item in epoch_value.items() if key != "record_id"}):
                fail(f"worker epoch-record content mismatch: {epoch_local}")
            validation_ref = candidate["validation_summaries"][epoch]
            compare(validation_ref["path"], validation_ref["file_sha256"], validation_ref["summary"])


def verify_worker_checkpoint_bytes(index_result: dict[str, Any], worker_root: Path) -> int:
    if not worker_root.is_dir():
        fail(f"worker checkpoint root is not a directory: {worker_root}")
    index = index_result["index"]
    remote_root = Path(index["candidates"][0]["candidate_root"]).parent
    count = 0
    for candidate in index["candidates"]:
        for checkpoint in candidate["checkpoints"]:
            local = map_worker_path(checkpoint["path"], remote_root, worker_root)
            if local.stat().st_size != checkpoint["checkpoint_bytes"] or file_sha256(local) != checkpoint["checkpoint_id"]:
                fail(f"worker checkpoint bytes mismatch: {local}")
            count += 1
    if count != EXPECTED_EPOCHS * len(CANDIDATES):
        fail("worker checkpoint byte count differs")
    return count


def run_upstream() -> None:
    commands = [
        [sys.executable, "tools/verify_w7_b1.py", "verify"],
        [sys.executable, "tools/verify_w7_a.py", "--no-upstream"],
        [sys.executable, "tools/verify_w5_training_system.py"],
        [sys.executable, "tools/verify_w6_complete.py"],
        [sys.executable, "tools/verify_w7_profile.py", "results/learned/w7/w7_pascal_profile.json"],
    ]
    for command in commands:
        try:
            subprocess.run(command, cwd=REPO, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            fail(f"upstream verifier failed: {' '.join(command)}: {exc}")


def verify_artifacts(
    artifact_dir: Path = REPO / "results/learned/w7",
    *,
    reauthenticate_upstream: bool = False,
    worker_metadata_root: Path | None = None,
    worker_root: Path | None = None,
) -> dict[str, Any]:
    if reauthenticate_upstream:
        run_upstream()
    index = load_artifact(artifact_dir / "w7_b2_reconciliation_index.json", INDEX_KEYS, INDEX_ROLE, "index_id", "w7b2rindex-")
    custody = load_artifact(artifact_dir / "w7_b2_checkpoint_custody.json", CUSTODY_KEYS, CUSTODY_ROLE, "custody_id", "w7b2rcustody-")
    noise = load_artifact(artifact_dir / "w7_b2_common_noise_audit.json", NOISE_KEYS, NOISE_ROLE, "audit_id", "w7b2rnoise-")
    reconciliation = load_artifact(artifact_dir / "w7_b2_reconciliation.json", RECONCILIATION_KEYS, RECONCILIATION_ROLE, "reconciliation_id", "w7b2rreconciliation-")
    completion = load_artifact(artifact_dir / "w7_b2_completion.json", COMPLETION_KEYS, COMPLETION_ROLE, "completion_id", "w7b2rcompletion-")
    index_result = verify_index(index)
    verify_custody(custody, index_result)
    verify_noise(noise, index_result)
    verify_reconciliation(reconciliation, index_result, custody, noise, completion)
    verify_completion(completion, index_result, custody, noise, reconciliation)
    if worker_metadata_root is not None:
        verify_worker_metadata(index_result, worker_metadata_root.resolve())
    checkpoint_count = None
    if worker_root is not None:
        checkpoint_count = verify_worker_checkpoint_bytes(index_result, worker_root.resolve())
    return {
        "status": "PASS",
        "campaign_id": CAMPAIGN_ID,
        "candidate_count": len(CANDIDATES),
        "completed_epoch_cycles": EXPECTED_EPOCHS * len(CANDIDATES),
        "checkpoint_count": checkpoint_count if checkpoint_count is not None else EXPECTED_EPOCHS * len(CANDIDATES),
        "index_id": index["index_id"],
        "custody_id": custody["custody_id"],
        "common_noise_audit_id": noise["audit_id"],
        "reconciliation_id": reconciliation["reconciliation_id"],
        "completion_id": completion["completion_id"],
        "g4_adjudication_run": 0,
        "lambda_decision": "NOT_PERFORMED",
        "lambda_core_updated": False,
        "w8_final_training_runs": 0,
        "test_model_facing_access": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify",))
    parser.add_argument("--artifact-dir", type=Path, default=REPO / "results/learned/w7")
    parser.add_argument("--worker-metadata-root", type=Path)
    parser.add_argument("--worker-root", type=Path)
    parser.add_argument("--skip-upstream", action="store_true", help="skip upstream verifiers; intended only for isolated unit tests")
    args = parser.parse_args(argv)
    result = verify_artifacts(args.artifact_dir, reauthenticate_upstream=not args.skip_upstream, worker_metadata_root=args.worker_metadata_root, worker_root=args.worker_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
