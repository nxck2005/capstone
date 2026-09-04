#!/usr/bin/env python3
"""Read-only W8-C terminal reconciliation verifier.

This verifier has deliberately no torch/model/data-loader dependency.  Live
verification authenticates the frozen worker namespace by JSON identities,
transaction hashes, checkpoint byte lengths and SHA-256 values.  It never
loads a checkpoint payload and never performs inference.  The compact
reconciliation record can subsequently be checked without worker custody.

Examples::

    python tools/verify_w8_c.py verify-live \
        --campaign-root /home/nick/w8-final-pascal-20260901-r1 \
        --authority-dir /home/nick/w8-r1-artifacts \
        --source-dir /home/nick/w8-r1-source \
        --inventory-output /tmp/w8-c-root-inventory.jsonl \
        --report-output /tmp/w8-c-live-report.json --check-custody

    python tools/verify_w8_c.py verify-compact \
        --reconciliation results/learned/w8/w8_c_reconciliation.json \
        --inventory results/learned/w8/w8_c_root_inventory.jsonl
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


SOURCE_COMMIT = "d52d85dd60bac0c816a7ba249e4453045723277b"
CAMPAIGN_ID = "w8-final-pascal-20260901-r1"
CAMPAIGN_ROOT = "/home/nick/w8-final-pascal-20260901-r1"
HEARTBEAT_PATH = "/home/nick/w8-final-pascal-20260901-r1.heartbeat.json"
STDOUT_PATH = "/home/nick/w8-final-pascal-20260901-r1.stdout.log"
GLOBAL_LOCK = "/tmp/capstone-w8-final-global.lock"
SOURCE_MANIFEST_ID = "w8source-bae4957e4945e812425cebd7f39994e10bada40cef4cee2f36deb812ca6d6eac"
SOURCE_MANIFEST_SHA256 = "bd6bc5061ab6df7e234e57939f4fb0a9d9c922ebea61adc8aa666d03ec18ebc7"
EXECUTION_AUTHORIZATION_ID = "w8auth-45373bb106eb140ce9ab6990f57b20e85c77907e1c50ddc164c4a43229340763"
EXECUTION_AUTHORIZATION_SHA256 = "d2511a7db17ceb10eeb911faf5418a0be4b46965155b6d63d1b512c5b143821c"
LAUNCH_AUTHORIZATION_ID = "w8blaunch-a7a8c92cf98acea0f5d55c137c4319b188fa7ec486921c1849ec761e6f3a85fa"
LAUNCH_AUTHORIZATION_SHA256 = "5a4a513a43cbb0c2d0992577074a4f61d1a64af98459c2e06b0f925b53d750a2"
LINEAGE_ID = "w8lineage-c3587b9c1b72739d906ebb10d14db732c30910e5acae8efa58af9996bc9e0954"
LINEAGE_SHA256 = "2cd83b0de8f52c1f48a0c380243038d3fc316ddc129fa9e68bb35fee220046e8"
GPU_UUID = "GPU-00214b86-48e7-fcf0-bf46-575fa7f85b6b"
PROFILE_ID = "confessor_pascal_cu126"
GPU_NAME = "NVIDIA GeForce GTX 1080 Ti"
EXECUTION_IMAGE = "pascal-cu126-requirements-pascal-lock-v1"
PASCAL_LOCK_SHA256 = "d3561c8e930797d328cf45df1bdc5085842833665f9b5c9e5617d4c891e31a82"
OLD_CAMPAIGN_ID = "w8-final-pascal-20260831"
OLD_INVENTORY_SHA256 = "da6f18be2545d59298c64cc890aa4842b436de1e7e6fc44dc6e22dd6f64cf4aa"
OLD_PARTIAL_CHECKPOINT_SHA256 = "ff89322a795e437994ff5eaaf5c7157fd7e751aed8bafb2f42cee950d371b55c"
OLD_INCIDENT_ID = "w8b2incident-feb598220d1a32143944e1d7a343fff00de43387e255d92c033289e4afcde8c2"
OLD_INCIDENT_SHA256 = "f244c48b237f9b5efbe5875653d6111d3ca5902a173451fe7638f16cd752a4c8"

EPOCHS = 100  # literal-ok: owner-frozen W8 six-run schedule
RUNS = 6  # literal-ok: owner-frozen W8 six-cell matrix
SAMPLES = 8469  # literal-ok: committed Imagenette-160 training denominator
MICROBATCHES = 265  # literal-ok: ceil(8469 / 32), frozen physical batching
FINAL_PHYSICAL_BATCH = 21  # literal-ok: 8469 modulo the physical batch
VALIDATION_TOTAL = 1000  # literal-ok: committed Imagenette-160 validation denominator
PHYSICAL_BATCH = 32  # literal-ok: owner-frozen Pascal effective batch
TRAIN_SNR_DB = 7  # literal-ok: frozen W8 training/checkpoint selection SNR
LAMBDA_CORE = 3.0  # literal-ok: frozen W7 G-4 primary-tier lambda

RUN_CELLS: tuple[dict[str, Any], ...] = (
    {"run_index": 1, "ratio": "r_1_6", "train_seed": 0, "channel_seed": 0, "k": 12800},
    {"run_index": 2, "ratio": "r_1_24", "train_seed": 0, "channel_seed": 0, "k": 3200},
    {"run_index": 3, "ratio": "r_1_6", "train_seed": 1, "channel_seed": 1, "k": 12800},
    {"run_index": 4, "ratio": "r_1_24", "train_seed": 1, "channel_seed": 1, "k": 3200},
    {"run_index": 5, "ratio": "r_1_6", "train_seed": 2, "channel_seed": 2, "k": 12800},
    {"run_index": 6, "ratio": "r_1_24", "train_seed": 2, "channel_seed": 2, "k": 3200},
)

EXPECTED_RUN_DIRS = tuple(
    f"run-{cell['run_index']:02d}-{cell['ratio']}-train{cell['train_seed']}-channel{cell['channel_seed']}"
    for cell in RUN_CELLS
)
EXPECTED_RUN_IDS = tuple(
    f"w8-{cell['ratio']}-train{cell['train_seed']}-channel{cell['channel_seed']}"
    for cell in RUN_CELLS
)

AUTHORITY_FILES = {
    "source_manifest": (("w8_r1_source_manifest.json", "w8_source_manifest.json"), "manifest_id", "w8source-", SOURCE_MANIFEST_ID, SOURCE_MANIFEST_SHA256),
    "execution_authorization": (("w8_r1_execution_authorization.json", "w8_execution_authorization.json"), "authorization_id", "w8auth-", EXECUTION_AUTHORIZATION_ID, EXECUTION_AUTHORIZATION_SHA256),
    "launch_authorization": (("w8_r1_launch_authorization.json", "w8_b_launch_authorization.json"), "authorization_id", "w8blaunch-", LAUNCH_AUTHORIZATION_ID, LAUNCH_AUTHORIZATION_SHA256),
    "lineage": (("w8_r1_successor_lineage.json",), "lineage_id", "w8lineage-", LINEAGE_ID, LINEAGE_SHA256),
}


class W8CHold(RuntimeError):
    """Fail-closed W8-C discrepancy."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise W8CHold(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise W8CHold(f"cannot hash {path}: {exc}") from None
    return digest.hexdigest()


def _safe_file(path: Path, label: str) -> None:
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing or unsafe: {path}")


def read_json(path: Path, label: str, *, canonical: bool = True) -> tuple[dict[str, Any], str]:
    _safe_file(path, label)
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise W8CHold(f"{label} is unreadable: {exc}") from None
    _require(isinstance(value, dict), f"{label} is not a JSON object")
    if canonical:
        _require(raw == canonical_bytes(value), f"{label} is not canonical immutable JSON")
    return value, _sha_bytes(raw)


def _full_sha(value: Any, width: int = 64) -> None:
    _require(
        isinstance(value, str)
        and len(value) == width
        and all(character in "0123456789abcdef" for character in value),
        f"invalid SHA-256/SHA-1 value: {value!r}",
    )


def _identified(value: dict[str, Any], field: str, prefix: str, label: str) -> str:
    _require(field in value, f"{label} lacks {field}")
    identifier = value[field]
    body = dict(value)
    body.pop(field)
    _require(identifier == prefix + canonical_sha256(body), f"{label} {field} does not reproduce")
    return str(identifier)


def _record_id(value: dict[str, Any], label: str) -> str:
    _require("record_id" in value, f"{label} lacks record_id")
    identifier = value["record_id"]
    body = dict(value)
    body.pop("record_id")
    _require(identifier == canonical_sha256(body), f"{label} record_id does not reproduce")
    _full_sha(identifier)
    return str(identifier)


def _expected_role_eligibility(role: str) -> dict[str, str]:
    if role == "W8_FINAL_MULTI_SEED_RUN":
        return {
            "artifact_role": role,
            "g10_eligibility": "NOT_ELIGIBLE_UNTIL_W8_RECONCILIATION",
            "reporting_eligibility": "NOT_ELIGIBLE_UNTIL_W8_RECONCILIATION",
            "scientific_status": "SCIENTIFIC_W8_FINAL_TRAINING",
            "selection_eligibility": "PER_RUN_VALIDATION_ONLY",
            "test_eligibility": "NOT_ELIGIBLE_FOR_TEST",
            "w8_eligibility": "ELIGIBLE_FOR_W8_FINAL_TRAINING_ONLY",
        }
    return {
        "artifact_role": "W8_SELECTED_CHECKPOINT",
        "g10_eligibility": "NOT_ELIGIBLE_UNTIL_W8_RECONCILIATION",
        "reporting_eligibility": "NOT_ELIGIBLE_UNTIL_W8_RECONCILIATION",
        "scientific_status": "SCIENTIFIC_W8_FINAL_TRAINING",
        "selection_eligibility": "ELIGIBLE_FOR_PER_RUN_PRE_TEST_VALIDATION_ONLY",
        "test_eligibility": "NOT_ELIGIBLE_FOR_TEST",
        "w8_eligibility": "SELECTED_W8_CHECKPOINT_PENDING_RECONCILIATION",
    }


def _run_name(cell: dict[str, Any]) -> str:
    return f"run-{cell['run_index']:02d}-{cell['ratio']}-train{cell['train_seed']}-channel{cell['channel_seed']}"


def _expected_profile() -> dict[str, Any]:
    return {
        "execution_profile_id": PROFILE_ID,
        "gpu_uuid": GPU_UUID,
        "gpu_name": GPU_NAME,
        "device": "cuda:0",
        "execution_image_family": EXECUTION_IMAGE,
        "requirements_lock": "requirements-pascal.lock",
        "requirements_lock_sha256": PASCAL_LOCK_SHA256,
        "physical_batch_size": PHYSICAL_BATCH,
        "effective_batch_size": PHYSICAL_BATCH,
        "accumulation_factor": 1,
        "validation_batch_size": PHYSICAL_BATCH,
        "train_samples": SAMPLES,
        "drop_last": False,
        "scientific_writer_host": "confessor",
    }


def verify_authorities(authority_dir: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    file_bindings: dict[str, dict[str, Any]] = {}
    for name, (filenames, field, prefix, expected_id, expected_sha) in AUTHORITY_FILES.items():
        candidates = [authority_dir / filename for filename in filenames if (authority_dir / filename).is_file()]
        _require(candidates, f"W8 {name} authority is missing")
        path = candidates[0]
        value, file_sha = read_json(path, f"W8 {name} authority")
        if file_sha != expected_sha and len(candidates) > 1:
            for candidate in candidates[1:]:
                candidate_value, candidate_sha = read_json(candidate, f"W8 {name} authority candidate")
                if candidate_sha == expected_sha:
                    path, value, file_sha = candidate, candidate_value, candidate_sha
                    break
        _require(file_sha == expected_sha, f"W8 {name} authority SHA differs")
        _require(value.get(field) == expected_id, f"W8 {name} authority ID differs")
        _identified(value, field, prefix, f"W8 {name} authority")
        values[name] = value
        file_bindings[name] = {"path": str(path), "id": expected_id, "sha256": file_sha}

    source = values["source_manifest"]
    auth = values["execution_authorization"]
    launch = values["launch_authorization"]
    lineage = values["lineage"]
    _require(source["artifact_role"] == "W8_SCIENTIFIC_SOURCE_MANIFEST", "W8 source manifest role differs")
    _require(source["source_commit"] == SOURCE_COMMIT, "W8 source manifest source differs")
    _require(source["entry_count"] == 53, "W8 source manifest entry count differs")  # literal-ok: frozen source manifest cardinality
    _require(source["runtime_root_included"] is False and source["scientific_w8_results_included"] is False, "W8 source manifest includes runtime/results")
    _require(source["test_access"] == 0, "W8 source manifest claims test access")
    _require(auth["artifact_role"] == "W8_EXECUTION_AUTHORIZATION" and auth["status"] == "FROZEN_PRE_EXECUTION", "W8 execution authority role/status differs")
    _require(auth["authorization_scope"] == "W8_SIX_CORE_RUNS_ONLY" and auth["scientific_execution_authorized"] == "SIX_CORE_RUNS_ONLY", "W8 execution scope differs")
    _require(auth["source_contains_no_w8_results"] is True, "W8 execution authority claims source results")
    _require(auth["scientific_source"]["source_commit"] == SOURCE_COMMIT, "W8 execution authority source differs")
    _require(auth["scientific_source"]["source_manifest_id"] == SOURCE_MANIFEST_ID, "W8 execution authority manifest ID differs")
    _require(auth["scientific_source"]["source_manifest_file_sha256"] == SOURCE_MANIFEST_SHA256, "W8 execution authority manifest SHA differs")
    _require(auth["campaign"]["campaign_id"] == CAMPAIGN_ID and auth["campaign"]["campaign_root"] == CAMPAIGN_ROOT, "W8 authority campaign differs")
    _require(auth["campaign"]["heartbeat_path"] == HEARTBEAT_PATH and auth["campaign"]["stdout_log_path"] == STDOUT_PATH, "W8 authority operational paths differ")
    _require(auth["campaign"]["run_count"] == RUNS and auth["campaign"]["run_cells"] == list(RUN_CELLS), "W8 authority six-cell matrix differs")
    _require(auth["campaign"]["order_rule"] == "seed_major_then_ratio_minor", "W8 authority run order differs")
    _require(auth["training"]["lambda"] == LAMBDA_CORE and auth["training"]["train_snr_db"] == TRAIN_SNR_DB and auth["training"]["epochs_per_run"] == EPOCHS, "W8 training authority differs")
    _require(auth["training"]["fresh_initialization"]["predecessor_checkpoint_id"] is None, "W8 authority genesis predecessor differs")
    _require(auth["training"]["w7_checkpoint_transfer_forbidden"] is True and auth["training"]["prior_w8_state_transfer_forbidden"] is True, "W8 state-transfer authority differs")
    _require(auth["checkpoint_selection"] == {
        "channel_seed_rule": "run_channel_seed",
        "cross_seed_selection": False,
        "fixed_noise_across_epochs": True,
        "forbidden_inputs": ["psnr", "papr", "reconstruction_loss"],
        "full_validation_every_completed_epoch": True,
        "metric": "validation_top1_accuracy",
        "mode": "max",
        "snr_db": TRAIN_SNR_DB,
        "snr_parameter": "params.learned_system.checkpoint_selection_snr_db",
        "snr_resolution": "params.channel.train_snr_db_fixed",
        "split": "validation",
        "tie_break": "earliest_epoch",
        "validation_denominator": VALIDATION_TOTAL,
    }, "W8 checkpoint-selection authority differs")
    _require(launch["artifact_role"] == "W8_B_LAUNCH_AUTHORIZATION" and launch["status"] == "AUTHORIZED", "W8 launch authority role/status differs")
    _require(launch["owner_authorization"] is True and launch["authorization_scope"] == "W8_SIX_CORE_RUNS_ONLY", "W8 launch authorization scope differs")
    _require(launch["campaign_id"] == CAMPAIGN_ID and launch["campaign_root"] == CAMPAIGN_ROOT, "W8 launch campaign differs")
    _require(launch["source_commit"] == SOURCE_COMMIT and launch["source_manifest_id"] == SOURCE_MANIFEST_ID and launch["source_manifest_sha256"] == SOURCE_MANIFEST_SHA256, "W8 launch source differs")
    _require(launch["profile"] == {
        "execution_profile_id": PROFILE_ID,
        "gpu_name": GPU_NAME,
        "gpu_uuid": GPU_UUID,
        "device": "cuda:0",
        "requirements_lock": "requirements-pascal.lock",
        "requirements_lock_sha256": PASCAL_LOCK_SHA256,
        "physical_batch_size": PHYSICAL_BATCH,
        "effective_batch_size": PHYSICAL_BATCH,
        "accumulation_factor": 1,
        "validation_batch_size": PHYSICAL_BATCH,
    }, "W8 launch profile differs")
    _require(launch["scope"] == {"core_runs": RUNS, "er2_randomized_training": False, "er9_training": False, "g10": False, "papr_constrained_training": False}, "W8 launch scope boundary differs")
    _require(launch["test"] == {"learned_inference": 0, "model_facing_access": 0, "status": "SEALED"}, "W8 launch test boundary differs")
    _require(launch["w8_a_authorization_id"] == EXECUTION_AUTHORIZATION_ID and launch["w8_a_authorization_sha256"] == EXECUTION_AUTHORIZATION_SHA256, "W8 launch execution-authority binding differs")
    _require(lineage["artifact_role"] == "W8_R1_SUCCESSOR_LINEAGE" and lineage["status"] == "IMMUTABLE_PROVENANCE_ONLY" and lineage["provenance_only"] is True, "W8 successor lineage role/status differs")
    _require(lineage["scientific_execution_started"] is False and lineage["test_access"] == 0, "W8 successor lineage boundary differs")
    predecessor = lineage["predecessor"]
    _require(predecessor["campaign_id"] == OLD_CAMPAIGN_ID and predecessor["historical_failed_optimizer_steps"] == 259, "W8 failed predecessor identity differs")  # literal-ok: immutable incident history
    _require(predecessor["partial_checkpoint_sha256"] == OLD_PARTIAL_CHECKPOINT_SHA256 and predecessor["incident_id"] == OLD_INCIDENT_ID and predecessor["incident_file_sha256"] == OLD_INCIDENT_SHA256, "W8 failed predecessor evidence differs")
    _require(predecessor["historical_accepted_coverage"] == 0 and predecessor["resume_eligible"] is False and predecessor["result_eligible"] is False and predecessor["g10_eligible"] is False and predecessor["test_eligible"] is False, "W8 failed predecessor eligibility differs")
    successor = lineage["successor"]
    _require(successor["campaign_id"] == CAMPAIGN_ID and successor["campaign_root"] == CAMPAIGN_ROOT and successor["source_commit"] == SOURCE_COMMIT, "W8 successor lineage campaign differs")
    _require(successor["execution_authorization_id"] == EXECUTION_AUTHORIZATION_ID and successor["execution_authorization_sha256"] == EXECUTION_AUTHORIZATION_SHA256, "W8 successor lineage authority differs")
    _require(successor["predecessor_checkpoint_id"] is None and successor["scientific_coverage"] == 0 and successor["initialization"] == "fresh deterministic genesis", "W8 successor genesis differs")
    return {"values": values, "files": file_bindings}


def verify_source_checkout(source_dir: Path) -> dict[str, Any]:
    _require(source_dir.is_dir() and not source_dir.is_symlink(), f"W8 source checkout is missing or unsafe: {source_dir}")
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source_dir, capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=source_dir, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise W8CHold(f"cannot inspect W8 source checkout: {exc}") from None
    _require(head == SOURCE_COMMIT, "W8 scientific source HEAD differs")
    _require(status == "", "W8 scientific source checkout is dirty")
    return {"path": str(source_dir), "head": head, "clean": True}


def _manifest_profile_matches(profile: dict[str, Any]) -> None:
    expected = _expected_profile()
    for field, value in expected.items():
        _require(profile.get(field) == value, f"W8 campaign profile differs: {field}")


def verify_campaign_manifest(manifest: dict[str, Any], authority: dict[str, Any]) -> None:
    _identified(manifest, "manifest_id", "w8campaignmanifest-", "W8 campaign manifest")
    _require(manifest["artifact_role"] == "W8_CAMPAIGN_MANIFEST" and manifest["schema_version"] == 1, "W8 campaign manifest role/version differs")  # literal-ok: frozen manifest schema
    _require(manifest["status"] == "FROZEN_BEFORE_FIRST_RUN" and manifest["campaign_id"] == CAMPAIGN_ID, "W8 campaign manifest status/ID differs")
    _require(manifest["source_commit"] == SOURCE_COMMIT and manifest["source_manifest_id"] == SOURCE_MANIFEST_ID and manifest["source_manifest_sha256"] == SOURCE_MANIFEST_SHA256, "W8 campaign manifest source differs")
    _require(manifest["authorization_id"] == EXECUTION_AUTHORIZATION_ID and manifest["launch_authorization_id"] == LAUNCH_AUTHORIZATION_ID, "W8 campaign manifest authority differs")
    _manifest_profile_matches(manifest["execution_profile"])
    _require(manifest["run_order"] == "seed_major_then_ratio_minor" and manifest["run_cells"] == list(RUN_CELLS), "W8 campaign manifest six-cell order differs")
    _require(manifest["checkpoint_selection"] == {
        "channel_seed_rule": "run_channel_seed",
        "cross_seed_selection": False,
        "fixed_noise_across_epochs": True,
        "forbidden_inputs": ["psnr", "papr", "reconstruction_loss"],
        "full_validation_every_completed_epoch": True,
        "metric": "validation_top1_accuracy",
        "mode": "max",
        "snr_db": TRAIN_SNR_DB,
        "snr_parameter": "params.learned_system.checkpoint_selection_snr_db",
        "snr_resolution": "params.channel.train_snr_db_fixed",
        "split": "validation",
        "tie_break": "earliest_epoch",
        "validation_denominator": VALIDATION_TOTAL,
    }, "W8 campaign manifest selection policy differs")
    _require(manifest["scientific_scope"]["g10_adjudications"] == 0 and manifest["scientific_scope"]["test_model_facing_access"] == 0, "W8 campaign manifest protected scope differs")
    _require(manifest["test_model_facing_access"] == 0 and manifest["learned_test_inference"] == 0, "W8 campaign manifest claims test access")
    _require(manifest["resume_rule"]["resume_rule"] == "latest_authenticated_completed_epoch_only" and manifest["resume_rule"]["corrupt_latest"] == "HOLD_NO_OLDER_FALLBACK", "W8 campaign resume rule differs")
    _require(authority["values"]["execution_authorization"]["protocol_hash"] == manifest["protocol_hash"], "W8 campaign protocol hash differs from authority")


def verify_selection(selection: dict[str, Any], cell: dict[str, Any], expected_epoch: int, expected_checkpoint_id: str, expected_n_correct: int) -> None:
    _identified(selection, "selection_id", "", "W8 selection")
    _require(selection["artifact_role"] == "W8_SELECTED_CHECKPOINT", "W8 selection role differs")
    _require(selection["eligibility"] == _expected_role_eligibility("W8_SELECTED_CHECKPOINT"), "W8 selection eligibility differs")
    for field, expected in {
        "campaign_id": CAMPAIGN_ID, "run_id": EXPECTED_RUN_IDS[cell["run_index"] - 1], "ratio": cell["ratio"],
        "k": cell["k"], "train_seed": cell["train_seed"], "channel_seed": cell["channel_seed"],
        "metric": "validation_top1_accuracy", "mode": "max", "tie_break": "earliest_epoch",
        "validation_snr_parameter": "params.learned_system.checkpoint_selection_snr_db",
        "validation_snr_resolution": "params.channel.train_snr_db_fixed", "validation_snr_db": TRAIN_SNR_DB,
        "validation_channel_seed_rule": "run_channel_seed", "selected_epoch": expected_epoch,
        "selected_checkpoint_id": expected_checkpoint_id, "n_correct": expected_n_correct,
        "n_total": VALIDATION_TOTAL, "cross_seed_selection": False, "psnr_selected": False,
        "papr_selected": False, "reconstruction_loss_selected": False,
    }.items():
        _require(selection.get(field) == expected, f"W8 selection differs: {field}")
    _require(selection["top1_accuracy"] == expected_n_correct / VALIDATION_TOTAL, "W8 selection top-1 is not count-derived")


def verify_validation_summary(summary: dict[str, Any], cell: dict[str, Any], epoch: int, checkpoint_id: str, previous_noise: str | None) -> tuple[str, int]:
    identifier = summary.get("summary_id")
    _require(isinstance(identifier, str) and identifier == canonical_sha256({key: value for key, value in summary.items() if key != "summary_id"}), "W8 validation summary digest differs")
    _require(summary["artifact_role"] == "W8_VALIDATION_EPOCH_SUMMARY" and summary["schema_version"] == 1, "W8 validation summary role/version differs")  # literal-ok: frozen validation schema
    _require(summary["eligibility"] == _expected_role_eligibility("W8_FINAL_MULTI_SEED_RUN"), "W8 validation summary eligibility differs")
    expected = {
        "campaign_id": CAMPAIGN_ID, "run_id": EXPECTED_RUN_IDS[cell["run_index"] - 1], "ratio": cell["ratio"],
        "k": cell["k"], "train_seed": cell["train_seed"], "channel_seed": cell["channel_seed"],
        "checkpoint_id": checkpoint_id, "epoch": epoch, "validation_split": "val",
        "validation_order": "stable_manifest_order", "validation_augmentation": False,
        "validation_batch_size": PHYSICAL_BATCH, "validation_snr_parameter": "params.learned_system.checkpoint_selection_snr_db",
        "validation_snr_resolution": "params.channel.train_snr_db_fixed", "validation_snr_db": TRAIN_SNR_DB,
        "validation_channel_seed_rule": "run_channel_seed", "validation_channel_seed": cell["channel_seed"],
        "validation_noise_policy": "keyed_per_image_fixed_snr_run_channel_seed_same_across_epochs",
        "validation_noise_id_count": VALIDATION_TOTAL, "n_total": VALIDATION_TOTAL,
        "forbidden_selection_inputs": ["psnr", "papr", "reconstruction_loss"], "test_model_facing_access": 0,
    }
    for field, value in expected.items():
        _require(summary.get(field) == value, f"W8 validation summary differs: {field}")
    _require(isinstance(summary["n_correct"], int) and not isinstance(summary["n_correct"], bool) and 0 <= summary["n_correct"] <= VALIDATION_TOTAL, "W8 validation correct count is invalid")
    _require(summary["top1_accuracy"] == summary["n_correct"] / VALIDATION_TOTAL, "W8 validation top-1 is not count-derived")
    noise = summary.get("validation_noise_id_digest")
    _full_sha(noise)
    if previous_noise is not None:
        _require(noise == previous_noise, "W8 validation noise digest changed within a run")
    return str(noise), int(summary["n_correct"])


def verify_root_inventory(inventory_path: Path, entries: list[tuple[str, int, str]], total_bytes: int, file_count: int) -> dict[str, Any]:
    _safe_file(inventory_path, "W8 root inventory")
    raw = inventory_path.read_bytes()
    actual_lines = raw.splitlines(keepends=True)
    expected_lines = [canonical_bytes([path, size, digest]) for path, size, digest in entries]
    _require(actual_lines == expected_lines, "W8 root inventory entries differ")
    _require(sum(size for _, size, _ in entries) == total_bytes and len(entries) == file_count, "W8 root inventory totals differ")
    inventory_sha = _sha_bytes(raw)
    return {
        "path": str(inventory_path),
        "inventory_id": "w8inventory-" + inventory_sha,
        "sha256": inventory_sha,
        "file_count": file_count,
        "byte_count": total_bytes,
    }


def verify_compact_inventory(inventory_path: Path, expected: dict[str, Any]) -> None:
    """Check the portable inventory's syntax and content-address binding."""

    _safe_file(inventory_path, "W8-C compact inventory")
    raw = inventory_path.read_bytes()
    _require(_sha_bytes(raw) == expected["sha256"], "W8-C compact inventory SHA differs")
    _require(expected["inventory_id"] == "w8inventory-" + expected["sha256"], "W8-C compact inventory ID differs")
    rows: list[list[Any]] = []
    for line in raw.splitlines(keepends=True):
        _require(line.endswith(b"\n"), "W8-C compact inventory is not LF terminated")
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise W8CHold(f"W8-C compact inventory row is unreadable: {exc}") from None
        _require(isinstance(row, list) and len(row) == 3 and canonical_bytes(row) == line, "W8-C compact inventory row is not canonical")
        relative, size, digest = row
        _require(isinstance(relative, str) and relative and not relative.startswith("/") and ".." not in Path(relative).parts, "W8-C compact inventory path is unsafe")
        _require(isinstance(size, int) and not isinstance(size, bool) and size >= 0, "W8-C compact inventory byte count is invalid")
        _full_sha(digest)
        rows.append(row)
    _require([row[0] for row in rows] == sorted(row[0] for row in rows), "W8-C compact inventory is not sorted")
    _require(len(rows) == expected["file_count"] and sum(row[1] for row in rows) == expected["byte_count"], "W8-C compact inventory totals differ")


def _verify_compact_transactions(value: dict[str, Any]) -> None:
    run_ids = [cell["run_id"] if "run_id" in cell else EXPECTED_RUN_IDS[index] for index, cell in enumerate(value["run_identities"])]
    _require(run_ids == list(EXPECTED_RUN_IDS) and len(value["run_identities"]) == RUNS, "W8-C compact run identity set/order differs")  # literal-ok: six-run terminal set
    for index, cell in enumerate(RUN_CELLS):
        identity = value["run_identities"][index]
        for field, expected in {"run_index": cell["run_index"], "run_directory": _run_name(cell), "run_id": EXPECTED_RUN_IDS[index], "ratio": cell["ratio"], "train_seed": cell["train_seed"], "channel_seed": cell["channel_seed"], "k": cell["k"]}.items():
            _require(identity.get(field) == expected, f"W8-C compact run identity differs: {field}")
    per_run = value["transaction_accounting"]["per_run"]
    _require(len(per_run) == RUNS and [item.get("run_id") for item in per_run] == list(EXPECTED_RUN_IDS), "W8-C compact per-run transaction set/order differs")  # literal-ok: six-run terminal set
    total_steps = 0
    total_skips = 0
    total_opportunities = 0
    all_checkpoint_ids: set[str] = set()
    for index, cell in enumerate(RUN_CELLS):
        item = per_run[index]
        _require(item["epoch_records"] == EPOCHS and item["checkpoint_payloads"] == EPOCHS and item["scientific_sidecars"] == EPOCHS and item["validation_summaries"] == EPOCHS, f"W8-C compact {EXPECTED_RUN_IDS[index]} transaction count differs")  # literal-ok: one hundred transactions per run
        transactions = item["transactions"]
        _require(len(transactions) == EPOCHS and [tx.get("epoch") for tx in transactions] == list(range(EPOCHS)), f"W8-C compact {EXPECTED_RUN_IDS[index]} epoch set/order differs")  # literal-ok: zero-based 100-epoch set
        expected_predecessor: str | None = None
        expected_global_step = 0
        run_steps = 0
        run_skips = 0
        run_opportunities = 0
        run_noise: str | None = None
        for tx in transactions:
            _require(tx["predecessor_checkpoint_id"] == expected_predecessor, f"W8-C compact {EXPECTED_RUN_IDS[index]} predecessor chain differs")
            for field in ("checkpoint_payload", "scientific_sidecar", "validation_summary"):
                _require(field in tx, f"W8-C compact {EXPECTED_RUN_IDS[index]} transaction lacks {field}")
            checkpoint = tx["checkpoint_payload"]
            sidecar = tx["scientific_sidecar"]
            validation = tx["validation_summary"]
            _full_sha(checkpoint["checkpoint_id"])
            _require(checkpoint["file_sha256"] == checkpoint["checkpoint_id"] and sidecar["checkpoint_id"] == checkpoint["checkpoint_id"] and validation["checkpoint_id"] == checkpoint["checkpoint_id"], f"W8-C compact {EXPECTED_RUN_IDS[index]} checkpoint binding differs")
            _full_sha(checkpoint["file_sha256"])
            _full_sha(sidecar["file_sha256"])
            _full_sha(validation["file_sha256"])
            _require(validation["n_total"] == VALIDATION_TOTAL and validation["noise_digest"], f"W8-C compact {EXPECTED_RUN_IDS[index]} validation denominator/noise differs")
            _full_sha(validation["noise_digest"])
            if run_noise is None:
                run_noise = validation["noise_digest"]
            else:
                _require(run_noise == validation["noise_digest"], f"W8-C compact {EXPECTED_RUN_IDS[index]} validation noise changed")
            _require(tx["optimizer_step_opportunities"] == MICROBATCHES and tx["optimizer_steps"] + tx["grad_scaler_skips"] == MICROBATCHES, f"W8-C compact {EXPECTED_RUN_IDS[index]} optimizer arithmetic differs")
            _require(tx["global_optimizer_step"] == expected_global_step + tx["optimizer_steps"], f"W8-C compact {EXPECTED_RUN_IDS[index]} global optimizer chain differs")
            run_steps += tx["optimizer_steps"]
            run_skips += tx["grad_scaler_skips"]
            run_opportunities += tx["optimizer_step_opportunities"]
            total_steps += tx["optimizer_steps"]
            total_skips += tx["grad_scaler_skips"]
            total_opportunities += tx["optimizer_step_opportunities"]
            _require(checkpoint["checkpoint_id"] not in all_checkpoint_ids, "W8-C compact checkpoint identity is duplicated")
            all_checkpoint_ids.add(checkpoint["checkpoint_id"])
            expected_predecessor = checkpoint["checkpoint_id"]
            expected_global_step = tx["global_optimizer_step"]
        _require(run_steps == item["optimizer_steps"] and run_skips == item["grad_scaler_skips"] and run_opportunities == item["optimizer_opportunities"], f"W8-C compact {EXPECTED_RUN_IDS[index]} accounting summary differs")
        _require(item["latest_epoch"] == EPOCHS - 1 and item["latest_checkpoint_id"] == expected_predecessor, f"W8-C compact {EXPECTED_RUN_IDS[index]} latest pointer differs")  # literal-ok: zero-based final epoch
        _require(item["validation_summaries"] == EPOCHS, f"W8-C compact {EXPECTED_RUN_IDS[index]} validation count differs")  # literal-ok: one validation summary per epoch
        _require(value["validation_noise"]["per_run"][index]["digest"] == run_noise, f"W8-C compact {EXPECTED_RUN_IDS[index]} noise summary differs")
    _require(total_opportunities == RUNS * EPOCHS * MICROBATCHES and total_steps + total_skips == total_opportunities, "W8-C compact campaign accounting does not close")  # literal-ok: six runs x 100 epochs x 265 opportunities
    _require(value["campaign_optimizer_accounting"]["opportunities"] == total_opportunities and value["campaign_optimizer_accounting"]["applied_optimizer_steps"] == total_steps and value["campaign_optimizer_accounting"]["grad_scaler_skips"] == total_skips, "W8-C compact campaign accounting differs")


def _verify_compact_selection(value: dict[str, Any]) -> None:
    selection = value["selection"]
    _require(selection["metric"] == "validation_top1_accuracy" and selection["mode"] == "max" and selection["tie_break"] == "earliest_epoch" and selection["validation_snr_db"] == TRAIN_SNR_DB and selection["validation_denominator"] == VALIDATION_TOTAL and selection["cross_seed_selection"] is False, "W8-C compact selection policy differs")
    per_run = selection["per_run"]
    _require(len(per_run) == RUNS and [item.get("run_id") for item in per_run] == list(EXPECTED_RUN_IDS), "W8-C compact selection run set/order differs")  # literal-ok: six-run selection set
    transactions = value["transaction_accounting"]["per_run"]
    for index, item in enumerate(per_run):
        independent = item["independently_reconstructed"]
        published = item["runner_published"]
        _require(item["equal"] is True and independent == {"epoch": published["epoch"], "n_correct": independent["n_correct"], "n_total": VALIDATION_TOTAL, "checkpoint_id": published["checkpoint_id"]}, f"W8-C compact selection equality differs for {item['run_id']}")
        epoch = independent["epoch"]
        _require(isinstance(epoch, int) and 0 <= epoch < EPOCHS and independent["n_total"] == VALIDATION_TOTAL, f"W8-C compact selected epoch/denominator differs for {item['run_id']}")
        tx = transactions[index]["transactions"][epoch]
        _require(independent["checkpoint_id"] == tx["checkpoint_payload"]["checkpoint_id"] and published["checkpoint_id"] == tx["checkpoint_payload"]["checkpoint_id"] and independent["n_correct"] == tx["validation_summary"]["n_correct"] and independent["n_total"] == tx["validation_summary"]["n_total"], f"W8-C compact selected checkpoint/validation differs for {item['run_id']}")
        _full_sha(published["result_id"])
        _full_sha(published["result_file_sha256"])
        _require(published["epoch"] == epoch and published["result_id"] and published["result_file_sha256"], f"W8-C compact published selection metadata differs for {item['run_id']}")
    _require(selection["equality_count"] == RUNS, "W8-C compact selection equality count differs")  # literal-ok: six selected models


def _verify_compact_initialization(value: dict[str, Any]) -> None:
    init = value["fresh_initialization"]
    _require(init["verified_runs"] == RUNS and init["total_runs"] == RUNS and init["cross_train_seed_initialization_ids_differ"] is True, "W8-C compact initialization summary differs")  # literal-ok: six fresh initializations
    per_run = init["per_run"]
    _require(len(per_run) == RUNS and [item.get("run_id") for item in per_run] == list(EXPECTED_RUN_IDS), "W8-C compact initialization run set/order differs")  # literal-ok: six-run initialization set
    hashes: dict[int, set[str]] = {}
    for item in per_run:
        _require(item["mode"] == "fresh_keyed_init" and item["predecessor_checkpoint_id"] is None and item["optimizer_state_transfer"] is False and item["scheduler_state_transfer"] is False and item["scaler_state_transfer"] is False and item["w7_checkpoint_transfer"] is False and item["prior_w8_state_transfer"] is False, f"W8-C compact fresh initialization differs for {item['run_id']}")
        _full_sha(item["initial_model_state_sha256"])
        seed = item["identity"]["train_seed"]
        hashes.setdefault(seed, set()).add(item["initial_model_state_sha256"])
    _require(len({digest for values in hashes.values() for digest in values}) == RUNS, "W8-C compact initialization identities are not separated")  # literal-ok: six fresh keyed initializations


def _verify_compact_noise_and_boundaries(value: dict[str, Any]) -> None:
    noise = value["validation_noise"]
    _require(noise["denominator"] == VALIDATION_TOTAL and noise["one_digest_per_run"] is True and len(noise["per_run"]) == RUNS, "W8-C compact validation-noise summary differs")
    _require([item.get("run_id") for item in noise["per_run"]] == list(EXPECTED_RUN_IDS), "W8-C compact validation-noise run set/order differs")  # literal-ok: six-run noise set
    for item in noise["per_run"]:
        _full_sha(item["digest"])
        _require(item["epoch_count"] == EPOCHS, f"W8-C compact validation-noise epoch count differs for {item['run_id']}")  # literal-ok: one digest across 100 epochs
    _require(value["monitor_terminal_display"]["classification"] == "MONITOR_PRESENTATION_DEFECT" and value["monitor_terminal_display"]["scope"] == "OPERATIONS ONLY" and value["monitor_terminal_display"]["scientific_impact"] == "ZERO", "W8-C monitor classification differs")


def _inventory_line(path: str, size: int, digest: str) -> bytes:
    return canonical_bytes([path, size, digest])


def build_root_inventory(root: Path, output: Path | None) -> tuple[dict[str, Any], list[tuple[str, int, str]]]:
    _require(root.is_dir() and not root.is_symlink(), f"W8 campaign root is missing or unsafe: {root}")
    entries: list[tuple[str, int, str]] = []

    def visit(directory: Path) -> None:
        try:
            children = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            raise W8CHold(f"cannot enumerate W8 campaign root: {exc}") from None
        for child in children:
            _require(not child.is_symlink(), f"W8 campaign root contains a symlink: {child}")
            if child.is_dir():
                visit(child)
            else:
                _require(child.is_file(), f"W8 campaign root contains a non-regular file: {child}")
                relative = child.relative_to(root).as_posix()
                size = child.stat().st_size
                entries.append((relative, size, sha256_file(child)))

    visit(root)
    entries.sort(key=lambda item: item[0])
    raw = b"".join(_inventory_line(*entry) for entry in entries)
    inventory_sha = _sha_bytes(raw)
    inventory = {
        "path": str(output) if output is not None else None,
        "inventory_id": "w8inventory-" + inventory_sha,
        "sha256": inventory_sha,
        "file_count": len(entries),
        "byte_count": sum(size for _, size, _ in entries),
    }
    if output is not None:
        _publish_bytes(output, raw, "W8 root inventory")
    return inventory, entries


def _publish_bytes(path: Path, raw: bytes, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        _require(path.is_file() and not path.is_symlink() and path.read_bytes() == raw, f"immutable {label} already differs: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def verify_custody(heartbeat_path: Path, stdout_path: Path, lock_path: Path) -> dict[str, Any]:
    heartbeat, heartbeat_sha = read_json(heartbeat_path, "W8 terminal heartbeat")
    _require(heartbeat["artifact_role"] == "W8_OPERATIONAL_HEARTBEAT" and heartbeat["campaign_id"] == CAMPAIGN_ID, "W8 heartbeat identity differs")
    _require(heartbeat["process_state"] == "COMPLETE_NOT_YET_RECONCILED" and heartbeat["completed_runs"] == RUNS and heartbeat["completed_epoch_cycles"] == RUNS * EPOCHS, "W8 heartbeat terminal state differs")  # literal-ok: six times 100 terminal cycles
    _require(heartbeat["total_runs"] == RUNS and heartbeat["current_epoch"] is None and heartbeat["current_run_index"] is None, "W8 heartbeat live-progress terminal fields differ")
    stdout_sha = sha256_file(stdout_path)
    _safe_file(stdout_path, "W8 stdout log")
    _safe_file(lock_path, "W8 global lock")
    try:
        with lock_path.open("rb") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    except (BlockingIOError, OSError) as exc:
        raise W8CHold(f"W8 global lock is not free: {exc}") from None

    process_rows: list[dict[str, Any]] = []
    stale_tmux_rows: list[dict[str, Any]] = []
    try:
        ps = subprocess.run(["ps", "-eo", "pid=,comm=,args="], capture_output=True, text=True, check=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise W8CHold(f"cannot inspect W8 processes: {exc}") from None
    for line in ps:
        fields = line.strip().split(None, 2)
        if len(fields) == 3 and "run_w8_campaign.py" in fields[2]:
            row = {"pid": int(fields[0]), "command": fields[2]}
            # A naturally exited detached run can leave the shared tmux
            # server's original launch argv behind while the W8 session is
            # gone.  It is custody history, not a live runner process.
            if fields[1].startswith("tmux") or fields[2].lstrip().startswith("tmux "):
                stale_tmux_rows.append(row)
            else:
                process_rows.append(row)
    _require(not process_rows, f"W8 campaign process is still present: {process_rows}")
    try:
        tmux = subprocess.run(["tmux", "has-session", "-t", "w8-final-r1"], capture_output=True, text=True)
        tmux_sessions = subprocess.run(["tmux", "list-sessions", "-F", "#{session_name}"], capture_output=True, text=True, check=False).stdout.splitlines()
    except OSError as exc:
        raise W8CHold(f"cannot inspect W8 tmux custody: {exc}") from None
    _require(tmux.returncode != 0 and "w8-final-r1" not in tmux_sessions, "W8 final tmux session still exists")
    try:
        nvidia = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise W8CHold(f"cannot inspect selected GTX CUDA processes: {exc}") from None
    cuda_rows: list[str] = []
    for line in nvidia:
        if line.strip():
            try:
                pid = int(line.split(",", 1)[0].strip())
            except ValueError:
                cuda_rows.append(line.strip())
                continue
            try:
                command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            except OSError:
                command = line.strip()
            if CAMPAIGN_ID in command or "run_w8_campaign.py" in command or CAMPAIGN_ROOT in command:
                cuda_rows.append(f"{pid}: {command}")
    _require(not cuda_rows, f"selected GTX has a W8 CUDA process: {cuda_rows}")
    return {
        "heartbeat": {"path": str(heartbeat_path), "sha256": heartbeat_sha, "state": heartbeat["process_state"], "completed_runs": heartbeat["completed_runs"], "completed_epoch_cycles": heartbeat["completed_epoch_cycles"]},
        "stdout": {"path": str(stdout_path), "sha256": stdout_sha},
        "global_lock": {"path": str(lock_path), "state": "FREE"},
        "w8_processes": [],
        "stale_shared_tmux_server_argv": stale_tmux_rows,
        "w8_final_tmux": {"session_present": False, "sessions": tmux_sessions},
        "selected_gtx_w8_cuda_processes": [],
    }


def _verify_init(init: dict[str, Any], cell: dict[str, Any]) -> None:
    _require(init == dict(init), "W8 initialization is not a JSON object")
    _require(init.get("purpose") == "init" and init.get("mode") == "fresh_keyed_init", "W8 initialization mode differs")
    _require(init.get("identity") == {"train_seed": cell["train_seed"], "component_path": "djscc.djscc_residual_v1"}, "W8 initialization key differs")
    _require(init.get("predecessor_checkpoint_id") is None, "W8 initialization has a predecessor")
    for field in ("optimizer_state_transfer", "scheduler_state_transfer", "scaler_state_transfer", "w7_checkpoint_transfer"):
        _require(init.get(field) is False, f"W8 initialization transfer flag is not false: {field}")
    # The frozen source's fresh_initialization_identity schema predates an
    # explicit prior-W8 key.  Absence is the source-authenticated false value;
    # if a successor writes the key, it must still be false.
    _require(init.get("prior_w8_state_transfer", False) is False, "W8 initialization transfers prior W8 state")
    _full_sha(init.get("initial_model_state_sha256"))


def _verify_lineage(value: dict[str, Any], cell: dict[str, Any], predecessor: str | None) -> None:
    _require(value.get("campaign_id") == CAMPAIGN_ID and value.get("bw_ratio") == cell["ratio"] and value.get("k") == cell["k"], "W8 epoch lineage run identity differs")
    _require(value.get("train_seed") == cell["train_seed"] and value.get("channel_seed") == cell["channel_seed"], "W8 epoch lineage seed pairing differs")
    _require(value.get("source_commit") == SOURCE_COMMIT and value.get("source_manifest_id") == SOURCE_MANIFEST_ID and value.get("source_manifest_sha256") == SOURCE_MANIFEST_SHA256, "W8 epoch lineage source differs")
    _require(value.get("execution_profile_id") == PROFILE_ID and value.get("gpu_uuid") == GPU_UUID, "W8 epoch lineage execution profile differs")
    _require(value.get("checkpoint_selection_snr_db") == TRAIN_SNR_DB and value.get("predecessor_checkpoint_id") == predecessor, "W8 epoch lineage predecessor/SNR differs")
    _require(value.get("initialization", {}).get("predecessor_checkpoint_id") is None, "W8 epoch lineage initialization has predecessor state")


def verify_run(root: Path, cell: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    run_dir = root / _run_name(cell)
    _require(run_dir.is_dir() and not run_dir.is_symlink(), f"W8 run directory is missing: {run_dir.name}")
    expected_top = {"checkpoints", "epochs", "validation", "latest.json", "run_completion.json", "selected_checkpoint.json"}
    _require({entry.name for entry in run_dir.iterdir()} == expected_top, f"W8 run namespace differs: {run_dir.name}")
    for name in ("checkpoints", "epochs", "validation"):
        directory = run_dir / name
        _require(directory.is_dir() and not directory.is_symlink(), f"W8 run directory is unsafe: {directory}")

    run_id = EXPECTED_RUN_IDS[cell["run_index"] - 1]
    run_completion, run_completion_sha = read_json(run_dir / "run_completion.json", f"{run_id} completion")
    _identified(run_completion, "completion_id", "w8runcompletion-", f"{run_id} completion")
    _require(run_completion["artifact_role"] == "W8_FINAL_TRAINING_RUN_COMPLETION" and run_completion["status"] == "COMPLETE" and run_completion["authentication_status"] == "PASSED", f"{run_id} completion status differs")
    _require(run_completion["eligibility"] == _expected_role_eligibility("W8_FINAL_MULTI_SEED_RUN"), f"{run_id} completion eligibility differs")
    for field, expected in {"campaign_id": CAMPAIGN_ID, "run_id": run_id, "ratio": cell["ratio"], "k": cell["k"], "train_seed": cell["train_seed"], "channel_seed": cell["channel_seed"], "lambda": LAMBDA_CORE, "train_snr_db": TRAIN_SNR_DB, "execution_profile_id": PROFILE_ID, "gpu_uuid": GPU_UUID, "source_commit": SOURCE_COMMIT, "source_manifest_id": SOURCE_MANIFEST_ID, "source_manifest_sha256": SOURCE_MANIFEST_SHA256, "epochs": EPOCHS, "completed_epoch_cycles": EPOCHS, "validation_summary_count": EPOCHS, "optimizer_step_opportunities": EPOCHS * MICROBATCHES, "test_model_facing_access": 0, "learned_test_inference": 0}.items():  # literal-ok: frozen six-run accounting
        _require(run_completion.get(field) == expected, f"{run_id} completion differs: {field}")
    _require(run_completion["protected_counters"]["g10_adjudications"] == 0 and run_completion["protected_counters"]["er2_randomized_training"] == 0 and run_completion["protected_counters"]["papr_constrained_training"] == 0 and run_completion["protected_counters"]["er9_training"] == 0 and run_completion["protected_counters"]["learned_test_inference"] == 0 and run_completion["protected_counters"]["test_model_facing_access"] == 0, f"{run_id} completion protected counter is nonzero")

    expected_checkpoint_names = {f"epoch-{epoch:04d}.pt" for epoch in range(EPOCHS)} | {f"epoch-{epoch:04d}.sidecar.json" for epoch in range(EPOCHS)}
    expected_epoch_names = {f"epoch-{epoch:04d}.json" for epoch in range(EPOCHS)}
    _require({entry.name for entry in (run_dir / "checkpoints").iterdir()} == expected_checkpoint_names, f"{run_id} checkpoint namespace has a gap/extra")
    _require({entry.name for entry in (run_dir / "epochs").iterdir()} == expected_epoch_names, f"{run_id} epoch namespace has a gap/extra")
    _require({entry.name for entry in (run_dir / "validation").iterdir()} == expected_epoch_names, f"{run_id} validation namespace has a gap/extra")

    init: dict[str, Any] | None = None
    previous_checkpoint: str | None = None
    previous_global_step = 0
    transactions: list[dict[str, Any]] = []
    validation_noise: str | None = None
    validation_counts: list[tuple[int, int, str]] = []
    applied = 0
    skips = 0

    for epoch in range(EPOCHS):
        checkpoint_path = run_dir / "checkpoints" / f"epoch-{epoch:04d}.pt"
        sidecar_path = run_dir / "checkpoints" / f"epoch-{epoch:04d}.sidecar.json"
        epoch_path = run_dir / "epochs" / f"epoch-{epoch:04d}.json"
        validation_path = run_dir / "validation" / f"epoch-{epoch:04d}.json"
        for path, label in ((checkpoint_path, "W8 checkpoint payload"), (sidecar_path, "W8 checkpoint sidecar"), (epoch_path, "W8 epoch record"), (validation_path, "W8 validation summary")):
            _safe_file(path, f"{run_id} {label}")
        checkpoint_size = checkpoint_path.stat().st_size
        checkpoint_id = sha256_file(checkpoint_path)
        sidecar, sidecar_sha = read_json(sidecar_path, f"{run_id} epoch {epoch} sidecar")
        record, record_sha = read_json(epoch_path, f"{run_id} epoch {epoch} record")
        summary, summary_sha = read_json(validation_path, f"{run_id} epoch {epoch} validation")
        record_id = _record_id(record, f"{run_id} epoch {epoch} record")
        _require(sidecar["artifact_role"] == "W8_FINAL_TRAINING_CHECKPOINT_SIDECAR" and sidecar["schema_version"] == 1, f"{run_id} epoch {epoch} sidecar role/version differs")  # literal-ok: frozen sidecar schema
        _require(sidecar["campaign_id"] == CAMPAIGN_ID and sidecar["run_id"] == run_id, f"{run_id} epoch {epoch} sidecar run differs")
        _require(sidecar["checkpoint_path"] == f"checkpoints/epoch-{epoch:04d}.pt" and sidecar["checkpoint_id"] == checkpoint_id and sidecar["checkpoint_bytes"] == checkpoint_size, f"{run_id} epoch {epoch} checkpoint bytes/SHA differs")
        _require(sidecar["completed_epoch"] == epoch and sidecar["next_epoch"] == epoch + 1 and sidecar["accumulation_position"] == 0, f"{run_id} epoch {epoch} sidecar epoch differs")  # literal-ok: zero-based epoch transaction
        _require(sidecar["predecessor_checkpoint_id"] == previous_checkpoint, f"{run_id} epoch {epoch} predecessor chain differs")
        _require(sidecar["epoch_record_path"] == f"epochs/epoch-{epoch:04d}.json" and sidecar["epoch_record_id"] == record_id and sidecar["epoch_record_sha256"] == record_sha, f"{run_id} epoch {epoch} sidecar/record binding differs")
        for field, expected in {"source_commit": SOURCE_COMMIT, "source_manifest_id": SOURCE_MANIFEST_ID, "source_manifest_sha256": SOURCE_MANIFEST_SHA256, "execution_image": EXECUTION_IMAGE, "execution_profile_id": PROFILE_ID, "gpu_uuid": GPU_UUID, "dataset": "imagenette160", "ratio": cell["ratio"], "k": cell["k"], "lambda": LAMBDA_CORE, "train_seed": cell["train_seed"], "channel_seed": cell["channel_seed"], "train_snr_db": TRAIN_SNR_DB, "checkpoint_selection_snr_db": TRAIN_SNR_DB, "checkpoint_selection_channel_seed_rule": "run_channel_seed", "config_hash": run_completion["config_hash"], "protocol_config_hash": run_completion["protocol_config_hash"]}.items():
            _require(sidecar.get(field) == expected, f"{run_id} epoch {epoch} sidecar differs: {field}")
        sidecar_init = sidecar["initialization"]
        _verify_init(sidecar_init, cell)
        if init is None:
            init = sidecar_init
        else:
            _require(sidecar_init == init, f"{run_id} initialization identity changed across epochs")

        _require(record["artifact_role"] == "W8_FINAL_TRAINING_EPOCH_RECORD" and record["schema_version"] == 1, f"{run_id} epoch {epoch} record role/version differs")  # literal-ok: frozen epoch schema
        _require(record["campaign_id"] == CAMPAIGN_ID and record["run_id"] == run_id and record["epoch"] == epoch and record["next_epoch"] == epoch + 1, f"{run_id} epoch {epoch} record identity differs")
        _require(record["lineage"]["config_hash"] == run_completion["config_hash"] and record["lineage"]["protocol_config_hash"] == run_completion["protocol_config_hash"], f"{run_id} epoch {epoch} lineage config differs")
        _verify_lineage(record["lineage"], cell, previous_checkpoint)
        _require(record["lineage"]["initialization"] == init, f"{run_id} epoch {epoch} lineage initialization differs")
        _require(record["samples"] == SAMPLES and record["expected_samples"] == SAMPLES and record["stable_id_count"] == SAMPLES, f"{run_id} epoch {epoch} sample accounting differs")  # literal-ok: frozen training denominator
        _require(record["microbatches"] == MICROBATCHES and record["expected_microbatches"] == MICROBATCHES and record["final_physical_batch"] == FINAL_PHYSICAL_BATCH, f"{run_id} epoch {epoch} physical accounting differs")  # literal-ok: frozen batch arithmetic
        _require(record["optimizer_step_opportunities"] == MICROBATCHES and record["optimizer_steps"] + record["grad_scaler_skips"] == MICROBATCHES, f"{run_id} epoch {epoch} optimizer/skips arithmetic differs")
        _require(record["global_optimizer_step"] == previous_global_step + record["optimizer_steps"], f"{run_id} epoch {epoch} global optimizer chain differs")
        _require(record["finite_loss"] is True and record["validation_noise_identity_rule"] == "run_channel_seed", f"{run_id} epoch {epoch} training record boundary differs")
        noise, correct = verify_validation_summary(summary, cell, epoch, checkpoint_id, validation_noise)
        validation_noise = noise
        validation_counts.append((epoch, correct, summary["n_total"]))
        transactions.append({
            "epoch": epoch,
            "epoch_record": {"path": str(epoch_path.relative_to(root).as_posix()), "record_id": record_id, "file_sha256": record_sha},
            "checkpoint_payload": {"path": str(checkpoint_path.relative_to(root).as_posix()), "checkpoint_id": checkpoint_id, "file_sha256": checkpoint_id, "bytes": checkpoint_size},
            "scientific_sidecar": {"path": str(sidecar_path.relative_to(root).as_posix()), "file_sha256": sidecar_sha, "checkpoint_id": checkpoint_id},
            "validation_summary": {"path": str(validation_path.relative_to(root).as_posix()), "summary_id": summary["summary_id"], "file_sha256": summary_sha, "checkpoint_id": checkpoint_id, "n_correct": correct, "n_total": summary["n_total"], "noise_digest": noise},
            "predecessor_checkpoint_id": previous_checkpoint,
            "optimizer_step_opportunities": record["optimizer_step_opportunities"],
            "optimizer_steps": record["optimizer_steps"],
            "grad_scaler_skips": record["grad_scaler_skips"],
            "global_optimizer_step": record["global_optimizer_step"],
        })
        applied += int(record["optimizer_steps"])
        skips += int(record["grad_scaler_skips"])
        previous_global_step = int(record["global_optimizer_step"])
        previous_checkpoint = checkpoint_id

    _require(applied + skips == EPOCHS * MICROBATCHES, f"{run_id} total optimizer accounting differs")  # literal-ok: 100 epochs x 265 opportunities
    latest, latest_sha = read_json(run_dir / "latest.json", f"{run_id} latest pointer")
    last_sidecar, _ = read_json(run_dir / "checkpoints" / f"epoch-{EPOCHS - 1:04d}.sidecar.json", f"{run_id} final sidecar")
    _require(latest == last_sidecar and latest["completed_epoch"] == EPOCHS - 1 and latest["checkpoint_id"] == previous_checkpoint, f"{run_id} latest pointer does not authenticate epoch 99")  # literal-ok: zero-based final epoch
    _require(run_completion["latest_checkpoint_id"] == previous_checkpoint and run_completion["optimizer_steps"] == applied and run_completion["grad_scaler_skips"] == skips and run_completion["global_optimizer_step"] == applied, f"{run_id} completion accounting/latest differs")
    _require(run_completion["protected_counters"]["w8_final_training_runs"] == 1 and run_completion["protected_counters"]["w8_scientific_optimizer_steps"] == applied, f"{run_id} W8 protected counters differ")  # literal-ok: one completed run
    expected_epoch, expected_correct, _ = max(validation_counts, key=lambda item: (item[1], -item[0]))
    selected, selected_sha = read_json(run_dir / "selected_checkpoint.json", f"{run_id} selected checkpoint")
    _identified(selected, "result_id", "", f"{run_id} selected checkpoint")
    _require(selected["artifact_role"] == "W8_SELECTED_CHECKPOINT" and selected["test_model_facing_access"] == 0, f"{run_id} selected result role/test boundary differs")
    _require(selected["campaign_id"] == CAMPAIGN_ID and selected["run_id"] == run_id and selected["ratio"] == cell["ratio"] and selected["train_seed"] == cell["train_seed"] and selected["channel_seed"] == cell["channel_seed"], f"{run_id} selected result identity differs")
    _require(selected["checkpoint_epoch"] == expected_epoch and selected["checkpoint_id"] == transactions[expected_epoch]["checkpoint_payload"]["checkpoint_id"], f"{run_id} selected result checkpoint differs")
    _require(selected["validation"] == json.loads((root / transactions[expected_epoch]["validation_summary"]["path"]).read_bytes()), f"{run_id} selected validation summary differs")
    _require(isinstance(selected["validation_rows"], list) and len(selected["validation_rows"]) == VALIDATION_TOTAL, f"{run_id} selected validation row custody differs")  # literal-ok: published validation denominator
    verify_selection(selected["selection"], cell, expected_epoch, selected["checkpoint_id"], expected_correct)
    _require(run_completion["selection"] == selected["selection"], f"{run_id} completion selection differs")
    _require(run_completion["selected_checkpoint_epoch"] == expected_epoch and run_completion["selected_checkpoint_id"] == selected["checkpoint_id"] and run_completion["selected_result_id"] == selected["result_id"] and run_completion["selected_result_sha256"] == selected_sha, f"{run_id} published selection differs")
    return {
        "run_index": cell["run_index"], "run_directory": run_dir.name, "run_id": run_id, "ratio": cell["ratio"], "train_seed": cell["train_seed"], "channel_seed": cell["channel_seed"], "k": cell["k"], "config_hash": run_completion["config_hash"], "protocol_config_hash": run_completion["protocol_config_hash"], "run_completion": {"path": str((run_dir / "run_completion.json").relative_to(root).as_posix()), "completion_id": run_completion["completion_id"], "sha256": run_completion_sha}, "initialization": {"identity": init["identity"], "mode": init["mode"], "initial_model_state_sha256": init["initial_model_state_sha256"], "predecessor_checkpoint_id": None, "optimizer_state_transfer": False, "scheduler_state_transfer": False, "scaler_state_transfer": False, "w7_checkpoint_transfer": False, "prior_w8_state_transfer": False, "prior_w8_state_transfer_evidence": "absent under frozen source fresh_initialization_identity schema; authority forbids transfer"}, "epoch_count": EPOCHS, "checkpoint_count": EPOCHS, "sidecar_count": EPOCHS, "validation_summary_count": EPOCHS, "latest_epoch": EPOCHS - 1, "latest_checkpoint_id": previous_checkpoint, "optimizer_opportunities": EPOCHS * MICROBATCHES, "optimizer_steps": applied, "grad_scaler_skips": skips, "validation_noise_id_digest": validation_noise, "validation_noise_digest_invariant": True, "independently_reconstructed_selection": {"epoch": expected_epoch, "n_correct": expected_correct, "n_total": VALIDATION_TOTAL, "checkpoint_id": transactions[expected_epoch]["checkpoint_payload"]["checkpoint_id"]}, "published_selection": {"epoch": selected["checkpoint_epoch"], "checkpoint_id": selected["checkpoint_id"], "result_id": selected["result_id"], "result_file_sha256": selected_sha}, "selection_equal": True, "transactions": transactions,
    }


def verify_campaign_root(root: Path, authority: dict[str, Any], inventory: dict[str, Any], source: dict[str, Any] | None) -> dict[str, Any]:
    expected_root_entries = {"campaign_manifest.json", "campaign_completion.json", *EXPECTED_RUN_DIRS}
    actual_root_entries = {entry.name for entry in root.iterdir()}
    _require(actual_root_entries == expected_root_entries, "W8 campaign root has a foreign run/file or missing namespace")
    _require(all((root / name).is_dir() and not (root / name).is_symlink() for name in EXPECTED_RUN_DIRS), "W8 campaign root run namespace differs")
    manifest, manifest_sha = read_json(root / "campaign_manifest.json", "W8 campaign manifest")
    verify_campaign_manifest(manifest, authority)
    runs = [verify_run(root, cell, manifest) for cell in RUN_CELLS]
    completion, completion_sha = read_json(root / "campaign_completion.json", "W8 campaign completion")
    _identified(completion, "completion_id", "w8campaigncompletion-", "W8 campaign completion")
    _require(completion["artifact_role"] == "W8_CAMPAIGN_COMPLETION" and completion["status"] == "COMPLETE_NOT_YET_RECONCILED", "W8 campaign completion status differs")
    _require(completion["campaign_id"] == CAMPAIGN_ID and completion["source_commit"] == SOURCE_COMMIT and completion["source_manifest_id"] == SOURCE_MANIFEST_ID, "W8 campaign completion source differs")
    _require(completion["authorization_id"] == EXECUTION_AUTHORIZATION_ID and completion["launch_authorization_id"] == LAUNCH_AUTHORIZATION_ID, "W8 campaign completion authority differs")
    _require(completion["run_count"] == RUNS and completion["w8_completed_runs"] == RUNS and completion["w8_final_training_runs"] == RUNS and completion["w8_scientific_checkpoints"] == RUNS * EPOCHS, "W8 campaign completion run totals differ")  # literal-ok: six runs x 100 epochs
    _require(completion["run_order"] == list(EXPECTED_RUN_IDS), "W8 campaign completion run order differs")
    _require(completion["selected_checkpoints"] == [{"run_id": run["run_id"], "checkpoint_id": run["published_selection"]["checkpoint_id"], "epoch": run["published_selection"]["epoch"]} for run in runs], "W8 campaign completion selected checkpoints differ")
    _require(completion["w8_scientific_optimizer_steps"] == sum(run["optimizer_steps"] for run in runs), "W8 campaign completion optimizer total differs")
    for field in ("g10_adjudications", "er2_randomized_training", "papr_constrained_training", "er9_training", "test_model_facing_access", "learned_test_inference"):
        _require(completion[field] == 0, f"W8 campaign completion protected counter is nonzero: {field}")
    _require(completion["reconciliation"] == "REQUIRED_BEFORE_DOWNSTREAM_VALIDATION_OR_G10" and completion["test"] == "SEALED", "W8 campaign completion downstream boundary differs")
    _require(inventory["file_count"] == RUNS * (3 + 2 * EPOCHS + EPOCHS + EPOCHS) + 2, "W8 root file count differs")  # literal-ok: 6*(three run metadata + four*100 transactions) + 2 root files = 2420
    return {"manifest": {"path": str((root / "campaign_manifest.json")), "manifest_id": manifest["manifest_id"], "sha256": manifest_sha}, "completion": {"path": str((root / "campaign_completion.json")), "completion_id": completion["completion_id"], "sha256": completion_sha, "status": completion["status"]}, "runs": runs, "source_checkout": source}


def verify_predecessor_exclusion(authority: dict[str, Any], runs: list[dict[str, Any]], completion: dict[str, Any], incident_path: Path | None) -> dict[str, Any]:
    lineage = authority["values"]["lineage"]["predecessor"]
    _require(lineage["partial_checkpoint_sha256"] == OLD_PARTIAL_CHECKPOINT_SHA256 and lineage["historical_accepted_coverage"] == 0 and lineage["resume_eligible"] is False and lineage["result_eligible"] is False and lineage["g10_eligible"] is False and lineage["test_eligible"] is False, "W8 failed predecessor exclusion binding differs")
    if incident_path is not None:
        incident, incident_sha = read_json(incident_path, "W8 predecessor incident")
        _require(incident_sha == OLD_INCIDENT_SHA256 and incident["incident_id"] == OLD_INCIDENT_ID, "W8 predecessor incident bytes differ")
        _require(incident["old_campaign_inventory"]["inventory_digest"] == OLD_INVENTORY_SHA256 and incident["partial_checkpoint_eligibility"]["not_eligible_for_resume"] is True and incident["partial_checkpoint_eligibility"]["not_eligible_for_w8_result"] is True and incident["partial_checkpoint_eligibility"]["not_eligible_for_g10"] is True and incident["partial_checkpoint_eligibility"]["not_eligible_for_test"] is True, "W8 predecessor incident status differs")
        incident_binding = {"path": str(incident_path), "incident_id": incident["incident_id"], "sha256": incident_sha, "old_inventory_sha256": incident["old_campaign_inventory"]["inventory_digest"], "partial_checkpoint_sha256": incident["partial_checkpoint_eligibility"]["checkpoint_path"] and OLD_PARTIAL_CHECKPOINT_SHA256, "resume": False, "result": False, "g10": False, "test": False, "historical_optimizer_steps": incident["accepted_coverage"]["historical_scientific_optimizer_steps_executed"]}
    else:
        incident_binding = {"path": None, "incident_id": OLD_INCIDENT_ID, "sha256": OLD_INCIDENT_SHA256, "old_inventory_sha256": OLD_INVENTORY_SHA256, "partial_checkpoint_sha256": OLD_PARTIAL_CHECKPOINT_SHA256, "resume": False, "result": False, "g10": False, "test": False, "historical_optimizer_steps": 259}  # literal-ok: immutable incident history
    serialized_ids = json.dumps({"runs": runs, "completion": completion}, sort_keys=True)
    _require(OLD_PARTIAL_CHECKPOINT_SHA256 not in serialized_ids, "historical c5a8 partial checkpoint appears in successor evidence")
    return {"incident": incident_binding, "successor_coverage": 0, "old_partial_checkpoint_in_successor": False, "historical_steps_excluded": True}


def build_live_report(args: argparse.Namespace) -> dict[str, Any]:
    authority = verify_authorities(Path(args.authority_dir))
    source = verify_source_checkout(Path(args.source_dir)) if args.source_dir else None
    custody = verify_custody(Path(args.heartbeat), Path(args.stdout), Path(args.lock)) if args.check_custody else None
    root = Path(args.campaign_root)
    inventory, entries = build_root_inventory(root, Path(args.inventory_output) if args.inventory_output else None)
    if args.inventory_output:
        verify_root_inventory(Path(args.inventory_output), entries, inventory["byte_count"], inventory["file_count"])
    root_report = verify_campaign_root(root, authority, inventory, source)
    predecessor = verify_predecessor_exclusion(authority, root_report["runs"], root_report["completion"], Path(args.incident_path) if args.incident_path else None)
    total_opportunities = sum(run["optimizer_opportunities"] for run in root_report["runs"])
    total_steps = sum(run["optimizer_steps"] for run in root_report["runs"])
    total_skips = sum(run["grad_scaler_skips"] for run in root_report["runs"])
    _require(total_opportunities == RUNS * EPOCHS * MICROBATCHES and total_steps + total_skips == total_opportunities, "W8 campaign optimizer accounting does not close")  # literal-ok: six runs x 100 epochs x 265 opportunities
    _require(len(root_report["runs"]) == RUNS and sum(run["epoch_count"] for run in root_report["runs"]) == RUNS * EPOCHS, "W8 campaign transaction coverage differs")  # literal-ok: six runs x 100 epochs
    return {
        "report_schema_version": 1,
        "artifact_role": "W8_C_LIVE_READ_ONLY_VERIFICATION_REPORT",
        "status": "PASS",
        "model_facing_recomputation": False,
        "scientific_runtime_modified": False,
        "campaign_root": str(root),
        "authority": authority["files"],
        "source": source,
        "custody": custody,
        "inventory": inventory,
        "campaign": root_report,
        "predecessor_exclusion": predecessor,
        "counts": {"runs": RUNS, "epoch_records": RUNS * EPOCHS, "checkpoint_payloads": RUNS * EPOCHS, "scientific_sidecars": RUNS * EPOCHS, "validation_summaries": RUNS * EPOCHS, "optimizer_opportunities": total_opportunities, "optimizer_steps": total_steps, "grad_scaler_skips": total_skips},
        "validation": {"denominator": VALIDATION_TOTAL, "selection_metric": "validation_top1_accuracy", "mode": "max", "tie_break": "earliest_epoch", "snr_db": TRAIN_SNR_DB, "noise_invariant_per_run": True},
        "protected_boundaries": {"g10": 0, "er2": 0, "papr_constrained_training": 0, "er9": 0, "test_model_facing_access": 0, "learned_test_inference": 0},
        "monitor_terminal_display": {"classification": "MONITOR_PRESENTATION_DEFECT", "scope": "OPERATIONS ONLY", "scientific_impact": "ZERO", "basis": "terminal heartbeat has no live run/epoch fields; direct root custody authenticates 600 epochs; monitor zero/n-a fallback is not accounting"},
    }


def _artifact_with_id(body: dict[str, Any], field: str, prefix: str) -> dict[str, Any]:
    result = dict(body)
    result[field] = prefix + canonical_sha256(body)
    return result


def write_json_immutable(path: Path, value: dict[str, Any], label: str) -> tuple[str, str]:
    raw = canonical_bytes(value)
    _publish_bytes(path, raw, label)
    return str(value[next(field for field in ("reconciliation_id", "completion_id") if field in value)]), _sha_bytes(raw)


def build_reconciliation(report: dict[str, Any]) -> dict[str, Any]:
    _require(report.get("status") == "PASS" and report.get("model_facing_recomputation") is False and report.get("scientific_runtime_modified") is False, "W8 live report is not a clean PASS")
    campaign = report["campaign"]
    completion = campaign["completion"]
    runs = campaign["runs"]
    body = {
        "schema_version": 1,
        "artifact_role": "W8_C_TERMINAL_RECONCILIATION",
        "status": "W8_RECONCILED_GREEN",
        "campaign": {"campaign_id": CAMPAIGN_ID, "campaign_root": report["campaign_root"], "manifest": campaign["manifest"], "completion": completion, "expected_terminal_status": "COMPLETE_NOT_YET_RECONCILED"},
        "source": {"source_commit": SOURCE_COMMIT, "source_checkout": report["source"], "source_manifest": report["authority"]["source_manifest"]},
        "authorities": {"execution_authorization": report["authority"]["execution_authorization"], "launch_authorization": report["authority"]["launch_authorization"], "successor_lineage": report["authority"]["lineage"]},
        "custody": {"heartbeat": report["custody"]["heartbeat"] if report.get("custody") else None, "stdout": report["custody"]["stdout"] if report.get("custody") else None, "global_lock": report["custody"]["global_lock"] if report.get("custody") else None, "inventory": report["inventory"]},
        "run_identities": [{key: run[key] for key in ("run_index", "run_directory", "run_id", "ratio", "train_seed", "channel_seed", "k", "config_hash", "protocol_config_hash")} for run in runs],
        "transaction_accounting": {"epoch_records": report["counts"]["epoch_records"], "checkpoint_payloads": report["counts"]["checkpoint_payloads"], "scientific_sidecars": report["counts"]["scientific_sidecars"], "validation_summaries": report["counts"]["validation_summaries"], "per_run": [{"run_id": run["run_id"], "epoch_records": run["epoch_count"], "checkpoint_payloads": run["checkpoint_count"], "scientific_sidecars": run["sidecar_count"], "validation_summaries": run["validation_summary_count"], "latest_epoch": run["latest_epoch"], "latest_checkpoint_id": run["latest_checkpoint_id"], "optimizer_opportunities": run["optimizer_opportunities"], "optimizer_steps": run["optimizer_steps"], "grad_scaler_skips": run["grad_scaler_skips"], "transactions": run["transactions"]} for run in runs]},
        "campaign_optimizer_accounting": {"opportunities": report["counts"]["optimizer_opportunities"], "applied_optimizer_steps": report["counts"]["optimizer_steps"], "grad_scaler_skips": report["counts"]["grad_scaler_skips"], "closure": "applied_optimizer_steps + grad_scaler_skips = opportunities"},
        "selection": {"metric": "validation_top1_accuracy", "mode": "max", "tie_break": "earliest_epoch", "validation_snr_db": TRAIN_SNR_DB, "validation_denominator": VALIDATION_TOTAL, "cross_seed_selection": False, "per_run": [{"run_id": run["run_id"], "independently_reconstructed": run["independently_reconstructed_selection"], "runner_published": run["published_selection"], "equal": run["selection_equal"]} for run in runs], "equality_count": sum(1 for run in runs if run["selection_equal"])},
        "fresh_initialization": {"verified_runs": sum(1 for run in runs if run["initialization"]["mode"] == "fresh_keyed_init"), "total_runs": RUNS, "per_run": [{"run_id": run["run_id"], **run["initialization"]} for run in runs], "cross_train_seed_initialization_ids_differ": len({run["initialization"]["initial_model_state_sha256"] for run in runs if run["train_seed"] == 0}) == 2 or len({run["initialization"]["initial_model_state_sha256"] for run in runs}) == RUNS},
        "validation_noise": {"denominator": VALIDATION_TOTAL, "one_digest_per_run": all(run["validation_noise_digest_invariant"] for run in runs), "per_run": [{"run_id": run["run_id"], "digest": run["validation_noise_id_digest"], "epoch_count": EPOCHS} for run in runs]},
        "predecessor_exclusion": report["predecessor_exclusion"],
        "protected_boundaries": report["protected_boundaries"],
        "monitor_terminal_display": report["monitor_terminal_display"],
        "eligibility": {"six_selected_checkpoints_frozen": True, "w8_result": "ELIGIBLE_FOR_DOWNSTREAM_VALIDATION_ONLY", "g10": "CLOSED_UNTIL_W8_C", "test": "NOT_ELIGIBLE_FOR_TEST", "best_seed_filtering": False},
    }
    _require(body["selection"]["equality_count"] == RUNS and body["fresh_initialization"]["verified_runs"] == RUNS and body["validation_noise"]["one_digest_per_run"] is True, "W8 reconciliation preconditions do not pass")  # literal-ok: six-run reconciliation gate
    return _artifact_with_id(body, "reconciliation_id", "w8creconcile-")


def verify_reconciliation(value: dict[str, Any], inventory_path: Path | None = None) -> None:
    _require(value.get("artifact_role") == "W8_C_TERMINAL_RECONCILIATION" and value.get("status") == "W8_RECONCILED_GREEN" and value.get("schema_version") == 1, "W8-C reconciliation role/status/version differs")  # literal-ok: reconciliation schema
    _identified(value, "reconciliation_id", "w8creconcile-", "W8-C reconciliation")
    _require(value["campaign"]["campaign_id"] == CAMPAIGN_ID and value["campaign"]["expected_terminal_status"] == "COMPLETE_NOT_YET_RECONCILED", "W8-C reconciliation campaign differs")
    _require(value["campaign"]["manifest"]["manifest_id"] == "w8campaignmanifest-96e119dbb70ba9311307e685c5714e0a27ecd0e701b128bd56a401897dd45467" and value["campaign"]["completion"]["completion_id"] == "w8campaigncompletion-843c7019d8b40bf2b57adf42b63a8e324354739ed68fe0d485f07c74e3ac7a7f" and value["campaign"]["completion"]["status"] == "COMPLETE_NOT_YET_RECONCILED", "W8-C campaign manifest/completion binding differs")
    _require(value["source"]["source_commit"] == SOURCE_COMMIT and value["source"]["source_manifest"]["id"] == SOURCE_MANIFEST_ID and value["source"]["source_manifest"]["sha256"] == SOURCE_MANIFEST_SHA256, "W8-C reconciliation source differs")
    _require(value["authorities"]["execution_authorization"]["id"] == EXECUTION_AUTHORIZATION_ID and value["authorities"]["execution_authorization"]["sha256"] == EXECUTION_AUTHORIZATION_SHA256 and value["authorities"]["launch_authorization"]["id"] == LAUNCH_AUTHORIZATION_ID and value["authorities"]["launch_authorization"]["sha256"] == LAUNCH_AUTHORIZATION_SHA256 and value["authorities"]["successor_lineage"]["id"] == LINEAGE_ID and value["authorities"]["successor_lineage"]["sha256"] == LINEAGE_SHA256, "W8-C reconciliation authority differs")
    counts = value["transaction_accounting"]
    _verify_compact_transactions(value)
    _require(counts["epoch_records"] == RUNS * EPOCHS and counts["checkpoint_payloads"] == RUNS * EPOCHS and counts["scientific_sidecars"] == RUNS * EPOCHS and counts["validation_summaries"] == RUNS * EPOCHS, "W8-C reconciliation transaction counts differ")  # literal-ok: six runs x 100 epochs
    _require(value["custody"]["inventory"]["file_count"] == RUNS * (3 + 2 * EPOCHS + EPOCHS + EPOCHS) + 2 and value["custody"]["inventory"]["byte_count"] == 11393422099, "W8-C reconciliation root inventory totals differ")  # literal-ok: authenticated six-run root inventory byte total
    _require(value["campaign_optimizer_accounting"]["opportunities"] == RUNS * EPOCHS * MICROBATCHES and value["campaign_optimizer_accounting"]["applied_optimizer_steps"] + value["campaign_optimizer_accounting"]["grad_scaler_skips"] == RUNS * EPOCHS * MICROBATCHES, "W8-C reconciliation optimizer closure differs")  # literal-ok: six runs x 100 epochs x 265 opportunities
    _verify_compact_selection(value)
    _verify_compact_initialization(value)
    _verify_compact_noise_and_boundaries(value)
    _require(value["predecessor_exclusion"]["old_partial_checkpoint_in_successor"] is False and value["predecessor_exclusion"]["historical_steps_excluded"] is True, "W8-C predecessor exclusion differs")
    _require(value["protected_boundaries"] == {"g10": 0, "er2": 0, "papr_constrained_training": 0, "er9": 0, "test_model_facing_access": 0, "learned_test_inference": 0}, "W8-C protected boundaries differ")
    _require(value["eligibility"]["six_selected_checkpoints_frozen"] is True and value["eligibility"]["best_seed_filtering"] is False and value["eligibility"]["test"] == "NOT_ELIGIBLE_FOR_TEST", "W8-C final eligibility differs")
    if inventory_path is not None:
        inventory = value["custody"]["inventory"]
        verify_compact_inventory(inventory_path, inventory)


def build_terminal_completion(reconciliation: dict[str, Any], reconciliation_path: Path, reconciliation_sha: str) -> dict[str, Any]:
    body = {
        "schema_version": 1,
        "artifact_role": "W8_TERMINAL_COMPLETION",
        "status": "W8_GREEN_CLOSED",
        "campaign_id": CAMPAIGN_ID,
        "reconciliation": {"path": str(reconciliation_path), "reconciliation_id": reconciliation["reconciliation_id"], "sha256": reconciliation_sha},
        "selected_checkpoints": [item["runner_published"] for item in reconciliation["selection"]["per_run"]],
        "selected_checkpoint_count": RUNS,
        "six_selected_checkpoints_frozen": True,
        "eligible_for_downstream_validation_only": True,
        "best_seed_filtering": False,
        "g10": "NOT_EXECUTED",
        "g10_count": 0,
        "er2": 0,
        "papr_constrained_training": 0,
        "er9": 0,
        "test": "SEALED",
        "test_model_facing_access": 0,
        "learned_test_inference": 0,
        "next_safe_action": "OWNER_AUDIT_THEN_W9_G10_VALIDATION_ONLY",
    }
    return _artifact_with_id(body, "completion_id", "w8completion-")


def verify_terminal_completion(
    value: dict[str, Any],
    reconciliation: dict[str, Any] | None = None,
    reconciliation_sha256: str | None = None,
) -> None:
    _require(value.get("artifact_role") == "W8_TERMINAL_COMPLETION" and value.get("status") == "W8_GREEN_CLOSED" and value.get("schema_version") == 1, "W8 terminal completion role/status/version differs")  # literal-ok: terminal completion schema
    _identified(value, "completion_id", "w8completion-", "W8 terminal completion")
    _require(value["campaign_id"] == CAMPAIGN_ID and value["selected_checkpoint_count"] == RUNS and value["six_selected_checkpoints_frozen"] is True and value["eligible_for_downstream_validation_only"] is True and value["best_seed_filtering"] is False, "W8 terminal completion selection boundary differs")  # literal-ok: six selected models
    _require(value["g10"] == "NOT_EXECUTED" and value["g10_count"] == 0 and value["er2"] == 0 and value["papr_constrained_training"] == 0 and value["er9"] == 0 and value["test"] == "SEALED" and value["test_model_facing_access"] == 0 and value["learned_test_inference"] == 0, "W8 terminal completion protected boundary differs")
    reconciliation_id = value["reconciliation"]["reconciliation_id"]
    _require(isinstance(reconciliation_id, str) and reconciliation_id.startswith("w8creconcile-"), "W8 terminal completion reconciliation ID is malformed")
    _full_sha(reconciliation_id.removeprefix("w8creconcile-"))
    _full_sha(value["reconciliation"]["sha256"])
    if reconciliation is not None:
        _require(value["reconciliation"]["reconciliation_id"] == reconciliation["reconciliation_id"], "W8 terminal completion reconciliation ID differs")
        _require(
            reconciliation_sha256 is not None
            and value["reconciliation"]["sha256"] == reconciliation_sha256,
            "W8 terminal completion reconciliation SHA differs",
        )
        _require(
            value["selected_checkpoints"] == [
                item["runner_published"] for item in reconciliation["selection"]["per_run"]
            ],
            "W8 terminal completion selected checkpoints differ",
        )


def _write_report(path: Path | None, report: dict[str, Any]) -> None:
    raw = canonical_bytes(report)
    if path is not None:
        _publish_bytes(path, raw, "W8 live report")
    print(json.dumps({"status": report["status"], "inventory": report["inventory"], "counts": report["counts"], "campaign_completion": report["campaign"]["completion"], "runs": [{"run_id": run["run_id"], "optimizer_steps": run["optimizer_steps"], "grad_scaler_skips": run["grad_scaler_skips"], "selected_epoch": run["published_selection"]["epoch"], "selected_checkpoint_id": run["published_selection"]["checkpoint_id"]} for run in report["campaign"]["runs"]]}, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    live = subparsers.add_parser("verify-live")
    live.add_argument("--campaign-root", required=True)
    live.add_argument("--authority-dir", required=True)
    live.add_argument("--source-dir")
    live.add_argument("--heartbeat", default=HEARTBEAT_PATH)
    live.add_argument("--stdout", default=STDOUT_PATH)
    live.add_argument("--lock", default=GLOBAL_LOCK)
    live.add_argument("--inventory-output")
    live.add_argument("--report-output")
    live.add_argument("--incident-path")
    live.add_argument("--check-custody", action="store_true")
    compact = subparsers.add_parser("verify-compact")
    compact.add_argument("--reconciliation", required=True)
    compact.add_argument("--inventory")
    compact.add_argument("--authority-dir")
    compact.add_argument("--source-dir")
    terminal = subparsers.add_parser("verify-terminal")
    terminal.add_argument("--completion", required=True)
    terminal.add_argument("--reconciliation")
    build = subparsers.add_parser("build-reconciliation")
    build.add_argument("--report", required=True)
    build.add_argument("--output", required=True)
    complete = subparsers.add_parser("build-terminal")
    complete.add_argument("--reconciliation", required=True)
    complete.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify-live":
            report = build_live_report(args)
            _write_report(Path(args.report_output) if args.report_output else None, report)
            return 0
        if args.command == "verify-compact":
            value, _ = read_json(Path(args.reconciliation), "W8-C reconciliation")
            verify_reconciliation(value, Path(args.inventory) if args.inventory else None)
            if args.authority_dir:
                verify_authorities(Path(args.authority_dir))
            if args.source_dir:
                verify_source_checkout(Path(args.source_dir))
            print(f"W8-C compact reconciliation PASS: {value['reconciliation_id']}")
            return 0
        if args.command == "verify-terminal":
            value, _ = read_json(Path(args.completion), "W8 terminal completion")
            reconciliation = None
            reconciliation_sha = None
            if args.reconciliation:
                reconciliation, reconciliation_sha = read_json(Path(args.reconciliation), "W8-C reconciliation")
                verify_reconciliation(reconciliation)
            verify_terminal_completion(value, reconciliation, reconciliation_sha)
            print(f"W8 terminal completion PASS: {value['completion_id']}")
            return 0
        if args.command == "build-reconciliation":
            report, _ = read_json(Path(args.report), "W8 live report")
            value = build_reconciliation(report)
            identifier, file_sha = write_json_immutable(Path(args.output), value, "W8-C reconciliation")
            print(json.dumps({"status": value["status"], "reconciliation_id": identifier, "sha256": file_sha, "path": args.output}, sort_keys=True))
            return 0
        if args.command == "build-terminal":
            reconciliation, reconciliation_sha = read_json(Path(args.reconciliation), "W8-C reconciliation")
            verify_reconciliation(reconciliation)
            value = build_terminal_completion(reconciliation, Path(args.reconciliation), reconciliation_sha)
            identifier, file_sha = write_json_immutable(Path(args.output), value, "W8 terminal completion")
            print(json.dumps({"status": value["status"], "completion_id": identifier, "sha256": file_sha, "path": args.output}, sort_keys=True))
            return 0
    except (W8CHold, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"W8-C HOLD — {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
