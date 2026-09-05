#!/usr/bin/env python3
"""Detached W8-B launcher for the already-frozen six-run campaign.

The ``preflight`` command is read-only with respect to scientific state.  The
``start`` command requires a separate W8-B owner launch authorization and is
intentionally never invoked by W8-A.  When invoked later, one process holds
one kernel lock while it runs the six cells sequentially on the authenticated
GTX 1080 Ti.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from config.params import REPO_ROOT, get  # noqa: E402
from config.run_config import config_hash as run_config_hash  # noqa: E402
from config.w8_execution import authenticate_w8_gpu, verify_frozen_w8_gpu_binding  # noqa: E402
from data.manifests import check_manifest, manifest_path, validate_manifest_bytes  # noqa: E402
from data.provenance import verify_extracted_dataset  # noqa: E402
from evaluation.w8_validation import (  # noqa: E402
    W8ValidationHold,
    _validate_selection,
    _validate_summary,
    evaluation_config_hash,
    evaluate_validation,
    selected_checkpoint_result,
    select_checkpoint_epoch,
    validate_selected_checkpoint_result,
    validation_stable_ids,
)
from gen_w8_execution_authorization import (  # noqa: E402
    AUTHORIZATION_ROLE,
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    G4_ID,
    G4_PATH,
    G4_SHA256,
    HEARTBEAT_PATH,
    PASCAL_LOCK_SHA256,
    STDOUT_LOG_PATH,
    W7_TERMINAL_ID,
    W7_TERMINAL_PATH,
    W7_TERMINAL_SHA256,
    _am94_predecessor_config_bindings,
    verify_authorization,
)
from gen_w8_source_manifest import verify_manifest  # noqa: E402
from runtime.w8_lock import W8CampaignLock, W8LockBusy, W8_GLOBAL_LOCK_PATH  # noqa: E402
from training.deterministic_core import canonical_bytes, canonical_sha256  # noqa: E402
from training.w8_final import (  # noqa: E402
    W8Hold,
    W8SourceLineage,
    W8Trainer,
    checkpoint_state_digest,
)
from training.w8_protocol import (  # noqa: E402
    W8_ACCUMULATION_FACTOR,
    W8_CAMPAIGN_COMPLETION_ROLE,
    W8_CHECKPOINT_ROLE,
    W8_CHANNEL_SEEDS,
    W8_TRAINING_EPOCH_ROLE,
    W8_DATASET,
    W8_EPOCHS,
    W8_EFFECTIVE_BATCH_SIZE,
    W8_EXPECTED_K,
    W8_EXPECTED_RATIOS,
    W8_EXECUTION_IMAGE_FAMILY,
    W8_PHYSICAL_BATCH_SIZE,
    W8_PROFILE_ID,
    W8_SELECTED_GPU_NAME,
    W8_SELECTED_GPU_UUID,
    W8_TRAIN_SAMPLE_COUNT,
    W8_TRAIN_SEEDS,
    W8_VALIDATION_BATCH_SIZE,
    W8_VALIDATION_SAMPLE_COUNT,
    W8_MIN_FREE_SPACE_GIB,
    checkpoint_selection_snr_db,
    eligibility_for_role,
    load_w8_config,
    protocol_config_hash,
    protocol_descriptor,
    run_cells,
)


RUN_COMPLETION_ROLE = "W8_FINAL_TRAINING_RUN_COMPLETION"
CAMPAIGN_MANIFEST_ROLE = "W8_CAMPAIGN_MANIFEST"
HEARTBEAT_ROLE = "W8_OPERATIONAL_HEARTBEAT"
LAUNCH_AUTHORIZATION_ROLE = "W8_B_LAUNCH_AUTHORIZATION"
CAMPAIGN_COMPLETION_ROLE = W8_CAMPAIGN_COMPLETION_ROLE
RUN_COMPLETION_PREFIX = "w8runcompletion-"
CAMPAIGN_MANIFEST_PREFIX = "w8campaignmanifest-"
CAMPAIGN_COMPLETION_PREFIX = "w8campaigncompletion-"


class W8CampaignHold(RuntimeError):
    """The detached launcher cannot safely continue the frozen campaign."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise W8CampaignHold(message)


def _sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise W8CampaignHold(f"cannot hash {path}: {exc}") from None


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing or unsafe")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise W8CampaignHold(f"{label} is unreadable: {exc}") from None
    _require(isinstance(value, dict), f"{label} is not a JSON object")
    return value


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise W8CampaignHold(f"git command failed in {repo}: {exc}") from None


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_new_json(path: Path, value: dict[str, Any]) -> None:
    """Publish immutable JSON bytes without replacing an existing pathname."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise W8CampaignHold(f"immutable W8 artifact already exists: {path}")
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
            raise W8CampaignHold(f"immutable W8 artifact already exists: {path}") from None
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _replace_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically replace an operational pointer outside the scientific root."""

    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.is_symlink() and (not path.exists() or path.is_file()), f"unsafe W8 heartbeat path: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_heartbeat(
    path: Path,
    *,
    campaign_id: str,
    current_run_index: int | None,
    ratio: str | None,
    train_seed: int | None,
    channel_seed: int | None,
    current_epoch: int | None,
    process_state: str,
    latest_checkpoint_id: str | None,
    completed_runs: int,
    completed_epoch_cycles: int,
) -> None:
    """Write only operational state; no loss, accuracy, PSNR or PAPR."""

    value = {
        "schema_version": 1,
        "artifact_role": HEARTBEAT_ROLE,
        "campaign_id": campaign_id,
        "current_run_index": current_run_index,
        "total_runs": 6,  # literal-ok: frozen W8 campaign cardinality
        "ratio": ratio,
        "train_seed": train_seed,
        "channel_seed": channel_seed,
        "current_epoch": current_epoch,
        "process_state": process_state,
        "latest_checkpoint_id": latest_checkpoint_id,
        "completed_runs": completed_runs,
        "completed_epoch_cycles": completed_epoch_cycles,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _replace_json(path, value)


def _campaign_run_root_name(cell: Any) -> str:
    return f"run-{cell.run_index:02d}-{cell.ratio}-train{cell.train_seed}-channel{cell.channel_seed}"


def _validate_campaign_root_namespace(root: Path) -> None:
    """Reject state outside the six immutable run namespaces."""

    allowed_files = {"campaign_manifest.json", "campaign_completion.json"}
    allowed_directories = {_campaign_run_root_name(cell) for cell in run_cells()}
    for entry in root.iterdir():
        _require(not entry.is_symlink(), f"W8 campaign root contains a symlink: {entry.name}")
        if entry.name in allowed_files:
            _require(entry.is_file(), f"W8 campaign root file is unsafe: {entry.name}")
        elif entry.name in allowed_directories:
            _require(entry.is_dir(), f"W8 campaign run namespace is unsafe: {entry.name}")
        else:
            raise W8CampaignHold(f"W8 campaign root contains foreign state: {entry.name}")


def _safe_root(root: Path, *, allow_existing: bool) -> None:
    _require(
        not root.is_symlink() and (not root.exists() or root.is_dir()),
        f"W8 campaign root is unsafe: {root}",
    )
    if not root.exists():
        return
    if not allow_existing:
        _require(not any(root.iterdir()), "W8 campaign root is not empty before W8-B")
        return
    _validate_campaign_root_namespace(root)


def _verify_test_boundary(repo: Path) -> None:
    _require(get("evaluation.test_access_gate") == "G-12", "W8 test release gate is not G-12")
    guarded = (repo / "src/data/test_access.py").resolve()
    _require(guarded.is_file(), "W8 guarded test boundary is missing")
    violations: list[str] = []
    for path in sorted((repo / "src").rglob("*.py")):
        if path.resolve() == guarded:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            raise W8CampaignHold(f"cannot audit W8 test boundary at {path}: {exc}") from None
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports_guard = any(
                    alias.name == "data.test_access" or alias.name.startswith("data.test_access.")
                    for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports_guard = module == "data.test_access" or module.startswith("data.test_access.") or (
                    module in {"data", ""} and any(alias.name == "test_access" for alias in node.names)
                )
            else:
                continue
            if imports_guard:
                violations.append(str(path.relative_to(repo)))
    _require(not violations, f"W8 production test boundary imports are unsafe: {violations}")


def _verify_dataset(repo: Path, *, require_extracted: bool) -> dict[str, Any]:
    """Verify only the W8 train/val dataset provenance and manifest bytes."""

    try:
        from data.provenance import archive_path, verify_archive

        archive_path_value = archive_path(W8_DATASET, repo)
        if archive_path_value.is_symlink():
            raise W8CampaignHold("W8 dataset archive path is a symlink")
        if not archive_path_value.exists() and not require_extracted:
            return {"status": "ARCHIVE_NOT_PRESENT_PREPARATION_REQUIRED"}
        archive = verify_archive(W8_DATASET, repo)
    except W8CampaignHold:
        raise
    except Exception as exc:
        raise W8CampaignHold(f"W8 dataset archive verification failed: {exc}") from None
    try:
        manifest_sha = check_manifest(W8_DATASET, repo)
        manifest_file = manifest_path(W8_DATASET, repo)
        rows = validate_manifest_bytes(W8_DATASET, manifest_file.read_bytes())
        counts = Counter(row.split for row in rows)
        _require(counts["train"] == W8_TRAIN_SAMPLE_COUNT, "W8 train manifest count differs from 8469")
        _require(counts["val"] == W8_VALIDATION_SAMPLE_COUNT, "W8 validation manifest count differs from 1000")
        _require(counts["test"] == int(get(f"datasets.{W8_DATASET}.test_images")), "W8 test manifest count differs")
        if require_extracted:
            verify_extracted_dataset(W8_DATASET, repo)
    except Exception as exc:
        raise W8CampaignHold(f"W8 dataset manifest/extraction verification failed: {exc}") from None
    return {
        "status": "VERIFIED",
        "archive_path": str(archive.path),
        "archive_bytes": archive.byte_length,
        "archive_sha256": archive.sha256,
        "manifest_path": str(manifest_file),
        "manifest_sha256": manifest_sha,
        "counts": {split: counts[split] for split in ("train", "val", "test")},
    }


def _run_w7_g4_verifier(repo: Path) -> None:
    """Invoke the direct upstream terminal verifier before any W8 work."""

    try:
        subprocess.run(
            [sys.executable, str(repo / "tools/verify_w7_g4.py"), "verify"],
            cwd=repo,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise W8CampaignHold(f"direct W7/G-4 terminal verification failed: {exc}") from None


def _expected_auth_keys() -> set[str]:
    return {
        "schema_version", "artifact_role", "status", "authorization_scope",
        "authorization_basis", "issued_at_utc", "scientific_execution_authorized",
        "w8_b_launch_authorization_required", "upstream", "scientific_source",
        "campaign", "training", "checkpoint_selection", "profile",
        "resume_and_custody", "boundary", "pre_execution_zero_counters",
        "protocol_hash", "source_contains_no_w8_results", "campaign_root_created_at_freeze",
        "authorization_id",
    }


def _expected_boundary() -> dict[str, Any]:
    return {
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


def _expected_zero_counters() -> dict[str, int]:
    return {
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


def _validate_authorization_contract(
    value: dict[str, Any], *, source_manifest: dict[str, Any], source_manifest_path: Path
) -> None:
    _require(set(value) == _expected_auth_keys(), "W8 execution authorization schema differs")
    _require(value["artifact_role"] == AUTHORIZATION_ROLE, "W8 authorization role differs")
    _require(value["status"] == "FROZEN_PRE_EXECUTION", "W8 authorization is not pre-execution frozen")
    _require(value["authorization_scope"] == "W8_SIX_CORE_RUNS_ONLY", "W8 authorization scope differs")
    _require(value["scientific_execution_authorized"] == "SIX_CORE_RUNS_ONLY", "W8 scientific scope differs")
    _require(value["w8_b_launch_authorization_required"] is True, "W8-B detached authorization is not required")
    _require(value["upstream"]["w7_terminal"] == {
        "completion_id": W7_TERMINAL_ID,
        "path": W7_TERMINAL_PATH,
        "file_sha256": W7_TERMINAL_SHA256,
        "status": "W7_GREEN_CLOSED",
    }, "W8 W7 terminal binding differs")
    _require(value["upstream"]["g4_adjudication"]["adjudication_id"] == G4_ID, "W8 G-4 ID differs")
    _require(value["upstream"]["g4_adjudication"]["path"] == G4_PATH, "W8 G-4 path differs")
    _require(value["upstream"]["g4_adjudication"]["file_sha256"] == G4_SHA256, "W8 G-4 file binding differs")
    _require(value["upstream"]["g4_adjudication"]["selected_lambda"] == 3.0, "W8 selected lambda differs")
    _require(value["upstream"]["g4_adjudication"]["lambda_status"] == "selected_at_G-4", "W8 lambda status differs")
    _require(value["scientific_source"]["source_manifest_id"] == source_manifest["manifest_id"], "W8 source manifest ID differs")
    _require(value["scientific_source"]["source_manifest_source_commit"] == source_manifest["source_commit"], "W8 source manifest commit differs")
    _require(value["scientific_source"]["source_manifest_file_sha256"] == _sha(source_manifest_path), "W8 source manifest file SHA differs")
    _require(value["campaign"]["campaign_id"] == CAMPAIGN_ID, "W8 campaign ID differs")
    _require(value["campaign"]["campaign_root"] == CAMPAIGN_ROOT, "W8 campaign root differs")
    _require(value["campaign"]["heartbeat_path"] == HEARTBEAT_PATH, "W8 heartbeat path differs")
    _require(value["campaign"]["stdout_log_path"] == STDOUT_LOG_PATH, "W8 stdout path differs")
    cells = [cell.to_dict() for cell in run_cells()]
    _require(value["campaign"]["run_count"] == 6 and value["campaign"]["run_cells"] == cells, "W8 six-cell matrix differs")
    _require(value["campaign"]["unique_ratios"] == list(W8_EXPECTED_RATIOS), "W8 unique ratio set differs")
    _require(value["campaign"]["k_by_ratio"] == dict(W8_EXPECTED_K), "W8 k-by-ratio binding differs")
    _require(value["campaign"]["train_seeds"] == list(W8_TRAIN_SEEDS), "W8 train seed set differs")
    _require(value["campaign"]["channel_seeds"] == list(W8_CHANNEL_SEEDS), "W8 channel seed set differs")
    _require(value["campaign"]["seed_pairing"] == "zipped_not_cross_product", "W8 seed pairing differs")
    expected_configs = []
    for cell in run_cells():
        config = load_w8_config(cell.ratio, cell.train_seed, cell.channel_seed)
        expected_configs.append({
            "run_index": cell.run_index,
            "config_hash": run_config_hash(config),
            "protocol_config_hash": protocol_config_hash(config),
        })
    if value["campaign"]["config_bindings"] != expected_configs:
        try:
            expected_configs = _am94_predecessor_config_bindings()
        except Exception as exc:
            raise W8CampaignHold(f"W8 AM-94 config compatibility differs: {exc}") from None
    _require(value["campaign"]["config_bindings"] == expected_configs, "W8 config bindings differ")
    _require(value["training"]["dataset"] == W8_DATASET, "W8 dataset binding differs")
    _require(value["training"]["lambda"] == 3.0 and value["training"]["lambda_status"] == "selected_at_G-4", "W8 training lambda differs")
    _require(value["training"]["train_snr_db"] == get("channel.train_snr_db_fixed"), "W8 train SNR differs")
    _require(value["training"]["epochs_per_run"] == W8_EPOCHS, "W8 epoch count differs")
    _require(value["training"]["fresh_initialization"]["predecessor_checkpoint_id"] is None, "W8 genesis predecessor is not fresh")
    _require(value["training"]["w7_checkpoint_transfer_forbidden"] is True and value["training"]["prior_w8_state_transfer_forbidden"] is True, "W8 state-transfer boundary differs")
    selection = value["checkpoint_selection"]
    _require(selection["split"] == "validation" and selection["metric"] == "validation_top1_accuracy" and selection["mode"] == "max" and selection["tie_break"] == "earliest_epoch", "W8 checkpoint rule differs")
    _require(selection["snr_parameter"] == "params.learned_system.checkpoint_selection_snr_db" and selection["snr_resolution"] == "params.channel.train_snr_db_fixed" and selection["snr_db"] == checkpoint_selection_snr_db(), "W8 checkpoint SNR authority differs")
    _require(selection["channel_seed_rule"] == "run_channel_seed" and selection["validation_denominator"] == W8_VALIDATION_SAMPLE_COUNT and selection["fixed_noise_across_epochs"] is True and selection["cross_seed_selection"] is False, "W8 checkpoint validation authority differs")
    profile = value["profile"]
    _require(profile["execution_profile_id"] == W8_PROFILE_ID and profile["scientific_writer_host"] == "confessor" and profile["execution_image_family"] == W8_EXECUTION_IMAGE_FAMILY, "W8 execution profile differs")
    _require(profile["gpu_name"] == W8_SELECTED_GPU_NAME and profile["gpu_uuid"] == W8_SELECTED_GPU_UUID and profile["device"] == "cuda:0", "W8 exact GPU differs")
    _require(profile["requirements_lock"] == "requirements-pascal.lock" and profile["requirements_lock_sha256"] == PASCAL_LOCK_SHA256, "W8 Pascal lock differs")
    _require(profile["physical_batch_size"] == W8_PHYSICAL_BATCH_SIZE and profile["accumulation_factor"] == W8_ACCUMULATION_FACTOR and profile["effective_batch_size"] == W8_EFFECTIVE_BATCH_SIZE and profile["validation_batch_size"] == W8_VALIDATION_BATCH_SIZE and profile["train_samples"] == W8_TRAIN_SAMPLE_COUNT and profile["drop_last"] is False, "W8 batch binding differs")
    _require(value["boundary"] == _expected_boundary(), "W8 forbidden boundary differs")
    _require(value["pre_execution_zero_counters"] == _expected_zero_counters(), "W8 pre-execution counters differ")
    _require(value["protocol_hash"] == canonical_sha256(protocol_descriptor()), "W8 authorization protocol digest differs")
    _require(value["source_contains_no_w8_results"] is True and value["campaign_root_created_at_freeze"] is False, "W8 source/root boundary differs")


def load_authority(authorization_path: Path, source_manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        authorization = verify_authorization(
            authorization_path,
            expected_source_manifest_path=source_manifest_path,
        )
        manifest = verify_manifest(
            source_manifest_path,
            expected_source_commit=authorization["scientific_source"]["source_commit"],
        )
    except Exception as exc:
        raise W8CampaignHold(f"W8 authority authentication failed: {exc}") from None
    _validate_authorization_contract(
        authorization, source_manifest=manifest, source_manifest_path=source_manifest_path
    )
    return authorization, manifest


def verify_launch_authorization(
    path: Path,
    *,
    w8_authorization: dict[str, Any],
    w8_authorization_path: Path,
    source_manifest: dict[str, Any],
    source_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Authenticate the distinct owner-issued W8-B launch permission."""

    value = _read_json(path, "W8-B launch authorization")
    required = {
        "schema_version", "artifact_role", "status", "authorization_scope",
        "authorization_id", "issued_at_utc", "w8_a_authorization_id",
        "w8_a_authorization_sha256", "source_commit", "source_manifest_id",
        "source_manifest_sha256", "campaign_id", "campaign_root", "profile",
        "scope", "test", "owner_authorization",
    }
    _require(set(value) == required, "W8-B launch authorization schema differs")
    body = dict(value)
    identifier = body.pop("authorization_id")
    _require(identifier == "w8blaunch-" + canonical_sha256(body), "W8-B launch authorization digest differs")
    _require(value["schema_version"] == 1 and value["artifact_role"] == LAUNCH_AUTHORIZATION_ROLE and value["status"] == "AUTHORIZED", "W8-B launch authorization role/status differs")
    _require(value["authorization_scope"] == "W8_SIX_CORE_RUNS_ONLY", "W8-B launch scope differs")
    _require(value["w8_a_authorization_id"] == w8_authorization["authorization_id"] and value["w8_a_authorization_sha256"] == _sha(w8_authorization_path), "W8-A authorization binding differs")
    _require(value["source_commit"] == source_manifest["source_commit"] and value["source_manifest_id"] == source_manifest["manifest_id"], "W8-B source binding differs")
    expected_manifest_sha = (
        _sha(source_manifest_path)
        if source_manifest_path is not None
        else w8_authorization["scientific_source"]["source_manifest_file_sha256"]
    )
    _require(value["source_manifest_sha256"] == expected_manifest_sha, "W8-B source manifest SHA differs")
    _require(value["campaign_id"] == CAMPAIGN_ID and value["campaign_root"] == CAMPAIGN_ROOT, "W8-B campaign binding differs")
    _require(value["profile"] == {
        "execution_profile_id": W8_PROFILE_ID,
        "gpu_name": W8_SELECTED_GPU_NAME,
        "gpu_uuid": W8_SELECTED_GPU_UUID,
        "device": "cuda:0",
        "requirements_lock": "requirements-pascal.lock",
        "requirements_lock_sha256": PASCAL_LOCK_SHA256,
        "physical_batch_size": W8_PHYSICAL_BATCH_SIZE,
        "accumulation_factor": W8_ACCUMULATION_FACTOR,
        "effective_batch_size": W8_EFFECTIVE_BATCH_SIZE,
        "validation_batch_size": W8_VALIDATION_BATCH_SIZE,
    }, "W8-B profile binding differs")
    _require(value["scope"] == {"core_runs": 6, "er2_randomized_training": False, "papr_constrained_training": False, "er9_training": False, "g10": False}, "W8-B scope opens forbidden work")
    _require(value["test"] == {"status": "SEALED", "model_facing_access": 0, "learned_inference": 0}, "W8-B test boundary differs")
    _require(value["owner_authorization"] is True, "W8-B owner authorization is absent")
    return value


def _verify_source_checkout(repo: Path, source_commit: str) -> None:
    _require(_git(repo, "rev-parse", "HEAD") == source_commit, "W8 scientific checkout HEAD differs from source authority")
    _require(_git(repo, "status", "--porcelain", "--untracked-files=all") == "", "W8 scientific checkout is dirty")


def _verify_free_space(root: Path) -> int:
    parent = root.parent if root.parent.exists() else Path("/")
    try:
        free = int(shutil.disk_usage(parent).free)
    except OSError as exc:
        raise W8CampaignHold(f"cannot measure W8 campaign free space: {exc}") from None
    minimum = W8_MIN_FREE_SPACE_GIB * (2**30)  # literal-ok: owner-required GiB-to-byte preflight conversion
    _require(free >= minimum, f"W8 campaign free space is below {W8_MIN_FREE_SPACE_GIB} GiB")
    return free


def preflight(
    *,
    repo: Path,
    authorization_path: Path,
    source_manifest_path: Path,
    require_data: bool,
) -> dict[str, Any]:
    authorization, manifest = load_authority(authorization_path, source_manifest_path)
    _verify_source_checkout(repo, authorization["scientific_source"]["source_commit"])
    _run_w7_g4_verifier(repo)
    _verify_test_boundary(repo)
    dataset = _verify_dataset(repo, require_extracted=require_data)
    root = Path(authorization["campaign"]["campaign_root"])
    _safe_root(root, allow_existing=require_data)
    free_space = _verify_free_space(root)
    return {
        "status": "PASS",
        "authorization_id": authorization["authorization_id"],
        "source_manifest_id": manifest["manifest_id"],
        "source_commit": manifest["source_commit"],
        "dataset": dataset,
        "campaign_root": str(root),
        "campaign_root_has_scientific_state": bool(root.exists() and any(root.iterdir())),
        "free_space_bytes": free_space,
        "minimum_free_space_gib": W8_MIN_FREE_SPACE_GIB,
        "test_model_facing_access": 0,
        "learned_test_inference": 0,
    }


def _campaign_manifest(
    authorization: dict[str, Any], manifest: dict[str, Any], launch: dict[str, Any]
) -> dict[str, Any]:
    body = {
        "schema_version": 1,
        "artifact_role": CAMPAIGN_MANIFEST_ROLE,
        "status": "FROZEN_BEFORE_FIRST_RUN",
        "campaign_id": CAMPAIGN_ID,
        "authorization_id": authorization["authorization_id"],
        "launch_authorization_id": launch["authorization_id"],
        "source_commit": manifest["source_commit"],
        "source_manifest_id": manifest["manifest_id"],
        "source_manifest_sha256": authorization["scientific_source"]["source_manifest_file_sha256"],
        "protocol_hash": authorization["protocol_hash"],
        "run_cells": [cell.to_dict() for cell in run_cells()],
        "run_order": "seed_major_then_ratio_minor",
        "execution_profile": dict(authorization["profile"]),
        "checkpoint_selection": dict(authorization["checkpoint_selection"]),
        "resume_rule": dict(authorization["resume_and_custody"]),
        "scientific_scope": dict(authorization["boundary"]),
        "test_model_facing_access": 0,
        "learned_test_inference": 0,
    }
    body["manifest_id"] = CAMPAIGN_MANIFEST_PREFIX + canonical_sha256(body)
    return body


def _load_epoch_summaries(root: Path) -> list[dict[str, Any]]:
    directory = root / "validation"
    if not directory.exists():
        return []
    _require(directory.is_dir() and not directory.is_symlink(), "W8 validation directory is unsafe")
    paths = sorted(directory.iterdir())
    expected: list[Path] = []
    values: list[dict[str, Any]] = []
    for path in paths:
        _require(path.is_file() and not path.is_symlink(), "W8 validation artifact is unsafe")
        _require(path.name.startswith("epoch-") and path.name.endswith(".json"), "W8 validation directory contains an unknown file")
        number = path.name.removeprefix("epoch-").removesuffix(".json")
        _require(len(number) == 4 and number.isdigit(), "W8 validation epoch filename is invalid")
        expected.append(path)
        values.append(_read_json(path, "W8 validation summary"))
    _require([value.get("epoch") for value in values] == list(range(len(values))), "W8 validation summaries are not an ordered prefix")
    return values


def _authenticate_summary_prefix(trainer: W8Trainer, summaries: list[dict[str, Any]]) -> None:
    for epoch, summary in enumerate(summaries):
        _validate_summary(
            summary,
            expected_epoch=epoch,
            expected_evaluation_config_hash=evaluation_config_hash(
                trainer, batch_size=W8_VALIDATION_BATCH_SIZE
            ),
        )
        _require(summary["campaign_id"] == trainer.campaign_id and summary["run_id"] == trainer.run_id, "W8 validation summary run differs")
        _require(summary["ratio"] == trainer.config.resolved["bw_ratio"] and summary["k"] == trainer.config.resolved["k"] and summary["train_seed"] == trainer.config.resolved["train_seed"] and summary["channel_seed"] == trainer.config.resolved["channel_seed"], "W8 validation summary config differs")
        sidecar_path = trainer.runtime_root / f"checkpoints/epoch-{epoch:04d}.sidecar.json"
        _require(sidecar_path.is_file() and not sidecar_path.is_symlink(), "W8 validation summary sidecar is missing")
        sidecar = _read_json(sidecar_path, "W8 validation summary sidecar")
        trainer._validate_sidecar(sidecar)
        _require(sidecar["completed_epoch"] == epoch and sidecar["checkpoint_id"] == summary["checkpoint_id"], "W8 validation summary checkpoint binding differs")


def _publish_validation(
    root: Path, trainer: W8Trainer, evaluation: Any
) -> None:
    summary = dict(evaluation.summary)
    _validate_summary(
        summary,
        expected_epoch=int(summary["epoch"]),
        expected_evaluation_config_hash=evaluation_config_hash(
            trainer, batch_size=int(summary["validation_batch_size"])
        ),
    )
    _publish_new_json(root / f"validation/epoch-{summary['epoch']:04d}.json", summary)


def _load_epoch_records(
    root: Path, trainer: W8Trainer | None = None
) -> list[dict[str, Any]]:
    """Load an exact ordered prefix and optionally authenticate it against a run."""

    directory = root / "epochs"
    _require(directory.is_dir() and not directory.is_symlink(), "W8 epoch directory is missing")
    paths = sorted(directory.iterdir())
    records: list[dict[str, Any]] = []
    previous_checkpoint: str | None = None
    previous_global_step = 0
    for index, path in enumerate(paths):
        _require(path.is_file() and not path.is_symlink(), "W8 epoch record is unsafe")
        _require(path.name == f"epoch-{index:04d}.json", "W8 epoch record prefix differs")
        value = _read_json(path, "W8 epoch record")
        record_id = value.pop("record_id", None)
        _require(isinstance(record_id, str) and record_id == canonical_sha256(value), "W8 epoch record digest differs")
        if trainer is not None:
            trainer._validate_epoch_record(
                value,
                record_id=record_id,
                expected_epoch=index,
                expected_predecessor=previous_checkpoint,
            )
        else:
            _require(value["artifact_role"] == W8_TRAINING_EPOCH_ROLE or value["artifact_role"] == "W8_NON_SCIENTIFIC_SMOKE_EPOCH_RECORD", "W8 epoch record role differs")
        # The checkpoint predecessor is authenticated by the matching sidecar.
        sidecar_path = root / f"checkpoints/epoch-{index:04d}.sidecar.json"
        if trainer is not None:
            _require(sidecar_path.is_file() and not sidecar_path.is_symlink(), "W8 epoch sidecar is missing")
            sidecar = _read_json(sidecar_path, "W8 epoch sidecar")
            trainer._validate_sidecar(sidecar)
            _require(sidecar["epoch_record_id"] == record_id, "W8 epoch/sidecar record binding differs")
            _require(sidecar["epoch_record_sha256"] == _sha(path), "W8 epoch/sidecar digest differs")
            _require(sidecar["completed_epoch"] == index, "W8 epoch/sidecar epoch differs")
            _require(value["global_optimizer_step"] == sidecar["global_optimizer_step"], "W8 epoch/sidecar optimizer step differs")
            _require(value["global_optimizer_step"] == previous_global_step + value["optimizer_steps"], "W8 epoch global optimizer chain differs")
            previous_global_step = int(value["global_optimizer_step"])
            previous_checkpoint = sidecar["checkpoint_id"]
        value["record_id"] = record_id
        records.append(value)
    return records


def _run_completion_value(
    trainer: W8Trainer,
    records: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    selection: dict[str, Any],
    selected: dict[str, Any],
) -> dict[str, Any]:
    optimizer_steps = sum(int(record["optimizer_steps"]) for record in records)
    skips = sum(int(record["grad_scaler_skips"]) for record in records)
    opportunities = sum(int(record["optimizer_step_opportunities"]) for record in records)
    _require(len(records) == W8_EPOCHS and len(summaries) == W8_EPOCHS, "W8 run completion evidence is incomplete")
    _require(optimizer_steps + skips == opportunities, "W8 run completion optimizer/skips accounting differs")
    _require(trainer.completed_epoch == W8_EPOCHS - 1 and trainer.global_optimizer_step == optimizer_steps, "W8 run completion trainer state differs")
    body = {
        "schema_version": 1,
        "artifact_role": RUN_COMPLETION_ROLE,
        "status": "COMPLETE",
        "authentication_status": "PASSED",
        "eligibility": eligibility_for_role("W8_FINAL_MULTI_SEED_RUN"),
        "campaign_id": trainer.campaign_id,
        "run_id": trainer.run_id,
        "config_hash": trainer.config_hash,
        "protocol_config_hash": trainer.protocol_hash,
        "source_commit": trainer.source_lineage.source_commit,
        "source_manifest_id": trainer.source_lineage.source_manifest_id,
        "source_manifest_sha256": trainer.source_lineage.source_manifest_sha256,
        "execution_profile_id": trainer.config.resolved["execution_profile_id"],
        "gpu_uuid": trainer.profile_binding["gpu_uuid"],
        "ratio": trainer.config.resolved["bw_ratio"],
        "k": trainer.config.resolved["k"],
        "train_seed": trainer.config.resolved["train_seed"],
        "channel_seed": trainer.config.resolved["channel_seed"],
        "lambda": trainer.config.resolved["lambda"],
        "train_snr_db": trainer.config.resolved["train_snr_db"],
        "epochs": W8_EPOCHS,
        "completed_epoch_cycles": len(records),
        "optimizer_step_opportunities": opportunities,
        "optimizer_steps": optimizer_steps,
        "grad_scaler_skips": skips,
        "global_optimizer_step": trainer.global_optimizer_step,
        "validation_summary_count": len(summaries),
        "selection": selection,
        "selected_checkpoint_id": selected["checkpoint_id"],
        "selected_checkpoint_epoch": selected["checkpoint_epoch"],
        "latest_checkpoint_id": trainer.predecessor_checkpoint_id,
        "selected_result_id": selected["result_id"],
        "selected_result_sha256": None,
        "protected_counters": {
            "w8_final_training_runs": 1,
            "w8_scientific_optimizer_steps": optimizer_steps,
            "g10_adjudications": 0,
            "er2_randomized_training": 0,
            "papr_constrained_training": 0,
            "er9_training": 0,
            "learned_test_inference": 0,
            "test_model_facing_access": 0,
        },
        "test_model_facing_access": 0,
        "learned_test_inference": 0,
    }
    return body


def _verify_run_completion(
    path: Path,
    trainer: W8Trainer,
    *,
    selected_path: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Reconcile a completed run before allowing the campaign to advance."""

    value = _read_json(path, "W8 run completion")
    required = {
        "schema_version", "artifact_role", "status", "authentication_status", "eligibility",
        "campaign_id", "run_id", "config_hash", "protocol_config_hash", "source_commit",
        "source_manifest_id", "source_manifest_sha256", "execution_profile_id", "gpu_uuid",
        "ratio", "k", "train_seed", "channel_seed", "lambda", "train_snr_db", "epochs",
        "completed_epoch_cycles", "optimizer_step_opportunities", "optimizer_steps",
        "grad_scaler_skips", "global_optimizer_step", "validation_summary_count", "selection",
        "selected_checkpoint_id", "selected_checkpoint_epoch", "latest_checkpoint_id",
        "selected_result_id", "selected_result_sha256", "protected_counters", "test_model_facing_access",
        "learned_test_inference", "completion_id",
    }
    _require(set(value) == required, "W8 run completion schema differs")
    body = dict(value)
    identifier = body.pop("completion_id")
    _require(identifier == RUN_COMPLETION_PREFIX + canonical_sha256(body), "W8 run completion digest differs")
    _require(value["schema_version"] == 1 and value["artifact_role"] == RUN_COMPLETION_ROLE and value["status"] == "COMPLETE" and value["authentication_status"] == "PASSED", "W8 run completion role/status differs")
    _require(value["eligibility"] == eligibility_for_role("W8_FINAL_MULTI_SEED_RUN"), "W8 run completion eligibility differs")
    for key, expected in {
        "campaign_id": trainer.campaign_id, "run_id": trainer.run_id, "config_hash": trainer.config_hash,
        "protocol_config_hash": trainer.protocol_hash, "source_commit": trainer.source_lineage.source_commit,
        "source_manifest_id": trainer.source_lineage.source_manifest_id, "source_manifest_sha256": trainer.source_lineage.source_manifest_sha256,
        "execution_profile_id": trainer.config.resolved["execution_profile_id"], "gpu_uuid": trainer.profile_binding["gpu_uuid"],
        "ratio": trainer.config.resolved["bw_ratio"], "k": trainer.config.resolved["k"], "train_seed": trainer.config.resolved["train_seed"],
        "channel_seed": trainer.config.resolved["channel_seed"], "lambda": trainer.config.resolved["lambda"], "train_snr_db": trainer.config.resolved["train_snr_db"],
    }.items():
        _require(value[key] == expected, f"W8 run completion {key} differs")
    _require(value["epochs"] == W8_EPOCHS and value["completed_epoch_cycles"] == W8_EPOCHS and value["validation_summary_count"] == W8_EPOCHS, "W8 run completion epoch coverage differs")
    _require(value["test_model_facing_access"] == 0 and value["learned_test_inference"] == 0, "W8 run completion claims test access")
    _require(path.is_file() and not path.is_symlink(), "W8 run completion path is unsafe")
    _require(selected_path.is_file() and not selected_path.is_symlink(), "W8 selected result is missing")
    _require(value["selected_result_sha256"] == _sha(selected_path), "W8 selected result SHA differs")
    selected = _read_json(selected_path, "W8 selected result")
    _require(selected.get("result_id") == value["selected_result_id"], "W8 selected result ID differs")
    validate_selected_checkpoint_result(
        selected,
        expected_evaluation_config_hash=evaluation_config_hash(
            trainer, batch_size=W8_VALIDATION_BATCH_SIZE
        ),
        expected_validation_ids=validation_stable_ids(repo_root=repo_root) if repo_root is not None else None,
    )
    _require(selected["campaign_id"] == trainer.campaign_id and selected["run_id"] == trainer.run_id, "W8 selected result run differs")
    _require(selected["checkpoint_id"] == value["selected_checkpoint_id"] and selected["checkpoint_epoch"] == value["selected_checkpoint_epoch"], "W8 selected checkpoint binding differs")
    _validate_selection(value["selection"])
    _require(value["selection"]["selected_checkpoint_id"] == value["selected_checkpoint_id"] and value["selection"]["selected_epoch"] == value["selected_checkpoint_epoch"], "W8 run completion selection differs")

    records = _load_epoch_records(path.parent, trainer)
    summaries = _load_epoch_summaries(path.parent)
    _require(len(records) == W8_EPOCHS and len(summaries) == W8_EPOCHS, "W8 completed run evidence prefix is incomplete")
    _authenticate_summary_prefix(trainer, summaries)
    trainer.resume()  # latest-only: corrupt latest is a HOLD, never an older fallback
    _require(trainer.completed_epoch == W8_EPOCHS - 1, "W8 completed run latest epoch differs")
    selected_sidecar = _read_json(path.parent / f"checkpoints/epoch-{value['selected_checkpoint_epoch']:04d}.sidecar.json", "W8 selected checkpoint sidecar")
    trainer._validate_sidecar(selected_sidecar)
    _require(selected_sidecar["checkpoint_id"] == value["selected_checkpoint_id"], "W8 selected checkpoint sidecar differs")
    trainer._load_authenticated_payload(selected_sidecar)
    latest_sidecar = _read_json(path.parent / f"checkpoints/epoch-{W8_EPOCHS - 1:04d}.sidecar.json", "W8 latest checkpoint sidecar")
    trainer._validate_sidecar(latest_sidecar)
    _require(value["latest_checkpoint_id"] == latest_sidecar["checkpoint_id"] == trainer.predecessor_checkpoint_id, "W8 latest checkpoint binding differs")

    optimizer_steps = sum(int(record["optimizer_steps"]) for record in records)
    skips = sum(int(record["grad_scaler_skips"]) for record in records)
    opportunities = sum(int(record["optimizer_step_opportunities"]) for record in records)
    _require(value["optimizer_steps"] == optimizer_steps and value["grad_scaler_skips"] == skips and value["optimizer_step_opportunities"] == opportunities, "W8 completed run optimizer accounting differs")
    _require(optimizer_steps + skips == opportunities, "W8 completed run optimizer/skips accounting differs")
    _require(value["global_optimizer_step"] == trainer.global_optimizer_step == optimizer_steps, "W8 completed run global optimizer step differs")
    _require(value["protected_counters"] == {
        "w8_final_training_runs": 1,
        "w8_scientific_optimizer_steps": optimizer_steps,
        "g10_adjudications": 0,
        "er2_randomized_training": 0,
        "papr_constrained_training": 0,
        "er9_training": 0,
        "learned_test_inference": 0,
        "test_model_facing_access": 0,
    }, "W8 completed run counters differ")
    return value


def _run_one_cell(
    *,
    authorization: dict[str, Any],
    manifest: dict[str, Any],
    root: Path,
    cell: Any,
    repo: Path,
    heartbeat: Path,
    completed_runs: int,
    completed_epoch_cycles: int,
) -> dict[str, Any]:
    config = load_w8_config(cell.ratio, cell.train_seed, cell.channel_seed)
    expected_binding = next(item for item in authorization["campaign"]["config_bindings"] if item["run_index"] == cell.run_index)
    _require(run_config_hash(config) == expected_binding["config_hash"] and protocol_config_hash(config) == expected_binding["protocol_config_hash"], "W8 cell config fingerprint differs")
    binding = authenticate_w8_gpu(config_hash=run_config_hash(config), expected_gpu_uuid=W8_SELECTED_GPU_UUID)
    verify_frozen_w8_gpu_binding(binding, config_hash=run_config_hash(config), source_commit=manifest["source_commit"])
    run_root = root / _campaign_run_root_name(cell)
    run_id = f"w8-{cell.ratio}-train{cell.train_seed}-channel{cell.channel_seed}"
    lineage = W8SourceLineage(manifest["source_commit"], manifest["manifest_id"], authorization["scientific_source"]["source_manifest_file_sha256"])
    trainer = W8Trainer(
        config,
        device="cuda:0",
        runtime_root=run_root,
        source_lineage=lineage,
        profile_binding=binding,
        campaign_id=CAMPAIGN_ID,
        run_id=run_id,
        num_workers=int(config.parameters["learned_system"]["dataloader_workers"]),
    )
    completion_path = run_root / "run_completion.json"
    selected_path = run_root / "selected_checkpoint.json"
    if completion_path.is_file() or completion_path.is_symlink():
        _require(not completion_path.is_symlink(), "W8 run completion path is unsafe")
        _require(selected_path.is_file() and not selected_path.is_symlink(), "W8 completed run lacks selected result")
        return _verify_run_completion(
            completion_path,
            trainer,
            selected_path=selected_path,
            repo_root=repo,
        )
    _require(not completion_path.exists() and not completion_path.is_symlink(), "W8 run completion path is unsafe")
    _require(not selected_path.exists() and not selected_path.is_symlink(), "W8 selected result exists without authenticated run completion")
    if run_root.exists() and any(run_root.iterdir()):
        latest_path = run_root / "latest.json"
        if latest_path.is_file() and not latest_path.is_symlink():
            trainer.resume()
        else:
            # A process may die during epoch-0 publication before the first
            # latest pointer exists.  Only the explicitly recognised,
            # unauthenticated genesis suffix may be discarded; all other
            # state was rejected by the trainer and remains a HOLD.
            trainer.discard_unpublished_genesis_suffix()
    summaries = _load_epoch_summaries(run_root)
    _require(len(summaries) <= trainer.completed_epoch + 1, "W8 validation summaries run past the authenticated checkpoint")
    _authenticate_summary_prefix(trainer, summaries)
    for epoch in range(len(summaries), trainer.completed_epoch + 1):
        sidecar = trainer.load_checkpoint_epoch(epoch)
        evaluation = evaluate_validation(trainer, checkpoint_id=sidecar["checkpoint_id"], repo_root=repo, retain_rows=False)
        _publish_validation(run_root, trainer, evaluation)
        summaries = _load_epoch_summaries(run_root)
    if trainer.completed_epoch >= 0:
        trainer.resume()
    while trainer.completed_epoch < W8_EPOCHS - 1:
        next_epoch = trainer.completed_epoch + 1
        write_heartbeat(
            heartbeat, campaign_id=CAMPAIGN_ID, current_run_index=cell.run_index,
            ratio=cell.ratio, train_seed=cell.train_seed, channel_seed=cell.channel_seed,
            current_epoch=next_epoch, process_state="TRAINING",
            latest_checkpoint_id=trainer.predecessor_checkpoint_id,
            completed_runs=completed_runs, completed_epoch_cycles=completed_epoch_cycles,
        )
        from data.djscc_training import TrainingDJSCCDataset

        dataset = TrainingDJSCCDataset(W8_DATASET, cell.train_seed, next_epoch, repo_root=repo)
        record = trainer.train_epoch(next_epoch, dataset)
        sidecar = trainer.save_checkpoint(record)
        trainer._validate_sidecar(sidecar)
        write_heartbeat(
            heartbeat, campaign_id=CAMPAIGN_ID, current_run_index=cell.run_index,
            ratio=cell.ratio, train_seed=cell.train_seed, channel_seed=cell.channel_seed,
            current_epoch=next_epoch, process_state="VALIDATING",
            latest_checkpoint_id=sidecar["checkpoint_id"],
            completed_runs=completed_runs, completed_epoch_cycles=completed_epoch_cycles + next_epoch + 1,
        )
        evaluation = evaluate_validation(trainer, checkpoint_id=sidecar["checkpoint_id"], repo_root=repo, retain_rows=False)
        _publish_validation(run_root, trainer, evaluation)
        summaries = _load_epoch_summaries(run_root)
        _authenticate_summary_prefix(trainer, summaries)
    _require(trainer.completed_epoch == W8_EPOCHS - 1, "W8 run did not reach 100 completed epochs")
    _require(len(summaries) == W8_EPOCHS, "W8 run lacks one validation summary per epoch")
    _authenticate_summary_prefix(trainer, summaries)
    selection = select_checkpoint_epoch(summaries, expected_epochs=W8_EPOCHS)
    sidecar = trainer.load_checkpoint_epoch(selection["selected_epoch"])
    selected = selected_checkpoint_result(trainer, selection=selection, repo_root=repo)
    _require(selected["checkpoint_id"] == sidecar["checkpoint_id"], "W8 selected checkpoint reauthentication differs")
    _publish_new_json(selected_path, selected)
    trainer.resume()  # completion accounting is for the final epoch, not the selected epoch
    completion = _run_completion_value(trainer, _load_epoch_records(run_root, trainer), summaries, selection, selected)
    completion["selected_result_sha256"] = _sha(selected_path)
    completion["completion_id"] = RUN_COMPLETION_PREFIX + canonical_sha256(completion)
    _publish_new_json(completion_path, completion)
    return completion


def _verify_campaign_completion(
    path: Path,
    *,
    authorization: dict[str, Any],
    manifest: dict[str, Any],
    launch: dict[str, Any],
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    value = _read_json(path, "W8 campaign completion")
    body = dict(value)
    identifier = body.pop("completion_id", None)
    _require(
        isinstance(identifier, str)
        and identifier == CAMPAIGN_COMPLETION_PREFIX + canonical_sha256(body),
        "W8 campaign completion digest differs",
    )
    expected = _campaign_completion(authorization, manifest, launch, runs)
    _require(value == expected, "W8 campaign completion differs")
    return value


def _campaign_completion(
    authorization: dict[str, Any], manifest: dict[str, Any], launch: dict[str, Any], runs: list[dict[str, Any]]
) -> dict[str, Any]:
    body = {
        "schema_version": 1,
        "artifact_role": CAMPAIGN_COMPLETION_ROLE,
        "status": "COMPLETE_NOT_YET_RECONCILED",
        "campaign_id": CAMPAIGN_ID,
        "authorization_id": authorization["authorization_id"],
        "launch_authorization_id": launch["authorization_id"],
        "source_commit": manifest["source_commit"],
        "source_manifest_id": manifest["manifest_id"],
        "run_count": len(runs),
        "run_order": [run["run_id"] for run in runs],
        "selected_checkpoints": [
            {"run_id": run["run_id"], "checkpoint_id": run["selected_checkpoint_id"], "epoch": run["selected_checkpoint_epoch"]}
            for run in runs
        ],
        "w8_final_training_runs": len(runs),
        "w8_completed_runs": len(runs),
        "w8_scientific_checkpoints": sum(int(run["validation_summary_count"]) for run in runs),
        "w8_scientific_optimizer_steps": sum(int(run["protected_counters"]["w8_scientific_optimizer_steps"]) for run in runs),
        "g10_adjudications": 0,
        "er2_randomized_training": 0,
        "papr_constrained_training": 0,
        "er9_training": 0,
        "test_model_facing_access": 0,
        "learned_test_inference": 0,
        "reconciliation": "REQUIRED_BEFORE_DOWNSTREAM_VALIDATION_OR_G10",
        "test": "SEALED",
    }
    body["completion_id"] = CAMPAIGN_COMPLETION_PREFIX + canonical_sha256(body)
    return body


def start(
    *,
    repo: Path,
    authorization_path: Path,
    source_manifest_path: Path,
    launch_path: Path,
) -> int:
    # This call is the only path that may create the scientific root, and it
    # cannot proceed without the detached owner-issued W8-B artifact.
    preflight(
        repo=repo, authorization_path=authorization_path,
        source_manifest_path=source_manifest_path, require_data=True,
    )
    authorization, manifest = load_authority(authorization_path, source_manifest_path)
    launch = verify_launch_authorization(
        launch_path,
        w8_authorization=authorization,
        w8_authorization_path=authorization_path,
        source_manifest=manifest,
        source_manifest_path=source_manifest_path,
    )
    _require(launch["source_manifest_sha256"] == authorization["scientific_source"]["source_manifest_file_sha256"], "W8-B source manifest file binding differs")
    root = Path(authorization["campaign"]["campaign_root"])
    heartbeat = Path(authorization["campaign"]["heartbeat_path"])
    lock = W8CampaignLock(
        campaign_id=CAMPAIGN_ID, source_commit=manifest["source_commit"],
        execution_image=W8_EXECUTION_IMAGE_FAMILY, gpu_uuid=W8_SELECTED_GPU_UUID,
        lock_path=W8_GLOBAL_LOCK_PATH,
    )
    runs: list[dict[str, Any]] = []
    # Acquire before the first operational heartbeat or campaign-root mutation.
    # A competing process therefore fails before it can write any W8 state.
    with lock:
        # Authenticate the exact scientific GPU before creating even the
        # operational campaign namespace.  A wrong-device or wrong-lock
        # launch must leave no misleading W8 root behind.
        first_cell = run_cells()[0]
        first_config = load_w8_config(
            first_cell.ratio, first_cell.train_seed, first_cell.channel_seed
        )
        first_config_hash = run_config_hash(first_config)
        first_binding = authenticate_w8_gpu(
            config_hash=first_config_hash,
            expected_gpu_uuid=W8_SELECTED_GPU_UUID,
        )
        verify_frozen_w8_gpu_binding(
            first_binding,
            config_hash=first_config_hash,
            source_commit=manifest["source_commit"],
        )
        _safe_root(root, allow_existing=True)
        root.mkdir(parents=True, exist_ok=True)
        write_heartbeat(
            heartbeat, campaign_id=CAMPAIGN_ID, current_run_index=None, ratio=None,
            train_seed=None, channel_seed=None, current_epoch=None,
            process_state="LOCK_ACQUIRED", latest_checkpoint_id=None,
            completed_runs=0, completed_epoch_cycles=0,
        )
        campaign_manifest = _campaign_manifest(authorization, manifest, launch)
        campaign_manifest_path = root / "campaign_manifest.json"
        if campaign_manifest_path.is_file():
            _require(
                not campaign_manifest_path.is_symlink()
                and _read_json(campaign_manifest_path, "W8 campaign manifest") == campaign_manifest,
                "W8 campaign manifest differs; supersession required",
            )
        else:
            _publish_new_json(campaign_manifest_path, campaign_manifest)

        cells = run_cells()
        # Authenticate the already-completed prefix first.  This is the resume
        # boundary: those runs are never rerun, and no later run can exist
        # before its predecessor is authenticated.
        prefix = 0
        for cell in cells:
            run_root = root / _campaign_run_root_name(cell)
            completion_path = run_root / "run_completion.json"
            if not completion_path.exists() and not completion_path.is_symlink():
                break
            _require(prefix == cell.run_index - 1, "W8 completed-run prefix is not ordered")
            run = _run_one_cell(
                authorization=authorization, manifest=manifest, root=root, cell=cell,
                repo=repo, heartbeat=heartbeat, completed_runs=len(runs),
                completed_epoch_cycles=sum(int(item["completed_epoch_cycles"]) for item in runs),
            )
            runs.append(run)
            prefix += 1

        completion_path = root / "campaign_completion.json"
        if completion_path.exists() or completion_path.is_symlink():
            _require(prefix == len(cells), "W8 campaign completion exists before all runs")
            _verify_campaign_completion(
                completion_path, authorization=authorization, manifest=manifest,
                launch=launch, runs=runs,
            )
            write_heartbeat(
                heartbeat, campaign_id=CAMPAIGN_ID, current_run_index=None, ratio=None,
                train_seed=None, channel_seed=None, current_epoch=None,
                process_state="COMPLETE_NOT_YET_RECONCILED", latest_checkpoint_id=None,
                completed_runs=len(runs),
                completed_epoch_cycles=sum(int(item["completed_epoch_cycles"]) for item in runs),
            )
        else:
            for index in range(prefix, len(cells)):
                cell = cells[index]
                for future in cells[index + 1 :]:
                    future_root = root / _campaign_run_root_name(future)
                    _require(
                        not future_root.exists() or not any(future_root.iterdir()),
                        "W8 future run has state before its frozen predecessor",
                    )
                write_heartbeat(
                    heartbeat, campaign_id=CAMPAIGN_ID, current_run_index=cell.run_index,
                    ratio=cell.ratio, train_seed=cell.train_seed, channel_seed=cell.channel_seed,
                    current_epoch=None, process_state="RUN_START", latest_checkpoint_id=None,
                    completed_runs=len(runs),
                    completed_epoch_cycles=sum(int(item["completed_epoch_cycles"]) for item in runs),
                )
                run = _run_one_cell(
                    authorization=authorization, manifest=manifest, root=root, cell=cell,
                    repo=repo, heartbeat=heartbeat, completed_runs=len(runs),
                    completed_epoch_cycles=sum(int(item["completed_epoch_cycles"]) for item in runs),
                )
                runs.append(run)
                write_heartbeat(
                    heartbeat, campaign_id=CAMPAIGN_ID, current_run_index=cell.run_index,
                    ratio=cell.ratio, train_seed=cell.train_seed, channel_seed=cell.channel_seed,
                    current_epoch=W8_EPOCHS - 1, process_state="RUN_COMPLETE",
                    latest_checkpoint_id=run["latest_checkpoint_id"], completed_runs=len(runs),
                    completed_epoch_cycles=sum(int(item["completed_epoch_cycles"]) for item in runs),
                )
            _require(len(runs) == len(cells), "W8 campaign did not complete exactly six runs")
            completion = _campaign_completion(authorization, manifest, launch, runs)
            _publish_new_json(completion_path, completion)
            write_heartbeat(
                heartbeat, campaign_id=CAMPAIGN_ID, current_run_index=None, ratio=None,
                train_seed=None, channel_seed=None, current_epoch=None,
                process_state="COMPLETE_NOT_YET_RECONCILED", latest_checkpoint_id=None,
                completed_runs=len(runs),
                completed_epoch_cycles=sum(int(item["completed_epoch_cycles"]) for item in runs),
            )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "start"))
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--launch-authorization", type=Path)
    parser.add_argument("--repo", type=Path, default=REPO)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo = args.repo.resolve()
    authorization = args.authorization.resolve()
    source_manifest = args.source_manifest.resolve()
    try:
        if args.command == "preflight":
            value = preflight(
                repo=repo, authorization_path=authorization,
                source_manifest_path=source_manifest, require_data=False,
            )
            print(json.dumps(value, sort_keys=True))
            return 0
        _require(args.launch_authorization is not None, "W8-B start requires a separate launch authorization")
        return start(
            repo=repo, authorization_path=authorization,
            source_manifest_path=source_manifest,
            launch_path=args.launch_authorization.resolve(),
        )
    except (W8CampaignHold, W8Hold, W8ValidationHold, W8LockBusy, ValueError, KeyError, OSError) as exc:
        print(json.dumps({"status": "HOLD", "reason": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
