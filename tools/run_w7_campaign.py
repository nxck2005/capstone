#!/usr/bin/env python3
"""Detached future W7-B campaign launcher (never run during W7-A).

One process owns the global flock and runs the five candidate roots in the
frozen order.  The command intentionally requires a separate execution
authorization artifact, which does not exist in W7-A.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from adjudication.w7_g4 import CANDIDATE_ELIGIBILITY, validate_candidate  # noqa: E402
from config.params import get  # noqa: E402
from config.run_config import config_hash as run_config_hash  # noqa: E402
from config.w7_execution import authenticate_w7_gpu, verify_frozen_gpu_binding  # noqa: E402
from evaluation.w7_validation import evaluate_validation, selected_checkpoint_result, select_checkpoint_epoch  # noqa: E402
from verify_w7_b1 import (  # noqa: E402
    AUTHORIZATION_ROLE,
    verify_authorization_path,
    verify_scientific_checkout,
    verify_source_path,
)
from training.deterministic_core import canonical_bytes, canonical_sha256  # noqa: E402
from training.w7_g4 import W7_G4_PILOT_POLICY, W7SourceLineage, W7Trainer  # noqa: E402
from training.w7_protocol import (
    W7_CALIBRATION_SNR_DB,
    W7_CHANNEL_SEED,
    W7_EXECUTION_IMAGE_FAMILY,
    W7_LAMBDA_GRID,
    W7_RATIO,
    W7_SELECTED_GPU_UUID,
    W7_TRAIN_SEED,
    W7_TRAINING_SNR_DB,
    W7_VALIDATION_BATCH_SIZE,
    W7_VALIDATION_NOISE_POLICY,
    load_w7_config,
    protocol_config_hash,
)  # noqa: E402
from verify_w7_a import verify_profile_freeze  # noqa: E402
from runtime.w7_lock import W7CampaignLock  # noqa: E402


# Compatibility name for the existing campaign-state tests; this is the B1
# successor verifier, never the historical gen_w7_source_manifest verifier.
verify_source_manifest = verify_source_path


CAMPAIGN_MANIFEST_ROLE = "W7_G4_CAMPAIGN_MANIFEST"
CAMPAIGN_ROLE = "W7_G4_CAMPAIGN_COMPLETE_NOT_ADJUDICATED"
HEARTBEAT_ROLE = "W7_OPERATIONAL_HEARTBEAT"
W7_A_COMPLETION_PATH = REPO / "results/learned/w7/w7_a_completion.json"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    """Publish one immutable campaign record without replacing a predecessor."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"immutable W7 campaign record already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            raise RuntimeError(f"immutable W7 campaign record already exists: {path}") from None
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_heartbeat(path: Path, *, campaign_id: str, lambda_value: float | None, epoch: int | None, state: str, checkpoint_id: str | None) -> None:
    value = {
        "schema_version": 1,
        "artifact_role": HEARTBEAT_ROLE,
        "campaign_id": campaign_id,
        "current_lambda": lambda_value,
        "current_epoch": epoch,
        "process_state": state,
        "checkpoint_id": checkpoint_id,
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RuntimeError(f"W7 heartbeat path is unsafe: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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


def _load_authorization(path: Path) -> dict[str, Any]:
    try:
        return verify_authorization_path(path, repo_root=REPO, verify_source=False)
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(str(exc)) from None


def _verify_upstream() -> None:
    for tool in ("tools/verify_w5_training_system.py", "tools/verify_w6_complete.py"):
        subprocess.run([sys.executable, str(REPO / tool)], cwd=REPO, check=True)


def _campaign_manifest(
    *,
    authorization: dict[str, Any],
    source_manifest: dict[str, Any],
    profile_freeze: dict[str, Any],
    campaign_id: str,
    source_commit: str,
    source_manifest_sha256: str,
    profile_freeze_sha256: str,
    execution_image: str,
    gpu_uuid: str,
    physical_batch: int,
    accumulation_factor: int,
    validation_batch: int,
) -> dict[str, Any]:
    body = {
        "schema_version": 1,
        "artifact_role": CAMPAIGN_MANIFEST_ROLE,
        "status": "FROZEN_BEFORE_FIRST_CANDIDATE",
        "campaign_id": campaign_id,
        "authorization_id": authorization["authorization_id"],
        "w7_a_completion_id": authorization["w7_a_completion_id"],
        "w7_test_hardening_completion_id": authorization["w7_test_hardening_completion_id"],
        "w7_test_hardening_completion_sha256": authorization["w7_test_hardening_completion_sha256"],
        "source_commit": source_commit,
        "execution_source_commit": source_manifest["source_commit"],
        "source_manifest_id": source_manifest["manifest_id"],
        "source_manifest_sha256": source_manifest_sha256,
        "profile_freeze_id": profile_freeze["profile_freeze_id"],
        "profile_freeze_sha256": profile_freeze_sha256,
        "execution_image_family": execution_image,
        "execution_profile_id": profile_freeze["execution_profile_id"],
        "gpu_uuid": gpu_uuid,
        "lambda_grid": list(W7_LAMBDA_GRID),
        "lambda_order": "exact_configured_lambda_grid_order",
        "train_seed": W7_TRAIN_SEED,
        "channel_seed": W7_CHANNEL_SEED,
        "training_snr_db": W7_TRAINING_SNR_DB,
        "calibration_snr_db": W7_CALIBRATION_SNR_DB,
        "ratio": W7_RATIO,
        "physical_batch_size": physical_batch,
        "accumulation_factor": accumulation_factor,
        "effective_batch_size": physical_batch * accumulation_factor,
        "validation_batch_size": validation_batch,
        "scientific_execution_authorization": "PRESENT",
        "g4_adjudication_run": 0,
        "lambda_core_updated": False,
    }
    body["manifest_id"] = "w7campaignmanifest-" + canonical_sha256(body)
    return body


def _load_epoch_summaries(root: Path) -> list[dict[str, Any]]:
    validation_root = root / "validation"
    paths = sorted(validation_root.glob("epoch-*.json")) if validation_root.is_dir() else []
    values: list[dict[str, Any]] = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("W7 validation summary is unsafe")
        try:
            value = json.loads(path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise RuntimeError("W7 validation summary is corrupt") from None
        if not isinstance(value, dict):
            raise RuntimeError("W7 validation summary is not a mapping")
        values.append(value)
    return values


def _authenticate_summary_prefix(trainer: W7Trainer, summaries: list[dict[str, Any]]) -> None:
    """Bind every retained summary to its immutable checkpoint sidecar."""

    expected_epochs = list(range(len(summaries)))
    if [summary.get("epoch") for summary in summaries] != expected_epochs:
        raise RuntimeError("W7 validation summaries are not an exact ordered prefix")
    for summary in summaries:
        required = {
            "schema_version", "artifact_role", "epoch", "checkpoint_id", "n_correct",
            "n_total", "top1_accuracy", "prediction_digest", "evaluation_config_hash",
            "noise_policy", "noise_policy_hash", "noise_id_digest", "row_digest", "summary_id",
        }
        if set(summary) != required:
            raise RuntimeError("W7 validation summary schema differs")
        if summary["schema_version"] != 1 or summary["artifact_role"] != "W7_VALIDATION_EPOCH_SUMMARY":
            raise RuntimeError("W7 validation summary role differs")
        summary_body = dict(summary)
        summary_id = summary_body.pop("summary_id", None)
        if summary_id != canonical_sha256(summary_body):
            raise RuntimeError("W7 validation summary digest differs")
        expected_total = int(get("datasets.imagenette160.val_images"))
        if summary["n_total"] != expected_total:
            raise RuntimeError("W7 validation denominator differs from the committed split")
        if not isinstance(summary["n_correct"], int) or isinstance(summary["n_correct"], bool) or not 0 <= summary["n_correct"] <= expected_total:
            raise RuntimeError("W7 validation correct-count is invalid")
        if summary["top1_accuracy"] != summary["n_correct"] / summary["n_total"]:
            raise RuntimeError("W7 validation top-1 is not count-derived")
        sidecar_path = trainer.runtime_root / f"checkpoints/epoch-{summary['epoch']:04d}.sidecar.json"
        if sidecar_path.is_symlink() or not sidecar_path.is_file():
            raise RuntimeError("W7 validation summary checkpoint sidecar is missing or unsafe")
        try:
            sidecar = json.loads(sidecar_path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise RuntimeError("W7 validation summary checkpoint sidecar is corrupt") from None
        trainer._validate_sidecar(sidecar)
        if sidecar["completed_epoch"] != summary["epoch"]:
            raise RuntimeError("W7 validation summary checkpoint epoch differs")
        if sidecar["checkpoint_id"] != summary["checkpoint_id"]:
            raise RuntimeError("W7 validation summary checkpoint binding differs")
        if summary["noise_policy"] != W7_VALIDATION_NOISE_POLICY:
            raise RuntimeError("W7 validation summary noise policy differs")


def _publish_validation(root: Path, evaluation: Any) -> None:
    summary = dict(evaluation.summary)
    path = root / f"validation/epoch-{summary['epoch']:04d}.json"
    _write_new_json(path, summary)


def _candidate_lineage(
    *,
    config: Any,
    source_lineage: W7SourceLineage,
    gpu_uuid: str,
    expected_epochs: int,
) -> dict[str, Any]:
    resolved = config.resolved
    return {
        "protocol_version": "w7-g4-pre-execution-v1",
        "source_commit": source_lineage.source_commit,
        "source_manifest_id": source_lineage.source_manifest_id,
        "source_manifest_sha256": source_lineage.source_manifest_sha256,
        "protocol_config_hash": protocol_config_hash(config),
        "execution_image": source_lineage.execution_image,
        "execution_profile_id": resolved["execution_profile_id"],
        "gpu_uuid": gpu_uuid,
        "dataset": resolved["dataset"],
        "split_manifest_hash": config.parameters["datasets"][resolved["dataset"]]["manifest_sha256"],
        "architecture": resolved["architecture"],
        "ratio": resolved["bw_ratio"],
        "k": resolved["k"],
        "train_seed": resolved["train_seed"],
        "channel_seed": resolved["channel_seed"],
        "train_snr_db": resolved["train_snr_db"],
        "epochs": expected_epochs,
        "optimizer": config.parameters["learned_system"]["optimizer"],
        "scheduler": config.parameters["learned_system"]["lr_schedule"],
        "checkpoint_selection": {
            "metric": "top1_accuracy",
            "mode": "max",
            "tie_break": "earliest_epoch",
            "snr_db": W7_CALIBRATION_SNR_DB,
        },
        "validation_noise_policy": W7_VALIDATION_NOISE_POLICY,
    }


def _authenticate_completed_candidate(
    *,
    trainer: W7Trainer,
    config: Any,
    root: Path,
    candidate: dict[str, Any],
    source_lineage: W7SourceLineage,
    gpu_uuid: str,
    expected_epochs: int,
) -> dict[str, Any]:
    """Skip only a complete candidate whose entire local lineage authenticates."""

    validated = validate_candidate(candidate)
    if validated["lambda"] != float(config.resolved["lambda"]):
        raise RuntimeError("existing W7 candidate lambda differs from its root")
    if validated["lineage"] != _candidate_lineage(
        config=config,
        source_lineage=source_lineage,
        gpu_uuid=gpu_uuid,
        expected_epochs=expected_epochs,
    ):
        raise RuntimeError("existing W7 candidate lineage differs from the current campaign")
    trainer.resume()
    if trainer.completed_epoch != expected_epochs - 1:
        raise RuntimeError("existing W7 candidate is not complete")
    summaries = _load_epoch_summaries(root)
    _authenticate_summary_prefix(trainer, summaries)
    selection = select_checkpoint_epoch(summaries, expected_epochs=expected_epochs)
    if validated["selected_validation"] != {
        "checkpoint_id": selection["selected_checkpoint_id"],
        "epoch": selection["selected_epoch"],
        "n_correct": selection["n_correct"],
        "n_total": selection["n_total"],
        "top1_accuracy": selection["top1_accuracy"],
    }:
        raise RuntimeError("existing W7 candidate selection differs from its validation history")
    evidence = validated["selected_evidence"]
    evidence_path = Path(str(evidence["path"]))
    if evidence_path.is_absolute() or ".." in evidence_path.parts or evidence_path != Path("selected_checkpoint_result.json"):
        raise RuntimeError("existing W7 selected evidence path is unsafe")
    selected_path = root / evidence_path
    if selected_path.is_symlink() or not selected_path.is_file():
        raise RuntimeError("existing W7 selected evidence is missing or unsafe")
    if hashlib.sha256(selected_path.read_bytes()).hexdigest() != evidence["file_sha256"]:
        raise RuntimeError("existing W7 selected evidence hash differs")
    selected = json.loads(selected_path.read_bytes())
    if not isinstance(selected, dict):
        raise RuntimeError("existing W7 selected evidence is not a mapping")
    selected_digest = selected.get("result_digest")
    selected_body = dict(selected)
    selected_body.pop("result_digest", None)
    if selected_digest != canonical_sha256(selected_body) or selected_digest != validated["selected_validation_result_digest"]:
        raise RuntimeError("existing W7 selected evidence digest differs")
    if selected.get("selection") != selection or selected.get("checkpoint_id") != selection["selected_checkpoint_id"]:
        raise RuntimeError("existing W7 selected evidence selection differs")
    if selected.get("calibration_validation") != summaries[selection["selected_epoch"]]:
        raise RuntimeError("existing W7 selected evidence calibration summary differs")
    sidecar = trainer.load_checkpoint_epoch(selection["selected_epoch"])
    if sidecar["checkpoint_id"] != selection["selected_checkpoint_id"]:
        raise RuntimeError("existing W7 selected checkpoint differs")
    expected_psnr = selected.get("psnr_evaluation")
    if not isinstance(expected_psnr, dict) or validated["psnr_evaluation"] != {
        "snr_db": expected_psnr.get("snr_db"),
        "denominator": expected_psnr.get("denominator"),
        "psnr_db": expected_psnr.get("psnr_db"),
        "data_range": expected_psnr.get("data_range"),
        "per_image_digest": expected_psnr.get("per_image_digest"),
    }:
        raise RuntimeError("existing W7 candidate PSNR evidence differs")
    return validated


def _candidate(
    *,
    config: Any,
    root: Path,
    source_lineage: W7SourceLineage,
    gpu_uuid: str,
    profile_binding: dict[str, Any],
    repo_root: Path | None,
    heartbeat: Path,
    campaign_id: str,
) -> dict[str, Any]:
    trainer = W7Trainer(
        config,
        device="cuda:0",
        runtime_root=root,
        source_lineage=source_lineage,
        profile_binding=profile_binding,
        policy=W7_G4_PILOT_POLICY,
    )
    expected_epochs = int(config.parameters["learned_system"]["epochs"][config.resolved["dataset"]])
    candidate_path = root / "candidate_completion.json"
    if candidate_path.is_symlink():
        raise RuntimeError("W7 candidate completion is a symlink")
    if candidate_path.is_file():
        try:
            existing = json.loads(candidate_path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise RuntimeError("W7 candidate completion is corrupt") from None
        if not isinstance(existing, dict):
            raise RuntimeError("W7 candidate completion is not a mapping")
        return _authenticate_completed_candidate(
            trainer=trainer,
            config=config,
            root=root,
            candidate=existing,
            source_lineage=source_lineage,
            gpu_uuid=gpu_uuid,
            expected_epochs=expected_epochs,
        )
    if root.exists():
        if (root / "latest.json").is_file():
            trainer.resume()
        elif any(root.iterdir()):
            raise RuntimeError(f"incomplete W7 candidate has no authenticated latest pointer: {root}")
    summaries = _load_epoch_summaries(root)
    if len(summaries) > trainer.completed_epoch + 1:
        raise RuntimeError("W7 validation summaries run past the authenticated checkpoint")
    _authenticate_summary_prefix(trainer, summaries)
    # A crash after checkpoint publication but before validation must replay
    # only the missing validation summaries, not retrain an authenticated epoch.
    for epoch in range(len(summaries), trainer.completed_epoch + 1):
        sidecar = trainer.load_checkpoint_epoch(epoch)
        evaluation = evaluate_validation(trainer, checkpoint_id=sidecar["checkpoint_id"], repo_root=repo_root, retain_rows=False)
        _publish_validation(root, evaluation)
        summaries = _load_epoch_summaries(root)
    if summaries:
        _authenticate_summary_prefix(trainer, summaries)
        trainer.resume()
    while trainer.completed_epoch < expected_epochs - 1:
        next_epoch = trainer.completed_epoch + 1
        _write_heartbeat(heartbeat, campaign_id=campaign_id, lambda_value=float(config.resolved["lambda"]), epoch=next_epoch, state="TRAINING", checkpoint_id=trainer.predecessor_checkpoint_id)
        record = trainer.train_epoch(
            next_epoch,
            __import__("data.djscc_training", fromlist=["TrainingDJSCCDataset"]).TrainingDJSCCDataset(
                str(config.resolved["dataset"]), int(config.resolved["train_seed"]), next_epoch, repo_root=repo_root
            ),
        )
        sidecar = trainer.save_checkpoint(record)
        _write_heartbeat(heartbeat, campaign_id=campaign_id, lambda_value=float(config.resolved["lambda"]), epoch=next_epoch, state="VALIDATING", checkpoint_id=sidecar["checkpoint_id"])
        evaluation = evaluate_validation(trainer, checkpoint_id=sidecar["checkpoint_id"], repo_root=repo_root, retain_rows=False)
        _publish_validation(root, evaluation)
        summaries = _load_epoch_summaries(root)
    if len(summaries) != expected_epochs:
        raise RuntimeError("W7 candidate lacks one validation summary per completed epoch")
    _authenticate_summary_prefix(trainer, summaries)
    selection = select_checkpoint_epoch(summaries, expected_epochs=expected_epochs)
    _write_heartbeat(heartbeat, campaign_id=campaign_id, lambda_value=float(config.resolved["lambda"]), epoch=selection["selected_epoch"], state="FINAL_EVALUATION", checkpoint_id=selection["selected_checkpoint_id"])
    selected = selected_checkpoint_result(trainer, selection=selection, repo_root=repo_root)
    selected_path = root / "selected_checkpoint_result.json"
    if selected_path.exists() or selected_path.is_symlink():
        raise RuntimeError("orphan W7 selected evidence exists without a candidate completion")
    _write_new_json(selected_path, selected)
    selected_file_sha = hashlib.sha256(selected_path.read_bytes()).hexdigest()
    candidate = {
        "schema_version": 1,
        "artifact_role": "W7_G4_LAMBDA_CANDIDATE_COMPLETION",
        "candidate_id": "w7candidate-" + canonical_sha256({"lambda": config.resolved["lambda"], "selected": selected["result_digest"]}),
        "status": "COMPLETE",
        "authentication_status": "PASSED",
        "eligibility": dict(CANDIDATE_ELIGIBILITY),
        "lambda": float(config.resolved["lambda"]),
        "lineage": _candidate_lineage(
            config=config,
            source_lineage=source_lineage,
            gpu_uuid=gpu_uuid,
            expected_epochs=expected_epochs,
        ),
        "selected_validation": {
            "checkpoint_id": selected["checkpoint_id"],
            "epoch": selected["checkpoint_epoch"],
            "n_correct": selected["selection"]["n_correct"],
            "n_total": selected["selection"]["n_total"],
            "top1_accuracy": selected["selection"]["top1_accuracy"],
        },
        "psnr_evaluation": {
            "snr_db": selected["psnr_evaluation"]["snr_db"],
            "denominator": selected["psnr_evaluation"]["denominator"],
            "psnr_db": selected["psnr_evaluation"]["psnr_db"],
            "data_range": selected["psnr_evaluation"]["data_range"],
            "per_image_digest": selected["psnr_evaluation"]["per_image_digest"],
        },
        "selected_validation_result_digest": selected["result_digest"],
        "selected_evidence": {
            "path": "selected_checkpoint_result.json",
            "result_digest": selected["result_digest"],
            "file_sha256": selected_file_sha,
        },
        "test_access": 0,
    }
    validate_candidate(candidate)
    _write_new_json(candidate_path, candidate)
    return candidate


def run(args: argparse.Namespace) -> int:
    authorization = _load_authorization(args.authorization)
    if authorization["campaign_id"] != args.campaign_id:
        raise RuntimeError("W7 campaign ID differs from execution authorization")
    if args.execution_image != authorization["execution_image_family"] or args.gpu_uuid != authorization["gpu_uuid"]:
        raise RuntimeError("W7 launch GPU/image differs from execution authorization")
    if args.gpu_uuid != W7_SELECTED_GPU_UUID:
        raise RuntimeError("W7 launch GPU differs from the frozen Pascal selection")
    manifest = verify_source_manifest(args.source_manifest, current=True, repo_root=REPO)
    manifest_sha256 = _sha256_path(args.source_manifest)
    if manifest["manifest_id"] != authorization["source_manifest_id"] or manifest_sha256 != authorization["source_manifest_sha256"] or manifest["source_commit"] != authorization["source_commit"]:
        raise RuntimeError("W7 source manifest differs from authorization")
    profile_freeze = verify_profile_freeze(json.loads(args.profile_freeze.read_bytes()))
    profile_freeze_sha256 = _sha256_path(args.profile_freeze)
    if profile_freeze.get("status") != "FROZEN" or profile_freeze.get("gpu_uuid") != args.gpu_uuid:
        raise RuntimeError("W7 Pascal profile freeze differs from authorization/launch")
    if profile_freeze.get("profile_freeze_id") != authorization["profile_freeze_id"] or profile_freeze_sha256 != authorization["profile_freeze_sha256"]:
        raise RuntimeError("W7 profile freeze differs from authorization")
    physical_batch = int(profile_freeze["physical_batch_size"])
    accumulation_factor = int(profile_freeze["accumulation_factor"])
    validation_batch = int(profile_freeze["validation_batch_size"])
    target_batch = int(get("learned_system.batch_size.imagenette160"))
    if physical_batch * accumulation_factor != target_batch or validation_batch != W7_VALIDATION_BATCH_SIZE:
        raise RuntimeError("W7 profile freeze batch policy differs from the frozen protocol")
    if args.execution_image != W7_EXECUTION_IMAGE_FAMILY:
        raise RuntimeError("W7 execution image family differs from the frozen contract")
    _verify_upstream()
    try:
        source_commit = verify_scientific_checkout(manifest["source_commit"], repo_root=REPO)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(str(exc)) from None
    source_lineage = W7SourceLineage(source_commit, manifest["manifest_id"], hashlib.sha256(args.source_manifest.read_bytes()).hexdigest(), args.execution_image)
    source_lineage.validate()
    _write_heartbeat(args.heartbeat, campaign_id=args.campaign_id, lambda_value=None, epoch=None, state="WAITING_FOR_LOCK", checkpoint_id=None)
    lock = W7CampaignLock(campaign_id=args.campaign_id, source_commit=source_commit, execution_image=args.execution_image, gpu_uuid=args.gpu_uuid)
    candidates: list[dict[str, Any]] = []
    with lock:
        campaign_root = args.campaign_root
        if campaign_root.is_symlink() or (campaign_root.exists() and not campaign_root.is_dir()):
            raise RuntimeError("W7 campaign root is unsafe")
        campaign_root.mkdir(parents=True, exist_ok=True)
        manifest_value = _campaign_manifest(
            authorization=authorization,
            source_manifest=manifest,
            profile_freeze=profile_freeze,
            campaign_id=args.campaign_id,
            source_commit=source_commit,
            source_manifest_sha256=manifest_sha256,
            profile_freeze_sha256=profile_freeze_sha256,
            execution_image=args.execution_image,
            gpu_uuid=args.gpu_uuid,
            physical_batch=physical_batch,
            accumulation_factor=accumulation_factor,
            validation_batch=validation_batch,
        )
        manifest_path = campaign_root / "campaign_manifest.json"
        if manifest_path.is_file():
            existing_manifest = json.loads(manifest_path.read_bytes())
            if existing_manifest != manifest_value:
                raise RuntimeError("W7 campaign manifest differs; explicit supersession is required")
        elif manifest_path.exists() or manifest_path.is_symlink():
            raise RuntimeError("W7 campaign manifest path is unsafe")
        else:
            _write_new_json(manifest_path, manifest_value)
        for lambda_value in W7_LAMBDA_GRID:
            config = load_w7_config(
                lambda_value=lambda_value,
                physical_batch_size=physical_batch,
                accumulation_factor=accumulation_factor,
                validation_batch_size=validation_batch,
            )
            binding = authenticate_w7_gpu(config_hash=run_config_hash(config), expected_gpu_uuid=args.gpu_uuid)
            verify_frozen_gpu_binding(binding, config_hash=run_config_hash(config))
            candidate_root = args.campaign_root / f"lambda-{lambda_value:g}"
            candidates.append(_candidate(config=config, root=candidate_root, source_lineage=source_lineage, gpu_uuid=args.gpu_uuid, profile_binding=binding, repo_root=args.repo_root, heartbeat=args.heartbeat, campaign_id=args.campaign_id))
        completion = {
            "schema_version": 1,
            "artifact_role": CAMPAIGN_ROLE,
            "status": "COMPLETE_NOT_ADJUDICATED",
            "campaign_id": args.campaign_id,
            "candidate_lambdas": [candidate["lambda"] for candidate in candidates],
            "candidate_paths": [str(args.campaign_root / f"lambda-{candidate['lambda']:g}/candidate_completion.json") for candidate in candidates],
            "candidates": candidates,
            "g4_adjudication_run": 0,
            "lambda_core_updated": False,
            "scientific_execution_authorization": "PRESENT",
        }
        completion["completion_id"] = "w7campaign-" + canonical_sha256(completion)
        _write_new_json(args.campaign_root / "campaign_completion.json", completion)
        _write_heartbeat(args.heartbeat, campaign_id=args.campaign_id, lambda_value=None, epoch=None, state="COMPLETE_NOT_ADJUDICATED", checkpoint_id=None)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--profile-freeze", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--heartbeat", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--execution-image", default=W7_EXECUTION_IMAGE_FAMILY)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
