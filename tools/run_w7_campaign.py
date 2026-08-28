#!/usr/bin/env python3
"""Detached future W7-B campaign launcher (never run during W7-A).

One process owns the global flock and runs the five candidate roots in the
frozen order.  The command intentionally requires a separate execution
authorization artifact, which does not exist in W7-A.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from adjudication.w7_g4 import CANDIDATE_ELIGIBILITY  # noqa: E402
from config.run_config import config_hash as run_config_hash  # noqa: E402
from config.w7_execution import authenticate_w7_gpu, verify_frozen_gpu_binding  # noqa: E402
from evaluation.w7_validation import evaluate_validation, selected_checkpoint_result  # noqa: E402
from gen_w7_source_manifest import verify as verify_source_manifest  # noqa: E402
from training.deterministic_core import canonical_bytes, canonical_sha256  # noqa: E402
from training.w7_g4 import W7_G4_PILOT_POLICY, W7SourceLineage, W7Trainer  # noqa: E402
from training.w7_protocol import W7_EXECUTION_IMAGE_FAMILY, W7_LAMBDA_GRID  # noqa: E402
from runtime.w7_lock import W7CampaignLock  # noqa: E402


AUTHORIZATION_ROLE = "W7_G4_SCIENTIFIC_EXECUTION_AUTHORIZATION"
CAMPAIGN_ROLE = "W7_G4_CAMPAIGN_COMPLETE_NOT_ADJUDICATED"
HEARTBEAT_ROLE = "W7_OPERATIONAL_HEARTBEAT"


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"immutable W7 campaign record already exists: {path}")
    path.write_bytes(canonical_bytes(value))


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
    path.write_bytes(canonical_bytes(value))


def _load_authorization(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("W7 execution authorization is absent; W7-B is not authorized")
    value = json.loads(path.read_bytes())
    required = {"schema_version", "artifact_role", "status", "authorization_id", "campaign_id", "source_manifest_id", "profile_freeze_id", "lambda_grid", "scientific_execution_authorization"}
    if set(value) != required:
        raise RuntimeError("W7 execution authorization schema differs")
    if value["artifact_role"] != AUTHORIZATION_ROLE or value["status"] != "AUTHORIZED":
        raise RuntimeError("W7 execution authorization is not active")
    if value["scientific_execution_authorization"] != "PRESENT":
        raise RuntimeError("W7 scientific execution authorization is not present")
    if value["lambda_grid"] != list(W7_LAMBDA_GRID):
        raise RuntimeError("W7 authorization lambda grid differs")
    return value


def _verify_upstream() -> None:
    for tool in ("tools/verify_w5_training_system.py", "tools/verify_w6_complete.py"):
        subprocess.run([sys.executable, str(REPO / tool)], cwd=REPO, check=True)


def _load_epoch_summaries(root: Path) -> list[dict[str, Any]]:
    validation_root = root / "validation"
    paths = sorted(validation_root.glob("epoch-*.json")) if validation_root.is_dir() else []
    values: list[dict[str, Any]] = []
    for path in paths:
        if path.is_symlink():
            raise RuntimeError("W7 validation summary is a symlink")
        values.append(json.loads(path.read_bytes()))
    return values


def _publish_validation(root: Path, evaluation: Any) -> None:
    summary = dict(evaluation.summary)
    path = root / f"validation/epoch-{summary['epoch']:04d}.json"
    _write_new_json(path, summary)


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
    if root.exists():
        if (root / "latest.json").is_file():
            trainer.resume()
        elif any(root.iterdir()):
            raise RuntimeError(f"incomplete W7 candidate has no authenticated latest pointer: {root}")
    expected_epochs = int(config.parameters["learned_system"]["epochs"][config.resolved["dataset"]])
    summaries = _load_epoch_summaries(root)
    if len(summaries) > trainer.completed_epoch + 1:
        raise RuntimeError("W7 validation summaries run past the authenticated checkpoint")
    if summaries and [item["epoch"] for item in summaries] != list(range(len(summaries))):
        raise RuntimeError("W7 validation summaries are not an exact prefix")
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
    from evaluation.w7_validation import select_checkpoint_epoch

    selection = select_checkpoint_epoch(summaries, expected_epochs=expected_epochs)
    _write_heartbeat(heartbeat, campaign_id=campaign_id, lambda_value=float(config.resolved["lambda"]), epoch=selection["selected_epoch"], state="FINAL_EVALUATION", checkpoint_id=selection["selected_checkpoint_id"])
    selected = selected_checkpoint_result(trainer, selection=selection, repo_root=repo_root)
    candidate = {
        "schema_version": 1,
        "artifact_role": "W7_G4_LAMBDA_CANDIDATE_COMPLETION",
        "candidate_id": "w7candidate-" + canonical_sha256({"lambda": config.resolved["lambda"], "selected": selected["result_digest"]}),
        "status": "COMPLETE",
        "authentication_status": "PASSED",
        "eligibility": dict(CANDIDATE_ELIGIBILITY),
        "lambda": float(config.resolved["lambda"]),
        "lineage": {
            "protocol_version": "w7-g4-pre-execution-v1",
            "source_commit": source_lineage.source_commit,
            "source_manifest_id": source_lineage.source_manifest_id,
            "source_manifest_sha256": source_lineage.source_manifest_sha256,
            "execution_image": source_lineage.execution_image,
            "execution_profile_id": config.resolved["execution_profile_id"],
            "gpu_uuid": gpu_uuid,
            "dataset": config.resolved["dataset"],
            "split_manifest_hash": config.parameters["datasets"][config.resolved["dataset"]]["manifest_sha256"],
            "architecture": config.resolved["architecture"],
            "ratio": config.resolved["bw_ratio"],
            "k": config.resolved["k"],
            "train_seed": config.resolved["train_seed"],
            "channel_seed": config.resolved["channel_seed"],
            "train_snr_db": config.resolved["train_snr_db"],
            "epochs": expected_epochs,
            "optimizer": config.parameters["learned_system"]["optimizer"],
            "scheduler": config.parameters["learned_system"]["lr_schedule"],
            "checkpoint_selection": {"metric": "top1_accuracy", "mode": "max", "tie_break": "earliest_epoch", "snr_db": 7},
            "validation_noise_policy": "keyed_channel_noise_same_per_image_across_lambda",
        },
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
        "test_access": 0,
        "selected_validation_result_digest": selected["result_digest"],
    }
    _write_new_json(root / "candidate_completion.json", candidate)
    return candidate


def run(args: argparse.Namespace) -> int:
    authorization = _load_authorization(args.authorization)
    if authorization["campaign_id"] != args.campaign_id:
        raise RuntimeError("W7 campaign ID differs from execution authorization")
    manifest = json.loads(args.source_manifest.read_bytes())
    verify_source_manifest(manifest, current=True)
    if manifest["manifest_id"] != authorization["source_manifest_id"]:
        raise RuntimeError("W7 source manifest differs from authorization")
    profile_freeze = json.loads(args.profile_freeze.read_bytes())
    if profile_freeze.get("status") != "FROZEN" or profile_freeze.get("gpu_uuid") != args.gpu_uuid:
        raise RuntimeError("W7 Pascal profile freeze differs from authorization/launch")
    if profile_freeze.get("profile_freeze_id") != authorization["profile_freeze_id"]:
        raise RuntimeError("W7 profile freeze ID differs")
    _verify_upstream()
    source_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    source_lineage = W7SourceLineage(source_commit, manifest["manifest_id"], __import__("hashlib").sha256(args.source_manifest.read_bytes()).hexdigest(), args.execution_image)
    source_lineage.validate()
    _write_heartbeat(args.heartbeat, campaign_id=args.campaign_id, lambda_value=None, epoch=None, state="WAITING_FOR_LOCK", checkpoint_id=None)
    lock = W7CampaignLock(campaign_id=args.campaign_id, source_commit=source_commit, execution_image=args.execution_image, gpu_uuid=args.gpu_uuid)
    candidates: list[dict[str, Any]] = []
    with lock:
        for lambda_value in W7_LAMBDA_GRID:
            config = __import__("training.w7_protocol", fromlist=["load_w7_config"]).load_w7_config(lambda_value=lambda_value)
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
