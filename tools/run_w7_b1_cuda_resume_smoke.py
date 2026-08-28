#!/usr/bin/env python3
"""Run the bounded, non-scientific CUDA checkpoint/resume smoke for W7-B1.

The two subcommands are intentionally separate process entry points.  They use
W7Trainer and its production checkpoint path, but a tiny deterministic fixture
and the NON_SCIENTIFIC_PROFILE role.  No validation or candidate machinery is
imported or called.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.data import Dataset

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from config.run_config import config_hash as run_config_hash  # noqa: E402
from config.w7_execution import authenticate_w7_gpu  # noqa: E402
from training.deterministic_core import canonical_bytes, canonical_sha256, state_tree_sha256  # noqa: E402
from training.w7_g4 import (  # noqa: E402
    NON_SCIENTIFIC_PROFILE_POLICY,
    W7SourceLineage,
    W7Trainer,
    checkpoint_state_digest,
    profile_config,
)
from training.w7_protocol import (  # noqa: E402
    W7_EXECUTION_IMAGE_FAMILY,
    W7_PROFILE_ID,
    W7_SELECTED_GPU_NAME,
    W7_SELECTED_GPU_UUID,
    W7_VALIDATION_BATCH_SIZE,
)
from verify_w7_a import verify_profile_freeze  # noqa: E402
from verify_w7_b1 import (  # noqa: E402
    SMOKE_PATH,
    W7B1Hold,
    verify_source_path,
)


SMOKE_DATASET_NAME = "synthetic_w7_b1_fixture"
SMOKE_SAMPLE_COUNT = 160  # literal-ok: five scaler attempts at the frozen 32-sample target, enough for one applied update
SMOKE_PHYSICAL_BATCH = 2  # literal-ok: one tiny physical microbatch
SMOKE_ACCUMULATION = 16  # literal-ok: preserves the configured effective batch
SMOKE_EPOCH_A = 0  # literal-ok: first completed checkpoint epoch
SMOKE_EPOCH_B = 1  # literal-ok: successor checkpoint epoch


class SmokeDataset(Dataset[tuple[torch.Tensor, int, str]]):
    """Small deterministic RGB fixture with production-style stable IDs."""

    def __init__(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return SMOKE_SAMPLE_COUNT

    def source_sample(self, index: int) -> SimpleNamespace:
        return SimpleNamespace(stable_sample_id=f"w7-b1-smoke-{index:04d}")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        sample = self.source_sample(index)
        value = float(((index + self.epoch) % 4 + 1) / 4)  # literal-ok: deterministic fixture values in the model's [0, 1] domain
        image = torch.full((3, 160, 160), value, dtype=torch.float32)  # literal-ok: Imagenette-160-shaped fixture
        return image, index % 2, sample.stable_sample_id  # literal-ok: two fixture labels


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise W7B1Hold(message)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise W7B1Hold(f"B1 smoke JSON is corrupt or missing: {path}") from None
    _require(isinstance(value, dict), f"B1 smoke JSON is not an object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.exists() and not path.is_symlink(), f"B1 smoke evidence already exists: {path}")
    path.write_bytes(canonical_bytes(value))


def _state_digests(trainer: W7Trainer) -> dict[str, str]:
    digest = checkpoint_state_digest(trainer)
    digest["scaler_state_sha256"] = state_tree_sha256(
        trainer.scaler.state_dict() if trainer.scaler is not None else None
    )
    return digest


def _context(args: argparse.Namespace) -> tuple[Any, dict[str, Any], dict[str, Any], str]:
    runtime_root = args.runtime_root.resolve()
    _require(not runtime_root.is_relative_to(REPO.resolve()), "B1 smoke runtime must be outside the scientific checkout")
    source = verify_source_path(args.source_manifest, current=True, repo_root=REPO)
    freeze = _load(args.profile_freeze)
    verify_profile_freeze(freeze)
    _require(freeze["execution_profile_id"] == W7_PROFILE_ID, "B1 smoke profile differs")
    _require(freeze["gpu_uuid"] == W7_SELECTED_GPU_UUID and freeze["gpu_name"] == W7_SELECTED_GPU_NAME, "B1 smoke GPU differs")
    config = profile_config(
        physical_batch_size=SMOKE_PHYSICAL_BATCH,
        accumulation_factor=SMOKE_ACCUMULATION,
    )
    config_digest = run_config_hash(config)
    binding = authenticate_w7_gpu(
        config_hash=config_digest,
        expected_gpu_uuid=W7_SELECTED_GPU_UUID,
        device="cuda:0",
        require_openjpeg=False,
    )
    _require(binding["gpu_uuid"] == W7_SELECTED_GPU_UUID, "B1 smoke authenticated GPU differs")
    _require(binding["git_commit"] == source["source_commit"], "B1 smoke checkout/source commit differs")
    _require(binding["execution_profile_id"] == W7_PROFILE_ID, "B1 smoke authenticated profile differs")
    return config, source, freeze, config_digest


def _lineage(source: dict[str, Any], source_sha: str) -> W7SourceLineage:
    return W7SourceLineage(
        source_commit=source["source_commit"],
        source_manifest_id=source["manifest_id"],
        source_manifest_sha256=source_sha,
        execution_image=W7_EXECUTION_IMAGE_FAMILY,
    )


def _trainer(
    *,
    config: Any,
    source: dict[str, Any],
    source_sha: str,
    freeze: dict[str, Any],
    runtime_root: Path,
) -> tuple[W7Trainer, dict[str, Any]]:
    binding = authenticate_w7_gpu(
        config_hash=run_config_hash(config),
        expected_gpu_uuid=W7_SELECTED_GPU_UUID,
        device="cuda:0",
        require_openjpeg=False,
    )
    trainer = W7Trainer(
        config,
        device="cuda:0",
        runtime_root=runtime_root,
        source_lineage=_lineage(source, source_sha),
        profile_binding=binding,
        policy=NON_SCIENTIFIC_PROFILE_POLICY,
        num_workers=0,
    )
    _require(trainer.amp_enabled is True and trainer.scaler is not None, "B1 smoke AMP/GradScaler is disabled")
    _require(freeze["gpu_uuid"] == binding["gpu_uuid"], "B1 smoke freeze/binding GPU differs")
    return trainer, binding


def _phase_a(args: argparse.Namespace) -> int:
    config, source, freeze, config_digest = _context(args)
    source_sha = _sha(args.source_manifest)
    _require(not args.runtime_root.exists(), "B1 smoke runtime already exists; no resume reuse is permitted")
    args.runtime_root.mkdir(parents=True, exist_ok=False)
    trainer, binding = _trainer(
        config=config,
        source=source,
        source_sha=source_sha,
        freeze=freeze,
        runtime_root=args.runtime_root,
    )
    record = trainer.train_epoch(SMOKE_EPOCH_A, SmokeDataset(SMOKE_EPOCH_A))
    _require(record["artifact_role"] == "NON_SCIENTIFIC_PROFILE_EPOCH_RECORD", "B1 smoke epoch role is not non-scientific")
    _require(record["optimizer_steps"] > 0, "B1 smoke process A applied no optimizer update")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    sidecar = trainer.save_checkpoint(record)
    digests = _state_digests(trainer)
    evidence = {
        "schema_version": 1,  # literal-ok: process evidence schema
        "artifact_role": "NON_SCIENTIFIC_PROFILE_CHECKPOINT",
        "process": "A",
        "process_id": os.getpid(),
        "process_boundary": "PROCESS_A_EXITED_AFTER_AUTHENTICATED_CHECKPOINT",
        "source_commit": source["source_commit"],
        "source_manifest_id": source["manifest_id"],
        "source_manifest_sha256": source_sha,
        "execution_image_family": W7_EXECUTION_IMAGE_FAMILY,
        "execution_profile_id": W7_PROFILE_ID,
        "gpu_uuid": binding["gpu_uuid"],
        "gpu_name": binding["gpu_name"],
        "device": binding["device"],
        "amp_enabled": trainer.amp_enabled,
        "grad_scaler_present": trainer.scaler is not None,
        "checkpoint_id": sidecar["checkpoint_id"],
        "checkpoint_path": sidecar["checkpoint_path"],
        "predecessor_checkpoint_id": sidecar["predecessor_checkpoint_id"],
        "completed_epoch": sidecar["completed_epoch"],
        "global_optimizer_step": sidecar["global_optimizer_step"],
        "optimizer_steps": record["optimizer_steps"],
        "model_state_sha256": digests["model_state_sha256"],
        "optimizer_state_sha256": digests["optimizer_state_sha256"],
        "scheduler_state_sha256": digests["scheduler_state_sha256"],
        "scaler_state_sha256": digests["scaler_state_sha256"],
        "validation_performed": False,
        "scientific_candidate": False,
        "smoke_config_hash": config_digest,
    }
    evidence["evidence_id"] = "w7b1processa-" + canonical_sha256(evidence)
    _write_once(args.evidence_root / "process_a.json", evidence)
    print(f"W7-B1 smoke process A PASS: {evidence['evidence_id']} checkpoint={sidecar['checkpoint_id']}")
    return 0


def _phase_b(args: argparse.Namespace) -> int:
    config, source, freeze, config_digest = _context(args)
    source_sha = _sha(args.source_manifest)
    phase_a_path = args.evidence_root / "process_a.json"
    phase_a = _load(phase_a_path)
    _require(phase_a["source_commit"] == source["source_commit"] and phase_a["source_manifest_id"] == source["manifest_id"] and phase_a["source_manifest_sha256"] == source_sha, "B1 smoke process A source lineage differs")
    _require(args.runtime_root.is_dir() and not args.runtime_root.is_symlink(), "B1 smoke runtime is missing or unsafe")
    trainer, binding = _trainer(
        config=config,
        source=source,
        source_sha=source_sha,
        freeze=freeze,
        runtime_root=args.runtime_root,
    )
    pointer = args.runtime_root / "latest.json"
    _require(pointer.is_file() and not pointer.is_symlink(), "B1 smoke latest pointer is missing")
    latest = _load(pointer)
    _require(latest["checkpoint_id"] == phase_a["checkpoint_id"], "B1 smoke latest pointer is not process A checkpoint")
    _require(not list((args.runtime_root / "checkpoints").glob("epoch-0001.*")), "B1 smoke successor already exists")
    restored = trainer.resume()
    restored_digests = _state_digests(trainer)
    seam = {
        "model_state_equal": restored_digests["model_state_sha256"] == phase_a["model_state_sha256"],
        "optimizer_state_equal": restored_digests["optimizer_state_sha256"] == phase_a["optimizer_state_sha256"],
        "scheduler_state_equal": restored_digests["scheduler_state_sha256"] == phase_a["scheduler_state_sha256"],
        "scaler_state_equal": restored_digests["scaler_state_sha256"] == phase_a["scaler_state_sha256"],
        "completed_epoch_equal": trainer.completed_epoch == phase_a["completed_epoch"],
        "global_optimizer_step_equal": trainer.global_optimizer_step == phase_a["global_optimizer_step"],
    }
    _require(all(seam.values()), "B1 smoke fresh-process resume seam differs")
    _require(restored["checkpoint_id"] == phase_a["checkpoint_id"], "B1 smoke restored checkpoint differs")
    record = trainer.train_epoch(SMOKE_EPOCH_B, SmokeDataset(SMOKE_EPOCH_B))
    _require(record["artifact_role"] == "NON_SCIENTIFIC_PROFILE_EPOCH_RECORD", "B1 smoke successor epoch role is not non-scientific")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    successor = trainer.save_checkpoint(record)
    _require(successor["predecessor_checkpoint_id"] == phase_a["checkpoint_id"], "B1 smoke successor predecessor differs")
    checker, _ = _trainer(
        config=config,
        source=source,
        source_sha=source_sha,
        freeze=freeze,
        runtime_root=args.runtime_root,
    )
    checked_latest = checker.resume()
    _require(checked_latest["checkpoint_id"] == successor["checkpoint_id"], "B1 smoke latest-only chain did not select successor")
    _require(checker.completed_epoch == SMOKE_EPOCH_B, "B1 smoke full chain epoch differs")
    final_digests = _state_digests(trainer)
    process_b = {
        "schema_version": 1,  # literal-ok: process evidence schema
        "artifact_role": "NON_SCIENTIFIC_PROFILE_CHECKPOINT",
        "process": "B",
        "process_id": os.getpid(),
        "process_boundary": "PROCESS_B_FRESH_PROCESS_RESTORED_AND_PUBLISHED_SUCCESSOR",
        "fresh_process": True,
        "source_commit": source["source_commit"],
        "source_manifest_id": source["manifest_id"],
        "source_manifest_sha256": source_sha,
        "execution_image_family": W7_EXECUTION_IMAGE_FAMILY,
        "execution_profile_id": W7_PROFILE_ID,
        "gpu_uuid": binding["gpu_uuid"],
        "gpu_name": binding["gpu_name"],
        "device": binding["device"],
        "amp_enabled": trainer.amp_enabled,
        "grad_scaler_present": trainer.scaler is not None,
        "restored_checkpoint_id": restored["checkpoint_id"],
        "checkpoint_id": successor["checkpoint_id"],
        "checkpoint_path": successor["checkpoint_path"],
        "predecessor_checkpoint_id": successor["predecessor_checkpoint_id"],
        "completed_epoch": successor["completed_epoch"],
        "global_optimizer_step": successor["global_optimizer_step"],
        "optimizer_steps": record["optimizer_steps"],
        "model_state_sha256": final_digests["model_state_sha256"],
        "optimizer_state_sha256": final_digests["optimizer_state_sha256"],
        "scheduler_state_sha256": final_digests["scheduler_state_sha256"],
        "scaler_state_sha256": final_digests["scaler_state_sha256"],
        "validation_performed": False,
        "scientific_candidate": False,
        "smoke_config_hash": config_digest,
    }
    process_b["evidence_id"] = "w7b1processb-" + canonical_sha256(process_b)
    resume_seam = {
        **seam,
        "scaler_state_before_restore_sha256": phase_a["scaler_state_sha256"],
        "scaler_state_after_restore_sha256": restored_digests["scaler_state_sha256"],
        "restored_checkpoint_id": restored["checkpoint_id"],
    }
    artifact = {
        "schema_version": 1,  # literal-ok: smoke evidence schema
        "artifact_role": "NON_SCIENTIFIC_W7_B1_CUDA_RESUME_SMOKE",
        "status": "PASSED",
        "scientific_status": "NON_SCIENTIFIC_ZERO_G4_COVERAGE",
        "role": "NON_SCIENTIFIC_PROFILE",
        "eligibility": {
            "artifact_role": "NON_SCIENTIFIC_PROFILE",
            "selection_eligibility": "NOT_ELIGIBLE_FOR_SELECTION",
            "reporting_eligibility": "NOT_ELIGIBLE_FOR_REPORTING",
            "w7_g4_eligibility": "NOT_ELIGIBLE_FOR_W7_G4",
            "w8_eligibility": "NOT_ELIGIBLE_FOR_W8",
            "test_eligibility": "NOT_ELIGIBLE_FOR_TEST",
        },
        "source_commit": source["source_commit"],
        "source_manifest_id": source["manifest_id"],
        "source_manifest_sha256": source_sha,
        "profile_freeze_id": freeze["profile_freeze_id"],
        "profile_freeze_sha256": _sha(args.profile_freeze),
        "execution_image_family": W7_EXECUTION_IMAGE_FAMILY,
        "execution_profile_id": W7_PROFILE_ID,
        "gpu_uuid": binding["gpu_uuid"],
        "gpu_name": binding["gpu_name"],
        "device": binding["device"],
        "amp_enabled": trainer.amp_enabled,
        "grad_scaler_present": trainer.scaler is not None,
        "smoke_config_hash": config_digest,
        "smoke_batch": {
            "dataset": SMOKE_DATASET_NAME,
            "sample_count": SMOKE_SAMPLE_COUNT,
            "physical_batch_size": SMOKE_PHYSICAL_BATCH,
            "accumulation_factor": SMOKE_ACCUMULATION,
            "validation_batch_size": W7_VALIDATION_BATCH_SIZE,
        },
        "process_a": phase_a,
        "process_b": process_b,
        "resume_seam": resume_seam,
        "checkpoint_chain": {
            "latest_only": True,
            "older_fallback": False,
            "full_chain_authenticated": True,
            "successor_predecessor_checkpoint_id": successor["predecessor_checkpoint_id"],
            "successor_checkpoint_id": successor["checkpoint_id"],
        },
        "validation": {"performed": False, "model_facing": False},
        "scientific_boundary": {
            "scientific_execution_authorization": "NOT_USED",
            "w7_scientific_optimizer_steps": 0,
            "w7_candidate_results": 0,
            "g4_adjudications": 0,
            "w8_final_training_runs": 0,
            "learned_test_inference": 0,
            "test_model_facing_access": 0,
            "test_access": "SEALED",
        },
        "non_scientific_w7_b1_resume_smoke_optimizer_steps": phase_a["optimizer_steps"] + process_b["optimizer_steps"],
        "protected_counters": {
            "w7_scientific_optimizer_steps": 0,
            "w7_lambda_pilot_runs": 0,
            "w7_candidate_results": 0,
            "g4_adjudications": 0,
            "w8_final_training_runs": 0,
            "learned_test_inference": 0,
            "test_model_facing_access": 0,
            "g8_scientific_changes": 0,
            "f1_reruns": 0,
            "f2_optimizer_steps_during_w7": 0,
            "f3_reruns": 0,
            "pass_one_reruns": 0,
            "pass_two_reruns": 0,
            "pass_three": 0,
            "bler_regeneration": 0,
        },
    }
    artifact["smoke_id"] = "w7b1smoke-" + canonical_sha256(artifact)
    _write_once(args.output, artifact)
    print(f"W7-B1 smoke process B PASS: {artifact['smoke_id']} successor={successor['checkpoint_id']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    for name in ("phase-a", "phase-b"):
        command = sub.add_parser(name)
        command.add_argument("--source-manifest", type=Path, required=True)
        command.add_argument("--profile-freeze", type=Path, required=True)
        command.add_argument("--runtime-root", type=Path, required=True)
        command.add_argument("--evidence-root", type=Path, required=True)
        if name == "phase-b":
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "phase-a":
        return _phase_a(args)
    return _phase_b(args)


if __name__ == "__main__":
    raise SystemExit(main())
