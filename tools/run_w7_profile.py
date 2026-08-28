#!/usr/bin/env python3
"""Run one complete non-scientific real-data Pascal W7 workload profile.

This command intentionally has no validation path and emits an ineligible
profile checkpoint.  The GTX 1080 Ti is attempted by the caller first; the
caller may invoke this command on the Titan only after a hard GTX failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

from config.w7_execution import W7ExecutionHold, authenticate_w7_gpu  # noqa: E402
from config.run_config import config_hash as run_config_hash  # noqa: E402
from training.w7_g4 import (  # noqa: E402
    NON_SCIENTIFIC_PROFILE_POLICY,
    W7SourceLineage,
    W7Trainer,
    profile_config,
)
from training.deterministic_core import canonical_bytes, canonical_sha256  # noqa: E402
from training.w7_protocol import W7_EXECUTION_IMAGE_FAMILY, W7_PROFILE_ID  # noqa: E402
from gen_w7_source_manifest import verify as verify_source_manifest  # noqa: E402


PROFILE_SCHEMA_VERSION = 1
PROFILE_ROLE = "NON_SCIENTIFIC_PROFILE"
PROFILE_REPORT_ROLE = "W7_NON_SCIENTIFIC_REAL_DATA_PROFILE"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def _write_report(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"profile report already exists; refusing overwrite: {path}")
    body = dict(value)
    body["report_id"] = "w7profile-" + canonical_sha256(body)
    path.write_bytes(canonical_bytes(body))
    return body


def _base_failure(args: argparse.Namespace, error: str) -> dict[str, Any]:
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "artifact_role": PROFILE_REPORT_ROLE,
        "status": "FAILED_HARD_PROFILE",
        "scientific_status": "NON_SCIENTIFIC_ZERO_G4_COVERAGE",
        "execution_profile_id": W7_PROFILE_ID,
        "gpu_uuid": args.gpu_uuid,
        "physical_batch_size": args.physical_batch_size,
        "accumulation_factor": args.accumulation_factor,
        "effective_batch_size": args.physical_batch_size * args.accumulation_factor,
        "error": error,
        "protected_counters": {
            "w7_scientific_optimizer_steps": 0,
            "w7_lambda_pilot_runs": 0,
            "w7_candidate_results": 0,
            "g4_adjudications": 0,
            "w8_final_training_runs": 0,
            "learned_test_inference": 0,
            "test_model_facing_access": 0,
        },
    }


def run(args: argparse.Namespace) -> int:
    if args.physical_batch_size * args.accumulation_factor != 32:  # literal-ok: W7 effective batch contract
        raise ValueError("profile effective batch must remain 32")
    if args.accumulation_factor not in {1, 2, 4, 8, 16}:
        raise ValueError("profile accumulation is outside the predetermined ladder")
    if args.gpu_uuid not in {
        "GPU-00214b86-48e7-fcf0-bf46-575fa7f85b6b",
        "GPU-46acd0f2-2ff5-1a43-cac9-2ae20e56dc9a",
    }:
        raise ValueError("profile GPU UUID is not a registered Pascal candidate")
    if args.execution_image != W7_EXECUTION_IMAGE_FAMILY:
        raise ValueError("profile execution image family differs")
    config = profile_config(
        physical_batch_size=args.physical_batch_size,
        accumulation_factor=args.accumulation_factor,
    )
    config_digest = run_config_hash(config)
    output = Path(args.output)
    runtime = Path(args.runtime_root)
    try:
        manifest = json.loads(Path(args.source_manifest).read_bytes())
        verify_source_manifest(manifest, current=True)
        current_commit = _git("rev-parse", "HEAD")
        if bool(_git("status", "--porcelain", "--untracked-files=all")):
            raise RuntimeError("profile checkout is dirty")
        if runtime.exists():
            raise RuntimeError(f"profile runtime already exists; refusing rerun: {runtime}")
        binding = authenticate_w7_gpu(
            config_hash=config_digest,
            expected_gpu_uuid=args.gpu_uuid,
            device="cuda:0",
            require_openjpeg=False,
        )
        lineage = W7SourceLineage(
            source_commit=current_commit,
            source_manifest_id=str(manifest["manifest_id"]),
            source_manifest_sha256=hashlib.sha256(Path(args.source_manifest).read_bytes()).hexdigest(),
            execution_image=args.execution_image,
        )
        lineage.validate()
        import torch

        torch.cuda.reset_peak_memory_stats(0)
        trainer = W7Trainer(
            config,
            device="cuda:0",
            runtime_root=runtime,
            source_lineage=lineage,
            profile_binding=binding,
            policy=NON_SCIENTIFIC_PROFILE_POLICY,
        )
        started = time.monotonic()
        records = trainer.run_epochs(final_epoch=0)
        elapsed = time.monotonic() - started
        sidecar = json.loads((runtime / "latest.json").read_bytes())
        record = records[-1]
        expected_train = int(__import__("config.params", fromlist=["get"]).get("datasets.imagenette160.train_images"))
        if record["samples"] != expected_train or record["stable_id_count"] != expected_train:
            raise RuntimeError("profile did not process the complete Imagenette train denominator")
        report = {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "artifact_role": PROFILE_REPORT_ROLE,
            "status": "PASSED",
            "scientific_status": "NON_SCIENTIFIC_ZERO_G4_COVERAGE",
            "eligibility": {
                "artifact_role": PROFILE_ROLE,
                "selection_eligibility": "NOT_ELIGIBLE_FOR_SELECTION",
                "reporting_eligibility": "NOT_ELIGIBLE_FOR_REPORTING",
                "w7_g4_eligibility": "NOT_ELIGIBLE_FOR_W7_G4",
                "w8_eligibility": "NOT_ELIGIBLE_FOR_W8",
                "test_eligibility": "NOT_ELIGIBLE_FOR_TEST",
            },
            "execution_profile_id": W7_PROFILE_ID,
            "execution_image_family": args.execution_image,
            "source_lineage": {
                "checkout_commit": current_commit,
                "execution_source_commit": manifest["source_commit"],
                "source_manifest_id": manifest["manifest_id"],
                "source_manifest_sha256": lineage.source_manifest_sha256,
            },
            "config": {
                "config_hash": config_digest,
                "ratio": config.resolved["bw_ratio"],
                "lambda": config.resolved["lambda"],
                "train_seed": config.resolved["train_seed"],
                "channel_seed": config.resolved["channel_seed"],
                "training_snr_db": config.resolved["train_snr_db"],
                "dataset": config.resolved["dataset"],
                "split": config.resolved["split"],
                "epochs_profiled": 1,
            },
            "gpu_binding": binding,
            "batch": {
                "physical_batch_size": args.physical_batch_size,
                "accumulation_factor": args.accumulation_factor,
                "effective_batch_size": args.physical_batch_size * args.accumulation_factor,
                "validation_batch_size": config.resolved["validation_batch_size"],
                "microbatches": record["microbatches"],
            },
            "training_epoch": {
                "expected_stable_ids": expected_train,
                "processed_stable_ids": record["stable_id_count"],
                "stable_id_order_sha256": record["stable_id_order_sha256"],
                "stable_id_set_sha256": record["stable_id_set_sha256"],
                "samples": record["samples"],
                "microbatches": record["microbatches"],
                "applied_optimizer_steps": record["optimizer_steps"],
                "grad_scaler_skips": record["grad_scaler_skips"],
                "global_optimizer_step": record["global_optimizer_step"],
                "finite_loss": record["finite_loss"],
                "gradient_checks": record["gradient_checks"],
                "elapsed_complete_epoch_seconds": elapsed,
                "checkpoint_path": sidecar["checkpoint_path"],
                "checkpoint_id": sidecar["checkpoint_id"],
                "checkpoint_bytes": sidecar["checkpoint_bytes"],
                "checkpoint_write_seconds": sidecar["checkpoint_write_seconds"],
            },
            "memory": {
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
            },
            "validation": {"performed": False, "reason": "profile is train-only"},
            "test_access": 0,
            "g4_coverage": 0,
            "w7_scientific_optimizer_steps": 0,
            "profile_optimizer_steps": record["optimizer_steps"],
            "runtime_root": str(runtime),
            "protected_counters": {
                "w7_scientific_optimizer_steps": 0,
                "w7_lambda_pilot_runs": 0,
                "w7_candidate_results": 0,
                "g4_adjudications": 0,
                "w8_final_training_runs": 0,
                "learned_test_inference": 0,
                "test_model_facing_access": 0,
            },
        }
        written = _write_report(output, report)
        print(f"W7 non-scientific real-data profile PASS: {output} {written['report_id']}")
        return 0
    except BaseException as exc:
        _write_report(output, _base_failure(args, f"{type(exc).__name__}: {exc}"))
        print(f"W7 non-scientific profile HARD FAIL: {exc}", file=sys.stderr)
        return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--physical-batch-size", type=int, required=True)
    parser.add_argument("--accumulation-factor", type=int, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-image", default=W7_EXECUTION_IMAGE_FAMILY)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
