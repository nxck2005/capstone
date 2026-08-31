#!/usr/bin/env python3
"""Freeze a conservative, non-scientific W8 runtime/storage estimate.

The estimate reuses the qualified one-epoch Pascal profile as performance
*evidence* only.  It is not a training result and this command never creates a
W8 model, optimizer step, checkpoint, validation result or campaign root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from training.deterministic_core import canonical_bytes, canonical_sha256  # noqa: E402
from training.w8_protocol import W8_EPOCHS, W8_MIN_FREE_SPACE_GIB, run_cells  # noqa: E402

PROFILE_PATH = REPO / "results/learned/w7/w7_pascal_profile.json"
PROFILE_ID = "w7profile-c2e70848dc6857fe4df3868c90af1ccff4d6e0c7d267cbad8b9ad49b228e5d69"
PROFILE_SHA256 = "938e3ad8420a8e543a5f4576aa7d44ef78cd50c71a8c2540910a552672bd6bf0"
ESTIMATE_ROLE = "W8_NON_SCIENTIFIC_RUNTIME_ESTIMATE"
ESTIMATE_PREFIX = "w8estimate-"
DEFAULT_OUTPUT = REPO / "results/learned/w8/w8_runtime_estimate.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(REPO))


def build_runtime_estimate(*, issued_at_utc: str | None = None) -> dict[str, Any]:
    if not PROFILE_PATH.is_file() or PROFILE_PATH.is_symlink():
        raise ValueError("qualified W7 Pascal profile is missing or unsafe")
    if _sha(PROFILE_PATH) != PROFILE_SHA256:
        raise ValueError("qualified W7 Pascal profile bytes differ")
    profile = json.loads(PROFILE_PATH.read_bytes())
    if not isinstance(profile, dict):
        raise ValueError("qualified W7 Pascal profile is not an object")
    if profile.get("report_id") != PROFILE_ID or profile.get("status") != "PASSED" or profile.get("scientific_status") != "NON_SCIENTIFIC_ZERO_G4_COVERAGE":
        raise ValueError("qualified W7 Pascal profile status differs")
    if profile.get("execution_profile_id") != "confessor_pascal_cu126" or profile.get("gpu_binding", {}).get("gpu_uuid") != "GPU-00214b86-48e7-fcf0-bf46-575fa7f85b6b":
        raise ValueError("qualified W7 Pascal profile GPU binding differs")
    batch = profile.get("batch", {})
    if batch != {
        "accumulation_factor": 1,
        "effective_batch_size": 32,
        "microbatches": 265,
        "physical_batch_size": 32,
        "validation_batch_size": 32,
    }:
        raise ValueError("qualified W7 Pascal profile batch evidence differs")
    training_epoch = profile.get("training_epoch", {})
    elapsed = float(training_epoch["elapsed_complete_epoch_seconds"])
    checkpoint_bytes = int(training_epoch["checkpoint_bytes"])
    run_count = len(run_cells())
    checkpoint_count = run_count * W8_EPOCHS
    raw_checkpoint_bytes = checkpoint_count * checkpoint_bytes
    reserved_bytes = 15 * (2**30)  # literal-ok: conservative 15 GiB W8 reservation
    minimum_free_bytes = W8_MIN_FREE_SPACE_GIB * (2**30)  # literal-ok: owner-required GiB-to-byte preflight conversion
    per_run_hours = {
        "r_1_6": 3.0,  # literal-ok: conservative non-scientific estimate below 4 h
        "r_1_24": 2.75,  # literal-ok: conservative non-scientific estimate below 4 h
    }
    total_hours = sum(per_run_hours[cell.ratio] for cell in run_cells())
    max_hours = float(get("compute.max_wall_clock_hours_per_run"))
    if any(hours >= max_hours for hours in per_run_hours.values()):
        raise ValueError("W8 runtime estimate does not fit the per-run wall-clock cap")
    body: dict[str, Any] = {
        "schema_version": 1,
        "artifact_role": ESTIMATE_ROLE,
        "status": "ESTIMATE_ONLY_NON_SCIENTIFIC",
        "issued_at_utc": issued_at_utc or datetime.now(timezone.utc).isoformat(),
        "profile_basis": {
            "path": _relative(PROFILE_PATH),
            "profile_id": PROFILE_ID,
            "file_sha256": PROFILE_SHA256,
            "execution_profile_id": "confessor_pascal_cu126",
            "gpu_name": "NVIDIA GeForce GTX 1080 Ti",
            "gpu_uuid": "GPU-00214b86-48e7-fcf0-bf46-575fa7f85b6b",
            "observed_complete_epoch_seconds": elapsed,
            "observed_checkpoint_bytes": checkpoint_bytes,
            "observed_optimizer_steps": int(profile["training_epoch"]["applied_optimizer_steps"]),
            "observed_grad_scaler_skips": int(profile["training_epoch"]["grad_scaler_skips"]),
            "profile_scientific_optimizer_steps": 0,
        },
        "kernel_equivalence": {
            "status": "PERFORMANCE_EVIDENCE_REUSED_AFTER_RESULT_AFFECTING_PATH_AUDIT",
            "same_model_builder": "src/models/djscc.py::build_djscc with keyed train_seed/component_path init",
            "same_loss": "src/training/djscc_loss.py::DJSCCObjective",
            "same_channel_noise": "training_channel_noise keyed identities and AM-91 unit complex noise",
            "same_data_order": "TrainingDJSCCDataset plus EpochPermutationSampler keyed by train_seed/epoch",
            "same_optimizer_update": "torch.optim.Adam plus deterministic_core.apply_optimizer_update and AM-91 GradScaler accounting",
            "same_scheduler": "zero-based epoch-start cosine schedule",
            "differences_included_in_estimate": [
                "W8 authenticated role/lineage/checkpoint records",
                "W8 full validation inference after every completed epoch",
                "W8 100-epoch schedule and two ratio-specific model shapes",
            ],
            "real_data_profile_rerun": False,
            "reason": "W7 Pascal profile exercises the same result-affecting train primitives; W8-only custody and validation overhead is covered by the conservative margin.",
        },
        "per_run_runtime_hours_estimate": {
            "r_1_6": per_run_hours["r_1_6"],
            "r_1_24": per_run_hours["r_1_24"],
        },
        "per_run_wall_clock_cap_hours": max_hours,
        "six_run_total_hours_estimate": total_hours,
        "timing_claim": "conservative planning estimate, not an observed W8 runtime and not a result",
        "checkpoint_storage_estimate": {
            "run_count": run_count,
            "epochs_per_run": W8_EPOCHS,
            "completed_epoch_checkpoint_count": checkpoint_count,
            "profile_checkpoint_bytes_used_as_upper_bound": checkpoint_bytes,
            "raw_checkpoint_payload_upper_bound_bytes": raw_checkpoint_bytes,
            "raw_checkpoint_payload_upper_bound_gib": raw_checkpoint_bytes / (2**30),
            "conservative_reserved_storage_bytes": reserved_bytes,
            "conservative_reserved_storage_gib": 15.0,  # literal-ok: same 15 GiB reservation
            "minimum_free_space_preflight_bytes": minimum_free_bytes,
            "minimum_free_space_preflight_gib": float(W8_MIN_FREE_SPACE_GIB),
            "includes": ["checkpoint payloads", "authenticated sidecars", "epoch records", "validation summaries", "selected result rows", "filesystem margin"],
            "storage_claim": "conservative estimate only; no W8 scientific state exists",
        },
        "boundary": {
            "w8_scientific_optimizer_steps": 0,
            "w8_scientific_checkpoints": 0,
            "w8_final_training_runs": 0,
            "g10": "NOT_AUTHORIZED",
            "er2_randomized_training": "NOT_AUTHORIZED",
            "papr_constrained_training": "NOT_AUTHORIZED",
            "er9_training": "NOT_AUTHORIZED",
            "test": "SEALED",
        },
    }
    body["estimate_id"] = ESTIMATE_PREFIX + canonical_sha256(body)
    return body


def verify_runtime_estimate(value: dict[str, Any]) -> dict[str, Any]:
    expected = build_runtime_estimate(issued_at_utc=value.get("issued_at_utc"))
    if value != expected:
        raise ValueError("W8 runtime estimate differs from the frozen profile-derived estimate")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_runtime_estimate(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable W8 runtime estimate already exists: {path}")
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
            raise FileExistsError(f"immutable W8 runtime estimate already exists: {path}") from None
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--issued-at-utc")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else REPO / args.output
    if args.check:
        value = json.loads(output.read_bytes())
        if not isinstance(value, dict):
            raise ValueError("W8 runtime estimate is not an object")
        verify_runtime_estimate(value)
        print(f"W8 runtime estimate PASS: {value['estimate_id']}")
    else:
        value = build_runtime_estimate(issued_at_utc=args.issued_at_utc)
        write_runtime_estimate(value, output)
        print(f"W8 runtime estimate written: {value['estimate_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
