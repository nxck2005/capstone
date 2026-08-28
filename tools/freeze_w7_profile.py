#!/usr/bin/env python3
"""Select and freeze the first qualifying Pascal real-data W7 profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from config.params import get  # noqa: E402
from training.deterministic_core import canonical_bytes, canonical_sha256  # noqa: E402
from training.w7_protocol import W7_PROFILE_ID  # noqa: E402
from verify_w7_profile import verify  # noqa: E402


FREEZE_ROLE = "W7_PASCAL_PROFILE_SELECTION_FREEZE"
GPU_ORDER = (
    "GPU-00214b86-48e7-fcf0-bf46-575fa7f85b6b",
    "GPU-46acd0f2-2ff5-1a43-cac9-2ae20e56dc9a",
)
GPU_NAMES = {
    GPU_ORDER[0]: "NVIDIA GeForce GTX 1080 Ti",
    GPU_ORDER[1]: "NVIDIA TITAN Xp",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"profile freeze already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(value)
    body["profile_freeze_id"] = "w7profilefreeze-" + canonical_sha256(body)
    path.write_bytes(canonical_bytes(body))


def run(args: argparse.Namespace) -> int:
    if len(args.reports) == 0 or len(args.reports) > len(GPU_ORDER):
        raise ValueError("provide one report per attempted Pascal GPU, in ladder order")
    reports: list[dict[str, Any]] = []
    passed_seen = False
    for index, path in enumerate(args.reports):
        value = json.loads(path.read_bytes())
        if passed_seen:
            raise ValueError("a later Pascal profile attempt is forbidden after the first qualifying GPU")
        gpu_uuid = value.get("gpu_uuid")
        if gpu_uuid is None and isinstance(value.get("gpu_binding"), dict):
            gpu_uuid = value["gpu_binding"].get("gpu_uuid")
        if gpu_uuid != GPU_ORDER[index]:
            raise ValueError("profile attempts are not in the GTX-first/Titan-fallback order")
        if value.get("status") == "PASSED":
            verify(value)
            passed_seen = True
        elif value.get("status") != "FAILED_HARD_PROFILE":
            raise ValueError("profile attempt status is neither pass nor hard failure")
        reports.append({
            "path": str(path),
            "report_id": value.get("report_id"),
            "sha256": _sha(path),
            "gpu_uuid": gpu_uuid,
            "gpu_name": GPU_NAMES.get(gpu_uuid),
            "status": value.get("status"),
            "physical_batch_size": value.get("physical_batch_size")
            if value.get("physical_batch_size") is not None
            else value.get("batch", {}).get("physical_batch_size"),
            "accumulation_factor": value.get("accumulation_factor")
            if value.get("accumulation_factor") is not None
            else value.get("batch", {}).get("accumulation_factor"),
        })
    selected_index = next((index for index, report in enumerate(reports) if report["status"] == "PASSED"), None)
    if selected_index is None:
        raise RuntimeError("W7-A HOLD: neither Pascal GPU has a qualifying real-data profile")
    if selected_index > 0 and any(report["status"] == "PASSED" for report in reports[:selected_index]):
        raise RuntimeError("fallback was used after an earlier qualifying GPU")
    selected = json.loads(args.reports[selected_index].read_bytes())
    epoch = selected["training_epoch"]
    epochs = int(get("learned_system.epochs.imagenette160"))
    epoch_seconds = float(epoch["elapsed_complete_epoch_seconds"])
    checkpoint_seconds = float(epoch["checkpoint_write_seconds"])
    # The profile does not run validation.  A conservative pre-result budget
    # reserves half an epoch for each validation and doubles observed checkpoint
    # I/O, then reserves one extra validation/finalization epoch.
    projected = {
        "training_seconds": epoch_seconds * epochs,
        "validation_reserve_seconds": epoch_seconds * epochs * 0.5,  # literal-ok: conservative pre-result reserve
        "checkpoint_io_reserve_seconds": checkpoint_seconds * epochs * 2.0,  # literal-ok: conservative pre-result reserve
        "finalization_reserve_seconds": epoch_seconds,
    }
    projected["one_lambda_seconds"] = sum(projected.values())
    cap = float(get("compute.max_wall_clock_hours_per_run")) * 3600.0
    if not math.isfinite(projected["one_lambda_seconds"]) or projected["one_lambda_seconds"] >= cap:
        raise RuntimeError(f"W7-A HOLD: projected one-lambda runtime {projected['one_lambda_seconds']} exceeds {cap}")
    value = {
        "schema_version": 1,
        "artifact_role": FREEZE_ROLE,
        "status": "FROZEN",
        "execution_profile_id": W7_PROFILE_ID,
        "selection_rule": "first qualifying report in GTX-1080-Ti then TITAN-Xp ladder",
        "attempts": reports,
        "selected_attempt_index": selected_index,
        "selected_report": {
            "path": str(args.reports[selected_index]),
            "report_id": selected["report_id"],
            "file_sha256": _sha(args.reports[selected_index]),
        },
        "gpu_uuid": selected["gpu_binding"]["gpu_uuid"],
        "gpu_name": selected["gpu_binding"]["gpu_name"],
        "gpu_compute_capability": selected["gpu_binding"]["gpu_compute_capability"],
        "physical_batch_size": selected["batch"]["physical_batch_size"],
        "accumulation_factor": selected["batch"]["accumulation_factor"],
        "effective_batch_size": selected["batch"]["effective_batch_size"],
        "validation_batch_size": selected["batch"]["validation_batch_size"],
        "profile_config_hash": selected["config"]["config_hash"],
        "source_lineage": selected["source_lineage"],
        "projected_runtime": {
            **projected,
            "cap_seconds": cap,
            "cap_hours": float(get("compute.max_wall_clock_hours_per_run")),
            "five_lambda_seconds": projected["one_lambda_seconds"] * 5.0,  # literal-ok: five frozen grid entries
        },
        "scientific_status": "NON_SCIENTIFIC_ZERO_G4_COVERAGE",
        "promotion": {
            "eligible_for_g4": False,
            "eligible_for_w8": False,
            "eligible_for_test": False,
            "profile_checkpoint_may_not_initialize_w7": True,
        },
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
    _write(args.output, value)
    print(f"W7 Pascal profile freeze written: {args.output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
