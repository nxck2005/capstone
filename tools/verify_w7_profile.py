#!/usr/bin/env python3
"""Offline verifier for one W7 non-scientific real-data profile report."""

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

from config.w7_execution import verify_frozen_gpu_binding  # noqa: E402
from config.params import get  # noqa: E402
from training.deterministic_core import canonical_sha256  # noqa: E402
from training.w7_protocol import W7_DATASET, W7_PROFILE_ID, W7_RATIO  # noqa: E402


PROFILE_ROLE = "W7_NON_SCIENTIFIC_REAL_DATA_PROFILE"
PROFILE_SCHEMA_VERSION = 1
GPU_UUIDS = {
    "GPU-00214b86-48e7-fcf0-bf46-575fa7f85b6b": "NVIDIA GeForce GTX 1080 Ti",
    "GPU-46acd0f2-2ff5-1a43-cac9-2ae20e56dc9a": "NVIDIA TITAN Xp",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: object, label: str) -> float:
    result = float(value)
    _require(math.isfinite(result), f"{label} is non-finite")
    return result


def verify(value: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version", "artifact_role", "status", "scientific_status", "eligibility",
        "execution_profile_id", "execution_image_family", "source_lineage", "config",
        "gpu_binding", "batch", "training_epoch", "memory", "validation", "test_access",
        "g4_coverage", "w7_scientific_optimizer_steps", "profile_optimizer_steps",
        "runtime_root", "protected_counters", "report_id",
    }
    _require(set(value) == required, "W7 profile report schema differs")
    _require(value["schema_version"] == PROFILE_SCHEMA_VERSION and value["artifact_role"] == PROFILE_ROLE, "W7 profile role/version differs")
    body = dict(value)
    report_id = body.pop("report_id")
    _require(report_id == "w7profile-" + canonical_sha256(body), "W7 profile report ID differs")
    _require(value["status"] == "PASSED", "W7 selected profile report is not a pass")
    _require(value["scientific_status"] == "NON_SCIENTIFIC_ZERO_G4_COVERAGE", "W7 profile scientific status differs")
    _require(value["execution_profile_id"] == W7_PROFILE_ID, "W7 profile ID differs")
    _require(value["test_access"] == 0 and value["g4_coverage"] == 0 and value["w7_scientific_optimizer_steps"] == 0, "W7 profile has scientific coverage")
    _require(value["eligibility"] == {
        "artifact_role": "NON_SCIENTIFIC_PROFILE",
        "selection_eligibility": "NOT_ELIGIBLE_FOR_SELECTION",
        "reporting_eligibility": "NOT_ELIGIBLE_FOR_REPORTING",
        "w7_g4_eligibility": "NOT_ELIGIBLE_FOR_W7_G4",
        "w8_eligibility": "NOT_ELIGIBLE_FOR_W8",
        "test_eligibility": "NOT_ELIGIBLE_FOR_TEST",
    }, "W7 profile eligibility differs")
    uuid = value["gpu_binding"].get("gpu_uuid")
    _require(uuid in GPU_UUIDS and value["gpu_binding"].get("gpu_name") == GPU_UUIDS[uuid], "W7 profile GPU is not a registered Pascal candidate")
    config = value["config"]
    _require(config["dataset"] == W7_DATASET and config["ratio"] == W7_RATIO and config["split"] == "train", "W7 profile dataset/ratio/split differs")
    _require(config["lambda"] == 1.0 and config["train_seed"] == 0 and config["channel_seed"] == 0 and config["training_snr_db"] == 7, "W7 profile choices differ")  # literal-ok: owner-frozen W7 profile choices
    _require(config["epochs_profiled"] == 1, "W7 profile did not profile one epoch")
    binding = dict(value["gpu_binding"])
    binding_hash = binding.pop("binding_sha256", None)
    binding["binding_sha256"] = binding_hash
    verify_frozen_gpu_binding({**binding, "binding_sha256": binding_hash}, config_hash=str(config["config_hash"]))
    batch = value["batch"]
    _require(batch["effective_batch_size"] == 32 and batch["physical_batch_size"] * batch["accumulation_factor"] == 32, "W7 profile effective batch differs")  # literal-ok: owner-frozen W7 effective batch
    _require(batch["validation_batch_size"] == 32, "W7 profile validation batch differs")  # literal-ok: owner-frozen W7 validation batch
    epoch = value["training_epoch"]
    expected_train = int(get(f"datasets.{W7_DATASET}.train_images"))
    _require(epoch["expected_stable_ids"] == expected_train == epoch["processed_stable_ids"] == epoch["samples"], "W7 profile train denominator differs")
    _require(epoch["microbatches"] > 0 and epoch["applied_optimizer_steps"] > 0 and epoch["grad_scaler_skips"] >= 0, "W7 profile optimizer accounting differs")
    _require(epoch["finite_loss"] is True, "W7 profile loss is not finite")
    for field in ("stable_id_order_sha256", "stable_id_set_sha256", "checkpoint_id"):
        _require(isinstance(epoch[field], str) and len(epoch[field]) == 64, f"W7 profile {field} is invalid")
    _require(epoch["checkpoint_bytes"] > 0, "W7 profile checkpoint is empty")
    elapsed = _finite(epoch["elapsed_complete_epoch_seconds"], "profile epoch time")
    _require(elapsed > 0, "profile epoch time is not positive")
    _require(_finite(value["memory"]["peak_allocated_bytes"], "peak allocated memory") > 0, "profile allocated memory is missing")
    _require(_finite(value["memory"]["peak_reserved_bytes"], "peak reserved memory") > 0, "profile reserved memory is missing")
    _require(value["validation"] == {"performed": False, "reason": "profile is train-only"}, "profile performed validation")
    _require(value["protected_counters"] and all(counter == 0 for counter in value["protected_counters"].values()), "W7 profile protected counter is nonzero")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    value = verify(json.loads(args.report.read_bytes()))
    print(f"W7 real-data profile PASS: {value['report_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
