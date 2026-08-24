#!/usr/bin/env python3
"""Read-only/synthetic F2 wall-clock preflight; performs no optimizer step."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from config.execution_profiles import authenticate_execution_profile
from data.classifier import EpochPermutationSampler
from models.frozen_reference_classifier import load_frozen_reference_classifier
from training.g8_f_f2 import (
    EXPECTED_EPOCHS,
    EXPECTED_MATERIALIZED,
    EXPECTED_OPTIMIZER_STEPS,
    EXPECTED_STEPS_PER_EPOCH,
    F2ArtifactDataset,
    canonical_json,
    f2_recipe,
    f2_recipe_sha256,
)
from training.g8_f_f2_authorization import EXPECTED_DEVICE, PROFILE_ID


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--f1-runtime", type=Path, default=ROOT / "results/baseline/g8_f/runtime")
    parser.add_argument("--device", default=EXPECTED_DEVICE)
    parser.add_argument("--batches", type=int, default=20)
    args = parser.parse_args()
    if args.batches < 5:
        parser.error("--batches must be at least 5")
    profile = authenticate_execution_profile(PROFILE_ID, device=args.device, config_hash=f2_recipe_sha256())
    if profile["git_dirty"]:
        raise RuntimeError("preflight requires a clean checkout")
    started = time.monotonic()
    dataset = F2ArtifactDataset.production(epoch=0, runtime_root=args.f1_runtime.resolve(), authenticate_objects=True)
    authentication_seconds = time.monotonic() - started
    recipe = f2_recipe()

    sample_count = min(len(dataset), int(recipe["batch_size"]) * args.batches)
    subset = Subset(dataset, list(range(sample_count)))
    load_started = time.monotonic()
    loaded = 0
    for inputs, _labels in DataLoader(
        subset,
        batch_size=int(recipe["batch_size"]),
        num_workers=int(recipe["dataloader_workers"]),
        pin_memory=bool(recipe["pin_memory"]),
    ):
        loaded += int(inputs.shape[0])
    load_seconds = time.monotonic() - load_started
    loader_examples_per_second = loaded / load_seconds
    loader_steps_per_second = (loaded / int(recipe["batch_size"])) / load_seconds

    device = torch.device(args.device)
    model = load_frozen_reference_classifier(device, allow_download=False)
    model.requires_grad_(True)
    model.train()
    loss_function = nn.CrossEntropyLoss(label_smoothing=float(recipe["label_smoothing"]))
    batch_size = int(recipe["batch_size"])
    inputs = torch.zeros((batch_size, 3, 160, 160), dtype=torch.float32, device=device)
    labels = torch.arange(batch_size, device=device) % 10
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        loss_function(model(inputs), labels).backward()
    torch.cuda.synchronize(device)
    compute_started = time.monotonic()
    for _ in range(args.batches):
        model.zero_grad(set_to_none=True)
        loss_function(model(inputs), labels).backward()
    torch.cuda.synchronize(device)
    compute_seconds = time.monotonic() - compute_started
    synthetic_steps_per_second = args.batches / compute_seconds
    synthetic_examples_per_second = batch_size * synthetic_steps_per_second
    peak_vram_bytes = torch.cuda.max_memory_allocated(device)

    with tempfile.TemporaryDirectory(prefix="g8-f2-preflight-") as directory:
        checkpoint_path = Path(directory) / "checkpoint.pt"
        checkpoint_started = time.monotonic()
        torch.save({"model_state": model.state_dict()}, checkpoint_path)
        with checkpoint_path.open("rb") as stream:
            while stream.read(1024 * 1024):
                pass
        checkpoint_seconds = time.monotonic() - checkpoint_started
        checkpoint_bytes = checkpoint_path.stat().st_size

    effective_steps_per_second = min(loader_steps_per_second, synthetic_steps_per_second)
    epoch_seconds = EXPECTED_STEPS_PER_EPOCH / effective_steps_per_second
    # Validation is forward-only and only 1,000 examples; this conservative
    # estimate charges it at full synthetic forward/backward batch time.
    validation_batches = 8
    validation_seconds_per_epoch = validation_batches / synthetic_steps_per_second
    projected_seconds = EXPECTED_EPOCHS * (epoch_seconds + validation_seconds_per_epoch + checkpoint_seconds)
    projection = {
        "schema_version": 1,
        "kind": "read_only_artifact_io_and_synthetic_resnet18_forward_backward_no_optimizer_step",
        "profile": profile,
        "dataset": {
            "logical_length": len(dataset),
            "unique_reconstruction_sha256": dataset.summary.unique_reconstruction_sha256,
            "authentication_seconds": authentication_seconds,
            "sampled_examples": loaded,
            "loader_examples_per_second": loader_examples_per_second,
            "loader_steps_per_second": loader_steps_per_second,
        },
        "synthetic_compute": {
            "batch_size": batch_size,
            "measured_batches": args.batches,
            "steps_per_second": synthetic_steps_per_second,
            "examples_per_second": synthetic_examples_per_second,
            "peak_vram_bytes": peak_vram_bytes,
            "peak_vram_gib": peak_vram_bytes / (1024 ** 3),
        },
        "checkpoint": {"bytes": checkpoint_bytes, "seconds": checkpoint_seconds, "cadence_epochs": 1},
        "projection": {
            "effective_steps_per_second": effective_steps_per_second,
            "epoch_seconds": epoch_seconds,
            "validation_seconds_per_epoch": validation_seconds_per_epoch,
            "epochs": EXPECTED_EPOCHS,
            "optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
            "total_seconds": projected_seconds,
            "total_hours": projected_seconds / 3600,
            "max_hours": 4,
            "within_max_wall_clock": projected_seconds <= 4 * 3600,
        },
        "protected_counters": {"scientific_optimizer_steps": 0, "f2_checkpoint_selection_validation_inference": 0, "f3_cached_sweep_inference": 0, "pass_two": 0, "fallback": 0, "learned_training": 0, "test_access": 0},
    }
    if len(dataset) != EXPECTED_MATERIALIZED or not projection["projection"]["within_max_wall_clock"]:
        raise RuntimeError("F2 preflight dataset or four-hour projection gate failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(projection))
    print(json.dumps(projection["projection"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
