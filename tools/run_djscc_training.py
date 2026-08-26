#!/usr/bin/env python3
"""Run only the owner-authorized W5 non-scientific DJSCC plumbing smoke."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from config.run_config import config_hash, load_experiment  # noqa: E402
from env import profile_environment_record, set_deterministic_backend  # noqa: E402
from training.djscc import (  # noqa: E402
    ELIGIBILITY,
    PROTECTED_COUNTERS,
    DJSCCTrainer,
    W5SmokeLimits,
    W5SourceLineage,
    deterministic_history,
    model_state_sha256,
    state_tree_sha256,
)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
    with temporary.open("wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/learned-w5-smoke.yaml")
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-manifest-id", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--final-epoch", type=int, required=True)
    parser.add_argument("--physical-batch-size", type=int, required=True)
    parser.add_argument("--effective-batch-size", type=int, required=True)
    parser.add_argument("--accumulation-factor", type=int, required=True)
    parser.add_argument("--max-microbatches-per-epoch", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ambient-rng-perturbation", type=int, default=0)
    parser.add_argument("--w5-non-scientific-smoke", action="store_true")
    args = parser.parse_args()
    if not args.w5_non_scientific_smoke:
        parser.error("the only authorized launcher requires --w5-non-scientific-smoke")
    if args.source_commit != _git_commit():
        raise RuntimeError("source commit must equal the exact checked-out HEAD")
    set_deterministic_backend()
    # Deliberately perturb every ambient stream before construction. AM-91's
    # keyed trajectory must be invariant to this process-local accident.
    if args.ambient_rng_perturbation:
        import random

        random.seed(args.ambient_rng_perturbation)
        np.random.seed(args.ambient_rng_perturbation)
        torch.manual_seed(args.ambient_rng_perturbation)
        _ = random.random(), np.random.standard_normal(17), torch.randn(19)  # literal-ok: non-scientific RNG perturbation counts
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.ambient_rng_perturbation)
            _ = torch.randn(23, device=args.device)  # literal-ok: non-scientific RNG perturbation count
    config = load_experiment(args.config)
    if args.device.startswith("cuda"):
        environment = profile_environment_record(
            config.resolved["execution_profile_id"],
            device=args.device,
            config_hash=config_hash(config),
            require_openjpeg=False,
        )
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    else:
        environment = {
            "execution_profile_id": config.resolved["execution_profile_id"],
            "device": args.device,
            "role": "unit_or_cpu_non_scientific_w5_plumbing",
        }
    lineage = W5SourceLineage(
        args.source_commit,
        args.source_manifest_id,
        args.source_manifest_sha256,
    )
    limits = W5SmokeLimits(
        args.physical_batch_size,
        args.effective_batch_size,
        args.accumulation_factor,
        args.max_microbatches_per_epoch,
        args.num_workers,
    )
    trainer = DJSCCTrainer(
        config,
        device=args.device,
        runtime_root=args.runtime_root,
        source_lineage=lineage,
        smoke_limits=limits,
    )
    if args.resume:
        trainer.resume()
    started = time.monotonic()
    checkpoints = trainer.run_epochs(final_epoch=args.final_epoch)
    wall_clock = time.monotonic() - started
    peak_bytes = (
        int(torch.cuda.max_memory_reserved(torch.device(args.device)))
        if args.device.startswith("cuda")
        else 0
    )
    result = {
        "schema_version": 1,
        "artifact_role": "w5_djscc_process_trajectory",
        "eligibility": dict(ELIGIBILITY),
        "config_hash": config_hash(config),
        "source_commit": args.source_commit,
        "source_manifest_id": args.source_manifest_id,
        "source_manifest_sha256": args.source_manifest_sha256,
        "execution_profile_id": config.resolved["execution_profile_id"],
        "dataset": config.resolved["dataset"],
        "bw_ratio": config.resolved["bw_ratio"],
        "k": config.resolved["k"],
        "lambda": config.resolved["lambda"],
        "lambda_status": config.parameters["learned_system"]["lambda_status"],
        "completed_epoch": trainer.completed_epoch,
        "global_optimizer_step": trainer.global_optimizer_step,
        "samples": sum(record["samples"] for record in trainer.training_history),
        "microbatches": sum(record["microbatches"] for record in trainer.training_history),
        "physical_batch_size": limits.physical_batch_size,
        "effective_batch_size": limits.effective_batch_size,
        "accumulation_factor": limits.accumulation_factor,
        "history": trainer.training_history,
        "deterministic_history": deterministic_history(trainer.training_history),
        "model_state_sha256": model_state_sha256(trainer.model),
        "optimizer_state_sha256": state_tree_sha256(trainer.optimizer.state_dict()),
        "scheduler_state_sha256": state_tree_sha256(trainer.scheduler.state_dict()),
        "scaler_state_sha256": None if trainer.scaler is None else state_tree_sha256(trainer.scaler.state_dict()),
        "checkpoints": checkpoints,
        "environment": environment,
        "wall_clock_seconds": wall_clock,
        "peak_vram_bytes": peak_bytes,
        "protected_counters": dict(PROTECTED_COUNTERS),
    }
    _write_json(args.runtime_root / "trajectory.json", result)
    print(json.dumps({key: result[key] for key in ("completed_epoch", "global_optimizer_step", "samples", "model_state_sha256")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
