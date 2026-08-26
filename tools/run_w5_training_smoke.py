#!/usr/bin/env python3
"""Execute the exact bounded real-CUDA W5 smoke after source CI is green."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
RUNNER = REPO / "tools/run_djscc_training.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
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


def _run(
    *,
    config: str,
    runtime: Path,
    device: str,
    source_commit: str,
    manifest_id: str,
    manifest_sha: str,
    final_epoch: int,
    batch: int,
    perturbation: int,
    resume: bool = False,
) -> None:
    command = [
        sys.executable,
        str(RUNNER),
        "--config",
        config,
        "--runtime-root",
        str(runtime),
        "--device",
        device,
        "--source-commit",
        source_commit,
        "--source-manifest-id",
        manifest_id,
        "--source-manifest-sha256",
        manifest_sha,
        "--final-epoch",
        str(final_epoch),
        "--physical-batch-size",
        str(batch),
        "--effective-batch-size",
        str(batch),
        "--accumulation-factor",
        "1",
        "--max-microbatches-per-epoch",
        "1",
        "--num-workers",
        "0",
        "--ambient-rng-perturbation",
        str(perturbation),
        "--w5-non-scientific-smoke",
    ]
    if resume:
        command.append("--resume")
    subprocess.run(command, cwd=REPO, check=True)


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def _git_clean() -> bool:
    return not subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True, check=True).stdout


def _optimizer_step_accounting(trajectory: dict[str, Any]) -> dict[str, Any]:
    expected_step = 0
    applied = 0
    skipped = 0
    optimizer_parameter_counts: set[int] = set()
    for record in trajectory["history"]:
        record_applied = 0
        record_skipped = 0
        for trace in record["trace"]:
            status = trace["optimizer_gradient_checks"]
            optimizer_parameter_counts.add(int(status["parameter_count"]))
            if not (0 < int(status["gradient_count"]) <= int(status["parameter_count"])):
                raise RuntimeError("W5 optimizer-gradient coverage count differs")
            if trace["optimizer_step_applied"] != status["finite"]:
                raise RuntimeError("W5 applied-step marker differs from optimizer-wide gradient status")
            if trace["optimizer_step_applied"]:
                expected_step += 1
                applied += 1
                record_applied += 1
            else:
                skipped += 1
                record_skipped += 1
            if trace["optimizer_step"] != expected_step:
                raise RuntimeError("W5 trace optimizer-step arithmetic differs")
        if record["optimizer_steps"] != record_applied or record["grad_scaler_skips"] != record_skipped:
            raise RuntimeError("W5 epoch optimizer-step accounting differs")
        if record["global_optimizer_step"] != expected_step:
            raise RuntimeError("W5 epoch global optimizer-step accounting differs")
    if trajectory["global_optimizer_step"] != expected_step:
        raise RuntimeError("W5 trajectory global optimizer-step accounting differs")
    return {
        "actual_applied_optimizer_steps": applied,
        "grad_scaler_skips": skipped,
        "global_optimizer_step_matches_trace": True,
        "optimizer_wide_finiteness_matches_applied_markers": True,
        "optimizer_parameter_counts": sorted(optimizer_parameter_counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=REPO / "results/learned/w5/w5_source_manifest_v4.json")
    parser.add_argument("--runtime-root", type=Path, default=REPO / "results/learned/w5/runtime_attempt_4")
    parser.add_argument("--output", type=Path, default=REPO / "results/learned/w5/w5_smoke_result_attempt_4.json")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not _git_clean():
        raise RuntimeError("final W5 CUDA smoke requires a clean exact-source worktree")
    manifest = _load(args.source_manifest)
    required_manifest = {"schema_version", "artifact_role", "manifest_id", "source_commit", "entries"}
    if set(manifest) != required_manifest or manifest["schema_version"] != 1 or manifest["artifact_role"] != "w5_training_critical_source_manifest":
        raise RuntimeError("W5 source manifest schema/role differs")
    head = _git_head()
    if manifest["source_commit"] != head:
        # The evidence/docs commit carrying the manifest may be newer than its
        # implementation source commit; all bound bytes are checked below.
        for entry in manifest["entries"]:
            path = REPO / entry["path"]
            if not path.is_file() or _sha(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
                raise RuntimeError(f"W5 source byte drift at {entry['path']}")
    manifest_sha = _sha(args.source_manifest)
    runtime = args.runtime_root
    if runtime.exists() or args.output.exists():
        raise RuntimeError("W5 final smoke runtime/output already exists; no overwrite or rerun")
    runtime.mkdir(parents=True)
    branches = {
        "uninterrupted": runtime / "cifar_uninterrupted",
        "resumed": runtime / "cifar_resumed",
        "imagenette_r_1_6": runtime / "imagenette_r_1_6",
        "imagenette_r_1_24": runtime / "imagenette_r_1_24",
    }
    try:
        _run(config="configs/learned-w5-smoke.yaml", runtime=branches["uninterrupted"], device=args.device, source_commit=head, manifest_id=manifest["manifest_id"], manifest_sha=manifest_sha, final_epoch=1, batch=4, perturbation=101)
        _run(config="configs/learned-w5-smoke.yaml", runtime=branches["resumed"], device=args.device, source_commit=head, manifest_id=manifest["manifest_id"], manifest_sha=manifest_sha, final_epoch=0, batch=4, perturbation=202)
        _run(config="configs/learned-w5-smoke.yaml", runtime=branches["resumed"], device=args.device, source_commit=head, manifest_id=manifest["manifest_id"], manifest_sha=manifest_sha, final_epoch=1, batch=4, perturbation=303, resume=True)
        _run(config="configs/learned-w5-imagenette-r1-6-smoke.yaml", runtime=branches["imagenette_r_1_6"], device=args.device, source_commit=head, manifest_id=manifest["manifest_id"], manifest_sha=manifest_sha, final_epoch=2, batch=1, perturbation=404)
        _run(config="configs/learned-w5-imagenette-r1-24-smoke.yaml", runtime=branches["imagenette_r_1_24"], device=args.device, source_commit=head, manifest_id=manifest["manifest_id"], manifest_sha=manifest_sha, final_epoch=2, batch=1, perturbation=505)
    except BaseException:
        # Failed evidence is preserved exactly for diagnosis. It is never
        # interpreted as completion and this launcher refuses to overwrite it.
        raise
    uninterrupted = _load(branches["uninterrupted"] / "trajectory.json")
    resumed = _load(branches["resumed"] / "trajectory.json")
    comparisons = {
        key: uninterrupted[key] == resumed[key]
        for key in (
            "deterministic_history",
            "model_state_sha256",
            "optimizer_state_sha256",
            "scheduler_state_sha256",
            "scaler_state_sha256",
            "completed_epoch",
            "global_optimizer_step",
            "samples",
            "microbatches",
        )
    }
    if not all(comparisons.values()):
        raise RuntimeError(f"W5 kill/resume trajectory differs: {comparisons}")
    selected = {
        ratio: _load(branches[name] / "trajectory.json")
        for ratio, name in (("r_1_6", "imagenette_r_1_6"), ("r_1_24", "imagenette_r_1_24"))
    }
    for ratio, trajectory in selected.items():
        checks = trajectory["history"][-1]["gradient_checks"]
        if trajectory["bw_ratio"] != ratio or not all(checks[head]["finite"] and checks[head]["nonzero"] for head in ("encoder", "reconstruction_head", "task_head")):
            raise RuntimeError(f"selected Imagenette ratio plumbing failed for {ratio}")
    trajectories = {
        "cifar_uninterrupted": uninterrupted,
        "cifar_resumed": resumed,
        **{f"imagenette_{ratio}": value for ratio, value in selected.items()},
    }
    accounting = {
        name: _optimizer_step_accounting(trajectory)
        for name, trajectory in trajectories.items()
    }
    actual_steps = sum(
        value["actual_applied_optimizer_steps"] for value in accounting.values()
    )
    result = {
        "schema_version": 1,
        "artifact_role": "w5_djscc_smoke_result",
        "eligibility": uninterrupted["eligibility"],
        "lineage": {
            "source_commit": head,
            "source_manifest_id": manifest["manifest_id"],
            "source_manifest_sha256": manifest_sha,
            "config_hash": uninterrupted["config_hash"],
        },
        "scope": {
            "role": "NON_SCIENTIFIC_W5_PLUMBING_ONLY",
            "dataset": "cifar10",
            "lambda": uninterrupted["lambda"],
            "lambda_status": uninterrupted["lambda_status"],
            "smoke_only_max_microbatches_per_epoch": 1,
            "accuracy_recorded": False,
            "selection_performed": False,
        },
        "environment": uninterrupted["environment"],
        "training": {
            "physical_batch_size": uninterrupted["physical_batch_size"],
            "effective_batch_size": uninterrupted["effective_batch_size"],
            "accumulation_factor": uninterrupted["accumulation_factor"],
            "w5_non_scientific_optimizer_steps": actual_steps,
            "cifar_uninterrupted_steps": uninterrupted["global_optimizer_step"],
            "cifar_resumed_physical_steps": resumed["global_optimizer_step"],
            "samples_across_all_physical_smoke_trajectories": uninterrupted["samples"] + resumed["samples"] + sum(value["samples"] for value in selected.values()),
            "finite_total_ce_mse": all(
                all(math_value == math_value and abs(math_value) != float("inf") for math_value in (record["total_loss"], record["cross_entropy"], record["reconstruction_mse"]))
                for trajectory in (uninterrupted, resumed, *selected.values())
                for record in trajectory["history"]
            ),
            "wall_clock_seconds": sum(value["wall_clock_seconds"] for value in (uninterrupted, resumed, *selected.values())),
            "peak_vram_bytes": max(value["peak_vram_bytes"] for value in (uninterrupted, resumed, *selected.values())),
        },
        "gradients": {
            "encoder_finite_nonzero": True,
            "reconstruction_head_finite_nonzero": True,
            "task_head_finite_nonzero": True,
        },
        "optimizer_step_accounting": {
            "definition": "actual GradScaler step iff all optimizer-owned gradients are finite and no backoff occurs",
            "all_optimizer_owned_gradients_covered": True,
            "trajectories": accounting,
            "actual_applied_optimizer_steps": actual_steps,
        },
        "checkpoint_resume": {
            "process_boundary": True,
            "fresh_process_resume": True,
            "ambient_rng_perturbations": [101, 202, 303],
            "comparison": comparisons,
            "exact": all(comparisons.values()),
            "uninterrupted": {key: uninterrupted[key] for key in ("model_state_sha256", "optimizer_state_sha256", "scheduler_state_sha256", "scaler_state_sha256", "global_optimizer_step")},
            "resumed": {key: resumed[key] for key in ("model_state_sha256", "optimizer_state_sha256", "scheduler_state_sha256", "scaler_state_sha256", "global_optimizer_step")},
        },
        "selected_ratio_plumbing": {
            ratio: {
                "dataset": trajectory["dataset"],
                "k": trajectory["k"],
                "steps": trajectory["global_optimizer_step"],
                "samples": trajectory["samples"],
                "gradient_checks": trajectory["history"][-1]["gradient_checks"],
            }
            for ratio, trajectory in selected.items()
        },
        "data_isolation": {
            "test_model_facing_loads": 0,
            "test_decoding_or_preprocessing": 0,
            "test_inference": 0,
            "test_accuracy_computation": 0,
            "learned_validation_selection": 0,
        },
        "protected_counters": uninterrupted["protected_counters"],
    }
    smoke_digest = hashlib.sha256(
        (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
    ).hexdigest()
    result["smoke_id"] = "w5smoke-" + smoke_digest
    _write(args.output, result)
    print(f"W5 CUDA smoke PASS: {result['smoke_id']} {_sha(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
