#!/usr/bin/env python3
"""Profile one full DJSCC training epoch for compute gate G-7."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from config.params import get  # noqa: E402
from config.run_config import FrozenMap, RunConfig, config_hash  # noqa: E402
from data.classifier import EpochPermutationSampler, TrainingClassifierDataset  # noqa: E402
from env import assert_cuda, environment_record, set_deterministic_backend  # noqa: E402
from models.djscc import build_djscc  # noqa: E402
from models.reference_classifier import build_reference_classifier  # noqa: E402
from training.djscc_loss import DJSCCObjective  # noqa: E402

DEFAULT_CONFIG = REPO / "configs/learned-g7-profile.yaml"
DEFAULT_REPORT = REPO / "results/profiling/g7_djscc_profile.json"
_PROFILE_CHOICE_KEYS = {
    "system",
    "dataset",
    "split",
    "bw_ratio",
    "channel",
    "train_snr_db",
    "lambda",
    "classifier_variant",
    "train_seed",
    "channel_seed",
    "batch_size",
    "num_workers",
    "architecture",
    "warmup_steps",
    "implementation_commit",
}


class ProfileError(ValueError):
    """A fail-closed profile-configuration or repository-state error."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(git_repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=git_repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise ProfileError(
            f"git {' '.join(args)} failed in {git_repo}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def load_profile_config(path: Path = DEFAULT_CONFIG) -> RunConfig:
    """Resolve the one committed G-7 profile recipe through current parameters."""

    path = path.resolve()
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict) or set(body) != {
        "experiment",
        "choices",
        "sweep_axes",
    }:
        raise ProfileError("G-7 profile config must contain experiment, choices, sweep_axes")
    if body["experiment"] != "learned_g7_profile":
        raise ProfileError("G-7 experiment name differs")
    if body["sweep_axes"] != {}:
        raise ProfileError("G-7 profile config must be fully concrete")
    choices = body["choices"]
    if not isinstance(choices, dict) or set(choices) != _PROFILE_CHOICE_KEYS:
        missing = _PROFILE_CHOICE_KEYS - set(choices) if isinstance(choices, dict) else _PROFILE_CHOICE_KEYS
        extra = set(choices) - _PROFILE_CHOICE_KEYS if isinstance(choices, dict) else set()
        raise ProfileError(
            f"G-7 choice keys differ: missing={sorted(missing)}, extra={sorted(extra)}"
        )

    expected = {
        "system": "learned",
        "dataset": "imagenette160",
        "split": "train",
        "bw_ratio": "r_1_2",
        "channel": "awgn",
        "train_snr_db": "train_snr_db_fixed",
        "lambda": "lambda_core",
        "classifier_variant": get("reference_classifier.clean_variant_name"),
        "train_seed": get("evaluation.train_seeds")[0],
        "channel_seed": get("evaluation.channel_seeds")[0],
        "batch_size": get("learned_system.batch_size.imagenette160"),
        "num_workers": 0,
        "architecture": get("learned_system.encoder_arch"),
    }
    for key, expected_value in expected.items():
        if choices[key] != expected_value:
            raise ProfileError(
                f"G-7 choice {key}={choices[key]!r}, expected {expected_value!r}"
            )
    if (
        not isinstance(choices["warmup_steps"], int)
        or isinstance(choices["warmup_steps"], bool)
        or choices["warmup_steps"] <= 0
    ):
        raise ProfileError("G-7 warmup_steps must be a positive integer")
    commit = choices["implementation_commit"]
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ProfileError("G-7 implementation_commit must be a full lowercase SHA")

    dataset = choices["dataset"]
    ratio = choices["bw_ratio"]
    resolved = dict(choices)
    resolved["train_snr_db"] = get("channel.train_snr_db_fixed")
    resolved["lambda"] = get("learned_system.lambda_core")
    resolved["dataset_version"] = get(
        f"datasets.{dataset}.{get('config.dataset_version_rule')}"
    )
    resolved["analysis_version"] = get("config.analysis_version")
    resolved["k"] = get(f"bandwidth.k_symbols.{dataset}.{ratio}")
    resolved["project_id"] = get("project.id")
    resolved["task"] = get("project.task")
    try:
        source = str(path.relative_to(REPO))
    except ValueError:
        source = str(path)
    return RunConfig(
        fingerprint_schema_version=get("config.fingerprint_schema_version"),
        experiment=body["experiment"],
        source=source,
        choices=FrozenMap.from_mapping(choices),
        sweep_axes=FrozenMap.from_mapping({}),
        resolved=FrozenMap.from_mapping(resolved),
        parameters=FrozenMap.from_mapping(
            {
                root: get(root)
                for root in get("config.fingerprint_parameter_roots")
            }
        ),
    )


def _build_training_state(config: RunConfig, device: torch.device):
    model = build_djscc(config, device=device).train()
    objective = DJSCCObjective.from_config(config)
    optimizer_name = config.parameters["learned_system"]["optimizer"]
    if optimizer_name != "adam":
        raise ProfileError(f"unsupported G-7 optimizer: {optimizer_name}")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.parameters["learned_system"]["lr"]
    )
    amp = bool(config.parameters["learned_system"]["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    return model, objective, optimizer, scaler, amp


def _step(
    batch: tuple[torch.Tensor, torch.Tensor],
    *,
    model,
    objective,
    optimizer,
    scaler,
    amp: bool,
    device: torch.device,
) -> tuple[int, torch.Tensor]:
    inputs, targets = batch
    inputs = inputs.to(device, non_blocking=False)
    targets = targets.to(device, non_blocking=False)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=amp):
        output = model(inputs, model_config_snr(model))
        loss = objective(output, targets, inputs)
    scaler.scale(loss.total).backward()
    scaler.step(optimizer)
    scaler.update()
    return inputs.shape[0], loss.total.detach()


def model_config_snr(model) -> float:
    """Return the config-bound training SNR installed by :func:`profile`."""

    return model.g7_train_snr_db


def profile(
    *,
    config_path: Path,
    report_path: Path,
    git_repo: Path,
    data_repo: Path,
) -> dict[str, Any]:
    config = load_profile_config(config_path)
    resolved = config.resolved
    implementation_commit = _git(git_repo, "rev-parse", "HEAD")
    dirty_output = _git(git_repo, "status", "--porcelain", "--untracked-files=all")
    git_dirty = bool(dirty_output)
    if implementation_commit != resolved["implementation_commit"]:
        raise ProfileError(
            f"profile target HEAD {implementation_commit} is not configured "
            f"implementation commit {resolved['implementation_commit']}"
        )
    if git_dirty:
        raise ProfileError("profile target repository is dirty")

    assert_cuda()
    if not torch.cuda.is_available():
        raise ProfileError("real CUDA is unavailable; G-7 cannot be profiled")
    set_deterministic_backend()
    device = torch.device("cuda", 0)
    properties = torch.cuda.get_device_properties(device)

    dataset_name = resolved["dataset"]
    dataset = TrainingClassifierDataset(
        dataset_name,
        int(resolved["train_seed"]),
        0,
        repo_root=data_repo,
    )
    sampler = EpochPermutationSampler(
        len(dataset), int(resolved["train_seed"]), 0
    )
    loader = DataLoader(
        dataset,
        batch_size=int(resolved["batch_size"]),
        sampler=sampler,
        num_workers=int(resolved["num_workers"]),
        drop_last=False,
    )
    expected_examples = int(config.parameters["datasets"][dataset_name]["train_images"])
    expected_batches = math.ceil(expected_examples / int(resolved["batch_size"]))
    if len(dataset) != expected_examples or len(loader) != expected_batches:
        raise ProfileError("manifest-backed training view has unexpected epoch size")

    model, objective, optimizer, scaler, amp = _build_training_state(config, device)
    model.g7_train_snr_db = float(resolved["train_snr_db"])
    warmup_steps = int(resolved["warmup_steps"])
    warmup_iterator = iter(loader)
    for _ in range(warmup_steps):
        try:
            batch = next(warmup_iterator)
        except StopIteration:
            warmup_iterator = iter(loader)
            batch = next(warmup_iterator)
        _step(
            batch,
            model=model,
            objective=objective,
            optimizer=optimizer,
            scaler=scaler,
            amp=amp,
            device=device,
        )
    torch.cuda.synchronize(device)
    del model, objective, optimizer, scaler, warmup_iterator
    torch.cuda.empty_cache()

    model, objective, optimizer, scaler, amp = _build_training_state(config, device)
    model.g7_train_snr_db = float(resolved["train_snr_db"])
    parameter_count = model.total_parameter_count
    reference_model = build_reference_classifier(dataset_name)
    reference_parameter_count = sum(
        parameter.numel() for parameter in reference_model.parameters()
    )
    del reference_model

    torch.cuda.reset_peak_memory_stats(device)
    examples = 0
    batches = 0
    loss_sum = torch.zeros((), device=device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for batch in loader:
        batch_examples, batch_loss = _step(
            batch,
            model=model,
            objective=objective,
            optimizer=optimizer,
            scaler=scaler,
            amp=amp,
            device=device,
        )
        examples += batch_examples
        batches += 1
        loss_sum += batch_loss * batch_examples
    torch.cuda.synchronize(device)
    epoch_time_s = time.perf_counter() - started
    epoch_mean_loss = float((loss_sum / examples).cpu())
    peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
    peak_reserved_bytes = torch.cuda.max_memory_reserved(device)

    epochs = int(config.parameters["learned_system"]["epochs"][dataset_name])
    projected_hours = epoch_time_s * epochs / 3600
    images_per_second = examples / epoch_time_s
    gib = 1024**3
    peak_allocated_gb = peak_allocated_bytes / gib
    peak_reserved_gb = peak_reserved_bytes / gib
    wall_limit = float(config.parameters["compute"]["max_wall_clock_hours_per_run"])
    vram_limit = float(config.parameters["compute"]["vram_budget_gb"])
    absolute_cap = int(
        float(config.parameters["learned_system"]["max_params_millions"])
        * 1_000_000
    )
    conditions = {
        "primary_architecture_full_epoch": "PASS"
        if batches == expected_batches and examples == expected_examples
        else "FAIL",
        "configured_batch_size_32": "PASS"
        if resolved["batch_size"]
        == config.parameters["learned_system"]["batch_size"][dataset_name]
        and resolved["batch_size"] >= 32
        else "FAIL",
        "peak_reserved_vram": "PASS"
        if peak_reserved_gb <= vram_limit
        else "FAIL",
        "projected_training_wall_time": "PASS"
        if projected_hours <= wall_limit
        else "FAIL",
        "absolute_parameter_cap": "PASS"
        if parameter_count <= absolute_cap
        else "FAIL",
        "reference_parameter_cap": "PASS"
        if parameter_count <= reference_parameter_count
        else "FAIL",
        "clean_implementation_commit": "PASS"
        if not git_dirty
        and implementation_commit == resolved["implementation_commit"]
        else "FAIL",
        "real_cuda": "PASS",
        "training_split_only": "PASS" if resolved["split"] == "train" else "FAIL",
        "report_consistency": "PASS",
    }
    verdict = "PASS" if all(value == "PASS" for value in conditions.values()) else "HOLD"

    manifest_path = (
        data_repo
        / config.parameters["datasets"]["manifest_dir"]
        / config.parameters["datasets"][dataset_name]["manifest_filename"]
    )
    run_environment = environment_record()
    report = {
        "schema_version": 1,
        "gate": "G-7",
        "verdict": verdict,
        "profiled_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_commit": implementation_commit,
        "git_dirty": git_dirty,
        "profile_config_path": "configs/learned-g7-profile.yaml",
        "profile_config_sha256": _sha256(config_path),
        "config_hash": config_hash(config),
        "resolved_config": config.to_dict(),
        "dataset": {
            "name": dataset_name,
            "split": resolved["split"],
            "dataset_version": resolved["dataset_version"],
            "archive_filename": config.parameters["datasets"][dataset_name][
                "archive_filename"
            ],
            "archive_bytes": config.parameters["datasets"][dataset_name][
                "archive_bytes"
            ],
            "manifest_path": str(manifest_path.relative_to(data_repo)),
            "manifest_sha256": _sha256(manifest_path),
            "configured_train_examples": expected_examples,
        },
        "environment": {
            "run_metadata": run_environment,
            "torch_cuda_available": torch.cuda.is_available(),
            "real_cuda": True,
            "device_index": device.index,
            "device_name": properties.name,
            "driver_version": run_environment["driver_version"],
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [
                properties.major,
                properties.minor,
            ],
        },
        "model": {
            "architecture": resolved["architecture"],
            "bw_ratio": resolved["bw_ratio"],
            "k": resolved["k"],
            "complex_channels": config.parameters["learned_system"][
                "encoder_output_complex_channels"
            ][resolved["bw_ratio"]],
            "parameter_count": parameter_count,
            "absolute_parameter_cap": absolute_cap,
            "reference_classifier_parameter_count": reference_parameter_count,
            "peak_constraint_enabled": False,
        },
        "training": {
            "optimizer": config.parameters["learned_system"]["optimizer"],
            "amp": amp,
            "batch_size": resolved["batch_size"],
            "num_workers": resolved["num_workers"],
            "warmup_steps": warmup_steps,
            "epochs_completed": 1,
            "num_batches": batches,
            "num_examples": examples,
            "epoch_time_s": epoch_time_s,
            "images_per_second": images_per_second,
            "epoch_mean_loss": epoch_mean_loss,
            "projected_epochs": epochs,
            "projected_training_hours": projected_hours,
        },
        "memory": {
            "peak_allocated_bytes": peak_allocated_bytes,
            "peak_reserved_bytes": peak_reserved_bytes,
            "peak_allocated_gb": peak_allocated_gb,
            "peak_reserved_gb": peak_reserved_gb,
        },
        "limits": {
            "configured_batch_size": config.parameters["learned_system"][
                "batch_size"
            ][dataset_name],
            "vram_budget_gb": vram_limit,
            "max_wall_clock_hours_per_run": wall_limit,
            "max_params": absolute_cap,
            "reference_classifier_params": reference_parameter_count,
        },
        "conditions": conditions,
        "data_isolation": {
            "scope": "training_only",
            "test_split_accessed": False,
            "test_inference": False,
            "test_accuracy_computed": False,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--git-repo",
        type=Path,
        default=REPO,
        help="clean worktree whose HEAD and dirty state bind the profile",
    )
    parser.add_argument(
        "--data-repo",
        type=Path,
        default=REPO,
        help="repository root containing verified local training data",
    )
    args = parser.parse_args()
    try:
        report = profile(
            config_path=args.config.resolve(),
            report_path=args.output.resolve(),
            git_repo=args.git_repo.resolve(),
            data_repo=args.data_repo.resolve(),
        )
    except (OSError, ProfileError, RuntimeError, ValueError) as exc:
        print(f"G-7 profiling FAILED: {exc}", file=sys.stderr)
        return 1
    print(
        "G-7 profiling "
        f"{report['verdict']}: epoch={report['training']['epoch_time_s']:.3f}s, "
        f"batch={report['training']['batch_size']}, "
        f"reserved={report['memory']['peak_reserved_gb']:.3f} GB, "
        f"projected={report['training']['projected_training_hours']:.3f} h"
    )
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
