"""Deterministic learned DJSCC training engine frozen for W5 (AM-91).

The only launcher currently exposed is the machine-labelled non-scientific W5
plumbing launcher. Later scientific launchers must supply a separately
owner-authorized eligibility policy; W5 checkpoints cannot be promoted.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from artifacts.rng import keyed_standard_normal
from config.params import REPO_ROOT, get
from config.run_config import RunConfig, canonical_sha256, config_hash
from data.classifier import EpochPermutationSampler
from data.djscc_training import TrainingDJSCCDataset
from models.djscc import DJSCC, build_djscc
from training.djscc_loss import DJSCCObjective

SCHEMA_VERSION = 1
CHECKPOINT_ROLE = "w5_djscc_training_checkpoint"
SIDECAR_ROLE = "w5_djscc_checkpoint_sidecar"
ELIGIBILITY = {
    "artifact_role": "W5_NON_SCIENTIFIC_PLUMBING_SMOKE",
    "selection_eligibility": "NOT_ELIGIBLE_FOR_SELECTION",
    "reporting_eligibility": "NOT_ELIGIBLE_FOR_REPORTING",
    "w7_g4_eligibility": "NOT_ELIGIBLE_FOR_W7_G4",
    "w8_eligibility": "NOT_ELIGIBLE_FOR_W8",
    "test_eligibility": "NOT_ELIGIBLE_FOR_TEST",
}
PROTECTED_COUNTERS = {
    "scientific_learned_training_runs": 0,
    "w7_lambda_pilot_runs": 0,
    "w8_final_training_runs": 0,
    "learned_validation_selection": 0,
    "learned_test_inference": 0,
    "test_access": 0,
    "f2_optimizer_steps_during_w5": 0,
    "f3_reruns": 0,
    "pass_two_reruns": 0,
    "pass_three": 0,
}
RNG_STATE_POLICY = {
    "python_random": "not_consumed",
    "numpy": "keyed_stateless_only",
    "torch_cpu": "not_consumed_after_isolated_keyed_initialization",
    "torch_cuda": "not_consumed_by_training_stochasticity",
    "channel": "keyed_training_channel_noise_per_sample_epoch",
    "augmentation": "keyed_per_stable_sample_train_seed_epoch",
    "batch_order": "keyed_per_train_seed_epoch",
    "serialized_sequential_rng_states": [],
}


class W5Hold(RuntimeError):
    """A fail-closed W5 schema, lineage, checkpoint or scope violation."""


_STANDARD_COMPLEX_SCALE = 1.0 / math.sqrt(2.0)


def keyed_training_complex_noise(
    identities: Sequence[Mapping[str, object]],
    k: int,
    *,
    dtype: torch.dtype = torch.complex64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Construct stateless per-sample AM-91 training-channel noise."""

    expected = list(get("artifacts.rng_identity_fields.training_channel_noise"))
    if not identities:
        raise ValueError("training noise requires one or more identities")
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")
    if dtype not in (torch.complex64, torch.complex128):
        raise TypeError("keyed training complex noise dtype must be complex64 or complex128")
    rows: list[np.ndarray] = []
    for identity in identities:
        missing = set(expected) - set(identity)
        extra = set(identity) - set(expected)
        if missing or extra:
            raise ValueError(
                "training noise identity differs: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        components = keyed_standard_normal(
            "training_channel_noise", identity, size=(2, k)
        )
        rows.append((components[0] + 1j * components[1]) * _STANDARD_COMPLEX_SCALE)
    return torch.as_tensor(np.stack(rows), dtype=dtype, device=device)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise W5Hold(message)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _full_sha(value: object, width: int = 64) -> bool:  # literal-ok: SHA-256 hexadecimal width
    return isinstance(value, str) and len(value) == width and all(
        character in "0123456789abcdef" for character in value
    )


def _canonical_bytes(value: Mapping[str, Any] | Sequence[Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):  # literal-ok: bounded hashing block
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_replace_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _publish_new_file(path: Path, writer: Callable[[Any], None]) -> None:
    """Durably publish a new immutable file without replacing an old one."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _portable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _portable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_portable(item) for item in value)
    if isinstance(value, list):
        return [_portable(item) for item in value]
    return value


def learned_recipe(config: RunConfig) -> dict[str, Any]:
    """Return every AM-91 output-affecting runtime choice explicitly."""

    learned = config.parameters["learned_system"]
    recipe = {
        "optimizer": learned["optimizer"],
        "optimizer_implementation": learned["optimizer_implementation"],
        "adam_beta1": learned["adam_beta1"],
        "adam_beta2": learned["adam_beta2"],
        "adam_epsilon": learned["adam_epsilon"],
        "adam_weight_decay": learned["adam_weight_decay"],
        "adam_amsgrad": learned["adam_amsgrad"],
        "adam_maximize": learned["adam_maximize"],
        "adam_foreach": learned["adam_foreach"],
        "adam_capturable": learned["adam_capturable"],
        "adam_differentiable": learned["adam_differentiable"],
        "adam_fused": learned["adam_fused"],
        "lr": learned["lr"],
        "lr_schedule": learned["lr_schedule"],
        "lr_schedule_equation": learned["lr_schedule_equation"],
        "lr_min": learned["lr_min"],
        "lr_warmup_epochs": learned["lr_warmup_epochs"],
        "scheduler_step_unit": learned["scheduler_step_unit"],
        "scheduler_epoch_indexing": learned["scheduler_epoch_indexing"],
        "scheduler_resume_state": learned["scheduler_resume_state"],
        "amp": learned["amp"],
        "amp_device_type": learned["amp_device_type"],
        "amp_dtype": learned["amp_dtype"],
        "grad_scaler_enabled": learned["grad_scaler_enabled"],
        "grad_scaler_init_scale": learned["grad_scaler_init_scale"],
        "grad_scaler_growth_factor": learned["grad_scaler_growth_factor"],
        "grad_scaler_backoff_factor": learned["grad_scaler_backoff_factor"],
        "grad_scaler_growth_interval": learned["grad_scaler_growth_interval"],
        "batch_order": learned["batch_order"],
        "drop_last": learned["drop_last"],
        "dataloader_workers": learned["dataloader_workers"],
        "pin_memory": learned["pin_memory"],
        "batch_size_policy": learned["batch_size_policy"],
        "accumulation_gradient_rule": learned["accumulation_gradient_rule"],
        "final_partial_accumulation": learned["final_partial_accumulation"],
        "scheduler_steps_under_accumulation": learned["scheduler_steps_under_accumulation"],
        "checkpoint_every_epochs": learned["checkpoint_every_epochs"],
        "checkpoint_timing": learned["checkpoint_timing"],
        "checkpoint_resume_unit": learned["checkpoint_resume_unit"],
        "corrupt_latest_checkpoint_policy": learned["corrupt_latest_checkpoint_policy"],
        "incomplete_epoch_policy": learned["incomplete_epoch_policy"],
        "checkpoint_schema_version": learned["checkpoint_schema_version"],
        "w5_checkpoint_selection": learned["w5_checkpoint_selection"],
        "loss": learned["loss"],
        "augmentation": list(learned["augmentation"]),
    }
    expected = {
        "optimizer": "adam",
        "optimizer_implementation": "torch.optim.Adam",
        "lr_schedule": "cosine",
        "lr_warmup_epochs": 0,
        "scheduler_step_unit": "epoch_start",
        "scheduler_epoch_indexing": "zero_based",
        "scheduler_resume_state": "completed_epoch",
        "amp": True,
        "amp_device_type": "cuda",
        "amp_dtype": "float16",
        "grad_scaler_enabled": True,
        "batch_order": "keyed_philox_permutation_per_epoch",
        "drop_last": False,
        "batch_size_policy": "effective_target_with_profile_bound_physical_microbatch",
        "accumulation_gradient_rule": "sample_weighted_mean_over_effective_batch",
        "final_partial_accumulation": "optimizer_step_over_all_remaining_samples_no_drop",
        "scheduler_steps_under_accumulation": "once_per_epoch_at_epoch_start_not_per_optimizer_step",
        "checkpoint_timing": "after_completed_epoch_and_before_next_epoch",
        "checkpoint_resume_unit": "authenticated_completed_epoch",
        "corrupt_latest_checkpoint_policy": "hold_no_older_fallback",
        "incomplete_epoch_policy": "replay_from_latest_authenticated_completed_epoch",
        "checkpoint_schema_version": SCHEMA_VERSION,
        "w5_checkpoint_selection": "prohibited_non_scientific_smoke_only",
        "loss": "CE + lambda * MSE",
        "augmentation": ["random_resized_crop", "horizontal_flip"],
    }
    for key, value in expected.items():
        _require(recipe[key] == value, f"learned recipe {key} differs from AM-91")
    return recipe


def learning_rate_for_epoch(config: RunConfig, epoch: int) -> float:
    learned = config.parameters["learned_system"]
    total = int(learned["epochs"][config.resolved["dataset"]])
    _require(_is_int(epoch) and 0 <= epoch < total, "epoch is outside configured learned schedule")
    base = float(learned["lr"])
    minimum = float(learned["lr_min"])
    return minimum + (base - minimum) * 0.5 * (  # literal-ok: AM-91 exact cosine equation
        1 + math.cos(math.pi * epoch / max(total - 1, 1))
    )


@dataclass(frozen=True)
class W5SourceLineage:
    source_commit: str
    source_manifest_id: str
    source_manifest_sha256: str

    def validate(self) -> None:
        _require(_full_sha(self.source_commit, 40), "source commit must be a full Git SHA-1")  # literal-ok: Git SHA-1 hexadecimal width
        _require(isinstance(self.source_manifest_id, str) and self.source_manifest_id, "source manifest ID is empty")
        _require(_full_sha(self.source_manifest_sha256), "source manifest SHA-256 is invalid")


@dataclass(frozen=True)
class W5SmokeLimits:
    physical_batch_size: int
    effective_batch_size: int
    accumulation_factor: int
    max_microbatches_per_epoch: int
    num_workers: int = 0

    def validate(self) -> None:
        for name in (
            "physical_batch_size",
            "effective_batch_size",
            "accumulation_factor",
            "max_microbatches_per_epoch",
        ):
            value = getattr(self, name)
            _require(_is_int(value) and value > 0, f"smoke {name} must be a positive integer")
        _require(self.effective_batch_size == self.physical_batch_size * self.accumulation_factor, "smoke batch arithmetic differs")
        _require(self.accumulation_factor == 1, "AM-91's active production/smoke accumulation factor is one")
        _require(_is_int(self.num_workers) and self.num_workers >= 0, "smoke num_workers is invalid")


class EpochScheduler:
    def __init__(self) -> None:
        self.completed_epoch = -1

    def state_dict(self) -> dict[str, int]:
        return {"completed_epoch": self.completed_epoch}

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        _require(set(value) == {"completed_epoch"}, "scheduler state schema differs")
        completed = value["completed_epoch"]
        _require(_is_int(completed) and completed >= -1, "scheduler completed epoch is invalid")
        self.completed_epoch = completed


class DJSCCTrainer:
    """Config-derived trainer with W5-only eligibility and exact-next resume."""

    def __init__(
        self,
        config: RunConfig,
        *,
        device: torch.device | str,
        runtime_root: Path,
        source_lineage: W5SourceLineage,
        smoke_limits: W5SmokeLimits,
        model: DJSCC | None = None,
    ) -> None:
        _require(isinstance(config, RunConfig), "trainer requires a resolved RunConfig")
        self.config = config
        self.device = torch.device(device)
        self.runtime_root = Path(runtime_root)
        self.source_lineage = source_lineage
        self.source_lineage.validate()
        self.smoke_limits = smoke_limits
        self.smoke_limits.validate()
        resolved = config.resolved
        _require(resolved.get("split") == "train", "learned trainer structurally requires split=train")
        _require(resolved.get("artifact_role") == ELIGIBILITY["artifact_role"], "W5 artifact role differs")
        for key, value in ELIGIBILITY.items():
            if key != "artifact_role":
                _require(resolved.get(key) == value, f"W5 {key} differs")
        profile = resolved.get("execution_profile_id")
        _require(profile in get("compute.execution_profile_policy.eligible_profiles"), "execution profile is not explicit and eligible")
        self.recipe = learned_recipe(config)
        self.recipe_sha256 = canonical_sha256(self.recipe)
        self.model = model or build_djscc(config, device=self.device)
        self.model.to(self.device)
        self.objective = DJSCCObjective.from_config(config)
        self.optimizer = self._new_optimizer(self.model)
        self.scheduler = EpochScheduler()
        self.amp_enabled = bool(self.recipe["amp"] and self.device.type == "cuda")
        self.scaler = self._new_scaler()
        self.completed_epoch = -1
        self.global_optimizer_step = 0
        self.training_history: list[dict[str, Any]] = []
        self.predecessor_checkpoint_id: str | None = None

    def _new_optimizer(self, model: nn.Module) -> Adam:
        return Adam(
            model.parameters(),
            lr=float(self.recipe["lr"]),
            betas=(float(self.recipe["adam_beta1"]), float(self.recipe["adam_beta2"])),
            eps=float(self.recipe["adam_epsilon"]),
            weight_decay=float(self.recipe["adam_weight_decay"]),
            amsgrad=bool(self.recipe["adam_amsgrad"]),
            maximize=bool(self.recipe["adam_maximize"]),
            foreach=bool(self.recipe["adam_foreach"]),
            capturable=bool(self.recipe["adam_capturable"]),
            differentiable=bool(self.recipe["adam_differentiable"]),
            fused=bool(self.recipe["adam_fused"]),
        )

    def _new_scaler(self) -> torch.amp.GradScaler | None:
        if not self.amp_enabled:
            return None
        return torch.amp.GradScaler(
            "cuda",
            init_scale=float(self.recipe["grad_scaler_init_scale"]),
            growth_factor=float(self.recipe["grad_scaler_growth_factor"]),
            backoff_factor=float(self.recipe["grad_scaler_backoff_factor"]),
            growth_interval=int(self.recipe["grad_scaler_growth_interval"]),
            enabled=bool(self.recipe["grad_scaler_enabled"]),
        )

    def _set_lr(self, epoch: int) -> float:
        value = learning_rate_for_epoch(self.config, epoch)
        for group in self.optimizer.param_groups:
            group["lr"] = value
        return value

    def _training_noise_identities(self, stable_ids: Sequence[str], epoch: int) -> list[dict[str, Any]]:
        resolved = self.config.resolved
        return [
            {
                "dataset_version": resolved["dataset_version"],
                "split_manifest_hash": get(
                    f"datasets.{resolved['dataset']}.manifest_sha256"
                ),
                "stable_sample_id": stable_id,
                "train_seed": resolved["train_seed"],
                "channel_seed": resolved["channel_seed"],
                "epoch": epoch,
                "channel": resolved["channel"],
                "bw_ratio": resolved["bw_ratio"],
                "k": resolved["k"],
                "train_snr_db": resolved["train_snr_db"],
            }
            for stable_id in stable_ids
        ]

    @staticmethod
    def _gradient_status(parameters: Sequence[nn.Parameter]) -> dict[str, bool]:
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        return {
            "present": bool(gradients),
            "finite": bool(gradients) and all(torch.isfinite(gradient).all().item() for gradient in gradients),
            "nonzero": bool(gradients) and any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients),
        }

    def train_epoch(
        self,
        epoch: int,
        dataset: Dataset[tuple[torch.Tensor, int, str]],
    ) -> dict[str, Any]:
        _require(epoch == self.completed_epoch + 1, "trainer epoch is not exact next")
        sampler = EpochPermutationSampler(len(dataset), int(self.config.resolved["train_seed"]), epoch)
        loader = DataLoader(
            dataset,
            batch_size=self.smoke_limits.physical_batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=self.smoke_limits.num_workers,
            drop_last=bool(self.recipe["drop_last"]),
            pin_memory=bool(self.recipe["pin_memory"] and self.device.type == "cuda"),
        )
        self.model.train()
        lr = self._set_lr(epoch)
        started = time.monotonic()
        total_weighted = 0.0
        ce_weighted = 0.0
        mse_weighted = 0.0
        samples = 0
        traces: list[dict[str, Any]] = []
        encoder_status = {"present": False, "finite": False, "nonzero": False}
        reconstruction_status = dict(encoder_status)
        task_status = dict(encoder_status)
        for microbatch, (inputs, labels, stable_ids) in enumerate(loader):
            if microbatch >= self.smoke_limits.max_microbatches_per_epoch:
                break
            ids = [str(value) for value in stable_ids]
            identities = self._training_noise_identities(ids, epoch)
            inputs = inputs.to(self.device, non_blocking=self.device.type == "cuda")
            labels = labels.to(self.device, non_blocking=self.device.type == "cuda")
            unit_noise = keyed_training_complex_noise(
                identities,
                int(self.config.resolved["k"]),
                dtype=torch.complex64,
                device=self.device,
            )
            self.optimizer.zero_grad(set_to_none=True)
            context = (
                torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
                if self.amp_enabled
                else nullcontext()
            )
            with context:
                output = self.model(
                    inputs,
                    self.config.resolved["train_snr_db"],
                    unit_noise=unit_noise,
                )
                loss = self.objective(output, labels, inputs)
            if self.scaler is None:
                loss.total.backward()
            else:
                self.scaler.scale(loss.total).backward()
                self.scaler.unscale_(self.optimizer)
            encoder_status = self._gradient_status(list(self.model.encoder.parameters()))
            reconstruction_status = self._gradient_status(list(self.model.decoder.reconstruction_head.parameters()))
            task_status = self._gradient_status(list(self.model.decoder.task_head.parameters()))
            _require(all(status["finite"] for status in (encoder_status, reconstruction_status, task_status)), "non-finite W5 gradient")
            _require(encoder_status["nonzero"] and task_status["nonzero"], "required W5 encoder/task gradient is zero")
            if float(self.config.resolved["lambda"]) > 0:
                _require(reconstruction_status["nonzero"], "nonzero-lambda reconstruction gradient is zero")
            if self.scaler is None:
                self.optimizer.step()
            else:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            self.global_optimizer_step += 1
            count = int(labels.numel())
            values = {
                "total": float(loss.total.detach().float().item()),
                "cross_entropy": float(loss.cross_entropy.detach().float().item()),
                "reconstruction_mse": float(loss.reconstruction_mse.detach().float().item()),
            }
            _require(all(math.isfinite(value) for value in values.values()), "non-finite W5 loss")
            total_weighted += values["total"] * count
            ce_weighted += values["cross_entropy"] * count
            mse_weighted += values["reconstruction_mse"] * count
            samples += count
            noise_bytes = torch.view_as_real(unit_noise).detach().cpu().contiguous().numpy().tobytes()
            traces.append(
                {
                    "microbatch": microbatch,
                    "stable_sample_ids": ids,
                    "augmentation_ids": [
                        canonical_sha256({"stable_sample_id": value, "train_seed": self.config.resolved["train_seed"], "epoch": epoch})
                        for value in ids
                    ],
                    "training_noise_ids": [canonical_sha256(identity) for identity in identities],
                    "training_noise_sha256": hashlib.sha256(noise_bytes).hexdigest(),
                    "optimizer_step": self.global_optimizer_step,
                    "lr": lr,
                    **values,
                }
            )
        _require(samples > 0, "W5 epoch processed no samples")
        self.completed_epoch = epoch
        self.scheduler.completed_epoch = epoch
        record = {
            "epoch": epoch,
            "lr": lr,
            "samples": samples,
            "microbatches": len(traces),
            "optimizer_steps": len(traces),
            "global_optimizer_step": self.global_optimizer_step,
            "total_loss": total_weighted / samples,
            "cross_entropy": ce_weighted / samples,
            "reconstruction_mse": mse_weighted / samples,
            "duration_seconds": time.monotonic() - started,
            "gradient_checks": {
                "encoder": encoder_status,
                "reconstruction_head": reconstruction_status,
                "task_head": task_status,
            },
            "trace": traces,
        }
        self.training_history.append(record)
        return record

    def run_epochs(
        self,
        *,
        final_epoch: int,
        dataset_factory: Callable[[int], Dataset[tuple[torch.Tensor, int, str]]] | None = None,
        repo_root: Path | None = None,
    ) -> list[dict[str, Any]]:
        _require(_is_int(final_epoch) and final_epoch >= self.completed_epoch + 1, "final epoch precedes exact next")
        total = int(self.config.parameters["learned_system"]["epochs"][self.config.resolved["dataset"]])
        _require(final_epoch < total, "final epoch exceeds configured schedule")
        interval = int(self.recipe["checkpoint_every_epochs"])
        _require(interval == int(get("compute.checkpoint_every_epochs")) == 1, "W5 checkpoint cadence differs")
        records: list[dict[str, Any]] = []
        for epoch in range(self.completed_epoch + 1, final_epoch + 1):
            dataset = (
                dataset_factory(epoch)
                if dataset_factory is not None
                else TrainingDJSCCDataset(
                    str(self.config.resolved["dataset"]),
                    int(self.config.resolved["train_seed"]),
                    epoch,
                    repo_root=repo_root,
                )
            )
            self.train_epoch(epoch, dataset)
            records.append(self.save_checkpoint())
        return records

    def _lineage(self) -> dict[str, Any]:
        resolved = self.config.resolved
        return {
            "source_commit": self.source_lineage.source_commit,
            "source_manifest_id": self.source_lineage.source_manifest_id,
            "source_manifest_sha256": self.source_lineage.source_manifest_sha256,
            "config_hash": config_hash(self.config),
            "resolved_config": self.config.to_dict(),
            "dataset": resolved["dataset"],
            "dataset_version": resolved["dataset_version"],
            "split_manifest_hash": get(
                f"datasets.{resolved['dataset']}.manifest_sha256"
            ),
            "execution_profile_id": resolved["execution_profile_id"],
            "architecture": resolved["architecture"],
            "bw_ratio": resolved["bw_ratio"],
            "k": resolved["k"],
            "train_seed": resolved["train_seed"],
            "channel_seed": resolved["channel_seed"],
            "train_snr_db": resolved["train_snr_db"],
            "lambda": resolved["lambda"],
            "recipe_sha256": self.recipe_sha256,
            "predecessor_checkpoint_id": self.predecessor_checkpoint_id,
        }

    def checkpoint_payload(self) -> dict[str, Any]:
        _require(self.completed_epoch >= 0, "cannot checkpoint before a completed W5 epoch")
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact_role": CHECKPOINT_ROLE,
            "eligibility": dict(ELIGIBILITY),
            "lineage": self._lineage(),
            "completed_epoch": self.completed_epoch,
            "next_epoch": self.completed_epoch + 1,
            "global_optimizer_step": self.global_optimizer_step,
            "accumulation_position": 0,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": None if self.scaler is None else self.scaler.state_dict(),
            "rng_state_policy": copy.deepcopy(RNG_STATE_POLICY),
            "training_history": copy.deepcopy(self.training_history),
            "protected_counters": dict(PROTECTED_COUNTERS),
        }

    def save_checkpoint(self) -> dict[str, Any]:
        checkpoint_rel = f"checkpoints/epoch-{self.completed_epoch:04d}.pt"
        checkpoint_path = self.runtime_root / checkpoint_rel
        _require(not checkpoint_path.exists() and not checkpoint_path.is_symlink(), "W5 checkpoint path already exists")
        payload = _portable(self.checkpoint_payload())
        _publish_new_file(checkpoint_path, lambda stream: torch.save(payload, stream))
        checkpoint_id = _sha256_file(checkpoint_path)
        sidecar = {
            "schema_version": SCHEMA_VERSION,
            "artifact_role": SIDECAR_ROLE,
            "eligibility": dict(ELIGIBILITY),
            "checkpoint_path": checkpoint_rel,
            "checkpoint_id": checkpoint_id,
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "completed_epoch": self.completed_epoch,
            "next_epoch": self.completed_epoch + 1,
            "global_optimizer_step": self.global_optimizer_step,
            "config_hash": config_hash(self.config),
            "source_commit": self.source_lineage.source_commit,
            "execution_profile_id": self.config.resolved["execution_profile_id"],
        }
        sidecar_path = checkpoint_path.with_suffix(".json")
        _publish_new_file(sidecar_path, lambda stream: stream.write(_canonical_bytes(sidecar)))
        _atomic_replace_bytes(self.runtime_root / "latest.json", _canonical_bytes(sidecar))
        self.predecessor_checkpoint_id = checkpoint_id
        return sidecar

    @staticmethod
    def _validate_sidecar(sidecar: object) -> dict[str, Any]:
        required = {
            "schema_version", "artifact_role", "eligibility", "checkpoint_path",
            "checkpoint_id", "checkpoint_bytes", "completed_epoch", "next_epoch",
            "global_optimizer_step", "config_hash", "source_commit", "execution_profile_id",
        }
        _require(isinstance(sidecar, Mapping) and set(sidecar) == required, "W5 checkpoint sidecar schema differs")
        value = dict(sidecar)
        _require(value["schema_version"] == SCHEMA_VERSION and value["artifact_role"] == SIDECAR_ROLE, "W5 sidecar role/version differs")
        _require(value["eligibility"] == ELIGIBILITY, "W5 sidecar eligibility differs")
        _require(_full_sha(value["checkpoint_id"]), "W5 sidecar checkpoint ID is invalid")
        _require(_is_int(value["checkpoint_bytes"]) and value["checkpoint_bytes"] > 0, "W5 sidecar byte count is invalid")
        completed = value["completed_epoch"]
        _require(_is_int(completed) and completed >= 0 and value["next_epoch"] == completed + 1, "W5 sidecar epoch differs")
        _require(value["checkpoint_path"] == f"checkpoints/epoch-{completed:04d}.pt", "W5 sidecar path differs")
        return value

    def _validate_history(self, history: object, completed: int, global_step: int) -> list[dict[str, Any]]:
        _require(isinstance(history, list) and len(history) == completed + 1, "W5 history is not an exact epoch prefix")
        records = copy.deepcopy(history)
        _require([record.get("epoch") for record in records] == list(range(completed + 1)), "W5 history epochs differ")
        required = {
            "epoch", "lr", "samples", "microbatches", "optimizer_steps",
            "global_optimizer_step", "total_loss", "cross_entropy",
            "reconstruction_mse", "duration_seconds", "gradient_checks", "trace",
        }
        prior_step = 0
        for record in records:
            _require(isinstance(record, Mapping) and set(record) == required, "W5 history record schema differs")
            _require(record["optimizer_steps"] == record["microbatches"] == len(record["trace"]), "W5 history step arithmetic differs")
            prior_step += record["optimizer_steps"]
            _require(record["global_optimizer_step"] == prior_step, "W5 history global step differs")
            _require(record["samples"] > 0 and all(math.isfinite(float(record[key])) for key in ("lr", "total_loss", "cross_entropy", "reconstruction_mse", "duration_seconds")), "W5 history numeric value differs")
        _require(prior_step == global_step, "W5 history/checkpoint global step differs")
        return records

    def resume(self, pointer_path: Path | None = None) -> dict[str, Any]:
        pointer = self.runtime_root / "latest.json" if pointer_path is None else Path(pointer_path)
        _require(pointer.is_file() and not pointer.is_symlink(), "W5 latest pointer is missing or unsafe")
        try:
            sidecar = self._validate_sidecar(json.loads(pointer.read_bytes()))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise W5Hold("W5 latest pointer is corrupt") from None
        _require(sidecar["config_hash"] == config_hash(self.config), "W5 sidecar config hash differs")
        _require(sidecar["source_commit"] == self.source_lineage.source_commit, "W5 sidecar source commit differs")
        _require(sidecar["execution_profile_id"] == self.config.resolved["execution_profile_id"], "W5 sidecar execution profile differs")
        checkpoint_path = self.runtime_root / sidecar["checkpoint_path"]
        _require(checkpoint_path.is_file() and not checkpoint_path.is_symlink(), "W5 checkpoint is missing or unsafe")
        _require(checkpoint_path.stat().st_size == sidecar["checkpoint_bytes"], "W5 checkpoint byte length differs")
        _require(_sha256_file(checkpoint_path) == sidecar["checkpoint_id"], "W5 checkpoint SHA-256 differs")
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, TypeError, ValueError):
            raise W5Hold("W5 authenticated checkpoint cannot be loaded") from None
        required = {
            "schema_version", "artifact_role", "eligibility", "lineage", "completed_epoch",
            "next_epoch", "global_optimizer_step", "accumulation_position", "model_state",
            "optimizer_state", "scheduler_state", "scaler_state", "rng_state_policy",
            "training_history", "protected_counters",
        }
        _require(isinstance(payload, Mapping) and set(payload) == required, "W5 checkpoint schema differs")
        _require(payload["schema_version"] == SCHEMA_VERSION and payload["artifact_role"] == CHECKPOINT_ROLE, "W5 checkpoint role/version differs")
        _require(payload["eligibility"] == ELIGIBILITY, "W5 checkpoint eligibility differs")
        _require(payload["protected_counters"] == PROTECTED_COUNTERS, "W5 checkpoint protected counters differ")
        _require(payload["rng_state_policy"] == RNG_STATE_POLICY, "W5 checkpoint RNG policy differs")
        completed = payload["completed_epoch"]
        _require(completed == sidecar["completed_epoch"] and payload["next_epoch"] == completed + 1, "W5 checkpoint epoch differs")
        _require(payload["global_optimizer_step"] == sidecar["global_optimizer_step"] and payload["accumulation_position"] == 0, "W5 checkpoint optimizer/accumulation state differs")
        expected_lineage = self._lineage()
        lineage = payload["lineage"]
        _require(isinstance(lineage, Mapping) and set(lineage) == set(expected_lineage), "W5 checkpoint lineage schema differs")
        for key, expected in expected_lineage.items():
            if key == "predecessor_checkpoint_id":
                predecessor = lineage[key]
                _require(predecessor is None or _full_sha(predecessor), "W5 predecessor checkpoint ID is invalid")
            else:
                _require(lineage[key] == expected, f"W5 checkpoint {key} differs")
        temporary_model = build_djscc(self.config, device="cpu")
        temporary_optimizer = self._new_optimizer(temporary_model)
        temporary_scheduler = EpochScheduler()
        try:
            temporary_model.load_state_dict(payload["model_state"], strict=True)
            temporary_optimizer.load_state_dict(payload["optimizer_state"])
            temporary_scheduler.load_state_dict(payload["scheduler_state"])
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise W5Hold(f"W5 checkpoint model/optimizer/scheduler state is invalid: {exc}") from None
        _require(temporary_scheduler.completed_epoch == completed, "W5 checkpoint scheduler epoch differs")
        expected_optimizer = self._new_optimizer(build_djscc(self.config, device="cpu"))
        expected_group = {key: value for key, value in expected_optimizer.param_groups[0].items() if key != "params"}
        expected_group["lr"] = learning_rate_for_epoch(self.config, completed)
        _require(len(temporary_optimizer.param_groups) == 1, "W5 optimizer group count differs")
        candidate_group = {key: value for key, value in temporary_optimizer.param_groups[0].items() if key != "params"}
        _require(candidate_group == expected_group, "W5 optimizer recipe/state differs")
        temporary_scaler = self._new_scaler()
        if temporary_scaler is None:
            _require(payload["scaler_state"] is None, "W5 CPU checkpoint unexpectedly carries scaler state")
        else:
            _require(isinstance(payload["scaler_state"], Mapping), "W5 CUDA checkpoint lacks scaler state")
            try:
                temporary_scaler.load_state_dict(payload["scaler_state"])
            except (KeyError, RuntimeError, TypeError, ValueError):
                raise W5Hold("W5 checkpoint scaler state is invalid") from None
        history = self._validate_history(payload["training_history"], completed, payload["global_optimizer_step"])
        self.model.load_state_dict(temporary_model.state_dict(), strict=True)
        self.optimizer.load_state_dict(temporary_optimizer.state_dict())
        self.scheduler.load_state_dict(temporary_scheduler.state_dict())
        if self.scaler is not None and temporary_scaler is not None:
            self.scaler.load_state_dict(temporary_scaler.state_dict())
        self.completed_epoch = completed
        self.global_optimizer_step = payload["global_optimizer_step"]
        self.training_history = history
        self.predecessor_checkpoint_id = sidecar["checkpoint_id"]
        return sidecar


def deterministic_history(history: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Strip wall-clock fields for exact uninterrupted/resumed comparison."""

    output = copy.deepcopy(list(history))
    for record in output:
        record.pop("duration_seconds", None)
    return output


def model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def state_tree_sha256(value: Any) -> str:
    """Deterministically hash nested optimizer/scaler state including tensors."""

    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            digest.update(b"tensor\0")
            digest.update(str(item.dtype).encode("ascii"))
            digest.update(str(tuple(item.shape)).encode("ascii"))
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda candidate: str(candidate)):
                visit(str(key))
                visit(item[key])
        elif isinstance(item, list | tuple):
            digest.update(b"sequence\0")
            for nested in item:
                visit(nested)
        else:
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(b"\0")
            digest.update(repr(item).encode("utf-8"))
            digest.update(b"\0")

    visit(value)
    return digest.hexdigest()


def default_source_lineage_for_tests() -> W5SourceLineage:
    """Explicit non-production lineage for unit fixtures only."""

    return W5SourceLineage("0" * 40, "w5source-test-fixture", "0" * 64)  # literal-ok: Git SHA-1 and SHA-256 fixture widths
