"""Production W7/G-4 pilot trainer with compact authenticated checkpoints.

The historical W5 trainer is left byte-identical.  This module supplies the
W7 policy on top of the small deterministic core: complete-epoch training,
profile-bound eligibility, exact fresh-process resume, validation-facing
checkpoint lineage, and O(number-of-epochs) worker custody.
"""

from __future__ import annotations

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
from config.run_config import RunConfig, config_hash as run_config_hash
from data.classifier import EpochPermutationSampler
from data.djscc_training import TrainingDJSCCDataset
from models.djscc import DJSCC, build_djscc
from runtime.w7_lock import W7CampaignLock
from training.deterministic_core import (
    OptimizerUpdate,
    apply_optimizer_update,
    canonical_bytes,
    canonical_sha256,
    gradient_status,
    module_gradient_status,
    optimizer_parameters,
    state_tree_sha256,
)
from training.djscc_loss import DJSCCObjective
from training.w7_protocol import (
    W7_CHANNEL_SEED,
    W7_DATASET,
    W7_PROFILE_ID,
    W7_TRAIN_SEED,
    W7_TRAINING_SNR_DB,
    eligibility_for_role,
    load_w7_config,
    protocol_config_hash,
    validate_profile_binding,
    validate_w7_config,
)


W7_CHECKPOINT_SCHEMA_VERSION = 1
W7_EPOCH_SCHEMA_VERSION = 1
W7_CHECKPOINT_ROLE = "W7_G4_SCIENTIFIC_PILOT_CHECKPOINT"
W7_SIDECAR_ROLE = "W7_G4_SCIENTIFIC_PILOT_CHECKPOINT_SIDECAR"
W7_EPOCH_ROLE = "W7_G4_EPOCH_RECORD"
W7_PROFILE_EPOCH_ROLE = "NON_SCIENTIFIC_PROFILE_EPOCH_RECORD"
W7_PROFILE_CHECKPOINT_ROLE = "NON_SCIENTIFIC_PROFILE_CHECKPOINT"
W7_RNG_STATE_POLICY = {
    "python_random": "not_consumed",
    "numpy": "keyed_stateless_only",
    "torch_cpu": "not_consumed_after_isolated_keyed_initialization",
    "torch_cuda": "not_consumed_by_training_stochasticity",
    "channel": "keyed_training_channel_noise_per_sample_epoch",
    "augmentation": "keyed_per_stable_sample_train_seed_epoch",
    "batch_order": "keyed_philox_per_train_seed_epoch",
    "validation_channel_noise": "keyed_per_image_snr_seed_ratio_not_lambda",
    "serialized_sequential_rng_states": [],
}
W7_PROTECTED_COUNTERS = {
    "w7_scientific_optimizer_steps": 0,
    "w7_lambda_pilot_runs": 0,
    "w7_candidate_results": 0,
    "g4_adjudications": 0,
    "w8_final_training_runs": 0,
    "learned_test_inference": 0,
    "test_model_facing_access": 0,
}
PROFILE_PROTECTED_COUNTERS = {
    **W7_PROTECTED_COUNTERS,
    "profile_optimizer_steps": 0,
    "profile_epochs": 0,
}


class W7Hold(RuntimeError):
    """A W7 training, lineage, checkpoint or eligibility violation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise W7Hold(message)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _full_sha(value: object, width: int) -> bool:
    return isinstance(value, str) and len(value) == width and all(
        character in "0123456789abcdef" for character in value
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):  # literal-ok: bounded hashing block size
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_new_bytes(path: Path, raw: bytes) -> None:
    """Publish a new immutable file without replacing a predecessor."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise W7Hold(f"refusing symlink at immutable W7 path {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            raise W7Hold(f"immutable W7 path already exists: {path}") from None
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _replace_pointer(path: Path, raw: bytes) -> None:
    """Atomically publish the mutable latest pointer, never through a symlink."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise W7Hold(f"latest W7 pointer is unsafe: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
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


def keyed_training_complex_noise(
    identities: Sequence[Mapping[str, Any]],
    k: int,
    *,
    dtype: torch.dtype = torch.complex64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Pure per-sample/per-epoch complex noise for the AM-91 training path."""

    expected = list(get("artifacts.rng_identity_fields.training_channel_noise"))
    if not identities:
        raise ValueError("training noise requires one or more identities")
    if not _is_int(k) or k <= 0:
        raise ValueError("k must be a positive integer")
    if dtype not in (torch.complex64, torch.complex128):
        raise TypeError("training noise dtype must be complex64 or complex128")
    rows: list[np.ndarray] = []
    for identity in identities:
        if set(identity) != set(expected):
            raise ValueError("training noise identity differs from params")
        components = keyed_standard_normal(
            "training_channel_noise", identity, size=(2, k)
        )
        rows.append((components[0] + 1j * components[1]) / math.sqrt(2.0))  # literal-ok: unit complex Gaussian convention
    return torch.as_tensor(np.stack(rows), dtype=dtype, device=device)


def _learned_recipe(config: RunConfig) -> dict[str, Any]:
    """Copy and authenticate the complete AM-91 recipe for W7."""

    learned = config.parameters["learned_system"]
    fields = (
        "optimizer",
        "optimizer_implementation",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "adam_weight_decay",
        "adam_amsgrad",
        "adam_maximize",
        "adam_foreach",
        "adam_capturable",
        "adam_differentiable",
        "adam_fused",
        "lr",
        "lr_schedule",
        "lr_schedule_equation",
        "lr_min",
        "lr_warmup_epochs",
        "scheduler_step_unit",
        "scheduler_epoch_indexing",
        "scheduler_resume_state",
        "amp",
        "amp_device_type",
        "amp_dtype",
        "grad_scaler_enabled",
        "grad_scaler_init_scale",
        "grad_scaler_growth_factor",
        "grad_scaler_backoff_factor",
        "grad_scaler_growth_interval",
        "batch_order",
        "drop_last",
        "dataloader_workers",
        "pin_memory",
        "batch_size_policy",
        "accumulation_gradient_rule",
        "final_partial_accumulation",
        "scheduler_steps_under_accumulation",
        "checkpoint_every_epochs",
        "checkpoint_timing",
        "checkpoint_resume_unit",
        "corrupt_latest_checkpoint_policy",
        "incomplete_epoch_policy",
        "checkpoint_schema_version",
        "w5_checkpoint_selection",
        "loss",
        "augmentation",
    )
    recipe = {field: learned[field] for field in fields}
    recipe["augmentation"] = list(recipe["augmentation"])
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
        "checkpoint_schema_version": W7_CHECKPOINT_SCHEMA_VERSION,
        "w5_checkpoint_selection": "prohibited_non_scientific_smoke_only",
        "loss": "CE + lambda * MSE",
        "augmentation": ["random_resized_crop", "horizontal_flip"],
    }
    for key, value in expected.items():
        _require(recipe[key] == value, f"W7 AM-91 recipe differs at {key}")
    return recipe


def learning_rate_for_epoch(config: RunConfig, epoch: int) -> float:
    learned = config.parameters["learned_system"]
    total = int(learned["epochs"][config.resolved["dataset"]])
    _require(_is_int(epoch) and 0 <= epoch < total, "W7 epoch is outside schedule")
    base = float(learned["lr"])
    minimum = float(learned["lr_min"])
    return minimum + (base - minimum) * 0.5 * (  # literal-ok: exact cosine schedule coefficient
        1 + math.cos(math.pi * epoch / max(total - 1, 1))
    )


@dataclass(frozen=True)
class W7SourceLineage:
    source_commit: str
    source_manifest_id: str
    source_manifest_sha256: str
    execution_image: str

    def validate(self) -> None:
        _require(_full_sha(self.source_commit, 40), "W7 source commit must be full SHA-1")  # literal-ok: Git SHA-1 width
        _require(bool(self.source_manifest_id), "W7 source manifest ID is empty")
        _require(_full_sha(self.source_manifest_sha256, 64), "W7 source manifest SHA is invalid")  # literal-ok: SHA-256 width
        _require(bool(self.execution_image), "W7 execution image is empty")


@dataclass(frozen=True)
class W7TrainingPolicy:
    """Explicit role policy; eligibility is never inferred from a filename."""

    role: str
    validation_enabled: bool
    scientific: bool
    protected_counters: Mapping[str, int]

    def validate(self, config: RunConfig) -> None:
        validate_w7_config(config)
        _require(config.resolved["artifact_role"] == self.role, "W7 policy/config role differs")
        _require(
            config.resolved.to_dict() | eligibility_for_role(self.role)
            == config.resolved.to_dict(),
            "W7 policy eligibility is not exact",
        )
        if self.role == "NON_SCIENTIFIC_PROFILE":
            _require(not self.validation_enabled and not self.scientific, "profile policy is scientific")
            _require(float(config.resolved["lambda"]) == 1.0, "profile lambda is not the fixed non-scientific value")  # literal-ok: owner-fixed profile lambda
        elif self.role == W7_CHECKPOINT_ROLE:
            _require(self.validation_enabled and self.scientific, "pilot policy is not scientific")
        else:
            raise W7Hold(f"unsupported W7 policy role {self.role!r}")
        _require(dict(self.protected_counters) in (W7_PROTECTED_COUNTERS, PROFILE_PROTECTED_COUNTERS), "W7 protected counter policy differs")


NON_SCIENTIFIC_PROFILE_POLICY = W7TrainingPolicy(
    role="NON_SCIENTIFIC_PROFILE",
    validation_enabled=False,
    scientific=False,
    protected_counters=PROFILE_PROTECTED_COUNTERS,
)
W7_G4_PILOT_POLICY = W7TrainingPolicy(
    role=W7_CHECKPOINT_ROLE,
    validation_enabled=True,
    scientific=True,
    protected_counters=W7_PROTECTED_COUNTERS,
)


class W7Trainer:
    """Full deterministic W7 trainer; one instance owns one candidate root."""

    def __init__(
        self,
        config: RunConfig,
        *,
        device: torch.device | str,
        runtime_root: Path,
        source_lineage: W7SourceLineage,
        profile_binding: Mapping[str, Any],
        policy: W7TrainingPolicy | None = None,
        model: DJSCC | None = None,
        num_workers: int | None = None,
    ) -> None:
        _require(isinstance(config, RunConfig), "W7 requires a resolved RunConfig")
        validate_w7_config(config)
        self.config = config
        self.device = torch.device(device)
        self.runtime_root = Path(runtime_root)
        self.source_lineage = source_lineage
        self.source_lineage.validate()
        self.profile_binding = dict(profile_binding)
        validate_profile_binding(self.profile_binding)
        expected_config_hash = run_config_hash(config)
        _require(
            self.profile_binding["config_hash"] == expected_config_hash,
            "W7 profile binding config hash differs",
        )
        _require(
            self.profile_binding["git_commit"] == self.source_lineage.source_commit,
            "W7 profile binding source commit differs",
        )
        self.policy = policy or W7_G4_PILOT_POLICY
        self.policy.validate(config)
        self.recipe = _learned_recipe(config)
        self.recipe_sha256 = canonical_sha256(self.recipe)
        self.config_hash = expected_config_hash
        self.protocol_hash = protocol_config_hash(config)
        self.num_workers = int(self.recipe["dataloader_workers"] if num_workers is None else num_workers)
        _require(self.num_workers >= 0, "W7 dataloader worker count is invalid")
        self.model = model or build_djscc(config, device=self.device)
        self.model.to(self.device)
        self.objective = DJSCCObjective.from_config(config)
        self.optimizer = self._new_optimizer(self.model)
        self.scheduler = _EpochScheduler()
        self.amp_enabled = bool(self.recipe["amp"] and self.device.type == "cuda")
        self.scaler = self._new_scaler()
        self.completed_epoch = -1
        self.global_optimizer_step = 0
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
                "split_manifest_hash": get(f"datasets.{resolved['dataset']}.manifest_sha256"),
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

    def _expected_ids(self, dataset: Dataset[tuple[torch.Tensor, int, str]], epoch: int) -> list[str] | None:
        source_sample = getattr(dataset, "source_sample", None)
        if not callable(source_sample):
            source = getattr(dataset, "_source", None)
            source_sample = getattr(source, "source_sample", None)
        if not callable(source_sample):
            return None
        base = [str(source_sample(index).stable_sample_id) for index in range(len(dataset))]
        order = tuple(EpochPermutationSampler(len(dataset), int(self.config.resolved["train_seed"]), epoch))
        return [base[index] for index in order]

    @staticmethod
    def _id_digest(ids: Sequence[str]) -> str:
        return hashlib.sha256("\n".join(ids).encode("ascii")).hexdigest()

    @staticmethod
    def _id_set_digest(ids: Sequence[str]) -> str:
        return hashlib.sha256("\n".join(sorted(ids)).encode("ascii")).hexdigest()

    def train_epoch(
        self,
        epoch: int,
        dataset: Dataset[tuple[torch.Tensor, int, str]],
    ) -> dict[str, Any]:
        _require(epoch == self.completed_epoch + 1, "W7 epoch is not exact next")
        _require(len(dataset) > 0, "W7 training dataset is empty")
        expected_ids = self._expected_ids(dataset, epoch)
        target_batch = int(get(f"learned_system.batch_size.{self.config.resolved['dataset']}"))
        physical_batch = int(self.config.resolved["physical_batch_size"])
        accumulation_factor = int(self.config.resolved["accumulation_factor"])
        _require(physical_batch * accumulation_factor == target_batch, "W7 batch arithmetic differs")
        sampler = EpochPermutationSampler(len(dataset), int(self.config.resolved["train_seed"]), epoch)
        loader = DataLoader(
            dataset,
            batch_size=physical_batch,
            sampler=sampler,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=bool(self.recipe["pin_memory"] and self.device.type == "cuda"),
        )
        self.model.train()
        lr = self._set_lr(epoch)
        started = time.monotonic()
        total_weighted = 0.0
        ce_weighted = 0.0
        mse_weighted = 0.0
        samples = 0
        microbatches = 0
        optimizer_steps = 0
        scaler_skips = 0
        observed_ids: list[str] = []
        observed_noise_ids: list[str] = []
        all_finite = True
        all_named_finite = True
        named_status_last: dict[str, Any] = {}
        optimizer_status_last: dict[str, Any] = {}
        optimizer_parameter_count: int | None = None
        optimizer_gradient_count_min: int | None = None
        optimizer_gradient_count_max: int | None = None
        group_samples = 0
        group_microbatches = 0
        self.optimizer.zero_grad(set_to_none=True)

        def apply_group() -> OptimizerUpdate:
            nonlocal group_samples, group_microbatches, optimizer_steps, scaler_skips
            _require(group_samples > 0 and group_microbatches > 0, "W7 empty accumulation group")
            update = apply_optimizer_update(
                self.optimizer,
                self.scaler,
                denominator=group_samples,
            )
            if update.applied:
                optimizer_steps += 1
                self.global_optimizer_step += 1
            else:
                scaler_skips += 1
            group_samples = 0
            group_microbatches = 0
            self.optimizer.zero_grad(set_to_none=True)
            return update

        for microbatch, (inputs, labels, stable_ids) in enumerate(loader):
            ids = [str(value) for value in stable_ids]
            if len(ids) != int(labels.numel()):
                raise W7Hold("W7 batch identity/label count differs")
            observed_ids.extend(ids)
            identities = self._training_noise_identities(ids, epoch)
            observed_noise_ids.extend(canonical_sha256(identity) for identity in identities)
            inputs = inputs.to(self.device, non_blocking=self.device.type == "cuda")
            labels = labels.to(self.device, non_blocking=self.device.type == "cuda")
            unit_noise = keyed_training_complex_noise(
                identities,
                int(self.config.resolved["k"]),
                dtype=torch.complex64,
                device=self.device,
            )
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
            values = {
                "total": float(loss.total.detach().float().item()),
                "cross_entropy": float(loss.cross_entropy.detach().float().item()),
                "reconstruction_mse": float(loss.reconstruction_mse.detach().float().item()),
            }
            _require(all(math.isfinite(value) for value in values.values()), "W7 loss is non-finite")
            count = int(labels.numel())
            # The objective is a per-sample mean.  Sum weighted microbatch
            # means, then divide once by the actual (possibly final-partial)
            # group size in apply_optimizer_update.
            weighted_loss = loss.total * count
            if self.scaler is None:
                weighted_loss.backward()
            else:
                self.scaler.scale(weighted_loss).backward()
            named_status = module_gradient_status(self.model)
            optimizer_status = gradient_status(optimizer_parameters(self.optimizer))
            named_status_last = named_status
            optimizer_status_last = optimizer_status
            optimizer_parameter_count = int(optimizer_status["parameter_count"])
            gradient_count = int(optimizer_status["gradient_count"])
            optimizer_gradient_count_min = gradient_count if optimizer_gradient_count_min is None else min(optimizer_gradient_count_min, gradient_count)
            optimizer_gradient_count_max = gradient_count if optimizer_gradient_count_max is None else max(optimizer_gradient_count_max, gradient_count)
            all_finite = all_finite and bool(optimizer_status["finite"])
            all_named_finite = all_named_finite and all(bool(status["finite"]) for status in named_status.values() if bool(status["present"]))
            group_samples += count
            group_microbatches += 1
            samples += count
            microbatches += 1
            total_weighted += values["total"] * count
            ce_weighted += values["cross_entropy"] * count
            mse_weighted += values["reconstruction_mse"] * count
            group_complete = group_samples >= target_batch
            last_batch = microbatch + 1 == len(loader)
            if group_complete or last_batch:
                update = apply_group()
                _require(
                    not update.applied or all(
                        bool(status["finite"])
                        and bool(status["nonzero"])
                        for name, status in named_status.items()
                        if name != "reconstruction_head"
                    ),
                    "W7 encoder/task gradient coverage is invalid",
                )
                if update.applied and float(self.config.resolved["lambda"]) > 0:
                    _require(
                        bool(named_status["reconstruction_head"]["finite"])
                        and bool(named_status["reconstruction_head"]["nonzero"]),
                        "W7 reconstruction gradient coverage is invalid",
                    )

        _require(samples == len(dataset), "W7 epoch did not process complete dataset")
        _require(len(observed_ids) == len(dataset), "W7 train denominator or final batch differs")
        _require(len(set(observed_ids)) == len(observed_ids), "W7 training stable ID duplicated")
        if expected_ids is not None:
            _require(observed_ids == expected_ids, "W7 keyed batch order or train IDs differ")
        self.completed_epoch = epoch
        self.scheduler.completed_epoch = epoch
        record = {
            "schema_version": W7_EPOCH_SCHEMA_VERSION,
            "artifact_role": W7_EPOCH_ROLE if self.policy.scientific else W7_PROFILE_EPOCH_ROLE,
            "eligibility": eligibility_for_role(self.policy.role),
            "lineage": self._lineage(predecessor=self.predecessor_checkpoint_id),
            "epoch": epoch,
            "next_epoch": epoch + 1,
            "samples": samples,
            "expected_samples": len(dataset),
            "stable_id_count": len(observed_ids),
            "stable_id_order_sha256": self._id_digest(observed_ids),
            "stable_id_set_sha256": self._id_set_digest(observed_ids),
            "training_noise_id_count": len(observed_noise_ids),
            "training_noise_id_sha256": self._id_digest(observed_noise_ids),
            "microbatches": microbatches,
            "optimizer_steps": optimizer_steps,
            "grad_scaler_skips": scaler_skips,
            "global_optimizer_step": self.global_optimizer_step,
            "lr": lr,
            "total_loss": total_weighted / samples,
            "cross_entropy": ce_weighted / samples,
            "reconstruction_mse": mse_weighted / samples,
            "duration_seconds": time.monotonic() - started,
            "finite_loss": True,
            "gradient_checks": {
                "optimizer_parameter_count": optimizer_parameter_count,
                "optimizer_gradient_count_min": optimizer_gradient_count_min,
                "optimizer_gradient_count_max": optimizer_gradient_count_max,
                "all_optimizer_gradients_finite": all_finite,
                "all_named_present_gradients_finite": all_named_finite,
                "last_named": named_status_last,
                "last_optimizer": optimizer_status_last,
            },
            "checkpoint_written_after_epoch": True,
        }
        return record

    def _lineage(self, *, predecessor: str | None) -> dict[str, Any]:
        resolved = self.config.resolved
        return {
            "source_commit": self.source_lineage.source_commit,
            "source_manifest_id": self.source_lineage.source_manifest_id,
            "source_manifest_sha256": self.source_lineage.source_manifest_sha256,
            "execution_image": self.source_lineage.execution_image,
            "config_hash": self.config_hash,
            "protocol_config_hash": self.protocol_hash,
            "resolved_config": self.config.to_dict(),
            "execution_profile_id": resolved["execution_profile_id"],
            "gpu_uuid": self.profile_binding["gpu_uuid"],
            "dataset": resolved["dataset"],
            "dataset_version": resolved["dataset_version"],
            "split_manifest_hash": get(f"datasets.{resolved['dataset']}.manifest_sha256"),
            "architecture": resolved["architecture"],
            "bw_ratio": resolved["bw_ratio"],
            "k": resolved["k"],
            "train_seed": resolved["train_seed"],
            "channel_seed": resolved["channel_seed"],
            "train_snr_db": resolved["train_snr_db"],
            "lambda": resolved["lambda"],
            "recipe_sha256": self.recipe_sha256,
            "predecessor_checkpoint_id": predecessor,
        }

    def _checkpoint_payload(self, record_id: str, record_sha256: str) -> dict[str, Any]:
        _require(self.completed_epoch >= 0, "cannot checkpoint before a completed epoch")
        return {
            "schema_version": W7_CHECKPOINT_SCHEMA_VERSION,
            "artifact_role": W7_CHECKPOINT_ROLE if self.policy.scientific else W7_PROFILE_CHECKPOINT_ROLE,
            "eligibility": eligibility_for_role(self.policy.role),
            "lineage": self._lineage(predecessor=self.predecessor_checkpoint_id),
            "execution_profile": dict(self.profile_binding),
            "completed_epoch": self.completed_epoch,
            "next_epoch": self.completed_epoch + 1,
            "global_optimizer_step": self.global_optimizer_step,
            "accumulation_position": 0,
            "model_state": _portable(self.model.state_dict()),
            "optimizer_state": _portable(self.optimizer.state_dict()),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": None if self.scaler is None else _portable(self.scaler.state_dict()),
            "rng_state_policy": dict(W7_RNG_STATE_POLICY),
            "epoch_manifest": {
                "path": f"epochs/epoch-{self.completed_epoch:04d}.json",
                "record_id": record_id,
                "record_sha256": record_sha256,
            },
            "predecessor_checkpoint_id": self.predecessor_checkpoint_id,
            "protected_counters": dict(self.policy.protected_counters),
        }

    def save_checkpoint(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Publish one compact epoch record, checkpoint, sidecar and pointer."""

        _require(record.get("epoch") == self.completed_epoch, "W7 record epoch differs")
        record_value = dict(record)
        record_value.pop("checkpoint_written_after_epoch", None)
        record_id = canonical_sha256(record_value)
        record_path = self.runtime_root / f"epochs/epoch-{self.completed_epoch:04d}.json"
        record_raw = canonical_bytes({**record_value, "record_id": record_id})
        record_sha256 = hashlib.sha256(record_raw).hexdigest()
        _publish_new_bytes(record_path, record_raw)

        checkpoint_rel = f"checkpoints/epoch-{self.completed_epoch:04d}.pt"
        checkpoint_path = self.runtime_root / checkpoint_rel
        payload = self._checkpoint_payload(record_id, record_sha256)
        checkpoint_started = time.monotonic()
        _publish_new_bytes(
            checkpoint_path,
            _torch_save_bytes(payload),
        )
        checkpoint_id = _sha256_file(checkpoint_path)
        sidecar = {
            "schema_version": W7_CHECKPOINT_SCHEMA_VERSION,
            "artifact_role": W7_SIDECAR_ROLE if self.policy.scientific else "NON_SCIENTIFIC_PROFILE_CHECKPOINT_SIDECAR",
            "eligibility": eligibility_for_role(self.policy.role),
            "checkpoint_path": checkpoint_rel,
            "checkpoint_id": checkpoint_id,
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "completed_epoch": self.completed_epoch,
            "next_epoch": self.completed_epoch + 1,
            "global_optimizer_step": self.global_optimizer_step,
            "accumulation_position": 0,
            "config_hash": self.config_hash,
            "protocol_config_hash": self.protocol_hash,
            "source_commit": self.source_lineage.source_commit,
            "source_manifest_id": self.source_lineage.source_manifest_id,
            "source_manifest_sha256": self.source_lineage.source_manifest_sha256,
            "execution_image": self.source_lineage.execution_image,
            "execution_profile_id": self.config.resolved["execution_profile_id"],
            "gpu_uuid": self.profile_binding["gpu_uuid"],
            "dataset": self.config.resolved["dataset"],
            "ratio": self.config.resolved["bw_ratio"],
            "k": self.config.resolved["k"],
            "lambda": self.config.resolved["lambda"],
            "train_seed": self.config.resolved["train_seed"],
            "channel_seed": self.config.resolved["channel_seed"],
            "train_snr_db": self.config.resolved["train_snr_db"],
            "predecessor_checkpoint_id": self.predecessor_checkpoint_id,
            "epoch_record_path": f"epochs/epoch-{self.completed_epoch:04d}.json",
            "epoch_record_id": record_id,
            "epoch_record_sha256": record_sha256,
            "checkpoint_write_seconds": time.monotonic() - checkpoint_started,
        }
        sidecar_path = checkpoint_path.with_suffix(".sidecar.json")
        _publish_new_bytes(sidecar_path, canonical_bytes(sidecar))
        _replace_pointer(self.runtime_root / "latest.json", canonical_bytes(sidecar))
        self.predecessor_checkpoint_id = checkpoint_id
        return sidecar

    def run_epochs(
        self,
        *,
        final_epoch: int,
        dataset_factory: Callable[[int], Dataset[tuple[torch.Tensor, int, str]]] | None = None,
        repo_root: Path | None = None,
    ) -> list[dict[str, Any]]:
        """Run through the inclusive epoch boundary, without a smoke cap."""

        total = int(get(f"learned_system.epochs.{self.config.resolved['dataset']}"))
        _require(_is_int(final_epoch) and self.completed_epoch < final_epoch < total, "W7 final epoch is outside schedule")
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
            record = self.train_epoch(epoch, dataset)
            self.save_checkpoint(record)
            records.append(record)
        return records

    def run_complete(
        self,
        *,
        dataset_factory: Callable[[int], Dataset[tuple[torch.Tensor, int, str]]] | None = None,
        repo_root: Path | None = None,
    ) -> list[dict[str, Any]]:
        total = int(get(f"learned_system.epochs.{self.config.resolved['dataset']}"))
        return self.run_epochs(final_epoch=total - 1, dataset_factory=dataset_factory, repo_root=repo_root)

    def resume(self) -> dict[str, Any]:
        """Authenticate exactly the latest completed epoch; never fall back."""

        pointer_path = self.runtime_root / "latest.json"
        _require(pointer_path.is_file() and not pointer_path.is_symlink(), "W7 latest pointer is missing or unsafe")
        try:
            sidecar = json.loads(pointer_path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise W7Hold("W7 latest pointer is corrupt") from None
        self._validate_sidecar(sidecar)
        completed = int(sidecar["completed_epoch"])
        self._validate_runtime_prefix(completed, sidecar)
        payload = self._load_authenticated_payload(sidecar)
        self._restore_payload(payload, int(sidecar["completed_epoch"]))
        return dict(sidecar)

    def load_checkpoint_epoch(self, epoch: int) -> dict[str, Any]:
        """Load one authenticated completed epoch for final independent evaluation."""

        _require(_is_int(epoch) and epoch >= 0, "W7 evaluation epoch is invalid")
        sidecars: list[dict[str, Any]] = []
        for index in range(epoch + 1):
            path = self.runtime_root / f"checkpoints/epoch-{index:04d}.sidecar.json"
            _require(path.is_file() and not path.is_symlink(), "W7 evaluation checkpoint chain has a gap")
            try:
                sidecar = json.loads(path.read_bytes())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                raise W7Hold("W7 evaluation sidecar is corrupt") from None
            self._validate_sidecar(sidecar)
            sidecars.append(dict(sidecar))
            expected_predecessor = None if index == 0 else sidecars[index - 1]["checkpoint_id"]
            _require(sidecar["predecessor_checkpoint_id"] == expected_predecessor, "W7 evaluation checkpoint predecessor differs")
        sidecar = sidecars[-1]
        payload = self._load_authenticated_payload(sidecar)
        self._restore_payload(payload, int(sidecar["completed_epoch"]))
        return sidecar

    def _load_authenticated_payload(self, sidecar: Mapping[str, Any]) -> Mapping[str, Any]:
        checkpoint_path = self.runtime_root / str(sidecar["checkpoint_path"])
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, TypeError, ValueError, EOFError):
            raise W7Hold("W7 authenticated checkpoint cannot be loaded") from None
        self._validate_payload(payload, sidecar)
        _require(isinstance(payload, Mapping), "W7 checkpoint payload is not a mapping")
        return payload

    def _restore_payload(self, payload: Mapping[str, Any], completed: int) -> None:
        try:
            self.model.load_state_dict(payload["model_state"], strict=True)
            self.optimizer.load_state_dict(payload["optimizer_state"])
            self.scheduler.load_state_dict(payload["scheduler_state"])
            if self.scaler is None:
                _require(payload["scaler_state"] is None, "W7 CPU resume unexpectedly carries scaler state")
            else:
                _require(isinstance(payload["scaler_state"], Mapping), "W7 CUDA resume lacks scaler state")
                self.scaler.load_state_dict(payload["scaler_state"])
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise W7Hold(f"W7 checkpoint state is invalid: {exc}") from None
        _require(self.scheduler.completed_epoch == completed, "W7 scheduler epoch differs")
        _require(self.optimizer.param_groups[0]["lr"] == learning_rate_for_epoch(self.config, completed), "W7 optimizer LR state differs")
        self.completed_epoch = completed
        self.global_optimizer_step = int(payload["global_optimizer_step"])
        self.predecessor_checkpoint_id = payload["predecessor_checkpoint_id"]

    def _validate_sidecar(self, sidecar: object) -> None:
        required = {
            "schema_version", "artifact_role", "eligibility", "checkpoint_path",
            "checkpoint_id", "checkpoint_bytes", "completed_epoch", "next_epoch",
            "global_optimizer_step", "accumulation_position", "config_hash",
            "protocol_config_hash", "source_commit", "source_manifest_id",
            "source_manifest_sha256", "execution_image", "execution_profile_id",
            "gpu_uuid", "dataset", "ratio", "k", "lambda", "train_seed",
            "channel_seed", "train_snr_db", "predecessor_checkpoint_id",
            "epoch_record_path", "epoch_record_id", "epoch_record_sha256",
            "checkpoint_write_seconds",
        }
        _require(isinstance(sidecar, Mapping) and set(sidecar) == required, "W7 checkpoint sidecar schema differs")
        value = dict(sidecar)
        expected_role = W7_SIDECAR_ROLE if self.policy.scientific else "NON_SCIENTIFIC_PROFILE_CHECKPOINT_SIDECAR"
        _require(value["schema_version"] == W7_CHECKPOINT_SCHEMA_VERSION and value["artifact_role"] == expected_role, "W7 sidecar role/version differs")
        _require(value["eligibility"] == eligibility_for_role(self.policy.role), "W7 sidecar eligibility differs")
        _require(_full_sha(value["checkpoint_id"], 64), "W7 checkpoint ID is invalid")  # literal-ok: SHA-256 width
        _require(_is_int(value["checkpoint_bytes"]) and value["checkpoint_bytes"] > 0, "W7 checkpoint byte count is invalid")
        completed = value["completed_epoch"]
        _require(_is_int(completed) and completed >= 0 and value["next_epoch"] == completed + 1, "W7 sidecar epoch differs")
        _require(value["checkpoint_path"] == f"checkpoints/epoch-{completed:04d}.pt", "W7 sidecar path differs")
        _require(value["epoch_record_path"] == f"epochs/epoch-{completed:04d}.json", "W7 epoch record path differs")
        _require(value["config_hash"] == self.config_hash, "W7 sidecar config hash differs")
        _require(value["protocol_config_hash"] == self.protocol_hash, "W7 sidecar protocol hash differs")
        _require(value["source_commit"] == self.source_lineage.source_commit, "W7 sidecar source commit differs")
        _require(value["source_manifest_id"] == self.source_lineage.source_manifest_id, "W7 sidecar source manifest differs")
        _require(value["source_manifest_sha256"] == self.source_lineage.source_manifest_sha256, "W7 sidecar source manifest SHA differs")
        _require(value["execution_image"] == self.source_lineage.execution_image, "W7 sidecar execution image differs")
        _require(value["execution_profile_id"] == self.config.resolved["execution_profile_id"], "W7 sidecar profile differs")
        _require(value["gpu_uuid"] == self.profile_binding["gpu_uuid"], "W7 sidecar GPU differs")
        checkpoint = self.runtime_root / value["checkpoint_path"]
        _require(checkpoint.is_file() and not checkpoint.is_symlink(), "W7 checkpoint is missing or unsafe")
        _require(checkpoint.stat().st_size == value["checkpoint_bytes"], "W7 checkpoint byte length differs")
        _require(_sha256_file(checkpoint) == value["checkpoint_id"], "W7 checkpoint SHA-256 differs")
        record = self.runtime_root / value["epoch_record_path"]
        _require(record.is_file() and not record.is_symlink(), "W7 epoch record is missing or unsafe")
        raw = record.read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == value["epoch_record_sha256"], "W7 epoch record SHA differs")
        record_value = json.loads(raw)
        record_id = record_value.pop("record_id", None)
        _require(record_id == value["epoch_record_id"], "W7 epoch record ID differs")
        _require(canonical_sha256(record_value) == record_id, "W7 epoch record content digest differs")

    def _validate_runtime_prefix(self, completed: int, latest: Mapping[str, Any]) -> None:
        checkpoint_dir = self.runtime_root / "checkpoints"
        epoch_dir = self.runtime_root / "epochs"
        _require(checkpoint_dir.is_dir() and epoch_dir.is_dir(), "W7 runtime directories are missing")
        checkpoints = sorted(checkpoint_dir.glob("epoch-*.pt"))
        sidecars = sorted(checkpoint_dir.glob("epoch-*.sidecar.json"))
        epochs = sorted(epoch_dir.glob("epoch-*.json"))
        expected_names = [f"epoch-{epoch:04d}" for epoch in range(completed + 1)]
        _require([path.stem for path in checkpoints] == expected_names, "W7 checkpoint prefix is not exact")
        _require([path.name.removesuffix(".sidecar.json") for path in sidecars] == expected_names, "W7 sidecar prefix is not exact")
        _require([path.stem for path in epochs] == expected_names, "W7 epoch-record prefix is not exact")
        for path in sidecars:
            self._validate_sidecar(json.loads(path.read_bytes()))
        _require(sidecars[-1].read_bytes() == (self.runtime_root / "latest.json").read_bytes(), "W7 latest pointer is not the newest authenticated sidecar")
        _require(latest["completed_epoch"] == completed, "W7 latest epoch differs")

    def _validate_payload(self, payload: object, sidecar: Mapping[str, Any]) -> None:
        required = {
            "schema_version", "artifact_role", "eligibility", "lineage",
            "execution_profile", "completed_epoch", "next_epoch",
            "global_optimizer_step", "accumulation_position", "model_state",
            "optimizer_state", "scheduler_state", "scaler_state",
            "rng_state_policy", "epoch_manifest", "predecessor_checkpoint_id",
            "protected_counters",
        }
        _require(isinstance(payload, Mapping) and set(payload) == required, "W7 checkpoint schema differs")
        expected_role = W7_CHECKPOINT_ROLE if self.policy.scientific else W7_PROFILE_CHECKPOINT_ROLE
        _require(payload["schema_version"] == W7_CHECKPOINT_SCHEMA_VERSION and payload["artifact_role"] == expected_role, "W7 checkpoint role/version differs")
        _require(payload["eligibility"] == eligibility_for_role(self.policy.role), "W7 checkpoint eligibility differs")
        _require(payload["execution_profile"] == self.profile_binding, "W7 checkpoint execution binding differs")
        _require(payload["rng_state_policy"] == W7_RNG_STATE_POLICY, "W7 checkpoint RNG policy differs")
        _require(payload["protected_counters"] == dict(self.policy.protected_counters), "W7 checkpoint protected counters differ")
        completed = int(sidecar["completed_epoch"])
        _require(payload["completed_epoch"] == completed and payload["next_epoch"] == completed + 1, "W7 checkpoint epoch differs")
        _require(payload["global_optimizer_step"] == sidecar["global_optimizer_step"] and payload["accumulation_position"] == 0, "W7 checkpoint optimizer state differs")
        _require(payload["predecessor_checkpoint_id"] == sidecar["predecessor_checkpoint_id"], "W7 predecessor differs")
        lineage = payload["lineage"]
        expected_lineage = self._lineage(predecessor=sidecar["predecessor_checkpoint_id"])
        _require(isinstance(lineage, Mapping) and dict(lineage) == expected_lineage, "W7 checkpoint lineage differs")
        manifest = payload["epoch_manifest"]
        _require(
            isinstance(manifest, Mapping)
            and manifest.get("path") == sidecar["epoch_record_path"]
            and manifest.get("record_id") == sidecar["epoch_record_id"]
            and manifest.get("record_sha256") == sidecar["epoch_record_sha256"],
            "W7 epoch manifest binding differs",
        )
        expected_optimizer = self._new_optimizer(build_djscc(self.config, device="cpu"))
        expected_group = {key: value for key, value in expected_optimizer.param_groups[0].items() if key != "params"}
        expected_group["lr"] = learning_rate_for_epoch(self.config, completed)
        candidate_optimizer = self._new_optimizer(build_djscc(self.config, device="cpu"))
        try:
            candidate_optimizer.load_state_dict(payload["optimizer_state"])
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise W7Hold(f"W7 optimizer state is invalid: {exc}") from None
        _require(len(candidate_optimizer.param_groups) == 1, "W7 optimizer group count differs")
        candidate_group = {key: value for key, value in candidate_optimizer.param_groups[0].items() if key != "params"}
        _require(candidate_group == expected_group, "W7 optimizer recipe/state differs")
        temporary_scheduler = _EpochScheduler()
        temporary_scheduler.load_state_dict(payload["scheduler_state"])
        _require(temporary_scheduler.completed_epoch == completed, "W7 scheduler state differs")
        if self.scaler is None:
            _require(payload["scaler_state"] is None, "W7 CPU checkpoint unexpectedly carries scaler")


class _EpochScheduler:
    def __init__(self) -> None:
        self.completed_epoch = -1

    def state_dict(self) -> dict[str, int]:
        return {"completed_epoch": self.completed_epoch}

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        _require(set(value) == {"completed_epoch"}, "W7 scheduler schema differs")
        _require(_is_int(value["completed_epoch"]) and value["completed_epoch"] >= -1, "W7 scheduler epoch is invalid")
        self.completed_epoch = int(value["completed_epoch"])


def _torch_save_bytes(value: Any) -> bytes:
    import io

    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


def profile_config(*, physical_batch_size: int = 32, accumulation_factor: int = 1) -> RunConfig:  # literal-ok: owner-frozen first profile rung
    """Resolve the fixed non-scientific real-data profile configuration."""

    return load_w7_config(
        lambda_value=1.0,
        role="NON_SCIENTIFIC_PROFILE",
        physical_batch_size=physical_batch_size,
        accumulation_factor=accumulation_factor,
        validation_batch_size=32,  # literal-ok: owner-frozen validation batch
    )


def future_campaign_lock(
    *,
    campaign_id: str,
    source_commit: str,
    execution_image: str,
    gpu_uuid: str,
) -> W7CampaignLock:
    """Construct the lock the detached five-λ launcher must hold."""

    return W7CampaignLock(
        campaign_id=campaign_id,
        source_commit=source_commit,
        execution_image=execution_image,
        gpu_uuid=gpu_uuid,
    )


def checkpoint_state_digest(trainer: W7Trainer) -> dict[str, str]:
    """Compact test/debug digest; no detailed history is retained in checkpoints."""

    return {
        "model_state_sha256": state_tree_sha256(trainer.model.state_dict()),
        "optimizer_state_sha256": state_tree_sha256(trainer.optimizer.state_dict()),
        "scheduler_state_sha256": state_tree_sha256(trainer.scheduler.state_dict()),
    }
