"""Prospective AM-95 randomized ER-2 training on the W9 v4 lifecycle."""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from artifacts.rng import keyed_standard_normal
from channels.awgn import keyed_complex_noise
from config.params import get
from config.run_config import RunConfig, config_hash as run_config_hash
from data.classifier import EpochPermutationSampler
from data.djscc_training import TrainingDJSCCDataset
from data.djscc_validation import ValidationDJSCCDataset, validation_noise_id
from models.djscc import build_djscc
from training.deterministic_core import apply_optimizer_update, canonical_sha256
from training.djscc_loss import DJSCCObjective
from training.er2_snr import select_training_snr_db
from training.w9_v4 import (
    V4TrainingStep,
    W9V4TrainingHold,
    W9V4TrainingLoop,
    W9V4TrainingRuntime,
    v4_training_step,
)


class ER2V4TrainingHold(W9V4TrainingHold):
    """An AM-95 randomized ER-2 v4 invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ER2V4TrainingHold(message)


def er2_v4_recipe(config: RunConfig) -> dict[str, Any]:
    learned = config.parameters["learned_system"]
    fields = (
        "optimizer", "optimizer_implementation", "adam_beta1", "adam_beta2",
        "adam_epsilon", "adam_weight_decay", "adam_amsgrad", "adam_maximize",
        "adam_foreach", "adam_capturable", "adam_differentiable", "adam_fused",
        "lr", "lr_schedule", "lr_min", "lr_warmup_epochs", "scheduler_step_unit",
        "scheduler_epoch_indexing", "amp", "amp_device_type", "amp_dtype",
        "grad_scaler_enabled", "grad_scaler_init_scale", "grad_scaler_growth_factor",
        "grad_scaler_backoff_factor", "grad_scaler_growth_interval", "batch_order",
        "grad_accumulation_allowed", "drop_last", "dataloader_workers", "pin_memory",
        "batch_size_policy", "accumulation_gradient_rule", "final_partial_accumulation",
        "checkpoint_every_epochs", "checkpoint_timing", "checkpoint_resume_unit",
        "corrupt_latest_checkpoint_policy", "incomplete_epoch_policy", "checkpoint_schema_version",
        "loss", "augmentation",
    )
    recipe = {field: learned[field] for field in fields}
    recipe["augmentation"] = list(recipe["augmentation"])
    _require(recipe["loss"] == "CE + lambda * MSE", "ER-2 v4 learned loss differs")
    _require(recipe["batch_order"] == "keyed_philox_permutation_per_epoch", "ER-2 v4 order differs")
    _require(recipe["checkpoint_timing"] == "after_completed_epoch_and_before_next_epoch", "ER-2 v4 checkpoint timing differs")
    _require(recipe["checkpoint_resume_unit"] == "authenticated_completed_epoch", "ER-2 v4 resume unit differs")
    return recipe


def er2_v4_learning_rate(config: RunConfig, epoch: int) -> float:
    learned = config.parameters["learned_system"]
    total = int(learned["epochs"][config.resolved["dataset"]])
    _require(0 <= epoch < total, "ER-2 v4 epoch is outside the configured schedule")
    base = float(learned["lr"])
    minimum = float(learned["lr_min"])
    return minimum + (base - minimum) * 0.5 * (  # literal-ok: exact cosine schedule coefficient
        1 + math.cos(math.pi * epoch / max(total - 1, 1))
    )


def _training_noise(
    config: RunConfig,
    stable_ids: Sequence[str],
    epoch: int,
    snrs: Sequence[int],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    resolved = config.resolved
    split_hash = str(get(f"datasets.{resolved['dataset']}.manifest_sha256"))
    rows: list[np.ndarray] = []
    identities: list[str] = []
    for stable_id, snr in zip(stable_ids, snrs, strict=True):
        identity = {
            "dataset_version": resolved["dataset_version"],
            "split_manifest_hash": split_hash,
            "stable_sample_id": stable_id,
            "train_seed": int(resolved["train_seed"]),
            "channel_seed": int(resolved["channel_seed"]),
            "epoch": epoch,
            "channel": resolved["channel"],
            "bw_ratio": resolved["bw_ratio"],
            "k": int(resolved["k"]),
            "train_snr_db": int(snr),
        }
        components = keyed_standard_normal("training_channel_noise", identity, size=(2, int(resolved["k"])))
        rows.append((components[0] + 1j * components[1]) / math.sqrt(2.0))
        identities.append(canonical_sha256(identity))
    array = np.stack(rows).astype(np.complex64, copy=False)
    return torch.from_numpy(array).to(device), identities


class ER2V4RandomizedTrainer:
    """One future randomized ER-2 run using the same v4 epoch chain."""

    ROLE = "W9_ER2_RANDOMIZED_CANDIDATE"

    def __init__(
        self,
        config: RunConfig,
        *,
        device: torch.device | str,
        runtime_root: Path,
        source_binding: Mapping[str, Any],
        campaign_id: str,
        run_id: str,
        live_authentication: Mapping[str, Any],
        num_workers: int | None = None,
        resume: bool = False,
    ) -> None:
        if config.resolved.get("system") != "learned_snr_randomised":
            raise ER2V4TrainingHold("ER-2 v4 trainer requires learned_snr_randomised")
        _require(int(config.resolved["train_seed"]) == 0 and int(config.resolved["channel_seed"]) == 0, "ER-2 v4 requires the first seed cell")
        _require(float(config.resolved["lambda"]) == 3.0, "ER-2 v4 lambda is not the frozen 3.0")
        _require(isinstance(source_binding, Mapping) and source_binding.get("source_commit"), "ER-2 v4 source binding is missing")
        environment = live_authentication.get("environment") if isinstance(live_authentication, Mapping) else None
        _require(isinstance(environment, Mapping) and environment.get("git_dirty") is False, "ER-2 v4 live Pascal authentication is not clean")
        self.config = config
        self.config_hash = run_config_hash(config)
        self.device = torch.device(device)
        self.runtime_root = Path(runtime_root)
        runtime_exists = self.runtime_root.exists() or self.runtime_root.is_symlink()
        if runtime_exists and not resume:
            raise ER2V4TrainingHold(f"ER-2 v4 fresh runtime already exists: {self.runtime_root}")
        if self.runtime_root.is_symlink():
            raise ER2V4TrainingHold(f"ER-2 v4 runtime is a symlink: {self.runtime_root}")
        self.source_binding = dict(source_binding)
        self.campaign_id = str(campaign_id)
        self.run_id = str(run_id)
        self.recipe = er2_v4_recipe(config)
        self.recipe_sha256 = canonical_sha256(self.recipe)
        self.num_workers = int(self.recipe["dataloader_workers"] if num_workers is None else num_workers)
        _require(self.num_workers >= 0, "ER-2 v4 worker count is negative")

        # These allocations are intentionally below the caller's live Pascal
        # authentication boundary.
        self.model = build_djscc(config, device=self.device)
        self.objective = DJSCCObjective.from_config(config)
        self.optimizer = Adam(
            self.model.parameters(),
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
        self.amp_enabled = bool(self.recipe["amp"] and self.device.type == "cuda")
        self.scaler = (
            torch.amp.GradScaler(
                "cuda",
                init_scale=float(self.recipe["grad_scaler_init_scale"]),
                growth_factor=float(self.recipe["grad_scaler_growth_factor"]),
                backoff_factor=float(self.recipe["grad_scaler_backoff_factor"]),
                growth_interval=int(self.recipe["grad_scaler_growth_interval"]),
                enabled=bool(self.recipe["grad_scaler_enabled"]),
            )
            if self.amp_enabled
            else None
        )
        total_epochs = int(self.config.parameters["learned_system"]["epochs"][self.config.resolved["dataset"]])
        self.identity = {
            "schema_version": 1,
            "campaign_id": self.campaign_id,
            "run_id": self.run_id,
            "source_binding": self.source_binding,
            "config_hash": self.config_hash,
            "system": "learned_snr_randomised",
            "lambda": float(config.resolved["lambda"]),
            "recipe_sha256": self.recipe_sha256,
            "test": "SEALED",
            "test_access": 0,
        }
        self.runtime = W9V4TrainingRuntime(
            self.runtime_root,
            identity=self.identity,
            total_epochs=total_epochs,
            role=self.ROLE,
        )
        self.loop = W9V4TrainingLoop(
            self.runtime,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            total_epochs=total_epochs,
            train_epoch=self._train_and_validate_epoch,
            selection_metric="validation_n_correct",
            tie_break="earliest_epoch",
            resume=resume,
        )
        prior = self.runtime.inspect()
        self.global_optimizer_step = sum(int(item.record.get("applied_optimizer_steps", 0)) for item in prior)

    @staticmethod
    def _id_digest(ids: Sequence[str]) -> str:
        return hashlib.sha256("\n".join(ids).encode("ascii")).hexdigest()

    def train_epoch(self, epoch: int) -> dict[str, Any]:
        _require(epoch == self.loop.completed_epoch + 1, "ER-2 v4 epoch is not the exact next epoch")
        dataset = TrainingDJSCCDataset(str(self.config.resolved["dataset"]), int(self.config.resolved["train_seed"]), epoch)
        physical_batch = int(self.config.resolved["physical_batch_size"])
        target_batch = int(self.config.resolved["effective_batch_size"])
        accumulation = int(self.config.resolved["accumulation_factor"])
        _require(physical_batch * accumulation == target_batch, "ER-2 v4 batch arithmetic differs")
        loader = DataLoader(
            dataset,
            batch_size=physical_batch,
            sampler=EpochPermutationSampler(len(dataset), int(self.config.resolved["train_seed"]), epoch),
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=bool(self.recipe["pin_memory"] and self.device.type == "cuda"),
        )
        source = getattr(dataset, "_source", None)
        _require(source is not None, "ER-2 v4 dataset source is missing")
        base_ids = [str(source.source_sample(index).stable_sample_id) for index in range(len(dataset))]
        expected_order = [
            base_ids[index]
            for index in EpochPermutationSampler(len(dataset), int(self.config.resolved["train_seed"]), epoch)
        ]
        observed_ids: list[str] = []
        observed_noise: list[str] = []
        self.model.train()
        lr = er2_v4_learning_rate(self.config, epoch)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        started = time.monotonic()
        weighted_total = 0.0
        weighted_ce = 0.0
        weighted_mse = 0.0
        samples = 0
        microbatches = 0
        opportunities = 0
        optimizer_steps = 0
        scaler_skips = 0
        group_samples = 0
        self.optimizer.zero_grad(set_to_none=True)
        last_status: dict[str, Any] = {}
        for microbatch, (inputs, labels, ids) in enumerate(loader):
            stable_ids = [str(value) for value in ids]
            _require(len(stable_ids) == int(labels.numel()), "ER-2 v4 IDs and labels differ")
            observed_ids.extend(stable_ids)
            snrs = [
                select_training_snr_db(
                    str(self.config.resolved["dataset_version"]),
                    str(get(f"datasets.{self.config.resolved['dataset']}.manifest_sha256")),
                    stable_id,
                    int(self.config.resolved["train_seed"]),
                    epoch,
                )
                for stable_id in stable_ids
            ]
            unit_noise, noise_ids = _training_noise(self.config, stable_ids, epoch, snrs, device=self.device)
            observed_noise.extend(noise_ids)
            inputs = inputs.to(self.device, non_blocking=self.device.type == "cuda")
            labels = labels.to(self.device, non_blocking=self.device.type == "cuda")
            snr_tensor = torch.as_tensor(snrs, dtype=torch.float32, device=self.device)
            loss_parts: dict[str, torch.Tensor] = {}

            def loss_fn(output: Any) -> torch.Tensor:
                loss = self.objective(output, labels, inputs)
                loss_parts["total"] = loss.total
                loss_parts["cross_entropy"] = loss.cross_entropy
                loss_parts["reconstruction_mse"] = loss.reconstruction_mse
                return loss.total * int(labels.numel())

            step: V4TrainingStep = v4_training_step(
                forward=lambda: self.model(inputs, snr_tensor, unit_noise=unit_noise),
                loss_fn=loss_fn,
                optimizer=self.optimizer,
                scaler=self.scaler,
                device=self.device,
                amp_enabled=self.amp_enabled,
                denominator=None,
            )
            count = int(labels.numel())
            values = {
                key: float(value.detach().float().item())
                for key, value in loss_parts.items()
            }
            _require(set(values) == {"total", "cross_entropy", "reconstruction_mse"}, "ER-2 v4 loss parts are incomplete")
            _require(all(math.isfinite(value) for value in values.values()), "ER-2 v4 loss is non-finite")
            last_status = step.gradient_status
            group_samples += count
            samples += count
            microbatches += 1
            weighted_total += values["total"] * count
            weighted_ce += values["cross_entropy"] * count
            weighted_mse += values["reconstruction_mse"] * count
            if group_samples >= target_batch or microbatch + 1 == len(loader):
                update = apply_optimizer_update(self.optimizer, self.scaler, denominator=group_samples)
                last_status = update.optimizer_gradients
                opportunities += 1
                if update.applied:
                    optimizer_steps += 1
                    self.global_optimizer_step += 1
                else:
                    scaler_skips += 1
                group_samples = 0
                self.optimizer.zero_grad(set_to_none=True)
        _require(samples == len(dataset), "ER-2 v4 epoch sample count differs")
        _require(observed_ids == expected_order, "ER-2 v4 keyed batch order differs")
        _require(len(set(observed_ids)) == len(observed_ids), "ER-2 v4 stable IDs duplicated")
        _require(opportunities == math.ceil(samples / target_batch), "ER-2 v4 optimizer opportunities differ")
        _require(optimizer_steps + scaler_skips == opportunities, "ER-2 v4 optimizer updates do not reconcile")
        return {
            "epoch": epoch,
            "system": "learned_snr_randomised",
            "samples": samples,
            "stable_id_order_sha256": self._id_digest(observed_ids),
            "training_noise_id_sha256": self._id_digest(observed_noise),
            "microbatches": microbatches,
            "optimizer_opportunities": opportunities,
            "applied_optimizer_steps": optimizer_steps,
            "grad_scaler_skips": scaler_skips,
            "global_optimizer_step": self.global_optimizer_step,
            "lr": lr,
            "loss": weighted_total / samples,
            "cross_entropy": weighted_ce / samples,
            "reconstruction_mse": weighted_mse / samples,
            "lambda": float(self.config.resolved["lambda"]),
            "loss_contract": "configured_learned_ce_plus_lambda_mse",
            "duration_seconds": time.monotonic() - started,
            "gradient_checks": {
                "optimizer_parameter_count": int(last_status["parameter_count"]),
                "optimizer_gradient_count": int(last_status["gradient_count"]),
                "all_optimizer_gradients_finite": bool(last_status["finite"]),
            },
            "snr_assignment_rule": "AM-95_one_keyed_selection_per_sample_per_epoch_reused_for_forward",
            "test_access": 0,
        }

    @torch.no_grad()
    def validate_checkpoint_path(self, epoch: int) -> dict[str, Any]:
        dataset = ValidationDJSCCDataset(str(self.config.resolved["dataset"]))
        loader = DataLoader(
            dataset,
            batch_size=int(self.config.resolved["validation_batch_size"]),
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=bool(self.recipe["pin_memory"] and self.device.type == "cuda"),
        )
        self.model.eval()
        correct = 0
        total = 0
        ids: list[str] = []
        for inputs, labels, stable_ids in loader:
            stable = [str(value) for value in stable_ids]
            ids.extend(stable)
            noise_ids = [
                validation_noise_id(
                    stable_sample_id=stable_id,
                    dataset_version=str(self.config.resolved["dataset_version"]),
                    split_manifest_hash=str(get(f"datasets.{self.config.resolved['dataset']}.manifest_sha256")),
                    channel_seed=int(self.config.resolved["channel_seed"]),
                    channel=str(self.config.resolved["channel"]),
                    ratio=str(self.config.resolved["bw_ratio"]),
                    k=int(self.config.resolved["k"]),
                    snr_db=int(self.config.resolved["train_snr_db"]),
                )
                for stable_id in stable
            ]
            unit_noise = keyed_complex_noise(tuple(noise_ids), int(self.config.resolved["k"]), dtype=torch.complex64, device=self.device)
            inputs = inputs.to(self.device, non_blocking=self.device.type == "cuda")
            labels = labels.to(self.device, non_blocking=self.device.type == "cuda")
            context = (
                torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=True)
                if self.amp_enabled
                else nullcontext()
            )
            with context:
                output = self.model(inputs, int(self.config.resolved["train_snr_db"]), unit_noise=unit_noise)
            correct += int((output.logits.argmax(dim=1) == labels).sum().item())
            total += int(labels.numel())
        _require(total == len(dataset), "ER-2 v4 validation denominator differs")
        return {
            "epoch": epoch,
            "snr_db": int(self.config.resolved["train_snr_db"]),
            "path": "real_analog_digital_jscc_validation_channel",
            "n_correct": correct,
            "n_total": total,
            "stable_id_order_sha256": self._id_digest(ids),
            "test_access": 0,
        }

    def _train_and_validate_epoch(self, epoch: int) -> dict[str, Any]:
        record = self.train_epoch(epoch)
        validation = self.validate_checkpoint_path(epoch)
        record["validation"] = validation
        record["validation_n_correct"] = int(validation["n_correct"])
        record["validation_n_total"] = int(validation["n_total"])
        record["validation_stable_id_order_sha256"] = validation["stable_id_order_sha256"]
        return record

    def run(self, *, max_epochs: int | None = None) -> dict[str, Any] | None:
        return self.loop.run(max_epochs=max_epochs)


def expected_er2_v4_identity(
    *,
    config: RunConfig,
    source_binding: Mapping[str, Any],
    campaign_id: str,
    run_id: str,
) -> dict[str, Any]:
    recipe = er2_v4_recipe(config)
    return {
        "schema_version": 1,
        "campaign_id": str(campaign_id),
        "run_id": str(run_id),
        "source_binding": dict(source_binding),
        "config_hash": run_config_hash(config),
        "system": "learned_snr_randomised",
        "lambda": float(config.resolved["lambda"]),
        "recipe_sha256": canonical_sha256(recipe),
        "test": "SEALED",
        "test_access": 0,
    }


__all__ = [
    "ER2V4RandomizedTrainer",
    "ER2V4TrainingHold",
    "er2_v4_learning_rate",
    "er2_v4_recipe",
    "expected_er2_v4_identity",
]
