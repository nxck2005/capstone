"""The single AM-95 randomized-SNR learned-system training run."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
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
from models.djscc import DJSCC, build_djscc
from training.deterministic_core import (
    apply_optimizer_update,
    canonical_bytes,
    canonical_sha256,
    gradient_status,
    optimizer_parameters,
)
from training.djscc_loss import DJSCCObjective
from training.er9 import _fsync_directory, _publish_immutable, _sha256_file
from training.er2_snr import select_training_snr_db


ER2_CHECKPOINT_SCHEMA_VERSION = 1


class ER2RandomizedHold(RuntimeError):
    """An AM-95 randomized ER-2 invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ER2RandomizedHold(message)


def _learning_rate_for_epoch(config: RunConfig, epoch: int) -> float:
    learned = config.parameters["learned_system"]
    total = int(learned["epochs"][config.resolved["dataset"]])
    _require(0 <= epoch < total, "randomized ER-2 epoch is outside the schedule")
    base = float(learned["lr"])
    minimum = float(learned["lr_min"])
    return minimum + (base - minimum) * 0.5 * (  # literal-ok: exact cosine schedule coefficient
        1 + math.cos(math.pi * epoch / max(total - 1, 1))
    )


def _recipe(config: RunConfig) -> dict[str, Any]:
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
    result = {field: learned[field] for field in fields}
    result["augmentation"] = list(result["augmentation"])
    _require(result["loss"] == "CE + lambda * MSE", "randomized ER-2 learned loss differs")
    _require(result["batch_order"] == "keyed_philox_permutation_per_epoch", "randomized ER-2 order differs")
    _require(result["checkpoint_timing"] == "after_completed_epoch_and_before_next_epoch", "randomized ER-2 checkpoint timing differs")
    _require(result["checkpoint_resume_unit"] == "authenticated_completed_epoch", "randomized ER-2 resume unit differs")
    return result


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
        rows.append((components[0] + 1j * components[1]) / math.sqrt(2.0))  # literal-ok: unit complex Gaussian convention
        identities.append(canonical_sha256(identity))
    return torch.as_tensor(np.stack(rows), dtype=torch.complex64, device=device), identities


def snr_assignment_audit(config: RunConfig) -> dict[str, Any]:
    """Digest all deterministic per-sample/epoch AM-95 assignments."""

    dataset = TrainingDJSCCDataset(
        str(config.resolved["dataset"]),
        int(config.resolved["train_seed"]),
        0,  # literal-ok: audit constructs each epoch view independently
    )
    source = getattr(dataset, "_source")
    stable_ids = [str(source.source_sample(index).stable_sample_id) for index in range(len(dataset))]
    epochs = int(config.parameters["learned_system"]["epochs"][config.resolved["dataset"]])
    domain = tuple(int(value) for value in get("channel.train_snr_db_set"))
    counts = {str(value): 0 for value in domain}
    digest = hashlib.sha256()
    per_epoch: list[dict[str, Any]] = []
    for epoch in range(epochs):
        epoch_counts = {str(value): 0 for value in domain}
        for stable_id in stable_ids:
            value = select_training_snr_db(
                str(config.resolved["dataset_version"]),
                str(get(f"datasets.{config.resolved['dataset']}.manifest_sha256")),
                stable_id,
                int(config.resolved["train_seed"]),
                epoch,
            )
            digest.update(f"{epoch}\0{stable_id}\0{value}\n".encode("ascii"))
            counts[str(value)] += 1
            epoch_counts[str(value)] += 1
        per_epoch.append({"epoch": epoch, "counts": epoch_counts})
    return {
        "schema_version": 1,  # literal-ok: AM-95 audit schema
        "artifact_role": "ER2_RANDOMIZED_SNR_ASSIGNMENT_AUDIT",
        "system": "learned_snr_randomised",
        "rng_purpose": "er2_snr_randomised_v1",
        "identity_fields": [
            "dataset_version",
            "split_manifest_hash",
            "stable_sample_id",
            "train_seed",
            "epoch",
        ],
        "config_hash": run_config_hash(config),
        "dataset_version": str(config.resolved["dataset_version"]),
        "split_manifest_hash": str(get(f"datasets.{config.resolved['dataset']}.manifest_sha256")),
        "domain": list(domain),
        "distribution": "discrete_uniform",
        "unit": "per_sample_per_epoch",
        "sample_count": len(stable_ids),
        "epoch_count": epochs,
        "assignment_count": len(stable_ids) * epochs,
        "assignment_digest": digest.hexdigest(),
        "global_counts": counts,
        "per_epoch_counts": per_epoch,
        "batching_independent": True,
        "channel_noise_separately_keyed": True,
        "test_access": 0,
    }


@torch.no_grad()
def evaluate_er2_at_snr(
    model: DJSCC,
    config: RunConfig,
    *,
    snr_db: int,
    device: torch.device | str,
    num_workers: int,
) -> dict[str, Any]:
    """Evaluate one randomized ER-2 checkpoint on validation only."""

    dataset = ValidationDJSCCDataset(str(config.resolved["dataset"]))
    loader = DataLoader(
        dataset,
        batch_size=int(config.resolved["validation_batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=bool(config.parameters["learned_system"]["pin_memory"] and torch.device(device).type == "cuda"),
    )
    model.eval()
    rows: list[dict[str, Any]] = []
    split_hash = str(get(f"datasets.{config.resolved['dataset']}.manifest_sha256"))
    correct = 0
    total = 0
    for inputs, labels, stable_ids in loader:
        ids = [str(value) for value in stable_ids]
        noise_ids = [
            validation_noise_id(
                stable_sample_id=stable_id,
                dataset_version=str(config.resolved["dataset_version"]),
                split_manifest_hash=split_hash,
                channel_seed=int(config.resolved["channel_seed"]),
                channel=str(config.resolved["channel"]),
                ratio=str(config.resolved["bw_ratio"]),
                k=int(config.resolved["k"]),
                snr_db=snr_db,
            )
            for stable_id in ids
        ]
        inputs = inputs.to(device, non_blocking=torch.device(device).type == "cuda")
        labels = labels.to(device, non_blocking=torch.device(device).type == "cuda")
        unit_noise = keyed_complex_noise(
            tuple(noise_ids),
            int(config.resolved["k"]),
            dtype=torch.complex64,
            device=device,
        )
        output = model(inputs, snr_db, unit_noise=unit_noise)
        predictions = output.logits.argmax(dim=1).detach().cpu().numpy()
        truth = labels.detach().cpu().numpy()
        for stable_id, prediction, target in zip(ids, predictions, truth, strict=True):
            outcome = bool(int(prediction) == int(target))
            rows.append({"stable_sample_id": stable_id, "correct": outcome})
            correct += int(outcome)
            total += 1
    _require(total == len(dataset), "randomized ER-2 validation did not cover all rows")
    return {
        "system": "learned_snr_randomised",
        "snr_db": int(snr_db),
        "n_correct": correct,
        "n_total": total,
        "outcomes": rows,
        "test_access": 0,
    }


class ER2RandomizedTrainer:
    """Exactly one fresh randomized learned training run."""

    def __init__(
        self,
        config: RunConfig,
        *,
        device: torch.device | str,
        runtime_root: Path,
        source_binding: Mapping[str, Any],
        campaign_id: str,
        run_id: str,
        num_workers: int | None = None,
    ) -> None:
        if config.resolved.get("system") != "learned_snr_randomised":
            raise ER2RandomizedHold("randomized ER-2 trainer requires learned_snr_randomised")
        if int(config.resolved["train_seed"]) != 0 or int(config.resolved["channel_seed"]) != 0:  # literal-ok: first zipped ER-2 seed cell
            raise ER2RandomizedHold("ER-2 randomized run must use the first zipped 0/0 seed cell")
        if float(config.resolved["lambda"]) != 3.0:  # literal-ok: frozen G-4 lambda
            raise ER2RandomizedHold("ER-2 randomized run lambda is not the frozen 3.0")
        self.config = config
        self.config_hash = run_config_hash(config)
        self.device = torch.device(device)
        self.runtime_root = Path(runtime_root)
        if self.runtime_root.exists() or self.runtime_root.is_symlink():
            raise ER2RandomizedHold(f"randomized ER-2 runtime already exists: {self.runtime_root}")
        self.runtime_root.mkdir(parents=True, exist_ok=False)
        self.source_binding = dict(source_binding)
        self.campaign_id = campaign_id
        self.run_id = run_id
        self.recipe = _recipe(config)
        self.recipe_sha256 = canonical_sha256(self.recipe)
        self.model: DJSCC = build_djscc(config, device=self.device)
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
        self.completed_epoch = -1
        self.global_optimizer_step = 0
        self.best_validation: dict[str, Any] | None = None

    def train_epoch(self, epoch: int) -> dict[str, Any]:
        _require(epoch == self.completed_epoch + 1, "randomized ER-2 epoch is not the exact next epoch")
        dataset = TrainingDJSCCDataset(
            str(self.config.resolved["dataset"]),
            int(self.config.resolved["train_seed"]),
            epoch,
        )
        physical_batch = int(self.config.resolved["physical_batch_size"])
        target_batch = int(self.config.resolved["effective_batch_size"])
        accumulation = int(self.config.resolved["accumulation_factor"])
        _require(physical_batch * accumulation == target_batch, "randomized ER-2 batch arithmetic differs")
        loader = DataLoader(
            dataset,
            batch_size=physical_batch,
            sampler=EpochPermutationSampler(len(dataset), int(self.config.resolved["train_seed"]), epoch),
            shuffle=False,
            num_workers=int(self.recipe["dataloader_workers"]),
            drop_last=False,
            pin_memory=bool(self.recipe["pin_memory"] and self.device.type == "cuda"),
        )
        source = getattr(dataset, "_source")
        base_ids = [str(source.source_sample(index).stable_sample_id) for index in range(len(dataset))]
        expected_order = [
            base_ids[index]
            for index in EpochPermutationSampler(len(dataset), int(self.config.resolved["train_seed"]), epoch)
        ]
        observed_ids: list[str] = []
        observed_noise: list[str] = []
        self.model.train()
        for group in self.optimizer.param_groups:
            group["lr"] = _learning_rate_for_epoch(self.config, epoch)
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
            unit_noise, noise_ids = _training_noise(
                self.config,
                stable_ids,
                epoch,
                snrs,
                device=self.device,
            )
            observed_noise.extend(noise_ids)
            inputs = inputs.to(self.device, non_blocking=self.device.type == "cuda")
            labels = labels.to(self.device, non_blocking=self.device.type == "cuda")
            snr_tensor = torch.as_tensor(snrs, dtype=torch.float32, device=self.device)
            context = (
                torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
                if self.amp_enabled
                else nullcontext()
            )
            with context:
                output = self.model(inputs, snr_tensor, unit_noise=unit_noise)
                loss = self.objective(output, labels, inputs)
            count = int(labels.numel())
            values = {
                "total": float(loss.total.detach().float().item()),
                "cross_entropy": float(loss.cross_entropy.detach().float().item()),
                "reconstruction_mse": float(loss.reconstruction_mse.detach().float().item()),
            }
            _require(all(math.isfinite(value) for value in values.values()), "randomized ER-2 loss is non-finite")
            weighted = loss.total * count
            if self.scaler is None:
                weighted.backward()
            else:
                self.scaler.scale(weighted).backward()
            last_status = gradient_status(optimizer_parameters(self.optimizer))
            if not last_status["finite"]:
                raise ER2RandomizedHold("randomized ER-2 optimizer gradient is non-finite")
            group_samples += count
            samples += count
            microbatches += 1
            weighted_total += values["total"] * count
            weighted_ce += values["cross_entropy"] * count
            weighted_mse += values["reconstruction_mse"] * count
            if group_samples >= target_batch or microbatch + 1 == len(loader):
                update = apply_optimizer_update(self.optimizer, self.scaler, denominator=group_samples)
                opportunities += 1
                if update.applied:
                    optimizer_steps += 1
                    self.global_optimizer_step += 1
                else:
                    scaler_skips += 1
                group_samples = 0
                self.optimizer.zero_grad(set_to_none=True)
        _require(samples == len(dataset), "randomized ER-2 epoch sample count differs")
        _require(observed_ids == expected_order, "randomized ER-2 keyed batch order differs")
        _require(len(set(observed_ids)) == len(observed_ids), "randomized ER-2 stable IDs duplicated")
        _require(opportunities == math.ceil(samples / target_batch), "randomized ER-2 optimizer opportunities differ")
        _require(optimizer_steps + scaler_skips == opportunities, "randomized ER-2 optimizer updates do not reconcile")
        self.completed_epoch = epoch
        return {
            "schema_version": 1,  # literal-ok: randomized ER-2 epoch schema
            "artifact_role": "ER2_RANDOMIZED_TRAINING_EPOCH",
            "campaign_id": self.campaign_id,
            "run_id": self.run_id,
            "epoch": epoch,
            "transmit_dim": None,
            "quantiser_bits": None,
            "samples": samples,
            "stable_id_order_sha256": hashlib.sha256("\n".join(observed_ids).encode("ascii")).hexdigest(),
            "training_noise_id_sha256": hashlib.sha256("\n".join(observed_noise).encode("ascii")).hexdigest(),
            "microbatches": microbatches,
            "optimizer_step_opportunities": opportunities,
            "optimizer_steps": optimizer_steps,
            "grad_scaler_skips": scaler_skips,
            "global_optimizer_step": self.global_optimizer_step,
            "lr": _learning_rate_for_epoch(self.config, epoch),
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
            num_workers=int(self.recipe["dataloader_workers"]),
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
            unit_noise = keyed_complex_noise(
                tuple(noise_ids),
                int(self.config.resolved["k"]),
                dtype=torch.complex64,
                device=self.device,
            )
            inputs = inputs.to(self.device, non_blocking=self.device.type == "cuda")
            labels = labels.to(self.device, non_blocking=self.device.type == "cuda")
            output = self.model(inputs, int(self.config.resolved["train_snr_db"]), unit_noise=unit_noise)
            correct += int((output.logits.argmax(dim=1) == labels).sum().item())
            total += int(labels.numel())
        _require(total == len(dataset), "randomized ER-2 validation denominator differs")
        return {
            "epoch": epoch,
            "snr_db": int(self.config.resolved["train_snr_db"]),
            "path": "real_analog_digital_jscc_validation_channel",
            "n_correct": correct,
            "n_total": total,
            "stable_id_order_sha256": hashlib.sha256("\n".join(ids).encode("ascii")).hexdigest(),
            "test_access": 0,
        }

    def save_checkpoint(self, epoch_record: Mapping[str, Any], validation: Mapping[str, Any]) -> dict[str, Any]:
        epoch = int(epoch_record["epoch"])
        record = {**dict(epoch_record), "validation": dict(validation), "checkpoint_written_after_epoch": True}
        _publish_immutable(
            self.runtime_root / "epochs" / f"epoch-{epoch:04d}.json",
            canonical_bytes(record),
        )
        checkpoint_path = self.runtime_root / "checkpoints" / f"epoch-{epoch:04d}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{checkpoint_path.name}.", suffix=".tmp", dir=checkpoint_path.parent
        )
        temporary = Path(temporary_name)
        payload = {
            "schema_version": ER2_CHECKPOINT_SCHEMA_VERSION,
            "artifact_role": "ER2_RANDOMIZED_CHECKPOINT",
            "campaign_id": self.campaign_id,
            "run_id": self.run_id,
            "source_binding": dict(self.source_binding),
            "config_hash": self.config_hash,
            "recipe": dict(self.recipe),
            "recipe_sha256": self.recipe_sha256,
            "completed_epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": None if self.scaler is None else self.scaler.state_dict(),
            "epoch_record": record,
            "test_access": 0,
        }
        try:
            with os.fdopen(descriptor, "wb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, checkpoint_path, follow_symlinks=False)
            temporary.unlink()
            _fsync_directory(checkpoint_path.parent)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        checkpoint_id = _sha256_file(checkpoint_path)
        sidecar = {
            "schema_version": ER2_CHECKPOINT_SCHEMA_VERSION,
            "artifact_role": "ER2_RANDOMIZED_CHECKPOINT_SIDECAR",
            "campaign_id": self.campaign_id,
            "run_id": self.run_id,
            "epoch": epoch,
            "checkpoint_path": str(checkpoint_path.relative_to(self.runtime_root)),
            "checkpoint_id": checkpoint_id,
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "test_access": 0,
        }
        _publish_immutable(
            checkpoint_path.with_suffix(".sidecar.json"),
            canonical_bytes(sidecar),
        )
        _replace = self.runtime_root / "latest.json"
        pointer = {
            "schema_version": ER2_CHECKPOINT_SCHEMA_VERSION,
            "completed_epoch": epoch,
            "checkpoint_id": checkpoint_id,
            "checkpoint_path": sidecar["checkpoint_path"],
            "test_access": 0,
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{_replace.name}.", suffix=".tmp", dir=_replace.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(canonical_bytes(pointer))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, _replace)
            _fsync_directory(_replace.parent)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return {"epoch": epoch, "checkpoint_path": sidecar["checkpoint_path"], "checkpoint_id": checkpoint_id}

    def run(self) -> dict[str, Any]:
        total_epochs = int(self.config.parameters["learned_system"]["epochs"][self.config.resolved["dataset"]])
        assignment = snr_assignment_audit(self.config)
        _publish_immutable(self.runtime_root / "snr_assignment_audit.json", canonical_bytes(assignment))
        for epoch in range(total_epochs):
            record = self.train_epoch(epoch)
            validation = self.validate_checkpoint_path(epoch)
            checkpoint = self.save_checkpoint(record, validation)
            candidate = {**validation, **checkpoint}
            if self.best_validation is None or int(validation["n_correct"]) > int(self.best_validation["n_correct"]):
                self.best_validation = candidate
        _require(self.best_validation is not None, "randomized ER-2 produced no selected checkpoint")
        selected = {
            "schema_version": 1,  # literal-ok: randomized ER-2 selected schema
            "artifact_role": "ER2_RANDOMIZED_SELECTED_CHECKPOINT",
            "campaign_id": self.campaign_id,
            "run_id": self.run_id,
            "recipe_sha256": self.recipe_sha256,
            "metric": "validation_n_correct",
            "mode": "max",
            "tie_break": "earliest_epoch",
            "selection": dict(self.best_validation),
            "test_access": 0,
        }
        _publish_immutable(self.runtime_root / "selected_checkpoint.json", canonical_bytes(selected))
        completion = {
            "schema_version": 1,  # literal-ok: randomized ER-2 completion schema
            "artifact_role": "ER2_RANDOMIZED_COMPLETION",
            "campaign_id": self.campaign_id,
            "run_id": self.run_id,
            "system": "learned_snr_randomised",
            "epochs": total_epochs,
            "recipe": dict(self.recipe),
            "recipe_sha256": self.recipe_sha256,
            "optimizer_steps": self.global_optimizer_step,
            "training_run_count": 1,
            "selected_checkpoint": selected["selection"],
            "selected_checkpoint_file": "selected_checkpoint.json",
            "snr_assignment_audit": "snr_assignment_audit.json",
            "test_access": 0,
        }
        _publish_immutable(self.runtime_root / "run_completion.json", canonical_bytes(completion))
        return completion


__all__ = [
    "ER2RandomizedHold",
    "ER2RandomizedTrainer",
    "evaluate_er2_at_snr",
    "snr_assignment_audit",
]
