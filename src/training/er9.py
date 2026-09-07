"""AM-96 ER-9 candidate training and checkpoint selection."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from config.params import get
from config.run_config import RunConfig, config_hash as run_config_hash
from data.classifier import EpochPermutationSampler
from data.djscc_training import TrainingDJSCCDataset
from data.djscc_validation import ValidationDJSCCDataset
from models.er9_digital import ER9DigitalModel, build_er9_model
from training.deterministic_core import (
    apply_optimizer_update,
    canonical_bytes,
    canonical_sha256,
    gradient_status,
    optimizer_parameters,
    state_tree_sha256,
)


ER9_CHECKPOINT_SCHEMA_VERSION = 1
ER9_EPOCH_SCHEMA_VERSION = 1


class ER9TrainingHold(RuntimeError):
    """An ER-9 training or checkpoint invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ER9TrainingHold(message)


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


def _publish_immutable(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ER9TrainingHold(f"ER-9 immutable artifact already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _replace_pointer(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ER9TrainingHold(f"ER-9 pointer is unsafe: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(dict(value)))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def er9_recipe(config: RunConfig) -> dict[str, Any]:
    """Return the learned recipe fields shared by ER-9 under AM-96."""

    learned = config.parameters["learned_system"]
    fields = (
        "optimizer", "optimizer_implementation", "adam_beta1", "adam_beta2",
        "adam_epsilon", "adam_weight_decay", "adam_amsgrad", "adam_maximize",
        "adam_foreach", "adam_capturable", "adam_differentiable", "adam_fused",
        "lr", "lr_schedule", "lr_schedule_equation", "lr_min", "lr_warmup_epochs",
        "scheduler_step_unit", "scheduler_epoch_indexing", "scheduler_resume_state",
        "amp", "amp_device_type", "amp_dtype", "grad_scaler_enabled",
        "grad_scaler_init_scale", "grad_scaler_growth_factor",
        "grad_scaler_backoff_factor", "grad_scaler_growth_interval", "batch_order",
        "grad_accumulation_allowed", "drop_last", "dataloader_workers", "pin_memory",
        "batch_size_policy", "accumulation_gradient_rule", "final_partial_accumulation",
        "scheduler_steps_under_accumulation", "checkpoint_every_epochs", "checkpoint_timing",
        "checkpoint_resume_unit", "corrupt_latest_checkpoint_policy", "incomplete_epoch_policy",
        "checkpoint_schema_version", "loss", "augmentation",
    )
    recipe = {field: learned[field] for field in fields}
    recipe["augmentation"] = list(recipe["augmentation"])
    recipe["inherited_learned_loss"] = recipe.pop("loss")
    recipe["loss"] = "cross_entropy"
    recipe["lambda"] = "not_applicable"
    required = {
        "optimizer": "adam",
        "optimizer_implementation": "torch.optim.Adam",
        "lr_schedule": "cosine",
        "lr_warmup_epochs": 0,  # literal-ok: inherited no-warmup recipe
        "scheduler_step_unit": "epoch_start",
        "scheduler_epoch_indexing": "zero_based",
        "amp": True,
        "amp_device_type": "cuda",
        "amp_dtype": "float16",
        "grad_scaler_enabled": True,
        "batch_order": "keyed_philox_permutation_per_epoch",
        "grad_accumulation_allowed": True,
        "drop_last": False,
        "checkpoint_timing": "after_completed_epoch_and_before_next_epoch",
        "checkpoint_resume_unit": "authenticated_completed_epoch",
        "corrupt_latest_checkpoint_policy": "hold_no_older_fallback",
        "incomplete_epoch_policy": "replay_from_latest_authenticated_completed_epoch",
        "inherited_learned_loss": "CE + lambda * MSE",
        "loss": "cross_entropy",
        "lambda": "not_applicable",
        "augmentation": ["random_resized_crop", "horizontal_flip"],
    }
    for key, expected in required.items():
        _require(recipe[key] == expected, f"ER-9 inherited recipe differs at {key}")
    return recipe


def learning_rate_for_epoch(config: RunConfig, epoch: int) -> float:
    learned = config.parameters["learned_system"]
    total = int(learned["epochs"][config.resolved["dataset"]])
    _require(0 <= epoch < total, "ER-9 epoch is outside the configured schedule")
    base = float(learned["lr"])
    minimum = float(learned["lr_min"])
    return minimum + (base - minimum) * 0.5 * (  # literal-ok: exact cosine schedule coefficient
        1 + math.cos(math.pi * epoch / max(total - 1, 1))
    )


class ER9Trainer:
    """One fresh ER-9 candidate, with exact epoch-boundary checkpoints."""

    def __init__(
        self,
        config: RunConfig,
        *,
        transmit_dim: int,
        quantiser_bits: int,
        device: torch.device | str,
        runtime_root: Path,
        source_binding: Mapping[str, Any],
        campaign_id: str,
        run_id: str,
        num_workers: int | None = None,
        resume: bool = False,
    ) -> None:
        if config.resolved.get("system") != "er9_digital":
            raise ER9TrainingHold("ER-9 trainer requires the er9_digital system")
        self.config = config
        self.config_hash = run_config_hash(config)
        self.transmit_dim = int(transmit_dim)
        self.quantiser_bits = int(quantiser_bits)
        self.device = torch.device(device)
        self.runtime_root = Path(runtime_root)
        runtime_exists = self.runtime_root.exists() or self.runtime_root.is_symlink()
        if runtime_exists and not resume:
            raise ER9TrainingHold(f"ER-9 fresh run namespace already exists: {self.runtime_root}")
        if runtime_exists and self.runtime_root.is_symlink():
            raise ER9TrainingHold(f"ER-9 runtime namespace is a symlink: {self.runtime_root}")
        self.runtime_root.mkdir(parents=True, exist_ok=resume)
        if not self.runtime_root.is_dir():
            raise ER9TrainingHold(f"ER-9 runtime namespace is not a directory: {self.runtime_root}")
        self.campaign_id = campaign_id
        self.run_id = run_id
        self.source_binding = dict(source_binding)
        profile_binding = self.source_binding.get("execution_profile_binding")
        _require(isinstance(profile_binding, Mapping), "ER-9 execution profile binding is missing")
        _require(
            profile_binding.get("config_hash") == self.config_hash,
            "ER-9 execution profile binding config differs",
        )
        self.recipe = er9_recipe(config)
        self.recipe_sha256 = canonical_sha256(self.recipe)
        self.num_workers = int(
            self.recipe["dataloader_workers"] if num_workers is None else num_workers
        )
        _require(self.num_workers >= 0, "ER-9 worker count must be non-negative")
        self.model = build_er9_model(
            config,
            transmit_dim=self.transmit_dim,
            quantiser_bits=self.quantiser_bits,
            device=self.device,
        )
        self.optimizer = self._new_optimizer()
        self.amp_enabled = bool(self.recipe["amp"] and self.device.type == "cuda")
        self.scaler = self._new_scaler()
        self.completed_epoch = -1
        self.global_optimizer_step = 0
        self.predecessor_checkpoint_id: str | None = None
        self.initial_model_state_sha256 = state_tree_sha256(self.model.state_dict())
        self.best_validation: dict[str, Any] | None = None
        if resume:
            self.resume()

    def _new_optimizer(self) -> Adam:
        return Adam(
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

    @staticmethod
    def _id_digest(ids: Sequence[str]) -> str:
        return hashlib.sha256("\n".join(ids).encode("ascii")).hexdigest()

    def _expected_ids(self, dataset: Dataset[tuple[torch.Tensor, int, str]], epoch: int) -> list[str]:
        source = getattr(dataset, "_source", None)
        source_sample = getattr(source, "source_sample", None)
        _require(callable(source_sample), "ER-9 dataset lacks source-bound stable IDs")
        base = [str(source_sample(index).stable_sample_id) for index in range(len(dataset))]
        order = tuple(EpochPermutationSampler(len(dataset), int(self.config.resolved["train_seed"]), epoch))
        return [base[index] for index in order]

    def train_epoch(self, epoch: int) -> dict[str, Any]:
        _require(epoch == self.completed_epoch + 1, "ER-9 epoch is not the exact next epoch")
        dataset = TrainingDJSCCDataset(
            str(self.config.resolved["dataset"]),
            int(self.config.resolved["train_seed"]),
            epoch,
        )
        target_batch = int(self.config.resolved["effective_batch_size"])
        physical_batch = int(self.config.resolved["physical_batch_size"])
        accumulation = int(self.config.resolved["accumulation_factor"])
        _require(physical_batch * accumulation == target_batch, "ER-9 batch arithmetic differs")
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
        expected_ids = self._expected_ids(dataset, epoch)
        observed_ids: list[str] = []
        self.model.train()
        lr = self._set_lr(epoch)
        started = time.monotonic()
        weighted_loss = 0.0
        weighted_ce = 0.0
        samples = 0
        microbatches = 0
        opportunities = 0
        optimizer_steps = 0
        scaler_skips = 0
        group_samples = 0
        group_microbatches = 0
        optimizer_status: dict[str, Any] = {}
        optimizer_parameter_count: int | None = None
        gradient_count_min: int | None = None
        gradient_count_max: int | None = None
        self.optimizer.zero_grad(set_to_none=True)

        for microbatch, (inputs, labels, stable_ids) in enumerate(loader):
            ids = [str(value) for value in stable_ids]
            _require(len(ids) == int(labels.numel()), "ER-9 batch IDs and labels differ")
            observed_ids.extend(ids)
            inputs = inputs.to(self.device, non_blocking=self.device.type == "cuda")
            labels = labels.to(self.device, non_blocking=self.device.type == "cuda")
            context = (
                torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
                if self.amp_enabled
                else nullcontext()
            )
            with context:
                output = self.model(inputs)
                cross_entropy = F.cross_entropy(output.logits, labels)
            loss_value = float(cross_entropy.detach().float().item())
            _require(math.isfinite(loss_value), "ER-9 loss is non-finite")
            count = int(labels.numel())
            weighted = cross_entropy * count
            if self.scaler is None:
                weighted.backward()
            else:
                self.scaler.scale(weighted).backward()
            status = gradient_status(optimizer_parameters(self.optimizer))
            optimizer_status = status
            optimizer_parameter_count = int(status["parameter_count"])
            gradient_count = int(status["gradient_count"])
            gradient_count_min = gradient_count if gradient_count_min is None else min(gradient_count_min, gradient_count)
            gradient_count_max = gradient_count if gradient_count_max is None else max(gradient_count_max, gradient_count)
            group_samples += count
            group_microbatches += 1
            samples += count
            microbatches += 1
            weighted_loss += loss_value * count
            weighted_ce += loss_value * count
            group_complete = group_samples >= target_batch
            last_batch = microbatch + 1 == len(loader)
            if group_complete or last_batch:
                update = apply_optimizer_update(
                    self.optimizer,
                    self.scaler,
                    denominator=group_samples,
                )
                opportunities += 1
                if update.applied:
                    optimizer_steps += 1
                    self.global_optimizer_step += 1
                else:
                    scaler_skips += 1
                group_samples = 0
                group_microbatches = 0
                self.optimizer.zero_grad(set_to_none=True)

        _require(samples == len(dataset), "ER-9 epoch did not process all train samples")
        _require(observed_ids == expected_ids, "ER-9 keyed batch order differs")
        _require(len(set(observed_ids)) == len(observed_ids), "ER-9 train stable IDs duplicated")
        expected_opportunities = math.ceil(samples / target_batch)
        _require(opportunities == expected_opportunities, "ER-9 optimizer opportunity count differs")
        _require(optimizer_steps + scaler_skips == opportunities, "ER-9 update opportunities do not reconcile")
        self.completed_epoch = epoch
        return {
            "schema_version": ER9_EPOCH_SCHEMA_VERSION,
            "artifact_role": "ER9_TRAINING_EPOCH",
            "campaign_id": self.campaign_id,
            "run_id": self.run_id,
            "epoch": epoch,
            "next_epoch": epoch + 1,
            "transmit_dim": self.transmit_dim,
            "quantiser_bits": self.quantiser_bits,
            "samples": samples,
            "stable_id_count": len(observed_ids),
            "stable_id_order_sha256": self._id_digest(observed_ids),
            "stable_id_set_sha256": self._id_digest(sorted(observed_ids)),
            "microbatches": microbatches,
            "optimizer_step_opportunities": opportunities,
            "optimizer_steps": optimizer_steps,
            "grad_scaler_skips": scaler_skips,
            "global_optimizer_step": self.global_optimizer_step,
            "lr": lr,
            "loss": weighted_loss / samples,
            "cross_entropy": weighted_ce / samples,
            "reconstruction_mse": None,
            "lambda": None,
            "loss_contract": "cross_entropy_only_no_reconstruction_no_lambda",
            "duration_seconds": time.monotonic() - started,
            "gradient_checks": {
                "optimizer_parameter_count": optimizer_parameter_count,
                "optimizer_gradient_count_min": gradient_count_min,
                "optimizer_gradient_count_max": gradient_count_max,
                "last_optimizer": optimizer_status,
            },
            "test_access": 0,
        }

    @torch.no_grad()
    def validate_training_path(self, epoch: int) -> dict[str, Any]:
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
        stable_ids: list[str] = []
        for inputs, labels, ids in loader:
            stable_ids.extend(str(value) for value in ids)
            inputs = inputs.to(self.device, non_blocking=self.device.type == "cuda")
            labels = labels.to(self.device, non_blocking=self.device.type == "cuda")
            logits = self.model(inputs).logits
            predictions = logits.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
        _require(total == len(dataset), "ER-9 validation denominator differs")
        return {
            "epoch": epoch,
            "snr_db": int(self.config.resolved["train_snr_db"]),
            "path": "training_ste_quantise_dequantise_without_digital_channel",
            "n_correct": correct,
            "n_total": total,
            "stable_id_order_sha256": self._id_digest(stable_ids),
            "test_access": 0,
        }

    def _runtime_path(self, relative: str) -> Path:
        path = (self.runtime_root / relative).resolve()
        _require(self.runtime_root.resolve() in path.parents, "ER-9 checkpoint path escapes its runtime")
        return path

    def _restore_payload(self, payload: Mapping[str, Any], sidecar: Mapping[str, Any]) -> None:
        expected = {
            "artifact_role": "ER9_CANDIDATE_CHECKPOINT",
            "campaign_id": self.campaign_id,
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "transmit_dim": self.transmit_dim,
            "quantiser_bits": self.quantiser_bits,
            "recipe_sha256": self.recipe_sha256,
            "completed_epoch": int(sidecar["completed_epoch"]),
            "next_epoch": int(sidecar["next_epoch"]),
            "global_optimizer_step": int(sidecar["global_optimizer_step"]),
            "predecessor_checkpoint_id": sidecar["predecessor_checkpoint_id"],
            "initial_model_state_sha256": self.initial_model_state_sha256,
            "test_access": 0,
        }
        for key, expected_value in expected.items():
            _require(payload.get(key) == expected_value, f"ER-9 checkpoint {key} differs on resume")
        _require(payload.get("checkpoint_id") == "pending", "ER-9 checkpoint identity placeholder differs")
        _require(isinstance(payload.get("model_state"), Mapping), "ER-9 checkpoint model state is missing")
        _require(isinstance(payload.get("optimizer_state"), Mapping), "ER-9 checkpoint optimizer state is missing")
        try:
            self.model.load_state_dict(payload["model_state"], strict=True)
            self.optimizer.load_state_dict(payload["optimizer_state"])
            if self.scaler is None:
                _require(payload.get("scaler_state") is None, "ER-9 CPU resume unexpectedly carries scaler state")
            else:
                _require(isinstance(payload.get("scaler_state"), Mapping), "ER-9 CUDA resume lacks scaler state")
                self.scaler.load_state_dict(payload["scaler_state"])
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise ER9TrainingHold(f"ER-9 checkpoint state cannot be restored: {exc}") from None
        self.completed_epoch = int(sidecar["completed_epoch"])
        self.global_optimizer_step = int(sidecar["global_optimizer_step"])
        self.predecessor_checkpoint_id = str(sidecar["checkpoint_id"])

    def resume(self) -> dict[str, Any]:
        """Restore the latest authenticated completed epoch for this run."""

        pointer_path = self.runtime_root / "latest.json"
        _require(pointer_path.is_file() and not pointer_path.is_symlink(), "ER-9 latest checkpoint pointer is missing or unsafe")
        try:
            pointer = json.loads(pointer_path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise ER9TrainingHold("ER-9 latest checkpoint pointer is corrupt") from None
        _require(
            isinstance(pointer, Mapping)
            and pointer.get("schema_version") == ER9_CHECKPOINT_SCHEMA_VERSION
            and pointer.get("test_access") == 0,
            "ER-9 latest checkpoint pointer schema differs",
        )
        sidecar_path = self._runtime_path(str(pointer["sidecar_path"]))
        _require(sidecar_path.is_file() and not sidecar_path.is_symlink(), "ER-9 latest sidecar is missing or unsafe")
        try:
            sidecar = json.loads(sidecar_path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise ER9TrainingHold("ER-9 latest sidecar is corrupt") from None
        _require(isinstance(sidecar, Mapping), "ER-9 latest sidecar is not a mapping")
        _require(
            pointer.get("sidecar_path") == str(sidecar_path.relative_to(self.runtime_root))
            and pointer.get("checkpoint_id") == sidecar.get("checkpoint_id")
            and pointer.get("completed_epoch") == sidecar.get("completed_epoch"),
            "ER-9 latest pointer/sidecar binding differs",
        )
        _require(sidecar.get("artifact_role") == "ER9_CANDIDATE_CHECKPOINT_SIDECAR", "ER-9 sidecar role differs")
        for key, expected_value in {
            "campaign_id": self.campaign_id,
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "recipe_sha256": self.recipe_sha256,
            "transmit_dim": self.transmit_dim,
            "quantiser_bits": self.quantiser_bits,
            "test_access": 0,
        }.items():
            _require(sidecar.get(key) == expected_value, f"ER-9 sidecar {key} differs")
        checkpoint_path = self._runtime_path(str(sidecar["checkpoint_path"]))
        _require(checkpoint_path.is_file() and not checkpoint_path.is_symlink(), "ER-9 latest checkpoint is missing or unsafe")
        _require(_sha256_file(checkpoint_path) == sidecar["checkpoint_id"], "ER-9 latest checkpoint hash differs")
        record_path = self._runtime_path(str(sidecar["epoch_record_path"]))
        _require(record_path.is_file() and not record_path.is_symlink(), "ER-9 latest epoch record is missing or unsafe")
        _require(_sha256_file(record_path) == sidecar["epoch_record_sha256"], "ER-9 latest epoch record hash differs")
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            record = json.loads(record_path.read_bytes())
        except (OSError, RuntimeError, TypeError, ValueError, EOFError, UnicodeDecodeError, json.JSONDecodeError):
            raise ER9TrainingHold("ER-9 latest checkpoint or epoch record is corrupt") from None
        _require(isinstance(payload, Mapping) and isinstance(record, Mapping), "ER-9 latest state is not a mapping")
        _require(record.get("epoch") == sidecar["completed_epoch"], "ER-9 latest epoch differs")
        _require(record.get("next_epoch") == sidecar["next_epoch"], "ER-9 latest next epoch differs")
        _require(record.get("global_optimizer_step") == sidecar["global_optimizer_step"], "ER-9 latest optimizer step differs")
        _require(record.get("transmit_dim") == self.transmit_dim and record.get("quantiser_bits") == self.quantiser_bits, "ER-9 latest candidate differs")
        _require(isinstance(record.get("validation"), Mapping), "ER-9 latest validation record is missing")
        self._restore_payload(payload, sidecar)
        best: dict[str, Any] | None = None
        for epoch in range(self.completed_epoch + 1):
            epoch_sidecar_path = self.runtime_root / "checkpoints" / f"epoch-{epoch:04d}.sidecar.json"
            epoch_record_path = self.runtime_root / "epochs" / f"epoch-{epoch:04d}.json"
            _require(epoch_sidecar_path.is_file() and epoch_record_path.is_file(), "ER-9 checkpoint prefix is incomplete")
            epoch_sidecar = json.loads(epoch_sidecar_path.read_bytes())
            epoch_record = json.loads(epoch_record_path.read_bytes())
            validation = dict(epoch_record["validation"])
            candidate = {
                **validation,
                "checkpoint_path": epoch_sidecar["checkpoint_path"],
                "checkpoint_id": epoch_sidecar["checkpoint_id"],
                "checkpoint_bytes": epoch_sidecar["checkpoint_bytes"],
                "sidecar_path": str(epoch_sidecar_path.relative_to(self.runtime_root)),
            }
            if best is None or int(candidate["n_correct"]) > int(best["n_correct"]):
                best = candidate
        self.best_validation = best
        _require(self.best_validation is not None, "ER-9 resume has no validation history")
        return dict(pointer)

    def _checkpoint_payload(self, epoch_record: Mapping[str, Any], checkpoint_id: str) -> dict[str, Any]:
        return {
            "schema_version": ER9_CHECKPOINT_SCHEMA_VERSION,
            "artifact_role": "ER9_CANDIDATE_CHECKPOINT",
            "campaign_id": self.campaign_id,
            "run_id": self.run_id,
            "source_binding": dict(self.source_binding),
            "config_hash": self.config_hash,
            "resolved_config": self.config.to_dict(),
            "transmit_dim": self.transmit_dim,
            "quantiser_bits": self.quantiser_bits,
            "completed_epoch": self.completed_epoch,
            "next_epoch": self.completed_epoch + 1,
            "global_optimizer_step": self.global_optimizer_step,
            "initial_model_state_sha256": self.initial_model_state_sha256,
            "recipe": dict(self.recipe),
            "recipe_sha256": self.recipe_sha256,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": None if self.scaler is None else self.scaler.state_dict(),
            "checkpoint_id": checkpoint_id,
            "predecessor_checkpoint_id": self.predecessor_checkpoint_id,
            "epoch_record": dict(epoch_record),
            "test_access": 0,
        }

    def save_checkpoint(self, epoch_record: Mapping[str, Any], validation: Mapping[str, Any]) -> dict[str, Any]:
        epoch = int(epoch_record["epoch"])
        epoch_path = self.runtime_root / "epochs" / f"epoch-{epoch:04d}.json"
        record = dict(epoch_record)
        record["validation"] = dict(validation)
        record["checkpoint_written_after_epoch"] = True
        _publish_immutable(epoch_path, canonical_bytes(record))
        checkpoint_path = self.runtime_root / "checkpoints" / f"epoch-{epoch:04d}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_path.exists() or checkpoint_path.is_symlink():
            raise ER9TrainingHold(f"ER-9 checkpoint already exists: {checkpoint_path}")
        payload = self._checkpoint_payload(record, "pending")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{checkpoint_path.name}.", suffix=".tmp", dir=checkpoint_path.parent
        )
        temporary = Path(temporary_name)
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
            "schema_version": ER9_CHECKPOINT_SCHEMA_VERSION,
            "artifact_role": "ER9_CANDIDATE_CHECKPOINT_SIDECAR",
            "campaign_id": self.campaign_id,
            "run_id": self.run_id,
            "epoch": epoch,
            "completed_epoch": epoch,
            "next_epoch": epoch + 1,
            "config_hash": self.config_hash,
            "recipe_sha256": self.recipe_sha256,
            "transmit_dim": self.transmit_dim,
            "quantiser_bits": self.quantiser_bits,
            "global_optimizer_step": self.global_optimizer_step,
            "predecessor_checkpoint_id": self.predecessor_checkpoint_id,
            "checkpoint_path": str(checkpoint_path.relative_to(self.runtime_root)),
            "checkpoint_id": checkpoint_id,
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "epoch_record_path": str(epoch_path.relative_to(self.runtime_root)),
            "epoch_record_sha256": _sha256_file(epoch_path),
            "test_access": 0,
        }
        sidecar_path = checkpoint_path.with_suffix(".sidecar.json")
        _publish_immutable(sidecar_path, sidecar)
        _replace_pointer(
            self.runtime_root / "latest.json",
            {
                "schema_version": ER9_CHECKPOINT_SCHEMA_VERSION,
                "completed_epoch": epoch,
                "checkpoint_id": checkpoint_id,
                "checkpoint_path": sidecar["checkpoint_path"],
                "sidecar_path": str(sidecar_path.relative_to(self.runtime_root)),
                "predecessor_checkpoint_id": sidecar["predecessor_checkpoint_id"],
                "test_access": 0,
            },
        )
        self.predecessor_checkpoint_id = checkpoint_id
        return {
            "epoch": epoch,
            "checkpoint_path": sidecar["checkpoint_path"],
            "checkpoint_id": checkpoint_id,
            "checkpoint_bytes": sidecar["checkpoint_bytes"],
            "sidecar_path": sidecar["sidecar_path"],
        }

    def run(self) -> dict[str, Any]:
        total_epochs = int(self.config.parameters["learned_system"]["epochs"][self.config.resolved["dataset"]])
        for epoch in range(self.completed_epoch + 1, total_epochs):
            record = self.train_epoch(epoch)
            validation = self.validate_training_path(epoch)
            checkpoint = self.save_checkpoint(record, validation)
            combined = {**validation, **checkpoint}
            if self.best_validation is None or int(validation["n_correct"]) > int(self.best_validation["n_correct"]):
                self.best_validation = combined
            elif int(validation["n_correct"]) == int(self.best_validation["n_correct"]):
                # The loop is ascending, so retaining the old record is the
                # explicit earliest-epoch tie rule.
                pass
        _require(self.best_validation is not None, "ER-9 produced no validation checkpoint")
        selected = {
            "schema_version": 1,  # literal-ok: compact selected-checkpoint schema
            "artifact_role": "ER9_SELECTED_CHECKPOINT",
            "campaign_id": self.campaign_id,
            "run_id": self.run_id,
            "transmit_dim": self.transmit_dim,
            "quantiser_bits": self.quantiser_bits,
            "recipe_sha256": self.recipe_sha256,
            "metric": "validation_n_correct",
            "mode": "max",
            "tie_break": "earliest_epoch",
            "selection": dict(self.best_validation),
            "test_access": 0,
        }
        _publish_immutable(
            self.runtime_root / "selected_checkpoint.json",
            canonical_bytes(selected),
        )
        completion = {
            "schema_version": 1,  # literal-ok: compact run-completion schema
            "artifact_role": "ER9_CANDIDATE_COMPLETION",
            "campaign_id": self.campaign_id,
            "run_id": self.run_id,
            "transmit_dim": self.transmit_dim,
            "quantiser_bits": self.quantiser_bits,
            "epochs": total_epochs,
            "recipe": dict(self.recipe),
            "recipe_sha256": self.recipe_sha256,
            "optimizer_steps": self.global_optimizer_step,
            "selected_checkpoint": selected["selection"],
            "selected_checkpoint_file": "selected_checkpoint.json",
            "training_run_count": 1,
            "test_access": 0,
        }
        _publish_immutable(self.runtime_root / "run_completion.json", canonical_bytes(completion))
        return completion


def load_er9_checkpoint(
    config: RunConfig,
    *,
    transmit_dim: int,
    quantiser_bits: int,
    checkpoint_path: Path,
    device: torch.device | str,
) -> ER9DigitalModel:
    """Load one authenticated selected checkpoint for validation/evaluation."""

    if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
        raise ER9TrainingHold(f"ER-9 checkpoint is missing or unsafe: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ER9TrainingHold("ER-9 checkpoint payload is not a mapping")
    if (
        payload.get("artifact_role") != "ER9_CANDIDATE_CHECKPOINT"
        or payload.get("transmit_dim") != transmit_dim
        or payload.get("quantiser_bits") != quantiser_bits
        or payload.get("config_hash") != run_config_hash(config)
        or payload.get("recipe_sha256") != canonical_sha256(er9_recipe(config))
        or not isinstance(payload.get("model_state"), Mapping)
    ):
        raise ER9TrainingHold("ER-9 checkpoint candidate differs")
    model = build_er9_model(
        config,
        transmit_dim=transmit_dim,
        quantiser_bits=quantiser_bits,
        device=device,
    )
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model


__all__ = [
    "ER9_CHECKPOINT_SCHEMA_VERSION",
    "ER9_EPOCH_SCHEMA_VERSION",
    "ER9Trainer",
    "ER9TrainingHold",
    "er9_recipe",
    "learning_rate_for_epoch",
    "load_er9_checkpoint",
]
