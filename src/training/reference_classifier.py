"""Deterministic, validation-only reference-classifier training (AM-78)."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer, SGD
from torch.utils.data import DataLoader, Dataset

from config.params import get
from config.run_config import RunConfig, config_hash
from data.classifier import EpochPermutationSampler, TrainingClassifierDataset, ValidationClassifierDataset
from models.reference_classifier import ReferenceClassifier, build_reference_classifier


@dataclass(frozen=True)
class ValidationResult:
    n_correct: int
    n_total: int

    @property
    def top1_accuracy(self) -> float:
        if self.n_total <= 0:
            raise RuntimeError("validation has no samples")
        return self.n_correct / self.n_total


@dataclass
class TrainingState:
    completed_epoch: int = -1
    best_validation_top1: float = float("-inf")
    best_epoch: int | None = None
    training_history: list[dict[str, Any]] = field(default_factory=list)
    validation_history: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_history: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class CheckpointRecord:
    """One finalized checkpoint, identified only after atomic replacement."""

    epoch: int
    path: Path
    checkpoint_id: str


@dataclass
class EpochScheduler:
    """The persisted state for the explicit AM-78 epoch-indexed schedule."""

    completed_epoch: int = -1

    def state_dict(self) -> dict[str, int]:
        return {"completed_epoch": self.completed_epoch}

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if set(value) != {"completed_epoch"}:
            raise ValueError("checkpoint scheduler state schema is invalid")
        completed = value["completed_epoch"]
        if not _is_int(completed) or completed < -1:
            raise ValueError("checkpoint scheduler completed_epoch is invalid")
        self.completed_epoch = completed


@dataclass(frozen=True)
class ResumeCandidate:
    """Fully validated, detached resume state, safe to apply to a live trainer."""

    model_state: Mapping[str, torch.Tensor]
    optimizer_state: Mapping[str, Any]
    scheduler_state: Mapping[str, int]
    state: TrainingState
    execution_mode: str
    smoke_steps: int | None
    smoke_val_batches: int | None
    full_run_requested: bool
    run_complete_requested: bool
    g1_eligible_requested: bool


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def re_full_sha256(value: str) -> bool:
    return re.fullmatch(r"[0-9a-f]{64}", value) is not None


def learning_rate_for_epoch(epoch: int, *, total_epochs: int | None = None) -> float:
    """The exact zero-based AM-78 warmup-plus-cosine formula."""

    total = int(get("reference_classifier.epochs")) if total_epochs is None else total_epochs
    warmup = int(get("reference_classifier.lr_warmup_epochs"))
    base = float(get("reference_classifier.lr"))
    start = float(get("reference_classifier.lr_warmup_start_factor"))
    minimum = float(get("reference_classifier.lr_min"))
    if epoch < 0 or epoch >= total:
        raise ValueError(f"epoch {epoch} is outside zero-based schedule length {total}")
    if epoch < warmup:
        return base * (start + (1 - start) * epoch / max(warmup - 1, 1))
    j = epoch - warmup
    cosine_epochs = total - warmup
    return minimum + (base - minimum) * 0.5 * (  # literal-ok: AM-78 cosine formula
        1 + math.cos(math.pi * j / max(cosine_epochs - 1, 1))
    )


def validate(
    model: nn.Module,
    dataset: Dataset[tuple[torch.Tensor, int]],
    *,
    batch_size: int,
    device: torch.device | str,
    num_workers: int = 0,
    max_batches: int | None = None,
) -> ValidationResult:
    """Evaluate one manifest-ordered validation view without gradients."""

    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    try:
        with torch.no_grad():
            for batch_index, (inputs, labels) in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                logits = model(inputs.to(device))
                predictions = logits.argmax(dim=1)
                labels = labels.to(device)
                correct += int((predictions == labels).sum().item())
                total += int(labels.numel())
    finally:
        model.train(was_training)
    return ValidationResult(correct, total)


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


def atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> str:
    """Write a portable checkpoint by replacement, then hash final file bytes."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(_portable(dict(payload)), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return hashlib.sha256(destination.read_bytes()).hexdigest()


class ReferenceClassifierTrainer:
    """Config-derived SGD trainer with direct-epoch resume semantics."""

    def __init__(
        self,
        config: RunConfig,
        *,
        model: ReferenceClassifier | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        resolved = config.resolved
        self.dataset = str(resolved["dataset"])
        self.architecture = str(resolved["architecture"])
        self.train_seed = int(resolved["train_seed"])
        self.model = model or build_reference_classifier(
            self.dataset,
            architecture=self.architecture,
            train_seed=self.train_seed,
            device=self.device,
        )
        self.model.to(self.device)
        if get("reference_classifier.optimizer") != "sgd_momentum":
            raise NotImplementedError("unsupported reference-classifier optimizer")
        if get("reference_classifier.loss") != "cross_entropy":
            raise NotImplementedError("unsupported reference-classifier loss")
        if get("reference_classifier.mixed_precision"):
            raise NotImplementedError("mixed precision is prohibited by AM-78")
        self.optimizer = self._new_optimizer(self.model)
        self.loss = nn.CrossEntropyLoss(label_smoothing=float(get("reference_classifier.label_smoothing")))
        self.state = TrainingState()
        self._scheduler = EpochScheduler()
        self.execution_mode: str | None = None
        self.smoke_steps: int | None = None
        self.smoke_val_batches: int | None = None
        self.full_run_requested = False
        self.run_complete_requested = False
        self.g1_eligible_requested = False

    @staticmethod
    def _new_optimizer(model: nn.Module) -> Optimizer:
        return SGD(
            model.parameters(),
            lr=learning_rate_for_epoch(0),
            momentum=float(get("reference_classifier.momentum")),
            weight_decay=float(get("reference_classifier.weight_decay")),
            nesterov=bool(get("reference_classifier.nesterov")),
        )

    @property
    def scheduler_state(self) -> dict[str, int]:
        return self._scheduler.state_dict()

    def _set_learning_rate(self, epoch: int, total_epochs: int | None = None) -> float:
        value = learning_rate_for_epoch(epoch, total_epochs=total_epochs)
        for group in self.optimizer.param_groups:
            group["lr"] = value
        return value

    def _mark_bounded_work(self, *, smoke_steps: int | None = None, smoke_val_batches: int | None = None) -> None:
        """Establish lineage before work; a bounded call permanently makes it smoke-only."""

        self._validate_smoke_bound("smoke_steps", smoke_steps)
        self._validate_smoke_bound("smoke_val_batches", smoke_val_batches)
        if smoke_steps is not None or smoke_val_batches is not None:
            self.execution_mode = "smoke"
            self.full_run_requested = False
            self.run_complete_requested = False
            self.g1_eligible_requested = False
            if smoke_steps is not None:
                self.smoke_steps = smoke_steps
            if smoke_val_batches is not None:
                self.smoke_val_batches = smoke_val_batches
        elif self.execution_mode is None:
            self.execution_mode = "full"
            self.full_run_requested = True

    def train_epoch(
        self,
        epoch: int,
        dataset: Dataset[tuple[torch.Tensor, int]] | None = None,
        *,
        batch_size: int | None = None,
        num_workers: int = 0,
        max_steps: int | None = None,
        total_epochs: int | None = None,
    ) -> dict[str, Any]:
        """Train one keyed epoch; bounded steps are smoke-only administrative limits."""

        self._mark_bounded_work(smoke_steps=max_steps)
        view = dataset or TrainingClassifierDataset(self.dataset, self.train_seed, epoch)
        size = int(get("reference_classifier.batch_size")) if batch_size is None else batch_size
        sampler = EpochPermutationSampler(len(view), self.train_seed, epoch)
        loader = DataLoader(view, batch_size=size, sampler=sampler, num_workers=num_workers, drop_last=False)
        self.model.train()
        learning_rate = self._set_learning_rate(epoch, total_epochs)
        total_loss = 0.0
        total_examples = 0
        steps = 0
        for inputs, labels in loader:
            if max_steps is not None and steps >= max_steps:
                break
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(inputs.to(self.device))
            loss = self.loss(logits, labels.to(self.device))
            loss.backward()
            self.optimizer.step()
            count = int(labels.numel())
            total_loss += float(loss.detach().item()) * count
            total_examples += count
            steps += 1
        if total_examples == 0:
            raise RuntimeError("training epoch processed no examples")
        record = {"epoch": epoch, "lr": learning_rate, "loss": total_loss / total_examples, "steps": steps, "sample_order": list(sampler)}
        self.state.completed_epoch = epoch
        self._scheduler.completed_epoch = epoch
        self.state.training_history.append(record)
        return record

    def validate_epoch(
        self,
        epoch: int,
        dataset: Dataset[tuple[torch.Tensor, int]] | None = None,
        *,
        batch_size: int | None = None,
        num_workers: int = 0,
        max_batches: int | None = None,
    ) -> ValidationResult:
        self._mark_bounded_work(smoke_val_batches=max_batches)
        view = dataset or ValidationClassifierDataset(self.dataset)
        size = int(get("reference_classifier.batch_size")) if batch_size is None else batch_size
        result = validate(self.model, view, batch_size=size, device=self.device, num_workers=num_workers, max_batches=max_batches)
        record = {"epoch": epoch, "n_correct": result.n_correct, "n_total": result.n_total, "top1_accuracy": result.top1_accuracy}
        self.state.validation_history.append(record)
        if result.top1_accuracy > self.state.best_validation_top1:
            self.state.best_validation_top1 = result.top1_accuracy
            self.state.best_epoch = epoch
        return result

    @staticmethod
    def _validate_smoke_bound(name: str, value: int | None) -> None:
        if value is not None and (not _is_int(value) or value <= 0):
            raise ValueError(f"{name} must be a positive integer or null")

    @classmethod
    def _validate_execution_lineage(
        cls,
        *,
        execution_mode: object,
        smoke_steps: object,
        smoke_val_batches: object,
        full_run_requested: object,
        run_complete: object,
        g1_eligible: object,
        lineage_g1_eligible: object,
        completed_epoch: int,
    ) -> tuple[str, int | None, int | None, bool]:
        if execution_mode not in {"smoke", "full"}:
            raise ValueError("checkpoint execution_mode must be 'smoke' or 'full'")
        if not isinstance(full_run_requested, bool):
            raise ValueError("checkpoint full_run_requested must be boolean")
        if not isinstance(run_complete, bool) or not isinstance(g1_eligible, bool) or not isinstance(lineage_g1_eligible, bool):
            raise ValueError("checkpoint completion and eligibility fields must be boolean")
        cls._validate_smoke_bound("checkpoint smoke_steps", smoke_steps)
        cls._validate_smoke_bound("checkpoint smoke_val_batches", smoke_val_batches)
        scheduled_final = int(get("reference_classifier.epochs")) - 1
        if execution_mode == "smoke":
            if full_run_requested or (smoke_steps is None and smoke_val_batches is None):
                raise ValueError("smoke checkpoint lineage is incomplete")
            if run_complete or g1_eligible or lineage_g1_eligible:
                raise ValueError("smoke checkpoint cannot be complete or G-1 eligible")
        else:
            if not full_run_requested:
                raise ValueError("full checkpoint lacks explicit full-run lineage")
            if smoke_steps is not None or smoke_val_batches is not None:
                raise ValueError("full checkpoint cannot carry smoke bounds")
            if run_complete and completed_epoch != scheduled_final:
                raise ValueError("full checkpoint marks completion before the configured final epoch")
            expected_eligibility = run_complete and completed_epoch == scheduled_final
            if g1_eligible != expected_eligibility or lineage_g1_eligible != expected_eligibility:
                raise ValueError("full checkpoint G-1 eligibility is inconsistent with completion")
        return execution_mode, smoke_steps, smoke_val_batches, full_run_requested

    def checkpoint_payload(self) -> dict[str, Any]:
        if self.execution_mode is None:
            raise ValueError("checkpoint lineage must be established before saving")
        scheduled_final = int(get("reference_classifier.epochs")) - 1
        run_complete = self.run_complete_requested and self.state.completed_epoch == scheduled_final
        g1_eligible = self.g1_eligible_requested and run_complete
        lineage_g1_eligible = g1_eligible
        self._validate_execution_lineage(
            execution_mode=self.execution_mode,
            smoke_steps=self.smoke_steps,
            smoke_val_batches=self.smoke_val_batches,
            full_run_requested=self.full_run_requested,
            run_complete=run_complete,
            g1_eligible=g1_eligible,
            lineage_g1_eligible=lineage_g1_eligible,
            completed_epoch=self.state.completed_epoch,
        )
        return {
            "checkpoint_schema_version": get("reference_classifier.checkpoint_schema_version"),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler_state,
            "completed_epoch": self.state.completed_epoch,
            "next_epoch": self.state.completed_epoch + 1,
            "best_validation_top1": self.state.best_validation_top1,
            "best_epoch": self.state.best_epoch,
            "resolved_run_config": self.config.to_dict(),
            "config_hash": config_hash(self.config),
            "dataset": self.dataset,
            "dataset_version": self.config.resolved["dataset_version"],
            "split_manifest_hash": self.config.resolved["split_manifest_hash"],
            "classifier_variant": self.config.resolved["classifier_variant"],
            "architecture": self.architecture,
            "train_seed": self.train_seed,
            "model_total_parameter_count": self.model.total_parameter_count,
            "model_trainable_parameter_count": self.model.trainable_parameter_count,
            "training_history": self.state.training_history,
            "validation_history": self.state.validation_history,
            "checkpoint_history": self.state.checkpoint_history,
            "execution_mode": self.execution_mode,
            "smoke_steps": self.smoke_steps,
            "smoke_val_batches": self.smoke_val_batches,
            "full_run_requested": self.full_run_requested,
            "run_complete": run_complete,
            "g1_eligible": g1_eligible,
            "lineage_g1_eligible": lineage_g1_eligible,
        }

    def save_checkpoint(self, destination: Path) -> str:
        if get("artifacts.checkpoint_id_form") != "sha256_of_exact_checkpoint_file_bytes":
            raise NotImplementedError("unsupported checkpoint identity form")
        checkpoint_id = atomic_torch_save(
            self.checkpoint_payload(),
            destination,
        )
        self.state.checkpoint_history.append(
            {"epoch": self.state.completed_epoch, "checkpoint_id": checkpoint_id}
        )
        return checkpoint_id

    def _validate_run_arguments(
        self,
        *,
        final_epoch: int,
        execution_mode: str,
        full_run_requested: bool,
        smoke_steps: int | None,
        smoke_val_batches: int | None,
        run_complete: bool,
        g1_eligible: bool,
    ) -> None:
        if execution_mode not in {"smoke", "full"}:
            raise ValueError("execution_mode must be 'smoke' or 'full'")
        self._validate_smoke_bound("smoke_steps", smoke_steps)
        self._validate_smoke_bound("smoke_val_batches", smoke_val_batches)
        if not isinstance(full_run_requested, bool) or not isinstance(run_complete, bool) or not isinstance(g1_eligible, bool):
            raise ValueError("run lineage flags must be boolean")
        if (smoke_steps is not None or smoke_val_batches is not None) and (run_complete or g1_eligible):
            raise ValueError("smoke bounds cannot coexist with completion or G-1 eligibility")
        if execution_mode == "smoke":
            if full_run_requested:
                raise ValueError("smoke execution mode cannot request a full run")
            if smoke_steps is None or smoke_val_batches is None:
                raise ValueError("smoke execution mode requires both smoke bounds")
            if run_complete or g1_eligible:
                raise ValueError("smoke execution mode cannot be complete or G-1 eligible")
            return
        if not full_run_requested:
            raise ValueError("full execution mode requires explicit full_run_requested=True")
        if smoke_steps is not None or smoke_val_batches is not None:
            raise ValueError("full execution mode cannot carry smoke bounds")
        scheduled_final = int(get("reference_classifier.epochs")) - 1
        if (run_complete or g1_eligible) and final_epoch != scheduled_final:
            raise ValueError("completion or G-1 eligibility requires the configured full epoch schedule")
        if g1_eligible and not run_complete:
            raise ValueError("G-1 eligibility requires run_complete=True")

    def run_epochs(
        self,
        *,
        final_epoch: int,
        checkpoint_dir: Path,
        num_workers: int = 0,
        training_dataset: Dataset[tuple[torch.Tensor, int]] | None = None,
        validation_dataset: Dataset[tuple[torch.Tensor, int]] | None = None,
        execution_mode: str = "smoke",
        full_run_requested: bool = False,
        smoke_steps: int | None = None,
        smoke_val_batches: int | None = None,
        run_complete: bool = False,
        g1_eligible: bool = False,
    ) -> list[CheckpointRecord]:
        """Run a contiguous full or bounded-smoke interval and checkpoint it."""

        self._validate_run_arguments(
            final_epoch=final_epoch,
            execution_mode=execution_mode,
            full_run_requested=full_run_requested,
            smoke_steps=smoke_steps,
            smoke_val_batches=smoke_val_batches,
            run_complete=run_complete,
            g1_eligible=g1_eligible,
        )
        if self.execution_mode == "smoke" and execution_mode == "full":
            raise ValueError("smoke-only trainer lineage cannot be promoted to full mode")
        start_epoch = self.state.completed_epoch + 1
        if final_epoch < start_epoch:
            raise ValueError(f"final epoch {final_epoch} precedes next epoch {start_epoch}")
        interval = int(get("compute.checkpoint_every_epochs"))
        if interval <= 0:
            raise ValueError("checkpoint_every_epochs must be positive")
        validation_interval = int(get("reference_classifier.validation_every_epochs"))
        if validation_interval <= 0:
            raise ValueError("validation_every_epochs must be positive")
        total_epochs = int(get("reference_classifier.epochs"))
        self.execution_mode = execution_mode
        self.smoke_steps = smoke_steps
        self.smoke_val_batches = smoke_val_batches
        self.full_run_requested = full_run_requested
        self.run_complete_requested = run_complete
        self.g1_eligible_requested = g1_eligible
        records: list[CheckpointRecord] = []
        for epoch in range(start_epoch, final_epoch + 1):
            self.train_epoch(
                epoch,
                training_dataset,
                num_workers=num_workers,
                max_steps=smoke_steps,
                total_epochs=total_epochs,
            )
            if (epoch + 1) % validation_interval == 0:
                self.validate_epoch(
                    epoch,
                    validation_dataset,
                    num_workers=num_workers,
                    max_batches=smoke_val_batches,
                )
            if (epoch + 1) % interval == 0 or epoch == final_epoch:
                checkpoint = checkpoint_dir / f"epoch-{epoch}.pt"
                checkpoint_id = self.save_checkpoint(checkpoint)
                records.append(CheckpointRecord(epoch, checkpoint, checkpoint_id))
        return records

    @staticmethod
    def _validate_history_entry(
        value: object,
        *,
        fields: set[str],
        label: str,
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError(f"checkpoint {label} history schema is invalid")
        result = dict(value)
        if not _is_int(result["epoch"]) or result["epoch"] < 0:
            raise ValueError(f"checkpoint {label} history epoch is invalid")
        return result

    @classmethod
    def _validate_history(cls, payload: Mapping[str, Any], completed: int) -> TrainingState:
        training = payload["training_history"]
        validation = payload["validation_history"]
        checkpoints = payload["checkpoint_history"]
        if not isinstance(training, list) or not isinstance(validation, list) or not isinstance(checkpoints, list):
            raise ValueError("checkpoint histories must be lists")
        training_records = [
            cls._validate_history_entry(
                item,
                fields={"epoch", "lr", "loss", "steps", "sample_order"},
                label="training",
            )
            for item in training
        ]
        if [item["epoch"] for item in training_records] != list(range(completed + 1)):
            raise ValueError("checkpoint training history does not match completed epoch")
        for item in training_records:
            if not isinstance(item["lr"], Real) or isinstance(item["lr"], bool) or not isinstance(item["loss"], Real) or isinstance(item["loss"], bool) or not _is_int(item["steps"]) or item["steps"] <= 0 or not isinstance(item["sample_order"], list) or not all(_is_int(index) and index >= 0 for index in item["sample_order"]):
                raise ValueError("checkpoint training history values are invalid")
        validation_records = [
            cls._validate_history_entry(
                item,
                fields={"epoch", "n_correct", "n_total", "top1_accuracy"},
                label="validation",
            )
            for item in validation
        ]
        if len({item["epoch"] for item in validation_records}) != len(validation_records) or any(item["epoch"] > completed for item in validation_records):
            raise ValueError("checkpoint validation history epochs are invalid")
        for item in validation_records:
            if not _is_int(item["n_correct"]) or not _is_int(item["n_total"]) or item["n_total"] <= 0 or not 0 <= item["n_correct"] <= item["n_total"] or not isinstance(item["top1_accuracy"], Real) or isinstance(item["top1_accuracy"], bool):
                raise ValueError("checkpoint validation history values are invalid")
        checkpoint_records = [
            cls._validate_history_entry(
                item,
                fields={"epoch", "checkpoint_id"},
                label="checkpoint",
            )
            for item in checkpoints
        ]
        for item in checkpoint_records:
            if item["epoch"] > completed or not isinstance(item["checkpoint_id"], str) or not re_full_sha256(item["checkpoint_id"]):
                raise ValueError("checkpoint checkpoint history values are invalid")
        best = payload["best_validation_top1"]
        best_epoch = payload["best_epoch"]
        if not isinstance(best, Real) or isinstance(best, bool):
            raise ValueError("checkpoint best_validation_top1 is invalid")
        if best_epoch is not None and not _is_int(best_epoch):
            raise ValueError("checkpoint best_epoch is invalid")
        if validation_records:
            if not math.isfinite(float(best)):
                raise ValueError("checkpoint best_validation_top1 is invalid")
            maximum = max(float(item["top1_accuracy"]) for item in validation_records)
            expected_best = next(item["epoch"] for item in validation_records if float(item["top1_accuracy"]) == maximum)
            if best_epoch != expected_best or float(best) != maximum:
                raise ValueError("checkpoint best validation state is inconsistent")
        elif best_epoch is not None or best != float("-inf"):
            raise ValueError("checkpoint empty validation history has a best metric")
        return TrainingState(
            completed_epoch=completed,
            best_validation_top1=float(best),
            best_epoch=best_epoch,
            training_history=training_records,
            validation_history=validation_records,
            checkpoint_history=checkpoint_records,
        )

    def _validated_resume_candidate(
        self,
        payload: Mapping[str, Any],
        *,
        requested_execution_mode: str,
    ) -> ResumeCandidate:
        required = {
            "checkpoint_schema_version", "model_state", "optimizer_state", "scheduler_state",
            "completed_epoch", "next_epoch", "best_validation_top1", "best_epoch",
            "resolved_run_config", "config_hash", "dataset", "dataset_version",
            "split_manifest_hash", "classifier_variant", "architecture", "train_seed",
            "model_total_parameter_count", "model_trainable_parameter_count", "training_history",
            "validation_history", "checkpoint_history", "execution_mode", "smoke_steps",
            "smoke_val_batches", "full_run_requested", "run_complete", "g1_eligible",
            "lineage_g1_eligible",
        }
        if set(payload) != required:
            raise ValueError("checkpoint top-level schema is invalid")
        expected = {
            "checkpoint_schema_version": get("reference_classifier.checkpoint_schema_version"),
            "config_hash": config_hash(self.config),
            "dataset": self.dataset,
            "dataset_version": self.config.resolved["dataset_version"],
            "split_manifest_hash": self.config.resolved["split_manifest_hash"],
            "classifier_variant": self.config.resolved["classifier_variant"],
            "architecture": self.architecture,
            "train_seed": self.train_seed,
            "model_total_parameter_count": self.model.total_parameter_count,
            "model_trainable_parameter_count": self.model.trainable_parameter_count,
        }
        for key, value in expected.items():
            if payload[key] != value:
                raise ValueError(f"checkpoint {key} mismatch: {payload[key]!r} != {value!r}")
        resolved = payload["resolved_run_config"]
        if not isinstance(resolved, Mapping) or _canonical_json(dict(resolved)) != _canonical_json(self.config.to_dict()):
            raise ValueError("checkpoint resolved_run_config mismatch")
        completed = payload["completed_epoch"]
        if not _is_int(completed) or completed < -1 or completed >= int(get("reference_classifier.epochs")):
            raise ValueError("checkpoint completed_epoch is invalid")
        if payload["next_epoch"] != completed + 1:
            raise ValueError("checkpoint completed/next epoch state is inconsistent")
        execution_mode, smoke_steps, smoke_val_batches, full_run_requested = self._validate_execution_lineage(
            execution_mode=payload["execution_mode"],
            smoke_steps=payload["smoke_steps"],
            smoke_val_batches=payload["smoke_val_batches"],
            full_run_requested=payload["full_run_requested"],
            run_complete=payload["run_complete"],
            g1_eligible=payload["g1_eligible"],
            lineage_g1_eligible=payload["lineage_g1_eligible"],
            completed_epoch=completed,
        )
        if requested_execution_mode not in {"smoke", "full"}:
            raise ValueError("requested execution_mode must be 'smoke' or 'full'")
        if execution_mode != requested_execution_mode:
            raise ValueError(f"cannot resume {execution_mode} checkpoint in {requested_execution_mode} mode")
        scheduler_state = payload["scheduler_state"]
        if not isinstance(scheduler_state, Mapping):
            raise ValueError("checkpoint scheduler state is invalid")
        temporary_scheduler = EpochScheduler()
        temporary_scheduler.load_state_dict(scheduler_state)
        if temporary_scheduler.completed_epoch != completed:
            raise ValueError("checkpoint completed/scheduler epoch state is inconsistent")
        state = self._validate_history(payload, completed)
        model_state = payload["model_state"]
        optimizer_state = payload["optimizer_state"]
        if not isinstance(model_state, Mapping) or not isinstance(optimizer_state, Mapping):
            raise ValueError("checkpoint model or optimizer state is invalid")
        current_state = self.model.state_dict()
        if set(model_state) != set(current_state):
            raise ValueError("checkpoint model state keys differ")
        for key, current in current_state.items():
            candidate = model_state[key]
            if not isinstance(candidate, torch.Tensor) or candidate.shape != current.shape or candidate.dtype != current.dtype:
                raise ValueError(f"checkpoint model tensor {key!r} is incompatible")
        temporary_model = copy.deepcopy(self.model)
        try:
            temporary_model.load_state_dict(model_state, strict=True)
            temporary_optimizer = self._new_optimizer(temporary_model)
            expected_optimizer_groups = [
                {
                    key: group[key]
                    for key in ("momentum", "weight_decay", "nesterov", "dampening")
                }
                for group in temporary_optimizer.param_groups
            ]
            temporary_optimizer.load_state_dict(optimizer_state)
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(f"checkpoint optimizer or model state is invalid: {exc}") from None
        for parameter, parameter_state in temporary_optimizer.state.items():
            if not isinstance(parameter_state, Mapping) or set(parameter_state) - {"momentum_buffer"}:
                raise ValueError("checkpoint optimizer parameter state is invalid")
            momentum = parameter_state.get("momentum_buffer")
            if momentum is not None and (not isinstance(momentum, torch.Tensor) or momentum.shape != parameter.shape or momentum.dtype != parameter.dtype):
                raise ValueError("checkpoint optimizer momentum state is incompatible")
        expected_lr = learning_rate_for_epoch(max(completed, 0))
        if len(temporary_optimizer.param_groups) != len(expected_optimizer_groups):
            raise ValueError("checkpoint optimizer param-group count mismatch")
        for group, expected_group in zip(
            temporary_optimizer.param_groups,
            expected_optimizer_groups,
            strict=True,
        ):
            expected_group["lr"] = expected_lr
            for key, expected_value in expected_group.items():
                candidate_value = group.get(key)
                if type(candidate_value) is not type(expected_value) or candidate_value != expected_value:
                    raise ValueError(
                        f"checkpoint optimizer {key} mismatch: "
                        f"{candidate_value!r} != {expected_value!r}"
                    )
        return ResumeCandidate(
            model_state=model_state,
            optimizer_state=optimizer_state,
            scheduler_state=temporary_scheduler.state_dict(),
            state=state,
            execution_mode=execution_mode,
            smoke_steps=smoke_steps,
            smoke_val_batches=smoke_val_batches,
            full_run_requested=full_run_requested,
            run_complete_requested=payload["run_complete"],
            g1_eligible_requested=payload["g1_eligible"],
        )

    def resume(self, checkpoint: Path, *, execution_mode: str = "full") -> None:
        """Transactionally restore a fully validated compatible checkpoint."""

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError("checkpoint must be a mapping")
        candidate = self._validated_resume_candidate(
            payload,
            requested_execution_mode=execution_mode,
        )
        checkpoint_id = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        self.model.load_state_dict(candidate.model_state, strict=True)
        self.optimizer.load_state_dict(candidate.optimizer_state)
        self._scheduler.load_state_dict(candidate.scheduler_state)
        self.state = candidate.state
        self.state.checkpoint_history.append(
            {"epoch": self.state.completed_epoch, "checkpoint_id": checkpoint_id}
        )
        self.execution_mode = candidate.execution_mode
        self.smoke_steps = candidate.smoke_steps
        self.smoke_val_batches = candidate.smoke_val_batches
        self.full_run_requested = candidate.full_run_requested
        self.run_complete_requested = candidate.run_complete_requested
        self.g1_eligible_requested = candidate.g1_eligible_requested
