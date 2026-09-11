"""Model/optimizer adapter over the shared W9 v4 transactional epoch store.

ER-9 and randomized ER-2 pass their own model and optimizer objects through
this small adapter.  The storage and recovery semantics stay identical, while
the historical v1-v3 trainers retain their original evidence contracts.
"""

from __future__ import annotations

import io
import os
import math
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from runtime.transactional_epochs import (
    CommittedEpoch,
    TransactionalEpochStore,
    TransactionalRuntimeHold,
    canonical_bytes,
    sha256_bytes,
)
from training.deterministic_core import OptimizerUpdate, apply_optimizer_update, gradient_status, optimizer_parameters


class W9V4TrainingHold(TransactionalRuntimeHold):
    """A model/optimizer state cannot be restored into the v4 identity."""


@dataclass(frozen=True)
class V4TrainingStep:
    """One forward/backward pass shared by scientific and synthetic paths."""

    output: Any
    loss: torch.Tensor
    loss_value: float
    gradient_status: dict[str, int | bool]
    optimizer_update: OptimizerUpdate | None


def v4_training_step(
    *,
    forward: Callable[[], Any],
    loss_fn: Callable[[Any], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    amp_enabled: bool,
    denominator: int | None = None,
) -> V4TrainingStep:
    """Run the canonical W9 v4 AMP forward/backward/update primitive.

    Callers may defer the update while accumulating microbatches by leaving
    ``denominator`` unset.  The same forward, loss, scaler and all-optimizer
    gradient path is then used by the real ER-9/ER-2 trainers and the
    synthetic CUDA fixture.
    """

    context = (
        torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True)
        if amp_enabled
        else nullcontext()
    )
    with context:
        output = forward()
        loss = loss_fn(output)
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        raise W9V4TrainingHold("v4 training loss must be a scalar tensor")
    loss_value = float(loss.detach().float().item())
    if not math.isfinite(loss_value):
        raise W9V4TrainingHold("v4 training loss is non-finite")
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()
    status = gradient_status(optimizer_parameters(optimizer))
    update = (
        apply_optimizer_update(optimizer, scaler, denominator=denominator)
        if denominator is not None
        else None
    )
    return V4TrainingStep(output, loss, loss_value, status, update)


class W9V4TrainingRuntime:
    """Shared checkpoint/resume facade for both prospective W9 systems."""

    def __init__(self, runtime_root: Path, *, identity: Mapping[str, Any], total_epochs: int, role: str) -> None:
        self.store = TransactionalEpochStore(runtime_root, identity=identity, total_epochs=total_epochs, role=role)
        self.identity = dict(identity)
        self.role = role

    def inspect(self) -> list[CommittedEpoch]:
        return self.store.inspect()

    @staticmethod
    def _checkpoint_bytes(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Any,
        *,
        epoch: int,
        identity: Mapping[str, Any],
        role: str,
    ) -> bytes:
        buffer = io.BytesIO()
        torch.save(
            {
                "schema_version": 1,
                "artifact_role": "W9_V4_MODEL_OPTIMIZER_CHECKPOINT",
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": None if scaler is None else scaler.state_dict(),
                "run_identity": dict(identity),
                "runtime_role": role,
                "test_access": 0,
            },
            buffer,
        )
        return buffer.getvalue()

    def commit_epoch(
        self,
        *,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Any,
        metrics: Mapping[str, Any],
    ) -> CommittedEpoch:
        record = dict(metrics)
        if "optimizer_opportunities" not in record and "optimizer_step_opportunities" in record:
            record["optimizer_opportunities"] = int(record["optimizer_step_opportunities"])
        if "applied_optimizer_steps" not in record and "optimizer_steps" in record:
            record["applied_optimizer_steps"] = int(record["optimizer_steps"])
        if "grad_scaler_skips" not in record:
            record["grad_scaler_skips"] = 0
        record["test_access"] = 0
        checkpoint = self._checkpoint_bytes(
            model,
            optimizer,
            scaler,
            epoch=epoch,
            identity=self.identity,
            role=self.role,
        )
        return self.store.publish_epoch(epoch, checkpoint, record)

    def restore_latest(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scaler: Any) -> int:
        epochs = self.store.inspect()
        if not epochs:
            return -1
        self.restore_epoch(epochs[-1].epoch, model, optimizer, scaler, epochs=epochs)
        return epochs[-1].epoch

    def restore_epoch(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Any,
        *,
        epochs: list[CommittedEpoch] | None = None,
    ) -> int:
        """Restore one already-authenticated epoch, including a selection."""

        committed = self.store.inspect() if epochs is None else epochs
        if not committed or epoch < 0 or epoch >= len(committed):
            raise W9V4TrainingHold("v4 requested checkpoint is not committed")
        item = committed[epoch]
        payload = torch.load(item.checkpoint_path, map_location="cpu", weights_only=False)
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != 1
            or payload.get("artifact_role") != "W9_V4_MODEL_OPTIMIZER_CHECKPOINT"
            or payload.get("run_identity") != self.identity
            or payload.get("runtime_role") != self.role
            or payload.get("epoch") != item.epoch
            or payload.get("test_access") != 0
        ):
            raise W9V4TrainingHold("v4 checkpoint is not a sealed model/optimizer payload")
        try:
            model.load_state_dict(payload["model_state"], strict=True)
            optimizer.load_state_dict(payload["optimizer_state"])
            if scaler is None:
                if payload.get("scaler_state") is not None:
                    raise W9V4TrainingHold("CPU v4 resume carries scaler state")
            else:
                if not isinstance(payload.get("scaler_state"), Mapping):
                    raise W9V4TrainingHold("CUDA v4 resume lacks scaler state")
                scaler.load_state_dict(payload["scaler_state"])
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise W9V4TrainingHold(f"v4 model/optimizer restore failed: {exc}") from None
        return item.epoch

    def publish_json_artifact(self, relative_path: str, value: Mapping[str, Any]) -> str:
        """Publish an immutable auxiliary record inside the v4 runtime."""

        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise W9V4TrainingHold("v4 auxiliary artifact escapes its runtime")
        path = self.store.runtime_root / relative
        if path.exists() or path.is_symlink():
            try:
                existing = path.read_bytes()
            except OSError as exc:
                raise W9V4TrainingHold(f"v4 auxiliary artifact cannot be read: {exc}") from None
            expected = canonical_bytes(dict(value))
            if existing != expected:
                raise W9V4TrainingHold(f"v4 auxiliary artifact differs: {relative_path}")
            return sha256_bytes(existing)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = path.open("xb")
        try:
            raw = canonical_bytes(dict(value))
            descriptor.write(raw)
            descriptor.flush()
            os.fsync(descriptor.fileno())
        finally:
            descriptor.close()
        descriptor_path = path.parent
        descriptor = os.open(descriptor_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return sha256_bytes(raw)

    def terminalize(self, *, selected_epoch: int, selection_metric: Mapping[str, Any], tie_break: str) -> tuple[dict[str, Any], bool]:
        return self.store.terminalize(selected_epoch=selected_epoch, selection_metric=selection_metric, tie_break=tie_break)


class W9V4TrainingLoop:
    """Common epoch/resume/selection lifecycle for ER-9, ER-2 and smoke."""

    def __init__(
        self,
        runtime: W9V4TrainingRuntime,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Any,
        total_epochs: int,
        train_epoch: Callable[[int], Mapping[str, Any]],
        selection_metric: str = "validation_n_correct",
        tie_break: str = "earliest_epoch",
        resume: bool = False,
    ) -> None:
        if total_epochs <= 0:
            raise W9V4TrainingHold("v4 total epoch count must be positive")
        if runtime.store.total_epochs != int(total_epochs):
            raise W9V4TrainingHold("v4 runtime epoch authority differs from loop")
        self.runtime = runtime
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.total_epochs = int(total_epochs)
        self.train_epoch = train_epoch
        self.selection_metric = selection_metric
        self.tie_break = tie_break
        self.completed_epoch = -1
        self.best_epoch: int | None = None
        self._resume = bool(resume)
        existing = runtime.inspect()
        if existing and not resume:
            raise W9V4TrainingHold("v4 runtime exists; resume must be explicit")
        if existing:
            self.completed_epoch = runtime.restore_latest(model, optimizer, scaler)
            self.best_epoch = self._best_epoch(existing)
        elif resume and runtime.store.runtime_root.exists():
            # An interrupted fresh initialisation may have created only the
            # runtime directory/epochs root.  It has no state to restore.
            self.completed_epoch = -1

    def _best_epoch(self, epochs: list[CommittedEpoch]) -> int | None:
        if not epochs:
            return None
        best: CommittedEpoch | None = None
        for item in epochs:
            left = item.record.get(self.selection_metric)
            if not isinstance(left, int) or isinstance(left, bool):
                raise W9V4TrainingHold("v4 selection metric is not an exact integer")
            if best is None:
                best = item
                continue
            right = best.record.get(self.selection_metric)
            if not isinstance(right, int) or isinstance(right, bool):
                raise W9V4TrainingHold("v4 selection metric is not an exact integer")
            if left > right:
                best = item
        return None if best is None else best.epoch

    def run(self, *, max_epochs: int | None = None) -> dict[str, Any] | None:
        remaining = self.total_epochs - (self.completed_epoch + 1)
        if max_epochs is not None and int(max_epochs) < 0:
            raise W9V4TrainingHold("v4 max_epochs must be non-negative")
        count = remaining if max_epochs is None else min(remaining, int(max_epochs))
        if count < 0:
            raise W9V4TrainingHold("v4 runtime has more epochs than its authority")
        for _ in range(count):
            epoch = self.completed_epoch + 1
            metrics = dict(self.train_epoch(epoch))
            if metrics.get("epoch") != epoch:
                raise W9V4TrainingHold("v4 trainer returned the wrong epoch")
            committed = self.runtime.commit_epoch(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                metrics=metrics,
            )
            self.completed_epoch = committed.epoch
            self.best_epoch = self._best_epoch(self.runtime.inspect())
        if self.completed_epoch + 1 < self.total_epochs:
            return None
        epochs = self.runtime.inspect()
        if len(epochs) != self.total_epochs or self.best_epoch is None:
            raise W9V4TrainingHold("v4 run does not have a complete committed prefix")
        selected = epochs[self.best_epoch]
        metric_value = selected.record.get(self.selection_metric)
        if not isinstance(metric_value, int) or isinstance(metric_value, bool):
            raise W9V4TrainingHold("v4 selected metric is not an exact integer")
        terminal, _ = self.runtime.terminalize(
            selected_epoch=self.best_epoch,
            selection_metric={
                "metric": self.selection_metric,
                "mode": "max",
                "value": metric_value,
            },
            tie_break=self.tie_break,
        )
        return terminal


__all__ = [
    "V4TrainingStep",
    "W9V4TrainingHold",
    "W9V4TrainingLoop",
    "W9V4TrainingRuntime",
    "v4_training_step",
]
