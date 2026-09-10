"""Model/optimizer adapter over the shared W9 v4 transactional epoch store.

ER-9 and randomized ER-2 pass their own model and optimizer objects through
this small adapter.  The storage and recovery semantics stay identical, while
the historical v1-v3 trainers retain their original evidence contracts.
"""

from __future__ import annotations

import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from runtime.transactional_epochs import CommittedEpoch, TransactionalEpochStore, TransactionalRuntimeHold


class W9V4TrainingHold(TransactionalRuntimeHold):
    """A model/optimizer state cannot be restored into the v4 identity."""


class W9V4TrainingRuntime:
    """Shared checkpoint/resume facade for both prospective W9 systems."""

    def __init__(self, runtime_root: Path, *, identity: Mapping[str, Any], total_epochs: int, role: str) -> None:
        self.store = TransactionalEpochStore(runtime_root, identity=identity, total_epochs=total_epochs, role=role)
        self.identity = dict(identity)

    def inspect(self) -> list[CommittedEpoch]:
        return self.store.inspect()

    @staticmethod
    def _checkpoint_bytes(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scaler: Any, *, epoch: int) -> bytes:
        buffer = io.BytesIO()
        torch.save(
            {
                "schema_version": 1,
                "artifact_role": "W9_V4_MODEL_OPTIMIZER_CHECKPOINT",
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": None if scaler is None else scaler.state_dict(),
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
        return self.store.publish_epoch(epoch, self._checkpoint_bytes(model, optimizer, scaler, epoch=epoch), record)

    def restore_latest(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scaler: Any) -> int:
        epochs = self.store.inspect()
        if not epochs:
            return -1
        payload = torch.load(epochs[-1].checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or payload.get("test_access") != 0:
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
        return epochs[-1].epoch

    def terminalize(self, *, selected_epoch: int, selection_metric: Mapping[str, Any], tie_break: str) -> tuple[dict[str, Any], bool]:
        return self.store.terminalize(selected_epoch=selected_epoch, selection_metric=selection_metric, tie_break=tie_break)


__all__ = ["W9V4TrainingHold", "W9V4TrainingRuntime"]
