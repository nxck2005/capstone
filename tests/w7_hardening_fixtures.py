"""Deterministic tiny fixtures for W7 pre-science behavioral regressions."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.utils.data import Dataset

from config.run_config import RunConfig, config_hash as run_config_hash
from models.djscc import DJSCCOutput
from training.deterministic_core import canonical_sha256
from training.w7_g4 import W7SourceLineage, W7Trainer
from training.w7_protocol import W7_SELECTED_GPU_UUID, load_w7_config


class TinyW7Dataset(Dataset[tuple[torch.Tensor, int, str]]):
    """Tiny deterministic train fixture with production-style stable identities."""

    def __init__(self, epoch: int, count: int = 5, *, duplicate: bool = False) -> None:
        self.epoch = epoch
        self.count = count
        self.duplicate = duplicate

    def __len__(self) -> int:
        return self.count

    def source_sample(self, index: int) -> SimpleNamespace:
        identity_index = 0 if self.duplicate and index == self.count - 1 else index
        return SimpleNamespace(stable_sample_id=f"w7-tiny-{identity_index:04d}")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        source = self.source_sample(index)
        value = ((index * 3 + self.epoch) % 17) / 16
        return torch.full((3, 1, 1), value, dtype=torch.float32), index % 3, source.stable_sample_id


class TinyValidationDataset(Dataset[tuple[torch.Tensor, int, str]]):
    def __init__(self, _dataset: str, *, repo_root: Any = None, count: int = 5) -> None:
        del repo_root
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        value = (index + 1) / (self.count + 1)
        return (
            torch.full((3, 1, 1), value, dtype=torch.float32),
            index % 3,
            f"w7-val-{index:04d}",
        )


class TinyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # ingress is deliberately shared and outside the three historical named
        # gradient regions (encoder/reconstruction_head/task_head).
        self.ingress = nn.Linear(4, 4)
        self.reconstruction_head = nn.Linear(4, 3)
        self.task_head = nn.Linear(4, 10)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = torch.tanh(self.ingress(latent))
        reconstruction = torch.sigmoid(self.reconstruction_head(shared)).reshape(-1, 3, 1, 1)
        return reconstruction, self.task_head(shared)


class TinyDJSCC(nn.Module):
    """Small dual-head model obeying the W7 trainer's production interface."""

    def __init__(self) -> None:
        super().__init__()
        # Isolate fixture initialization from ambient Torch RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7341)
            self.encoder = nn.Linear(3, 4)
            self.decoder = TinyDecoder()

    def forward(
        self,
        inputs: torch.Tensor,
        snr_db: float | torch.Tensor,
        *,
        unit_noise: torch.Tensor | None = None,
    ) -> DJSCCOutput:
        del snr_db
        latent = self.encoder(inputs.reshape(inputs.shape[0], 3))
        if unit_noise is not None:
            latent = latent + unit_noise[:, :4].real.to(latent.dtype) * 0.01
        reconstruction, logits = self.decoder(latent)
        transmitted = torch.complex(latent, torch.zeros_like(latent))
        power = transmitted.abs().square()
        papr = 10 * torch.log10(power.amax(dim=1) / power.mean(dim=1).clamp_min(1e-12))
        return DJSCCOutput(
            transmitted_symbols=transmitted,
            received_symbols=transmitted,
            reconstruction=reconstruction,
            logits=logits,
            papr_db=papr,
        )


def tiny_config(*, lambda_value: float = 0.3, epochs: int = 3) -> RunConfig:
    value = copy.deepcopy(
        load_w7_config(
            lambda_value=lambda_value,
            physical_batch_size=2,
            accumulation_factor=16,
            validation_batch_size=32,
        ).to_dict()
    )
    value["parameters"]["learned_system"]["epochs"]["imagenette160"] = epochs
    return RunConfig.from_dict(value)


def lineage() -> W7SourceLineage:
    return W7SourceLineage(
        source_commit="a" * 40,
        source_manifest_id="w7source-tiny-fixture",
        source_manifest_sha256="b" * 64,
        execution_image="w7-tiny-test-image",
    )


def profile_binding(config: RunConfig) -> dict[str, Any]:
    return {
        "authentication_status": "PASSED",
        "execution_profile_id": "confessor_pascal_cu126",
        "gpu_uuid": W7_SELECTED_GPU_UUID,
        "gpu_name": "NVIDIA GeForce GTX 1080 Ti",
        "gpu_compute_capability": "6.1",
        "lock_file_sha256": "d3561c8e930797d328cf45df1bdc5085842833665f9b5c9e5617d4c891e31a82",
        "git_commit": "a" * 40,
        "config_hash": run_config_hash(config),
    }


def tiny_trainer(root: Path, *, lambda_value: float = 0.3, epochs: int = 3) -> W7Trainer:
    config = tiny_config(lambda_value=lambda_value, epochs=epochs)
    return W7Trainer(
        config,
        device="cpu",
        runtime_root=root,
        source_lineage=lineage(),
        profile_binding=profile_binding(config),
        model=TinyDJSCC(),
        num_workers=0,
    )


def validation_get(actual_get, count: int):
    def get_value(path: str):
        if path == "datasets.imagenette160.val_images":
            return count
        return actual_get(path)

    return get_value


def fake_validation_summary(trainer: W7Trainer, checkpoint_id: str) -> dict[str, Any]:
    epoch = trainer.completed_epoch
    total = 5
    correct = (epoch + 2) % (total + 1)
    body = {
        "schema_version": 1,
        "artifact_role": "W7_VALIDATION_EPOCH_SUMMARY",
        "epoch": epoch,
        "checkpoint_id": checkpoint_id,
        "n_correct": correct,
        "n_total": total,
        "top1_accuracy": correct / total,
        "prediction_digest": canonical_sha256([epoch, "predictions"]),
        "evaluation_config_hash": canonical_sha256(["evaluation", epoch]),
        "noise_policy": "keyed_channel_noise_same_per_image_across_lambda",
        "noise_policy_hash": canonical_sha256(["noise-policy", epoch]),
        "noise_id_digest": canonical_sha256(["noise-ids", epoch]),
        "row_digest": canonical_sha256(["rows", epoch]),
    }
    body["summary_id"] = canonical_sha256(body)
    return body
