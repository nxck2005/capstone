"""Small deterministic W8 fixtures; every optimizer step is non-scientific."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.utils.data import Dataset

from config.run_config import RunConfig, config_hash as run_config_hash
from models.djscc import DJSCCOutput
from training.deterministic_core import canonical_sha256
from training.w8_final import (
    W8SourceLineage,
    W8Trainer,
    W8_CORE_TRAINING_POLICY,
    W8_SMOKE_POLICY,
)
from training.w8_protocol import (
    W8_EXECUTION_IMAGE_FAMILY,
    W8_PROFILE_ID,
    W8_SELECTED_GPU_NAME,
    W8_SELECTED_GPU_UUID,
    eligibility_for_role,
    load_w8_config,
)


class TinyW8Dataset(Dataset[tuple[torch.Tensor, int, str]]):
    """Tiny train fixture with stable IDs and a production-style sampler hook."""

    def __init__(self, epoch: int, count: int = 5, *, duplicate: bool = False) -> None:
        self.epoch = epoch
        self.count = count
        self.duplicate = duplicate

    def __len__(self) -> int:
        return self.count

    def source_sample(self, index: int) -> SimpleNamespace:
        identity_index = 0 if self.duplicate and index == self.count - 1 else index
        return SimpleNamespace(stable_sample_id=f"w8-tiny-{identity_index:04d}")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        source = self.source_sample(index)
        value = ((index * 3 + self.epoch) % 17) / 16
        return (
            torch.full((3, 1, 1), value, dtype=torch.float32),
            index % 3,
            source.stable_sample_id,
        )


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
            f"w8-val-{index:04d}",
        )


class TinyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # This shared ingress is deliberately outside the named head regions;
        # the W5/AM-91 regression must classify it through optimizer ownership.
        self.ingress = nn.Linear(4, 4)
        self.reconstruction_head = nn.Linear(4, 3)
        self.task_head = nn.Linear(4, 10)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = torch.tanh(self.ingress(latent))
        reconstruction = torch.sigmoid(self.reconstruction_head(shared)).reshape(
            -1, 3, 1, 1
        )
        return reconstruction, self.task_head(shared)


class TinyDJSCC(nn.Module):
    """Small dual-head model implementing the DJSCC trainer interface."""

    def __init__(self, seed: int = 7341) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
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


def tiny_config(*, ratio: str = "r_1_6", train_seed: int = 0, channel_seed: int = 0, role: str = "W8_NON_SCIENTIFIC_SMOKE") -> RunConfig:
    return load_w8_config(
        ratio,
        train_seed,
        channel_seed,
        role=role,
    )


def lineage() -> W8SourceLineage:
    return W8SourceLineage(
        source_commit="a" * 40,
        source_manifest_id="w8source-tiny-fixture",
        source_manifest_sha256="b" * 64,
        execution_image=W8_EXECUTION_IMAGE_FAMILY,
    )


def profile_binding(config: RunConfig, *, gpu_uuid: str = W8_SELECTED_GPU_UUID) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "authentication_status": "PASSED",
        "execution_profile_id": W8_PROFILE_ID,
        "gpu_uuid": gpu_uuid,
        "gpu_name": W8_SELECTED_GPU_NAME,
        "gpu_compute_capability": "6.1",
        "cuda_visible_devices": W8_SELECTED_GPU_UUID,
        "device": "cuda:0",
        "profile_environment": {
            "execution_profile_id": W8_PROFILE_ID,
            "lock_file": "requirements-pascal.lock",
            "lock_file_sha256": "d3561c8e930797d328cf45df1bdc5085842833665f9b5c9e5617d4c891e31a82",
            "gpu_uuid": gpu_uuid,
            "gpu_name": W8_SELECTED_GPU_NAME,
            "gpu_compute_capability": "6.1",
            "git_commit": "a" * 40,
            "git_dirty": False,
            "config_hash": run_config_hash(config),
        },
        "lock_file": "requirements-pascal.lock",
        "lock_file_sha256": "d3561c8e930797d328cf45df1bdc5085842833665f9b5c9e5617d4c891e31a82",
        "git_commit": "a" * 40,
        "git_dirty": False,
        "config_hash": run_config_hash(config),
    }
    value["binding_sha256"] = canonical_sha256(value)
    return value


def tiny_trainer(
    root: Path,
    *,
    ratio: str = "r_1_6",
    train_seed: int = 0,
    channel_seed: int = 0,
    role: str = "W8_NON_SCIENTIFIC_SMOKE",
    campaign_id: str = "w8-test-campaign",
    run_id: str | None = None,
    model: nn.Module | None = None,
    policy=None,
) -> W8Trainer:
    config = tiny_config(
        ratio=ratio,
        train_seed=train_seed,
        channel_seed=channel_seed,
        role=role,
    )
    kwargs: dict[str, Any] = {
        "device": "cpu",
        "runtime_root": root,
        "source_lineage": lineage(),
        "profile_binding": profile_binding(config),
        "campaign_id": campaign_id,
        "run_id": run_id,
        "num_workers": 0,
    }
    # Core trainers must exercise the production constructor, which builds the
    # model from the keyed init identity.  The module-level builder is
    # monkeypatched to TinyDJSCC by the bounded tests.  Only the explicitly
    # non-scientific smoke role may use an injected tiny model seam.
    if model is not None:
        kwargs["model"] = model
    elif role != "W8_FINAL_MULTI_SEED_RUN":
        kwargs["model"] = TinyDJSCC()
    kwargs["policy"] = policy or (
        W8_CORE_TRAINING_POLICY if role == "W8_FINAL_MULTI_SEED_RUN" else W8_SMOKE_POLICY
    )
    return W8Trainer(config, **kwargs)


def fake_summary(
    *,
    epoch: int,
    ratio: str = "r_1_6",
    train_seed: int = 0,
    channel_seed: int = 0,
    correct: int = 500,
    total: int = 1000,
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Build a fully shaped, count-derived selection fixture without data."""

    body: dict[str, Any] = {
        "schema_version": 1,
        "artifact_role": "W8_VALIDATION_EPOCH_SUMMARY",
        "eligibility": eligibility_for_role("W8_FINAL_MULTI_SEED_RUN"),
        "campaign_id": "w8-test-campaign",
        "run_id": f"w8-{ratio}-train{train_seed}-channel{channel_seed}",
        "ratio": ratio,
        "k": 12800 if ratio == "r_1_6" else 3200,
        "train_seed": train_seed,
        "channel_seed": channel_seed,
        "checkpoint_id": checkpoint_id or f"{epoch + 1:064x}",
        "epoch": epoch,
        "validation_split": "val",
        "validation_order": "stable_manifest_order",
        "validation_augmentation": False,
        "validation_batch_size": 32,
        "validation_snr_parameter": "params.learned_system.checkpoint_selection_snr_db",
        "validation_snr_resolution": "params.channel.train_snr_db_fixed",
        "validation_snr_db": 7,
        "validation_channel_seed_rule": "run_channel_seed",
        "validation_channel_seed": channel_seed,
        "validation_noise_policy": "keyed_per_image_fixed_snr_run_channel_seed_same_across_epochs",
        # Validation noise is deliberately epoch-independent in W8; only the
        # synthetic outcome digests vary across fixture epochs.
        "validation_noise_id_digest": canonical_sha256(["noise"]),
        "validation_noise_id_count": total,
        "n_correct": correct,
        "n_total": total,
        "top1_accuracy": correct / total,
        "prediction_digest": canonical_sha256(["prediction", epoch]),
        "row_digest": canonical_sha256(["rows", epoch]),
        "evaluation_config_hash": canonical_sha256(["evaluation", epoch]),
        "forbidden_selection_inputs": ["psnr", "papr", "reconstruction_loss"],
        "test_model_facing_access": 0,
    }
    # The fixture helper leaves eligibility to the test, which fills it from
    # the live protocol before authenticating the summary.
    body["summary_id"] = canonical_sha256(body)
    return body
