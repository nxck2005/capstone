"""Deterministic, train/validation-only DJSCC evaluation data views."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import Dataset

from config.run_config import canonical_sha256
from data.preprocessing import evaluation_input
from data.registry import load_dataset


class ValidationDataHold(RuntimeError):
    """Validation data was not the complete committed non-test split."""


class ValidationDJSCCDataset(Dataset[tuple[torch.Tensor, int, str]]):
    """Return canonical evaluation tensors in committed manifest order.

    There is deliberately no split argument.  This class cannot be pointed at
    the test split, and the registry itself rejects model-facing test loading.
    """

    def __init__(self, dataset: str, *, repo_root=None) -> None:
        self.dataset = dataset
        self._source = load_dataset(dataset, "val", repo_root) if repo_root is not None else load_dataset(dataset, "val")
        if len(self._source) <= 0:
            raise ValidationDataHold("validation dataset is empty")
        identifiers = [self._source.source_sample(index).stable_sample_id for index in range(len(self._source))]
        if len(set(identifiers)) != len(identifiers):
            raise ValidationDataHold("validation stable IDs are not unique")
        if identifiers != sorted(identifiers):
            raise ValidationDataHold("validation IDs are not in stable manifest order")

    def __len__(self) -> int:
        return len(self._source)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        product, label = self._source[index]
        return evaluation_input(product), int(label), product.stable_sample_id


def validation_noise_id(
    *,
    stable_sample_id: str,
    dataset_version: str,
    split_manifest_hash: str,
    channel_seed: int,
    channel: str,
    ratio: str,
    k: int,
    snr_db: int | float,
) -> str:
    """Derive one λ-independent identity for one validation image/SNR."""

    identity: Mapping[str, Any] = {
        "dataset_version": dataset_version,
        "split_manifest_hash": split_manifest_hash,
        "stable_sample_id": stable_sample_id,
        "channel_seed": channel_seed,
        "channel": channel,
        "bw_ratio": ratio,
        "k": k,
        "snr_db": snr_db,
        "rng_purpose": "channel_noise",
    }
    return canonical_sha256(identity)
