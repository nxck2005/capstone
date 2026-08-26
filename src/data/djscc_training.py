"""Train-only DJSCC dataset view retaining stable IDs for keyed noise."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from data.classifier import _load
from data.preprocessing import training_input


class TrainingDJSCCDataset(Dataset[tuple[torch.Tensor, int, str]]):
    """Common train-data path with no model-facing test-split argument."""

    def __init__(self, dataset: str, train_seed: int, epoch: int, *, repo_root=None) -> None:
        if not isinstance(train_seed, int) or isinstance(train_seed, bool):
            raise TypeError("train_seed must be an integer")
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.dataset = dataset
        self.train_seed = train_seed
        self.epoch = epoch
        self._source = _load(dataset, "train", repo_root)

    def __len__(self) -> int:
        return len(self._source)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        product, label = self._source[index]
        identity = {
            "stable_sample_id": product.stable_sample_id,
            "train_seed": self.train_seed,
            "epoch": self.epoch,
        }
        return training_input(product, identity), label, product.stable_sample_id
