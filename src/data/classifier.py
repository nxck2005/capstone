"""Manifest-backed immutable train/validation views for the reference classifier."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import Dataset, Sampler

from artifacts.rng import keyed_generator
from data.preprocessing import evaluation_input, training_input
from data.registry import CanonicalDataset, load_dataset


class TrainingClassifierDataset(Dataset[tuple[torch.Tensor, int]]):
    """One immutable augmented training view bound to a seed and epoch."""

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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        product, label = self._source[index]
        return training_input(
            product,
            {
                "stable_sample_id": product.stable_sample_id,
                "train_seed": self.train_seed,
                "epoch": self.epoch,
            },
        ), label


class TrainingDJSCCDataset(Dataset[tuple[torch.Tensor, int, str]]):
    """Train-only DJSCC view retaining stable IDs for keyed channel noise."""

    def __init__(self, dataset: str, train_seed: int, epoch: int, *, repo_root=None) -> None:
        if not isinstance(train_seed, int) or isinstance(train_seed, bool):
            raise TypeError("train_seed must be an integer")
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.dataset = dataset
        self.train_seed = train_seed
        self.epoch = epoch
        # This class deliberately has no split argument. Model-facing test data
        # remains reachable only through data.test_access after G-12.
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


class ValidationClassifierDataset(Dataset[tuple[torch.Tensor, int]]):
    """The stable-manifest-order deterministic validation view."""

    def __init__(self, dataset: str, *, repo_root=None) -> None:
        self.dataset = dataset
        self._source = _load(dataset, "val", repo_root)

    def __len__(self) -> int:
        return len(self._source)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        product, label = self._source[index]
        return evaluation_input(product), label


def _load(dataset: str, split: str, repo_root: object) -> CanonicalDataset:
    return load_dataset(dataset, split) if repo_root is None else load_dataset(dataset, split, repo_root)


def epoch_permutation(dataset_length: int, train_seed: int, epoch: int) -> tuple[int, ...]:
    """Return the direct keyed-Philox permutation for one epoch only."""

    if not isinstance(dataset_length, int) or isinstance(dataset_length, bool) or dataset_length < 0:
        raise ValueError("dataset_length must be a non-negative integer")
    order = keyed_generator(
        "batch_order", {"train_seed": train_seed, "epoch": epoch}
    ).permutation(dataset_length)
    return tuple(int(index) for index in order)


class EpochPermutationSampler(Sampler[int]):
    """A stateless sampler; workers cannot retain an old epoch's order."""

    def __init__(self, dataset_length: int, train_seed: int, epoch: int) -> None:
        self.dataset_length = dataset_length
        self.train_seed = train_seed
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        return iter(epoch_permutation(self.dataset_length, self.train_seed, self.epoch))

    def __len__(self) -> int:
        return self.dataset_length
