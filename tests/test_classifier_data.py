"""Classifier-specific manifest views and direct keyed batch order tests."""

from __future__ import annotations

from pathlib import Path

import torch

from data.classifier import (
    EpochPermutationSampler,
    TrainingClassifierDataset,
    ValidationClassifierDataset,
    epoch_permutation,
)
from data.preprocessing import evaluation_input, training_input


def test_training_view_uses_immutable_epoch_specific_augmentation(
    synthetic_dataset_repo: Path,
):
    first = TrainingClassifierDataset("cifar10", 17, 4, repo_root=synthetic_dataset_repo)
    repeated = TrainingClassifierDataset("cifar10", 17, 4, repo_root=synthetic_dataset_repo)
    changed_epoch = TrainingClassifierDataset("cifar10", 17, 5, repo_root=synthetic_dataset_repo)
    product, expected_label = first._source[0]

    expected = training_input(
        product,
        {"stable_sample_id": product.stable_sample_id, "train_seed": 17, "epoch": 4},
    )
    actual, label = first[0]
    other, _ = changed_epoch[0]

    assert label == expected_label
    assert torch.equal(actual, expected)
    assert torch.equal(actual, repeated[0][0])
    assert not torch.equal(actual, other)
    assert (first.train_seed, first.epoch) == (17, 4)


def test_validation_view_preserves_manifest_order_and_uses_no_augmentation(
    synthetic_dataset_repo: Path,
):
    view = ValidationClassifierDataset("stl10", repo_root=synthetic_dataset_repo)
    product, expected_label = view._source[0]
    actual, label = view[0]

    assert label == expected_label
    assert torch.equal(actual, evaluation_input(product))
    assert [view._source.source_sample(index).stable_sample_id for index in range(len(view))] == [
        view._source.source_sample(index).stable_sample_id for index in range(len(view))
    ]


def test_epoch_permutation_is_complete_direct_and_global_rng_independent():
    torch.manual_seed(97)
    before = torch.get_rng_state().clone()
    first = epoch_permutation(10, 17, 4)
    after = torch.get_rng_state()

    assert first == epoch_permutation(10, 17, 4)
    assert first == tuple(EpochPermutationSampler(10, 17, 4))
    assert sorted(first) == list(range(10))
    assert first != epoch_permutation(10, 17, 5)
    assert first != epoch_permutation(10, 18, 4)
    assert torch.equal(before, after)
