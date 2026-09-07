"""AM-95 randomized ER-2 SNR assignment tests."""

from __future__ import annotations

import ast
from pathlib import Path

from config.params import REPO_ROOT, get
from training.er2_snr import (
    assignment_record,
    assignments_for_samples,
    select_training_snr_db,
    snr_selection_identity,
)


DATASET_VERSION = "dataset-version-sha256"
SPLIT_MANIFEST_HASH = "split-manifest-sha256"


def _select(sample_id: str, epoch: int, seed: int = 17) -> int:
    return select_training_snr_db(
        DATASET_VERSION,
        SPLIT_MANIFEST_HASH,
        sample_id,
        seed,
        epoch,
    )


def test_assignments_are_in_the_exact_configured_domain():
    expected = set(get("channel.train_snr_db_set"))
    assert expected == {1, 4, 7, 13, 19}
    assert {_select(f"sample-{index}", index) for index in range(32)} <= expected


def test_same_sample_epoch_and_seed_replay_exactly():
    assert _select("sample-a", 4) == _select("sample-a", 4)
    assert assignment_record(DATASET_VERSION, SPLIT_MANIFEST_HASH, "sample-a", 17, 4) == assignment_record(
        DATASET_VERSION, SPLIT_MANIFEST_HASH, "sample-a", 17, 4
    )


def test_assignment_is_stable_under_dataloader_reordering():
    sample_ids = ["sample-a", "sample-b", "sample-c", "sample-d"]
    first = assignments_for_samples(
        sample_ids,
        dataset_version=DATASET_VERSION,
        split_manifest_hash=SPLIT_MANIFEST_HASH,
        train_seed=17,
        epoch=4,
    )
    reordered = assignments_for_samples(
        list(reversed(sample_ids)),
        dataset_version=DATASET_VERSION,
        split_manifest_hash=SPLIT_MANIFEST_HASH,
        train_seed=17,
        epoch=4,
    )
    assert first == reordered


def test_assignment_is_stable_under_batch_size_changes():
    sample_ids = [f"sample-{index}" for index in range(17)]

    def batched(size: int) -> dict[str, int]:
        result: dict[str, int] = {}
        for start in range(0, len(sample_ids), size):
            result.update(
                assignments_for_samples(
                    sample_ids[start : start + size],
                    dataset_version=DATASET_VERSION,
                    split_manifest_hash=SPLIT_MANIFEST_HASH,
                    train_seed=17,
                    epoch=4,
                )
            )
        return result

    assert batched(1) == batched(4) == batched(9)


def test_assignment_is_stable_under_gradient_accumulation_changes():
    sample_ids = [f"sample-{index}" for index in range(19)]

    def accumulated(factor: int) -> dict[str, int]:
        result: dict[str, int] = {}
        for start in range(0, len(sample_ids), factor):
            result.update(
                assignments_for_samples(
                    sample_ids[start : start + factor],
                    dataset_version=DATASET_VERSION,
                    split_manifest_hash=SPLIT_MANIFEST_HASH,
                    train_seed=17,
                    epoch=4,
                )
            )
        return result

    assert accumulated(1) == accumulated(3) == accumulated(8)


def test_different_epochs_can_produce_different_assignments():
    assignments = {_select("sample-a", epoch) for epoch in range(32)}
    assert len(assignments) > 1


def test_sample_ids_are_independently_keyed():
    first = snr_selection_identity(DATASET_VERSION, SPLIT_MANIFEST_HASH, "sample-a", 17, 4)
    second = snr_selection_identity(DATASET_VERSION, SPLIT_MANIFEST_HASH, "sample-b", 17, 4)
    assert first != second
    records = {
        assignment_record(DATASET_VERSION, SPLIT_MANIFEST_HASH, sample, 17, 4)["identity"][
            "stable_sample_id"
        ]
        for sample in ("sample-a", "sample-b")
    }
    assert records == {"sample-a", "sample-b"}


def test_checkpoint_resume_preserves_future_assignments():
    epochs = range(12)
    uninterrupted = [_select("sample-a", epoch) for epoch in epochs]
    completed_before_checkpoint = uninterrupted[:5]
    resumed_future = [_select("sample-a", epoch) for epoch in range(5, 12)]
    assert completed_before_checkpoint == uninterrupted[:5]
    assert completed_before_checkpoint + resumed_future == uninterrupted


def test_selector_does_not_touch_test_split_or_freeze_manifest():
    source = (REPO_ROOT / "src/training/er2_snr.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert "data.test_access" not in imports
    assert "test_access" not in source
    assert "freeze_manifest" not in source


def test_fixed_snr_w8_and_g10_contract_is_unchanged():
    assert get("channel.train_snr_db_fixed") == 7
    assert get("learned_system.train_snr_protocol") == "one_model_per_ratio_at_fixed_snr"
    assert get("learned_system.checkpoint_selection_snr_db") == "train_snr_db_fixed"
    assert get("evaluation.g10_ratio") == "r_1_6"
    assert get("evaluation.g10_exact_arithmetic") == "rational_correct_count_over_denominator_no_tolerance"
