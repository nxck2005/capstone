"""W8 validation/checkpoint-selection tests using synthetic data only."""

from __future__ import annotations

import copy
import random

import numpy as np
import pytest
import torch

import evaluation.w8_validation as validation
from training.deterministic_core import canonical_sha256
from training.w8_final import W8Hold, W8_CORE_TRAINING_POLICY
from training.w8_protocol import W8_VALIDATION_SAMPLE_COUNT
from tests.w8_hardening_fixtures import TinyDJSCC, TinyValidationDataset, fake_summary, tiny_trainer


@pytest.fixture
def tiny_validation(monkeypatch):
    actual_get = validation.get

    def get_value(path: str):
        if path == "datasets.imagenette160.val_images":
            return 5
        return actual_get(path)

    monkeypatch.setattr(validation, "get", get_value)
    monkeypatch.setattr(validation, "W8_VALIDATION_SAMPLE_COUNT", 5)
    monkeypatch.setattr(
        validation,
        "ValidationDJSCCDataset",
        lambda dataset, repo_root=None: TinyValidationDataset(
            dataset, repo_root=repo_root, count=5
        ),
    )
    # The core constructor still owns model creation; this replaces only the
    # architecture with the bounded fixture and does not inject state.
    monkeypatch.setattr(
        "training.w8_final.build_djscc",
        lambda *_args, **_kwargs: TinyDJSCC(),
    )
    return monkeypatch


def _scientific_trainer(tmp_path, *, ratio: str = "r_1_6", train_seed: int = 0):
    return tiny_trainer(
        tmp_path,
        ratio=ratio,
        train_seed=train_seed,
        channel_seed=train_seed,
        role="W8_FINAL_MULTI_SEED_RUN",
        policy=W8_CORE_TRAINING_POLICY,
    )


def test_validation_is_complete_count_derived_and_non_test(tiny_validation, tmp_path):
    trainer = _scientific_trainer(tmp_path)
    trainer.completed_epoch = 0
    evaluation = validation.evaluate_validation(
        trainer, checkpoint_id="a" * 64, retain_rows=True
    )
    summary = evaluation.summary
    rows = list(evaluation.rows)
    assert summary["validation_split"] == "val"
    assert summary["n_total"] == len(rows) == 5
    assert summary["n_correct"] == sum(row["correct"] for row in rows)
    assert summary["top1_accuracy"] == summary["n_correct"] / 5
    assert summary["test_model_facing_access"] == 0
    validation._validate_summary(summary)
    assert [row["stable_sample_id"] for row in rows] == sorted(
        row["stable_sample_id"] for row in rows
    )


def test_validation_noise_is_batching_and_ambient_rng_invariant(tmp_path):
    trainer = _scientific_trainer(tmp_path, ratio="r_1_24")
    ids = [f"w8-val-{index:04d}" for index in range(5)]
    first_ids = validation._noise_ids(trainer, ids, 7)
    random.seed(19)
    np.random.seed(23)
    torch.manual_seed(29)
    _ = torch.randn(17)
    second_ids = validation._noise_ids(trainer, ids[:2], 7) + validation._noise_ids(
        trainer, ids[2:], 7
    )
    assert first_ids == second_ids
    full = validation.keyed_complex_noise(first_ids, 3200)
    pieces = torch.cat(
        [
            validation.keyed_complex_noise(first_ids[:2], 3200),
            validation.keyed_complex_noise(first_ids[2:], 3200),
        ],
        dim=0,
    )
    assert torch.equal(full, pieces)


def test_validation_dataset_has_no_model_facing_test_route(monkeypatch):
    import data.djscc_validation as validation_data

    calls = []

    class Source:
        def __len__(self):
            return 1

        def source_sample(self, index):
            return type("Sample", (), {"stable_sample_id": f"id-{index}"})()

    def load_dataset(dataset, split, repo_root=None):
        del repo_root
        calls.append((dataset, split))
        return Source()

    monkeypatch.setattr(validation_data, "load_dataset", load_dataset)
    assert len(validation_data.ValidationDJSCCDataset("imagenette160")) == 1
    assert calls == [("imagenette160", "val")]
    with pytest.raises(TypeError):
        validation_data.ValidationDJSCCDataset("imagenette160", split="test")  # type: ignore[call-arg]


def test_checkpoint_selection_is_max_top1_with_earliest_exact_tie(tiny_validation):
    summaries = [
        fake_summary(epoch=0, correct=3, total=5),
        fake_summary(epoch=1, correct=4, total=5),
        fake_summary(epoch=2, correct=4, total=5),
    ]
    selection = validation.select_checkpoint_epoch(summaries, expected_epochs=3)
    assert selection["selected_epoch"] == 1
    assert selection["metric"] == "validation_top1_accuracy"
    assert selection["cross_seed_selection"] is False
    validation._validate_selection(selection)

    reversed_summaries = list(reversed(summaries))
    with pytest.raises(validation.W8ValidationHold, match="epoch order"):
        validation.select_checkpoint_epoch(reversed_summaries, expected_epochs=3)


def test_cross_seed_or_cross_ratio_selection_is_rejected_after_resigning(tiny_validation):
    summaries = [fake_summary(epoch=index, correct=3 + index, total=5) for index in range(3)]
    summaries[2]["train_seed"] = 1
    summaries[2]["channel_seed"] = 1
    summaries[2]["validation_channel_seed"] = 1
    summaries[2]["run_id"] = "w8-r_1_6-train1-channel1"
    body = dict(summaries[2])
    body.pop("summary_id")
    summaries[2]["summary_id"] = canonical_sha256(body)
    with pytest.raises(validation.W8ValidationHold, match="cross-seed"):
        validation.select_checkpoint_epoch(summaries, expected_epochs=3)


def test_summary_mutations_fail_closed(tiny_validation):
    value = fake_summary(epoch=0, correct=3, total=5)
    mutations = [
        ("denominator", lambda item: item.__setitem__("n_total", 4)),
        ("noise", lambda item: item.__setitem__("validation_channel_seed", 1)),
        ("forbidden", lambda item: item.__setitem__("forbidden_selection_inputs", [])),
    ]
    for label, mutation in mutations:
        candidate = copy.deepcopy(value)
        mutation(candidate)
        body = dict(candidate)
        body.pop("summary_id")
        candidate["summary_id"] = canonical_sha256(body)
        with pytest.raises(validation.W8ValidationHold, match="denominator|seed|inputs"):
            validation._validate_summary(candidate)


def test_selection_never_uses_psnr_papr_or_reconstruction_loss(tiny_validation):
    selection = validation.select_checkpoint_epoch(
        [fake_summary(epoch=0, correct=4, total=5), fake_summary(epoch=1, correct=3, total=5), fake_summary(epoch=2, correct=3, total=5)],
        expected_epochs=3,
    )
    assert selection["psnr_selected"] is False
    assert selection["papr_selected"] is False
    assert selection["reconstruction_loss_selected"] is False
    assert selection["n_total"] == 5
    assert W8_VALIDATION_SAMPLE_COUNT == 1000
