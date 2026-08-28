"""W7 validation/common-noise/checkpoint-selection behavioral regressions."""

from __future__ import annotations

import copy
import math
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import data.djscc_validation as validation_data
import evaluation.w7_validation as validation
import training.w7_g4 as w7_training
from training.deterministic_core import canonical_sha256
from training.w7_g4 import W7Hold
from tests.w7_hardening_fixtures import (
    TinyDJSCC,
    TinyValidationDataset,
    TinyW7Dataset,
    tiny_trainer,
    validation_get,
)


@pytest.fixture(autouse=True)
def _tiny_validation_environment(monkeypatch):
    actual_get = validation.get
    monkeypatch.setattr(validation, "get", validation_get(actual_get, 5))
    monkeypatch.setattr(
        validation,
        "ValidationDJSCCDataset",
        lambda dataset, repo_root=None: TinyValidationDataset(dataset, repo_root=repo_root, count=5),
    )
    monkeypatch.setattr(w7_training, "build_djscc", lambda *_args, **_kwargs: TinyDJSCC())


def _ready_trainer(root: Path, *, lambda_value: float = 0.3):
    trainer = tiny_trainer(root, lambda_value=lambda_value, epochs=3)
    trainer.completed_epoch = 0
    trainer.scheduler.completed_epoch = 0
    return trainer


def test_validation_dataset_has_no_test_split_route(monkeypatch):
    calls: list[tuple[str, str]] = []

    class _Source:
        def __len__(self):
            return 2

        def source_sample(self, index):
            return SimpleNamespace(stable_sample_id=f"id-{index}")

    def fake_load(dataset, split, repo_root=None):
        del repo_root
        calls.append((dataset, split))
        return _Source()

    monkeypatch.setattr(validation_data, "load_dataset", fake_load)
    dataset = validation_data.ValidationDJSCCDataset("imagenette160")
    assert len(dataset) == 2
    assert calls == [("imagenette160", "val")]
    with pytest.raises(TypeError):
        validation_data.ValidationDJSCCDataset("imagenette160", split="test")  # type: ignore[call-arg]


def test_validation_complete_denominator_stable_order_and_count_recomputable(tmp_path: Path):
    trainer = _ready_trainer(tmp_path)
    evaluated = validation.evaluate_validation(
        trainer, checkpoint_id="c" * 64, batch_size=2, retain_rows=True
    )
    summary = evaluated.summary
    rows = list(evaluated.rows)
    assert summary["n_total"] == len(rows) == 5
    assert [row["stable_sample_id"] for row in rows] == sorted(row["stable_sample_id"] for row in rows)
    assert len({row["stable_sample_id"] for row in rows}) == 5
    correct = sum(int(row["prediction"] == row["label"]) for row in rows)
    assert summary["n_correct"] == correct
    assert summary["top1_accuracy"] == correct / len(rows)
    assert summary["row_digest"] == canonical_sha256(rows)


def test_validation_common_noise_is_lambda_and_checkpoint_independent(tmp_path: Path):
    first = _ready_trainer(tmp_path / "lambda-0", lambda_value=0.0)
    second = _ready_trainer(tmp_path / "lambda-3", lambda_value=3.0)
    a = validation.evaluate_validation(first, checkpoint_id="1" * 64, retain_rows=True)
    b = validation.evaluate_validation(second, checkpoint_id="2" * 64, retain_rows=True)
    assert [row["noise_id"] for row in a.rows] == [row["noise_id"] for row in b.rows]
    assert a.summary["noise_id_digest"] == b.summary["noise_id_digest"]
    assert a.summary["noise_policy_hash"] == b.summary["noise_policy_hash"]


def test_ambient_python_numpy_torch_rng_cannot_change_keyed_validation_noise(tmp_path: Path):
    trainer = _ready_trainer(tmp_path)
    first = validation.evaluate_validation(trainer, checkpoint_id="a" * 64, retain_rows=True)
    random.seed(19)
    np.random.seed(23)
    torch.manual_seed(29)
    _ = [random.random() for _ in range(17)]
    _ = np.random.default_rng(31).normal(size=200)
    _ = torch.randn(200)
    second = validation.evaluate_validation(trainer, checkpoint_id="b" * 64, retain_rows=True)
    assert [row["noise_id"] for row in first.rows] == [row["noise_id"] for row in second.rows]
    assert [row["prediction"] for row in first.rows] == [row["prediction"] for row in second.rows]
    assert [row["psnr_db"] for row in first.rows] == [row["psnr_db"] for row in second.rows]


def test_psnr_is_independently_recomputable_from_frozen_definition(tmp_path: Path):
    trainer = _ready_trainer(tmp_path)
    evaluated = validation.evaluate_validation(trainer, checkpoint_id="d" * 64, retain_rows=True)
    for row in evaluated.rows:
        expected = "inf" if row["mse"] == 0 else 10.0 * math.log10(1.0 / row["mse"])
        if expected == "inf":
            assert row["psnr_db"] == "inf"
        else:
            assert row["psnr_db"] == pytest.approx(expected, abs=1e-10)


def _summary(epoch: int, correct: int, *, total: int = 5) -> dict:
    body = {
        "schema_version": 1,
        "artifact_role": "W7_VALIDATION_EPOCH_SUMMARY",
        "epoch": epoch,
        "checkpoint_id": f"{epoch + 1:064x}",
        "n_correct": correct,
        "n_total": total,
        "top1_accuracy": correct / total,
        "prediction_digest": canonical_sha256(["pred", epoch]),
        "evaluation_config_hash": canonical_sha256(["eval", epoch]),
        "noise_policy": "keyed_channel_noise_same_per_image_across_lambda",
        "noise_policy_hash": canonical_sha256(["policy", epoch]),
        "noise_id_digest": canonical_sha256(["noise", epoch]),
        "row_digest": canonical_sha256(["rows", epoch]),
    }
    body["summary_id"] = canonical_sha256(body)
    return body


def test_earliest_epoch_tie_break_and_resigned_denominator_mutation_hold(monkeypatch):
    actual_get = validation.get
    monkeypatch.setattr(validation, "get", validation_get(actual_get, 5))
    summaries = [_summary(0, 3), _summary(1, 4), _summary(2, 4)]
    selection = validation.select_checkpoint_epoch(list(reversed(summaries)), expected_epochs=3)
    assert selection["selected_epoch"] == 1
    assert selection["selected_checkpoint_id"] == summaries[1]["checkpoint_id"]

    mutated = copy.deepcopy(summaries)
    mutated[1]["n_total"] = 6
    mutated[1]["top1_accuracy"] = mutated[1]["n_correct"] / 6
    mutated[1].pop("summary_id")
    mutated[1]["summary_id"] = canonical_sha256(mutated[1])
    with pytest.raises(W7Hold, match="denominator"):
        validation.select_checkpoint_epoch(mutated, expected_epochs=3)


def test_selected_checkpoint_is_independently_reloaded_and_authenticated(tmp_path: Path):
    trainer = tiny_trainer(tmp_path, epochs=2)
    summaries = []
    for epoch in range(2):
        record = trainer.train_epoch(epoch, TinyW7Dataset(epoch, count=5))
        sidecar = trainer.save_checkpoint(record)
        summaries.append(
            validation.evaluate_validation(
                trainer, checkpoint_id=sidecar["checkpoint_id"], retain_rows=False
            ).summary
        )
    selection = validation.select_checkpoint_epoch(summaries, expected_epochs=2)
    fresh = tiny_trainer(tmp_path, epochs=2)
    result = validation.selected_checkpoint_result(fresh, selection=selection)
    assert result["checkpoint_id"] == selection["selected_checkpoint_id"]
    assert result["checkpoint_epoch"] == selection["selected_epoch"]
    assert result["calibration_validation"] == summaries[selection["selected_epoch"]]
    assert result["psnr_evaluation"]["denominator"] == 5
    assert result["protected_counters"] == {
        "w7_candidate_results": 0,
        "learned_test_inference": 0,
        "test_model_facing_access": 0,
    }

    wrong = dict(selection)
    wrong["selected_checkpoint_id"] = "f" * 64
    with pytest.raises(W7Hold, match="checkpoint ID"):
        validation.selected_checkpoint_result(tiny_trainer(tmp_path, epochs=2), selection=wrong)

    other_epoch = 1 - int(selection["selected_epoch"])
    wrong_epoch = dict(selection)
    wrong_epoch["selected_epoch"] = other_epoch
    wrong_epoch["selected_checkpoint_id"] = summaries[other_epoch]["checkpoint_id"]
    wrong_epoch["n_correct"] = -1
    wrong_epoch["top1_accuracy"] = -0.2
    with pytest.raises(W7Hold, match="reauthentication"):
        validation.selected_checkpoint_result(
            tiny_trainer(tmp_path, epochs=2), selection=wrong_epoch
        )
