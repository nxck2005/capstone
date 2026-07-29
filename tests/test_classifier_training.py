"""Bounded deterministic trainer and validation tests."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from config.params import REPO_ROOT, get
from config.run_config import load_reference_classifier_config
from training.reference_classifier import ReferenceClassifierTrainer, learning_rate_for_epoch, validate
import training.reference_classifier as trainer_module


class TinyDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self) -> None:
        self.rows = [(torch.tensor([float(index), 1.0]), index % 2) for index in range(6)]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.rows[index]


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(2, 2)
        self.total_parameter_count = sum(item.numel() for item in self.parameters())
        self.trainable_parameter_count = self.total_parameter_count

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layer(value)


def _config():
    return load_reference_classifier_config(
        REPO_ROOT / get("config.dir") / "reference-classifier-clean.yaml",
        dataset="cifar10",
    )


def _trainer() -> ReferenceClassifierTrainer:
    torch.manual_seed(123)
    return ReferenceClassifierTrainer(_config(), model=TinyModel())


def test_exact_zero_based_scheduler_boundaries():
    assert learning_rate_for_epoch(0) == pytest.approx(0.01)
    assert learning_rate_for_epoch(4) == pytest.approx(0.1)
    assert learning_rate_for_epoch(5) == pytest.approx(0.1)
    assert learning_rate_for_epoch(99) == pytest.approx(0.0)


def test_validation_uses_integer_counts_and_restores_training_mode():
    model = TinyModel().train()
    result = validate(model, TinyDataset(), batch_size=2, device="cpu")

    assert result.n_total == 6
    assert 0 <= result.n_correct <= result.n_total
    assert result.top1_accuracy == result.n_correct / result.n_total
    assert model.training


def test_training_order_learning_rates_and_exact_tie_rule():
    trainer = _trainer()
    data = TinyDataset()
    first = trainer.train_epoch(0, data, batch_size=2, total_epochs=2)
    trainer.validate_epoch(0, data, batch_size=2)
    best = trainer.state.best_epoch
    trainer.validate_epoch(1, data, batch_size=2)

    assert first["sample_order"] == first["sample_order"]
    assert first["lr"] == pytest.approx(0.01)
    assert trainer.state.validation_history[0]["top1_accuracy"] == trainer.state.validation_history[1]["top1_accuracy"]
    assert trainer.state.best_epoch == best


def test_unbounded_direct_epoch_helpers_are_irreversibly_smoke_only(tmp_path):
    trainer = _trainer()
    data = TinyDataset()

    trainer.train_epoch(0, data, batch_size=2, total_epochs=2)
    trainer.validate_epoch(0, data, batch_size=2)
    checkpoint = tmp_path / "direct-unbounded.pt"
    trainer.save_checkpoint(checkpoint)

    payload = torch.load(checkpoint, weights_only=False)
    assert payload["execution_mode"] == "smoke"
    assert payload["full_run_requested"] is False
    assert payload["run_complete"] is False
    assert payload["g1_eligible"] is False
    assert payload["lineage_g1_eligible"] is False
    with pytest.raises(ValueError, match="cannot resume smoke checkpoint in full mode"):
        _trainer().resume(checkpoint, execution_mode="full")
    with pytest.raises(ValueError, match="cannot be promoted to full mode"):
        trainer.run_epochs(
            final_epoch=int(get("reference_classifier.epochs")) - 1,
            checkpoint_dir=tmp_path / "forbidden",
            execution_mode="full",
            full_run_requested=True,
        )


@pytest.mark.parametrize("argument", ["training_dataset", "validation_dataset"])
def test_full_run_rejects_external_datasets_before_state_or_artifact_mutation(tmp_path, argument):
    trainer = _trainer()
    kwargs = {argument: TinyDataset()}

    with pytest.raises(ValueError, match="constructs its .* dataset internally"):
        trainer.run_epochs(
            final_epoch=int(get("reference_classifier.epochs")) - 1,
            checkpoint_dir=tmp_path / "checkpoints",
            execution_mode="full",
            full_run_requested=True,
            **kwargs,
        )

    assert trainer.execution_mode is None
    assert trainer.state.completed_epoch == -1
    assert trainer.state.training_history == []
    assert trainer.state.validation_history == []
    assert not (tmp_path / "checkpoints").exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"smoke_steps": 1},
        {"smoke_val_batches": 1},
        {"final_epoch": 0},
    ],
)
def test_full_run_rejects_result_affecting_overrides_before_work(tmp_path, kwargs):
    trainer = _trainer()
    run_kwargs = {
        "final_epoch": int(get("reference_classifier.epochs")) - 1,
        "checkpoint_dir": tmp_path / "checkpoints",
        "execution_mode": "full",
        "full_run_requested": True,
    }
    run_kwargs.update(kwargs)

    with pytest.raises(ValueError):
        trainer.run_epochs(**run_kwargs)

    assert trainer.execution_mode is None
    assert trainer.state.completed_epoch == -1
    assert not (tmp_path / "checkpoints").exists()


def test_full_run_constructs_only_module_owned_production_views(monkeypatch, tmp_path):
    original_get = trainer_module.get
    constructors: list[str] = []

    def tiny_schedule(key: str):
        if key in {
            "reference_classifier.epochs",
            "reference_classifier.validation_every_epochs",
            "compute.checkpoint_every_epochs",
        }:
            return 1
        if key == "reference_classifier.batch_size":
            return 2
        return original_get(key)

    def training_view(*_args):
        constructors.append("training")
        return TinyDataset()

    def validation_view(*_args):
        constructors.append("validation")
        return TinyDataset()

    monkeypatch.setattr(trainer_module, "get", tiny_schedule)
    monkeypatch.setattr(trainer_module, "TrainingClassifierDataset", training_view)
    monkeypatch.setattr(trainer_module, "ValidationClassifierDataset", validation_view)
    trainer = _trainer()

    records = trainer.run_epochs(
        final_epoch=0,
        checkpoint_dir=tmp_path / "checkpoints",
        execution_mode="full",
        full_run_requested=True,
        run_complete=True,
        g1_eligible=True,
    )

    assert constructors == ["training", "validation"]
    assert records[-1].path.exists()
    payload = torch.load(records[-1].path, weights_only=False)
    assert payload["execution_mode"] == "full"
    assert payload["full_run_requested"] is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"smoke_steps": 1, "smoke_val_batches": 1, "run_complete": True},
        {"smoke_steps": 1, "smoke_val_batches": 1, "g1_eligible": True},
        {
            "execution_mode": "smoke",
            "smoke_steps": 1,
            "smoke_val_batches": 1,
            "run_complete": True,
        },
        {
            "execution_mode": "full",
            "full_run_requested": True,
            "smoke_steps": 1,
            "smoke_val_batches": 1,
        },
        {"execution_mode": "full", "full_run_requested": False},
    ],
    ids=[
        "smoke-steps-with-completion",
        "smoke-bounds-with-eligibility",
        "smoke-mode-with-completion",
        "full-mode-with-smoke-bounds",
        "full-mode-without-explicit-request",
    ],
)
def test_run_epochs_rejects_contradictory_lineage_before_training(tmp_path, kwargs):
    trainer = _trainer()

    with pytest.raises(ValueError):
        trainer.run_epochs(
            final_epoch=0,
            checkpoint_dir=tmp_path / "checkpoints",
            training_dataset=TinyDataset(),
            validation_dataset=TinyDataset(),
            **kwargs,
        )

    assert trainer.state.completed_epoch == -1
    assert not (tmp_path / "checkpoints").exists()
