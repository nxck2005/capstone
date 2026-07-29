"""Atomic checkpoint and direct-epoch resume tests."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from tests.test_classifier_training import TinyDataset, _trainer


def _tensor_state(trainer):
    return {key: value.detach().clone() for key, value in trainer.model.state_dict().items()}


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _equal(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, (tuple, list)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(
            _equal(value, item) for value, item in zip(left, right, strict=True)
        )
    return left == right


def _trainer_snapshot(trainer):
    return {
        "model": _tensor_state(trainer),
        "optimizer": copy.deepcopy(trainer.optimizer.state_dict()),
        "scheduler": trainer.scheduler_state,
        "state": copy.deepcopy(trainer.state),
        "execution_mode": trainer.execution_mode,
        "smoke_steps": trainer.smoke_steps,
        "smoke_val_batches": trainer.smoke_val_batches,
        "full_run_requested": trainer.full_run_requested,
        "run_complete_requested": trainer.run_complete_requested,
        "g1_eligible_requested": trainer.g1_eligible_requested,
    }


def _assert_snapshot_unchanged(trainer, snapshot):
    assert _equal(_tensor_state(trainer), snapshot["model"])
    assert _equal(trainer.optimizer.state_dict(), snapshot["optimizer"])
    assert trainer.scheduler_state == snapshot["scheduler"]
    assert trainer.state == snapshot["state"]
    assert trainer.execution_mode == snapshot["execution_mode"]
    assert trainer.smoke_steps == snapshot["smoke_steps"]
    assert trainer.smoke_val_batches == snapshot["smoke_val_batches"]
    assert trainer.full_run_requested == snapshot["full_run_requested"]
    assert trainer.run_complete_requested == snapshot["run_complete_requested"]
    assert trainer.g1_eligible_requested == snapshot["g1_eligible_requested"]


def test_interrupted_resume_matches_uninterrupted_training(tmp_path: Path):
    data = TinyDataset()
    uninterrupted = _trainer()
    uninterrupted.train_epoch(0, data, batch_size=2, total_epochs=2)
    uninterrupted.validate_epoch(0, data, batch_size=2)
    uninterrupted.train_epoch(1, data, batch_size=2, total_epochs=2)
    uninterrupted.validate_epoch(1, data, batch_size=2)

    interrupted = _trainer()
    interrupted.train_epoch(0, data, batch_size=2, total_epochs=2)
    interrupted.validate_epoch(0, data, batch_size=2)
    path = tmp_path / "checkpoint.pt"
    checkpoint_id = interrupted.save_checkpoint(path)
    resumed = _trainer()
    resumed.resume(path)
    resumed.train_epoch(1, data, batch_size=2, total_epochs=2)
    resumed.validate_epoch(1, data, batch_size=2)

    assert len(checkpoint_id) == 64
    assert interrupted.state.training_history[0]["sample_order"] == resumed.state.training_history[0]["sample_order"]
    assert uninterrupted.state.training_history == resumed.state.training_history
    assert uninterrupted.state.validation_history == resumed.state.validation_history
    assert uninterrupted.state.best_epoch == resumed.state.best_epoch
    assert uninterrupted.state.best_validation_top1 == resumed.state.best_validation_top1
    assert all(torch.equal(value, _tensor_state(resumed)[key]) for key, value in _tensor_state(uninterrupted).items())
    assert _equal(uninterrupted.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert uninterrupted.scheduler_state == resumed.scheduler_state


def test_run_epochs_checkpoints_and_marks_smoke_ineligible(tmp_path: Path):
    trainer = _trainer()
    records = trainer.run_epochs(
        final_epoch=1,
        checkpoint_dir=tmp_path / "checkpoints",
        training_dataset=TinyDataset(),
        validation_dataset=TinyDataset(),
        smoke_steps=1,
        smoke_val_batches=1,
    )

    assert [record.epoch for record in records] == [0, 1]
    payload = torch.load(records[-1].path, weights_only=False)
    required = {
        "checkpoint_schema_version", "model_state", "optimizer_state", "scheduler_state",
        "completed_epoch", "next_epoch", "best_validation_top1", "best_epoch",
        "resolved_run_config", "config_hash", "dataset", "dataset_version",
        "split_manifest_hash", "classifier_variant", "architecture", "train_seed",
        "model_total_parameter_count", "model_trainable_parameter_count", "training_history",
        "execution_mode", "smoke_steps", "smoke_val_batches", "full_run_requested",
        "run_complete", "g1_eligible", "lineage_g1_eligible",
    }
    assert required <= payload.keys()
    assert payload["execution_mode"] == "smoke"
    assert payload["smoke_steps"] == 1
    assert payload["smoke_val_batches"] == 1
    assert payload["run_complete"] is False
    assert payload["g1_eligible"] is False
    assert payload["lineage_g1_eligible"] is False
    assert records[-1].checkpoint_id == __import__("hashlib").sha256(records[-1].path.read_bytes()).hexdigest()
    assert not list((tmp_path / "checkpoints").glob("*.tmp"))


def test_checkpoint_mismatch_rejects_before_model_mutation(tmp_path: Path):
    data = TinyDataset()
    trainer = _trainer()
    trainer.train_epoch(0, data, batch_size=2, total_epochs=2)
    path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(path)
    payload = torch.load(path, weights_only=False)
    payload["dataset"] = "stl10"
    torch.save(payload, path)
    target = _trainer()
    before = _tensor_state(target)

    with pytest.raises(ValueError, match="dataset mismatch"):
        target.resume(path)

    assert all(torch.equal(value, _tensor_state(target)[key]) for key, value in before.items())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["model_state"].pop("layer.weight"),
        lambda payload: payload["model_state"].update({"unexpected": torch.zeros(1)}),
        lambda payload: payload["model_state"].update({"layer.weight": torch.zeros(3, 2)}),
        lambda payload: payload.update({"optimizer_state": {}}),
        lambda payload: payload.update({"scheduler_state": {"completed_epoch": "bad"}}),
        lambda payload: payload.update({"training_history": "malformed"}),
        lambda payload: payload["resolved_run_config"]["resolved"].update({"k": 999}),
        lambda payload: payload.update({"next_epoch": 99}),
    ],
    ids=[
        "missing-model-key",
        "unexpected-model-key",
        "changed-tensor-shape",
        "invalid-optimizer",
        "invalid-scheduler",
        "malformed-history",
        "different-resolved-config",
        "invalid-epoch-counters",
    ],
)
def test_rejected_checkpoint_is_transactional(tmp_path: Path, mutation):
    source = _trainer()
    source.train_epoch(0, TinyDataset(), batch_size=2, total_epochs=2)
    source.validate_epoch(0, TinyDataset(), batch_size=2)
    path = tmp_path / "checkpoint.pt"
    source.save_checkpoint(path)
    payload = torch.load(path, weights_only=False)
    mutation(payload)
    torch.save(payload, path)

    target = _trainer()
    target.train_epoch(0, TinyDataset(), batch_size=2, total_epochs=2)
    target.validate_epoch(0, TinyDataset(), batch_size=2)
    before = _trainer_snapshot(target)

    with pytest.raises(ValueError):
        target.resume(path)

    _assert_snapshot_unchanged(target, before)


def test_smoke_checkpoint_cannot_resume_in_full_mode(tmp_path: Path):
    source = _trainer()
    source.run_epochs(
        final_epoch=0,
        checkpoint_dir=tmp_path / "smoke",
        training_dataset=TinyDataset(),
        validation_dataset=TinyDataset(),
        smoke_steps=1,
        smoke_val_batches=1,
    )
    checkpoint = tmp_path / "smoke" / "epoch-0.pt"
    target = _trainer()
    before = _trainer_snapshot(target)

    with pytest.raises(ValueError, match="cannot resume smoke checkpoint in full mode"):
        target.resume(checkpoint, execution_mode="full")

    _assert_snapshot_unchanged(target, before)


def test_direct_bounded_training_irreversibly_creates_smoke_lineage(tmp_path: Path):
    source = _trainer()
    source.train_epoch(0, TinyDataset(), batch_size=2, max_steps=1, total_epochs=2)
    checkpoint = tmp_path / "direct-smoke.pt"

    source.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, weights_only=False)

    assert payload["execution_mode"] == "smoke"
    assert payload["smoke_steps"] == 1
    assert payload["smoke_val_batches"] is None
    assert payload["full_run_requested"] is False
    assert payload["run_complete"] is False
    assert payload["g1_eligible"] is False
    source.train_epoch(1, TinyDataset(), batch_size=2, total_epochs=2)
    assert source.execution_mode == "smoke"
    with pytest.raises(ValueError, match="cannot be promoted to full mode"):
        source.run_epochs(
            final_epoch=2,
            checkpoint_dir=tmp_path / "forbidden-full",
            training_dataset=TinyDataset(),
            validation_dataset=TinyDataset(),
            execution_mode="full",
            full_run_requested=True,
        )
    with pytest.raises(ValueError, match="cannot resume smoke checkpoint in full mode"):
        _trainer().resume(checkpoint, execution_mode="full")


def test_direct_bounded_validation_downgrades_full_lineage(tmp_path: Path):
    source = _trainer()
    source.train_epoch(0, TinyDataset(), batch_size=2, total_epochs=2)
    source.validate_epoch(0, TinyDataset(), batch_size=2, max_batches=1)
    checkpoint = tmp_path / "bounded-validation.pt"

    source.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, weights_only=False)

    assert payload["execution_mode"] == "smoke"
    assert payload["smoke_steps"] is None
    assert payload["smoke_val_batches"] == 1
    assert payload["full_run_requested"] is False


def test_smoke_checkpoint_can_resume_in_smoke_mode(tmp_path: Path):
    source = _trainer()
    source.run_epochs(
        final_epoch=0,
        checkpoint_dir=tmp_path / "smoke",
        training_dataset=TinyDataset(),
        validation_dataset=TinyDataset(),
        smoke_steps=1,
        smoke_val_batches=1,
    )
    checkpoint = tmp_path / "smoke" / "epoch-0.pt"
    target = _trainer()

    target.resume(checkpoint, execution_mode="smoke")

    assert target.execution_mode == "smoke"
    assert target.smoke_steps == 1
    assert target.smoke_val_batches == 1
    assert target.state.completed_epoch == 0


def test_full_lineage_checkpoint_resumes_in_full_mode(tmp_path: Path):
    source = _trainer()
    source.train_epoch(0, TinyDataset(), batch_size=2, total_epochs=2)
    source.validate_epoch(0, TinyDataset(), batch_size=2)
    checkpoint = tmp_path / "full.pt"
    source.save_checkpoint(checkpoint)

    target = _trainer()
    target.resume(checkpoint, execution_mode="full")

    assert target.execution_mode == "full"
    assert target.full_run_requested is True
    assert target.state.completed_epoch == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("momentum", 0.123),
        ("weight_decay", 0.123),
        ("nesterov", True),
        ("dampening", 0.123),
        ("lr", 0.123),
    ],
)
def test_resume_rejects_optimizer_recipe_mismatch_transactionally(
    tmp_path: Path,
    field: str,
    value,
):
    source = _trainer()
    source.train_epoch(0, TinyDataset(), batch_size=2, total_epochs=2)
    checkpoint = tmp_path / f"bad-{field}.pt"
    source.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, weights_only=False)
    payload["optimizer_state"]["param_groups"][0][field] = value
    torch.save(payload, checkpoint)
    target = _trainer()
    before = _trainer_snapshot(target)

    with pytest.raises(ValueError, match=rf"optimizer {field} mismatch"):
        target.resume(checkpoint)

    _assert_snapshot_unchanged(target, before)
