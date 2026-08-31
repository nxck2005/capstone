"""W8 checkpoint, resume and AM-91 accounting regressions (all synthetic)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

import training.w8_final as w8_training
from training.deterministic_core import canonical_bytes, state_tree_sha256
from training.w8_final import W8Hold, checkpoint_state_digest
from tests.w8_hardening_fixtures import TinyDJSCC, TinyW8Dataset, tiny_config, tiny_trainer


@pytest.fixture(autouse=True)
def _tiny_checkpoint_model(monkeypatch):
    # Payload validation reconstructs the configured architecture.  A tiny
    # fixture keeps these checks bounded and does not execute a real W8 epoch.
    monkeypatch.setattr(w8_training, "build_djscc", lambda *_args, **_kwargs: TinyDJSCC())


def _run_epoch_and_save(trainer, epoch: int, count: int = 5):
    record = trainer.train_epoch(epoch, TinyW8Dataset(epoch, count))
    return record, trainer.save_checkpoint(record)


def test_synthetic_smoke_is_explicitly_ineligible_and_resume_is_exact(tmp_path: Path):
    uninterrupted = tiny_trainer(tmp_path / "uninterrupted")
    _run_epoch_and_save(uninterrupted, 0)
    _run_epoch_and_save(uninterrupted, 1)

    seam_root = tmp_path / "seam"
    first = tiny_trainer(seam_root)
    _, epoch0 = _run_epoch_and_save(first, 0)
    resumed = tiny_trainer(seam_root)
    resumed.resume()
    _, epoch1 = _run_epoch_and_save(resumed, 1)

    assert checkpoint_state_digest(resumed) == checkpoint_state_digest(uninterrupted)
    assert resumed.completed_epoch == 1
    assert resumed.global_optimizer_step == uninterrupted.global_optimizer_step == 2
    assert epoch1["predecessor_checkpoint_id"] == epoch0["checkpoint_id"]
    payload = torch.load(
        seam_root / epoch1["checkpoint_path"], map_location="cpu", weights_only=False
    )
    assert payload["artifact_role"] == "W8_NON_SCIENTIFIC_SMOKE_CHECKPOINT"
    assert payload["eligibility"]["w8_eligibility"] == "NOT_ELIGIBLE_FOR_W8_RESULT"
    assert resumed.initialization["predecessor_checkpoint_id"] is None
    assert "initial_model_state_sha256" in resumed.initialization


def test_resume_restores_optimizer_scheduler_and_scaler_state(tmp_path: Path):
    class FakeScaler:
        def __init__(self, scale: float) -> None:
            self.scale_value = scale

        def get_scale(self) -> float:
            return self.scale_value

        def scale(self, loss: torch.Tensor) -> torch.Tensor:
            return loss * self.scale_value

        def unscale_(self, optimizer) -> None:
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.div_(self.scale_value)

        def step(self, optimizer) -> None:
            optimizer.step()

        def update(self) -> None:
            pass

        def state_dict(self) -> dict[str, float]:
            return {"scale": self.scale_value}

        def load_state_dict(self, value) -> None:
            assert set(value) == {"scale"}
            self.scale_value = float(value["scale"])

    root = tmp_path / "scaler"
    trainer = tiny_trainer(root)
    trainer.scaler = FakeScaler(17.0)  # type: ignore[assignment]
    record, _sidecar = _run_epoch_and_save(trainer, 0)
    assert record["optimizer_steps"] == 1

    resumed = tiny_trainer(root)
    resumed.scaler = FakeScaler(1.0)  # type: ignore[assignment]
    resumed.resume()
    assert resumed.completed_epoch == 0
    assert resumed.scheduler.completed_epoch == 0
    assert resumed.global_optimizer_step == 1
    assert resumed.scaler.scale_value == 17.0  # type: ignore[union-attr]
    assert resumed.optimizer.param_groups[0]["lr"] == record["lr"]


def test_corrupt_latest_holds_without_older_fallback(tmp_path: Path):
    root = tmp_path / "corrupt"
    trainer = tiny_trainer(root)
    _run_epoch_and_save(trainer, 0)
    _record, latest = _run_epoch_and_save(trainer, 1)
    checkpoint = root / latest["checkpoint_path"]
    checkpoint.write_bytes(checkpoint.read_bytes()[:17])
    with pytest.raises(W8Hold, match="byte length|SHA-256|cannot be loaded"):
        tiny_trainer(root).resume()
    assert tiny_trainer(root).completed_epoch == -1


def test_incomplete_next_epoch_suffix_is_replayed_from_authenticated_latest(tmp_path: Path):
    root = tmp_path / "suffix"
    trainer = tiny_trainer(root)
    _run_epoch_and_save(trainer, 0)
    # A crash after record publication but before the next latest pointer is a
    # replayable suffix because epoch zero remains authenticated.
    (root / "epochs/epoch-0001.json").write_text("{}", encoding="ascii")
    resumed = tiny_trainer(root)
    resumed.resume()
    assert resumed.completed_epoch == 0
    assert not (root / "epochs/epoch-0001.json").exists()
    _run_epoch_and_save(resumed, 1)
    assert resumed.completed_epoch == 1


def test_incomplete_genesis_suffix_is_explicitly_discarded_and_replayed(tmp_path: Path):
    root = tmp_path / "genesis-suffix"
    writer = tiny_trainer(root)
    record = writer.train_epoch(0, TinyW8Dataset(0, count=5))
    (root / "epochs").mkdir(parents=True)
    (root / "checkpoints").mkdir(parents=True)
    (root / "epochs/epoch-0000.json").write_bytes(canonical_bytes(record))
    (root / "checkpoints/epoch-0000.pt").write_bytes(b"crash-left checkpoint")
    (root / "checkpoints/.epoch-0000.pt.crash.tmp").write_bytes(b"crash-left temp")

    resumed = tiny_trainer(root)
    resumed.discard_unpublished_genesis_suffix()
    assert not list((root / "epochs").iterdir())
    assert not list((root / "checkpoints").iterdir())
    _run_epoch_and_save(resumed, 0)
    assert resumed.completed_epoch == 0


def test_foreign_w8_runtime_is_not_a_resume_source(tmp_path: Path):
    source_root = tmp_path / "source"
    source = tiny_trainer(source_root, run_id="w8-r1-source")
    _run_epoch_and_save(source, 0)
    foreign = tiny_trainer(source_root, run_id="w8-r1-foreign")
    with pytest.raises(W8Hold, match="sidecar run|run differs"):
        foreign.resume()


def test_production_partial_batch_and_duplicate_accounting(monkeypatch, tmp_path: Path):
    # The constants are locally projected only to this synthetic fixture.  No
    # production denominator or result artifact is changed by the test.
    trainer = tiny_trainer(tmp_path / "partial", role="W8_NON_SCIENTIFIC_SMOKE")
    record = trainer.train_epoch(0, TinyW8Dataset(0, count=35))
    assert record["samples"] == record["expected_samples"] == 35
    assert record["microbatches"] == record["expected_microbatches"] == 2
    assert record["final_physical_batch"] == 3
    assert record["optimizer_step_opportunities"] == 2
    assert record["optimizer_steps"] + record["grad_scaler_skips"] == 2

    duplicate = tiny_trainer(tmp_path / "duplicate", role="W8_NON_SCIENTIFIC_SMOKE")
    with pytest.raises(W8Hold, match="duplicated"):
        duplicate.train_epoch(0, TinyW8Dataset(0, count=35, duplicate=True))


class _UnitTestScaler:
    """CPU-compatible GradScaler semantic double for the AM-91 regression."""

    def __init__(self, *, target: torch.nn.Parameter, inject: float | None = None) -> None:
        self.target = target
        self.inject = inject
        self.scale_value = 64.0
        self.skipped: bool | None = None
        self.after_unscale: torch.Tensor | None = None
        self.at_step: torch.Tensor | None = None

    def get_scale(self) -> float:
        return self.scale_value

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.scale_value

    def unscale_(self, optimizer) -> None:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.div_(self.scale_value)
        if self.inject is not None:
            assert self.target.grad is not None
            self.target.grad.view(-1)[0] = self.inject
        assert self.target.grad is not None
        self.after_unscale = self.target.grad.detach().clone()

    def step(self, optimizer) -> None:
        assert self.target.grad is not None
        self.at_step = self.target.grad.detach().clone()
        finite = all(
            torch.isfinite(parameter.grad).all().item()
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        )
        self.skipped = not finite
        if finite:
            optimizer.step()

    def update(self) -> None:
        if self.skipped:
            self.scale_value /= 2


def test_gradscaler_inf_in_any_optimizer_owned_region_skips_update(monkeypatch, tmp_path: Path):
    trainer = tiny_trainer(tmp_path / "inf")
    target = trainer.model.decoder.ingress.weight
    before = {name: parameter.detach().clone() for name, parameter in trainer.model.named_parameters()}
    trainer.scaler = _UnitTestScaler(target=target, inject=float("inf"))  # type: ignore[assignment]
    record = trainer.train_epoch(0, TinyW8Dataset(0, count=5))
    assert record["gradient_checks"]["all_optimizer_gradients_finite"] is False
    assert record["optimizer_steps"] == record["global_optimizer_step"] == 0
    assert record["grad_scaler_skips"] == 1
    assert trainer.scaler.skipped is True  # type: ignore[union-attr]
    assert trainer.scaler.get_scale() == 32.0  # type: ignore[union-attr]
    for name, parameter in trainer.model.named_parameters():
        assert torch.equal(parameter, before[name]), name


def test_gradscaler_finite_update_divides_by_actual_partial_denominator(tmp_path: Path):
    trainer = tiny_trainer(tmp_path / "finite")
    target = trainer.model.decoder.ingress.weight
    before = target.detach().clone()
    scaler = _UnitTestScaler(target=target)
    trainer.scaler = scaler  # type: ignore[assignment]
    record = trainer.train_epoch(0, TinyW8Dataset(0, count=5))
    assert record["optimizer_steps"] == record["global_optimizer_step"] == 1
    assert record["grad_scaler_skips"] == 0
    assert not torch.equal(target, before)
    assert scaler.after_unscale is not None and scaler.at_step is not None
    assert torch.allclose(scaler.at_step, scaler.after_unscale / 5, rtol=0, atol=1e-8)
