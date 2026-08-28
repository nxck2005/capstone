"""Behavioral W7 trainer/resume/GradScaler regressions (tiny, non-scientific)."""

from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch

import training.w7_g4 as w7_training
from training.deterministic_core import apply_optimizer_update, state_tree_sha256
from training.w7_g4 import W7Hold, checkpoint_state_digest, keyed_training_complex_noise
from tests.w7_hardening_fixtures import TinyDJSCC, TinyW7Dataset, tiny_trainer


@pytest.fixture(autouse=True)
def _tiny_checkpoint_model(monkeypatch):
    # Payload authentication independently reconstructs the configured model.
    # Use the same deterministic tiny architecture, not the large production
    # fixture, while exercising the production checkpoint validator verbatim.
    monkeypatch.setattr(w7_training, "build_djscc", lambda *_args, **_kwargs: TinyDJSCC())


def _run_epoch_and_save(trainer, epoch: int, count: int = 5):
    record = trainer.train_epoch(epoch, TinyW7Dataset(epoch, count))
    sidecar = trainer.save_checkpoint(record)
    return record, sidecar


def test_complete_denominator_no_duplicate_or_drop_and_exact_step_arithmetic(tmp_path: Path):
    trainer = tiny_trainer(tmp_path, epochs=2)
    record = trainer.train_epoch(0, TinyW7Dataset(0, count=35))
    assert record["samples"] == record["expected_samples"] == record["stable_id_count"] == 35
    assert record["microbatches"] == 18
    # 32 samples in the first effective batch and a final partial group of 3.
    assert record["optimizer_steps"] == record["global_optimizer_step"] == 2
    assert record["grad_scaler_skips"] == 0

    duplicate = tiny_trainer(tmp_path / "duplicate", epochs=2)
    with pytest.raises(W7Hold, match="duplicated"):
        duplicate.train_epoch(0, TinyW7Dataset(0, count=5, duplicate=True))


def test_sample_weighted_accumulation_and_final_partial_match_full_batch_math(tmp_path: Path):
    trainer = tiny_trainer(tmp_path, epochs=2)
    initial = copy.deepcopy(trainer.model.state_dict())
    dataset = TinyW7Dataset(0, count=5)
    record = trainer.train_epoch(0, dataset)
    assert record["microbatches"] == 3
    assert record["optimizer_steps"] == 1
    assert record["samples"] == 5

    reference = TinyDJSCC()
    reference.load_state_dict(initial)
    optimizer = trainer._new_optimizer(reference)
    for group in optimizer.param_groups:
        group["lr"] = record["lr"]
    inputs = torch.stack([dataset[index][0] for index in range(len(dataset))])
    labels = torch.tensor([dataset[index][1] for index in range(len(dataset))])
    ids = [dataset[index][2] for index in range(len(dataset))]
    identities = trainer._training_noise_identities(ids, 0)
    noise = keyed_training_complex_noise(identities, int(trainer.config.resolved["k"]))
    output = reference(inputs, trainer.config.resolved["train_snr_db"], unit_noise=noise)
    loss = trainer.objective(output, labels, inputs)
    loss.total.backward()
    optimizer.step()

    assert record["total_loss"] == pytest.approx(float(loss.total.detach()), abs=1e-6)
    assert record["cross_entropy"] == pytest.approx(float(loss.cross_entropy.detach()), abs=1e-6)
    assert record["reconstruction_mse"] == pytest.approx(float(loss.reconstruction_mse.detach()), abs=1e-6)
    for name, value in trainer.model.state_dict().items():
        assert torch.allclose(value, reference.state_dict()[name], rtol=2e-6, atol=2e-7), name


def test_fresh_instance_resume_is_exact_and_preserves_latest_predecessor(tmp_path: Path):
    uninterrupted = tiny_trainer(tmp_path / "uninterrupted", epochs=3)
    _run_epoch_and_save(uninterrupted, 0)
    _run_epoch_and_save(uninterrupted, 1)

    seam_root = tmp_path / "seam"
    before = tiny_trainer(seam_root, epochs=3)
    _, epoch0 = _run_epoch_and_save(before, 0)

    # Perturb all ambient RNGs before constructing the fresh resume process.
    random.seed(987)
    np.random.seed(654)
    torch.manual_seed(321)
    resumed = tiny_trainer(seam_root, epochs=3)
    resumed.resume()
    _, epoch1 = _run_epoch_and_save(resumed, 1)

    assert checkpoint_state_digest(resumed) == checkpoint_state_digest(uninterrupted)
    assert state_tree_sha256(resumed.scaler.state_dict() if resumed.scaler else None) == state_tree_sha256(None)
    assert resumed.global_optimizer_step == uninterrupted.global_optimizer_step == 2
    assert resumed.completed_epoch == uninterrupted.completed_epoch == 1
    assert epoch1["predecessor_checkpoint_id"] == epoch0["checkpoint_id"]

    # A second fresh process must authenticate the complete resumed chain.
    second_fresh = tiny_trainer(seam_root, epochs=3)
    second_fresh.resume()
    assert checkpoint_state_digest(second_fresh) == checkpoint_state_digest(uninterrupted)
    assert second_fresh.global_optimizer_step == 2


def test_corrupt_or_truncated_latest_holds_without_older_fallback(tmp_path: Path):
    root = tmp_path / "runtime"
    trainer = tiny_trainer(root, epochs=3)
    _run_epoch_and_save(trainer, 0)
    _, latest = _run_epoch_and_save(trainer, 1)
    checkpoint = root / latest["checkpoint_path"]
    checkpoint.write_bytes(checkpoint.read_bytes()[:31])

    fresh = tiny_trainer(root, epochs=3)
    with pytest.raises(W7Hold, match="byte length|SHA-256|cannot be loaded"):
        fresh.resume()
    assert fresh.completed_epoch == -1
    assert fresh.global_optimizer_step == 0


def _rewrite_sidecar(root: Path, mutation) -> None:
    sidecar_path = root / "checkpoints/epoch-0000.sidecar.json"
    value = json.loads(sidecar_path.read_bytes())
    mutation(value)
    raw = w7_training.canonical_bytes(value)
    sidecar_path.write_bytes(raw)
    (root / "latest.json").write_bytes(raw)


@pytest.mark.parametrize(
    "label,mutation,match",
    [
        ("source", lambda value: value.__setitem__("source_commit", "c" * 40), "source commit"),
        ("gpu", lambda value: value.__setitem__("gpu_uuid", "GPU-foreign"), "GPU"),
        ("lambda", lambda value: value.__setitem__("lambda", 3.0), "lambda"),
        ("seed", lambda value: value.__setitem__("train_seed", 2), "train_seed"),
        ("config", lambda value: value.__setitem__("config_hash", "d" * 64), "config hash"),
        ("profile", lambda value: value.__setitem__("execution_profile_id", "local_4060_cu130"), "profile"),
    ],
)
def test_resigned_sidecar_lineage_mismatches_hold(tmp_path: Path, label, mutation, match):
    root = tmp_path / label
    trainer = tiny_trainer(root, epochs=2)
    _run_epoch_and_save(trainer, 0)
    _rewrite_sidecar(root, mutation)
    with pytest.raises(W7Hold, match=match):
        tiny_trainer(root, epochs=2).resume()


def test_resigned_checkpoint_predecessor_mismatch_holds(tmp_path: Path):
    root = tmp_path / "predecessor"
    trainer = tiny_trainer(root, epochs=3)
    _run_epoch_and_save(trainer, 0)
    _run_epoch_and_save(trainer, 1)

    sidecar_path = root / "checkpoints/epoch-0001.sidecar.json"
    sidecar = json.loads(sidecar_path.read_bytes())
    sidecar["predecessor_checkpoint_id"] = "e" * 64
    raw = w7_training.canonical_bytes(sidecar)
    sidecar_path.write_bytes(raw)
    (root / "latest.json").write_bytes(raw)
    with pytest.raises(W7Hold, match="predecessor chain"):
        tiny_trainer(root, epochs=3).resume()


class _UnitTestScaler:
    """CPU-compatible GradScaler semantic double with real optimizer stepping."""

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

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.div_(self.scale_value)
        if self.inject is not None:
            assert self.target.grad is not None
            self.target.grad.view(-1)[0] = self.inject
        assert self.target.grad is not None
        self.after_unscale = self.target.grad.detach().clone()

    def step(self, optimizer: torch.optim.Optimizer) -> None:
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


def test_w7_shared_decoder_nonfinite_is_optimizer_wide_skip_and_backoff(tmp_path: Path):
    trainer = tiny_trainer(tmp_path, epochs=2)
    target = trainer.model.decoder.ingress.weight
    named_ids = {
        id(parameter)
        for module in (
            trainer.model.encoder,
            trainer.model.decoder.reconstruction_head,
            trainer.model.decoder.task_head,
        )
        for parameter in module.parameters()
    }
    assert id(target) not in named_ids
    before = {name: parameter.detach().clone() for name, parameter in trainer.model.named_parameters()}
    scaler = _UnitTestScaler(target=target, inject=float("inf"))
    trainer.scaler = scaler  # type: ignore[assignment]

    record = trainer.train_epoch(0, TinyW7Dataset(0, count=5))
    assert record["gradient_checks"]["all_optimizer_gradients_finite"] is False
    assert record["optimizer_steps"] == record["global_optimizer_step"] == 0
    assert record["grad_scaler_skips"] == 1
    assert scaler.skipped is True
    assert scaler.get_scale() == 32.0
    for name, parameter in trainer.model.named_parameters():
        assert torch.equal(parameter, before[name]), name


def test_w7_finite_gradients_genuinely_update_and_denominator_is_after_unscale(tmp_path: Path):
    trainer = tiny_trainer(tmp_path, epochs=2)
    target = trainer.model.decoder.ingress.weight
    before = target.detach().clone()
    scaler = _UnitTestScaler(target=target)
    trainer.scaler = scaler  # type: ignore[assignment]

    record = trainer.train_epoch(0, TinyW7Dataset(0, count=5))
    assert record["optimizer_steps"] == record["global_optimizer_step"] == 1
    assert record["grad_scaler_skips"] == 0
    assert scaler.skipped is False
    assert not torch.equal(target, before)
    assert scaler.after_unscale is not None and scaler.at_step is not None
    # The five-sample final partial denominator is applied after unscale and
    # before the real optimizer step.
    assert torch.allclose(scaler.at_step, scaler.after_unscale / 5, rtol=0, atol=1e-8)
