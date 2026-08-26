"""W5 learned trainer recipe, keyed noise, checkpoint and resume proofs."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from config.params import get
from config.run_config import RunConfig, load_experiment
from training.djscc import (
    ELIGIBILITY,
    PROTECTED_COUNTERS,
    DJSCCTrainer,
    W5Hold,
    W5SmokeLimits,
    default_source_lineage_for_tests,
    deterministic_history,
    keyed_training_complex_noise,
    learned_recipe,
    learning_rate_for_epoch,
    model_state_sha256,
    state_tree_sha256,
)


class _TinyCifarTrain(Dataset[tuple[torch.Tensor, int, str]]):
    def __init__(self, epoch: int, count: int = 4) -> None:
        self.epoch = epoch
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        # The content is deterministic and in the same unit-interval model input
        # contract as TrainingDJSCCDataset. Epoch affects augmentation in the
        # real common path, so it is represented without ambient RNG here too.
        value = ((index + self.epoch) % 17) / 16  # literal-ok: bounded fixture pattern
        return torch.full((3, 32, 32), value), index % 10, f"w5-sample-{index}"


def _config(**changes: object) -> RunConfig:
    value = load_experiment("configs/learned-w5-smoke.yaml").to_dict()
    value["resolved"].update(changes)
    return RunConfig.from_dict(value)


def _trainer(root: Path, config: RunConfig | None = None) -> DJSCCTrainer:
    return DJSCCTrainer(
        config or _config(),
        device="cpu",
        runtime_root=root,
        source_lineage=default_source_lineage_for_tests(),
        smoke_limits=W5SmokeLimits(2, 2, 1, 1, 0),
    )


def _factory(epoch: int) -> _TinyCifarTrain:
    return _TinyCifarTrain(epoch)


def test_recipe_is_explicit_and_cosine_has_frozen_endpoints():
    config = _config()
    recipe = learned_recipe(config)
    assert recipe["optimizer_implementation"] == "torch.optim.Adam"
    assert (recipe["adam_beta1"], recipe["adam_beta2"], recipe["adam_epsilon"]) == (0.9, 0.999, 1e-8)
    assert recipe["adam_weight_decay"] == 0
    assert not any(recipe[name] for name in ("adam_amsgrad", "adam_maximize", "adam_foreach", "adam_capturable", "adam_differentiable", "adam_fused"))
    assert recipe["lr_warmup_epochs"] == 0
    assert recipe["amp_device_type"] == "cuda" and recipe["amp_dtype"] == "float16"
    assert learning_rate_for_epoch(config, 0) == recipe["lr"]
    final = config.parameters["learned_system"]["epochs"]["cifar10"] - 1
    assert learning_rate_for_epoch(config, final) == recipe["lr_min"]


def test_training_noise_is_ambient_rng_and_batch_order_independent():
    config = _config()
    trainer = _trainer(Path("unused-w5-test-runtime"), config)
    identities = trainer._training_noise_identities(["a", "b", "c"], 3)
    random.seed(7)
    np.random.seed(8)
    torch.manual_seed(9)
    first = keyed_training_complex_noise(identities, config.resolved["k"])
    random.seed(700)
    np.random.seed(800)
    torch.manual_seed(900)
    split = torch.cat(
        [
            keyed_training_complex_noise(identities[:1], config.resolved["k"]),
            keyed_training_complex_noise(identities[1:], config.resolved["k"]),
        ]
    )
    reversed_draw = keyed_training_complex_noise(list(reversed(identities)), config.resolved["k"]).flip(0)
    assert torch.equal(first, split)
    assert torch.equal(first, reversed_draw)


@pytest.mark.parametrize("ratio", ["r_1_6", "r_1_24"])
def test_selected_imagenette_ratios_instantiate_exact_k_and_backpropagate(tmp_path: Path, ratio: str):
    config = _config(
        dataset="imagenette160",
        dataset_version=get("datasets.imagenette160.archive_sha256"),
        split_manifest_hash=get("datasets.imagenette160.manifest_sha256"),
        bw_ratio=ratio,
        k=get(f"bandwidth.k_symbols.imagenette160.{ratio}"),
    )
    trainer = _trainer(tmp_path / ratio, config)
    record = trainer.train_epoch(0, _TinyImagenetteTrain())
    assert record["samples"] == 1
    assert record["gradient_checks"]["encoder"]["nonzero"]
    assert record["gradient_checks"]["reconstruction_head"]["nonzero"]
    assert record["gradient_checks"]["task_head"]["nonzero"]


class _TinyImagenetteTrain(Dataset[tuple[torch.Tensor, int, str]]):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        return torch.full((3, 160, 160), 0.5), 0, "w5-imagenette-selected-ratio"


def test_lambda_zero_uses_same_trainer_and_keeps_mse_measurable(tmp_path: Path):
    trainer = _trainer(tmp_path, _config(**{"lambda": 0.0}))
    record = trainer.train_epoch(0, _TinyCifarTrain(0, 2))
    assert record["reconstruction_mse"] > 0
    assert record["total_loss"] == record["cross_entropy"]
    assert record["gradient_checks"]["encoder"]["nonzero"]
    assert record["gradient_checks"]["task_head"]["nonzero"]


def test_epoch_checkpoint_and_resume_reproduce_exact_trajectory(tmp_path: Path):
    uninterrupted = _trainer(tmp_path / "uninterrupted")
    uninterrupted.run_epochs(final_epoch=1, dataset_factory=_factory)

    resumed_root = tmp_path / "resumed"
    first_process = _trainer(resumed_root)
    first_process.run_epochs(final_epoch=0, dataset_factory=_factory)
    second_process = _trainer(resumed_root)
    sidecar = second_process.resume()
    assert sidecar["completed_epoch"] == 0
    second_process.run_epochs(final_epoch=1, dataset_factory=_factory)

    assert deterministic_history(uninterrupted.training_history) == deterministic_history(second_process.training_history)
    assert model_state_sha256(uninterrupted.model) == model_state_sha256(second_process.model)
    assert state_tree_sha256(uninterrupted.optimizer.state_dict()) == state_tree_sha256(second_process.optimizer.state_dict())
    assert uninterrupted.scheduler.state_dict() == second_process.scheduler.state_dict()
    assert uninterrupted.global_optimizer_step == second_process.global_optimizer_step == 2
    assert (resumed_root / "checkpoints/epoch-0000.pt").is_file()
    assert (resumed_root / "checkpoints/epoch-0001.pt").is_file()
    assert json.loads((resumed_root / "latest.json").read_bytes())["eligibility"] == ELIGIBILITY


def _mutate_authenticated_checkpoint(root: Path, mutation) -> None:
    latest_path = root / "latest.json"
    latest = json.loads(latest_path.read_bytes())
    checkpoint = root / latest["checkpoint_path"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    mutation(payload)
    torch.save(payload, checkpoint)
    latest["checkpoint_bytes"] = checkpoint.stat().st_size
    latest["checkpoint_id"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    latest_path.write_text(json.dumps(latest, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda p: p.__setitem__("schema_version", 2), "role/version"),
        (lambda p: p["eligibility"].__setitem__("w8_eligibility", "ELIGIBLE"), "eligibility"),
        (lambda p: p["lineage"].__setitem__("config_hash", "f" * 64), "config_hash"),
        (lambda p: p["lineage"].__setitem__("dataset_version", "f" * 64), "dataset_version"),
        (lambda p: p["lineage"].__setitem__("split_manifest_hash", "f" * 64), "split_manifest_hash"),
        (lambda p: p["lineage"].__setitem__("source_commit", "f" * 40), "source_commit"),
        (lambda p: p["lineage"].__setitem__("source_manifest_sha256", "f" * 64), "source_manifest_sha256"),
        (lambda p: p["lineage"].__setitem__("execution_profile_id", "confessor_pascal_cu126"), "execution_profile_id"),
        (lambda p: p["lineage"].__setitem__("bw_ratio", "r_1_6"), "bw_ratio"),
        (lambda p: p["lineage"].__setitem__("k", 1), "k differs"),
        (lambda p: p["lineage"].__setitem__("train_seed", 9), "train_seed"),
        (lambda p: p["lineage"].__setitem__("channel_seed", 9), "channel_seed"),
        (lambda p: p["lineage"].__setitem__("architecture", "other"), "architecture"),
        (lambda p: p["lineage"].__setitem__("train_snr_db", 8), "train_snr_db"),
        (lambda p: p["lineage"].__setitem__("lambda", 0.3), "lambda differs"),
        (lambda p: p["optimizer_state"]["param_groups"][0].__setitem__("eps", 1e-7), "optimizer recipe"),
        (lambda p: p.__setitem__("scheduler_state", {"completed_epoch": 9}), "scheduler epoch"),
        (lambda p: p.__setitem__("scaler_state", {"scale": 1}), "unexpectedly carries scaler"),
        (lambda p: p.__setitem__("protected_counters", {**PROTECTED_COUNTERS, "test_access": 1}), "protected counters"),
        (lambda p: p.__setitem__("rng_state_policy", {"channel": "ambient"}), "RNG policy"),
    ],
)
def test_resume_rejects_authenticated_checkpoint_mutations(tmp_path: Path, mutation, match: str):
    root = tmp_path / "runtime"
    original = _trainer(root)
    original.run_epochs(final_epoch=0, dataset_factory=_factory)
    _mutate_authenticated_checkpoint(root, mutation)
    with pytest.raises(W5Hold, match=match):
        _trainer(root).resume()


def test_resume_rejects_hash_drift_truncation_and_silent_fresh_start(tmp_path: Path):
    root = tmp_path / "runtime"
    original = _trainer(root)
    original.run_epochs(final_epoch=0, dataset_factory=_factory)
    latest = json.loads((root / "latest.json").read_bytes())
    checkpoint = root / latest["checkpoint_path"]
    checkpoint.write_bytes(checkpoint.read_bytes()[:100])  # literal-ok: truncation attack fixture
    with pytest.raises(W5Hold, match="byte length|SHA-256"):
        _trainer(root).resume()
    empty = _trainer(tmp_path / "empty")
    with pytest.raises(W5Hold, match="missing"):
        empty.resume()


def test_immutable_checkpoint_cannot_be_overwritten(tmp_path: Path):
    trainer = _trainer(tmp_path)
    trainer.train_epoch(0, _TinyCifarTrain(0, 2))
    trainer.save_checkpoint()
    with pytest.raises(W5Hold, match="already exists"):
        trainer.save_checkpoint()


def test_checkpoint_has_no_sequential_rng_state_and_all_protected_counters_zero(tmp_path: Path):
    trainer = _trainer(tmp_path)
    trainer.run_epochs(final_epoch=0, dataset_factory=_factory)
    latest = json.loads((tmp_path / "latest.json").read_bytes())
    payload = torch.load(tmp_path / latest["checkpoint_path"], map_location="cpu", weights_only=False)
    assert payload["rng_state_policy"]["serialized_sequential_rng_states"] == []
    assert payload["protected_counters"] == PROTECTED_COUNTERS
    assert all(value == 0 for value in payload["protected_counters"].values())
    assert payload["scaler_state"] is None
    assert payload["eligibility"] == ELIGIBILITY


def test_trainer_rejects_test_split_and_ineligible_smoke_role(tmp_path: Path):
    with pytest.raises(W5Hold, match="split=train"):
        _trainer(tmp_path / "test", _config(split="test"))
    with pytest.raises(W5Hold, match="artifact role"):
        _trainer(tmp_path / "eligible", _config(artifact_role="SCIENTIFIC"))
