"""W8-A protocol boundary tests; no dataset or scientific execution."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from config.run_config import RunConfig
from models.djscc import build_djscc
from training.w8_final import W8Hold, W8Trainer
from training.w8_protocol import (
    W8_CHECKPOINT_SELECTION_CHANNEL_SEED_RULE,
    W8_EPOCHS,
    W8_EXPECTED_K,
    W8_EXPECTED_LAMBDA,
    W8_EXPECTED_RATIOS,
    W8_PROFILE_ID,
    W8_SELECTED_GPU_UUID,
    W8_TRAIN_SAMPLE_COUNT,
    W8_VALIDATION_BATCH_SIZE,
    W8_VALIDATION_SAMPLE_COUNT,
    checkpoint_selection_snr_db,
    fresh_initialization_identity,
    load_w8_config,
    run_cells,
    unique_core_ratios,
    validate_run_order,
    validate_w8_config,
)
from tests.w8_hardening_fixtures import TinyDJSCC, lineage, profile_binding, tiny_config


def test_dec11_resolves_exactly_two_unique_ratios_and_seed_major_order() -> None:
    assert unique_core_ratios() == ("r_1_6", "r_1_24")
    expected = (
        (1, "r_1_6", 0, 0, 12800),
        (2, "r_1_24", 0, 0, 3200),
        (3, "r_1_6", 1, 1, 12800),
        (4, "r_1_24", 1, 1, 3200),
        (5, "r_1_6", 2, 2, 12800),
        (6, "r_1_24", 2, 2, 3200),
    )
    assert tuple(
        (cell.run_index, cell.ratio, cell.train_seed, cell.channel_seed, cell.k)
        for cell in run_cells()
    ) == expected
    validate_run_order(run_cells())
    assert len({(cell.train_seed, cell.channel_seed) for cell in run_cells()}) == 3
    assert {(cell.train_seed, cell.channel_seed) for cell in run_cells()} == {
        (0, 0), (1, 1), (2, 2)
    }
    assert len(run_cells()) == 6


def test_every_frozen_cell_resolves_lambda_snr_shape_profile_and_batch() -> None:
    for cell in run_cells():
        config = load_w8_config(cell.ratio, cell.train_seed, cell.channel_seed)
        validate_w8_config(config)
        resolved = config.resolved
        assert resolved["bw_ratio"] in W8_EXPECTED_RATIOS
        assert resolved["k"] == W8_EXPECTED_K[cell.ratio] == cell.k
        assert resolved["lambda"] == W8_EXPECTED_LAMBDA == 3.0
        assert resolved["train_snr_db"] == checkpoint_selection_snr_db() == 7
        assert resolved["checkpoint_selection_snr_db"] == 7
        assert resolved["checkpoint_selection_snr_parameter"] == (
            "params.learned_system.checkpoint_selection_snr_db"
        )
        assert resolved["execution_profile_id"] == W8_PROFILE_ID
        assert resolved["physical_batch_size"] == 32
        assert resolved["accumulation_factor"] == 1
        assert resolved["effective_batch_size"] == 32
        assert resolved["validation_batch_size"] == W8_VALIDATION_BATCH_SIZE == 32
        assert resolved["architecture"] == "djscc_residual_v1"
    descriptor = load_w8_config("r_1_6", 0, 0).parameters["learned_system"]
    assert descriptor["epochs"]["imagenette160"] == W8_EPOCHS == 100
    assert W8_TRAIN_SAMPLE_COUNT == 8469
    assert W8_VALIDATION_SAMPLE_COUNT == 1000
    assert W8_CHECKPOINT_SELECTION_CHANNEL_SEED_RULE == "run_channel_seed"


def test_zipped_pairing_rejects_cross_product_and_mutated_cells() -> None:
    with pytest.raises(ValueError, match="zipped"):
        load_w8_config("r_1_6", 0, 1)
    with pytest.raises(ValueError, match="zipped"):
        load_w8_config("r_1_6", 2, 1)

    cells = list(run_cells())
    cells[1], cells[2] = cells[2], cells[1]
    with pytest.raises(ValueError, match="order"):
        validate_run_order(cells)


def test_fresh_initialization_identity_is_explicit_and_keyed() -> None:
    assert fresh_initialization_identity(0) == fresh_initialization_identity(0)
    assert fresh_initialization_identity(0) != fresh_initialization_identity(1)
    assert fresh_initialization_identity(0)["predecessor_checkpoint_id"] is None
    assert fresh_initialization_identity(0)["optimizer_state_transfer"] is False


def test_keyed_model_initialization_is_repeatable_and_seed_specific() -> None:
    first = build_djscc(load_w8_config("r_1_24", 0, 0), device="cpu")
    second = build_djscc(load_w8_config("r_1_24", 0, 0), device="cpu")
    other = build_djscc(load_w8_config("r_1_24", 1, 1), device="cpu")
    assert all(torch.equal(first.state_dict()[key], second.state_dict()[key]) for key in first.state_dict())
    assert any(not torch.equal(first.state_dict()[key], other.state_dict()[key]) for key in first.state_dict())


def test_w7_or_foreign_w8_initial_checkpoint_is_rejected_before_model_use(tmp_path: Path) -> None:
    config = tiny_config()
    kwargs = {
        "device": "cpu",
        "runtime_root": tmp_path,
        "source_lineage": lineage(),
        "profile_binding": profile_binding(config),
        "model": TinyDJSCC(),
        "num_workers": 0,
    }
    with pytest.raises(W8Hold, match="initial checkpoint"):
        W8Trainer(config, initial_checkpoint=Path("results/learned/w7/checkpoints/epoch-0000.pt"), **kwargs)
    with pytest.raises(W8Hold, match="initial checkpoint"):
        W8Trainer(config, initial_checkpoint=Path("other-w8/epoch-0000.pt"), **kwargs)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value["resolved"].__setitem__("bw_ratio", "r_1_24"), "k"),
        (lambda value: value["resolved"].__setitem__("train_seed", 2), "seed"),
        (lambda value: value["resolved"].__setitem__("channel_seed", 1), "seed"),
        (lambda value: value["resolved"].__setitem__("physical_batch_size", 16), "batch"),
        (lambda value: value["resolved"].__setitem__("lambda", 1.0), "lambda"),
    ],
)
def test_mutated_w8_config_fails_closed(mutation, match: str) -> None:
    value = copy.deepcopy(load_w8_config("r_1_6", 0, 0).to_dict())
    mutation(value)
    with pytest.raises((ValueError, TypeError), match=match):
        validate_w8_config(RunConfig.from_dict(value))
