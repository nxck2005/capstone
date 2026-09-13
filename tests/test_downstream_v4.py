"""Synthetic-only checks for the final W9 successor contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.downstream_v4 import EXPECTED_SNR_GRID, FINAL_PAIR, PRODUCTION_CELLS, validate_final_er9_cell
from evaluation.h4_precision import H4PrecisionError, aggregate_three_cell_differences
from evaluation.w10_rehearsal import W10_SYSTEMS, closeout as w10_closeout, execute as execute_w10, work_units
from training.er2_snr import select_training_snr_db


def _er9_validation_fixture() -> dict:
    checkpoint = "a" * 64
    ids = [f"val-{index:04d}" for index in range(1000)]
    points = []
    for snr in EXPECTED_SNR_GRID:
        rows = [{"stable_sample_id": stable_id, "split": "val", "test_snr_db": snr, "run_id": f"run-{snr}", "correct": index % 2 == 0} for index, stable_id in enumerate(ids)]
        points.append({
            "snr_db": snr,
            "validation_total": 1000,
            "validation_n_correct": 500,
            "run_id": f"run-{snr}",
            "checkpoint_id": checkpoint,
            "task_head_identity": "er9taskhead-" + "b" * 64,
            "row_binding": {
                "checkpoint_id": checkpoint,
                "train_seed": 0,
                "channel_seed": 0,
                "transmit_dim": 2048,
                "quantiser_bits": 2,
                "task_head_identity": "er9taskhead-" + "b" * 64,
                "test_access": 0,
            },
            "per_image": rows,
            "test_access": 0,
        })
    return {
        "artifact_role": "ER9_FINAL_PRODUCTION_VALIDATION_CELL_V4",
        "train_seed": 0, "channel_seed": 0, "selected_pair": FINAL_PAIR,
        "checkpoint": {"sha256": checkpoint},
        "task_head": {"kind": "own_task_head", "identity": "er9taskhead-" + "b" * 64},
        "snr_grid_db": list(EXPECTED_SNR_GRID), "points": points,
        "test": "SEALED", "test_access": 0,
    }


def test_final_er9_validation_binds_real_checkpoint_head_cells_and_grid() -> None:
    value = _er9_validation_fixture()
    validate_final_er9_cell(value, cell=(0, 0))
    value["checkpoint"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="sentinel"):
        validate_final_er9_cell(value, cell=(0, 0))


def test_final_er9_validation_rejects_clean_reference_head_and_bad_denominator() -> None:
    value = _er9_validation_fixture()
    value["task_head"] = {"kind": "clean", "identity": "reference-classifier"}
    with pytest.raises(RuntimeError, match="own-task-head"):
        validate_final_er9_cell(value, cell=(0, 0))
    value = _er9_validation_fixture()
    value["points"][0]["per_image"].pop()
    with pytest.raises(RuntimeError, match="denominator"):
        validate_final_er9_cell(value, cell=(0, 0))


def test_production_scope_is_exact_fresh_three_cells() -> None:
    assert FINAL_PAIR == {"transmit_dim": 2048, "quantiser_bits": 2}
    assert PRODUCTION_CELLS == ((0, 0), (1, 1), (2, 2))
    roots = {f"checkpoints/er9_production_pascal_v4/train{a}_channel{b}" for a, b in PRODUCTION_CELLS}
    assert len(roots) == 3
    assert all("stage1" not in root for root in roots)


def test_am95_assignment_is_order_and_batch_invariant() -> None:
    args = ("imagenette160-v1", "f" * 64, "stable-image-17", 0, 42)
    first = select_training_snr_db(*args)
    reordered = [select_training_snr_db("imagenette160-v1", "f" * 64, stable, 0, 42) for stable in ("stable-image-99", "stable-image-17")]
    assert first == reordered[1]
    assert first in {1, 4, 7, 13, 19}


def test_g11_requires_three_zipped_cells_and_rejects_randomized_er2() -> None:
    ids = ("a", "b")
    learned = {f"{a}/{b}": {stable: {"-8": True} for stable in ids} for a, b in PRODUCTION_CELLS}
    er9 = {f"{a}/{b}": {stable: {"-8": False} for stable in ids} for a, b in PRODUCTION_CELLS}
    value = aggregate_three_cell_differences(learned, er9, snr_grid_db=(-8,))
    assert value["cells"] == [list(cell) for cell in PRODUCTION_CELLS]
    with pytest.raises(H4PrecisionError, match="randomized"):
        aggregate_three_cell_differences(learned, er9, snr_grid_db=(-8,), learned_arm="randomized_er2")
    del er9["2/2"]["b"]
    with pytest.raises(H4PrecisionError, match="stable IDs"):
        aggregate_three_cell_differences(learned, er9, snr_grid_db=(-8,))


def test_w10_is_one_cell_full_grid_all_configured_systems_and_validation_only(tmp_path: Path) -> None:
    expected = work_units()
    assert {tuple((unit["train_seed"], unit["channel_seed"])) for unit in expected} == {(0, 0)}
    assert tuple(dict.fromkeys(unit["snr_db"] for unit in expected)) == EXPECTED_SNR_GRID
    assert tuple(dict.fromkeys(unit["system"] for unit in expected)) == W10_SYSTEMS
    authority = {
        "authority_kind": "W10_VALIDATION_REHEARSAL_AUTHORITY",
        "cell": {"train_seed": 0, "channel_seed": 0},
        "snr_grid_db": list(EXPECTED_SNR_GRID),
        "systems": list(W10_SYSTEMS),
        "validation_only": True,
        "test_authorized": False,
        "test_access": 0,
    }
    results = execute_w10(tmp_path, authority=authority, evaluator=lambda unit: {"n_correct": unit["ordinal"] % 10, "n_total": 10})
    terminal = w10_closeout(results, source_commit="c" * 40, authority_id="authority")
    assert terminal["split"] == "val" and terminal["cell"] == [0, 0]
    assert terminal["test_access"] == 0 and terminal["test"] == "SEALED"


def test_w10_rejects_authority_that_can_touch_test(tmp_path: Path) -> None:
    authority = {
        "authority_kind": "W10_VALIDATION_REHEARSAL_AUTHORITY",
        "cell": {"train_seed": 0, "channel_seed": 0},
        "snr_grid_db": list(EXPECTED_SNR_GRID),
        "systems": list(W10_SYSTEMS),
        "validation_only": True,
        "test_authorized": True,
        "test_access": 0,
    }
    with pytest.raises(RuntimeError, match="test boundary"):
        execute_w10(tmp_path, authority=authority, evaluator=lambda unit: {"n_correct": 0, "n_total": 1})
