"""Synthetic identity and strict-pairing tests for SR-18."""

from __future__ import annotations

from copy import deepcopy

import pytest

from artifacts.ids import (
    join_pair_trajectories,
    make_analysis_cell_id,
    make_noise_id,
    make_pair_id,
    make_run_id,
)


def _run_fields() -> dict[str, object]:
    return {
        "system": "learned",
        "dataset": "imagenette160",
        "dataset_version": "dataset-sha256",
        "split": "test",
        "split_manifest_hash": "split-sha256",
        "bw_ratio": "r_1_6",
        "test_snr_db": -8,
        "train_seed": 17,
        "channel_seed": 101,
        "config_hash": "config-sha256",
        "checkpoint_id": "checkpoint-sha256",
        "classifier_variant": "clean",
        "ldpc_rate": None,
        "modulation": None,
        "quantiser_bits": None,
        "transmit_dim": 1024,
        "lambda": 0.3,
        "analysis_version": 1,
    }


def _row_fields() -> dict[str, object]:
    values = _run_fields()
    values.update(
        {
            "stable_sample_id": "sample-sha256",
            "channel": "awgn",
            "k": 8533,
            "block_index": 0,
            "rng_purpose": "channel_noise",
            "comparison": "learned_vs_classical",
        }
    )
    values["analysis_cell_id"] = make_analysis_cell_id(values)
    values["noise_id"] = make_noise_id(values)
    return values


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("split", "validation"),
        ("checkpoint_id", "other-checkpoint"),
        ("config_hash", "other-config"),
    ],
)
def test_split_checkpoint_and_config_change_run_id(field: str, changed: str):
    original = _run_fields()
    mutated = deepcopy(original)
    mutated[field] = changed

    assert make_run_id(mutated) != make_run_id(original)


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("system", "classical_adaptive"),
        ("comparison", "learned_vs_er9"),
    ],
)
def test_excluded_fields_never_change_pair_id(field: str, changed: str):
    left = _row_fields()
    right = deepcopy(left)
    right[field] = changed

    assert make_pair_id(right) == make_pair_id(left)


def test_comparison_arms_share_pair_and_noise_but_not_run_id():
    learned = _row_fields()
    classical = deepcopy(learned)
    classical["system"] = "classical_adaptive"
    classical["comparison"] = "learned_vs_classical"

    assert make_pair_id(learned) == make_pair_id(classical)
    assert make_noise_id(learned) == make_noise_id(classical)
    assert make_run_id(learned) != make_run_id(classical)


def test_identity_requires_every_declared_key():
    values = _run_fields()
    del values["checkpoint_id"]

    with pytest.raises(ValueError, match="checkpoint_id"):
        make_run_id(values)


def test_er10_join_is_one_to_one():
    first = _row_fields()
    second = deepcopy(first)
    second["stable_sample_id"] = "sample-two"
    second["noise_id"] = make_noise_id(second)
    first["pair_id"] = make_pair_id(first)
    second["pair_id"] = make_pair_id(second)

    left = [second, first]
    right = [
        {**first, "system": "classical_adaptive"},
        {**second, "system": "classical_adaptive"},
    ]

    joined = join_pair_trajectories(left, right)

    assert len(joined) == 2
    assert all(a["pair_id"] == b["pair_id"] for a, b in joined)


def test_er10_join_rejects_missing_trajectory():
    row = _row_fields()
    row["pair_id"] = make_pair_id(row)

    with pytest.raises(ValueError, match="missing_from_right"):
        join_pair_trajectories([row], [])


def test_er10_join_rejects_duplicate_trajectory():
    row = _row_fields()
    row["pair_id"] = make_pair_id(row)

    with pytest.raises(ValueError, match="duplicates pair_id"):
        join_pair_trajectories([row, dict(row)], [row])
