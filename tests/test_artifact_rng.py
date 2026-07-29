"""Synthetic keyed-Philox invariance tests for SR-18."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest

from artifacts.ids import make_noise_id
from artifacts.rng import keyed_generator, keyed_standard_normal
from config.params import get


def _identity(purpose: str) -> dict[str, object]:
    identities: dict[str, dict[str, object]] = {
        "channel_noise": {"noise_id": "noise-sha256"},
        "outage_label": {
            "split_manifest_hash": "split-sha256",
            "stable_sample_id": "sample-a",
            "channel_seed": 101,
        },
        "augmentation": {
            "stable_sample_id": "sample-a",
            "train_seed": 17,
            "epoch": 4,
        },
        "init": {
            "train_seed": 17,
            "component_path": "encoder.stem.weight",
        },
        "batch_order": {"train_seed": 17, "epoch": 4},
    }
    return identities[purpose]


def _noise_fields(sample: str, block_index: int) -> dict[str, object]:
    return {
        "dataset_version": "dataset-sha256",
        "split_manifest_hash": "split-sha256",
        "stable_sample_id": sample,
        "test_snr_db": -8,
        "channel_seed": 101,
        "channel": "awgn",
        "k": 8533,
        "block_index": block_index,
        "rng_purpose": "channel_noise",
    }


def _evaluate_batches(
    batches: Iterable[Iterable[dict[str, object]]],
) -> dict[str, tuple[str, np.ndarray]]:
    evaluated: dict[str, tuple[str, np.ndarray]] = {}
    for batch in batches:
        for row in batch:
            noise_id = make_noise_id(row)
            draw = keyed_standard_normal(
                "channel_noise",
                {"noise_id": noise_id},
                size=(4,),
            )
            evaluated[str(row["stable_sample_id"])] = (noise_id, draw)
    return evaluated


def test_reordering_and_rebatching_change_no_noise_id_or_draw():
    rows = [
        _noise_fields("sample-a", 0),
        _noise_fields("sample-b", 0),
        _noise_fields("sample-c", 1),
    ]

    one_batch = _evaluate_batches([rows])
    reordered_batches = _evaluate_batches([[rows[2]], [rows[0], rows[1]]])

    assert one_batch.keys() == reordered_batches.keys()
    for sample_id, (noise_id, draw) in one_batch.items():
        other_noise_id, other_draw = reordered_batches[sample_id]
        assert other_noise_id == noise_id
        np.testing.assert_array_equal(other_draw, draw)


def test_distinct_identity_changes_draw():
    first = keyed_standard_normal(
        "channel_noise",
        {"noise_id": "first"},
        size=(4,),
    )
    second = keyed_standard_normal(
        "channel_noise",
        {"noise_id": "second"},
        size=(4,),
    )

    assert not np.array_equal(first, second)


def test_every_declared_rng_purpose_is_supported():
    for purpose in get("artifacts.rng_purposes"):
        first = keyed_generator(purpose, _identity(purpose)).random()
        second = keyed_generator(purpose, _identity(purpose)).random()
        assert first == second


def test_unknown_rng_purpose_raises():
    with pytest.raises(ValueError, match="unknown RNG purpose"):
        keyed_generator("not-a-purpose", {"fixture": "value"})


@pytest.mark.parametrize("purpose", get("artifacts.rng_purposes"))
def test_rng_purposes_reject_missing_and_extra_identity_fields(purpose):
    complete = _identity(purpose)
    missing = dict(complete)
    missing.pop(next(iter(missing)))
    extra = {**complete, "unexpected": "value"}

    with pytest.raises(ValueError, match="missing="):
        keyed_generator(purpose, missing)
    with pytest.raises(ValueError, match="extra=.*unexpected"):
        keyed_generator(purpose, extra)


def test_init_component_path_is_model_qualified():
    with pytest.raises(ValueError, match="stable model-qualified name"):
        keyed_generator(
            "init",
            {"train_seed": 17, "component_path": "weight"},
        )
