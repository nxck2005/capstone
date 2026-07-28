"""Synthetic keyed-Philox invariance tests for SR-18."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest

from artifacts.ids import make_noise_id
from artifacts.rng import keyed_generator, keyed_standard_normal
from config.params import get


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
        first = keyed_generator(purpose, {"fixture": "same"}).random()
        second = keyed_generator(purpose, {"fixture": "same"}).random()
        assert first == second


def test_unknown_rng_purpose_raises():
    with pytest.raises(ValueError, match="unknown RNG purpose"):
        keyed_generator("not-a-purpose", {"fixture": "value"})
