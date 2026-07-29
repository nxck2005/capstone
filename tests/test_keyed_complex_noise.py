"""Direct keyed complex-noise batching and identity invariance."""

from __future__ import annotations

import pytest
import torch

from channels.awgn import keyed_complex_noise
from config.params import get


def test_keyed_noise_matches_declared_identity_exactly():
    assert get("artifacts.rng_identity_fields.channel_noise") == ["noise_id"]
    noise = keyed_complex_noise("sample-noise", 16)
    assert noise.shape == (1, 16)
    assert noise.dtype == torch.complex64


def test_keyed_noise_reordering_and_batch_splitting_are_exact():
    ids = ["noise-a", "noise-b", "noise-c"]
    together = keyed_complex_noise(ids, 128)
    reordered = keyed_complex_noise([ids[2], ids[0], ids[1]], 128)
    split = torch.cat(
        [keyed_complex_noise(ids[:1], 128), keyed_complex_noise(ids[1:], 128)]
    )

    assert torch.equal(reordered, together[[2, 0, 1]])
    assert torch.equal(split, together)
    assert torch.equal(keyed_complex_noise(ids, 128), together)
    assert not torch.equal(together[0], together[1])


@pytest.mark.parametrize("bad_ids", [[], [""], [1]])
def test_keyed_noise_rejects_invalid_noise_ids(bad_ids):
    with pytest.raises(ValueError, match="noise_ids"):
        keyed_complex_noise(bad_ids, 8)
