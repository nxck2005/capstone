from __future__ import annotations

import numpy as np
import pytest
import torch

from evaluation.er9_protocol import (
    decode_message,
    encode_message,
    expected_level_values,
    factorisation_for_dimension,
    fit_static_entropy_model,
    flatten_features,
    raw_bound_bits,
    unflatten_features,
    UniformScalarQuantizer,
)


def test_am96_factorisation_is_exact_for_every_configured_dimension() -> None:
    expected = {
        64: 1,
        128: 2,
        256: 4,
        512: 8,
        1024: 16,
        2048: 32,
        4096: 64,
        8192: 128,
    }
    for dimension, channels in expected.items():
        value = factorisation_for_dimension(dimension)
        assert value.shape == (channels, 8, 8)
        assert value.exactly_realised
        assert channels * 8 * 8 == dimension


def test_am96_flatten_unflatten_is_contiguous_nchw_round_trip() -> None:
    factorisation = factorisation_for_dimension(256)
    source = torch.arange(2 * 4 * 8 * 8, dtype=torch.float32).reshape(2, 4, 8, 8)
    flattened = flatten_features(source, factorisation)
    assert torch.equal(unflatten_features(flattened, factorisation), source)
    assert torch.equal(flattened[0, :8], source[0, 0, 0, :8])


def test_am96_quantiser_has_endpoint_levels_and_fixed_range() -> None:
    for bits in (2, 4, 6, 8):
        quantiser = UniformScalarQuantizer(bits)
        levels = expected_level_values(bits)
        assert quantiser.levels == 2**bits
        assert levels.size == quantiser.levels
        assert float(levels[0]) == -1.0
        assert float(levels[-1]) == 1.0
        inputs = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0])
        indices = quantiser.quantise(inputs)
        assert int(indices.min()) == 0
        assert int(indices.max()) == quantiser.levels - 1
        assert quantiser.dequantise(indices).min() >= -1.0
        assert quantiser.dequantise(indices).max() <= 1.0


def test_am96_quantiser_ste_retains_tanh_gradient() -> None:
    quantiser = UniformScalarQuantizer(4)
    inputs = torch.tensor([[-0.7, 0.3]], requires_grad=True)
    values, indices = quantiser.straight_through(inputs)
    values.sum().backward()
    assert indices.dtype == torch.long
    assert inputs.grad is not None
    assert torch.allclose(inputs.grad, 1.0 - torch.tanh(inputs.detach()) ** 2)


def test_am96_range_and_raw_message_round_trip() -> None:
    quantiser = UniformScalarQuantizer(6)
    source = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 63], dtype=np.int64)
    model = fit_static_entropy_model(source, quantiser.levels)
    encoded = encode_message(source, quantiser, model, packet_payload_bits=4248)
    restored, branch = decode_message(
        encoded.message_bits,
        dimension=source.size,
        quantiser=quantiser,
        entropy_model=model,
    )
    assert np.array_equal(restored, source)
    assert branch == encoded.selector
    assert encoded.counted_bits <= encoded.packet_payload_bits


def test_am96_raw_bound_counts_the_selector_and_rejects_over_budget() -> None:
    assert raw_bound_bits(2048, 2) == 4097
    quantiser = UniformScalarQuantizer(8)
    model = fit_static_entropy_model(np.zeros(64, dtype=np.int64), quantiser.levels)
    with pytest.raises(ValueError, match="exceeds"):
        encode_message(
            np.zeros(64, dtype=np.int64),
            quantiser,
            model,
            packet_payload_bits=10,
        )
