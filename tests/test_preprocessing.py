"""Synthetic executable checks for the canonical preprocessing contract (SR-19)."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest
import torch

from config.params import get
from data.preprocessing import (
    CanonicalProduct,
    canonical_uint8_to_tensor,
    canonicalize_image,
    channel_normalisation_stats,
    clip_reconstruction_for_metrics,
    codec_downsample,
    codec_input,
    codec_upsample,
    encoder_input,
    evaluation_input,
    reconstruction_metrics,
    reconstruction_psnr,
    stable_sample_id,
    training_input,
)


def _synthetic_rgb(height: int = 53, width: int = 91) -> np.ndarray:
    rows, columns = np.indices((height, width))
    return np.stack(
        (
            (rows * 7 + columns * 3) % 256,
            (rows * 11 + columns * 5) % 256,
            (rows * 13 + columns * 17) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


def _product(dataset: str = "cifar10") -> CanonicalProduct:
    return canonicalize_image(
        _synthetic_rgb(),
        dataset=dataset,
        source_bytes=b"fixture/original/sample-payload",
    )


def test_codec_and_encoder_paths_are_bit_identical_for_same_sample_id():
    product = _product()

    codec_pixels = codec_input(product)
    codec_path_tensor = canonical_uint8_to_tensor(codec_pixels)
    encoder_path_tensor = encoder_input(product)

    assert product.stable_sample_id == stable_sample_id(
        b"fixture/original/sample-payload"
    )
    np.testing.assert_array_equal(codec_pixels, product.canonical_image)
    assert torch.equal(codec_path_tensor, encoder_path_tensor)
    expected = (
        torch.from_numpy(product.canonical_image.copy())
        .permute(2, 0, 1)
        .to(torch.float32)
        / np.iinfo(np.uint8).max
    )
    assert torch.equal(encoder_path_tensor, expected)
    assert float(encoder_path_tensor.min()) >= 0
    assert float(encoder_path_tensor.max()) <= 1


def test_source_identity_is_independent_of_preprocessing_and_pixels_are_not():
    source_bytes = b"one immutable per-sample source payload"
    source = _synthetic_rgb()

    cifar = canonicalize_image(
        source,
        dataset="cifar10",
        source_bytes=source_bytes,
    )
    stl = canonicalize_image(
        source,
        dataset="stl10",
        source_bytes=source_bytes,
    )

    expected = hashlib.sha256(source_bytes).hexdigest()[:16]
    assert cifar.stable_sample_id == expected == stl.stable_sample_id
    assert cifar.canonical_image.shape != stl.canonical_image.shape
    assert stable_sample_id(source_bytes + b"-changed") != expected


def test_eval_path_is_deterministic_and_never_normalises():
    product = _product()

    first = evaluation_input(product)
    second = evaluation_input(product)

    assert torch.equal(first, second)
    assert torch.equal(first, encoder_input(product))
    assert channel_normalisation_stats() == (
        tuple(get("preprocessing.channel_mean")),
        tuple(get("preprocessing.channel_std")),
    )


def test_train_path_is_keyed_reproducible_and_identity_sensitive():
    product = _product(dataset="stl10")
    first_identity = {"train_seed": 17, "epoch": 4}
    other_identity = {"train_seed": 17, "epoch": 5}

    first = training_input(product, first_identity)
    repeated = training_input(product, first_identity)
    other = training_input(product, other_identity)

    assert torch.equal(first, repeated)
    assert not torch.equal(first, other)
    assert first.shape == encoder_input(product).shape


def test_train_path_rejects_a_mismatched_or_incomplete_rng_identity():
    product = _product()

    with pytest.raises(ValueError, match="must not be empty"):
        training_input(product, {})
    with pytest.raises(ValueError, match="does not match"):
        training_input(
            product,
            {"stable_sample_id": "a-different-sample", "epoch": 0},
        )


def test_codec_resampling_pins_direction_and_preserves_aspect():
    source = _synthetic_rgb(height=24, width=40)

    downsampled = codec_downsample(source, shorter_side=12)
    restored = codec_upsample(downsampled, output_hw=(24, 40))

    assert downsampled.shape == (12, 20, 3)
    assert restored.shape == source.shape
    assert downsampled.dtype == restored.dtype == np.uint8
    with pytest.raises(ValueError, match="must not upscale"):
        codec_downsample(source, shorter_side=25)
    with pytest.raises(ValueError, match="preserve"):
        codec_upsample(downsampled, output_hw=(24, 41))


def test_psnr_matches_an_analytically_known_mse():
    reference = np.zeros((16, 16, 3), dtype=np.float64)
    reconstruction = np.full_like(reference, 0.25)
    mse = float(np.mean((reference - reconstruction) ** 2))

    actual = reconstruction_psnr(reference, reconstruction)

    assert actual == pytest.approx(10 * math.log10(1.0 / mse))
    metrics = reconstruction_metrics(reference, reconstruction)
    assert metrics.psnr_db == pytest.approx(actual)
    assert math.isfinite(metrics.ssim)


def test_out_of_range_reconstruction_is_really_clipped_before_metrics():
    reference = np.zeros((16, 16, 3), dtype=np.float64)
    reconstruction = np.full_like(reference, 2.0)

    clipped = clip_reconstruction_for_metrics(reconstruction)
    clipped_psnr = reconstruction_psnr(reference, reconstruction)
    raw_mse = float(np.mean((reference - reconstruction) ** 2))
    raw_psnr = 10 * math.log10(1.0 / raw_mse)

    assert np.max(clipped) == get("preprocessing.psnr_data_range")
    assert clipped_psnr == pytest.approx(0.0)
    assert clipped_psnr != pytest.approx(raw_psnr)


def test_invalid_canonical_pixels_fail_closed():
    with pytest.raises(TypeError, match="dtype uint8"):
        CanonicalProduct(
            stable_sample_id="sample",
            canonical_image=np.zeros((8, 8, 3), dtype=np.float32),
        )
