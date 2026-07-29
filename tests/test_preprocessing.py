"""Synthetic executable checks for the canonical preprocessing contract (SR-19)."""

from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from config.params import get
import data.preprocessing as preprocessing
from data.preprocessing import (
    CanonicalProduct,
    canonicalize_source,
    channel_normalisation_stats,
    clip_reconstruction_for_metrics,
    codec_downsample,
    codec_input,
    codec_upsample,
    encoder_input,
    evaluation_input,
    reconstruction_metrics,
    reconstruction_input,
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
    return canonicalize_source(b"fixture/original/sample-payload", dataset)


@pytest.fixture(autouse=True)
def fixture_source_decoders(monkeypatch: pytest.MonkeyPatch):
    decoders = {
        dataset: (lambda source_bytes: _synthetic_rgb())
        for dataset in ("cifar10", "stl10", "imagenette160")
    }
    monkeypatch.setattr(preprocessing, "_SOURCE_DECODERS", decoders)


def test_codec_and_encoder_paths_are_bit_identical_for_same_sample_id():
    product = _product()

    codec_pixels = codec_input(product)
    codec_path_tensor = (
        torch.from_numpy(codec_pixels.copy()).permute(2, 0, 1).to(torch.float32)
        / np.iinfo(np.uint8).max
    )
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

    preprocessing._SOURCE_DECODERS = {
        "cifar10": lambda payload: source,
        "stl10": lambda payload: source,
    }
    cifar = canonicalize_source(source_bytes, "cifar10")
    stl = canonicalize_source(source_bytes, "stl10")

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


def test_reconstruction_input_is_byte_and_tensor_identical_to_evaluation_input():
    product = _product()
    reconstructed = product.canonical_image.copy()

    tensor = reconstruction_input(reconstructed)

    np.testing.assert_array_equal(reconstructed, product.canonical_image)
    assert torch.equal(tensor, evaluation_input(product))
    assert tensor.dtype == torch.float32
    assert tensor.shape == (3, *product.canonical_image.shape[:2])
    assert float(tensor.min()) >= 0
    assert float(tensor.max()) <= 1


@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros((8, 8), dtype=np.uint8),
        np.zeros((8, 8, 4), dtype=np.uint8),
        np.zeros((8, 8, 3), dtype=np.float32),
    ],
)
def test_reconstruction_input_accepts_only_uint8_rgb_hwc(invalid: np.ndarray):
    with pytest.raises((TypeError, ValueError)):
        reconstruction_input(invalid)


def test_train_path_is_keyed_reproducible_and_identity_sensitive():
    product = _product(dataset="stl10")
    first_identity = {
        "stable_sample_id": product.stable_sample_id,
        "train_seed": 17,
        "epoch": 4,
    }
    other_identity = {
        "stable_sample_id": product.stable_sample_id,
        "train_seed": 17,
        "epoch": 5,
    }

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
            {
                "stable_sample_id": "a-different-sample",
                "train_seed": 0,
                "epoch": 0,
            },
        )
    with pytest.raises(ValueError, match="extra=.*unexpected"):
        training_input(
            product,
            {
                "stable_sample_id": product.stable_sample_id,
                "train_seed": 0,
                "epoch": 0,
                "unexpected": True,
            },
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


def test_identical_image_ssim_and_fixed_gaussian_fixture():
    rows, columns = np.indices((16, 16))
    reference = np.stack(
        ((rows + columns) / 30, rows / 15, columns / 15),
        axis=-1,
    )
    reconstruction = reference.copy()
    reconstruction[4:12, 5:11, :] = np.clip(
        reconstruction[4:12, 5:11, :] * 0.8 + 0.05,
        0,
        1,
    )

    assert reconstruction_metrics(reference, reference).ssim == pytest.approx(1.0)
    assert reconstruction_metrics(reference, reconstruction).ssim == pytest.approx(
        0.9581761548549639,
        abs=1e-15,
    )


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
    with pytest.raises(TypeError, match="Protocols cannot be instantiated"):
        CanonicalProduct()


def test_source_bytes_reach_only_registry_selected_decoder(monkeypatch):
    calls: list[tuple[str, bytes]] = []
    payload = b"opaque encoded sample payload"
    monkeypatch.setattr(
        preprocessing,
        "_SOURCE_DECODERS",
        {
            "cifar10": lambda source: (
                calls.append(("cifar10", source)) or _synthetic_rgb()
            ),
            "stl10": lambda source: (
                calls.append(("stl10", source)) or _synthetic_rgb()
            ),
        },
    )

    product = canonicalize_source(payload, "stl10")

    assert calls == [("stl10", payload)]
    assert product.stable_sample_id == stable_sample_id(payload)


def test_production_code_has_no_supported_pixel_injection_or_product_constructor():
    module_path = Path(preprocessing.__file__).resolve()
    source_root = module_path.parents[1]
    forbidden_imports: list[str] = []
    forbidden_calls: list[str] = []
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "data.preprocessing":
                for alias in node.names:
                    if alias.name.startswith("_") or alias.name in {
                        "canonicalize_image",
                        "canonical_uint8_to_tensor",
                    }:
                        forbidden_imports.append(str(path))
            if path != module_path and isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in {
                    "_CanonicalProduct",
                    "_canonicalize_decoded",
                }:
                    forbidden_calls.append(str(path))

    assert not hasattr(preprocessing, "canonicalize_image")
    assert not hasattr(preprocessing, "canonical_uint8_to_tensor")
    assert forbidden_imports == []
    assert forbidden_calls == []
