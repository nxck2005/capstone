"""JPEG 2000 source-codec, search, and cache checks."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

import baseline.j2k as j2k
from baseline.j2k import J2KCodec, J2KCodecError


def _image(axis: int = 64) -> np.ndarray:
    rows, columns = np.indices((axis, axis))
    return np.stack(
        (
            (rows * 7 + columns * 3) % 256,
            (rows * 11 + columns * 5) % 256,
            (rows * 13 + columns * 17) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


def _identity(image: np.ndarray) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


@pytest.mark.external_codec_runtime
def test_raw_codestream_search_decode_budget_and_cache(tmp_path: Path):
    image = _image()
    codec = J2KCodec(tmp_path / "cache")

    first = codec.encode_to_budget(
        image,
        canonical_pixels_sha256=_identity(image),
        budget_bytes=900,
        encode_axis_px=64,
    )
    second = codec.encode_to_budget(
        image,
        canonical_pixels_sha256=_identity(image),
        budget_bytes=900,
        encode_axis_px=64,
    )

    assert first.feasible and first.decode_success
    assert first.codestream is not None
    assert first.codestream.startswith(b"\xff\x4f")
    assert first.emitted_byte_count == len(first.codestream) <= 900
    assert first.codestream_sha256 == hashlib.sha256(first.codestream).hexdigest()
    np.testing.assert_array_equal(first.decoded_image, second.decoded_image)
    assert not first.cache_hit and second.cache_hit
    assert first.cache_key == second.cache_key
    assert len(tuple((tmp_path / "cache").glob("*.j2kcache"))) == 1


def test_runtime_failure_precedes_cache_artifact_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_root = tmp_path / "must-not-exist"
    codec = J2KCodec(cache_root)
    monkeypatch.setattr(
        j2k,
        "assert_j2k_runtime",
        lambda: (_ for _ in ()).throw(RuntimeError("wrong OpenJPEG")),
    )

    with pytest.raises(RuntimeError, match="wrong OpenJPEG"):
        codec.encode_to_budget(
            _image(),
            canonical_pixels_sha256=_identity(_image()),
            budget_bytes=900,
            encode_axis_px=64,
        )
    assert not cache_root.exists()


@pytest.mark.external_codec_runtime
def test_nonmonotone_search_keeps_largest_observed_fitting_codestream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    image = _image()
    codec = J2KCodec(tmp_path / "cache")
    codec.max_iterations = 5
    codec.byte_tolerance = 0
    sizes = iter((150, 50, 90, 95, 80))

    def encode_once(source: np.ndarray, ratio: float) -> bytes:
        return b"\xff\x4f" + bytes(next(sizes) - 2)

    monkeypatch.setattr(codec, "_encode_once", encode_once)
    monkeypatch.setattr(codec, "_decode_codestream", lambda stream: image.copy())
    result = codec.encode_to_budget(
        image,
        canonical_pixels_sha256=_identity(image),
        budget_bytes=100,
        encode_axis_px=64,
    )

    assert result.feasible
    assert result.emitted_byte_count == 95
    assert [point.emitted_bytes for point in result.search_trace] == [
        150,
        50,
        90,
        95,
        80,
    ]


@pytest.mark.external_codec_runtime
def test_explicit_infeasible_result_is_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    image = _image()
    codec = J2KCodec(tmp_path / "cache")
    monkeypatch.setattr(
        codec,
        "_encode_once",
        lambda source, ratio: b"\xff\x4f" + bytes(148),
    )
    first = codec.encode_to_budget(
        image,
        canonical_pixels_sha256=_identity(image),
        budget_bytes=100,
        encode_axis_px=64,
    )
    second = codec.encode_to_budget(
        image,
        canonical_pixels_sha256=_identity(image),
        budget_bytes=100,
        encode_axis_px=64,
    )

    assert not first.feasible and not first.decode_success
    assert first.codestream is None and first.decoded_image is None
    assert second.cache_hit and not second.feasible


@pytest.mark.external_codec_runtime
def test_partial_cache_entry_is_rejected(tmp_path: Path):
    image = _image()
    codec = J2KCodec(tmp_path / "cache")
    result = codec.encode_to_budget(
        image,
        canonical_pixels_sha256=_identity(image),
        budget_bytes=900,
        encode_axis_px=64,
    )
    cache_path = tmp_path / "cache" / f"{result.cache_key}.j2kcache"
    cache_path.write_bytes(b"partial")

    with pytest.raises(J2KCodecError, match="invalid JPEG 2000 cache"):
        codec.encode_to_budget(
            image,
            canonical_pixels_sha256=_identity(image),
            budget_bytes=900,
            encode_axis_px=64,
        )


@pytest.mark.external_codec_runtime
def test_jp2_container_and_invalid_pixels_are_rejected(tmp_path: Path):
    codec = J2KCodec(tmp_path / "cache")
    with pytest.raises(J2KCodecError, match="JP2 container"):
        codec._validate_raw_codestream(b"\x00\x00\x00\x0cjP  \r\n\x87\n")
    with pytest.raises(TypeError, match="uint8"):
        codec.encode_to_budget(
            _image().astype(np.float32),
            canonical_pixels_sha256=_identity(_image()),
            budget_bytes=900,
            encode_axis_px=64,
        )
