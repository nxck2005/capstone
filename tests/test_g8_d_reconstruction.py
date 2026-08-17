from __future__ import annotations

import hashlib
import json
import numpy as np
import pytest

from baseline.g8_d import (
    BR11Accounting,
    CODEC_INFEASIBILITY,
    CodecConfigurationIdentity,
    EmittedFileIdentity,
    G8DContractError,
    ImageIdentity,
    ReconstructionCache,
    ValidationSplitIdentity,
    account_br11,
    aggregate_br11,
    canonical_json,
    sha256_bytes,
)


def _codestream(tile_payload: bytes, *, psot: int) -> bytes:
    # SOT + Lsot + Isot + Psot + TPsot + TNsot is twelve bytes including the
    # marker.  The nonzero Psot fixture is deliberately not a one-byte toy.
    sot = b"\xff\x90\x00\x0a\x00\x00" + psot.to_bytes(4, "big") + b"\x00\x00"
    return b"\xff\x4f" + sot + b"\xff\x93" + tile_payload + b"\xff\xd9"


def _two_tile_codestream() -> bytes:
    def tile(payload: bytes) -> bytes:
        return b"\xff\x90\x00\x0a\x00\x00" + (17).to_bytes(4, "big") + b"\x00\x00" + b"\xff\x93" + payload

    return b"\xff\x4f" + tile(b"abc") + tile(b"def") + b"\xff\xd9"


def _split() -> ValidationSplitIdentity:
    return ValidationSplitIdentity("fixture", "val", "a" * 64, "b" * 64)


def _image(source: bytes = b"source-v1") -> ImageIdentity:
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    return ImageIdentity.from_pixels(
        split_identity=_split(),
        stable_sample_id=hashlib.sha256(source).hexdigest()[:16],
        source_bytes=source,
        canonical_pixels=pixels,
    )


def _codec(snapshot_mutation: str | None = None) -> CodecConfigurationIdentity:
    snapshot = {
        "baseline": {"source_codec": "jpeg2000", "rate_control": "emitted_bytes"},
        "preprocessing": {"upsample": "configured"},
        "environment": {"openjpeg": "fixture", "binding": "fixture"},
    }
    if snapshot_mutation is not None:
        snapshot["baseline"]["rate_control"] = snapshot_mutation
    return CodecConfigurationIdentity(
        snapshot=snapshot,
        configuration_hash=sha256_bytes(canonical_json(snapshot)),
        runtime_version="fixture-openjpeg",
    )


def _emitted(codestream: bytes, *, key: str = "g8dsearch-" + "c" * 64, budget: int = 64) -> EmittedFileIdentity:
    return EmittedFileIdentity(
        codec_search_key_id=key,
        codestream_sha256=sha256_bytes(codestream),
        emitted_bytes=len(codestream),
        payload_budget_bytes=budget,
        filler_bytes=budget - len(codestream),
    )


def test_br11_known_answer_covers_multi_tile_and_psot_zero() -> None:
    codestream = _two_tile_codestream()
    row = account_br11(
        codestream,
        emitted_file_identity=_emitted(codestream, budget=50),
        bytes_sent=50,
        verdict="delivered",
    )
    assert (row.header_bytes, row.payload_bytes) == (32, 6)
    assert row.emitted_codestream_bytes == 38
    assert row.payload_filler_bytes == 12

    psot_zero = _codestream(b"abc", psot=0)
    failed = account_br11(
        psot_zero,
        emitted_file_identity=_emitted(psot_zero, key="g8dsearch-" + "d" * 64, budget=50),
        bytes_sent=50,
        verdict="decode_failure",
    )
    assert (failed.header_bytes, failed.payload_bytes) == (18, 3)
    aggregate = aggregate_br11((row, failed))
    assert aggregate.denominator == 2
    assert aggregate.verdict_counts == {"decode_failure": 1, "delivered": 1}
    assert aggregate.as_dict()["header_bytes"] == 25.0
    assert aggregate.as_dict()["payload_bytes"] == 4.5


def test_br11_accounting_includes_decode_failure_and_rejects_mutations() -> None:
    codestream = _codestream(b"abc", psot=17)
    emitted = _emitted(codestream, budget=24)
    row = account_br11(
        codestream,
        emitted_file_identity=emitted,
        bytes_sent=24,
        verdict="decode_failure",
    )
    assert row.as_dict()["verdict"] == "decode_failure"
    with pytest.raises(G8DContractError, match="codestream bytes"):
        account_br11(b"x" + codestream[1:], emitted_file_identity=emitted, bytes_sent=24, verdict="delivered")
    with pytest.raises(G8DContractError, match="bytes_sent"):
        account_br11(codestream, emitted_file_identity=emitted, bytes_sent=23, verdict="delivered")
    with pytest.raises(G8DContractError, match="requires delivered"):
        account_br11(codestream, emitted_file_identity=emitted, bytes_sent=24, verdict=CODEC_INFEASIBILITY)

    mutated = row.as_dict()
    mutated["payload_bytes"] += 1
    with pytest.raises(G8DContractError, match="does not reconcile"):
        BR11Accounting.from_mapping(mutated)


def test_empty_br11_aggregate_has_null_means() -> None:
    result = aggregate_br11(())
    assert result.as_dict()["denominator"] == 0
    assert result.as_dict()["header_bytes"] is None
    assert result.as_dict()["payload_bytes"] is None


class _Decoder:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, codestream: bytes) -> np.ndarray:
        assert codestream
        self.calls += 1
        return np.full((4, 4, 3), 7, dtype=np.uint8)


def test_reconstruction_cache_upsamples_once_and_reuses_immutable_bytes(tmp_path) -> None:
    decoder = _Decoder()
    codestream = _codestream(b"abc", psot=17)
    emitted = _emitted(codestream, budget=30)
    cache = ReconstructionCache(tmp_path, _codec(), decoder=decoder)
    first = cache.get_or_create(
        image_identity=_image(),
        emitted_file_identity=emitted,
        codestream=codestream,
        output_shape=(8, 8, 3),
    )
    second = cache.get_or_create(
        image_identity=_image(),
        emitted_file_identity=emitted,
        codestream=codestream,
        output_shape=(8, 8, 3),
    )
    assert first.reconstruction.shape == (8, 8, 3)
    assert first.reconstruction.dtype == np.uint8
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.cache_object_id == first.cache_object_id
    assert np.array_equal(first.reconstruction, second.reconstruction)
    assert decoder.calls == 1


def test_reconstruction_cache_key_binds_image_codec_and_emitted_content(tmp_path) -> None:
    decoder = _Decoder()
    first_bytes = _codestream(b"abc", psot=17)
    second_bytes = _codestream(b"xyz", psot=17)
    cache = ReconstructionCache(tmp_path, _codec(), decoder=decoder)
    first = cache.get_or_create(
        image_identity=_image(),
        emitted_file_identity=_emitted(first_bytes, budget=30),
        codestream=first_bytes,
        output_shape=(8, 8, 3),
    )
    changed_image = cache.get_or_create(
        image_identity=_image(b"source-v2"),
        emitted_file_identity=_emitted(first_bytes, budget=30),
        codestream=first_bytes,
        output_shape=(8, 8, 3),
    )
    changed_emitted = cache.get_or_create(
        image_identity=_image(),
        emitted_file_identity=_emitted(second_bytes, key="g8dsearch-" + "e" * 64, budget=30),
        codestream=second_bytes,
        output_shape=(8, 8, 3),
    )
    changed_codec = ReconstructionCache(tmp_path, _codec("different"), decoder=decoder).get_or_create(
        image_identity=_image(),
        emitted_file_identity=_emitted(first_bytes, budget=30),
        codestream=first_bytes,
        output_shape=(8, 8, 3),
    )
    assert len({first.identity.identity_id, changed_image.identity.identity_id, changed_emitted.identity.identity_id, changed_codec.identity.identity_id}) == 4
    assert decoder.calls == 4


def test_reconstruction_cache_corruption_and_downsample_fail_closed(tmp_path) -> None:
    decoder = _Decoder()
    codestream = _codestream(b"abc", psot=17)
    emitted = _emitted(codestream, budget=30)
    cache = ReconstructionCache(tmp_path, _codec(), decoder=decoder)
    result = cache.get_or_create(
        image_identity=_image(),
        emitted_file_identity=emitted,
        codestream=codestream,
        output_shape=(8, 8, 3),
    )
    path = tmp_path / "reconstruction" / f"{result.identity.identity_id}.json"
    corrupted = json.loads(path.read_bytes())
    corrupted["decoded_pixels_sha256"] = "0" * 64
    path.write_text(json.dumps(corrupted))
    with pytest.raises(G8DContractError):
        cache.get_or_create(
            image_identity=_image(),
            emitted_file_identity=emitted,
            codestream=codestream,
            output_shape=(8, 8, 3),
        )

    class LargerDecoder:
        def __call__(self, codestream: bytes) -> np.ndarray:
            return np.zeros((9, 9, 3), dtype=np.uint8)

    with pytest.raises(G8DContractError, match="downsample"):
        ReconstructionCache(tmp_path / "larger", _codec(), decoder=LargerDecoder()).get_or_create(
            image_identity=_image(),
            emitted_file_identity=emitted,
            codestream=codestream,
            output_shape=(8, 8, 3),
        )
