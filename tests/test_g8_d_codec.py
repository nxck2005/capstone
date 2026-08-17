from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from baseline.g8_d import (
    CODEC_FEASIBLE,
    CODEC_INFEASIBILITY,
    STRUCTURAL_INFEASIBILITY,
    BudgetIdentity,
    CodecConfigurationIdentity,
    CodecSearchEngine,
    G8DContractError,
    ImageIdentity,
    ValidationSplitIdentity,
    canonical_json,
    sha256_bytes,
)


def _identity(source: bytes = b"source", pixels: np.ndarray | None = None) -> ImageIdentity:
    if pixels is None:
        pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    split = ValidationSplitIdentity("fixture", "val", "a" * 64, "b" * 64)
    return ImageIdentity.from_pixels(
        split_identity=split,
        stable_sample_id=hashlib.sha256(source).hexdigest()[:16],
        source_bytes=source,
        canonical_pixels=pixels,
    )


def _budget(payload_bytes: int = 20) -> BudgetIdentity:
    return BudgetIdentity(
        bw_ratio="fixture",
        bytes_sent=payload_bytes,
        payload_bytes=payload_bytes,
        packet_accounting={"payload_bytes": payload_bytes, "channel_bits": payload_bytes * 8},
    )


class FakeCodec:
    def __init__(self, mode: str = "feasible", codestream: bytes = b"0123456789") -> None:
        self.mode = mode
        self.codestream = codestream
        self.calls = 0
        self.snapshot = {
            "baseline": {"source_codec": "jpeg2000", "rate_control": "emitted_bytes"},
            "preprocessing": {"downsample": "bilinear", "upsample": "bicubic"},
            "environment": {"openjpeg": "fixture", "binding": "fixture"},
        }
        self.configuration_hash = sha256_bytes(canonical_json(self.snapshot))

    def encode_to_budget(self, image: np.ndarray, **kwargs: object) -> SimpleNamespace:
        self.calls += 1
        if self.mode == "infeasible":
            return SimpleNamespace(
                feasible=False,
                codestream=None,
                emitted_byte_count=None,
                compression_ratio_argument=1.0,
                search_trace=(),
                cache_key="fake-infeasible",
            )
        return SimpleNamespace(
            feasible=True,
            codestream=self.codestream,
            emitted_byte_count=len(self.codestream),
            compression_ratio_argument=1.0,
            search_trace=(
                SimpleNamespace(iteration=1, compression_ratio=1.0, emitted_bytes=len(self.codestream), within_budget=len(self.codestream) <= int(kwargs["budget_bytes"])),
            ),
            cache_key="fake-backend-cache",
        )


def _engine(tmp_path, backend: FakeCodec | None = None) -> tuple[CodecSearchEngine, FakeCodec]:
    backend = backend or FakeCodec()
    identity = CodecConfigurationIdentity(
        backend.snapshot,
        backend.configuration_hash,
        "fixture",
    )
    return CodecSearchEngine(tmp_path, backend=backend, codec_identity=identity), backend


def test_emitted_bytes_are_authoritative_and_search_is_cached(tmp_path) -> None:
    engine, backend = _engine(tmp_path)
    image = _identity()
    first = engine.search(
        image_identity=image,
        encoded_image=np.zeros((8, 8, 3), dtype=np.uint8),
        budget=_budget(20),
        encode_axis_px=8,
    )
    second = engine.search(
        image_identity=image,
        encoded_image=np.zeros((8, 8, 3), dtype=np.uint8),
        budget=_budget(20),
        encode_axis_px=8,
    )
    assert first.status == second.status == CODEC_FEASIBLE
    assert first.emitted_identity is not None
    assert first.emitted_identity.emitted_bytes == len(b"0123456789")
    assert first.emitted_identity.filler_bytes == 10
    assert first.requested_compression_ratio == 1.0  # provenance only
    assert first.cache_hit is False and second.cache_hit is True
    assert backend.calls == 1


def test_structural_infeasibility_is_distinct_and_does_not_call_codec(tmp_path) -> None:
    engine, backend = _engine(tmp_path)
    result = engine.search(
        image_identity=_identity(),
        encoded_image=np.zeros((8, 8, 3), dtype=np.uint8),
        budget=_budget(),
        encode_axis_px=8,
        structurally_feasible=False,
        structural_reason="no legal packet plan",
    )
    assert result.status == STRUCTURAL_INFEASIBILITY
    assert result.reason == "no legal packet plan"
    assert backend.calls == 0


def test_packet_plan_path_binds_actual_payload_accounting(tmp_path) -> None:
    from baseline.classical.channel_transport import build_accounting
    from baseline.ldpc.transport import build_packet_plan

    packet = build_packet_plan(128, "qpsk", "1/2")
    accounting = build_accounting(packet)
    engine, backend = _engine(tmp_path)
    budget = BudgetIdentity(
        bw_ratio="fixture",
        bytes_sent=accounting.payload_bytes,
        payload_bytes=accounting.payload_bytes,
        packet_accounting=accounting.as_dict(),
    )
    result = engine.search_with_packet_plan(
        image_identity=_identity(),
        encoded_image=np.zeros((8, 8, 3), dtype=np.uint8),
        budget=budget,
        encode_axis_px=8,
        k_symbols=128,
        modulation="qpsk",
        ldpc_rate="1/2",
    )
    assert result.status == CODEC_FEASIBLE
    assert backend.calls == 1


def test_codec_infeasibility_is_preserved_and_cached(tmp_path) -> None:
    engine, backend = _engine(tmp_path, FakeCodec(mode="infeasible"))
    kwargs = {
        "image_identity": _identity(),
        "encoded_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "budget": _budget(),
        "encode_axis_px": 8,
    }
    first = engine.search(**kwargs)
    second = engine.search(**kwargs)
    assert first.status == second.status == CODEC_INFEASIBILITY
    assert first.emitted_codestream is None and first.emitted_identity is None
    assert second.cache_hit is True
    assert backend.calls == 1


def test_cache_key_changes_for_source_content_and_codec_configuration(tmp_path) -> None:
    engine, backend = _engine(tmp_path)
    image = _identity()
    original = engine.search(
        image_identity=image,
        encoded_image=np.zeros((8, 8, 3), dtype=np.uint8),
        budget=_budget(),
        encode_axis_px=8,
    )
    changed_source = engine.search(
        image_identity=_identity(b"different-source"),
        encoded_image=np.zeros((8, 8, 3), dtype=np.uint8),
        budget=_budget(),
        encode_axis_px=8,
    )
    assert original.search_key.identity_id != changed_source.search_key.identity_id
    assert backend.calls == 2

    changed_backend = FakeCodec()
    changed_backend.snapshot["baseline"]["rate_control"] = "mutated"
    changed_backend.configuration_hash = sha256_bytes(canonical_json(changed_backend.snapshot))
    changed_engine, _ = _engine(tmp_path, changed_backend)
    changed_codec = changed_engine.search(
        image_identity=image,
        encoded_image=np.zeros((8, 8, 3), dtype=np.uint8),
        budget=_budget(),
        encode_axis_px=8,
    )
    assert original.search_key.identity_id != changed_codec.search_key.identity_id


def test_actual_emitted_bytes_over_budget_fail_closed(tmp_path) -> None:
    engine, _ = _engine(tmp_path, FakeCodec(codestream=b"x" * 21))
    with pytest.raises(G8DContractError, match="exceeds payload budget"):
        engine.search(
            image_identity=_identity(),
            encoded_image=np.zeros((8, 8, 3), dtype=np.uint8),
            budget=_budget(20),
            encode_axis_px=8,
        )


def test_upscale_axis_is_rejected_before_codec(tmp_path) -> None:
    engine, backend = _engine(tmp_path)
    with pytest.raises(G8DContractError, match="upscale"):
        engine.search(
            image_identity=_identity(),
            encoded_image=np.zeros((9, 9, 3), dtype=np.uint8),
            budget=_budget(),
            encode_axis_px=9,
        )
    assert backend.calls == 0


def test_corrupted_cache_object_is_not_accepted(tmp_path) -> None:
    engine, _ = _engine(tmp_path)
    kwargs = {
        "image_identity": _identity(),
        "encoded_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "budget": _budget(),
        "encode_axis_px": 8,
    }
    result = engine.search(**kwargs)
    path = tmp_path / "codec_search" / f"{result.search_key.identity_id}.json"
    data = path.read_bytes().replace(b"MDEyMzQ1Njc4OQ==", b"eHh4eHh4eHh4eA==")
    path.write_bytes(data)
    with pytest.raises(G8DContractError):
        engine.search(**kwargs)
