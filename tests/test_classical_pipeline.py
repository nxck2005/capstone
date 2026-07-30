"""Classical end-to-end pipeline: verdict taxonomy, budgets, cache and isolation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

import baseline.classical.pipeline as pipeline
import data.preprocessing as preprocessing
from baseline.classical.pipeline import (
    BUDGET_EXCEEDED,
    CODEC_CONFIGURATION_ERROR,
    CODEC_INFEASIBILITY,
    DECODE_FAILURE,
    DELIVERED,
    STRUCTURAL_INFEASIBILITY,
    VERDICTS,
    ChannelIdentity,
    ClassicalPipelineError,
    configured_axes,
    run_classical_pipeline,
)
from baseline.j2k import J2KCodec
from baseline.ldpc.transport import build_packet_plan
from config.params import get
from data.preprocessing import canonicalize_source, codec_input
from data.registry import DatasetRegistryError, load_dataset

HIGH_SNR_DB = 20.0
LOW_SNR_DB = -10.0
CIFAR10_K = get("bandwidth.k_symbols.cifar10")


def _synthetic_rgb(axis: int = 64) -> np.ndarray:
    rows, columns = np.indices((axis, axis))
    return np.stack(
        (
            (rows * 7 + columns * 3) % 256,
            (rows * 11 + columns * 5) % 256,
            (rows * 13 + columns * 17) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


@pytest.fixture(autouse=True)
def fixture_source_decoders(monkeypatch: pytest.MonkeyPatch):
    decoders = {
        dataset: (lambda source_bytes: _synthetic_rgb())
        for dataset in ("cifar10", "stl10", "imagenette160")
    }
    monkeypatch.setattr(preprocessing, "_SOURCE_DECODERS", decoders)


@pytest.fixture
def product():
    return canonicalize_source(b"classical/pipeline/sample-a", "cifar10")


@pytest.fixture
def codec(tmp_path: Path):
    return J2KCodec(tmp_path / "j2k-cache")


@pytest.fixture
def identity():
    return ChannelIdentity(
        dataset_version="dataset-version-fixture",
        split_manifest_hash="split-manifest-fixture",
        channel_seed=int(get("evaluation.channel_seeds")[0]),
    )


def _run(product, codec, identity, **kwargs):
    parameters = {
        "dataset": "cifar10",
        "k_symbols": CIFAR10_K["r_1_2"],
        "modulation": get("baseline.core_modulation"),
        "ldpc_rate": "1/2",
        "snr_db": HIGH_SNR_DB,
    }
    parameters.update(kwargs)
    return run_classical_pipeline(
        product, codec=codec, channel_identity=identity, **parameters
    )


# --- the three-way failure taxonomy ------------------------------------------


def test_high_snr_round_trip_delivers_a_decoded_canonical_image(
    product, codec, identity
):
    result = _run(product, codec, identity)
    assert result.verdict == DELIVERED and result.delivered
    assert result.codestream_recovered_exactly is True
    assert result.decoded_image is not None
    assert result.decoded_image.shape == codec_input(product).shape
    assert result.decoded_image.dtype == np.uint8
    assert result.transport is not None and result.transport.crc_ok
    assert result.accounting is not None and result.accounting.reconciles
    assert result.noise_id and len(result.noise_id) == len(hashlib.sha256().hexdigest())


def test_structural_infeasibility_is_detected_before_any_encoding(
    product, codec, identity
):
    result = _run(
        product, codec, identity, k_symbols=CIFAR10_K["r_1_48"], modulation="bpsk",
        ldpc_rate="1/3",
    )
    assert result.verdict == STRUCTURAL_INFEASIBILITY
    assert result.packet_feasible is False
    assert result.structural_reason == "no_legal_byte_aligned_A_within_nominal_budget"
    # nothing downstream ran
    assert result.accounting is None
    assert result.source_coding is None
    assert result.transport is None
    assert result.noise_id is None
    assert not tuple((codec.cache_root).glob("*.j2kcache")) if codec.cache_root.exists() else True


def test_codec_infeasibility_is_distinct_from_structural_infeasibility(
    product, codec, identity
):
    result = _run(
        product, codec, identity, k_symbols=CIFAR10_K["r_1_48"], modulation="qpsk",
        ldpc_rate="1/2",
    )
    assert result.verdict == CODEC_INFEASIBILITY
    # a packetisation *does* exist here, which is what makes this the other class
    assert result.packet_feasible is True
    assert result.structural_reason is None
    assert result.accounting is not None and result.accounting.reconciles
    assert result.source_coding is not None and not result.source_coding.feasible
    assert result.source_coding.axes_attempted == configured_axes("cifar10", 32)
    assert result.transport is None
    reasons = dict(result.source_coding.axis_reasons)
    assert set(reasons) == set(result.source_coding.axes_attempted)
    assert reasons[32] == BUDGET_EXCEEDED
    assert all(
        value == BUDGET_EXCEEDED or value.startswith(CODEC_CONFIGURATION_ERROR)
        for value in reasons.values()
    )


def test_decode_failure_is_its_own_verdict_after_a_real_transmission(
    product, codec, identity
):
    result = _run(product, codec, identity, snr_db=LOW_SNR_DB)
    assert result.verdict == DECODE_FAILURE
    assert result.packet_feasible is True
    assert result.source_coding is not None and result.source_coding.feasible
    assert result.transport is not None and not result.transport.crc_ok
    assert result.decoded_image is None
    assert result.codestream_recovered_exactly is None
    # the transmission happened, so its measurements survive the failure
    assert result.transport.realised_symbol_energy > 0
    assert result.noise_id is not None


def test_every_verdict_is_one_of_the_declared_four(product, codec, identity):
    observed = set()
    for k_symbols, modulation, rate, snr in (
        (CIFAR10_K["r_1_2"], "qpsk", "1/2", HIGH_SNR_DB),
        (CIFAR10_K["r_1_2"], "qpsk", "1/2", LOW_SNR_DB),
        (CIFAR10_K["r_1_48"], "bpsk", "1/3", HIGH_SNR_DB),
        (CIFAR10_K["r_1_48"], "qpsk", "1/2", HIGH_SNR_DB),
    ):
        result = _run(
            product, codec, identity, k_symbols=k_symbols, modulation=modulation,
            ldpc_rate=rate, snr_db=snr,
        )
        assert result.verdict in VERDICTS
        observed.add(result.verdict)
    assert observed == set(VERDICTS)


# --- budgets, filler and the emitted-byte authority ---------------------------


def test_emitted_codestream_is_authoritative_and_slack_becomes_filler(
    product, codec, identity
):
    assert get("baseline.j2k_emitted_size_authoritative") is True
    result = _run(product, codec, identity)
    source = result.source_coding
    accounting = result.accounting
    assert source is not None and accounting is not None
    assert source.payload_capacity_bytes == accounting.payload_bytes
    # the measured emitted size, not the requested ratio, is what counts
    assert 0 < source.emitted_bytes <= source.payload_capacity_bytes
    assert (
        source.emitted_bytes + source.payload_filler_bytes
        == source.payload_capacity_bytes
    )
    assert source.payload_filler_bits == source.payload_filler_bytes * 8
    assert source.payload_filler_bits + source.emitted_bytes * 8 == accounting.payload_bits


def test_j2k_cache_identity_binds_exactly_the_configured_cache_key(product, codec):
    """Every ``baseline.j2k_cache_key`` field is bound, and each one moves the key.

    The implementation spells the library-version field ``openjpeg_version``
    where the parameter spells it ``j2k_impl_version``.  The bound *value* is
    the same, and the field is not renamed here because the committed
    transparency-probe evidence records cache keys produced under the current
    spelling.
    """

    pixels = hashlib.sha256(codec_input(product).tobytes()).hexdigest()
    base = {
        "canonical_pixels_sha256": pixels,
        "budget_bytes": 190,
        "encode_axis_px": 32,
    }
    key, cache_identity = codec._cache_identity(**base)

    bound = dict(cache_identity)
    assert bound.pop("openjpeg_version") == get("baseline.j2k_impl_version")
    bound["j2k_impl_version"] = get("baseline.j2k_impl_version")
    assert set(bound) == set(get("baseline.j2k_cache_key"))
    assert cache_identity["codec_config_hash"] == codec.configuration_hash
    assert cache_identity["canonical_pixels_sha256"] == pixels

    for field, value in (
        ("canonical_pixels_sha256", hashlib.sha256(b"other").hexdigest()),
        ("budget_bytes", 191),
        ("encode_axis_px", 24),
    ):
        other, _ = codec._cache_identity(**{**base, field: value})
        assert other != key, field


def test_repeated_invocation_hits_the_cache_and_is_byte_identical(
    product, codec, identity
):
    first = _run(product, codec, identity)
    second = _run(product, codec, identity)
    assert first.source_coding.cache_hit is False
    assert second.source_coding.cache_hit is True
    assert first.source_coding.cache_key == second.source_coding.cache_key
    assert first.source_coding.codestream_sha256 == second.source_coding.codestream_sha256
    assert first.source_coding.emitted_bytes == second.source_coding.emitted_bytes
    np.testing.assert_array_equal(first.decoded_image, second.decoded_image)
    assert first.transport.unit_noise_sha256 == second.transport.unit_noise_sha256


def test_configured_axes_are_descending_and_never_upscale():
    assert get("baseline.downsample_axis_never_upscales") is True
    for dataset, shorter_side in (("cifar10", 32), ("imagenette160", 160), ("stl10", 96)):
        axes = configured_axes(dataset, shorter_side)
        assert axes == tuple(sorted(axes, reverse=True))
        assert max(axes) <= shorter_side
        assert set(axes) <= {int(value) for value in get("baseline.downsample_axis_px")[dataset]}
    assert configured_axes("imagenette160", 96) == (96, 64)
    with pytest.raises(ClassicalPipelineError, match="no configured downsample axis"):
        configured_axes("imagenette160", 32)


def test_requested_axis_may_not_upscale_the_source(product, codec, identity):
    with pytest.raises(ClassicalPipelineError, match="would upscale"):
        _run(product, codec, identity, encode_axis_px=64)


# --- explicit encode axes are a selection, not a second configuration source ---


def test_configured_explicit_axis_is_accepted(product, codec, identity):
    assert 32 in configured_axes("cifar10", 32)
    outcome = _run(product, codec, identity, encode_axis_px=32)
    assert outcome.source_coding is not None
    assert outcome.source_coding.encode_axis_px == 32
    assert outcome.source_coding.axes_attempted == (32,)


def test_unconfigured_explicit_axis_is_rejected(product, codec, identity):
    """A smaller, non-upscaling, but unconfigured axis must not reach the codec."""

    axes = {int(value) for value in get("baseline.downsample_axis_px")["cifar10"]}
    assert 28 not in axes and 28 < 32
    with pytest.raises(ClassicalPipelineError, match="is not configured"):
        _run(product, codec, identity, encode_axis_px=28)


def test_unconfigured_explicit_axis_is_rejected_before_the_codec_runs(
    product, codec, identity, monkeypatch
):
    """Rejection must precede JPEG 2000, or it produces cache keys and evidence
    for a configuration the spec never authorised."""

    def refuse(*args, **kwargs):
        raise AssertionError("the codec must not run for an unconfigured axis")

    monkeypatch.setattr(J2KCodec, "encode_to_budget", refuse)
    monkeypatch.setattr(pipeline, "codec_downsample", refuse)

    with pytest.raises(ClassicalPipelineError, match="is not configured"):
        _run(product, codec, identity, encode_axis_px=28)
    # the same guard fires for an upscaling axis, also before any encoding
    with pytest.raises(ClassicalPipelineError, match="would upscale"):
        _run(product, codec, identity, encode_axis_px=64)


def test_automatic_axis_iteration_is_descending_and_only_configured_axes(
    product, codec, identity, monkeypatch
):
    seen: list[int] = []
    original = pipeline.codec_downsample

    def spy(image, axis):
        seen.append(int(axis))
        return original(image, axis)

    monkeypatch.setattr(pipeline, "codec_downsample", spy)
    _run(product, codec, identity)

    permitted = configured_axes("cifar10", 32)
    assert seen, "automatic selection must attempt at least one axis"
    assert seen == sorted(seen, reverse=True)
    assert set(seen) <= set(permitted)
    assert seen[0] == max(permitted)


def test_unconfigured_ldpc_rate_is_rejected(product, codec, identity):
    with pytest.raises(ValueError, match="unconfigured LDPC rate"):
        _run(product, codec, identity, ldpc_rate="3/4")


# --- identity and split isolation --------------------------------------------


def test_noise_identity_covers_the_declared_key_and_separates_invocations(identity):
    packet = build_packet_plan(CIFAR10_K["r_1_2"], "qpsk", "1/2")
    assert packet.feasible
    base = {
        "stable_sample_id": "sample-a",
        "test_snr_db": HIGH_SNR_DB,
        "k": CIFAR10_K["r_1_2"],
        "block_index": 0,
    }
    reference = identity.noise_id(**base)
    assert reference == identity.noise_id(**base)
    for field, value in (
        ("stable_sample_id", "sample-b"),
        ("test_snr_db", 0.0),
        ("k", CIFAR10_K["r_1_3"]),
        ("block_index", 1),
    ):
        assert identity.noise_id(**{**base, field: value}) != reference
    other_seed = ChannelIdentity(
        dataset_version=identity.dataset_version,
        split_manifest_hash=identity.split_manifest_hash,
        channel_seed=identity.channel_seed + 1,
    )
    assert other_seed.noise_id(**base) != reference


def test_pipeline_loads_validation_data_only_and_test_access_stays_sealed():
    with pytest.raises(DatasetRegistryError, match="sealed"):
        load_dataset("cifar10", "test")
    assert get("evaluation.test_access_gate") in ("G-12", "W11")
    source = Path(__file__).resolve().parents[1] / "src/baseline/classical"
    for module in sorted(source.glob("*.py")):
        text = module.read_text(encoding="utf-8")
        assert "test_access" not in text
        assert '"test"' not in text
