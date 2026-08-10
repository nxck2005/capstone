"""Mutation tests: each of the nine W4 classical defect classes must be caught.

Every test here injects one defect and asserts that the project rejects it.  A
test that merely exercises the correct path proves nothing about the guard, so
each case also asserts the unmutated behaviour it is contrasted against.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import baseline.classical.channel_transport as channel_transport
import baseline.classical.pipeline as pipeline
import channels.registry as channel_registry
import data.preprocessing as preprocessing
from baseline.classical.channel_transport import (
    build_accounting,
    modulate,
    transport_round_trip,
)
from baseline.classical.pipeline import (
    STRUCTURAL_INFEASIBILITY,
    ChannelIdentity,
    ClassicalPipelineError,
    SourceCoding,
    run_classical_pipeline,
)
from baseline.j2k import J2KCodec, J2KResult
from baseline.ldpc.adapter import SionnaLDPCAdapter
from baseline.ldpc.modulation import interleave, map_bits
from baseline.ldpc.transport import build_packet_plan, transmit_transport
from channels.awgn import keyed_complex_noise
from config.params import get

HIGH_SNR_DB = 20.0
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
    monkeypatch.setattr(
        preprocessing,
        "_SOURCE_DECODERS",
        {
            dataset: (lambda source_bytes: _synthetic_rgb())
            for dataset in ("cifar10", "stl10", "imagenette160")
        },
    )


@pytest.fixture
def product():
    return preprocessing.canonicalize_source(b"classical/mutation/sample", "cifar10")


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


def _packet(modulation: str = "qpsk", rate: str = "1/2", k: int = 256):
    packet = build_packet_plan(k, modulation, rate)
    assert packet.feasible
    return packet, build_accounting(packet)


def _payload(accounting, seed: int = 5) -> np.ndarray:
    return np.random.default_rng(seed).integers(
        0, 2, size=accounting.payload_bits, dtype=np.uint8
    )


def _run(product, codec, identity, **kwargs):
    parameters = {
        "dataset": "cifar10",
        "k_symbols": CIFAR10_K["r_1_2"],
        "modulation": "qpsk",
        "ldpc_rate": "1/2",
        "snr_db": HIGH_SNR_DB,
    }
    parameters.update(kwargs)
    return run_classical_pipeline(
        product, codec=codec, channel_identity=identity, **parameters
    )


# 1 --- LLR sign reversal ------------------------------------------------------


def test_llr_sign_reversal_breaks_the_round_trip(monkeypatch: pytest.MonkeyPatch):
    packet, accounting = _packet()
    payload = _payload(accounting)
    clean = transport_round_trip(
        payload, packet, snr_db=HIGH_SNR_DB, noise_id="llr-sign-clean"
    )
    assert clean.crc_ok

    original = channel_transport.max_log_llr
    monkeypatch.setattr(
        channel_transport,
        "max_log_llr",
        lambda symbols, modulation, n0: -original(symbols, modulation, n0),
    )
    mutated = transport_round_trip(
        payload, packet, snr_db=HIGH_SNR_DB, noise_id="llr-sign-clean"
    )
    assert not mutated.crc_ok
    assert mutated.payload_bits is None


# 2 --- disabled 16-QAM bit interleaver ---------------------------------------


def test_disabled_qam16_interleaver_is_rejected_and_corrupts_the_link(
    monkeypatch: pytest.MonkeyPatch,
):
    packet, accounting = _packet(modulation="qam16")
    payload = _payload(accounting)
    assert transport_round_trip(
        payload, packet, snr_db=HIGH_SNR_DB, noise_id="interleaver-clean"
    ).crc_ok

    # (a) turning the requirement off is refused outright, not silently honoured
    original_get = channel_transport.get
    monkeypatch.setattr(
        channel_transport,
        "get",
        lambda path: (
            False
            if path == "baseline.modulation_bit_interleaver_required"
            else original_get(path)
        ),
    )
    with pytest.raises(NotImplementedError, match="modulation_bit_interleaver_required"):
        modulate(transmit_transport(payload, packet), "qam16")
    monkeypatch.undo()

    # (b) applying it a *second* time at the transmitter — the PB_1C defect —
    # destroys the link, because the receiver only ever undoes it once
    clean = transport_round_trip(
        payload, packet, snr_db=HIGH_SNR_DB, noise_id="interleaver-clean"
    )
    assert clean.crc_ok

    original = channel_transport.mapper_input_bits

    def doubly_interleaved(blocks, modulation):
        q_m = channel_transport.bits_per_symbol(modulation)
        bits = original(blocks, modulation)
        offset = 0
        pieces = []
        for block in blocks:
            size = np.asarray(block).size
            pieces.append(interleave(bits[offset : offset + size], q_m))
            offset += size
        return np.concatenate(pieces)

    monkeypatch.setattr(channel_transport, "mapper_input_bits", doubly_interleaved)
    mutated = transport_round_trip(
        payload, packet, snr_db=HIGH_SNR_DB, noise_id="interleaver-clean"
    )
    assert not mutated.crc_ok


def test_qam16_interleaver_is_not_a_no_op():
    """Non-trivial at the seam that owns it: the adapter, not ``modulate()``."""

    packet, accounting = _packet(modulation="qam16")
    layout = packet.segmentation
    assert layout is not None
    adapter = SionnaLDPCAdapter(
        layout.k_prime, packet.e_r[0], packet.q_m, layout.base_graph, "cpu"
    )
    assert adapter.encoder.num_bits_per_symbol == 4
    assert not np.array_equal(
        adapter.encoder.out_int.numpy(), np.arange(packet.e_r[0])
    )
    # and the project layer must not add a second one
    blocks = transmit_transport(_payload(accounting), packet)
    assert np.array_equal(
        modulate(blocks, "qam16"), map_bits(np.concatenate(blocks), "qam16")
    )


# 3 --- wrong k ----------------------------------------------------------------


def test_wrong_k_fails_reconciliation_and_stops_transmission():
    packet, accounting = _packet()
    assert accounting.reconciles

    # channel budget no longer equals the sum of the per-block rate-matched bits
    inflated = dataclasses.replace(packet, channel_bits=packet.channel_bits + packet.q_m)
    mutated = build_accounting(inflated)
    assert not mutated.channel_reconciles and not mutated.reconciles
    with pytest.raises(AssertionError, match="does not reconcile"):
        transport_round_trip(
            _payload(accounting), inflated, snr_db=HIGH_SNR_DB, noise_id="wrong-k"
        )

    # a channel budget that is not a whole number of symbols is not exact
    ragged = build_accounting(
        dataclasses.replace(packet, channel_bits=packet.channel_bits + 1, e_r=(packet.channel_bits + 1,))
    )
    assert not ragged.channel_uses_exact and not ragged.reconciles


def test_pipeline_rejects_a_plan_built_for_a_different_k(
    product, codec, identity, monkeypatch: pytest.MonkeyPatch
):
    wrong = build_packet_plan(CIFAR10_K["r_1_3"], "qpsk", "1/2")
    monkeypatch.setattr(pipeline, "build_packet_plan", lambda *args, **kwargs: wrong)
    with pytest.raises(ClassicalPipelineError, match="does not carry the requested k"):
        _run(product, codec, identity)


# 4 --- dropped filler ---------------------------------------------------------


def test_dropped_ldpc_filler_fails_reconciliation():
    packet, accounting = _packet()
    assert accounting.ldpc_filler_bits_total > 0
    layout = packet.segmentation
    mutated = build_accounting(
        dataclasses.replace(
            packet, segmentation=dataclasses.replace(layout, filler_bits_per_block=0)
        )
    )
    assert not mutated.systematic_reconciles and not mutated.reconciles


@pytest.mark.external_codec_runtime
def test_dropped_payload_filler_is_caught_before_transmission(
    product, codec, identity, monkeypatch: pytest.MonkeyPatch
):
    original = pipeline._encode_source

    def understating_filler(**kwargs):
        source, result, image = original(**kwargs)
        assert source.payload_filler_bytes
        return dataclasses.replace(source, payload_filler_bytes=0), result, image

    monkeypatch.setattr(pipeline, "_encode_source", understating_filler)
    with pytest.raises(ClassicalPipelineError, match="does not fill the transport block"):
        _run(product, codec, identity)


# 5 --- unaccounted CRC --------------------------------------------------------


def test_unaccounted_transport_block_crc_fails_reconciliation():
    packet, accounting = _packet()
    assert accounting.tb_crc_bits > 0
    mutated = build_accounting(
        dataclasses.replace(
            packet,
            segmentation=dataclasses.replace(packet.segmentation, tb_crc_bits=0),
        )
    )
    assert not mutated.systematic_reconciles and not mutated.reconciles


def test_unaccounted_code_block_crc_fails_reconciliation():
    packet, accounting = _packet(modulation="qam16", k=3200, rate="2/3")
    assert accounting.code_blocks > 1 and accounting.cb_crc_bits_total > 0
    mutated = build_accounting(
        dataclasses.replace(
            packet,
            segmentation=dataclasses.replace(packet.segmentation, code_block_crc_bits=0),
        )
    )
    assert mutated.cb_crc_bits_total == 0
    assert not mutated.systematic_reconciles and not mutated.reconciles


# 6 --- codec size above budget ------------------------------------------------


@pytest.mark.external_codec_runtime
def test_codestream_above_the_payload_budget_is_refused(
    product, codec, identity, monkeypatch: pytest.MonkeyPatch
):
    original = J2KCodec.encode_to_budget

    def oversized(self, image, **kwargs):
        result = original(self, image, **kwargs)
        if not result.feasible or result.codestream is None:
            return result
        overshoot = result.requested_budget_bytes + 1 - result.emitted_byte_count
        assert overshoot > 0
        return dataclasses.replace(
            result,
            codestream=result.codestream + b"\x00" * overshoot,
            emitted_byte_count=result.emitted_byte_count + overshoot,
        )

    monkeypatch.setattr(J2KCodec, "encode_to_budget", oversized)
    with pytest.raises(ClassicalPipelineError, match="exceeds the payload capacity"):
        _run(product, codec, identity)


@pytest.mark.external_codec_runtime
def test_every_delivered_codestream_is_within_its_budget(product, codec, identity):
    result = _run(product, codec, identity)
    source = result.source_coding
    assert source.emitted_bytes <= source.payload_capacity_bytes
    assert source.payload_filler_bytes >= 0


# 7 --- silently skipped infeasibility ----------------------------------------


def test_infeasible_configurations_return_a_verdict_and_are_never_skipped(
    product, codec, identity
):
    # the one structurally infeasible cell in the committed packetisation record
    result = _run(
        product, codec, identity, k_symbols=CIFAR10_K["r_1_48"], modulation="bpsk",
        ldpc_rate="1/3",
    )
    assert result.verdict == STRUCTURAL_INFEASIBILITY
    assert result.structural_reason
    # and it cannot be walked past: accounting refuses an infeasible plan
    with pytest.raises(ValueError, match="structurally infeasible"):
        build_accounting(build_packet_plan(CIFAR10_K["r_1_48"], "bpsk", "1/3"))
    with pytest.raises(ValueError, match="structurally infeasible"):
        transmit_transport(
            np.zeros(8, dtype=np.uint8),
            build_packet_plan(CIFAR10_K["r_1_48"], "bpsk", "1/3"),
        )


@pytest.mark.external_codec_runtime
def test_codec_infeasibility_records_a_reason_for_every_axis_it_tried(
    product, codec, identity
):
    result = _run(
        product, codec, identity, k_symbols=CIFAR10_K["r_1_48"], modulation="qpsk",
        ldpc_rate="1/2",
    )
    source = result.source_coding
    assert not source.feasible
    assert source.axes_attempted
    assert {axis for axis, _ in source.axis_reasons} == set(source.axes_attempted)
    assert all(reason for _, reason in source.axis_reasons)


# 8 --- a different channel implementation substituted -------------------------


class _LookalikeAWGN(nn.Module):
    """A plausible but separate AWGN: same shape, its own noise scaling."""

    def forward(self, symbols, snr_db, *, unit_noise=None):
        scale = 10.0 ** (-float(snr_db) / 20.0)
        return symbols + (unit_noise if unit_noise is not None else 0) * scale


def test_a_substituted_channel_implementation_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
):
    packet, accounting = _packet()
    payload = _payload(accounting)
    assert transport_round_trip(
        payload, packet, snr_db=HIGH_SNR_DB, noise_id="channel-clean"
    ).crc_ok

    monkeypatch.setitem(
        channel_registry._CHANNELS,
        channel_transport._CHANNEL_MODEL,
        lambda **kwargs: _LookalikeAWGN(),
    )
    with pytest.raises(RuntimeError, match="must use the shared channels.awgn.AWGN"):
        transport_round_trip(
            payload, packet, snr_db=HIGH_SNR_DB, noise_id="channel-clean"
        )


def test_a_locally_constructed_channel_bypassing_the_registry_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
):
    packet, accounting = _packet()
    monkeypatch.setattr(
        channel_transport, "build_channel", lambda name, **kwargs: _LookalikeAWGN()
    )
    with pytest.raises(RuntimeError, match="must use the shared channels.awgn.AWGN"):
        transport_round_trip(
            _payload(accounting), packet, snr_db=HIGH_SNR_DB, noise_id="channel-local"
        )


def test_an_unregistered_channel_model_is_rejected(monkeypatch: pytest.MonkeyPatch):
    packet, accounting = _packet()
    monkeypatch.setattr(channel_transport, "_CHANNEL_MODEL", "rayleigh_block")
    with pytest.raises(NotImplementedError, match="models_supported"):
        transport_round_trip(
            _payload(accounting), packet, snr_db=HIGH_SNR_DB, noise_id="channel-unknown"
        )


# 9 --- sequential rather than keyed noise -------------------------------------


def _keyed_noise_invariance_holds(packet, accounting) -> bool:
    """The same noise identity must yield the same realisation, always."""

    payload = _payload(accounting)
    first = transport_round_trip(
        payload, packet, snr_db=HIGH_SNR_DB, noise_id="invariance"
    )
    keyed_complex_noise("an-unrelated-intervening-draw", 4096)
    transport_round_trip(payload, packet, snr_db=HIGH_SNR_DB, noise_id="another")
    second = transport_round_trip(
        payload, packet, snr_db=HIGH_SNR_DB, noise_id="invariance"
    )
    return first.unit_noise_sha256 == second.unit_noise_sha256


def test_sequential_noise_breaks_the_keyed_invariance_the_project_relies_on(
    monkeypatch: pytest.MonkeyPatch,
):
    packet, accounting = _packet()
    assert _keyed_noise_invariance_holds(packet, accounting)

    generator = torch.Generator().manual_seed(0)

    def sequential(noise_ids, k, *, dtype=torch.complex64, device=None):
        real = torch.randn(1, k, generator=generator)
        imaginary = torch.randn(1, k, generator=generator)
        return torch.complex(real, imaginary).mul(0.5**0.5).to(dtype)

    monkeypatch.setattr(channel_transport, "keyed_complex_noise", sequential)
    assert not _keyed_noise_invariance_holds(packet, accounting)


def test_keyed_noise_is_a_pure_function_of_the_noise_id():
    packet, accounting = _packet()
    outcome = transport_round_trip(
        _payload(accounting), packet, snr_db=HIGH_SNR_DB, noise_id="pure-function"
    )
    import hashlib

    expected = hashlib.sha256(
        keyed_complex_noise("pure-function", accounting.k_symbols).numpy().tobytes()
    ).hexdigest()
    assert outcome.unit_noise_sha256 == expected
    assert get("artifacts.rng_stream") == "counter_based_keyed_not_sequential"
