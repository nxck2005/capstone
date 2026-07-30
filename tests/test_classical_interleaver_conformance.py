"""Independent TS 38.212 §5.4.2.2 transmitter-order conformance for the W4 arm.

PB_1's round-trip tests are *self-consistency* tests: a transmitter permutation
that the receiver undoes is invisible end to end, because CRC still passes, the
codestream is still recovered byte-exactly, and every bit-count identity is
permutation-invariant.  That is exactly how a duplicated modulation bit
interleaver survived PB_1.

So nothing here calls ``modulate()`` to decide what ``modulate()`` should
produce.  The reference is built two ways that do not depend on the project's
own interleaver:

* ``_ts_38212_out_int`` — a test-only, independently spelled implementation of
  the §5.4.2.2 permutation (write by rows of Qm, read by columns);
* ``_encode_uninterleaved`` — a Sionna encoder built *without*
  ``num_bits_per_symbol``, so it rate-matches and stops.

Their composition is the standards-conformant rate-matched, once-interleaved
sequence.  The single owner of that permutation is Sionna (verified against the
installed source at PB_1C/C1.1: ``sionna/phy/fec/ldpc/encoding.py:791-793``
applies it after rate matching, and ``decoding.py:1646-1649`` applies the
inverse before rate recovery), so the project layer must hand the adapter's
output to the mapper *unchanged*.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sionna.phy.fec.ldpc import LDPC5GEncoder

import baseline.classical.channel_transport as channel_transport
from baseline.classical.channel_transport import (
    build_accounting,
    demodulate,
    mapper_input_bits,
    modulate,
    split_llr_blocks,
)
from baseline.ldpc.adapter import SionnaLDPCAdapter
from baseline.ldpc.modulation import bits_per_symbol, max_log_llr, n0_from_esn0_db
from baseline.ldpc.transport import build_packet_plan, transmit_transport
from config.params import get

CIFAR10_K = get("bandwidth.k_symbols.cifar10")


# --- independent reference ----------------------------------------------------


def _ts_38212_out_int(length: int, q_m: int) -> np.ndarray:
    """TS 38.212 §5.4.2.2, written independently of ``ldpc.modulation``.

    The standard writes the rate-matched sequence into a ``Qm``-row matrix row
    by row and reads it out column by column, so output position ``j*Qm + i``
    carries input position ``i*(E/Qm) + j``.  Spelled here as a reshape and a
    transpose rather than as an index generator, so a shared mistake in the
    project's formula cannot hide behind an identically shaped copy of itself.
    """

    if length <= 0 or length % q_m:
        raise ValueError("E must be a positive multiple of Qm")
    return np.arange(length).reshape(q_m, length // q_m).T.reshape(-1)


def _encode_uninterleaved(block: np.ndarray, k_prime: int, e_r: int, bg: int):
    """Rate-matched Sionna output with the §5.4.2.2 interleaver switched off."""

    encoder = LDPC5GEncoder(k=k_prime, n=e_r, bg=f"bg{bg}", device="cpu")
    assert encoder.num_bits_per_symbol is None
    tensor = torch.as_tensor(block, dtype=torch.float32).reshape(1, -1)
    return encoder(tensor).detach().cpu().numpy().astype(np.uint8)[0]


def _plan(modulation: str, ratio: str = "r_1_2", rate: str = "1/2"):
    packet = build_packet_plan(int(CIFAR10_K[ratio]), modulation, rate)
    assert packet.feasible and packet.segmentation is not None
    return packet


def _payload(packet, seed: int) -> np.ndarray:
    assert packet.segmentation is not None
    return np.random.default_rng(seed).integers(
        0, 2, size=packet.segmentation.payload_bits, dtype=np.uint8
    )


def _expected_mapper_input(packet, payload: np.ndarray) -> np.ndarray:
    """Independently derived: rate-match each block, interleave once, concatenate."""

    from baseline.ldpc.segmentation import segment

    layout = packet.segmentation
    assert layout is not None
    expected = []
    for block, e_r in zip(segment(payload, layout), packet.e_r, strict=True):
        rate_matched = _encode_uninterleaved(
            block[: layout.k_prime], layout.k_prime, e_r, layout.base_graph
        )
        expected.append(rate_matched[_ts_38212_out_int(e_r, packet.q_m)])
    return np.concatenate(expected)


# --- the reference is itself checked ------------------------------------------


@pytest.mark.parametrize("q_m", [1, 2, 4])
def test_reference_permutation_matches_the_installed_sionna_encoder(q_m):
    """The test-only formula must agree with the interleaver Sionna really uses."""

    encoder = LDPC5GEncoder(k=200, n=400, num_bits_per_symbol=q_m, bg="bg2", device="cpu")
    assert np.array_equal(encoder.out_int.numpy(), _ts_38212_out_int(400, q_m))
    assert np.array_equal(
        encoder.out_int_inv.numpy(), np.argsort(_ts_38212_out_int(400, q_m))
    )


def test_reference_permutation_is_the_identity_only_for_bpsk():
    assert np.array_equal(_ts_38212_out_int(400, 1), np.arange(400))
    for q_m in (2, 4):
        assert not np.array_equal(_ts_38212_out_int(400, q_m), np.arange(400))


# --- A. complete transmitter ordering -----------------------------------------


@pytest.mark.parametrize("modulation", ["qpsk", "qam16", "bpsk"])
def test_mapper_input_is_the_rate_matched_once_interleaved_sequence(modulation):
    """The bits entering the mapper must be interleaved exactly once.

    Fails against the pre-PB_1C implementation for QPSK and 16-QAM because the
    project applied Sionna's own permutation a second time, making the realised
    transmit order the permutation *squared*.
    """

    packet = _plan(modulation)
    payload = _payload(packet, seed=17)
    observed = mapper_input_bits(transmit_transport(payload, packet), modulation)
    expected = _expected_mapper_input(packet, payload)

    assert observed.shape == expected.shape
    assert np.array_equal(observed, expected)


@pytest.mark.parametrize("modulation", ["qpsk", "qam16"])
def test_mapper_input_is_not_the_doubly_interleaved_sequence(modulation):
    """Name the specific defect, so a regression cannot pass quietly."""

    packet = _plan(modulation)
    payload = _payload(packet, seed=23)
    observed = mapper_input_bits(transmit_transport(payload, packet), modulation)

    doubled = []
    offset = 0
    expected = _expected_mapper_input(packet, payload)
    for e_r in packet.e_r:
        block = expected[offset : offset + e_r]
        doubled.append(block[_ts_38212_out_int(e_r, packet.q_m)])
        offset += e_r
    doubled = np.concatenate(doubled)

    assert not np.array_equal(doubled, expected), "the defect must be observable"
    assert not np.array_equal(observed, doubled)


def test_mapped_symbols_are_the_reference_symbols():
    """Carry the ordering conclusion through to the actual channel input."""

    packet = _plan("qam16")
    payload = _payload(packet, seed=5)
    from baseline.ldpc.modulation import map_bits

    symbols = modulate(transmit_transport(payload, packet), "qam16")
    reference = map_bits(_expected_mapper_input(packet, payload), "qam16")
    assert np.array_equal(symbols, reference)


# --- B. single-owner seam -----------------------------------------------------


@pytest.mark.parametrize("modulation", ["qpsk", "qam16"])
def test_adapter_encode_returns_modulation_interleaved_bits(modulation):
    """Sionna, not the project, owns the permutation."""

    packet = _plan(modulation)
    layout = packet.segmentation
    assert layout is not None
    payload = _payload(packet, seed=31)
    from baseline.ldpc.segmentation import segment

    block = segment(payload, layout)[0][: layout.k_prime]
    e_r = packet.e_r[0]

    adapter = SionnaLDPCAdapter(
        layout.k_prime, e_r, packet.q_m, layout.base_graph, "cpu"
    )
    encoded = adapter.encode(block[None, :])[0]
    plain = _encode_uninterleaved(block, layout.k_prime, e_r, layout.base_graph)

    assert np.array_equal(encoded, plain[_ts_38212_out_int(e_r, packet.q_m)])
    assert not np.array_equal(encoded, plain)


def test_adapter_is_built_with_the_selected_modulation_qm():
    """``num_bits_per_symbol`` must equal Qm on every adapter transmit builds."""

    for modulation in ("bpsk", "qpsk", "qam16"):
        packet = _plan(modulation)
        layout = packet.segmentation
        assert layout is not None
        adapter = SionnaLDPCAdapter(
            layout.k_prime, packet.e_r[0], packet.q_m, layout.base_graph, "cpu"
        )
        assert adapter.encoder.num_bits_per_symbol == bits_per_symbol(modulation)
        assert adapter.encoder.num_bits_per_symbol == packet.q_m
        assert adapter.decoder.encoder is adapter.encoder


@pytest.mark.parametrize("modulation", ["qpsk", "qam16"])
def test_encoded_blocks_reach_the_mapper_unchanged(modulation):
    """Concatenation is the *only* thing the project does between the two."""

    packet = _plan(modulation)
    blocks = transmit_transport(_payload(packet, seed=41), packet)
    observed = mapper_input_bits(blocks, modulation)
    assert np.array_equal(observed, np.concatenate(blocks))


def test_multi_code_block_concatenation_is_in_plan_order():
    packet = build_packet_plan(
        int(get("bandwidth.k_symbols.imagenette160")["r_1_24"]), "qam16", "2/3"
    )
    assert packet.feasible and packet.segmentation is not None
    assert packet.segmentation.code_blocks > 1
    blocks = transmit_transport(_payload(packet, seed=53), packet)
    observed = mapper_input_bits(blocks, "qam16")

    offset = 0
    for block, e_r in zip(blocks, packet.e_r, strict=True):
        assert np.array_equal(observed[offset : offset + e_r], block)
        offset += e_r
    assert offset == observed.size == packet.channel_bits


@pytest.mark.parametrize("modulation", ["qpsk", "qam16"])
def test_demapper_llrs_reach_the_decoder_unchanged_except_for_splitting(modulation):
    """The receiver must not apply a second inverse permutation."""

    packet = _plan(modulation)
    accounting = build_accounting(packet)
    symbols = modulate(transmit_transport(_payload(packet, seed=61), packet), modulation)
    n0 = n0_from_esn0_db(20.0)

    flat = max_log_llr(np.asarray(symbols).reshape(1, -1), modulation, n0).reshape(-1)
    blocks = demodulate(symbols, modulation, n0, accounting.rate_matched_bits)

    offset = 0
    for block, e_r in zip(blocks, packet.e_r, strict=True):
        assert block.size == e_r
        assert np.array_equal(block, flat[offset : offset + e_r])
        offset += e_r
    assert offset == flat.size


def test_llr_blocks_are_split_at_exact_e_r_boundaries():
    packet = build_packet_plan(
        int(get("bandwidth.k_symbols.imagenette160")["r_1_24"]), "qam16", "2/3"
    )
    assert packet.feasible
    llrs = np.arange(packet.channel_bits, dtype=np.float32)
    blocks = split_llr_blocks(llrs, packet.e_r)

    assert tuple(block.size for block in blocks) == packet.e_r
    assert np.array_equal(np.concatenate(blocks), llrs)
    with pytest.raises(ValueError):
        split_llr_blocks(llrs[:-1], packet.e_r)


# --- C. required mutations ----------------------------------------------------


def test_mutation_adapter_built_without_num_bits_per_symbol():
    """The sole required interleaver must not be bypassable."""

    packet = _plan("qam16")
    layout = packet.segmentation
    assert layout is not None
    payload = _payload(packet, seed=71)
    from baseline.ldpc.segmentation import segment

    block = segment(payload, layout)[0][: layout.k_prime]
    e_r = packet.e_r[0]

    mutated = _encode_uninterleaved(block, layout.k_prime, e_r, layout.base_graph)
    expected = mutated[_ts_38212_out_int(e_r, packet.q_m)]
    assert not np.array_equal(mutated, expected)


def test_mutation_wrong_qm_passed_to_the_adapter():
    packet = _plan("qam16")
    layout = packet.segmentation
    assert layout is not None
    payload = _payload(packet, seed=73)
    from baseline.ldpc.segmentation import segment

    block = segment(payload, layout)[0][: layout.k_prime]
    e_r = packet.e_r[0]

    correct = SionnaLDPCAdapter(
        layout.k_prime, e_r, packet.q_m, layout.base_graph, "cpu"
    ).encode(block[None, :])[0]
    wrong = SionnaLDPCAdapter(
        layout.k_prime, e_r, 2, layout.base_graph, "cpu"
    ).encode(block[None, :])[0]

    plain = _encode_uninterleaved(block, layout.k_prime, e_r, layout.base_graph)
    assert np.array_equal(correct, plain[_ts_38212_out_int(e_r, 4)])
    assert not np.array_equal(wrong, plain[_ts_38212_out_int(e_r, 4)])


@pytest.mark.parametrize("modulation", ["qpsk", "qam16"])
def test_mutation_additional_transmitter_interleaver(modulation, monkeypatch):
    """Re-introducing the removed project-side permutation must be caught."""

    packet = _plan(modulation)
    payload = _payload(packet, seed=79)
    blocks = transmit_transport(payload, packet)
    expected = _expected_mapper_input(packet, payload)

    original = channel_transport.mapper_input_bits

    def mutated(block_list, mod):
        q_m = bits_per_symbol(mod)
        clean = original(block_list, mod)
        offset = 0
        pieces = []
        for block in block_list:
            size = np.asarray(block).size
            pieces.append(
                clean[offset : offset + size][_ts_38212_out_int(size, q_m)]
            )
            offset += size
        return np.concatenate(pieces)

    monkeypatch.setattr(channel_transport, "mapper_input_bits", mutated)
    assert not np.array_equal(
        channel_transport.mapper_input_bits(blocks, modulation), expected
    )


def test_mutation_additional_receiver_inverse_interleaver():
    """A second inverse permutation breaks the LLR-passthrough property."""

    packet = _plan("qam16")
    accounting = build_accounting(packet)
    symbols = modulate(transmit_transport(_payload(packet, seed=83), packet), "qam16")
    n0 = n0_from_esn0_db(20.0)
    flat = max_log_llr(np.asarray(symbols).reshape(1, -1), "qam16", n0).reshape(-1)

    clean = demodulate(symbols, "qam16", n0, accounting.rate_matched_bits)
    inverse = np.argsort(_ts_38212_out_int(packet.e_r[0], packet.q_m))
    mutated = [block[inverse] for block in clean]

    assert np.array_equal(clean[0], flat[: packet.e_r[0]])
    assert not np.array_equal(mutated[0], flat[: packet.e_r[0]])


def test_mutation_incorrect_code_block_concatenation():
    packet = build_packet_plan(
        int(get("bandwidth.k_symbols.imagenette160")["r_1_24"]), "qam16", "2/3"
    )
    assert packet.feasible and packet.segmentation is not None
    assert packet.segmentation.code_blocks > 1
    payload = _payload(packet, seed=89)
    blocks = transmit_transport(payload, packet)
    expected = _expected_mapper_input(packet, payload)

    assert np.array_equal(mapper_input_bits(blocks, "qam16"), expected)
    assert not np.array_equal(
        mapper_input_bits(list(reversed(blocks)), "qam16"), expected
    )


def test_mutation_incorrect_llr_block_splitting():
    packet = build_packet_plan(
        int(get("bandwidth.k_symbols.imagenette160")["r_1_24"]), "qam16", "2/3"
    )
    assert packet.feasible
    llrs = np.arange(packet.channel_bits, dtype=np.float32)
    correct = split_llr_blocks(llrs, packet.e_r)

    shifted = (packet.e_r[0] - packet.q_m,) + packet.e_r[1:-1] + (
        packet.e_r[-1] + packet.q_m,
    )
    assert sum(shifted) == sum(packet.e_r)
    mutated = split_llr_blocks(llrs, shifted)

    assert tuple(block.size for block in correct) == packet.e_r
    assert tuple(block.size for block in mutated) != packet.e_r
    assert not np.array_equal(mutated[0], correct[0][: mutated[0].size]) or (
        mutated[0].size != correct[0].size
    )
