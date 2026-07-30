"""Classical channel transport: exact accounting, shared channel, keyed noise."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import baseline.classical.channel_transport as channel_transport
import channels.registry as channel_registry
from baseline.classical.channel_transport import (
    build_accounting,
    demodulate,
    modulate,
    transport_round_trip,
)
from sionna.phy.fec.ldpc import LDPC5GEncoder

from baseline.ldpc.adapter import SionnaLDPCAdapter
from baseline.ldpc.modulation import (
    bits_per_symbol,
    map_bits,
    max_log_llr,
    n0_from_esn0_db,
    realised_symbol_energy,
)
from baseline.ldpc.transport import build_packet_plan, transmit_transport
from channels.awgn import AWGN, keyed_complex_noise
from channels.power import symbol_papr_db
from config.params import get

ROOT = Path(__file__).resolve().parents[1]
HIGH_SNR_DB = 20.0
LOW_SNR_DB = -15.0


def _payload(accounting, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(
        0, 2, size=accounting.payload_bits, dtype=np.uint8
    )


def _record_rows():
    record = json.loads((ROOT / "spec/evidence/packetisation_record.json").read_text())
    return record["configurations"]


# --- exact channel uses and bit reconciliation --------------------------------


def test_channel_uses_are_exact_for_every_ratio_and_modulation():
    ratios = get("bandwidth.ratios")
    checked = 0
    infeasible = 0
    for dataset, per_ratio in get("bandwidth.k_symbols").items():
        for ratio in ratios:
            k_symbols = int(per_ratio[ratio])
            for modulation in get("baseline.modulations"):
                for rate in get("baseline.ldpc_rates"):
                    packet = build_packet_plan(k_symbols, modulation, rate)
                    if not packet.feasible:
                        infeasible += 1
                        continue
                    accounting = build_accounting(packet)
                    q_m = bits_per_symbol(modulation)
                    assert accounting.k_symbols == k_symbols, (dataset, ratio, modulation)
                    assert accounting.channel_bits == k_symbols * q_m
                    assert accounting.channel_uses_exact
                    assert accounting.channel_bits_equal_k_times_qm
                    checked += 1
    assert checked == 215
    assert infeasible == 1


def test_bit_reconciliation_matches_every_committed_packetisation_row():
    for row in _record_rows():
        packet = build_packet_plan(row["k"], row["modulation"], row["nominal_rate_str"])
        if not row["feasible"]:
            assert not packet.feasible
            continue
        accounting = build_accounting(packet)
        assert accounting.payload_bits == row["A"]
        assert accounting.payload_bytes == row["source_bytes"]
        assert accounting.tb_crc_bits == row["tb_crc_bits"]
        assert accounting.cb_crc_bits_total == row["cb_crc_bits_total"]
        assert accounting.ldpc_filler_bits_total == row["filler_bits_total"]
        assert list(accounting.rate_matched_bits) == row["E"]
        assert accounting.rate_matched_bits_total == row["E_sum"]
        # the two identities the phase exists to guarantee
        assert (
            accounting.payload_bits
            + accounting.tb_crc_bits
            + accounting.cb_crc_bits_total
            + accounting.ldpc_filler_bits_total
            == accounting.systematic_bits_total
        ), row["tag"]
        assert accounting.rate_matched_bits_total == accounting.channel_bits, row["tag"]
        assert accounting.reconciles, row["tag"]


def test_worked_example_reproduces_br10_partial_final_block_accounting():
    accounting = build_accounting(build_packet_plan(25600, "qpsk", "5/6"))
    assert accounting.payload_bits == 42624
    assert accounting.payload_bytes == 5328
    assert accounting.tb_crc_name == "crc24a"
    assert accounting.payload_bits + accounting.tb_crc_bits == 42648
    assert accounting.code_blocks == 6
    assert accounting.k_prime == 7132
    assert accounting.lifting_size == 352
    assert accounting.systematic_bits_per_block == 7744
    assert accounting.ldpc_filler_bits_per_block == 612
    assert accounting.ldpc_filler_bits_total == 3672
    assert accounting.rate_matched_bits == (8532, 8532, 8534, 8534, 8534, 8534)
    assert accounting.rate_matched_bits_total == 51200


def test_build_accounting_rejects_a_structurally_infeasible_packet():
    packet = build_packet_plan(64, "bpsk", "1/3")
    assert not packet.feasible
    with pytest.raises(ValueError, match="structurally infeasible"):
        build_accounting(packet)


# --- round trips --------------------------------------------------------------


@pytest.mark.parametrize("modulation", ["bpsk", "qpsk", "qam16"])
def test_all_modulations_round_trip_at_high_snr(modulation: str):
    packet = build_packet_plan(256, modulation, "1/2")
    accounting = build_accounting(packet)
    payload = _payload(accounting, 11)
    outcome = transport_round_trip(
        payload, packet, snr_db=HIGH_SNR_DB, noise_id=f"round-trip-{modulation}"
    )
    assert outcome.crc_ok and outcome.received.tb_crc_ok
    assert np.array_equal(outcome.payload_bits, payload)
    assert outcome.accounting.modulation == modulation


@pytest.mark.parametrize("rate", ["1/3", "1/2", "2/3", "5/6"])
def test_all_configured_ldpc_rates_round_trip(rate: str):
    assert rate in get("baseline.ldpc_rates")
    packet = build_packet_plan(512, "qpsk", rate)
    accounting = build_accounting(packet)
    assert accounting.nominal_rate == rate
    payload = _payload(accounting, 23)
    outcome = transport_round_trip(
        payload, packet, snr_db=HIGH_SNR_DB, noise_id=f"rate-{rate}"
    )
    assert outcome.crc_ok
    assert np.array_equal(outcome.payload_bits, payload)


def test_multi_code_block_round_trip_with_uniform_budgets():
    packet = build_packet_plan(3200, "qam16", "2/3")
    accounting = build_accounting(packet)
    assert accounting.code_blocks == 2
    assert len(set(accounting.rate_matched_bits)) == 1
    assert accounting.cb_crc_bits_total == 2 * int(get("baseline.cb_crc_bits"))
    payload = _payload(accounting, 31)
    outcome = transport_round_trip(
        payload, packet, snr_db=HIGH_SNR_DB, noise_id="multi-uniform"
    )
    assert outcome.crc_ok
    assert outcome.received.code_block_crc_ok == (True, True)
    assert np.array_equal(outcome.payload_bits, payload)


def test_partial_final_code_block_round_trips_and_is_accounted():
    packet = build_packet_plan(6400, "qam16", "2/3")
    accounting = build_accounting(packet)
    assert accounting.code_blocks == 3
    assert accounting.rate_matched_bits == (8532, 8532, 8536)
    assert len(set(accounting.rate_matched_bits)) > 1
    payload = _payload(accounting, 37)
    outcome = transport_round_trip(
        payload, packet, snr_db=HIGH_SNR_DB, noise_id="partial-final"
    )
    assert outcome.crc_ok
    assert outcome.received.code_block_crc_ok == (True, True, True)
    assert np.array_equal(outcome.payload_bits, payload)


def test_decode_failure_is_reported_not_raised():
    packet = build_packet_plan(256, "qpsk", "1/2")
    accounting = build_accounting(packet)
    payload = _payload(accounting, 41)
    outcome = transport_round_trip(
        payload, packet, snr_db=LOW_SNR_DB, noise_id="decode-failure"
    )
    assert not outcome.crc_ok
    assert not outcome.received.tb_crc_ok
    assert outcome.payload_bits is None
    # the transmission still happened, so the measurements are still emitted
    assert outcome.realised_symbol_energy > 0
    assert outcome.accounting.reconciles


# --- shared channel and keyed noise ------------------------------------------


def test_classical_path_builds_the_shared_registry_awgn_channel(
    monkeypatch: pytest.MonkeyPatch,
):
    assert channel_transport._CHANNEL_MODEL in get("channel.models_supported")
    built: list[torch.nn.Module] = []

    def recording_factory(**kwargs):
        channel = AWGN(**kwargs)
        built.append(channel)
        return channel

    monkeypatch.setitem(
        channel_registry._CHANNELS,
        channel_transport._CHANNEL_MODEL,
        recording_factory,
    )
    packet = build_packet_plan(256, "qpsk", "1/2")
    accounting = build_accounting(packet)
    transport_round_trip(
        _payload(accounting, 43), packet, snr_db=HIGH_SNR_DB, noise_id="shared-channel"
    )
    assert len(built) == 1
    assert isinstance(built[0], AWGN)
    assert not built[0].training


def test_noise_is_keyed_and_identical_across_systems_for_one_identity():
    packet = build_packet_plan(256, "qpsk", "1/2")
    accounting = build_accounting(packet)
    payload = _payload(accounting, 47)
    expected = hashlib.sha256(
        keyed_complex_noise("shared-identity", accounting.k_symbols).numpy().tobytes()
    ).hexdigest()

    first = transport_round_trip(
        payload, packet, snr_db=LOW_SNR_DB, noise_id="shared-identity"
    )
    # an unrelated intervening draw must not move the stream
    keyed_complex_noise("unrelated", 4096)
    transport_round_trip(payload, packet, snr_db=LOW_SNR_DB, noise_id="other-identity")
    second = transport_round_trip(
        payload, packet, snr_db=LOW_SNR_DB, noise_id="shared-identity"
    )

    assert first.unit_noise_sha256 == expected == second.unit_noise_sha256
    assert first.received.tb_crc_ok == second.received.tb_crc_ok
    assert get("artifacts.rng_stream") == "counter_based_keyed_not_sequential"


def test_channel_noise_identity_is_exactly_the_declared_noise_id_key():
    assert get("artifacts.rng_identity_fields.channel_noise") == ["noise_id"]
    assert get("artifacts.noise_id_key") == [
        "dataset_version",
        "split_manifest_hash",
        "stable_sample_id",
        "test_snr_db",
        "channel_seed",
        "channel",
        "k",
        "block_index",
        "rng_purpose",
    ]


# --- normalisation, interleaving and demapping conventions --------------------


def test_modulation_applies_only_the_fixed_constellation_normalisation():
    assert get("baseline.per_packet_power_rescaling_permitted") is False
    packet = build_packet_plan(256, "qam16", "1/2")
    accounting = build_accounting(packet)
    blocks = transmit_transport(_payload(accounting, 53), packet)
    symbols = modulate(blocks, "qam16")
    # the blocks arrive already interleaved by Sionna, so mapping is all that is
    # left to do — see tests/test_classical_interleaver_conformance.py
    expected = map_bits(np.concatenate(blocks), "qam16")
    assert np.array_equal(symbols, expected)
    # a renormalised packet would measure exactly unit energy; this one must not
    assert realised_symbol_energy(symbols) != 1.0
    assert abs(realised_symbol_energy(symbols) - 1.0) < 0.2


def test_realised_symbol_energy_and_papr_are_measured_and_returned():
    packet = build_packet_plan(256, "qam16", "1/2")
    accounting = build_accounting(packet)
    payload = _payload(accounting, 59)
    blocks = transmit_transport(payload, packet)
    symbols = modulate(blocks, "qam16")
    outcome = transport_round_trip(
        payload, packet, snr_db=HIGH_SNR_DB, noise_id="measurements"
    )
    assert get("baseline.realised_symbol_energy_logged") is True
    assert outcome.realised_symbol_energy == pytest.approx(
        realised_symbol_energy(symbols)
    )
    assert outcome.papr_db == pytest.approx(
        float(symbol_papr_db(torch.as_tensor(symbols).reshape(1, -1))[0])
    )
    assert outcome.papr_db > 0
    assert outcome.per_packet_power_rescaling_applied is False


def test_qam16_bit_interleaver_is_required_and_actually_changes_the_symbols():
    """Required, non-trivial, and applied by its single owner — the adapter.

    PB_1 asserted this against ``modulate()``, which is why a *second* copy of
    the permutation there went unnoticed.  The claim is the same; the place it
    is checked moved to the seam that actually owns it.
    """

    assert get("baseline.modulation_bit_interleaver") == "ts_38212_5_4_2_2"
    assert get("baseline.modulation_bit_interleaver_required") is True
    packet = build_packet_plan(256, "qam16", "1/2")
    accounting = build_accounting(packet)
    layout = packet.segmentation
    assert layout is not None
    blocks = transmit_transport(_payload(accounting, 61), packet)

    plain = LDPC5GEncoder(
        k=layout.k_prime, n=packet.e_r[0], bg=f"bg{layout.base_graph}", device="cpu"
    )
    assert plain.num_bits_per_symbol is None
    interleaved = SionnaLDPCAdapter(
        layout.k_prime, packet.e_r[0], packet.q_m, layout.base_graph, "cpu"
    )
    assert interleaved.encoder.num_bits_per_symbol == accounting.q_m
    assert not np.array_equal(interleaved.encoder.out_int.numpy(), np.arange(packet.e_r[0]))

    # and the project adds nothing on top of it
    assert np.array_equal(
        modulate(blocks, "qam16"), map_bits(np.concatenate(blocks), "qam16")
    )


def test_demapper_convention_is_log_p1_over_p0_and_deinterleaves():
    assert get("baseline.demapper") == "max_log_app"
    assert get("baseline.ldpc_llr_convention") == "log_p1_over_p0"
    n0 = n0_from_esn0_db(HIGH_SNR_DB)
    # BPSK maps bit 0 to +1, so a clean +1 must give a negative log P1/P0
    assert float(max_log_llr(np.asarray([1.0 + 0j]), "bpsk", n0)[0]) < 0
    assert float(max_log_llr(np.asarray([-1.0 + 0j]), "bpsk", n0)[0]) > 0

    packet = build_packet_plan(256, "qam16", "1/2")
    accounting = build_accounting(packet)
    blocks = transmit_transport(_payload(accounting, 67), packet)
    symbols = modulate(blocks, "qam16")
    recovered = demodulate(symbols, "qam16", n0, accounting.rate_matched_bits)
    assert [block.size for block in recovered] == list(accounting.rate_matched_bits)
    for sent, llrs in zip(blocks, recovered, strict=True):
        assert np.array_equal((llrs > 0).astype(np.uint8), sent)


def test_demodulate_rejects_a_packet_whose_llr_count_disagrees():
    with pytest.raises(ValueError, match="does not match the packet plan"):
        demodulate(np.zeros(8, dtype=np.complex64), "qpsk", 1.0, (32,))
