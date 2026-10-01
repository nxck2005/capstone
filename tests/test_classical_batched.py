"""The batched classical transport reproduces ``transport_round_trip`` exactly."""

from __future__ import annotations

import numpy as np
import pytest

from baseline.classical.batched import BatchedTransport
from baseline.classical.channel_transport import transport_round_trip
from baseline.ldpc.transport import build_packet_plan


@pytest.mark.parametrize(
    ("modulation", "rate", "snr_db"),
    [("bpsk", "1/3", -4.5), ("qpsk", "1/2", 1.0), ("qam16", "5/6", 9.0)],
)
def test_batched_round_trip_equals_the_per_packet_path(modulation: str, rate: str, snr_db: float) -> None:
    packet = build_packet_plan(12800, modulation, rate)
    assert packet.feasible and packet.segmentation is not None
    rng = np.random.default_rng(7)
    payloads = [rng.integers(0, 2, packet.segmentation.payload_bits).astype(np.uint8) for _ in range(5)]
    noise_ids = [f"test-batched-{modulation}-{index}" for index in range(5)]

    batched = BatchedTransport(packet, device="cpu").round_trip(payloads, snr_db=snr_db, noise_ids=noise_ids)
    for payload, noise_id, outcome in zip(payloads, noise_ids, batched, strict=True):
        single = transport_round_trip(payload, packet, snr_db=snr_db, noise_id=noise_id, device="cpu")
        assert outcome.accounting == single.accounting
        assert outcome.unit_noise_sha256 == single.unit_noise_sha256
        assert outcome.realised_symbol_energy == single.realised_symbol_energy
        assert outcome.papr_db == single.papr_db
        assert outcome.received.crc_ok == single.received.crc_ok
        assert outcome.received.tb_crc_ok == single.received.tb_crc_ok
        assert outcome.received.code_block_crc_ok == single.received.code_block_crc_ok
        if single.received.payload_bits is None:
            assert outcome.received.payload_bits is None
        else:
            assert np.array_equal(outcome.received.payload_bits, single.received.payload_bits)
