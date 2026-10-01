"""AM-100: Sionna's zero-pruning edge case at base graph 2, rate 1/5."""

from __future__ import annotations

import numpy as np

from baseline.ldpc.adapter import SionnaLDPCAdapter
from baseline.ldpc.transport import build_packet_plan
from baseline.ldpc.unpruned_decoder import needs_unpruned_decoder, repair_zero_pruning


def _adapter(rate: str) -> tuple[SionnaLDPCAdapter, int]:
    packet = build_packet_plan(12800, "bpsk", rate)
    layout = packet.segmentation
    assert layout is not None
    return SionnaLDPCAdapter(layout.k_prime, packet.e_r[0], 1, layout.base_graph, "cpu"), layout.k_prime


def test_rate_one_fifth_needs_and_receives_the_unpruned_decoder() -> None:
    adapter, k_prime = _adapter("1/5")
    assert needs_unpruned_decoder(adapter)
    repair_zero_pruning(adapter)
    assert not needs_unpruned_decoder(adapter)
    bits = np.random.default_rng(0).integers(0, 2, (4, k_prime))
    codeword = adapter.encode(bits)
    llrs = np.where(codeword == 1, 8.0, -8.0)  # log p1/p0 convention
    assert np.array_equal(adapter.decode(llrs), bits)


def test_rate_one_third_decoder_is_left_untouched() -> None:
    adapter, _ = _adapter("1/3")
    decoder = adapter.decoder
    assert not needs_unpruned_decoder(adapter)
    assert repair_zero_pruning(adapter).decoder is decoder
