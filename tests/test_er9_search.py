from __future__ import annotations

from evaluation.er9_search import (
    feasible_pairs,
    packetisation_floor,
    select_phy,
    select_stage1,
    select_stage2,
    stage1_candidates,
    stage2_candidates,
)


def test_er9_packet_floor_and_stage1_order_are_exact() -> None:
    floor = packetisation_floor(12800)
    assert floor.payload_bits == 4248
    assert floor.source_bytes == 531
    assert [item.transmit_dim for item in stage1_candidates(floor.payload_bits)] == [
        64, 128, 256, 512, 1024, 2048
    ]
    assert all(item.quantiser_bits == 2 for item in stage1_candidates(floor.payload_bits))


def test_er9_admissible_pair_count_and_stage2_order() -> None:
    pairs = feasible_pairs(4248)
    assert len(pairs) == 19
    assert [item.quantiser_bits for item in stage2_candidates(4248, 1024)] == [2, 4]
    assert [item.quantiser_bits for item in stage2_candidates(4248, 2048)] == [2]


def test_er9_stage1_exact_tie_chooses_smallest_dimension() -> None:
    selected = select_stage1(
        [
            {"transmit_dim": 256, "n_correct": 900},
            {"transmit_dim": 64, "n_correct": 900},
            {"transmit_dim": 128, "n_correct": 899},
        ]
    )
    assert selected["transmit_dim"] == 64


def test_er9_stage2_exact_tie_chooses_smallest_width_and_reuse_is_explicit() -> None:
    selected = select_stage2(
        [
            {"transmit_dim": 256, "quantiser_bits": 6, "n_correct": 900},
            {"transmit_dim": 256, "quantiser_bits": 2, "n_correct": 900},
            {"transmit_dim": 256, "quantiser_bits": 4, "n_correct": 899},
        ]
    )
    assert selected["quantiser_bits"] == 2
    assert [item.quantiser_bits for item in stage2_candidates(4248, 2048)] == [2]


def test_er9_phy_tie_is_exact_and_not_a_float_comparison() -> None:
    selected = select_phy(
        [
            {"modulation": "qam16", "ldpc_rate": "1/2", "n_correct": 500},
            {"modulation": "qpsk", "ldpc_rate": "5/6", "n_correct": 500},
        ]
    )
    assert selected["modulation"] == "qpsk"
