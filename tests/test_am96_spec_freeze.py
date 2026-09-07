"""AM-96 pre-science ER-9 semantic-freeze checks."""

from __future__ import annotations

from pathlib import Path

from evaluation.am96_spec_compatibility import (
    ALLOWED_PARAMETER_PATHS,
    FREEZE_ID,
    load,
)


REPO = Path(__file__).resolve().parents[1]


def test_am96_freeze_closes_all_known_p1_7_items() -> None:
    value = load(REPO, allow_downstream=True)
    assert value["freeze_id"] == FREEZE_ID
    assert value["audit"]["closed_items"] == 16
    assert len(ALLOWED_PARAMETER_PATHS) == 32
    assert value["scientific_boundary"] == {
        "g10_model_facing_evaluations": 63,
        "g10_reruns": 0,
        "er9_training": 0,
        "er2_randomized_training": 0,
        "g11": 0,
        "w10": 0,
        "learned_test_inference": 0,
        "model_facing_test_access": 0,
        "test_split": "SEALED",
    }


def test_am96_contract_is_fixed_before_science() -> None:
    contract = load(REPO, allow_downstream=True)["semantic_contract"]
    assert contract["factorisation"]["channel_rule"] == "D_div_64"
    assert contract["factorisation"]["pool_height"] == 8
    assert contract["factorisation"]["pool_width"] == 8
    assert contract["dimension_scope"].endswith("frozen_across_snr")
    assert contract["quantiser"]["range"] == [-1, 1]
    assert contract["quantiser"]["scale_side_information_bits"] == 0
    assert contract["search"]["cross_product"] is False
    assert contract["entropy_and_framing"]["stream_length_field"] == "none_decode_exact_D_symbols"
