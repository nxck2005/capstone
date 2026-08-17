from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from baseline import g8_d
from baseline.classical.composition import BlerIdentity


HEX_A = "a" * 64
HEX_B = "b" * 64


def _split() -> g8_d.ValidationSplitIdentity:
    return g8_d.ValidationSplitIdentity("fixture", "val", HEX_A, HEX_B)


def _image() -> g8_d.ImageIdentity:
    return g8_d.ImageIdentity.from_pixels(
        split_identity=_split(),
        stable_sample_id="fixture-sample",
        source_bytes=b"source-v1",
        canonical_pixels=np.zeros((8, 8, 3), dtype=np.uint8),
    )


def _budget() -> g8_d.BudgetIdentity:
    return g8_d.BudgetIdentity(
        bw_ratio="1/12",
        bytes_sent=80,
        payload_bytes=80,
        packet_accounting={"payload_bytes": 80, "channel_bits": 640, "k_symbols": 320},
    )


def _codec() -> g8_d.CodecConfigurationIdentity:
    snapshot = {"baseline": {"source_codec": "jpeg2000"}, "preprocessing": {}, "environment": {}}
    return g8_d.CodecConfigurationIdentity(
        snapshot=snapshot,
        configuration_hash=g8_d.sha256_bytes(g8_d.canonical_json(snapshot)),
        runtime_version="fixture-openjpeg",
    )


def _bler_identity() -> dict[str, object]:
    return {
        "k_and_n": [128, 256],
        "base_graph": 2,
        "lifting_size": 22,
        "modulation": "qpsk",
        "decoder_algorithm": "offset_min_sum",
        "decoder_offset": 0.5,
        "iterations": 50,
        "snr_convention": "es_n0_per_symbol",
        "rate": "1/2",
    }


def test_contract_artifact_and_independent_verifier_pass() -> None:
    contract = g8_d.build_g8_d_contract()
    assert contract["checkpoint"] == "D6"
    assert contract["status"] == "bounded_smoke_ready"
    assert contract["next_gate"] == "G8_D/D7"
    assert contract["g8_c_binding"]["table_id"].startswith("g8pblertable-")
    assert contract["g8_c_binding"]["measured_points"] == 3213
    assert all(item["split"] == "val" for item in contract["validation_split_bindings"])
    assert contract["safety"]["test_access"] == 0


def test_validation_identity_refuses_test_split_and_extra_fields() -> None:
    with pytest.raises(g8_d.G8DContractError, match="only use split 'val'"):
        g8_d.ValidationSplitIdentity("fixture", "test", HEX_A, HEX_B)

    data = _split().as_dict()
    data["unexpected"] = True
    with pytest.raises(g8_d.G8DContractError, match="schema differs"):
        g8_d.ValidationSplitIdentity.from_mapping(data)

    image = _image().as_dict()
    image["split"] = "test"
    with pytest.raises(g8_d.G8DContractError, match="only use split 'val'"):
        g8_d.ImageIdentity.from_mapping(image)


def test_image_identity_binds_source_bytes_and_canonical_pixels() -> None:
    image = _image()
    source_mutation = g8_d.ImageIdentity.from_pixels(
        split_identity=_split(),
        stable_sample_id="fixture-sample",
        source_bytes=b"source-v2",
        canonical_pixels=np.zeros((8, 8, 3), dtype=np.uint8),
    )
    pixel_mutation = g8_d.ImageIdentity.from_pixels(
        split_identity=_split(),
        stable_sample_id="fixture-sample",
        source_bytes=b"source-v1",
        canonical_pixels=np.ones((8, 8, 3), dtype=np.uint8),
    )
    assert image.identity_id != source_mutation.identity_id
    assert image.identity_id != pixel_mutation.identity_id


def test_codec_search_key_binds_every_non_snr_input() -> None:
    image = _image()
    budget = _budget()
    codec = _codec()
    key = g8_d.CodecSearchKey(image.identity_id, budget.identity_id, codec.identity_id, 8)
    assert "snr" not in g8_d.canonical_json(key.payload()).decode()
    assert replace(key, image_identity_id="other").identity_id != key.identity_id
    assert replace(key, budget_identity_id="other").identity_id != key.identity_id
    assert replace(key, codec_configuration_id="other").identity_id != key.identity_id
    assert replace(key, encode_axis_px=7).identity_id != key.identity_id


def test_candidate_binds_full_bler_identity_table_and_snr() -> None:
    candidate = g8_d.CandidateIdentity(
        image_identity_id=_image().identity_id,
        budget_identity_id=_budget().identity_id,
        codec_configuration_id=_codec().identity_id,
        g8_c_table_identity_id="g8dtable-" + HEX_A,
        bler_identity=_bler_identity(),
        snr_db=3.0,
        encode_axis_px=8,
    )
    BlerIdentity.from_mapping(candidate.bler_identity)
    assert replace(candidate, snr_db=4.0).identity_id != candidate.identity_id
    changed = dict(candidate.bler_identity)
    changed["rate"] = "5/6"
    assert replace(candidate, bler_identity=changed).identity_id != candidate.identity_id


def test_emitted_and_reconstruction_identities_reconcile_bytes() -> None:
    key = g8_d.CodecSearchKey("image", "budget", "codec", 8)
    emitted = g8_d.EmittedFileIdentity(key.identity_id, HEX_A, 70, 80, 10)
    reconstruction = g8_d.ReconstructionIdentity(
        "image", emitted.identity_id, "codec", (16, 16, 3), "bicubic", True
    )
    assert reconstruction.identity_id.startswith("g8drecon-")
    with pytest.raises(g8_d.G8DContractError, match="arithmetic"):
        g8_d.EmittedFileIdentity(key.identity_id, HEX_A, 71, 80, 10)


def test_table_identity_round_trips_wrapped_schema() -> None:
    contract = g8_d.build_g8_d_contract()
    identity = g8_d.G8CTableIdentity.from_mapping(contract["g8_c_binding"])
    assert identity.table_id == "g8pblertable-69ecc729f3b7dc3d67c0a3a5d8cf071cab927ad0a1e0cd5b18a6bbe674b9126f"
    assert identity.predecessor_table_contribution == "none"
