"""G8_B B1 frozen-contract tests.

Every random array here is a tiny synthetic fixture proving deterministic
stream semantics. Nothing in this file executes a scientific work unit, runs an
LDPC encoder or decoder, generates a scientific channel realisation, touches
campaign state, or performs bounded smoke.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import re
import socket
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable

import numpy as np
import pytest

import config.params as params_module
import gen_g8_bler_tooling_contract as generator
import verify_g8_bler_tooling_contract as contract_verifier
from baseline import g8_bler_contract as contract
from baseline.g8_campaign import canonical_json, rendered_json

# --------------------------------------------------------------------------
# Independently calculated fixed vectors.
#
# These are *not* produced by calling the helpers under test. The seeds and
# material digests come from hashing the literal pre-image bytes; the raw words
# and normal draws come from NumPy directly; the bits are extracted with plain
# Python integer arithmetic.
# --------------------------------------------------------------------------

EXPECTED_SEED_MATERIAL = {
    "information_bits": (
        b'["capstone:g8:bler-seed:v1","g8-fixture-campaign","bler-fixture-unit",'
        b'"information_bits"]'
    ),
    "awgn_real": (
        b'["capstone:g8:bler-seed:v1","g8-fixture-campaign","bler-fixture-unit","awgn_real"]'
    ),
    "awgn_imag": (
        b'["capstone:g8:bler-seed:v1","g8-fixture-campaign","bler-fixture-unit","awgn_imag"]'
    ),
}
EXPECTED_MATERIAL_SHA256 = {
    "information_bits": "54ce9ccd70afed9b721b078049cdb876234db196e25e2546ce3941baf5a766e6",
    "awgn_real": "1c7c8fad40e16110e649e7a79f3306c28e029c6dd5b4b42b8897b1be46a50323",
    "awgn_imag": "e4bb66584ffdb13901e67281a086e100b1666a732ea3ec80c958e0b6b065c3cb",
}
EXPECTED_SEED_UINT64 = {
    "information_bits": 6110994150561148315,
    "awgn_real": 2052673504454730000,
    "awgn_imag": 16481879790777643321,
}
EXPECTED_FIRST_RAW_WORDS = {
    "information_bits": [
        3688305290296260437,
        15804894354361011202,
        15787647027830899325,
        3711645746825559366,
    ],
    "awgn_real": [
        10330529237113510068,
        6419582030516705304,
        2500550259890786980,
        10491274558385273237,
    ],
    "awgn_imag": [
        10331417318171454595,
        8380918203836260281,
        6946468110326417903,
        15346635912537046511,
    ],
}
EXPECTED_BITS_0_TO_8 = [1, 0, 1, 0, 1, 0, 1, 0]
EXPECTED_BITS_60_TO_68 = [1, 1, 0, 0, 0, 1, 0, 0]
EXPECTED_FIRST_NORMALS = {
    "awgn_real": [
        0.9216810155880256,
        0.5781483793956664,
        0.1512876155480005,
        -0.924367554903119,
    ],
    "awgn_imag": [
        0.747589359934289,
        -1.2454842320670447,
        -0.03364133403329626,
        -1.7571909989743366,
    ],
}
EXPECTED_WILSON = {
    "zero_errors_16_trials": [0.0, 0.19360768053443644],
    "one_error_16_trials": [0.011119344764642547, 0.28328737570298934],
    "all_errors_16_trials": [0.8063923194655636, 1.0],
}

FIXTURE_CAMPAIGN = contract.FIXTURE_CAMPAIGN_ID
FIXTURE_UNIT = contract.FIXTURE_WORK_UNIT_ID
CONTRACT_SOURCE = Path(__file__).resolve().parents[1] / "src/baseline/g8_bler_contract.py"
SMOKE_TRIALS = contract.BOUNDED_SMOKE_MAX_TRIALS_PER_UNIT


# --------------------------------------------------------------------------
# Shared fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def corrected_b1c_contract_artifact(
    tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
) -> Path:
    """Exercise v2 builders against an isolated generated artifact.

    C5 installs the artifact into the tracked path.  Until then, the live
    state must continue to bind the B1 bytes, so these source-level tests use
    a temporary corrected artifact without touching campaign state.
    """

    path = tmp_path_factory.mktemp("g8-b1c-contract") / "bler_tooling_contract.json"
    path.write_bytes(rendered_json(generator.build()))
    patcher = pytest.MonkeyPatch()
    patcher.setattr(contract, "TOOLING_CONTRACT_ARTIFACT", path)
    request.addfinalizer(patcher.undo)
    return path


@pytest.fixture(scope="module")
def required_unit_id() -> str:
    return sorted(contract.required_work_unit_index())[0]


@pytest.fixture(scope="module")
def full_request(required_unit_id: str) -> dict[str, Any]:
    return contract.build_full_strength_request(required_unit_id)


@pytest.fixture(scope="module")
def smoke_request() -> dict[str, Any]:
    return contract.build_bounded_smoke_request(
        work_unit_id="bler-smoke-fixture",
        bler_identity={
            "k_and_n": [16, 32],
            "base_graph": 2,
            "lifting_size": 8,
            "modulation": "bpsk",
            "decoder_algorithm": "offset_min_sum",
            "decoder_offset": 0.5,
            "iterations": 50,
            "snr_convention": "es_n0_per_symbol",
            "rate": "1/2",
        },
        snr_db=0.0,
        source_packet_config_ids=["pkt-smoke-fixture"],
        trials_requested=SMOKE_TRIALS,
    )


def _complete_full_result(request: dict[str, Any], *, bit_errors: int, block_errors: int) -> dict:
    return contract.build_work_unit_result(
        request=request,
        status=contract.STATUS_COMPLETE,
        trials_completed=request["trials_requested"],
        bit_errors=bit_errors,
        block_errors=block_errors,
    )


# --------------------------------------------------------------------------
# 1-14  Seed derivation and random-stream separation
# --------------------------------------------------------------------------


def test_seed_is_stable_for_the_same_inputs() -> None:
    for purpose in contract.SEED_PURPOSES:
        first = contract.derive_seed(FIXTURE_CAMPAIGN, FIXTURE_UNIT, purpose)
        second = contract.derive_seed(FIXTURE_CAMPAIGN, FIXTURE_UNIT, purpose)
        assert first == second == EXPECTED_SEED_UINT64[purpose]


def test_changing_campaign_id_changes_the_seed() -> None:
    baseline = contract.derive_seed(FIXTURE_CAMPAIGN, FIXTURE_UNIT, "information_bits")
    other = contract.derive_seed(FIXTURE_CAMPAIGN + "x", FIXTURE_UNIT, "information_bits")
    assert baseline != other


def test_changing_work_unit_id_changes_the_seed() -> None:
    baseline = contract.derive_seed(FIXTURE_CAMPAIGN, FIXTURE_UNIT, "information_bits")
    other = contract.derive_seed(FIXTURE_CAMPAIGN, FIXTURE_UNIT + "x", "information_bits")
    assert baseline != other


def test_changing_purpose_changes_the_seed() -> None:
    seeds = {
        purpose: contract.derive_seed(FIXTURE_CAMPAIGN, FIXTURE_UNIT, purpose)
        for purpose in contract.SEED_PURPOSES
    }
    assert len(set(seeds.values())) == len(contract.SEED_PURPOSES)


def test_unknown_purpose_is_rejected() -> None:
    with pytest.raises(contract.G8BlerContractError, match="unknown random purpose"):
        contract.derive_seed(FIXTURE_CAMPAIGN, FIXTURE_UNIT, "channel_state")


@pytest.mark.parametrize(
    "campaign_id, work_unit_id",
    [("", FIXTURE_UNIT), ("   ", FIXTURE_UNIT), (FIXTURE_CAMPAIGN, ""), (FIXTURE_CAMPAIGN, "\t")],
)
def test_blank_identifiers_are_rejected(campaign_id: str, work_unit_id: str) -> None:
    with pytest.raises(contract.G8BlerContractError, match="non-blank"):
        contract.derive_seed(campaign_id, work_unit_id, "information_bits")


@pytest.mark.parametrize("identifier", [123, None, b"bytes", ["list"]])
def test_non_string_identifiers_are_rejected(identifier: Any) -> None:
    with pytest.raises(contract.G8BlerContractError, match="non-blank"):
        contract.derive_seed(identifier, FIXTURE_UNIT, "information_bits")


def test_delimiter_like_text_cannot_create_a_collision() -> None:
    # A naive "a|b|c" pre-image would collide these two; JSON escaping cannot.
    left = contract.derive_seed("camp", 'unit","information_bits', "awgn_real")
    right = contract.derive_seed("camp", "unit", "information_bits")
    assert left != right
    assert contract.seed_material("camp", 'a","b', "awgn_real") != contract.seed_material(
        "camp", 'a", "b', "awgn_real"
    )


def test_seed_lies_in_the_unsigned_64_bit_range() -> None:
    for index in range(64):
        seed = contract.derive_seed(FIXTURE_CAMPAIGN, f"bler-range-{index}", "awgn_imag")
        assert isinstance(seed, int) and not isinstance(seed, bool)
        assert 0 <= seed < 2**64


def test_fixed_seed_vectors_match_independently_computed_values() -> None:
    for purpose in contract.SEED_PURPOSES:
        material = contract.seed_material(FIXTURE_CAMPAIGN, FIXTURE_UNIT, purpose)
        assert material == EXPECTED_SEED_MATERIAL[purpose]
        assert hashlib.sha256(material).hexdigest() == EXPECTED_MATERIAL_SHA256[purpose]
        record = contract.seed_record(FIXTURE_CAMPAIGN, FIXTURE_UNIT, purpose)
        assert record["material_sha256"] == EXPECTED_MATERIAL_SHA256[purpose]
        assert record["seed_uint64"] == EXPECTED_SEED_UINT64[purpose]
        assert record["seed_derivation_identity"] == contract.SEED_DERIVATION_IDENTITY
        assert record["seed_domain_separator"] == contract.SEED_DOMAIN_SEPARATOR


def test_live_state_seed_identity_equals_the_frozen_constant() -> None:
    from baseline.g8_campaign import load_campaign_state

    state = load_campaign_state()
    assert state["identity"]["seed_derivation_identity"] == contract.SEED_DERIVATION_IDENTITY


def test_contract_module_never_calls_python_hash() -> None:
    tree = ast.parse(CONTRACT_SOURCE.read_text(encoding="utf-8"), filename=str(CONTRACT_SOURCE))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "hash" not in called
    assert "id" not in called


def test_seed_is_independent_of_shard_and_enumeration_order() -> None:
    ids = [f"bler-order-{index}" for index in range(8)]
    forward = [contract.derive_seed(FIXTURE_CAMPAIGN, unit, "awgn_real") for unit in ids]
    backward = [contract.derive_seed(FIXTURE_CAMPAIGN, unit, "awgn_real") for unit in reversed(ids)]
    assert forward == list(reversed(backward))
    for forbidden in contract.SEED_FORBIDDEN_INPUTS:
        assert forbidden.encode() not in contract.seed_material(
            FIXTURE_CAMPAIGN, FIXTURE_UNIT, "awgn_real"
        )


def test_information_and_noise_streams_are_distinct() -> None:
    seeds = {
        purpose: contract.derive_seed(FIXTURE_CAMPAIGN, FIXTURE_UNIT, purpose)
        for purpose in contract.SEED_PURPOSES
    }
    real = contract.normal_stream(seeds["awgn_real"], 32)
    imag = contract.normal_stream(seeds["awgn_imag"], 32)
    assert not np.array_equal(real, imag)
    words = {
        purpose: contract.philox_words(seed, 0, 8).tolist() for purpose, seed in seeds.items()
    }
    assert len({tuple(value) for value in words.values()}) == len(contract.SEED_PURPOSES)


def test_fixed_raw_word_and_bit_vectors_match_numpy_directly() -> None:
    for purpose, expected in EXPECTED_FIRST_RAW_WORDS.items():
        seed = EXPECTED_SEED_UINT64[purpose]
        assert [int(word) for word in np.random.Philox(key=seed).random_raw(4)] == expected
        assert contract.philox_words(seed, 0, 4).tolist() == expected
    seed = EXPECTED_SEED_UINT64["information_bits"]
    words = EXPECTED_FIRST_RAW_WORDS["information_bits"]
    assert [(words[0] >> index) & 1 for index in range(8)] == EXPECTED_BITS_0_TO_8
    assert [
        (words[index // 64] >> (index % 64)) & 1 for index in range(60, 68)
    ] == EXPECTED_BITS_60_TO_68
    assert contract.information_bit_stream(seed, 0, 8).tolist() == EXPECTED_BITS_0_TO_8
    assert contract.information_bit_stream(seed, 60, 8).tolist() == EXPECTED_BITS_60_TO_68
    for purpose, expected_normals in EXPECTED_FIRST_NORMALS.items():
        drawn = contract.normal_stream(EXPECTED_SEED_UINT64[purpose], 4)
        assert [float(value) for value in drawn] == expected_normals


def test_information_bit_stream_dtype_and_domain() -> None:
    seed = EXPECTED_SEED_UINT64["information_bits"]
    bits = contract.information_bit_stream(seed, 0, 257)
    assert bits.dtype == np.uint8
    assert bits.flags["C_CONTIGUOUS"]
    assert set(np.unique(bits)).issubset({0, 1})
    assert contract.normal_stream(seed, 5).dtype == np.float64


@pytest.mark.parametrize("boundary", [0, 1, 62, 63, 64, 65, 127, 128])
def test_information_bit_stream_is_invariant_across_every_named_boundary(boundary: int) -> None:
    seed = EXPECTED_SEED_UINT64["information_bits"]
    whole = contract.information_bit_stream(seed, 0, 256).tolist()
    head = contract.information_bit_stream(seed, 0, boundary).tolist()
    tail = contract.information_bit_stream(seed, boundary, 256 - boundary).tolist()
    assert head + tail == whole


@pytest.mark.parametrize("length", [1, 7, 63, 65, 100, 127, 129, 257])
def test_information_lengths_not_divisible_by_64_are_exact(length: int) -> None:
    seed = EXPECTED_SEED_UINT64["information_bits"]
    reference = contract.information_bit_stream(seed, 0, 512).tolist()
    assert contract.information_bit_stream(seed, 0, length).tolist() == reference[:length]
    # A shorter unit must not carry its unused trailing bits into the next one.
    assert contract.information_bit_stream(seed, length, 32).tolist() == reference[
        length : length + 32
    ]


@pytest.mark.parametrize("chunks", [[200], [100, 100], [3, 61, 1, 70, 65], [1] * 200])
def test_every_stream_is_chunk_boundary_invariant(chunks: list[int]) -> None:
    seeds = {
        purpose: contract.derive_seed(FIXTURE_CAMPAIGN, FIXTURE_UNIT, purpose)
        for purpose in contract.SEED_PURPOSES
    }
    bit_seed = seeds["information_bits"]
    pieces: list[int] = []
    start = 0
    for size in chunks:
        pieces.extend(contract.information_bit_stream(bit_seed, start, size).tolist())
        start += size
    assert pieces == contract.information_bit_stream(bit_seed, 0, 200).tolist()

    for purpose in ("awgn_real", "awgn_imag"):
        generator_ = np.random.Generator(np.random.Philox(key=seeds[purpose]))
        chunked: list[float] = []
        for size in chunks:
            chunked.extend(float(value) for value in generator_.standard_normal(size))
        assert chunked == [float(value) for value in contract.normal_stream(seeds[purpose], 200)]


def test_philox_words_are_addressable_from_any_offset() -> None:
    seed = EXPECTED_SEED_UINT64["awgn_imag"]
    whole = contract.philox_words(seed, 0, 32).tolist()
    for start in range(0, 24):
        assert contract.philox_words(seed, start, 8).tolist() == whole[start : start + 8]


def test_gaussian_stream_is_declared_non_addressable() -> None:
    assert contract.NORMAL_STREAM_ADDRESSABLE is False
    assert contract.INFORMATION_BIT_STREAM_ADDRESSABLE is True
    assert contract.MID_WORK_UNIT_RESUME_PERMITTED is False
    assert contract.RESUME_GRANULARITY == "work_unit_atomic"


def test_restarting_a_unit_from_trial_zero_reproduces_the_same_bytes() -> None:
    seeds = {
        purpose: contract.derive_seed(FIXTURE_CAMPAIGN, FIXTURE_UNIT, purpose)
        for purpose in contract.SEED_PURPOSES
    }
    first = (
        contract.information_bit_stream(seeds["information_bits"], 0, 128).tobytes(),
        contract.normal_stream(seeds["awgn_real"], 32).tobytes(),
        contract.normal_stream(seeds["awgn_imag"], 32).tobytes(),
    )
    second = (
        contract.information_bit_stream(seeds["information_bits"], 0, 128).tobytes(),
        contract.normal_stream(seeds["awgn_real"], 32).tobytes(),
        contract.normal_stream(seeds["awgn_imag"], 32).tobytes(),
    )
    assert first == second


# --------------------------------------------------------------------------
# 15-18  Trial-count ownership
# --------------------------------------------------------------------------


def _patch_params(monkeypatch: pytest.MonkeyPatch, mutate: Callable[[dict], None]) -> None:
    tree = copy.deepcopy(params_module.load_params())
    mutate(tree)
    monkeypatch.setattr(params_module, "load_params", lambda: tree)


def test_full_strength_count_comes_from_its_own_parameter() -> None:
    assert contract.FULL_STRENGTH_TRIAL_COUNT_PARAMETER == "baseline.bler_characterisation_trials"
    assert contract.full_strength_trial_count() == params_module.get(
        "baseline.bler_characterisation_trials"
    )


def test_mutating_the_g2_blocks_per_snr_does_not_change_the_g8_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = contract.full_strength_trial_count()
    # The two values are currently equal; this proves the G-8 contract does not
    # read the G-2 reference experiment's key even so.
    _patch_params(
        monkeypatch,
        lambda tree: tree["baseline"]["ldpc_bler_reference"].__setitem__("blocks_per_snr", 7),
    )
    assert params_module.get("baseline.ldpc_bler_reference.blocks_per_snr") == 7
    assert contract.full_strength_trial_count() == before


def test_mutating_the_g8_trial_count_changes_the_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    before = contract.full_strength_trial_count()
    _patch_params(
        monkeypatch,
        lambda tree: tree["baseline"].__setitem__("bler_characterisation_trials", before + 1),
    )
    assert contract.full_strength_trial_count() == before + 1


@pytest.mark.parametrize("bad", [0, -1, True, 5.0, "5000", None])
def test_malformed_trial_count_is_rejected(monkeypatch: pytest.MonkeyPatch, bad: Any) -> None:
    _patch_params(
        monkeypatch, lambda tree: tree["baseline"].__setitem__("bler_characterisation_trials", bad)
    )
    with pytest.raises(contract.G8BlerContractError):
        contract.full_strength_trial_count()


def test_no_adaptive_stopping_or_observed_error_threshold_exists() -> None:
    assert contract.ADAPTIVE_STOPPING_PERMITTED is False
    assert "no adaptive stopping" in contract.NO_EARLY_STOPPING_RULE
    source = CONTRACT_SOURCE.read_text(encoding="utf-8")
    for forbidden in ("min_block_errors", "target_block_errors", "stop_after", "waterfall_target"):
        assert forbidden not in source
    # The trial count cannot depend on observations: it takes no arguments.
    assert contract.full_strength_trial_count.__code__.co_argcount == 0
    # An adaptive stopping rule needs a loop to stop out of; this module has none.
    tree = ast.parse(source, filename=str(CONTRACT_SOURCE))
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.While)]
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Break)]


# --------------------------------------------------------------------------
# 19-27  Request schema
# --------------------------------------------------------------------------


def test_exact_required_work_unit_produces_a_valid_full_strength_request(
    full_request: dict[str, Any], required_unit_id: str
) -> None:
    unit = contract.required_work_unit(required_unit_id)
    assert full_request["execution_class"] == contract.EXECUTION_CLASS_FULL_STRENGTH
    assert full_request["work_unit_id"] == required_unit_id
    assert full_request["bler_identity"] == unit["identity"]
    assert full_request["snr_db"] == unit["snr_db"]
    assert full_request["source_packet_config_ids"] == list(unit["source_packet_config_ids"])
    assert full_request["trials_requested"] == contract.full_strength_trial_count()
    assert full_request["trial_count_source"] == "params.baseline.bler_characterisation_trials"
    assert full_request["scientific_evidence"] is True
    assert full_request["merge_eligible"] is False
    assert full_request["test_split_access"] == 0
    assert contract.require_full_strength_request(full_request) == full_request


def test_unknown_work_unit_id_is_rejected_without_a_nearest_match() -> None:
    with pytest.raises(contract.G8BlerContractError, match="not an exact required BLER identity"):
        contract.build_full_strength_request("bler-does-not-exist")


def test_campaign_binding_mutations_do_not_change_later_requests(required_unit_id: str) -> None:
    first = contract.campaign_bindings()
    expected = dict(first)
    first["campaign_id"] = "corrupted-campaign"
    first["campaign_manifest_sha256"] = "corrupted-manifest"
    second = contract.campaign_bindings()
    request = contract.build_full_strength_request(required_unit_id)
    assert second == expected
    assert request["campaign_id"] == expected["campaign_id"]
    assert request["campaign_manifest_sha256"] == expected["campaign_manifest_sha256"]


def test_nested_authority_mutations_do_not_change_required_work_unit(
    required_unit_id: str,
) -> None:
    exposed = contract.required_work_unit(required_unit_id)
    exposed["identity"]["iterations"] = 1
    exposed["identity"]["k_and_n"][0] = 1
    exposed["source_packet_config_ids"].append("pkt-corruption")

    fresh = contract.required_work_unit(required_unit_id)
    request = contract.build_full_strength_request(required_unit_id)
    assert fresh["identity"]["iterations"] == 50
    assert fresh["identity"]["k_and_n"] == request["bler_identity"]["k_and_n"]
    assert fresh["source_packet_config_ids"] == request["source_packet_config_ids"]


def test_index_authority_mutations_do_not_change_single_work_unit_lookup(
    required_unit_id: str,
) -> None:
    exposed_index = contract.required_work_unit_index()
    exposed_index[required_unit_id]["identity"]["iterations"] = 2
    exposed_index[required_unit_id]["identity"]["k_and_n"].reverse()
    exposed_index[required_unit_id]["source_packet_config_ids"].reverse()
    exposed_index["corrupted-unit"] = exposed_index[required_unit_id]

    fresh = contract.required_work_unit(required_unit_id)
    assert fresh["identity"]["iterations"] == 50
    assert fresh["identity"]["k_and_n"][0] == 7128
    assert fresh["source_packet_config_ids"] == sorted(fresh["source_packet_config_ids"])
    assert "corrupted-unit" not in contract.required_work_unit_index()


def test_authority_mutation_cannot_make_corrupted_request_validate(
    required_unit_id: str,
) -> None:
    authority = contract.required_work_unit(required_unit_id)
    authority["identity"]["iterations"] = 1
    corrupted = contract.build_full_strength_request(required_unit_id)
    corrupted["bler_identity"]["iterations"] = 1
    with pytest.raises(contract.G8BlerContractError, match="identity does not match"):
        contract.validate_work_unit_request(corrupted)


def test_valid_request_still_matches_required_identity_after_mutation_attempt(
    required_unit_id: str,
) -> None:
    index = contract.required_work_unit_index()
    index[required_unit_id]["identity"]["k_and_n"][0] = 0
    index[required_unit_id]["source_packet_config_ids"].clear()
    unit = contract.required_work_unit(required_unit_id)
    request = contract.build_full_strength_request(required_unit_id)
    assert request["bler_identity"] == unit["identity"]
    assert request["source_packet_config_ids"] == unit["source_packet_config_ids"]


def test_identity_mismatch_is_rejected(full_request: dict[str, Any]) -> None:
    mutated = copy.deepcopy(full_request)
    mutated["bler_identity"]["iterations"] += 1
    with pytest.raises(contract.G8BlerContractError, match="identity does not match"):
        contract.validate_work_unit_request(mutated)


def test_snr_mismatch_is_rejected(full_request: dict[str, Any]) -> None:
    mutated = copy.deepcopy(full_request)
    mutated["snr_db"] = mutated["snr_db"] + 1
    with pytest.raises(contract.G8BlerContractError, match="SNR does not match"):
        contract.validate_work_unit_request(mutated)


def test_snr_is_never_coerced_to_an_equal_float(full_request: dict[str, Any]) -> None:
    mutated = copy.deepcopy(full_request)
    assert isinstance(mutated["snr_db"], int)
    mutated["snr_db"] = float(mutated["snr_db"])
    with pytest.raises(contract.G8BlerContractError, match="SNR does not match"):
        contract.validate_work_unit_request(mutated)


def test_work_unit_id_mismatch_is_rejected(full_request: dict[str, Any]) -> None:
    mutated = copy.deepcopy(full_request)
    mutated["work_unit_id"] = sorted(contract.required_work_unit_index())[1]
    with pytest.raises(contract.G8BlerContractError):
        contract.validate_work_unit_request(mutated)


def test_missing_or_wrong_source_packet_ids_are_rejected(full_request: dict[str, Any]) -> None:
    for mutate in (
        lambda request: request.__setitem__("source_packet_config_ids", []),
        lambda request: request.__setitem__("source_packet_config_ids", ["pkt-not-real"]),
        lambda request: request["source_packet_config_ids"].append("pkt-extra"),
    ):
        mutated = copy.deepcopy(full_request)
        mutate(mutated)
        with pytest.raises(contract.G8BlerContractError):
            contract.validate_work_unit_request(mutated)


def test_unsorted_or_duplicated_packet_ids_are_rejected(full_request: dict[str, Any]) -> None:
    mutated = copy.deepcopy(full_request)
    mutated["source_packet_config_ids"] = mutated["source_packet_config_ids"] * 2
    with pytest.raises(contract.G8BlerContractError, match="unique and canonically ordered"):
        contract.validate_work_unit_request(mutated)


def test_unknown_request_fields_are_rejected(full_request: dict[str, Any]) -> None:
    mutated = dict(full_request)
    mutated["batch_size"] = 250
    with pytest.raises(contract.G8BlerContractError, match="unknown fields"):
        contract.validate_work_unit_request(mutated)


@pytest.mark.parametrize("field", ["bler_tooling_contract_id", "bler_tooling_contract_sha256"])
def test_request_contract_binding_fields_are_mandatory(
    full_request: dict[str, Any], field: str
) -> None:
    mutated = copy.deepcopy(full_request)
    del mutated[field]
    with pytest.raises(contract.G8BlerContractError, match="missing fields"):
        contract.validate_work_unit_request(mutated)


@pytest.mark.parametrize(
    "field, value",
    [
        ("bler_tooling_contract_id", "g8bler-" + "0" * 64),
        ("bler_tooling_contract_sha256", "0" * 64),
    ],
)
def test_request_contract_binding_must_match_current_artifact(
    full_request: dict[str, Any], field: str, value: str
) -> None:
    mutated = copy.deepcopy(full_request)
    mutated[field] = value
    with pytest.raises(contract.G8BlerContractError, match="tooling contract"):
        contract.validate_work_unit_request(mutated)


def test_schema_v1_request_is_rejected(full_request: dict[str, Any]) -> None:
    mutated = copy.deepcopy(full_request)
    mutated["schema_version"] = 1
    with pytest.raises(contract.G8BlerContractError, match="unsupported work-unit request schema_version"):
        contract.validate_work_unit_request(mutated)


def test_v2_request_and_result_survive_serialization_and_revalidation(
    full_request: dict[str, Any],
) -> None:
    request_copy = json.loads(json.dumps(full_request))
    assert request_copy["schema_version"] == 2
    assert contract.validate_work_unit_request(request_copy) == request_copy
    result = _complete_full_result(full_request, bit_errors=5, block_errors=1)
    result_copy = json.loads(json.dumps(result))
    assert result_copy["schema_version"] == 2
    assert contract.validate_work_unit_result(result_copy, request=request_copy) == result_copy


@pytest.mark.parametrize("field", ["trials_requested", "snr_db", "stream_seeds", "bler_identity"])
def test_omitted_request_fields_never_take_a_default(
    full_request: dict[str, Any], field: str
) -> None:
    mutated = dict(full_request)
    del mutated[field]
    with pytest.raises(contract.G8BlerContractError, match="missing fields"):
        contract.validate_work_unit_request(mutated)


def test_full_strength_request_with_fewer_trials_is_rejected(full_request: dict[str, Any]) -> None:
    mutated = copy.deepcopy(full_request)
    mutated["trials_requested"] = SMOKE_TRIALS
    with pytest.raises(contract.G8BlerContractError, match="exactly the configured trial count"):
        contract.validate_work_unit_request(mutated)


def test_full_strength_request_may_not_borrow_the_g2_trial_source(
    full_request: dict[str, Any],
) -> None:
    mutated = copy.deepcopy(full_request)
    mutated["trial_count_source"] = "params.baseline.ldpc_bler_reference.blocks_per_snr"
    with pytest.raises(contract.G8BlerContractError, match="bler_characterisation_trials"):
        contract.validate_work_unit_request(mutated)


def test_tampered_stream_seed_is_rejected(full_request: dict[str, Any]) -> None:
    mutated = copy.deepcopy(full_request)
    mutated["stream_seeds"]["awgn_real"]["seed_uint64"] += 1
    with pytest.raises(contract.G8BlerContractError, match="does not reproduce"):
        contract.validate_work_unit_request(mutated)


def test_request_may_never_claim_merge_eligibility(full_request: dict[str, Any]) -> None:
    mutated = copy.deepcopy(full_request)
    mutated["merge_eligible"] = True
    with pytest.raises(contract.G8BlerContractError, match="never merge eligible"):
        contract.validate_work_unit_request(mutated)


def test_request_may_never_claim_test_split_access(full_request: dict[str, Any]) -> None:
    mutated = copy.deepcopy(full_request)
    mutated["test_split_access"] = 1
    with pytest.raises(contract.G8BlerContractError, match="test-split access"):
        contract.validate_work_unit_request(mutated)


def test_bounded_smoke_request_is_explicitly_non_scientific(smoke_request: dict[str, Any]) -> None:
    assert smoke_request["execution_class"] == contract.EXECUTION_CLASS_BOUNDED_SMOKE
    assert smoke_request["scientific_evidence"] is False
    assert smoke_request["merge_eligible"] is False
    assert smoke_request["label"] == contract.BOUNDED_SMOKE_LABEL
    assert "NON-SCIENTIFIC" in smoke_request["label"]
    assert smoke_request["trials_requested"] <= SMOKE_TRIALS
    assert smoke_request["trials_requested"] < contract.full_strength_trial_count()
    assert smoke_request["trial_count_source"] != contract.FULL_STRENGTH_TRIAL_COUNT_SOURCE


def test_bounded_smoke_cannot_validate_as_full_strength(smoke_request: dict[str, Any]) -> None:
    with pytest.raises(contract.G8BlerContractError, match="not 'full_strength'"):
        contract.require_full_strength_request(smoke_request)


def test_bounded_smoke_ceiling_is_enforced() -> None:
    assert contract.BOUNDED_SMOKE_MAX_WORK_UNITS == 3
    assert contract.BOUNDED_SMOKE_MAX_TRIALS_PER_UNIT == 16
    with pytest.raises(contract.G8BlerContractError, match="may not exceed"):
        contract.build_bounded_smoke_request(
            work_unit_id="bler-smoke-too-big",
            bler_identity={
                "k_and_n": [16, 32],
                "base_graph": 2,
                "lifting_size": 8,
                "modulation": "bpsk",
                "decoder_algorithm": "offset_min_sum",
                "decoder_offset": 0.5,
                "iterations": 50,
                "snr_convention": "es_n0_per_symbol",
                "rate": "1/2",
            },
            snr_db=0.0,
            source_packet_config_ids=["pkt-smoke-fixture"],
            trials_requested=SMOKE_TRIALS + 1,
        )


def test_smoke_request_relabelled_as_full_strength_is_still_rejected(
    smoke_request: dict[str, Any],
) -> None:
    mutated = copy.deepcopy(smoke_request)
    mutated["execution_class"] = contract.EXECUTION_CLASS_FULL_STRENGTH
    mutated["scientific_evidence"] = True
    mutated["label"] = contract.EXECUTION_CLASS_FULL_STRENGTH
    mutated["trial_count_source"] = contract.FULL_STRENGTH_TRIAL_COUNT_SOURCE
    with pytest.raises(contract.G8BlerContractError, match="not an exact required BLER identity"):
        contract.validate_work_unit_request(mutated)


def test_incomplete_physical_identity_is_rejected(smoke_request: dict[str, Any]) -> None:
    mutated = copy.deepcopy(smoke_request)
    del mutated["bler_identity"]["lifting_size"]
    with pytest.raises(Exception, match="incomplete BLER lookup key"):
        contract.validate_work_unit_request(mutated)


# --------------------------------------------------------------------------
# 28-43  Result schema
# --------------------------------------------------------------------------


def test_zero_block_errors_at_the_full_trial_count_is_characterized_evidence(
    full_request: dict[str, Any],
) -> None:
    result = _complete_full_result(full_request, bit_errors=0, block_errors=0)
    assert result["status"] == contract.STATUS_COMPLETE
    assert result["measurement"]["bler"] == 0.0
    assert result["measurement"]["ber"] == 0.0
    assert result["disposition"]["merge_eligible"] is True
    assert result["disposition"]["required_coverage_contribution"] == 1
    assert result["measurement"]["bler_confidence_high"] > 0.0


def test_every_block_failing_is_characterized_evidence(full_request: dict[str, Any]) -> None:
    trials = full_request["trials_requested"]
    k = full_request["bler_identity"]["k_and_n"][0]
    result = _complete_full_result(full_request, bit_errors=trials * k, block_errors=trials)
    assert result["measurement"]["bler"] == 1.0
    assert result["measurement"]["ber"] == 1.0
    assert result["disposition"]["merge_eligible"] is True


@pytest.mark.parametrize("field", ["trials_completed", "bit_errors", "block_errors"])
def test_negative_counts_are_rejected(full_request: dict[str, Any], field: str) -> None:
    result = _complete_full_result(full_request, bit_errors=1, block_errors=1)
    result["measurement"][field] = -1
    with pytest.raises(contract.G8BlerContractError, match="non-negative"):
        contract.validate_work_unit_result(result)


@pytest.mark.parametrize("field", ["trials_completed", "bit_errors", "block_errors", "information_bits"])
def test_boolean_counts_are_rejected(full_request: dict[str, Any], field: str) -> None:
    result = _complete_full_result(full_request, bit_errors=1, block_errors=1)
    result["measurement"][field] = True
    with pytest.raises(contract.G8BlerContractError, match="not a boolean"):
        contract.validate_work_unit_result(result)


def test_block_errors_exceeding_trials_are_rejected(full_request: dict[str, Any]) -> None:
    result = _complete_full_result(full_request, bit_errors=1, block_errors=1)
    result["measurement"]["block_errors"] = result["measurement"]["trials_completed"] + 1
    with pytest.raises(contract.G8BlerContractError, match="block_errors exceeds"):
        contract.validate_work_unit_result(result)


def test_bit_errors_exceeding_information_bits_are_rejected(full_request: dict[str, Any]) -> None:
    result = _complete_full_result(full_request, bit_errors=1, block_errors=1)
    result["measurement"]["bit_errors"] = result["measurement"]["information_bits"] + 1
    with pytest.raises(contract.G8BlerContractError, match="bit_errors exceeds"):
        contract.validate_work_unit_result(result)


def test_zero_bit_errors_with_a_block_error_are_rejected(full_request: dict[str, Any]) -> None:
    result = _complete_full_result(full_request, bit_errors=1, block_errors=1)
    result["measurement"]["bit_errors"] = 0
    with pytest.raises(contract.G8BlerContractError, match="block_errors == 0 iff bit_errors == 0"):
        contract.validate_work_unit_result(result)


def test_nonzero_bit_errors_with_zero_block_errors_are_rejected(full_request: dict[str, Any]) -> None:
    result = _complete_full_result(full_request, bit_errors=1, block_errors=1)
    result["measurement"]["block_errors"] = 0
    with pytest.raises(contract.G8BlerContractError, match="block_errors == 0 iff bit_errors == 0"):
        contract.validate_work_unit_result(result)


def test_bit_errors_cannot_be_fewer_than_erroneous_blocks(full_request: dict[str, Any]) -> None:
    result = _complete_full_result(full_request, bit_errors=5, block_errors=1)
    result["measurement"]["block_errors"] = 6
    with pytest.raises(contract.G8BlerContractError, match="between block_errors"):
        contract.validate_work_unit_result(result)


def test_bit_errors_cannot_exceed_k_per_erroneous_block(full_request: dict[str, Any]) -> None:
    k = full_request["bler_identity"]["k_and_n"][0]
    result = _complete_full_result(full_request, bit_errors=k, block_errors=1)
    result["measurement"]["bit_errors"] = k + 1
    with pytest.raises(contract.G8BlerContractError, match="between block_errors"):
        contract.validate_work_unit_result(result)


def test_decoder_exception_placeholder_cannot_be_completed_evidence(
    full_request: dict[str, Any],
) -> None:
    # A decoder exception has no decoded K-bit vector.  Treating it as one
    # block error with zero compared bit errors is therefore not evidence.
    result = _complete_full_result(full_request, bit_errors=1, block_errors=1)
    result["measurement"]["bit_errors"] = 0
    with pytest.raises(contract.G8BlerContractError, match="block_errors == 0 iff bit_errors == 0"):
        contract.validate_work_unit_result(result)


def test_one_bit_error_per_erroneous_block_is_valid(full_request: dict[str, Any]) -> None:
    result = _complete_full_result(full_request, bit_errors=3, block_errors=3)
    assert result["measurement"]["bit_errors"] == 3
    assert result["measurement"]["block_errors"] == 3


def test_exactly_k_bit_errors_per_erroneous_block_is_valid(full_request: dict[str, Any]) -> None:
    k = full_request["bler_identity"]["k_and_n"][0]
    result = _complete_full_result(full_request, bit_errors=3 * k, block_errors=3)
    assert result["measurement"]["bit_errors"] == 3 * k


def test_intermediate_error_count_is_valid(full_request: dict[str, Any]) -> None:
    k = full_request["bler_identity"]["k_and_n"][0]
    result = _complete_full_result(full_request, bit_errors=k + 2, block_errors=3)
    assert result["measurement"]["bit_errors"] == k + 2


@pytest.mark.parametrize("status", [contract.STATUS_INCOMPLETE, contract.STATUS_FAILED])
def test_zero_trial_noncharacterized_result_has_zero_counts_and_null_rates(
    full_request: dict[str, Any], status: str
) -> None:
    result = contract.build_work_unit_result(
        request=full_request,
        status=status,
        trials_completed=0,
        bit_errors=0,
        block_errors=0,
    )
    assert result["measurement"]["information_bits"] == 0
    assert result["measurement"]["ber"] is None
    assert result["measurement"]["bler"] is None


def test_incorrect_information_bit_total_is_rejected(full_request: dict[str, Any]) -> None:
    result = _complete_full_result(full_request, bit_errors=1, block_errors=1)
    result["measurement"]["information_bits"] += 1
    with pytest.raises(contract.G8BlerContractError, match="trials_completed x K"):
        contract.validate_work_unit_result(result)


def test_incorrect_stored_bler_is_rejected(full_request: dict[str, Any]) -> None:
    result = _complete_full_result(full_request, bit_errors=10, block_errors=2)
    result["measurement"]["bler"] = 0.0
    with pytest.raises(contract.G8BlerContractError, match="stored bler does not reproduce"):
        contract.validate_work_unit_result(result)


def test_incorrect_stored_ber_is_rejected(full_request: dict[str, Any]) -> None:
    result = _complete_full_result(full_request, bit_errors=10, block_errors=2)
    result["measurement"]["ber"] = 0.0
    with pytest.raises(contract.G8BlerContractError, match="stored ber does not reproduce"):
        contract.validate_work_unit_result(result)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nan_and_infinity_are_rejected(full_request: dict[str, Any], value: float) -> None:
    result = _complete_full_result(full_request, bit_errors=10, block_errors=2)
    result["measurement"]["bler"] = value
    with pytest.raises(contract.G8BlerContractError):
        contract.validate_work_unit_result(result)


def test_incorrect_confidence_bounds_are_rejected(full_request: dict[str, Any]) -> None:
    result = _complete_full_result(full_request, bit_errors=10, block_errors=2)
    result["measurement"]["bler_confidence_low"] = 0.0
    with pytest.raises(contract.G8BlerContractError, match="does not reproduce"):
        contract.validate_work_unit_result(result)


def test_confidence_bounds_reproduce_from_counts(full_request: dict[str, Any]) -> None:
    result = _complete_full_result(full_request, bit_errors=1000, block_errors=25)
    trials = result["measurement"]["trials_completed"]
    z = NormalDist().inv_cdf(0.5 + 95 / 200.0)
    p = 25 / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    assert result["measurement"]["bler_confidence_low"] == pytest.approx(centre - margin, abs=1e-15)
    assert result["measurement"]["bler_confidence_high"] == pytest.approx(centre + margin, abs=1e-15)
    assert result["measurement"]["confidence_interval_percent"] == 95
    assert "diagnostic only" in result["measurement"]["confidence_interval_role"]


def test_fixed_wilson_vectors_match_independently_computed_values() -> None:
    assert list(contract.wilson_interval(0, 16)) == EXPECTED_WILSON["zero_errors_16_trials"]
    assert list(contract.wilson_interval(1, 16)) == EXPECTED_WILSON["one_error_16_trials"]
    assert list(contract.wilson_interval(16, 16)) == EXPECTED_WILSON["all_errors_16_trials"]


def test_incomplete_full_strength_result_is_not_merge_eligible(
    full_request: dict[str, Any],
) -> None:
    result = contract.build_work_unit_result(
        request=full_request,
        status=contract.STATUS_INCOMPLETE,
        trials_completed=full_request["trials_requested"] // 2,
        bit_errors=3,
        block_errors=1,
    )
    assert result["status"] == contract.STATUS_INCOMPLETE
    assert result["disposition"]["merge_eligible"] is False
    assert result["disposition"]["required_coverage_contribution"] == 0


def test_failed_result_is_not_merge_eligible(full_request: dict[str, Any]) -> None:
    result = contract.build_work_unit_result(
        request=full_request,
        status=contract.STATUS_FAILED,
        trials_completed=0,
        bit_errors=0,
        block_errors=0,
    )
    assert result["disposition"]["merge_eligible"] is False
    assert result["measurement"]["bler"] is None
    assert result["measurement"]["ber"] is None
    assert result["measurement"]["bler_confidence_low"] is None
    assert result["measurement"]["bler_confidence_high"] is None


def test_zero_completed_trials_may_not_report_a_zero_rate(full_request: dict[str, Any]) -> None:
    result = contract.build_work_unit_result(
        request=full_request,
        status=contract.STATUS_FAILED,
        trials_completed=0,
        bit_errors=0,
        block_errors=0,
    )
    result["measurement"]["bler"] = 0.0
    with pytest.raises(contract.G8BlerContractError, match="null at zero completed trials"):
        contract.validate_work_unit_result(result)


def test_completed_status_requires_positive_trials(full_request: dict[str, Any]) -> None:
    result = contract.build_work_unit_result(
        request=full_request,
        status=contract.STATUS_FAILED,
        trials_completed=0,
        bit_errors=0,
        block_errors=0,
    )
    result["status"] = contract.STATUS_COMPLETE
    with pytest.raises(contract.G8BlerContractError, match="trials_completed > 0"):
        contract.validate_work_unit_result(result)


def test_completed_full_strength_result_is_merge_eligible(full_request: dict[str, Any]) -> None:
    result = _complete_full_result(full_request, bit_errors=42, block_errors=7)
    assert result["disposition"]["merge_eligible"] is True
    assert result["disposition"]["scientific_evidence"] is True
    assert result["measurement"]["trials_completed"] == contract.full_strength_trial_count()


def test_merge_eligibility_cannot_be_asserted_by_hand(full_request: dict[str, Any]) -> None:
    result = contract.build_work_unit_result(
        request=full_request,
        status=contract.STATUS_INCOMPLETE,
        trials_completed=10,
        bit_errors=0,
        block_errors=0,
    )
    result["disposition"]["merge_eligible"] = True
    with pytest.raises(contract.G8BlerContractError, match="merge eligibility must follow"):
        contract.validate_work_unit_result(result)


def test_smoke_result_is_never_merge_eligible(smoke_request: dict[str, Any]) -> None:
    result = contract.build_work_unit_result(
        request=smoke_request,
        status=contract.STATUS_COMPLETE,
        trials_completed=smoke_request["trials_requested"],
        bit_errors=1,
        block_errors=1,
    )
    assert result["disposition"]["merge_eligible"] is False
    assert result["disposition"]["scientific_evidence"] is False
    assert result["disposition"]["required_coverage_contribution"] == 0
    result["disposition"]["merge_eligible"] = True
    with pytest.raises(contract.G8BlerContractError):
        contract.validate_work_unit_result(result)


def test_runtime_metadata_does_not_change_the_measurement_identity(
    full_request: dict[str, Any],
) -> None:
    bare = _complete_full_result(full_request, bit_errors=5, block_errors=1)
    annotated = contract.build_work_unit_result(
        request=full_request,
        status=contract.STATUS_COMPLETE,
        trials_completed=full_request["trials_requested"],
        bit_errors=5,
        block_errors=1,
        execution_metadata={
            "wall_time_s": 931.5,
            "hostname": "some-host",
            "device": "cuda:0",
            "shard_index": 7,
            "shard_count": 32,
            "attempt": 3,
        },
    )
    assert annotated["execution_metadata"] != bare["execution_metadata"]
    assert contract.measurement_identity_digest(annotated) == contract.measurement_identity_digest(
        bare
    )


def test_unknown_execution_metadata_is_rejected(full_request: dict[str, Any]) -> None:
    with pytest.raises(contract.G8BlerContractError, match="unknown fields"):
        contract.build_work_unit_result(
            request=full_request,
            status=contract.STATUS_COMPLETE,
            trials_completed=full_request["trials_requested"],
            bit_errors=1,
            block_errors=1,
            execution_metadata={"decoder_exception": True},
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -1.0, True, "1"])
def test_wall_time_metadata_is_finite_nonnegative_real(
    full_request: dict[str, Any], value: Any
) -> None:
    with pytest.raises(contract.G8BlerContractError):
        contract.build_work_unit_result(
            request=full_request,
            status=contract.STATUS_FAILED,
            trials_completed=0,
            bit_errors=0,
            block_errors=0,
            execution_metadata={"wall_time_s": value},
        )


@pytest.mark.parametrize("field", ["hostname", "device"])
def test_hostname_and_device_must_be_nonblank_strings(
    full_request: dict[str, Any], field: str
) -> None:
    for value in ("", " \t", 1, True):
        with pytest.raises(contract.G8BlerContractError):
            contract.build_work_unit_result(
                request=full_request,
                status=contract.STATUS_FAILED,
                trials_completed=0,
                bit_errors=0,
                block_errors=0,
                execution_metadata={field: value},
            )


@pytest.mark.parametrize("value", [-1, True, "0"])
def test_shard_index_requires_an_exact_nonnegative_integer(
    full_request: dict[str, Any], value: Any
) -> None:
    with pytest.raises(contract.G8BlerContractError):
        contract.build_work_unit_result(
            request=full_request,
            status=contract.STATUS_FAILED,
            trials_completed=0,
            bit_errors=0,
            block_errors=0,
            execution_metadata={"shard_index": value, "shard_count": 2},
        )


@pytest.mark.parametrize("value", [0, -1, True, "2"])
def test_shard_count_requires_an_exact_positive_integer(
    full_request: dict[str, Any], value: Any
) -> None:
    with pytest.raises(contract.G8BlerContractError):
        contract.build_work_unit_result(
            request=full_request,
            status=contract.STATUS_FAILED,
            trials_completed=0,
            bit_errors=0,
            block_errors=0,
            execution_metadata={"shard_index": 0, "shard_count": value},
        )


@pytest.mark.parametrize(
    "metadata",
    [{"shard_index": 0}, {"shard_count": 2}, {"shard_index": 2, "shard_count": 2}],
)
def test_shard_pair_is_complete_and_in_range(
    full_request: dict[str, Any], metadata: dict[str, Any]
) -> None:
    with pytest.raises(contract.G8BlerContractError):
        contract.build_work_unit_result(
            request=full_request,
            status=contract.STATUS_FAILED,
            trials_completed=0,
            bit_errors=0,
            block_errors=0,
            execution_metadata=metadata,
        )


@pytest.mark.parametrize("value", [0, -1, True, "1"])
def test_attempt_requires_an_exact_positive_integer(
    full_request: dict[str, Any], value: Any
) -> None:
    with pytest.raises(contract.G8BlerContractError):
        contract.build_work_unit_result(
            request=full_request,
            status=contract.STATUS_FAILED,
            trials_completed=0,
            bit_errors=0,
            block_errors=0,
            execution_metadata={"attempt": value},
        )


def test_all_null_execution_metadata_is_valid(full_request: dict[str, Any]) -> None:
    result = contract.build_work_unit_result(
        request=full_request,
        status=contract.STATUS_FAILED,
        trials_completed=0,
        bit_errors=0,
        block_errors=0,
        execution_metadata={name: None for name in contract.RESULT_EXECUTION_METADATA_FIELDS},
    )
    assert result["execution_metadata"] == {
        name: None for name in contract.RESULT_EXECUTION_METADATA_FIELDS
    }


def test_changed_identity_with_copied_counts_is_rejected(
    full_request: dict[str, Any], required_unit_id: str
) -> None:
    result = _complete_full_result(full_request, bit_errors=5, block_errors=1)
    other_id = sorted(contract.required_work_unit_index())[1]
    result["identity"]["work_unit_id"] = other_id
    with pytest.raises(contract.G8BlerContractError, match="does not reproduce"):
        contract.validate_work_unit_result(result)


def test_result_must_bind_the_request_it_claims(
    full_request: dict[str, Any], smoke_request: dict[str, Any]
) -> None:
    result = _complete_full_result(full_request, bit_errors=5, block_errors=1)
    with pytest.raises(contract.G8BlerContractError, match="does not bind the request"):
        contract.validate_work_unit_result(result, request=smoke_request)


def test_result_sections_are_closed(full_request: dict[str, Any]) -> None:
    result = _complete_full_result(full_request, bit_errors=5, block_errors=1)
    extra = copy.deepcopy(result)
    extra["measurement"]["observed_waterfall"] = 0.01
    with pytest.raises(contract.G8BlerContractError, match="measurement section"):
        contract.validate_work_unit_result(extra)
    missing = copy.deepcopy(result)
    del missing["execution_metadata"]
    with pytest.raises(contract.G8BlerContractError, match="missing or unknown sections"):
        contract.validate_work_unit_result(missing)


def test_result_may_never_claim_test_split_access(full_request: dict[str, Any]) -> None:
    result = _complete_full_result(full_request, bit_errors=5, block_errors=1)
    result["disposition"]["test_split_access"] = 1
    with pytest.raises(contract.G8BlerContractError, match="test-split access"):
        contract.validate_work_unit_result(result)


@pytest.mark.parametrize("field", ["bler_tooling_contract_id", "bler_tooling_contract_sha256"])
def test_result_contract_binding_fields_are_mandatory(
    full_request: dict[str, Any], field: str
) -> None:
    result = _complete_full_result(full_request, bit_errors=5, block_errors=1)
    del result["identity"][field]
    with pytest.raises(contract.G8BlerContractError, match="identity section"):
        contract.validate_work_unit_result(result)


@pytest.mark.parametrize(
    "field, value",
    [
        ("bler_tooling_contract_id", "g8bler-" + "0" * 64),
        ("bler_tooling_contract_sha256", "0" * 64),
    ],
)
def test_result_contract_binding_must_match_its_request(
    full_request: dict[str, Any], field: str, value: str
) -> None:
    result = _complete_full_result(full_request, bit_errors=5, block_errors=1)
    result["identity"][field] = value
    with pytest.raises(contract.G8BlerContractError, match="tooling contract"):
        contract.validate_work_unit_result(result)


def test_result_copied_from_another_tooling_contract_is_rejected(
    full_request: dict[str, Any],
) -> None:
    result = _complete_full_result(full_request, bit_errors=5, block_errors=1)
    result["identity"]["bler_tooling_contract_id"] = "g8bler-" + "1" * 64
    with pytest.raises(contract.G8BlerContractError, match="tooling contract"):
        contract.validate_work_unit_result(result)


def test_recomputed_digest_cannot_hide_a_changed_contract_binding(
    full_request: dict[str, Any],
) -> None:
    result = _complete_full_result(full_request, bit_errors=5, block_errors=1)
    result["identity"]["bler_tooling_contract_sha256"] = "0" * 64
    rebuilt = {
        "schema_version": contract.BLER_WORK_UNIT_REQUEST_SCHEMA_VERSION,
        "artifact_role": contract.REQUEST_ARTIFACT_ROLE,
        "execution_class": result["identity"]["execution_class"],
        "campaign_id": result["identity"]["campaign_id"],
        "bler_tooling_contract_id": result["identity"]["bler_tooling_contract_id"],
        "bler_tooling_contract_sha256": result["identity"]["bler_tooling_contract_sha256"],
        "campaign_manifest_sha256": result["identity"]["campaign_manifest_sha256"],
        "required_bler_artifact_sha256": result["identity"]["required_bler_artifact_sha256"],
        "selection_policy_sha256": result["identity"]["selection_policy_sha256"],
        "work_unit_id": result["identity"]["work_unit_id"],
        "bler_identity": result["identity"]["bler_identity"],
        "snr_db": result["identity"]["snr_db"],
        "source_packet_config_ids": result["identity"]["source_packet_config_ids"],
        "trials_requested": result["identity"]["trials_requested"],
        "trial_count_source": result["identity"]["trial_count_source"],
        "seed_derivation_identity": result["identity"]["seed_derivation_identity"],
        "seed_domain_separator": result["identity"]["seed_domain_separator"],
        "stream_seeds": result["identity"]["stream_seeds"],
        "scientific_evidence": True,
        "merge_eligible": False,
        "test_split_access": 0,
        "label": contract.EXECUTION_CLASS_FULL_STRENGTH,
    }
    result["identity"]["request_sha256"] = contract.request_digest(rebuilt)
    with pytest.raises(contract.G8BlerContractError, match="tooling contract"):
        contract.validate_work_unit_result(result)


def test_result_dependency_binding_must_match(full_request: dict[str, Any]) -> None:
    result = _complete_full_result(full_request, bit_errors=5, block_errors=1)
    result["identity"]["implementation"]["rng_library_version"] = "0.0.0"
    with pytest.raises(contract.G8BlerContractError, match="dependency binding"):
        contract.validate_work_unit_result(result)


# --------------------------------------------------------------------------
# 44-56  Generated artifact and independent verifier
# --------------------------------------------------------------------------


def test_corrected_loader_and_independent_verifier_accept_generated_b1c_artifact(
    corrected_b1c_contract_artifact: Path,
) -> None:
    loaded = contract.load_bler_tooling_contract(corrected_b1c_contract_artifact)
    verified = contract_verifier.verify(corrected_b1c_contract_artifact)
    assert loaded["schema_version"] == 2
    assert loaded["checkpoint"] == "B1C"
    assert verified["contract_id"] == loaded["contract_id"]
    binding = contract.tooling_contract_binding()
    assert binding["bler_tooling_contract_id"] == loaded["contract_id"]
    assert binding["bler_tooling_contract_sha256"] == hashlib.sha256(
        corrected_b1c_contract_artifact.read_bytes()
    ).hexdigest()


def test_generator_write_then_check_is_byte_identical() -> None:
    payload = generator.build()
    assert generator.BLER_TOOLING_CONTRACT.read_bytes() == rendered_json(payload)
    assert generator.main(["--check"]) == 0


def test_independent_verifier_passes_on_the_committed_artifact() -> None:
    payload = contract_verifier.verify()
    assert payload["contract_id"].startswith("g8bler-")
    assert payload["scientific_execution_performed"] is False
    assert payload["characterization_started"] is False
    assert payload["bounded_smoke_started"] is False


def test_contract_id_reproduces_and_excludes_itself() -> None:
    payload = json.loads(generator.BLER_TOOLING_CONTRACT.read_text(encoding="utf-8"))
    assert payload["contract_id"] == generator.contract_identifier(payload)
    without = {key: value for key, value in payload.items() if key != "contract_id"}
    assert generator.contract_identifier(payload) == generator.contract_identifier(without)


def test_contract_is_independent_of_timestamps_and_absolute_paths() -> None:
    text = generator.BLER_TOOLING_CONTRACT.read_text(encoding="utf-8")
    payload = json.loads(text)
    for entry in payload["contract_sources"]:
        assert not Path(entry["path"]).is_absolute()
    assert str(Path.home()) not in text
    assert socket.gethostname() not in text
    # "hostname" appears only as a declared *non-identity* result field name.
    assert payload["result_schema"]["execution_metadata_fields"] == list(
        contract.NON_IDENTITY_EXECUTION_METADATA
    )
    assert "/home/" not in text

    def keys(node: Any) -> list[str]:
        if isinstance(node, dict):
            return list(node) + [name for value in node.values() for name in keys(value)]
        if isinstance(node, list):
            return [name for value in node for name in keys(value)]
        return []

    for forbidden in ("timestamp", "generated_at", "commit_sha", "created"):
        assert forbidden not in keys(payload)
    assert not re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", text)
    # Rebuilding at a different wall-clock time yields the identical contract.
    assert generator.build()["contract_id"] == payload["contract_id"]
    assert rendered_json(generator.build()) == generator.BLER_TOOLING_CONTRACT.read_bytes()


def test_contract_binds_no_future_runner_or_its_own_output() -> None:
    payload = json.loads(generator.BLER_TOOLING_CONTRACT.read_text(encoding="utf-8"))
    bound = [entry["path"] for entry in payload["contract_sources"]]
    assert bound == list(generator.CONTRACT_SOURCES)
    assert "results/baseline/g8/bler_tooling_contract.json" not in bound
    for forbidden in ("shard", "checkpoint", "merge", "runner"):
        assert not any(forbidden in path for path in bound)


def _mutated_contract(tmp_path: Path, mutate: Callable[[dict], None], *, refresh_id: bool = True) -> Path:
    payload = generator.build()
    mutate(payload)
    if refresh_id:
        payload.pop("contract_id", None)
        payload["contract_id"] = generator.contract_identifier(payload)
    path = tmp_path / "bler_tooling_contract.json"
    path.write_bytes(rendered_json(payload))
    return path


MUTATIONS: dict[str, Callable[[dict], None]] = {
    "unknown_top_level": lambda p: p.__setitem__("unexpected", True),
    "tooling_schema_version": lambda p: p.__setitem__("schema_version", 1),
    "campaign_hash": lambda p: p["campaign_bindings"]["campaign_manifest"].__setitem__(
        "sha256", "0" * 64
    ),
    "campaign_id": lambda p: p["campaign_bindings"].__setitem__("campaign_id", "g8-other"),
    "required_artifact_hash": lambda p: p["campaign_bindings"][
        "required_bler_identities"
    ].__setitem__("sha256", "0" * 64),
    "required_work_unit_count": lambda p: p["campaign_bindings"].__setitem__(
        "required_work_unit_count", 1
    ),
    "selection_policy_hash": lambda p: p["campaign_bindings"].__setitem__(
        "selection_policy_sha256", "0" * 64
    ),
    "trial_count_source": lambda p: p["trial_count"].__setitem__(
        "parameter", "baseline.ldpc_bler_reference.blocks_per_snr"
    ),
    "trial_count_value": lambda p: p["trial_count"].__setitem__("full_strength_trials", 7),
    "adaptive_stopping": lambda p: p["trial_count"].__setitem__(
        "adaptive_stopping_permitted", True
    ),
    "seed_domain": lambda p: p["seed"].__setitem__("domain_separator", "capstone:g8:other:v1"),
    "seed_identity": lambda p: p["seed"].__setitem__("derivation_identity", "sha256(x)-v2"),
    "seed_input_encoding": lambda p: p["seed"].__setitem__("input_encoding", "other"),
    "seed_output_rule": lambda p: p["seed"].__setitem__("output_rule", "other"),
    "seed_vector": lambda p: p["seed"]["test_vectors"]["seeds"]["awgn_real"].__setitem__(
        "seed_uint64", 1
    ),
    "seed_material_hash": lambda p: p["seed"]["test_vectors"]["seeds"][
        "information_bits"
    ].__setitem__("material_sha256", "0" * 64),
    "seed_bit_vector": lambda p: p["seed"]["test_vectors"]["seeds"][
        "information_bits"
    ].__setitem__("bits_60_to_68", [0] * 8),
    "seed_purposes": lambda p: p["seed"].__setitem__("purposes", ["information_bits"]),
    "request_schema": lambda p: p["request_schema"].__setitem__("fields", ["schema_version"]),
    "request_merge_rule": lambda p: p["request_schema"].__setitem__(
        "request_is_never_merge_eligible", False
    ),
    "result_schema": lambda p: p["result_schema"].__setitem__("measurement_fields", ["bler"]),
    "result_statuses": lambda p: p["result_schema"].__setitem__("statuses", ["complete"]),
    "result_metadata_rules": lambda p: p["result_schema"]["execution_metadata_rules"].__setitem__(
        "attempt", "anything"
    ),
    "smoke_merge_rule": lambda p: p["merge_rules"].__setitem__(
        "bounded_smoke_is_merge_eligible", True
    ),
    "smoke_evidence_rule": lambda p: p["execution_classes"].__setitem__(
        "bounded_smoke_is_scientific_evidence", True
    ),
    "incomplete_merge_rule": lambda p: p["merge_rules"].__setitem__(
        "incomplete_result_is_merge_eligible", True
    ),
    "confidence_role": lambda p: p["confidence"].__setitem__("role", "used in BR-4 ranking"),
    "confidence_percent": lambda p: p["confidence"].__setitem__("percent", 90),
    "confidence_method": lambda p: p["confidence"].__setitem__("method", "normal_approximation"),
    "wilson_vector": lambda p: p["seed"]["test_vectors"]["wilson"].__setitem__(
        "all_errors_16_trials", [0.0, 1.0]
    ),
    "rng_bit_generator": lambda p: p["rng"].__setitem__("bit_generator", "PCG64"),
    "rng_version": lambda p: p["rng"].__setitem__("library_version", "0.0.0"),
    "rng_extraction": lambda p: p["rng"].__setitem__(
        "information_bit_extraction", "bit_i = (word[i // 8] >> (i % 8)) & 1"
    ),
    "rng_invariance_claim": lambda p: p["rng"].__setitem__("chunk_boundary_invariant", False),
    "resume_granularity": lambda p: p["resume"].__setitem__("granularity", "trial"),
    "mid_unit_resume": lambda p: p["resume"].__setitem__("mid_work_unit_resume_permitted", True),
    "counts_authority": lambda p: p["count_authority"].__setitem__(
        "counts_override_stored_floats", False
    ),
    "zero_error_evidence": lambda p: p["count_authority"].__setitem__(
        "zero_errors_is_characterized_evidence", False
    ),
    "cross_count_invariants": lambda p: p["count_authority"].__setitem__(
        "cross_count_invariants", []
    ),
    "execution_class": lambda p: p["execution_classes"].__setitem__(
        "full_strength", "other"
    ),
    "smoke_selection_rule": lambda p: p["execution_classes"].__setitem__(
        "bounded_smoke_selection_rule", "other"
    ),
    "smoke_label": lambda p: p["execution_classes"].__setitem__(
        "bounded_smoke_label", "other"
    ),
    "source_binding": lambda p: p["contract_sources"][0].__setitem__("sha256", "0" * 64),
    "phase": lambda p: p.__setitem__("phase", "G8_C"),
    "execution_claim": lambda p: p.__setitem__("scientific_execution_performed", True),
    "no_interpolation": lambda p: p["rules"].__setitem__(
        "no_interpolation_or_extrapolation", "other"
    ),
    "supersession_claim": lambda p: p.__setitem__("supersession_reason", "other"),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_every_contract_mutation_fails_the_independent_verifier(
    tmp_path: Path, name: str
) -> None:
    path = _mutated_contract(tmp_path, MUTATIONS[name])
    with pytest.raises(contract_verifier.G8BlerToolingError):
        contract_verifier.verify(path)


def test_mutated_contract_id_alone_fails(tmp_path: Path) -> None:
    path = _mutated_contract(tmp_path, lambda p: p.__setitem__("contract_id", "g8bler-" + "0" * 64),
                             refresh_id=False)
    with pytest.raises(contract_verifier.G8BlerToolingError, match="contract_id does not reproduce"):
        contract_verifier.verify(path)


def test_noncanonical_contract_bytes_are_rejected(tmp_path: Path) -> None:
    payload = generator.build()
    path = tmp_path / "bler_tooling_contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(contract_verifier.G8BlerToolingError, match="canonical rendered JSON"):
        contract_verifier.verify(path)


def test_contract_records_the_frozen_seed_and_rng_contract() -> None:
    payload = json.loads(generator.BLER_TOOLING_CONTRACT.read_text(encoding="utf-8"))
    assert payload["seed"]["derivation_identity"] == "sha256(campaign_id,work_unit_id,purpose)-v1"
    assert payload["seed"]["domain_separator"] == "capstone:g8:bler-seed:v1"
    assert payload["seed"]["purposes"] == ["information_bits", "awgn_real", "awgn_imag"]
    assert payload["seed"]["test_vectors"]["seeds"]["information_bits"]["seed_uint64"] == (
        EXPECTED_SEED_UINT64["information_bits"]
    )
    assert payload["rng"]["bit_generator"] == "Philox"
    assert payload["rng"]["information_bit_extraction"] == "bit_i = (word[i // 64] >> (i % 64)) & 1"
    assert payload["trial_count"]["full_strength_trials"] == contract.full_strength_trial_count()
    assert payload["trial_count"]["source"] == "params.baseline.bler_characterisation_trials"


def test_contract_module_imports_no_execution_stack() -> None:
    tree = ast.parse(CONTRACT_SOURCE.read_text(encoding="utf-8"), filename=str(CONTRACT_SOURCE))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    for forbidden in ("torch", "sionna", "data", "datasets", "training", "baseline.ldpc"):
        assert not any(name == forbidden or name.startswith(f"{forbidden}.") for name in imported)
    source = CONTRACT_SOURCE.read_text(encoding="utf-8")
    assert "G8Authorization" not in source
    assert "BlerTable" not in source


def test_canonical_identity_comparison_distinguishes_int_and_float() -> None:
    assert canonical_json(13) != canonical_json(13.0)
