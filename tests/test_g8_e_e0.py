"""E0 opening boundary tests; no validation data path is used."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from baseline.g8_e import E0_PATH, G8EContractError, rendered_json, validate_e0_opening, verify_e0_file


def test_committed_e0_is_open_and_zero_coverage(post_g10_am94) -> None:
    del post_g10_am94
    value = verify_e0_file(E0_PATH)
    assert value["status"] == "OPEN"
    assert value["safety"]["g8_e_measurement_coverage"] == 0
    assert value["safety"]["test_access"] == 0
    assert value["declarations"]["validation_image_decoding_required_to_open"] is False


def test_e0_binds_portable_epoch_and_current_d7(post_g10_am94) -> None:
    del post_g10_am94
    value = verify_e0_file(E0_PATH)
    assert value["g8_c"]["portable_verification_epoch"]["epoch"] == "g8-c-portable-scientific-runtime-v1"
    assert value["g8_c"]["portable_verification_epoch"]["legacy_tree_digest_is_historical_only"] is True
    assert value["g8_d"]["contract_id"].startswith("g8dcontract-")
    assert value["g8_d"]["handoff_id"].startswith("g8dhandoff-")


@pytest.mark.parametrize(
    "mutation",
    [
        ("wrong portable manifest", lambda value: value["g8_c"].__setitem__("portable_manifest_id", "wrong")),
        ("wrong portable provenance", lambda value: value["g8_c"].__setitem__("portable_provenance_id", "wrong")),
        ("wrong D contract", lambda value: value["g8_d"].__setitem__("contract_id", "wrong")),
        ("coverage", lambda value: value["safety"].__setitem__("g8_e_measurement_coverage", 1)),
        ("test access", lambda value: value["safety"].__setitem__("test_access", 1)),
        ("E2 started", lambda value: value["safety"].__setitem__("e2_started", True)),
    ],
    ids=lambda item: item[0],
)
def test_e0_mutations_fail_closed(mutation, post_g10_am94) -> None:
    del post_g10_am94
    value = copy.deepcopy(verify_e0_file(E0_PATH))
    mutation[1](value)
    with pytest.raises(G8EContractError):
        validate_e0_opening(value)


def test_e0_file_is_canonical(post_g10_am94) -> None:
    del post_g10_am94
    raw = E0_PATH.read_bytes()
    assert raw == rendered_json(json.loads(raw))
