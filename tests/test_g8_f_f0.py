"""Fail-closed mutations for the frozen G8_F/F0 opening."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from baseline.g8_f_f0 import (
    AUTHORIZATION_PATH,
    AUTHORIZATION_PREFIX,
    V1_AUTHORIZATION_ID,
    V1_AUTHORIZATION_PATH,
    V1_FILE_SHA256,
    G8FF0Error,
    canonical_json,
    rendered_json,
    sha256_bytes,
    verify_f0_authorization,
    verify_f0_v1_historical,
)


@pytest.fixture(scope="module")
def committed() -> dict:
    return verify_f0_authorization()


def _reject(tmp_path: Path, value: dict) -> None:
    mutated = copy.deepcopy(value)
    mutated.pop("authorization_id", None)
    mutated["authorization_id"] = AUTHORIZATION_PREFIX + sha256_bytes(canonical_json(mutated))
    path = tmp_path / "f0.json"
    path.write_bytes(rendered_json(mutated))
    with pytest.raises(G8FF0Error):
        verify_f0_authorization(path)


@pytest.mark.parametrize(
    "mutation",
    ["am88_id", "am88_sha", "ordered", "pair_set", "variants", "attempts"],
)
def test_sampler_identity_and_multiplicity_mutations_hold(tmp_path: Path, committed: dict, mutation: str) -> None:
    value = copy.deepcopy(committed)
    if mutation == "am88_id":
        value["protocol"]["am88_sampler_plan"]["plan_id"] = "foreign"
    elif mutation == "am88_sha":
        value["protocol"]["am88_sampler_plan"]["sha256"] = "0" * 64
    elif mutation == "ordered":
        value["protocol"]["ordered_pair_sha256"] = "0" * 64
    elif mutation == "pair_set":
        value["protocol"]["pair_set_sha256"] = "0" * 64
    elif mutation == "variants":
        value["protocol"]["sampler"]["variants_per_training_image"] = 5
    else:
        value["protocol"]["nominal_attempt_count"] = 1_016_280
    _reject(tmp_path, value)


@pytest.mark.parametrize("mutation", ["manifest", "codec", "lock", "source"])
def test_data_runtime_and_source_mutations_hold(tmp_path: Path, committed: dict, mutation: str) -> None:
    value = copy.deepcopy(committed)
    if mutation == "manifest":
        value["data"]["training_manifest"]["sha256"] = "0" * 64
    elif mutation == "codec":
        value["codec"]["configuration_hash"] = "0" * 64
    elif mutation == "lock":
        value["execution"]["lock_file_sha256"] = "0" * 64
    else:
        value["source"]["closure"][0]["sha256"] = "0" * 64
    _reject(tmp_path, value)


def test_nonzero_protected_starting_state_holds(tmp_path: Path, committed: dict) -> None:
    value = copy.deepcopy(committed)
    value["protected_starting_state"]["artifact_classifier_optimizer_steps"] = 1
    _reject(tmp_path, value)


def test_f0_explicitly_does_not_authorize_or_start_f1(committed: dict) -> None:
    assert committed["owner_authorization"]["scope"] == "G8_F_F0_V2_REPAIR_ONLY"
    assert committed["owner_authorization"]["f1_launch_authorized"] is False
    assert committed["protected_starting_state"]["f1_started"] is False
    assert committed["protected_starting_state"]["materialized_artifact_objects"] == 0
    assert committed["protected_starting_state"]["artifact_classifier_optimizer_steps"] == 0
    assert committed["protected_starting_state"]["pass_two"] == 0
    assert committed["protected_starting_state"]["test_access"] == 0


def test_f0_v1_is_byte_preserved_and_historically_authenticates(committed: dict) -> None:
    del committed
    historical = verify_f0_v1_historical()
    assert historical["authorization_id"] == V1_AUTHORIZATION_ID
    assert sha256_bytes(V1_AUTHORIZATION_PATH.read_bytes()) == V1_FILE_SHA256
    assert historical["protected_starting_state"]["f1_started"] is False
    assert historical["protected_starting_state"]["materialized_artifact_objects"] == 0


def test_f0_v2_explicitly_supersedes_v1_before_f1(committed: dict) -> None:
    assert committed["supersession"]["state"] == "superseded_before_F1"
    assert committed["supersession"]["prior_authorization"]["authorization_id"] == V1_AUTHORIZATION_ID
    assert committed["supersession"]["prior_authorization"]["file_sha256"] == V1_FILE_SHA256
    assert committed["supersession"]["prior_production_coverage"] == 0
    assert committed["protected_starting_state"]["f1_launch_authorized"] is False


def test_authorization_file_is_canonical_and_self_authenticating(committed: dict) -> None:
    raw = AUTHORIZATION_PATH.read_bytes()
    assert raw == rendered_json(json.loads(raw))
    body = dict(committed)
    identity = body.pop("authorization_id")
    assert identity == AUTHORIZATION_PREFIX + sha256_bytes(canonical_json(body))
