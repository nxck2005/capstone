"""Fail-closed W7-B1 source and execution-authorization regressions."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import run_w7_campaign as campaign
import verify_w7_b1 as b1
from gen_w7_source_manifest import verify as verify_historical_source
from training.deterministic_core import canonical_sha256



def _resigned(value: dict, mutate) -> dict:
    result = copy.deepcopy(value)
    mutate(result)
    body = dict(result)
    body.pop("authorization_id", None)
    result["authorization_id"] = "w7auth-" + canonical_sha256(body)
    return result


def _authorization() -> dict:
    return json.loads(b1.AUTHORIZATION_PATH.read_bytes())


def test_successor_source_manifest_is_accepted_by_real_launcher_verification_after_w7c():
    # B1's source authority remains historical and immutable. W7-C changes
    # generated normative views additively, so authenticate this predecessor
    # manifest without asserting that its old bytes are still current.
    value = campaign.verify_source_manifest(b1.B1_SOURCE_PATH, current=False, repo_root=campaign.REPO)
    assert value["artifact_role"] == b1.B1_SOURCE_ROLE
    assert value["source_commit"] != json.loads(b1.HARDENING_SOURCE_PATH.read_bytes())["source_commit"]


def test_historical_v1_is_rejected_as_current_scientific_source():
    with pytest.raises(b1.W7B1Hold, match="schema differs"):
        campaign.verify_source_manifest(b1.HISTORICAL_SOURCE_PATH, current=True, repo_root=campaign.REPO)

    # The historical verifier also fails closed against the post-W7-C current
    # checkout; its source authority is not silently rewritten.
    historical = json.loads(b1.HISTORICAL_SOURCE_PATH.read_bytes())
    with pytest.raises(ValueError, match="W7 current source byte drift:"):
        verify_historical_source(historical, current=True)


def test_execution_authorization_binds_current_source_and_hardening_authority():
    value = b1.verify_authorization_path(b1.AUTHORIZATION_PATH, verify_source=True)
    assert value["w7_a_completion_id"] == "w7acompletion-e623063c65348e23833fcec31588ef04ac43f793eeb2be0272af158071b7ba17"
    assert value["w7_test_hardening_completion_id"] == "w7testhardening-a7011b78b327184bd08bd58c6e85cd04253bc583e617a376f74392f467effab3"


@pytest.mark.parametrize(
    "label,mutation",
    [
        ("source", lambda value: value.__setitem__("source_commit", "f" * 40)),
        ("hardening", lambda value: value.__setitem__("w7_test_hardening_completion_id", "f" * 64)),
        ("gpu", lambda value: value.__setitem__("gpu_uuid", "GPU-foreign")),
        ("image", lambda value: value.__setitem__("execution_image_family", "foreign-image")),
        ("lambda", lambda value: value.__setitem__("lambda_grid", [0.0])),
        ("seed", lambda value: value.__setitem__("train_seed", 99)),
        ("freeze", lambda value: value.__setitem__("profile_freeze_id", "w7profilefreeze-foreign")),
        ("test-seal", lambda value: value.__setitem__("test_access", "OPEN")),
    ],
)
def test_resigned_authorization_inner_mutations_hold(tmp_path: Path, label, mutation):
    value = _resigned(_authorization(), mutation)
    path = tmp_path / f"{label}.json"
    path.write_bytes(b1.canonical_bytes(value))
    with pytest.raises(b1.W7B1Hold):
        b1.verify_authorization_path(path, verify_source=True)


def test_unknown_authorization_schema_and_role_hold(tmp_path: Path):
    value = _authorization()
    value["unexpected"] = True
    path = tmp_path / "unknown.json"
    path.write_bytes(b1.canonical_bytes(value))
    with pytest.raises(b1.W7B1Hold, match="schema differs"):
        b1.verify_authorization_path(path)

    value = _resigned(_authorization(), lambda item: item.__setitem__("authorization_role", "FOREIGN"))
    path.write_bytes(b1.canonical_bytes(value))
    with pytest.raises(b1.W7B1Hold, match="role/status"):
        b1.verify_authorization_path(path)


def test_completion_binds_authorization_and_non_scientific_smoke():
    value = b1.verify_completion(json.loads(b1.COMPLETION_PATH.read_bytes()))
    assert value["scientific_pilots_run"] == 0
    assert value["g4_adjudications"] == 0
    assert value["lambda_selected"] is False
    assert value["w8"] == "UNOPENED"
    assert value["test"] == "SEALED"


def test_wrong_head_and_dirty_checkout_hold(monkeypatch):
    with pytest.raises(b1.W7B1Hold, match="HEAD differs"):
        b1.verify_scientific_checkout("f" * 40, repo_root=campaign.REPO)

    actual_head = b1._git(campaign.REPO, "rev-parse", "HEAD")
    original = b1._git

    def fake_git(_root: Path, *args: str) -> str:
        if args[:2] == ("rev-parse", "HEAD"):
            return actual_head
        if args[:2] == ("status", "--porcelain"):
            return " M scientific.py"
        return original(_root, *args)

    monkeypatch.setattr(b1, "_git", fake_git)
    with pytest.raises(b1.W7B1Hold, match="dirty"):
        b1.verify_scientific_checkout(actual_head, repo_root=campaign.REPO)


def test_missing_authorization_holds_before_any_candidate_or_training(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(campaign, "W7Trainer", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("verification trained")))
    with pytest.raises(RuntimeError, match="absent"):
        campaign._load_authorization(tmp_path / "missing-authorization.json")


def test_authorization_cannot_implicitly_open_g4_w8_or_test(tmp_path: Path):
    for field, value in (("g4_adjudication", "AUTHORIZED"), ("w8", "AUTHORIZED"), ("test_access", "OPEN")):
        mutation = lambda item, field=field, value=value: item.__setitem__(field, value)
        path = tmp_path / f"{field}.json"
        path.write_bytes(b1.canonical_bytes(_resigned(_authorization(), mutation)))
        with pytest.raises(b1.W7B1Hold):
            b1.verify_authorization_path(path, verify_source=True)


def test_verification_only_paths_do_not_construct_a_trainer(monkeypatch):
    monkeypatch.setattr(campaign, "W7Trainer", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("verification constructed trainer")))
    assert b1.verify_source_path(b1.B1_SOURCE_PATH, current=False)["artifact_role"] == b1.B1_SOURCE_ROLE
    assert b1.verify_authorization_path(b1.AUTHORIZATION_PATH, verify_source=True)["status"] == "AUTHORIZED"
