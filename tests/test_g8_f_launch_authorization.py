"""Fail-closed checks for the separate owner-issued G8_F/F1 launch artifact."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from baseline.g8_f_f0 import AUTHORIZATION_PATH, rendered_json, verify_f0_authorization
from baseline.g8_f_materializer import canonical_json
from gen_g8_f_f1_launch_authorization import LAUNCH_PATH
from run_g8_f_f1 import F1LaunchAuthorizationError, verify_separate_f1_launch


def _rewrite_identity(value: dict) -> dict:
    body = copy.deepcopy(value)
    body.pop("launch_id", None)
    body["launch_id"] = "g8ff1launch-" + hashlib.sha256(canonical_json(body)).hexdigest()
    return body


@pytest.fixture(scope="module")
def committed() -> tuple[dict, dict]:
    f0 = verify_f0_authorization(require_zero_prefix=False)
    launch = verify_separate_f1_launch(LAUNCH_PATH, AUTHORIZATION_PATH, f0)
    return f0, launch


def test_launch_binds_active_pascal_f0_and_source(committed: tuple[dict, dict]) -> None:
    f0, launch = committed
    assert f0["execution"]["execution_profile_id"] == "confessor_pascal_cu126"
    assert f0["execution"]["device"] == "cuda:0"
    assert launch["status"] == "OWNER_AUTHORIZED_F1_LAUNCH"
    assert launch["scope"] == "G8_F_F1_ONLY"
    assert launch["f0_authorization_id"] == f0["authorization_id"]
    assert launch["f0_file_sha256"] == hashlib.sha256(AUTHORIZATION_PATH.read_bytes()).hexdigest()
    assert launch["intended_f1_source_commit"] == f0["source"]["intended_f1_source_commit"]


@pytest.mark.parametrize("field", ["f0_authorization_id", "f0_file_sha256", "intended_f1_source_commit", "scope", "owner_statement"])
def test_launch_binding_mutations_hold(tmp_path: Path, committed: tuple[dict, dict], field: str) -> None:
    f0, launch = committed
    value = copy.deepcopy(launch)
    value[field] = "foreign"
    value = _rewrite_identity(value)
    path = tmp_path / "launch.json"
    path.write_bytes(rendered_json(value))
    with pytest.raises(F1LaunchAuthorizationError):
        verify_separate_f1_launch(path, AUTHORIZATION_PATH, f0)


def test_launch_file_is_canonical_and_content_derived(committed: tuple[dict, dict]) -> None:
    _f0, launch = committed
    raw = LAUNCH_PATH.read_bytes()
    assert raw == rendered_json(json.loads(raw))
    body = dict(launch)
    launch_id = body.pop("launch_id")
    assert launch_id == "g8ff1launch-" + hashlib.sha256(canonical_json(body)).hexdigest()
