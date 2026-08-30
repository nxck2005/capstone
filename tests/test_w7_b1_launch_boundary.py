"""W7-B1 detached-launch/source-authority boundary regressions."""

from __future__ import annotations

import json

import pytest

import run_w7_campaign as campaign
import verify_w7_b1 as b1
from gen_w7_source_manifest import verify as verify_historical_source
from gen_w7_test_hardening import verify_source as verify_hardening_source


def test_successor_source_manifest_uses_real_launcher_verifier_after_w7c():
    # B1's source authority is historical and remains byte-authenticated. W7-C
    # updates the generated normative parameter views additively, so the
    # post-closeout check must not pretend those historical bytes are current.
    value = campaign.verify_source_manifest(
        b1.B1_SOURCE_PATH,
        current=False,
        repo_root=campaign.REPO,
    )
    assert value["artifact_role"] == b1.B1_SOURCE_ROLE
    assert value["source_commit"] != json.loads(b1.HARDENING_SOURCE_PATH.read_bytes())["source_commit"]


def test_accepted_v2_authority_is_authenticated_as_w7a_predecessor():
    value = json.loads(b1.HARDENING_SOURCE_PATH.read_bytes())
    assert verify_hardening_source(value, current=False)["manifest_id"] == "w7testsource-1cf7ce96ec6a7134ed900ef7bb2c45bb3d292df007123d522916c9d8784c679b"


def test_historical_v1_cannot_be_used_as_current_scientific_source():
    with pytest.raises(b1.W7B1Hold, match="schema differs"):
        campaign.verify_source_manifest(
            b1.HISTORICAL_SOURCE_PATH,
            current=True,
            repo_root=campaign.REPO,
        )

    # The historical verifier also fails closed against the post-W7-C current
    # checkout; its source authority is not silently rewritten.
    historical = json.loads(b1.HISTORICAL_SOURCE_PATH.read_bytes())
    with pytest.raises(ValueError, match="W7 current source byte drift:"):
        verify_historical_source(historical, current=True)
