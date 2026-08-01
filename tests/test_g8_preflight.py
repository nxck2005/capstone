"""G8_A pre-data contract tests; no scientific runner is imported or called."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

import pytest

import verify_g8_preflight as verifier
from baseline.classical.g8_campaign import (
    CAMPAIGN_MANIFEST,
    campaign_identifier,
    rendered_json,
)


def _payload() -> dict[str, Any]:
    return json.loads(CAMPAIGN_MANIFEST.read_text(encoding="utf-8"))


def _mutated(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    *,
    refresh_id: bool = True,
) -> Path:
    payload = copy.deepcopy(_payload())
    mutate(payload)
    if refresh_id:
        payload["campaign_id"] = campaign_identifier(payload)
    path = tmp_path / "campaign_manifest.json"
    path.write_bytes(rendered_json(payload))
    return path


def test_committed_campaign_contract_verifies() -> None:
    payload = verifier.verify()
    assert payload["authorization_issued"] is False
    assert payload["campaign_started"] is False
    assert payload["test_split_access"] == 0


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda p: p["w4_adjudication"].__setitem__("sha256", "0" * 64), "bound SHA-256"),
        (lambda p: p["selection_policy"].__setitem__("selection_policy_sha256", "0" * 64), "policy hash"),
        (lambda p: p["selection_sources"][0].__setitem__("sha256", "0" * 64), "bound SHA-256"),
        (lambda p: p["normative_sources"][0].__setitem__("sha256", "0" * 64), "bound SHA-256"),
        (lambda p: p["dataset_split_manifests"][0].__setitem__("sha256", "0" * 64), "bound SHA-256"),
        (lambda p: p.__setitem__("phase_order", list(reversed(p["phase_order"]))), "phase order"),
        (lambda p: p.__setitem__("campaign_started", True), "campaign_started"),
        (lambda p: p.__setitem__("authorization_issued", True), "authorization_issued"),
        (lambda p: p["contract_sources"].pop(), "contract_sources"),
    ],
)
def test_manifest_mutations_fail_closed(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    path = _mutated(tmp_path, mutate)
    with pytest.raises(verifier.G8PreflightError, match=match):
        verifier.verify(path)


def test_stale_campaign_id_is_rejected_before_other_claims(tmp_path: Path) -> None:
    path = _mutated(
        tmp_path,
        lambda p: p.__setitem__("campaign_started", True),
        refresh_id=False,
    )
    with pytest.raises(verifier.G8PreflightError, match="campaign_id"):
        verifier.verify(path)


def test_noncanonical_manifest_bytes_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "campaign_manifest.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    with pytest.raises(verifier.G8PreflightError, match="canonical rendered JSON"):
        verifier.verify(path)
