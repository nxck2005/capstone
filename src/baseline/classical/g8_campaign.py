"""Fail-closed contracts for the validation-only G-8 campaign.

G8_A freezes metadata and state machinery only.  This module deliberately has
no simulation, codec, dataset-decoding, classifier, training, selection, or
authorization entry point.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from config.params import REPO_ROOT

CAMPAIGN = "G-8"
CAMPAIGN_MANIFEST = REPO_ROOT / "results/baseline/g8/campaign_manifest.json"
PHASE_ORDER = tuple(f"G8_{letter}" for letter in "ABCDEFG")
PB3C_TERMINAL_SHA = "39c43e327573f33011c561c6de22bd05ff93c068"
SELECTION_POLICY_FIELDS = (
    "tie_break_order",
    "tie_equality",
    "fixed_modulation.source",
    "fixed_modulation.configured_value",
    "selection_passes",
    "selection_termination_pass",
)
PRE_DATA_FLAGS = {
    "campaign_started": False,
    "characterization_started": False,
    "validation_measurements_started": False,
    "pass_one_executed": False,
    "training_started": False,
    "pass_two_executed": False,
    "adjudication_complete": False,
    "test_split_access": 0,
    "authorization_issued": False,
}


class G8ContractError(RuntimeError):
    """The persisted campaign contract is missing, malformed, or has drifted."""


def canonical_json(value: Any) -> bytes:
    """Canonical identity bytes; presentation whitespace is never identity."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def rendered_json(value: Any) -> bytes:
    """Stable tracked-file rendering."""

    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def campaign_identifier(payload: Mapping[str, Any]) -> str:
    """Derive the stable ID from every manifest field except the ID itself."""

    basis = dict(payload)
    basis.pop("campaign_id", None)
    return f"g8-{sha256_bytes(canonical_json(basis))}"


def load_campaign_manifest(path: Path = CAMPAIGN_MANIFEST) -> dict[str, Any]:
    """Load and minimally type-check a G8_A manifest without trusting it."""

    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise G8ContractError(f"cannot read campaign manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise G8ContractError("campaign manifest is not a JSON object")
    if raw != rendered_json(payload):
        raise G8ContractError("campaign manifest is not canonical rendered JSON")
    if payload.get("schema_version") != 1:
        raise G8ContractError("unsupported campaign manifest schema_version")
    if payload.get("campaign") != CAMPAIGN:
        raise G8ContractError("campaign manifest names the wrong campaign")
    if payload.get("campaign_id") != campaign_identifier(payload):
        raise G8ContractError("campaign_id does not reproduce from manifest content")
    return payload
