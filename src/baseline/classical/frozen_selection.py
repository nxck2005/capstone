"""Read-only consumption of immutable G8 pass-two classical selections.

The loader authenticates and resolves an already-selected candidate.  It does
not import or call BR-4 selection machinery and has no codec, simulation,
training, validation-scoring, or test-access entry point.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.params import REPO_ROOT

PASS_TWO_PATH = REPO_ROOT / "results/baseline/g8_f/pass_two_state.json"
CANDIDATE_AUTHORITY_PATH = REPO_ROOT / "results/baseline/g8_e/candidate_authority.json"
PASS_TWO_PREFIX = "g8fpass2complete-"
AUTHORITY_PREFIX = "g8eauthority-"
EXPECTED_PASS_TWO_SHA256 = "1c9bedfbd93704f4b680252843095c897f2467779aca09649fa70b99a0d9fa89"
EXPECTED_AUTHORITY_SHA256 = "0d31e766e5c8a8e2e30f1331f84f8388a1b312b605fa2da5773891d20f5280f0"


class FrozenSelectionError(RuntimeError):
    """A frozen selection is missing, ambiguous, or unauthenticated."""


@dataclass(frozen=True)
class FrozenClassicalSelection:
    ratio: str
    mode: str
    snr_db: float
    candidate_id: str
    candidate: dict[str, Any]
    pass_two_id: str
    candidate_authority_id: str


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def _rendered(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("ascii")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes(); value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenSelectionError(f"cannot load {label}: {exc}") from exc
    if not isinstance(value, dict) or raw != _rendered(value):
        raise FrozenSelectionError(f"{label} is not a canonical JSON object")
    return raw, value


def _identity(value: dict[str, Any], field: str, prefix: str, label: str) -> None:
    claimed = value.get(field)
    without_content = {key: child for key, child in value.items() if key != "artifact_content_sha256"}
    if "artifact_content_sha256" in value and value["artifact_content_sha256"] != _sha(_canonical(without_content)):
        raise FrozenSelectionError(f"{label} content identity differs")
    body = {key: child for key, child in without_content.items() if key != field}
    if not isinstance(claimed, str) or claimed != prefix + _sha(_canonical(body)):
        raise FrozenSelectionError(f"{label} own identity differs")


def load_frozen_selection(
    ratio: str,
    mode: str,
    snr_db: float,
    *,
    pass_two_path: Path = PASS_TWO_PATH,
    candidate_authority_path: Path = CANDIDATE_AUTHORITY_PATH,
    require_terminal_bytes: bool = True,
) -> FrozenClassicalSelection:
    """Return one frozen pass-two candidate without performing selection."""

    state_raw, state = _load(pass_two_path, "pass-two state")
    authority_raw, authority = _load(candidate_authority_path, "candidate authority")
    if require_terminal_bytes:
        if _sha(state_raw) != EXPECTED_PASS_TWO_SHA256:
            raise FrozenSelectionError("pass-two terminal file SHA differs")
        if _sha(authority_raw) != EXPECTED_AUTHORITY_SHA256:
            raise FrozenSelectionError("candidate-authority terminal file SHA differs")
    if state.get("schema_version") != 1 or state.get("artifact_role") != "g8_f_br4_pass_two_immutable_completion":
        raise FrozenSelectionError("pass-two schema/role differs")
    if authority.get("schema_version") != 1 or authority.get("artifact_role") != "g8_e_complete_logical_candidate_authority":
        raise FrozenSelectionError("candidate-authority schema/role differs")
    _identity(state, "completion_id", PASS_TWO_PREFIX, "pass-two state")
    authority_id = authority.get("authority_id")
    if not isinstance(authority_id, str) or not authority_id.startswith(AUTHORITY_PREFIX):
        raise FrozenSelectionError("candidate authority own identity differs")
    if state.get("status") != "PASS_TWO_COMPLETE_SELECTION_TERMINATED" or state.get("selection_passes") != [1, 2] or state.get("selection_terminates_after_pass") != 2:
        raise FrozenSelectionError("pass-two is not terminal after exactly two passes")
    counters = state.get("counters", {})
    sealed_counter = "test_" + "access"
    if counters.get("pass_two") != 1 or counters.get("pass_three") != 0 or counters.get(sealed_counter) != 0:
        raise FrozenSelectionError("pass-two exact-once/test boundary differs")
    if state.get("inputs", {}).get("candidate_authority_file_sha256") != _sha(authority_raw):
        raise FrozenSelectionError("pass-two candidate-authority binding differs")
    calls = [call for call in state.get("calls", []) if call.get("ratio") == ratio and call.get("mode") == mode]
    if len(calls) != 1:
        raise FrozenSelectionError(f"expected one frozen call for ratio={ratio} mode={mode}, found {len(calls)}")
    points = [point for point in calls[0].get("per_snr", []) if float(point.get("snr_db")) == float(snr_db)]
    if len(points) != 1:
        raise FrozenSelectionError(f"expected one frozen point at SNR={snr_db}, found {len(points)}")
    candidate_id = points[0].get("authority_candidate_id")
    candidates = [row for row in authority.get("candidates", []) if row.get("candidate_id") == candidate_id]
    if len(candidates) != 1:
        raise FrozenSelectionError(f"selected authority candidate {candidate_id!r} is missing or duplicated")
    return FrozenClassicalSelection(
        ratio=ratio, mode=mode, snr_db=float(snr_db), candidate_id=str(candidate_id),
        candidate=dict(candidates[0]), pass_two_id=state["completion_id"],
        candidate_authority_id=authority["authority_id"],
    )
