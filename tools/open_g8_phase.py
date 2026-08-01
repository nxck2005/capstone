#!/usr/bin/env python3
"""Open exactly one adjacent G-8 phase without claiming scientific work."""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from baseline.g8_campaign import (  # noqa: E402
    CAMPAIGN_STATE,
    COUNTER_FIELDS,
    PHASE_ORDER,
    STATE_STAGES,
    G8ContractError,
    load_campaign_state,
    validate_state_transition,
    write_campaign_state_atomically,
)

# Keep the production-path cleanliness check anchored to the tracked path even
# when tests inject a temporary state path.
_TRACKED_CAMPAIGN_STATE = CAMPAIGN_STATE
_PRESERVED_IDENTITY_FIELDS = (
    "campaign_id",
    "campaign_manifest_sha256",
    "completed_work_unit_ids",
    "in_progress_work_unit_id",
    "produced_artifacts",
    "seed_derivation_identity",
    "counters",
)


def _git_state_file_is_clean(path: Path) -> bool:
    """Return whether the tracked campaign state has no staged or unstaged diff."""

    if path.resolve() != _TRACKED_CAMPAIGN_STATE.resolve():
        return True
    relative = path.resolve().relative_to(REPO.resolve()).as_posix()
    result = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", relative],
        cwd=REPO,
        check=False,
    )
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    raise G8ContractError("could not determine whether campaign_state.json is clean")


def _require_clean_state(path: Path) -> None:
    if not _git_state_file_is_clean(path):
        raise G8ContractError("campaign state is dirty; commit or discard no state here")


def _require_zero_science(identity: dict[str, object]) -> None:
    if identity["completed_work_unit_ids"] != []:
        raise G8ContractError("source state claims completed scientific work units")
    if identity["in_progress_work_unit_id"] is not None:
        raise G8ContractError("source state claims an in-progress scientific work unit")
    counters = identity["counters"]
    if not isinstance(counters, dict) or any(counters[name] != 0 for name in COUNTER_FIELDS):
        raise G8ContractError("source state has nonzero scientific counters")


def open_phase(
    phase: str,
    restart_command: str,
    *,
    state_path: Path | None = None,
) -> str:
    """Open one explicitly requested adjacent phase and return its state digest."""

    path = CAMPAIGN_STATE if state_path is None else Path(state_path)
    _require_clean_state(path)
    previous = load_campaign_state(path)
    previous_identity = previous["identity"]

    if previous_identity["phase"] != "G8_A":
        raise G8ContractError(
            "G8_B can open only from source phase G8_A; "
            f"current phase is {previous_identity['phase']}"
        )
    if previous_identity["stage"] != "preflight_complete":
        raise G8ContractError(
            "G8_B can open only from source stage G8_A/preflight_complete; "
            f"current stage is {previous_identity['stage']}"
        )
    _require_zero_science(previous_identity)

    if phase not in PHASE_ORDER:
        raise G8ContractError(f"unknown target G-8 phase: {phase!r}")
    expected_phase = PHASE_ORDER[PHASE_ORDER.index(previous_identity["phase"]) + 1]
    if phase != expected_phase:
        raise G8ContractError(
            f"target phase {phase!r} is not the exact next phase {expected_phase!r}"
        )
    if phase not in STATE_STAGES or not STATE_STAGES[phase]:
        raise G8ContractError(f"target phase {phase!r} has no opening stage")
    if not isinstance(restart_command, str) or not restart_command.strip():
        raise G8ContractError("restart command must be nonblank")

    current = copy.deepcopy(previous)
    current_identity = current["identity"]
    current_identity["phase"] = phase
    current_identity["stage"] = STATE_STAGES[phase][0]
    current_identity["restart_command"] = restart_command

    for field in _PRESERVED_IDENTITY_FIELDS:
        if current_identity[field] != previous_identity[field]:
            raise G8ContractError(f"phase opening changed preserved identity field {field}")
    if current["metadata"] != previous["metadata"]:
        raise G8ContractError("phase opening changed campaign metadata")

    # This is deliberately called even though the checks above are narrower:
    # the frozen state machine remains the authority for boundary transitions.
    validate_state_transition(previous, current)
    return write_campaign_state_atomically(path, current)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)
    parser.add_argument("--restart-command", required=True)
    args = parser.parse_args(argv)
    try:
        path = CAMPAIGN_STATE
        previous = load_campaign_state(path)
        digest = open_phase(args.phase, args.restart_command, state_path=path)
        current = load_campaign_state(path)
    except G8ContractError as exc:
        raise SystemExit(f"G8 phase opening HOLD: {exc}") from exc
    print(
        "G8 phase opening PASS: "
        f"previous={previous['identity']['phase']}/{previous['identity']['stage']} "
        f"resulting={current['identity']['phase']}/{current['identity']['stage']} "
        f"state_sha256={digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
