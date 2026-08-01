#!/usr/bin/env python3
"""Register one produced G-8 artifact in the live campaign state.

Narrow by design: it appends a hash binding and updates the restart command.
It never claims a completed or in-progress scientific work unit, never touches a
counter, and never advances the phase or stage.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from baseline.g8_campaign import (  # noqa: E402
    CAMPAIGN_STATE,
    COUNTER_FIELDS,
    G8ContractError,
    load_campaign_state,
    sha256_bytes,
    validate_state_transition,
    write_campaign_state_atomically,
)
from config.params import REPO_ROOT  # noqa: E402

ARTIFACT_ROOT = "results/baseline/g8/"
REQUIRED_PHASE = "G8_B"
REQUIRED_STAGE = "tooling_open"


def register(
    relative_path: str,
    restart_command: str,
    *,
    state_path: Path | None = None,
) -> str:
    """Bind one existing artifact and return the resulting state digest."""

    path = CAMPAIGN_STATE if state_path is None else Path(state_path)
    previous = load_campaign_state(path)
    identity = previous["identity"]

    if identity["phase"] != REQUIRED_PHASE or identity["stage"] != REQUIRED_STAGE:
        raise G8ContractError(
            f"artifact registration requires {REQUIRED_PHASE}/{REQUIRED_STAGE}; "
            f"current state is {identity['phase']}/{identity['stage']}"
        )
    if identity["completed_work_unit_ids"]:
        raise G8ContractError("state claims completed scientific work units")
    if identity["in_progress_work_unit_id"] is not None:
        raise G8ContractError("state claims an in-progress scientific work unit")
    if any(identity["counters"][name] != 0 for name in COUNTER_FIELDS):
        raise G8ContractError("state has nonzero scientific counters")

    if not isinstance(restart_command, str) or not restart_command.strip():
        raise G8ContractError("restart command must be nonblank")
    if Path(relative_path).is_absolute():
        raise G8ContractError("artifact path must be repository-relative")
    normalized = Path(relative_path).as_posix()
    if not normalized.startswith(ARTIFACT_ROOT) or ".." in Path(normalized).parts:
        raise G8ContractError(f"artifact path must live under {ARTIFACT_ROOT}")
    target = REPO_ROOT / normalized
    try:
        body = target.read_bytes()
    except OSError as exc:
        raise G8ContractError(f"cannot read artifact {normalized}: {exc}") from exc

    binding = {"path": normalized, "sha256": sha256_bytes(body), "bytes": len(body)}
    current = copy.deepcopy(previous)
    artifacts = current["identity"]["produced_artifacts"]
    for existing in artifacts:
        if existing["path"] != normalized:
            continue
        if existing != binding:
            raise G8ContractError(f"conflicting existing binding for {normalized}")
        break
    else:
        artifacts.append(binding)
    artifacts.sort(key=lambda entry: entry["path"])
    current["identity"]["restart_command"] = restart_command

    for entry in previous["identity"]["produced_artifacts"]:
        if entry not in artifacts:
            raise G8ContractError(f"registration dropped an existing binding: {entry['path']}")
    validate_state_transition(previous, current)
    return write_campaign_state_atomically(path, current)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--restart-command", required=True)
    args = parser.parse_args(argv)
    try:
        digest = register(args.path, args.restart_command)
        state = load_campaign_state(CAMPAIGN_STATE)
    except G8ContractError as exc:
        raise SystemExit(f"G8 artifact registration HOLD: {exc}") from exc
    identity = state["identity"]
    print(
        "G8 artifact registration PASS: "
        f"path={args.path} "
        f"phase={identity['phase']}/{identity['stage']} "
        f"produced_artifacts={len(identity['produced_artifacts'])} "
        f"completed_work_units={len(identity['completed_work_unit_ids'])} "
        f"in_progress={identity['in_progress_work_unit_id']} "
        f"state_sha256={digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
