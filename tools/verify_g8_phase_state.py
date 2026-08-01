#!/usr/bin/env python3
"""Verify the current G-8 phase state without reopening the G8_A assertion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import verify_g8_preflight as preflight  # noqa: E402
from baseline.g8_campaign import (  # noqa: E402
    CAMPAIGN,
    CAMPAIGN_MANIFEST,
    CAMPAIGN_STATE,
    PHASE_ORDER,
    PRE_DATA_FLAGS,
    REQUIRED_BLER_IDENTITIES,
    SELECTION_POLICY_FIELDS,
    STATE_STAGES,
    G8ContractError,
    initial_campaign_state,
    load_campaign_manifest,
    load_campaign_state,
    load_required_bler_identities,
    sha256_file,
)


class G8PhaseStateError(RuntimeError):
    """The current phase state or its immutable G8_A inputs is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G8PhaseStateError(message)


def _verify_manifest_and_required_artifact() -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        manifest = load_campaign_manifest(CAMPAIGN_MANIFEST)
        required = load_required_bler_identities(REQUIRED_BLER_IDENTITIES)
    except G8ContractError as exc:
        raise G8PhaseStateError(str(exc)) from exc

    _require(manifest.get("campaign") == CAMPAIGN, "campaign manifest names the wrong campaign")
    _require(manifest.get("phase_order") == list(PHASE_ORDER), "campaign phase order changed")
    for name, expected in PRE_DATA_FLAGS.items():
        _require(manifest.get(name) == expected, f"manifest pre-data flag {name} changed")
    _require(manifest.get("stage") == "preflight_contract_only", "campaign manifest stage changed")

    for group, expected_paths in {
        "normative_sources": preflight.EXPECTED_NORMATIVE_SOURCES,
        "dataset_split_manifests": preflight.EXPECTED_SPLIT_MANIFESTS,
        "contract_sources": preflight.EXPECTED_CONTRACT_SOURCES,
    }.items():
        entries = manifest.get(group)
        _require(isinstance(entries, list), f"manifest {group} is malformed")
        _require(
            [entry.get("path") for entry in entries if isinstance(entry, dict)]
            == list(expected_paths),
            f"manifest {group} path set changed",
        )
        for entry in entries:
            try:
                preflight._verify_binding(entry)
            except preflight.G8PreflightError as exc:
                raise G8PhaseStateError(str(exc)) from exc

    adjudication_binding = manifest.get("w4_adjudication")
    _require(isinstance(adjudication_binding, dict), "W4 adjudication binding is malformed")
    try:
        preflight._verify_binding(adjudication_binding)
        adjudication = json.loads(
            (REPO / adjudication_binding["path"]).read_text(encoding="utf-8")
        )
    except preflight.G8PreflightError as exc:
        raise G8PhaseStateError(str(exc)) from exc
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise G8PhaseStateError(f"cannot read W4 adjudication binding: {exc}") from exc
    machinery = adjudication.get("selection_machinery") or {}
    _require(
        machinery.get("selection_policy_sha256")
        == preflight._policy_fingerprint(machinery),
        "W4 selection policy hash does not reproduce",
    )
    policy = manifest.get("selection_policy") or {}
    _require(isinstance(policy, dict), "selection policy binding is malformed")
    for field in ("selection_policy_sha256", "fields", "tie_break_order", "tie_equality"):
        _require(
            policy.get(field)
            == (
                machinery.get(field)
                if field != "fields"
                else list(SELECTION_POLICY_FIELDS)
            ),
            f"selection policy field {field} changed",
        )
    _require(policy.get("frozen_before_data") is True, "selection policy is not frozen before data")

    sources = manifest.get("selection_sources")
    recorded_sources = adjudication.get("selection_sources") or []
    _require(isinstance(sources, list) and len(sources) == len(recorded_sources),
             "selection-source bindings changed")
    for current, recorded in zip(sources, recorded_sources, strict=True):
        _require(isinstance(recorded, dict), "recorded selection source is malformed")
        try:
            preflight._verify_binding(current)
        except preflight.G8PreflightError as exc:
            raise G8PhaseStateError(str(exc)) from exc
        for field in ("path", "role", "sha256", "bytes"):
            _require(current.get(field) == recorded.get(field),
                     f"selection source {field} changed")
        _require(current.get("w4_bound_at") == recorded.get("bound_at"),
                 "selection source bound-at changed")

    generated = manifest.get("generated_preflight_artifacts")
    _require(isinstance(generated, list) and len(generated) == 1,
             "required-BLER artifact binding is missing")
    _require(isinstance(generated[0], dict), "required-BLER artifact binding is malformed")
    _require(generated[0].get("path") == str(REQUIRED_BLER_IDENTITIES.relative_to(REPO)),
             "required-BLER artifact path changed")
    try:
        preflight._verify_binding(generated[0])
        preflight.verify_required_structure(required)
        preflight.verify_no_tracked_authorization_construction()
    except (preflight.G8PreflightError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise G8PhaseStateError(str(exc)) from exc
    return manifest, required


def verify(
    *,
    phase: str,
    stage: str,
    require_zero_science: bool = False,
    state_path: Path | None = None,
) -> dict[str, Any]:
    """Verify one requested current phase and return the validated state."""

    if phase not in PHASE_ORDER:
        raise G8PhaseStateError(f"unknown requested phase: {phase!r}")
    if stage not in STATE_STAGES[phase]:
        raise G8PhaseStateError(f"unknown requested stage {stage!r} for phase {phase!r}")

    manifest, _required = _verify_manifest_and_required_artifact()
    path = CAMPAIGN_STATE if state_path is None else Path(state_path)
    try:
        state = load_campaign_state(path)
    except G8ContractError as exc:
        raise G8PhaseStateError(str(exc)) from exc
    identity = state["identity"]
    _require(identity["campaign_id"] == manifest["campaign_id"],
             "campaign state campaign ID does not match the manifest")
    _require(identity["campaign_manifest_sha256"] == sha256_file(CAMPAIGN_MANIFEST),
             "campaign state manifest hash does not match the manifest")

    expected_opening = initial_campaign_state(stage="preflight_complete")
    _require(
        identity["produced_artifacts"] == expected_opening["identity"]["produced_artifacts"],
        "campaign produced-artifact bindings changed",
    )

    if require_zero_science:
        _require(identity["completed_work_unit_ids"] == [],
                 "campaign state contains completed scientific work units")
        _require(identity["in_progress_work_unit_id"] is None,
                 "campaign state contains an in-progress scientific work unit")
        counters = identity["counters"]
        _require(all(value == 0 for value in counters.values()),
                 "campaign state has nonzero scientific counters")

    _require(identity["phase"] == phase,
             f"current phase is {identity['phase']!r}, expected {phase!r}")
    _require(identity["stage"] == stage,
             f"current stage is {identity['stage']!r}, expected {stage!r}")
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--require-zero-science", action="store_true")
    args = parser.parse_args(argv)
    try:
        state = verify(
            phase=args.phase,
            stage=args.stage,
            require_zero_science=args.require_zero_science,
        )
    except G8PhaseStateError as exc:
        raise SystemExit(f"G8 phase-state verification HOLD: {exc}") from exc
    identity = state["identity"]
    counters = identity["counters"]
    zero_status = all(value == 0 for value in counters.values()) and not identity[
        "completed_work_unit_ids"
    ] and identity["in_progress_work_unit_id"] is None
    print(
        "G8 phase-state PASS: "
        f"campaign_id={identity['campaign_id']}, "
        f"phase={identity['phase']}, stage={identity['stage']}, "
        f"manifest_sha256={identity['campaign_manifest_sha256']}, "
        f"zero_science={'true' if zero_status else 'false'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
