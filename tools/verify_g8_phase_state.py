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
    PB3C_TERMINAL_SHA,
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

G8_ARTIFACT_ROOT = (REPO / "results/baseline/g8").resolve()
BLER_TOOLING_CONTRACT_PATH = "results/baseline/g8/bler_tooling_contract.json"


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

    # The B0 gap: the named verifier checked bindings but never the manifest's
    # own scientific-base and interpretation clauses, so a later phase could
    # have rebased the campaign's identity without this verifier noticing.
    base = manifest.get("scientific_base") or {}
    _require(base.get("commit_sha") == PB3C_TERMINAL_SHA, "manifest scientific base commit changed")
    _require(
        base.get("source_state_mode") == "content_hashes_with_pb3c_base",
        "manifest scientific source-state mode changed",
    )
    _require(
        base.get("future_g8a_final_commit_not_part_of_identity") is True,
        "manifest improperly depends on a future G8_A commit",
    )
    rules = manifest.get("interpretation_rules") or {}
    _require(rules.get("pre_data_contract_not_authorization") is True,
             "manifest contract claims authorization")
    _require(rules.get("later_phases_may_not_silently_reinterpret_earlier_artifacts") is True,
             "later-phase reinterpretation is no longer prohibited")
    _require(rules.get("changed_bound_scientific_policy_invalidates_campaign") is True,
             "policy drift no longer invalidates the campaign")

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
    _require(
        policy.get("changing_bound_policy_after_campaign_start_invalidates_campaign") is True,
        "the selection-policy invalidation rule was weakened",
    )

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


def _verify_produced_artifacts(
    identity: dict[str, Any],
    require_artifacts: tuple[str, ...],
) -> None:
    """Require every G8_A base binding and allow later phases to add more.

    Comparing the whole list to the G8_A opening list was correct at B0 but
    would forbid every legitimate later artifact. The base bindings stay
    mandatory and byte-identical; additions are permitted and are already
    hash-validated by the strict campaign-state loader.
    """

    artifacts = identity["produced_artifacts"]
    paths = [entry["path"] for entry in artifacts]
    _require(len(paths) == len(set(paths)), "produced artifact paths are duplicated")
    _require(paths == sorted(paths), "produced artifact bindings are not canonically sorted")

    by_path = {entry["path"]: entry for entry in artifacts}
    base = initial_campaign_state(stage="preflight_complete")["identity"]["produced_artifacts"]
    for expected in base:
        current = by_path.get(expected["path"])
        _require(current is not None,
                 f"base G8_A produced-artifact binding is missing: {expected['path']}")
        _require(current == expected,
                 f"base G8_A produced-artifact binding changed: {expected['path']}")
    for wanted in require_artifacts:
        _require(wanted in by_path, f"required produced-artifact binding is absent: {wanted}")


def _verify_produced_artifact_paths(identity: dict[str, Any]) -> None:
    """Reject traversal, outside-root and normalized-alias artifact paths."""

    seen: dict[Path, str] = {}
    raw_seen: set[str] = set()
    for entry in identity["produced_artifacts"]:
        raw = entry.get("path")
        _require(isinstance(raw, str) and raw, "produced artifact path is not a string")
        _require(raw not in raw_seen, f"produced artifact paths are duplicated: {raw}")
        raw_seen.add(raw)
        relative = Path(raw)
        _require(not relative.is_absolute(), f"produced artifact path is absolute: {raw}")
        _require(".." not in relative.parts, f"produced artifact path contains '..': {raw}")
        resolved = (REPO / relative).resolve(strict=False)
        try:
            resolved.relative_to(G8_ARTIFACT_ROOT)
        except ValueError:
            raise G8PhaseStateError(
                f"produced artifact path resolves outside the G8 root: {raw}"
            ) from None
        prior = seen.get(resolved)
        _require(
            prior is None,
            f"produced artifact path aliases {prior!r} after normalization: {raw}",
        )
        seen[resolved] = raw


def _verify_required_tooling_seed_identity(
    state: dict[str, Any],
    require_artifacts: tuple[str, ...],
) -> None:
    """Bind the live seed identity to the required B1C contract artifact."""

    if BLER_TOOLING_CONTRACT_PATH not in require_artifacts:
        return
    try:
        payload = json.loads((REPO / BLER_TOOLING_CONTRACT_PATH).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise G8PhaseStateError(f"cannot read required BLER tooling contract: {exc}") from exc
    seed = payload.get("seed")
    _require(isinstance(seed, dict), "required BLER tooling contract seed section is missing")
    _require(
        state["identity"]["seed_derivation_identity"] == seed.get("derivation_identity"),
        "campaign state seed derivation identity does not match the tooling contract",
    )


def _precheck_state_artifact_paths(path: Path) -> None:
    """Run the phase-owned path boundary before the generic state loader.

    The generic loader deliberately validates bytes and schema first; this
    precheck ensures the phase verifier itself reports unsafe paths before a
    malformed absolute entry can be classified only as a generic binding
    error.
    """

    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError):
        return
    if isinstance(payload, dict) and isinstance(payload.get("identity"), dict):
        artifacts = payload["identity"].get("produced_artifacts")
        if isinstance(artifacts, list):
            _verify_produced_artifact_paths(payload["identity"])


def verify(
    *,
    phase: str,
    stage: str,
    require_zero_science: bool = False,
    state_path: Path | None = None,
    require_artifacts: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Verify one requested current phase and return the validated state."""

    if phase not in PHASE_ORDER:
        raise G8PhaseStateError(f"unknown requested phase: {phase!r}")
    if stage not in STATE_STAGES[phase]:
        raise G8PhaseStateError(f"unknown requested stage {stage!r} for phase {phase!r}")

    manifest, _required = _verify_manifest_and_required_artifact()
    path = CAMPAIGN_STATE if state_path is None else Path(state_path)
    _precheck_state_artifact_paths(path)
    try:
        state = load_campaign_state(path)
    except G8ContractError as exc:
        raise G8PhaseStateError(str(exc)) from exc
    identity = state["identity"]
    _require(identity["campaign_id"] == manifest["campaign_id"],
             "campaign state campaign ID does not match the manifest")
    _require(identity["campaign_manifest_sha256"] == sha256_file(CAMPAIGN_MANIFEST),
             "campaign state manifest hash does not match the manifest")

    _verify_produced_artifact_paths(identity)
    _verify_produced_artifacts(identity, require_artifacts)
    _verify_required_tooling_seed_identity(state, require_artifacts)

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
    parser.add_argument(
        "--require-artifact",
        action="append",
        default=[],
        metavar="PATH",
        help="repository-relative produced artifact that must be bound; repeatable",
    )
    args = parser.parse_args(argv)
    try:
        state = verify(
            phase=args.phase,
            stage=args.stage,
            require_zero_science=args.require_zero_science,
            require_artifacts=tuple(args.require_artifact),
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
        f"produced_artifacts={len(identity['produced_artifacts'])}, "
        f"zero_science={'true' if zero_status else 'false'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
