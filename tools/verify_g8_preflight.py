#!/usr/bin/env python3
"""Independently verify the G8_A pre-data contract; never execute G-8."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from baseline.g8_campaign import (
    CAMPAIGN_MANIFEST,
    PB3C_TERMINAL_SHA,
    PHASE_ORDER,
    PRE_DATA_FLAGS,
    REQUIRED_BLER_IDENTITIES,
    CAMPAIGN_STATE,
    SELECTION_POLICY_FIELDS,
    G8ContractError,
    build_structural_preflight,
    campaign_identifier,
    load_campaign_manifest,
    load_campaign_state,
    load_required_bler_identities,
    sha256_bytes,
    verify_historical_contract_sources,
    verify_historical_normative_sources,
)
from config.params import REPO_ROOT, get

EXPECTED_NORMATIVE_SOURCES = ("spec/SPEC.md", "spec/params.generated.yaml")
EXPECTED_SPLIT_MANIFESTS = (
    "data/manifests/imagenette160.csv",
    "data/manifests/stl10.csv",
    "data/manifests/cifar10.csv",
)
EXPECTED_CONTRACT_SOURCES = (
    "instructions/G8.txt",
    "instructions/G8_A.txt",
    "instructions/G8_B.txt",
    "instructions/G8_C.txt",
    "instructions/G8_D.txt",
    "instructions/G8_E.txt",
    "instructions/G8_F.txt",
    "instructions/G8_G.txt",
    "src/baseline/g8_campaign.py",
    "tools/gen_g8_campaign_manifest.py",
    "tools/update_g8_campaign_state.py",
    "tools/verify_g8_preflight.py",
)


class G8PreflightError(RuntimeError):
    """An independently checked preflight invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G8PreflightError(message)


def _verify_binding(entry: Any) -> None:
    _require(isinstance(entry, dict), "file binding is not an object")
    path = entry.get("path")
    _require(isinstance(path, str) and not Path(path).is_absolute(), "invalid bound path")
    target = REPO_ROOT / path
    try:
        body = target.read_bytes()
    except OSError as exc:
        raise G8PreflightError(f"cannot read bound path {path}: {exc}") from exc
    _require(entry.get("bytes") == len(body), f"bound byte length changed: {path}")
    _require(entry.get("sha256") == sha256_bytes(body), f"bound SHA-256 changed: {path}")


def _policy_fingerprint(machinery: dict[str, Any]) -> str:
    covered: list[list[Any]] = []
    for field in SELECTION_POLICY_FIELDS:
        head, _, tail = field.partition(".")
        value = machinery.get(head)
        if tail:
            _require(isinstance(value, dict), f"policy field {field} is malformed")
            value = value.get(tail)
        covered.append([field, value])
    canonical = json.dumps(covered, separators=(",", ":"), ensure_ascii=True)
    return sha256_bytes(canonical.encode("utf-8"))


def verify_required_structure(required: dict[str, Any]) -> None:
    """Independently reject malformed, incomplete, or optimistic enumeration."""

    axes = required.get("axes") or {}
    expected_datasets = [
        {"name": name, "role": get(f"datasets.{name}.role")}
        for name in ("imagenette160", "stl10")
    ]
    _require(axes.get("datasets") == expected_datasets, "required grid dataset axis changed")
    _require(axes.get("ratios") == list(get("bandwidth.ratios")), "required grid ratio axis changed")
    _require(axes.get("source_codecs") == [get("baseline.source_codec")],
             "required grid source-codec axis changed")
    _require(
        axes.get("encode_axis_px")
        == {name: list(get(f"baseline.downsample_axis_px.{name}")) for name in ("imagenette160", "stl10")},
        "required grid encode-axis changed",
    )
    _require(axes.get("modulations") == list(get("baseline.modulations")),
             "required grid modulation axis changed")
    _require(axes.get("ldpc_rates") == list(get("baseline.ldpc_rates")),
             "required grid LDPC-rate axis changed")
    _require(axes.get("snr_grid_db") == list(get("channel.test_snr_grid_db")),
             "required grid SNR axis changed")
    candidates = required.get("structural_candidates") or []
    work_units = required.get("required_bler_work_units") or []
    candidate_ids = [row.get("candidate_id") for row in candidates]
    work_unit_ids = [row.get("work_unit_id") for row in work_units]
    _require(candidate_ids == sorted(candidate_ids), "structural candidate ordering is nondeterministic")
    _require(len(candidate_ids) == len(set(candidate_ids)), "duplicate candidate ID")
    _require(work_unit_ids == sorted(work_unit_ids), "BLER work-unit ordering is nondeterministic")
    _require(len(work_unit_ids) == len(set(work_unit_ids)), "duplicate BLER work-unit ID")
    counts = required.get("counts") or {}
    _require(counts.get("structural_candidates") == len(candidates), "candidate count is false")
    _require(counts.get("required_unique_bler_work_units") == len(work_units), "work-unit count is false")
    coverage = required.get("g2_comparison") or {}
    _require(coverage.get("coverage_complete") is False, "G-2 coverage is falsely complete")
    _require(coverage.get("complete_coverage_claim_permitted") is False,
             "preflight permits a false complete-coverage claim")
    exact = set(coverage.get("already_characterized_exact") or [])
    missing = set(coverage.get("missing_required") or [])
    mismatch = set(coverage.get("uncharacterized_identity_mismatch") or [])
    snr_missing = set(coverage.get("uncharacterized_snr_support") or [])
    _require(exact.isdisjoint(missing) and exact | missing == set(work_unit_ids),
             "G-2 exact/missing coverage partition is false")
    _require(mismatch.isdisjoint(snr_missing) and mismatch | snr_missing == missing,
             "G-2 missing-coverage categories are false")
    _require(coverage.get("interpolation_used") is False, "G-2 interpolation was used")
    _require(coverage.get("extrapolation_used") is False, "G-2 extrapolation was used")
    _require(required.get("scientific_execution_performed") is False,
             "required-BLER artifact claims scientific execution")
    _require(required.get("dataset_pixels_loaded") == 0, "preflight loaded dataset pixels")
    _require(required.get("fallback_invoked") is False, "preflight invoked fallback")


def authorization_constructions(source: str, path: str) -> list[str]:
    """Return line-qualified real constructor calls, ignoring comments/strings."""

    tree = ast.parse(source, filename=path)
    findings: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = function.id if isinstance(function, ast.Name) else (
            function.attr if isinstance(function, ast.Attribute) else None
        )
        if name == "G8Authorization":
            findings.append(f"{path}:{node.lineno}")
    return findings


def verify_no_tracked_authorization_construction() -> None:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "*.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    paths = sorted(path for path in result.stdout.decode().split("\0") if path)
    findings: list[str] = []
    for path in paths:
        target = REPO_ROOT / path
        if path.startswith("tests/") or not target.is_file():
            continue
        findings.extend(authorization_constructions(target.read_text(), path))
    _require(not findings, f"tracked non-test G8Authorization construction: {findings}")


def verify(path: Path = CAMPAIGN_MANIFEST) -> dict[str, Any]:
    try:
        payload = load_campaign_manifest(path)
    except G8ContractError as exc:
        raise G8PreflightError(str(exc)) from exc
    _require(payload.get("campaign_id") == campaign_identifier(payload), "campaign ID drift")
    _require(payload.get("phase_order") == list(PHASE_ORDER), "G-8 phase order changed")
    for name, expected in PRE_DATA_FLAGS.items():
        _require(payload.get(name) == expected, f"pre-data flag {name} is not {expected!r}")
    _require(payload.get("stage") == "preflight_contract_only", "wrong campaign stage")

    scientific_base = payload.get("scientific_base") or {}
    _require(scientific_base.get("commit_sha") == PB3C_TERMINAL_SHA, "wrong PB_3C terminal SHA")
    _require(
        scientific_base.get("source_state_mode") == "content_hashes_with_pb3c_base",
        "scientific source state is not content-addressed",
    )
    _require(scientific_base.get("future_g8a_final_commit_not_part_of_identity") is True,
             "manifest improperly depends on a future G8_A commit")

    adjudication_binding = payload.get("w4_adjudication")
    _verify_binding(adjudication_binding)
    adjudication = json.loads((REPO_ROOT / adjudication_binding["path"]).read_text())
    machinery = adjudication.get("selection_machinery") or {}
    recorded_policy = machinery.get("selection_policy_sha256")
    _require(recorded_policy == _policy_fingerprint(machinery), "W4 policy hash does not reproduce")
    policy = payload.get("selection_policy") or {}
    _require(policy.get("selection_policy_sha256") == recorded_policy, "selection policy hash changed")
    _require(policy.get("fields") == list(SELECTION_POLICY_FIELDS), "policy coverage fields changed")
    _require(policy.get("tie_break_order") == machinery.get("tie_break_order"), "tie-break order changed")
    _require(policy.get("tie_equality") == machinery.get("tie_equality"), "exact-equality rule changed")
    _require(policy.get("frozen_before_data") is True, "policy is not frozen before data")

    recorded_sources = adjudication.get("selection_sources") or []
    manifest_sources = payload.get("selection_sources") or []
    _require(len(manifest_sources) == len(recorded_sources), "selection source count changed")
    for current, recorded in zip(manifest_sources, recorded_sources, strict=True):
        _verify_binding(current)
        for field in ("path", "role", "sha256", "bytes"):
            _require(current.get(field) == recorded.get(field), f"selection source {field} changed")
        _require(current.get("w4_bound_at") == recorded.get("bound_at"), "selection bound-at changed")

    expected_groups = {
        "normative_sources": EXPECTED_NORMATIVE_SOURCES,
        "dataset_split_manifests": EXPECTED_SPLIT_MANIFESTS,
        "contract_sources": EXPECTED_CONTRACT_SOURCES,
    }
    for group, expected_paths in expected_groups.items():
        entries = payload.get(group)
        _require(isinstance(entries, list) and entries, f"{group} is empty or malformed")
        paths = [entry.get("path") for entry in entries if isinstance(entry, dict)]
        _require(paths == list(expected_paths), f"{group} path set or order changed")
        if group == "normative_sources":
            try:
                verify_historical_normative_sources(entries)
            except G8ContractError as exc:
                raise G8PreflightError(str(exc)) from exc
        elif group == "contract_sources":
            try:
                verify_historical_contract_sources(entries)
            except G8ContractError as exc:
                raise G8PreflightError(str(exc)) from exc
        else:
            for entry in entries:
                _verify_binding(entry)

    rules = payload.get("interpretation_rules") or {}
    _require(rules.get("pre_data_contract_not_authorization") is True, "contract claims authorization")
    _require(rules.get("later_phases_may_not_silently_reinterpret_earlier_artifacts") is True,
             "later-phase reinterpretation is not prohibited")
    _require(rules.get("changed_bound_scientific_policy_invalidates_campaign") is True,
             "policy drift does not invalidate campaign")

    generated = payload.get("generated_preflight_artifacts")
    _require(isinstance(generated, list) and len(generated) == 1,
             "required generated-preflight binding is missing")
    _require(generated[0].get("path") == "results/baseline/g8/required_bler_identities.json",
             "required-BLER artifact path changed")
    _verify_binding(generated[0])
    required = load_required_bler_identities(REQUIRED_BLER_IDENTITIES)
    verify_required_structure(required)
    _require(required == build_structural_preflight(), "structural enumeration does not reproduce")
    verify_no_tracked_authorization_construction()
    state = load_campaign_state(CAMPAIGN_STATE)
    state_identity = state["identity"]
    _require(state_identity["phase"] == "G8_A", "campaign state exposes a later phase")
    _require(state_identity["stage"] in ("contract_open", "preflight_complete"),
             "campaign state exposes a future stage")
    _require(state_identity["completed_work_unit_ids"] == [],
             "G8_A state claims completed scientific work")
    _require(state_identity["in_progress_work_unit_id"] is None,
             "G8_A state claims in-progress scientific work")
    _require(all(value == 0 for value in state_identity["counters"].values()),
             "G8_A campaign counters are not all zero")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=CAMPAIGN_MANIFEST)
    args = parser.parse_args()
    try:
        payload = verify(args.manifest)
    except G8PreflightError as exc:
        raise SystemExit(f"G8 preflight verification HOLD: {exc}") from exc
    print(
        "G8 preflight contract PASS: "
        f"campaign_id={payload['campaign_id']}, phases={len(payload['phase_order'])}, "
        "authorization=false, execution=false, test_split_access=0, coverage=incomplete-as-required"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
