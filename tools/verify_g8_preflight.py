#!/usr/bin/env python3
"""Independently verify the G8_A pre-data contract; never execute G-8."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from baseline.classical.g8_campaign import (
    CAMPAIGN_MANIFEST,
    PB3C_TERMINAL_SHA,
    PHASE_ORDER,
    PRE_DATA_FLAGS,
    SELECTION_POLICY_FIELDS,
    G8ContractError,
    campaign_identifier,
    load_campaign_manifest,
    sha256_bytes,
)
from config.params import REPO_ROOT

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
    "src/baseline/classical/g8_campaign.py",
    "tools/gen_g8_campaign_manifest.py",
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
        for entry in entries:
            _verify_binding(entry)

    rules = payload.get("interpretation_rules") or {}
    _require(rules.get("pre_data_contract_not_authorization") is True, "contract claims authorization")
    _require(rules.get("later_phases_may_not_silently_reinterpret_earlier_artifacts") is True,
             "later-phase reinterpretation is not prohibited")
    _require(rules.get("changed_bound_scientific_policy_invalidates_campaign") is True,
             "policy drift does not invalidate campaign")
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
        "authorization=false, execution=false, test_split_access=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
