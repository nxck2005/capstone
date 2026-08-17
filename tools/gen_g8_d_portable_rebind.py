#!/usr/bin/env python3
"""Record the additive, non-scientific G8_D source-binding rebind."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline import g8_d  # noqa: E402


OUT = REPO / "results/baseline/g8_d/portable_rebind_provenance.json"
OLD_CONTRACT = REPO / "results/baseline/g8_d/measurement_contract_pre_portable_repair.json"
CURRENT_CONTRACT = REPO / "results/baseline/g8_d/measurement_contract.json"
OLD_HANDOFF = REPO / "results/baseline/g8_d/d7_handoff_pre_portable_repair.json"
CURRENT_HANDOFF = REPO / "results/baseline/g8_d/d7_handoff.json"


def _read(path: Path) -> tuple[dict[str, object], str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or raw != g8_d.rendered_json(value):
        raise g8_d.G8DContractError(f"{path} is not canonical rendered JSON")
    return value, g8_d.sha256_bytes(raw)


def build(*, repair_commit: str, repo_root: Path = REPO) -> dict[str, object]:
    old_contract, old_contract_sha = _read(OLD_CONTRACT)
    current_contract, current_contract_sha = _read(CURRENT_CONTRACT)
    old_handoff, old_handoff_sha = _read(OLD_HANDOFF)
    current_handoff, current_handoff_sha = _read(CURRENT_HANDOFF)
    old_sources = {item["path"]: item for item in old_contract["source_bindings"]}
    current_sources = {item["path"]: item for item in current_contract["source_bindings"]}
    changed_sources = [
        {
            "path": path,
            "role": current_sources[path]["role"],
            "old_sha256": old_sources[path]["sha256"],
            "current_sha256": current_sources[path]["sha256"],
        }
        for path in sorted(set(old_sources) | set(current_sources))
        if old_sources.get(path, {}).get("sha256") != current_sources.get(path, {}).get("sha256")
    ]
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_role": "g8_d_portable_loader_rebind_provenance",
        "phase": "G8_D",
        "checkpoint": "D7-CORRECTIVE",
        "provenance_id": None,
        "repair_commit": repair_commit,
        "classification": "non_scientific_contract_and_provenance_rebind",
        "reason": "G8_C frozen successor loader now authenticates portable scientific runtime evidence; its legacy tree digest remains historical only",
        "historical_contract": {
            "path": str(OLD_CONTRACT.relative_to(repo_root)),
            "contract_id": old_contract["contract_id"],
            "sha256": old_contract_sha,
            "preserved": True,
        },
        "current_contract": {
            "path": str(CURRENT_CONTRACT.relative_to(repo_root)),
            "contract_id": current_contract["contract_id"],
            "sha256": current_contract_sha,
            "source_binding_changed": True,
        },
        "historical_handoff": {
            "path": str(OLD_HANDOFF.relative_to(repo_root)),
            "artifact_id": old_handoff["artifact_id"],
            "contract_id": old_handoff["contract_id"],
            "sha256": old_handoff_sha,
            "preserved": True,
        },
        "current_handoff": {
            "path": str(CURRENT_HANDOFF.relative_to(repo_root)),
            "artifact_id": current_handoff["artifact_id"],
            "contract_id": current_handoff["contract_id"],
            "sha256": current_handoff_sha,
            "pytest_count": current_handoff["tests"]["full_pytest_passed"],
        },
        "changed_source_bindings": changed_sources,
        "semantics": {
            "g8_c_science_unchanged": True,
            "g8_c_table_unchanged": True,
            "g8_d_scientific_semantics_unchanged": True,
            "bounded_non_scientific_smoke_refreshed": True,
            "validation_campaign_started": False,
            "selection_started": False,
            "g8_e_started": False,
            "test_split_accessed": False,
        },
    }
    body["provenance_id"] = "g8dportableprov-" + g8_d.sha256_bytes(g8_d.canonical_json(body))
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-commit", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = g8_d.rendered_json(build(repair_commit=args.repair_commit))
    if args.check:
        if OUT.read_bytes() != expected:
            print(f"FAIL: {OUT} is stale", file=sys.stderr)
            return 1
        print(f"PASS: {OUT.relative_to(REPO)} is current")
        return 0
    OUT.write_bytes(expected)
    print(f"PASS: wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
