#!/usr/bin/env python3
"""Generate the pre-data G-8 campaign-opening contract; never run science."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from baseline.classical.g8_campaign import (
    CAMPAIGN,
    CAMPAIGN_MANIFEST,
    PB3C_TERMINAL_SHA,
    PHASE_ORDER,
    PRE_DATA_FLAGS,
    REQUIRED_BLER_IDENTITIES,
    SELECTION_POLICY_FIELDS,
    build_structural_preflight,
    campaign_identifier,
    rendered_json,
    sha256_bytes,
)
from baseline.classical.outage import write_json_atomically
from config.params import REPO_ROOT

W4_ADJUDICATION = Path("results/baseline/w4/integration_adjudication.json")
CONTRACT_SOURCES = (
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
SPLIT_MANIFESTS = (
    "data/manifests/imagenette160.csv",
    "data/manifests/stl10.csv",
    "data/manifests/cifar10.csv",
)


def _binding(path: str, *, role: str) -> dict[str, Any]:
    body = (REPO_ROOT / path).read_bytes()
    return {"path": path, "role": role, "sha256": sha256_bytes(body), "bytes": len(body)}


def _policy_fingerprint(machinery: dict[str, Any]) -> str:
    covered: list[list[Any]] = []
    for field in SELECTION_POLICY_FIELDS:
        head, _, tail = field.partition(".")
        value = machinery[head]
        if tail:
            value = value[tail]
        covered.append([field, value])
    canonical = json.dumps(covered, separators=(",", ":"), ensure_ascii=True)
    return sha256_bytes(canonical.encode("utf-8"))


def build() -> dict[str, Any]:
    adjudication_path = REPO_ROOT / W4_ADJUDICATION
    adjudication = json.loads(adjudication_path.read_text(encoding="utf-8"))
    machinery = adjudication["selection_machinery"]
    observed_policy = machinery["selection_policy_sha256"]
    reproduced_policy = _policy_fingerprint(machinery)
    if observed_policy != reproduced_policy:
        raise RuntimeError("W4 selection policy hash does not reproduce")

    selection_sources: list[dict[str, Any]] = []
    for recorded in adjudication["selection_sources"]:
        current = _binding(recorded["path"], role=recorded["role"])
        if current != {key: recorded[key] for key in current}:
            raise RuntimeError(f"W4 selection source drift: {recorded['path']}")
        current["w4_bound_at"] = recorded["bound_at"]
        selection_sources.append(current)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "campaign": CAMPAIGN,
        "stage": "preflight_contract_only",
        **PRE_DATA_FLAGS,
        "phase_order": list(PHASE_ORDER),
        "scientific_base": {
            "kind": "pb3c_terminal_handoff",
            "commit_sha": PB3C_TERMINAL_SHA,
            "actual_subject": "fix: fix push failure due to gpg for resume.md",
            "source_state_mode": "content_hashes_with_pb3c_base",
            "future_g8a_final_commit_not_part_of_identity": True,
        },
        "w4_adjudication": _binding(str(W4_ADJUDICATION), role="frozen_w4_contract"),
        "selection_policy": {
            "selection_policy_sha256": observed_policy,
            "fields": list(SELECTION_POLICY_FIELDS),
            "tie_break_order": machinery["tie_break_order"],
            "tie_equality": machinery["tie_equality"],
            "frozen_before_data": True,
            "changing_bound_policy_after_campaign_start_invalidates_campaign": True,
        },
        "selection_sources": selection_sources,
        "normative_sources": [
            _binding("spec/SPEC.md", role="normative_spec"),
            _binding("spec/params.generated.yaml", role="generated_parameters"),
        ],
        "dataset_split_manifests": [
            _binding(path, role="split_manifest_bytes_only") for path in SPLIT_MANIFESTS
        ],
        "contract_sources": [
            _binding(path, role="g8a_contract_source") for path in CONTRACT_SOURCES
        ],
        "interpretation_rules": {
            "pre_data_contract_not_authorization": True,
            "later_phases_may_not_silently_reinterpret_earlier_artifacts": True,
            "changed_bound_scientific_policy_invalidates_campaign": True,
        },
        "generated_preflight_artifacts": [
            _binding(
                str(REQUIRED_BLER_IDENTITIES.relative_to(REPO_ROOT)),
                role="required_bler_structural_grid",
            )
        ],
    }
    payload["campaign_id"] = campaign_identifier(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    required = build_structural_preflight()
    if args.check:
        if not REQUIRED_BLER_IDENTITIES.exists():
            raise SystemExit("missing required_bler_identities.json")
        if REQUIRED_BLER_IDENTITIES.read_bytes() != rendered_json(required):
            raise SystemExit("required_bler_identities.json is stale")
        payload = build()
        if not CAMPAIGN_MANIFEST.exists():
            raise SystemExit(f"missing {CAMPAIGN_MANIFEST.relative_to(REPO_ROOT)}")
        expected = rendered_json(payload)
        actual = CAMPAIGN_MANIFEST.read_bytes()
        if actual != expected:
            raise SystemExit("campaign_manifest.json is stale")
        print(
            "ok: campaign manifest matches regenerated pre-data contract "
            f"campaign_id={payload['campaign_id']}"
        )
        return 0
    required_digest = write_json_atomically(REQUIRED_BLER_IDENTITIES, required)
    payload = build()
    digest = write_json_atomically(CAMPAIGN_MANIFEST, payload)
    print(
        f"wrote {CAMPAIGN_MANIFEST.relative_to(REPO_ROOT)} "
        f"campaign_id={payload['campaign_id']} sha256={digest}; "
        f"required_bler_sha256={required_digest} "
        f"candidates={required['counts']['structural_candidates']} "
        f"work_units={required['counts']['required_unique_bler_work_units']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
