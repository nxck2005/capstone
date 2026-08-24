#!/usr/bin/env python3
"""Freeze/check the narrow AM-89 historical G8 compatibility chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "results/baseline/g8_f/am89_f2_source_compatibility.json"
PRIOR_PATH = ROOT / "results/baseline/g8_f/am88_post_campaign_source_compatibility.json"
PRIOR_COMMIT = "1bca1fb2e3455a4b424766c6b3296af2911e72ef"
FILES = (
    "spec/SPEC.md",
    "spec/params.generated.yaml",
    "src/baseline/g8_f_f0.py",
    "src/baseline/g8_campaign.py",
    "src/baseline/g8_pascal_production.py",
    "src/baseline/g8_d.py",
    "src/baseline/g8_e.py",
    "src/baseline/g8_e_corrected_v3s.py",
    "src/baseline/g8_f_sampler_plan.py",
    "tools/verify_g8_f_sampler_plan.py",
    "tools/verify_w4_baseline_integration.py",
)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def rendered(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


def leaves(old: Any, new: Any, prefix: str = "") -> set[str]:
    if isinstance(old, dict) and isinstance(new, dict):
        result: set[str] = set()
        for key in set(old) | set(new):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in old or key not in new:
                result.add(child)
            else:
                result.update(leaves(old[key], new[key], child))
        return result
    return set() if old == new else {prefix}


def build() -> dict[str, Any]:
    prior_raw = PRIOR_PATH.read_bytes()
    prior = json.loads(prior_raw)
    entries = []
    raw_by_path = {}
    for relative in FILES:
        archived = subprocess.run(["git", "show", f"{PRIOR_COMMIT}:{relative}"], cwd=ROOT, capture_output=True, check=True).stdout
        current = (ROOT / relative).read_bytes()
        raw_by_path[relative] = (archived, current)
        entries.append({"path": relative, "archived_bytes": len(archived), "archived_sha256": sha(archived), "current_bytes": len(current), "current_sha256": sha(current)})
    old_params, new_params = raw_by_path["spec/params.generated.yaml"]
    allowed = sorted(leaves(yaml.safe_load(old_params), yaml.safe_load(new_params)))
    body = {
        "schema_version": 1,
        "artifact_role": "g8_f_am89_f2_historical_source_compatibility",
        "amendment": "AM-89",
        "timing": "f1_green_closed_pre_f2_optimizer_step_1",
        "prior_commit": PRIOR_COMMIT,
        "prior_compatibility": {"path": str(PRIOR_PATH.relative_to(ROOT)), "compatibility_id": prior["compatibility_id"], "sha256": sha(prior_raw)},
        "entries": entries,
        "allowed_parameter_paths": allowed,
        "allowed_parameter_prefix": "reference_classifier.artifact_finetune_recipe.",
        "protected_boundary": {"f1_rerun": 0, "f2_optimizer_steps": 0, "f2_validation_inference": 0, "f3_cached_sweep_inference": 0, "pass_two": 0, "fallback": 0, "learned_training": 0, "test_access": 0, "g8_c_changed": False, "g8_d_changed": False, "g8_e_changed": False, "pass_one_rerun": False},
        "interpretation": "additive protocol completion only; historical G8/F0/F1 bindings retain archived bytes",
    }
    body["compatibility_id"] = "g8postsource-" + sha(canonical(body))
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = rendered(build())
    if args.check:
        if PATH.read_bytes() != expected:
            raise SystemExit("AM-89 compatibility artifact differs")
        print("PASS")
    else:
        PATH.parent.mkdir(parents=True, exist_ok=True)
        PATH.write_bytes(expected)
        print(json.loads(expected)["compatibility_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
