#!/usr/bin/env python3
"""Generate and verify the W7 training/evaluation source manifest.

The manifest binds an execution-source commit.  It is published in a later
carrier commit, so the manifest never contains the hash of the commit that
adds itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "results/learned/w7/w7_source_manifest.json"
SOURCES = {
    "configs/learned-w7-g4-pilot.yaml": "w7_run_config",
    "instructions/W7-A.txt": "w7a_contract_text",
    "results/learned/w5/w5_gradscaler_accounting_repair_completion.json": "w5_terminal_authority",
    "results/learned/w5/w5_source_manifest_v4.json": "w5_source_authority",
    "results/baseline/w6/w6_completion.json": "w6_terminal_authority",
    "requirements-pascal.lock": "pascal_execution_lock",
    "results/learned/w7/w7_a_contract.json": "w7a_contract_artifact",
    "spec/params.generated.yaml": "resolved_parameters",
    "spec/schemas/w7_g4_artifacts.schema.json": "w7_artifact_schema",
    "src/artifacts/rng.py": "keyed_rng",
    "src/channels/awgn.py": "awgn_channel",
    "src/channels/power.py": "power_normalisation_and_papr",
    "src/channels/registry.py": "channel_registry",
    "src/config/execution_profiles.py": "profile_registry_authenticator",
    "src/config/params.py": "parameter_loader",
    "src/config/run_config.py": "resolved_run_config",
    "src/config/w7_execution.py": "uuid_bound_profile_authenticator",
    "src/data/classifier.py": "keyed_order_sampler",
    "src/data/djscc_training.py": "training_dataset",
    "src/data/djscc_validation.py": "validation_dataset",
    "src/data/preprocessing.py": "canonical_preprocessing_and_psnr",
    "src/data/registry.py": "dataset_registry_test_sealed",
    "src/env.py": "determinism_environment",
    "src/evaluation/w7_validation.py": "validation_evaluator",
    "src/models/djscc.py": "djscc_model",
    "src/models/reference_classifier.py": "model_normalisation",
    "src/models/task_heads.py": "task_head_registry",
    "src/runtime/w7_lock.py": "campaign_sole_writer_lock",
    "src/training/deterministic_core.py": "generic_deterministic_training_core",
    "src/training/djscc_loss.py": "dual_head_loss",
    "src/training/w5_compatibility.py": "w5_compatibility_policy",
    "src/training/w7_g4.py": "w7_training_policy_and_checkpoints",
    "src/training/w7_protocol.py": "w7_protocol_resolution",
    "src/adjudication/w7_g4.py": "frozen_g4_adjudicator",
    "tools/gen_w7_a_contract.py": "contract_generator",
    "tools/gen_w7_a_completion.py": "completion_generator",
    "tools/gen_w7_source_manifest.py": "source_manifest_generator",
    "tools/run_w7_profile.py": "real_data_profile_runner",
    "tools/run_w7_campaign.py": "detached_scientific_launcher",
    "tools/verify_w7_a.py": "w7a_verifier",
    "tools/verify_w7_profile.py": "profile_verifier",
    "tools/freeze_w7_profile.py": "profile_freeze_builder",
    "tests/test_w7_protocol.py": "protocol_regressions",
    "tests/test_w7_lock.py": "sole_writer_regressions",
    "tests/test_w7_adjudicator.py": "g4_fixture_regressions",
}


def _run(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, check=True).stdout


def _canonical(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def build(commit: str) -> dict[str, Any]:
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("source commit must be a full Git SHA-1")
    entries = []
    for path, role in sorted(SOURCES.items()):
        raw = _run("show", f"{commit}:{path}")
        blob = _run("rev-parse", f"{commit}:{path}").decode().strip()
        entries.append({
            "path": path,
            "role": role,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "git_blob": blob,
        })
    body = {
        "schema_version": 1,
        "artifact_role": "w7_training_critical_source_manifest",
        "source_commit": commit,
        "entries": entries,
        "scientific_execution_authorization": "ABSENT_DURING_W7_A",
    }
    body["manifest_id"] = "w7source-" + hashlib.sha256(_canonical(body)).hexdigest()
    return body


def verify(value: dict[str, Any], *, current: bool) -> None:
    required = {"schema_version", "artifact_role", "source_commit", "entries", "scientific_execution_authorization", "manifest_id"}
    if set(value) != required or value["schema_version"] != 1 or value["artifact_role"] != "w7_training_critical_source_manifest":
        raise ValueError("W7 source manifest schema/role differs")
    if value["scientific_execution_authorization"] != "ABSENT_DURING_W7_A":
        raise ValueError("W7 source manifest authorization boundary differs")
    expected = build(str(value["source_commit"]))
    if value != expected:
        raise ValueError("W7 source manifest differs from bound Git bytes")
    if current:
        for entry in value["entries"]:
            path = REPO / entry["path"]
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"W7 current source missing/unsafe: {entry['path']}")
            raw = path.read_bytes()
            if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise ValueError(f"W7 current source byte drift: {entry['path']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--check-current", action="store_true")
    args = parser.parse_args()
    if args.check:
        value = json.loads(args.output.read_bytes())
        verify(value, current=args.check_current)
        print(f"W7 source manifest PASS: {value['manifest_id']}")
        return 0
    if not args.source_commit:
        parser.error("generation requires --source-commit")
    value = build(args.source_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical(value))
    print(f"wrote {args.output.relative_to(REPO)}: {value['manifest_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
