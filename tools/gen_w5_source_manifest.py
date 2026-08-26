#!/usr/bin/env python3
"""Generate/verify the immutable W5 training-critical source manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
OUTPUT = REPO / "results/learned/w5/w5_source_manifest.json"
SOURCES = {
    "instructions/W5.txt": "w5_contract",
    "spec/SPEC.md": "normative_specification",
    "spec/params.generated.yaml": "resolved_parameters",
    "spec/schemas/w5_training_artifacts.schema.json": "artifact_schema",
    "configs/learned-w5-smoke.yaml": "cifar_smoke_config",
    "configs/learned-w5-imagenette-r1-6-smoke.yaml": "selected_ratio_smoke_config",
    "configs/learned-w5-imagenette-r1-24-smoke.yaml": "selected_ratio_smoke_config",
    "src/artifacts/rng.py": "keyed_rng",
    "src/channels/awgn.py": "training_channel",
    "src/channels/power.py": "power_normalisation",
    "src/channels/registry.py": "channel_registry",
    "src/config/params.py": "parameter_loader",
    "src/config/run_config.py": "resolved_run_config",
    "src/data/classifier.py": "training_dataset_sampler",
    "src/data/preprocessing.py": "augmentation_preprocessing",
    "src/data/registry.py": "common_dataset_registry",
    "src/env.py": "determinism_execution_profile",
    "src/models/djscc.py": "djscc_model",
    "src/models/reference_classifier.py": "model_owned_normalisation",
    "src/models/task_heads.py": "task_head_registry",
    "src/training/djscc.py": "production_training_engine",
    "src/training/djscc_loss.py": "dual_head_objective",
    "tools/run_djscc_training.py": "process_runner",
    "tools/run_w5_training_smoke.py": "fresh_process_smoke_orchestrator",
    "tools/verify_w5_training_system.py": "w5_verifier",
}


def _run(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, check=True).stdout


def _git_bytes(commit: str, path: str) -> bytes:
    return _run("show", f"{commit}:{path}")


def _canonical(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def build(commit: str) -> dict[str, Any]:
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("source commit must be a full Git SHA-1")
    entries = []
    for path, role in sorted(SOURCES.items()):
        raw = _git_bytes(commit, path)
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
        "artifact_role": "w5_training_critical_source_manifest",
        "source_commit": commit,
        "entries": entries,
    }
    body["manifest_id"] = "w5source-" + hashlib.sha256(_canonical(body)).hexdigest()
    return body


def verify(value: dict[str, Any], *, current: bool) -> None:
    required = {"schema_version", "artifact_role", "manifest_id", "source_commit", "entries"}
    if set(value) != required or value["schema_version"] != 1 or value["artifact_role"] != "w5_training_critical_source_manifest":
        raise ValueError("W5 source manifest schema/role differs")
    expected = build(value["source_commit"])
    if value != expected:
        raise ValueError("W5 source manifest differs from bound Git bytes")
    if current:
        for entry in value["entries"]:
            path = REPO / entry["path"]
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"W5 current source missing/unsafe: {entry['path']}")
            raw = path.read_bytes()
            if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise ValueError(f"W5 current source byte drift: {entry['path']}")


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
        print(f"W5 source manifest PASS: {value['manifest_id']}")
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
