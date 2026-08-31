#!/usr/bin/env python3
"""Create/verify the immutable W8 scientific source-epoch manifest.

The manifest reads bytes from ``git show <source_commit>:<path>``.  It never
uses the working tree to answer what source produced a future result, and it
never includes W8 runtime/results bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from training.deterministic_core import canonical_bytes, canonical_sha256  # noqa: E402


MANIFEST_ROLE = "W8_SCIENTIFIC_SOURCE_MANIFEST"
MANIFEST_PREFIX = "w8source-"
DEFAULT_OUTPUT = REPO / "results/learned/w8/w8_source_manifest.json"

# This is deliberately explicit.  Adding a result-affecting W8 dependency
# requires a new source epoch and a new manifest, not a silent omission.
CRITICAL_SOURCES: tuple[tuple[str, str], ...] = (
    ("src/training/w8_final.py", "w8_trainer_checkpoint_resume"),
    ("src/training/w8_protocol.py", "w8_protocol_and_matrix"),
    ("src/evaluation/w8_validation.py", "w8_validation_and_selection"),
    ("src/runtime/w8_lock.py", "w8_campaign_kernel_lock"),
    ("src/config/w8_execution.py", "w8_gpu_profile_authentication"),
    ("src/training/deterministic_core.py", "optimizer_and_gradscaler_accounting"),
    ("src/training/djscc_loss.py", "learned_loss"),
    ("src/artifacts/rng.py", "keyed_rng_core"),
    ("src/models/djscc.py", "djscc_residual_v1_architecture"),
    ("src/models/reference_classifier.py", "classifier_and_normalisation_dependency"),
    ("src/models/task_heads.py", "task_head_dependency"),
    ("src/channels/awgn.py", "awgn_channel"),
    ("src/channels/power.py", "power_normalisation"),
    ("src/channels/registry.py", "channel_registry"),
    ("src/data/djscc_training.py", "training_data_view"),
    ("src/data/djscc_validation.py", "validation_data_view"),
    ("src/data/classifier.py", "keyed_batch_order"),
    ("src/data/adapters.py", "dataset_source_adapter"),
    ("src/data/provenance.py", "dataset_archive_provenance"),
    ("src/data/preprocessing.py", "canonical_preprocessing"),
    ("src/data/registry.py", "dataset_registry_and_split_boundary"),
    ("src/data/test_access.py", "sealed_test_boundary"),
    ("src/data/manifests.py", "manifest_loader"),
    ("src/data/identity.py", "stable_identity"),
    ("src/config/params.py", "parameter_loader"),
    ("src/config/run_config.py", "config_fingerprint"),
    ("src/config/execution_profiles.py", "execution_profile_registry"),
    ("src/env.py", "deterministic_environment_and_profile_boundary"),
    ("configs/learned-w8-final.yaml", "w8_choices_and_sweep_schema"),
    ("spec/SPEC.md", "normative_specification_at_source_freeze"),
    ("spec/params.generated.yaml", "generated_parameter_identity"),
    ("spec/schemas/w8_final_artifacts.schema.json", "w8_artifact_schema"),
    ("requirements-pascal.lock", "pascal_execution_environment_lock"),
    ("tools/run_w8_campaign.py", "detached_w8_campaign_runner"),
    ("tools/verify_w8_a.py", "w8_pre_execution_verifier"),
    ("tools/gen_w8_source_manifest.py", "w8_source_manifest_builder"),
    ("tools/gen_w8_execution_authorization.py", "w8_execution_authority_builder"),
    ("tools/gen_w8_smoke.py", "w8_non_scientific_smoke_builder"),
    ("tools/gen_w8_data_verification.py", "w8_data_provenance_builder"),
    ("tools/gen_w8_runtime_estimate.py", "w8_runtime_estimate_builder"),
    ("tools/gen_w8_a_completion.py", "w8_pre_execution_completion_builder"),
    ("src/baseline/w8_spec_compatibility.py", "w8_spec_compatibility_verifier"),
    ("src/baseline/w7c_source_compatibility.py", "w7c_successor_compatibility_verifier"),
    ("src/baseline/w6_evidence.py", "historical_w6_evidence_compatibility_verifier"),
    ("results/learned/w7/w7_spec_additive_compatibility.json", "w7_spec_additive_compatibility_authority"),
    ("results/learned/w7/w8_spec_additive_compatibility.json", "w8_spec_additive_compatibility_authority"),
    ("tools/verify_w7_g4.py", "upstream_w7_g4_verifier"),
    ("tools/verify_w7_b2r.py", "upstream_w7_b2r_verifier"),
    ("tools/verify_w7_b1.py", "upstream_w7_b1_verifier"),
    ("tools/verify_w5_training_system.py", "upstream_w5_verifier"),
    ("tools/verify_w6_complete.py", "upstream_w6_verifier"),
)


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, check=check
    )


def _validate_commit(commit: str) -> str:
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):  # literal-ok: Git SHA-1 width
        raise ValueError("W8 source commit must be a full lowercase SHA-1")
    actual = _git("rev-parse", "--verify", f"{commit}^{{commit}}").stdout.decode("ascii").strip()
    if actual != commit:
        raise ValueError("W8 source commit is not an exact commit object")
    return commit


def _source_tree_has_w8_results(commit: str) -> bool:
    listing = _git(
        "ls-tree", "-r", "--name-only", commit, "--", "results/learned/w8"
    ).stdout.decode("utf-8")
    return any(
        item == "results/learned/w8" or item.startswith("results/learned/w8/")
        for item in listing.splitlines()
    )


def _source_bytes(commit: str, path: str) -> tuple[bytes, str]:
    try:
        raw = _git("show", f"{commit}:{path}").stdout
        blob = _git("rev-parse", f"{commit}:{path}").stdout.decode("ascii").strip()
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"W8 source path is absent at {commit}: {path}") from exc
    if not raw or len(blob) != 40:
        raise ValueError(f"W8 source path is empty or has an invalid blob: {path}")
    return raw, blob


def _entry(commit: str, path: str, role: str) -> dict[str, Any]:
    raw, blob = _source_bytes(commit, path)
    return {
        "path": path,
        "role": role,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "git_blob_sha1": blob,
    }


def build_manifest(source_commit: str) -> dict[str, Any]:
    commit = _validate_commit(source_commit)
    if _source_tree_has_w8_results(commit):
        raise ValueError("W8 scientific results are present in the source epoch")
    entries = [_entry(commit, path, role) for path, role in CRITICAL_SOURCES]
    body: dict[str, Any] = {
        "schema_version": 1,
        "artifact_role": MANIFEST_ROLE,
        "status": "IMMUTABLE_SOURCE_EPOCH",
        "source_commit": commit,
        "source_commit_comparison": "git_show_exact_source_commit_not_current_head",
        "entries": entries,
        "entry_count": len(entries),
        "scientific_w8_results_included": False,
        "scientific_execution_authorization": "ABSENT_AT_SOURCE_FREEZE",
        "runtime_root_included": False,
        "test_access": 0,
    }
    body["manifest_id"] = MANIFEST_PREFIX + canonical_sha256(body)
    return body


def _read_json(path: Path) -> dict[str, Any]:
    import json

    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def verify_manifest(path: Path, *, expected_source_commit: str | None = None) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"W8 source manifest path is missing or unsafe: {path}")
    value = _read_json(path)
    required = {
        "schema_version", "artifact_role", "status", "source_commit",
        "source_commit_comparison", "entries", "entry_count",
        "scientific_w8_results_included", "scientific_execution_authorization",
        "runtime_root_included", "test_access", "manifest_id",
    }
    if set(value) != required:
        raise ValueError("W8 source manifest schema differs")
    body = dict(value)
    identifier = body.pop("manifest_id")
    if identifier != MANIFEST_PREFIX + canonical_sha256(body):
        raise ValueError("W8 source manifest ID does not authenticate its body")
    if value["schema_version"] != 1 or value["artifact_role"] != MANIFEST_ROLE or value["status"] != "IMMUTABLE_SOURCE_EPOCH":
        raise ValueError("W8 source manifest role/status differs")
    if value["source_commit_comparison"] != "git_show_exact_source_commit_not_current_head":
        raise ValueError("W8 source manifest comparison rule differs")
    commit = _validate_commit(str(value["source_commit"]))
    if expected_source_commit is not None and commit != expected_source_commit:
        raise ValueError("W8 source manifest source commit differs from the expected epoch")
    entries = value["entries"]
    if not isinstance(entries, list) or len(entries) != len(CRITICAL_SOURCES) or value["entry_count"] != len(entries):
        raise ValueError("W8 source manifest entry count differs")
    expected_paths = [path for path, _role in CRITICAL_SOURCES]
    if any(not isinstance(item, dict) for item in entries):
        raise ValueError("W8 source manifest entries must be objects")
    if [item["path"] for item in entries] != expected_paths:
        raise ValueError("W8 source manifest path/order differs")
    expected_entries = [_entry(commit, path, role) for path, role in CRITICAL_SOURCES]
    if entries != expected_entries:
        raise ValueError("W8 source manifest bytes differ from the exact source commit")
    if value["scientific_w8_results_included"] is not False or value["scientific_execution_authorization"] != "ABSENT_AT_SOURCE_FREEZE" or value["runtime_root_included"] is not False or value["test_access"] != 0:
        raise ValueError("W8 source manifest scientific boundary differs")
    if _source_tree_has_w8_results(commit):
        raise ValueError("W8 scientific results are present in the source epoch")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_manifest(value: dict[str, Any], path: Path) -> None:
    """Publish an immutable source manifest without replacing a final name."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable W8 source manifest already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            raise FileExistsError(f"immutable W8 source manifest already exists: {path}") from None
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="verify an existing manifest")
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else REPO / args.output
    if args.check:
        value = verify_manifest(output, expected_source_commit=args.source_commit)
        print(f"W8 source manifest PASS: {value['manifest_id']}")
    else:
        value = build_manifest(args.source_commit)
        write_manifest(value, output)
        print(f"W8 source manifest written: {value['manifest_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
