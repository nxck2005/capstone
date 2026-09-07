#!/usr/bin/env python3
"""Build the immutable ER-9/ER-2 pre-science source manifest."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
from collections.abc import Mapping
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from training.deterministic_core import canonical_bytes, canonical_sha256  # noqa: E402


MANIFEST_ROLE = "ER9_ER2_PRE_SCIENCE_SOURCE_MANIFEST"
PREFIX = "ersource-"
DEFAULT_OUTPUT = REPO / "results/learned/er9/er_execution_source_manifest.json"

CRITICAL_SOURCES: tuple[tuple[str, str], ...] = (
    ("src/evaluation/er9_protocol.py", "am96_er9_interface_and_range_coder"),
    ("src/evaluation/er9_search.py", "am96_two_stage_search_and_packet_floor"),
    ("src/evaluation/er9_transport.py", "existing_phy_batched_adapter"),
    ("src/evaluation/er9_campaign.py", "validation_real_chain_evaluation"),
    ("src/evaluation/h4_precision.py", "g11_h4_validation_precision_simulation"),
    ("src/models/er9_digital.py", "er9_shared_trunk_and_task_head"),
    ("src/training/er9.py", "er9_training_checkpoint_and_selection"),
    ("src/training/er2_randomized.py", "single_am95_randomized_er2_training"),
    ("src/training/er2_snr.py", "am95_keyed_snr_selection"),
    ("src/training/deterministic_core.py", "optimizer_accounting_and_canonical_ids"),
    ("src/artifacts/rng.py", "keyed_rng_core"),
    ("src/models/djscc.py", "shared_djscc_architecture_and_channel"),
    ("src/models/reference_classifier.py", "channel_normalisation_dependency"),
    ("src/models/task_heads.py", "shared_task_head_family"),
    ("src/channels/awgn.py", "shared_awgn"),
    ("src/channels/registry.py", "channel_registry"),
    ("src/baseline/classical/channel_transport.py", "existing_digital_phy"),
    ("src/baseline/ldpc/adapter.py", "sionna_ldpc_adapter"),
    ("src/baseline/ldpc/transport.py", "exact_packetisation_solver"),
    ("src/baseline/ldpc/segmentation.py", "transport_segmentation"),
    ("src/baseline/ldpc/modulation.py", "configured_mapper_and_demapper"),
    ("src/data/djscc_training.py", "train_only_data_view"),
    ("src/data/djscc_validation.py", "validation_only_data_view"),
    ("src/data/classifier.py", "keyed_batch_order"),
    ("src/data/registry.py", "test_sealed_registry_boundary"),
    ("src/data/preprocessing.py", "canonical_input_contract"),
    ("src/config/params.py", "parameter_loader"),
    ("src/config/run_config.py", "resolved_config_hash"),
    ("src/config/execution_profiles.py", "profile_authentication"),
    ("src/env.py", "deterministic_backend"),
    ("src/evaluation/am96_spec_compatibility.py", "am96_semantic_compatibility_contract"),
    ("configs/er9-digital.yaml", "er9_config"),
    ("configs/learned-er2-randomized.yaml", "randomized_er2_config"),
    ("spec/SPEC.md", "normative_specification"),
    ("spec/params.generated.yaml", "generated_parameter_snapshot"),
    ("requirements-pascal.lock", "qualified_execution_lock"),
    ("tools/run_er9_campaign.py", "scientific_campaign_runner"),
    ("tools/gen_er9_source_manifest.py", "source_manifest_builder"),
    ("tools/gen_er9_stage1_authorization.py", "stage1_authorization_builder"),
    ("tools/verify_er9.py", "er9_verifier"),
    ("tools/verify_er2_randomized.py", "er2_verifier"),
    ("tools/verify_g11.py", "g11_verifier"),
    ("tests/test_er9_protocol.py", "er9_protocol_tests"),
    ("tests/test_er9_search.py", "er9_search_tests"),
    ("tests/test_er9_safety.py", "er9_safety_tests"),
    ("tests/test_er2_snr.py", "am95_assignment_tests"),
)


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, check=True).stdout


def _commit(commit: str) -> str:
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):  # literal-ok: Git SHA-1 width
        raise ValueError("source commit must be a full lowercase SHA-1")
    actual = _git("rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip()
    if actual != commit:
        raise ValueError("source commit is not an exact commit object")
    return commit


def _source(commit: str, path: str, role: str) -> dict[str, Any]:
    try:
        raw = _git("show", f"{commit}:{path}")
        blob = _git("rev-parse", f"{commit}:{path}").decode().strip()
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"source path is absent at {commit}: {path}") from exc
    if not raw or len(blob) != 40:
        raise ValueError(f"source path is empty or invalid: {path}")
    return {
        "path": path,
        "role": role,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "git_blob_sha1": blob,
    }


def build_manifest(source_commit: str) -> dict[str, Any]:
    commit = _commit(source_commit)
    listing = _git("ls-tree", "-r", "--name-only", commit, "--", "results/learned/er9").decode()
    if any(line == "results/learned/er9" or line.startswith("results/learned/er9/") for line in listing.splitlines()):
        raise ValueError("ER-9 scientific results exist in the implementation source epoch")
    entries = [_source(commit, path, role) for path, role in CRITICAL_SOURCES]
    body: dict[str, Any] = {
        "schema_version": 1,  # literal-ok: source manifest schema
        "artifact_role": MANIFEST_ROLE,
        "status": "IMMUTABLE_PRE_SCIENCE_SOURCE_EPOCH",
        "source_commit": commit,
        "source_commit_comparison": "git_show_exact_source_commit_not_current_head",
        "entries": entries,
        "entry_count": len(entries),
        "scientific_results_included": False,
        "er9_training": 0,
        "randomized_er2_training": 0,
        "g11": 0,
        "test_access": 0,
        "test": "SEALED",
    }
    body["manifest_id"] = PREFIX + canonical_sha256(body)
    return body


def write_immutable(value: Mapping[str, Any], path: Path) -> None:
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable source manifest exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(dict(value)))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else REPO / args.output
    value = build_manifest(args.source_commit)
    write_immutable(value, output)
    print(f"ER source manifest written: {value['manifest_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
