#!/usr/bin/env python3
"""Freeze pre-execution Imagenette archive, extraction and manifest evidence.

This is provenance-only preparation.  It does not construct a model-facing
loader, decode an image, train, validate a model, or open the test split.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from config.params import get  # noqa: E402
from data.manifests import check_manifest, manifest_path, validate_manifest_bytes  # noqa: E402
from data.provenance import dataset_root, verify_extracted_dataset  # noqa: E402
from training.deterministic_core import canonical_bytes, canonical_sha256  # noqa: E402
from training.w8_protocol import (  # noqa: E402
    W8_DATASET,
    W8_TRAIN_SAMPLE_COUNT,
    W8_VALIDATION_SAMPLE_COUNT,
)

DATA_ROLE = "W8_DATA_PROVENANCE_PRE_EXECUTION"
DATA_PREFIX = "w8data-"
DEFAULT_OUTPUT = REPO / "results/learned/w8/w8_data_verification.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path, repo: Path) -> str:
    try:
        return str(path.relative_to(repo))
    except ValueError as exc:
        raise ValueError(f"W8 data path is outside its checkout: {path}") from exc


def build_data_verification(*, repo: Path = REPO, issued_at_utc: str | None = None) -> dict[str, Any]:
    provenance = verify_extracted_dataset(W8_DATASET, repo)
    manifest_file = manifest_path(W8_DATASET, repo)
    root = dataset_root(W8_DATASET, repo)
    marker = root / ".archive-sha256"
    if provenance.path.is_symlink() or not provenance.path.is_file() or root.is_symlink() or not root.is_dir() or marker.is_symlink() or not marker.is_file() or manifest_file.is_symlink() or not manifest_file.is_file():
        raise ValueError("W8 dataset provenance paths are unsafe")
    manifest_sha = check_manifest(W8_DATASET, repo)
    rows = validate_manifest_bytes(W8_DATASET, manifest_file.read_bytes())
    counts = Counter(row.split for row in rows)
    expected_test = int(get(f"datasets.{W8_DATASET}.test_images"))
    expected_counts = {
        "train": W8_TRAIN_SAMPLE_COUNT,
        "val": W8_VALIDATION_SAMPLE_COUNT,
        "test": expected_test,
    }
    if {split: counts[split] for split in expected_counts} != expected_counts:
        raise ValueError(f"W8 manifest counts differ: {dict(counts)} != {expected_counts}")
    body: dict[str, Any] = {
        "schema_version": 1,
        "artifact_role": DATA_ROLE,
        "status": "VERIFIED_PROVENANCE_ONLY",
        "issued_at_utc": issued_at_utc or datetime.now(timezone.utc).isoformat(),
        "dataset": W8_DATASET,
        "archive": {
            "path": _relative(provenance.path, repo),
            "filename": provenance.filename,
            "source_url": provenance.url,
            "bytes": provenance.byte_length,
            "sha256": provenance.sha256,
        },
        "extraction": {
            "root": _relative(root, repo),
            "marker_path": _relative(marker, repo),
            "marker_sha256": _sha(marker),
            "marker_content_sha256": provenance.sha256,
            "status": "VERIFIED_AGAINST_ARCHIVE_SHA256",
        },
        "manifest": {
            "path": _relative(manifest_file, repo),
            "sha256": manifest_sha,
            "counts": expected_counts,
            "validation_denominator": W8_VALIDATION_SAMPLE_COUNT,
        },
        "access_boundary": {
            "model_facing_train_access": 0,
            "model_facing_validation_access": 0,
            "test_model_facing_access": 0,
            "learned_test_inference": 0,
            "test_state": "SEALED",
            "operation": "ARCHIVE_EXTRACTION_AND_PROVENANCE_MANIFEST_VERIFICATION_ONLY",
        },
    }
    body["verification_id"] = DATA_PREFIX + canonical_sha256(body)
    return body


def verify_data_verification(
    value: dict[str, Any],
    *,
    repo: Path = REPO,
    require_local_data: bool = True,
) -> dict[str, Any]:
    required = {
        "schema_version", "artifact_role", "status", "issued_at_utc", "dataset",
        "archive", "extraction", "manifest", "access_boundary", "verification_id",
    }
    if set(value) != required:
        raise ValueError("W8 data-verification schema differs")
    body = dict(value)
    identifier = body.pop("verification_id")
    if identifier != DATA_PREFIX + canonical_sha256(body):
        raise ValueError("W8 data-verification ID differs")
    if value["schema_version"] != 1 or value["artifact_role"] != DATA_ROLE or value["status"] != "VERIFIED_PROVENANCE_ONLY":
        raise ValueError("W8 data-verification role/status differs")
    if value["dataset"] != W8_DATASET:
        raise ValueError("W8 data-verification dataset differs")
    archive = value["archive"]
    archive_path = repo / str(archive["path"])
    provenance = None
    if require_local_data:
        if archive_path.is_symlink() or not archive_path.is_file():
            raise ValueError("W8 data archive path is missing or unsafe")
        provenance = verify_extracted_dataset(W8_DATASET, repo)
    expected = {
        "path": _relative(archive_path, repo),
        "filename": str(get(f"datasets.{W8_DATASET}.archive_filename")),
        "source_url": str(get(f"datasets.{W8_DATASET}.source_url")),
        "bytes": int(get(f"datasets.{W8_DATASET}.archive_bytes")),
        "sha256": str(get(f"datasets.{W8_DATASET}.archive_sha256")),
    }
    if archive != expected:
        raise ValueError("W8 data archive binding differs")
    if require_local_data and (
        provenance is None
        or provenance.path != archive_path
        or provenance.byte_length != expected["bytes"]
        or provenance.sha256 != expected["sha256"]
    ):
        raise ValueError("W8 data archive binding differs")
    extraction = value["extraction"]
    if not isinstance(extraction, Mapping):
        raise ValueError("W8 data extraction binding is not an object")
    root = dataset_root(W8_DATASET, repo)
    marker = root / ".archive-sha256"
    if require_local_data and (root.is_symlink() or not root.is_dir() or marker.is_symlink() or not marker.is_file()):
        raise ValueError("W8 data extraction path is missing or unsafe")
    recorded_marker_sha = extraction.get("marker_sha256")
    if (
        not isinstance(recorded_marker_sha, str)
        or len(recorded_marker_sha) != 64
        or any(character not in "0123456789abcdef" for character in recorded_marker_sha)
    ):
        raise ValueError("W8 data extraction marker digest is invalid")
    actual_marker_sha = _sha(marker) if require_local_data else recorded_marker_sha
    if extraction != {
        "root": _relative(root, repo),
        "marker_path": _relative(marker, repo),
        "marker_sha256": actual_marker_sha,
        "marker_content_sha256": expected["sha256"],
        "status": "VERIFIED_AGAINST_ARCHIVE_SHA256",
    }:
        raise ValueError("W8 data extraction binding differs")
    manifest = value["manifest"]
    manifest_file = manifest_path(W8_DATASET, repo)
    if require_local_data and (manifest_file.is_symlink() or not manifest_file.is_file()):
        raise ValueError("W8 data manifest path is missing or unsafe")
    rows = validate_manifest_bytes(W8_DATASET, manifest_file.read_bytes()) if require_local_data else []
    counts = Counter(row.split for row in rows)
    expected_manifest = {
        "path": _relative(manifest_file, repo),
        "sha256": str(get(f"datasets.{W8_DATASET}.manifest_sha256")),
        "counts": {
            "train": W8_TRAIN_SAMPLE_COUNT,
            "val": W8_VALIDATION_SAMPLE_COUNT,
            "test": int(get(f"datasets.{W8_DATASET}.test_images")),
        },
        "validation_denominator": W8_VALIDATION_SAMPLE_COUNT,
    }
    if manifest != expected_manifest:
        raise ValueError("W8 data manifest binding differs")
    if require_local_data and (
        _sha(manifest_file) != expected_manifest["sha256"]
        or {key: counts[key] for key in expected_manifest["counts"]} != expected_manifest["counts"]
    ):
        raise ValueError("W8 data manifest binding differs")
    if value["access_boundary"] != {
        "model_facing_train_access": 0,
        "model_facing_validation_access": 0,
        "test_model_facing_access": 0,
        "learned_test_inference": 0,
        "test_state": "SEALED",
        "operation": "ARCHIVE_EXTRACTION_AND_PROVENANCE_MANIFEST_VERIFICATION_ONLY",
    }:
        raise ValueError("W8 data access boundary differs")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_data_verification(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable W8 data verification already exists: {path}")
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
            raise FileExistsError(f"immutable W8 data verification already exists: {path}") from None
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--issued-at-utc")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    output = args.output if args.output.is_absolute() else repo / args.output
    if args.check:
        value = verify_data_verification(json_load(output), repo=repo)
        print(f"W8 data provenance PASS: {value['verification_id']}")
    else:
        value = build_data_verification(repo=repo, issued_at_utc=args.issued_at_utc)
        write_data_verification(value, output)
        print(f"W8 data provenance written: {value['verification_id']}")
    return 0


def json_load(path: Path) -> dict[str, Any]:
    import json

    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
