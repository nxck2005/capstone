#!/usr/bin/env python3
"""Freeze the excluded W8 predecessor and fresh successor lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from gen_w8_execution_authorization import verify_authorization  # noqa: E402
from gen_w8_source_manifest import verify_manifest  # noqa: E402
from training.deterministic_core import canonical_bytes, canonical_sha256  # noqa: E402


LINEAGE_ROLE = "W8_R1_SUCCESSOR_LINEAGE"
LINEAGE_PREFIX = "w8lineage-"
DEFAULT_OUTPUT = REPO / "results/learned/w8/w8_r1_successor_lineage.json"
INCIDENT_ID = "w8b2incident-feb598220d1a32143944e1d7a343fff00de43387e255d92c033289e4afcde8c2"
INCIDENT_SHA256 = "f244c48b237f9b5efbe5875653d6111d3ca5902a173451fe7638f16cd752a4c8"
FAILED_SOURCE_COMMIT = "c5a8b70563b1a9e4056c42bca785414924c11fa2"
FAILED_CAMPAIGN_ID = "w8-final-pascal-20260831"
PARTIAL_CHECKPOINT_SHA256 = "ff89322a795e437994ff5eaaf5c7157fd7e751aed8bafb2f42cee950d371b55c"
SUCCESSOR_SOURCE_COMMIT = "d52d85dd60bac0c816a7ba249e4453045723277b"
SUCCESSOR_CAMPAIGN_ID = "w8-final-pascal-20260901-r1"
SUCCESSOR_CAMPAIGN_ROOT = "/home/nick/w8-final-pascal-20260901-r1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_lineage(
    *,
    source_manifest_path: Path,
    execution_authorization_path: Path,
    issued_at_utc: str,
) -> dict[str, Any]:
    manifest = verify_manifest(
        source_manifest_path, expected_source_commit=SUCCESSOR_SOURCE_COMMIT
    )
    authorization = verify_authorization(
        execution_authorization_path,
        expected_source_commit=SUCCESSOR_SOURCE_COMMIT,
        expected_source_manifest_path=source_manifest_path,
    )
    body: dict[str, Any] = {
        "schema_version": 1,
        "artifact_role": LINEAGE_ROLE,
        "status": "IMMUTABLE_PROVENANCE_ONLY",
        "issued_at_utc": issued_at_utc,
        "predecessor": {
            "incident_id": INCIDENT_ID,
            "incident_file_sha256": INCIDENT_SHA256,
            "source_commit": FAILED_SOURCE_COMMIT,
            "campaign_id": FAILED_CAMPAIGN_ID,
            "partial_checkpoint_sha256": PARTIAL_CHECKPOINT_SHA256,
            "historical_failed_optimizer_steps": 259,
            "historical_accepted_coverage": 0,
            "resume_eligible": False,
            "result_eligible": False,
            "g10_eligible": False,
            "test_eligible": False,
        },
        "successor": {
            "source_commit": manifest["source_commit"],
            "source_manifest_id": manifest["manifest_id"],
            "source_manifest_sha256": _sha(source_manifest_path),
            "execution_authorization_id": authorization["authorization_id"],
            "execution_authorization_sha256": _sha(execution_authorization_path),
            "campaign_id": SUCCESSOR_CAMPAIGN_ID,
            "campaign_root": SUCCESSOR_CAMPAIGN_ROOT,
            "initialization": "fresh deterministic genesis",
            "predecessor_checkpoint_id": None,
            "scientific_coverage": 0,
        },
        "provenance_only": True,
        "scientific_execution_started": False,
        "g10": False,
        "er2_randomized_training": False,
        "papr_constrained_training": False,
        "er9_training": False,
        "test_access": 0,
    }
    body["lineage_id"] = LINEAGE_PREFIX + canonical_sha256(body)
    return body


def verify_lineage(
    path: Path,
    *,
    source_manifest_path: Path,
    execution_authorization_path: Path,
) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("W8 R1 successor lineage is not an object")
    expected = build_lineage(
        source_manifest_path=source_manifest_path,
        execution_authorization_path=execution_authorization_path,
        issued_at_utc=str(value.get("issued_at_utc", "")),
    )
    if value != expected:
        raise ValueError("W8 R1 successor lineage differs")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_lineage(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable W8 R1 successor lineage already exists: {path}")
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
            raise FileExistsError(
                f"immutable W8 R1 successor lineage already exists: {path}"
            ) from None
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--issued-at-utc", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    paths = {
        "source": args.source_manifest.resolve(),
        "authorization": args.execution_authorization.resolve(),
        "output": args.output.resolve(),
    }
    if args.check:
        value = verify_lineage(
            paths["output"],
            source_manifest_path=paths["source"],
            execution_authorization_path=paths["authorization"],
        )
    else:
        value = build_lineage(
            source_manifest_path=paths["source"],
            execution_authorization_path=paths["authorization"],
            issued_at_utc=args.issued_at_utc,
        )
        write_lineage(value, paths["output"])
    print(
        json.dumps(
            {
                "status": "PASS",
                "path": str(paths["output"].relative_to(REPO)),
                "lineage_id": value["lineage_id"],
                "file_sha256": _sha(paths["output"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
