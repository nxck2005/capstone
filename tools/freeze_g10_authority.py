#!/usr/bin/env python3
"""Freeze the W9-A/G-10 authority after the clean scientific source epoch."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evaluation.g10_protocol import (  # noqa: E402
    AUTHORIZATION_PATH,
    CLASSICAL_EXTRACT_PATH,
    REPO_ROOT,
    SOURCE_MANIFEST_PATH,
    G10ProtocolHold,
    build_authorization,
    rendered_json,
    sha256_file,
    verify_authorization,
)


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def _publish_once(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise G10ProtocolHold(f"refusing to replace immutable authority artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _publish_or_reuse(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink():
        if path.read_bytes() != raw:
            raise G10ProtocolHold(f"existing immutable artifact differs: {path}")
        return
    _publish_once(path, raw)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--profile-id", default="confessor_pascal_cu126")
    args = parser.parse_args()
    try:
        source_commit = str(args.source_commit)
        if _head() != source_commit:
            raise G10ProtocolHold("authority must be generated from the exact clean source epoch HEAD")
        authorization, source_manifest, classical = build_authorization(
            root=REPO, source_commit=source_commit, profile_id=str(args.profile_id)
        )
        _publish_or_reuse(REPO / CLASSICAL_EXTRACT_PATH, rendered_json(classical))
        _publish_once(REPO / SOURCE_MANIFEST_PATH, rendered_json(source_manifest))
        _publish_once(REPO / AUTHORIZATION_PATH, rendered_json(authorization))
        verified = verify_authorization(root=REPO)
    except (G10ProtocolHold, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"G-10 AUTHORITY HOLD — {exc}", file=sys.stderr)
        return 1
    print(
        "G-10 pre-execution authority PASS: "
        f"{verified['authorization_id']} "
        f"source={verified['scientific_source']['commit']} "
        f"grid={verified['protocol']['snr_grid_db']} "
        f"cells={verified['protocol']['matrix_cell_count']} "
        f"classical_extract_sha256={sha256_file(REPO / CLASSICAL_EXTRACT_PATH)} "
        "g10_outcomes=0 test=SEALED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
