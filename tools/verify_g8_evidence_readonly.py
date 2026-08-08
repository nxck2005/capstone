#!/usr/bin/env python3
"""Verify committed G8 evidence without importing or executing a runner.

The source-epoch verifier performs the authenticated campaign census.  This
wrapper adds the evidence-lane-only checks for canonical JSON and transient
coordination artefacts.  It deliberately has no production execution path.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))

from baseline.g8_campaign import canonical_json  # noqa: E402
import verify_g8_bler_characterization_manifest_v2 as epoch2  # noqa: E402


EVIDENCE_ROOT = REPO / "results/baseline/g8/work_units"
TRANSIENT_PARTS = frozenset({"lock", "locks", "staging", "stage", "tmp", "temp"})
TRANSIENT_SUFFIXES = frozenset({".lock", ".staging", ".tmp", ".temp"})


class EvidenceVerificationError(RuntimeError):
    """The committed evidence is not suitable for an evidence-only check."""


def _tracked_paths() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "results/baseline/g8/work_units"],
        check=True,
        capture_output=True,
    )
    return [REPO / name.decode("utf-8") for name in result.stdout.split(b"\0") if name]


def _verify_no_transient_files(paths: list[Path]) -> None:
    for path in paths:
        relative = path.relative_to(REPO)
        if any(part in TRANSIENT_PARTS for part in relative.parts):
            raise EvidenceVerificationError(f"tracked transient evidence path: {relative}")
        if path.suffix in TRANSIENT_SUFFIXES:
            raise EvidenceVerificationError(f"tracked transient evidence suffix: {relative}")


def _verify_canonical_json(paths: list[Path]) -> None:
    for path in paths:
        if path.suffix != ".json":
            raise EvidenceVerificationError(f"non-JSON tracked evidence path: {path.relative_to(REPO)}")
        try:
            raw = path.read_bytes()
            value = json.loads(raw)
        except (OSError, ValueError, TypeError) as exc:
            raise EvidenceVerificationError(f"invalid evidence JSON: {path}: {exc}") from exc
        if raw != canonical_json(value):
            raise EvidenceVerificationError(f"non-canonical evidence JSON: {path.relative_to(REPO)}")


def verify() -> dict[str, object]:
    paths = _tracked_paths()
    if not paths:
        raise EvidenceVerificationError("no tracked G8 work-unit evidence")
    _verify_no_transient_files(paths)
    _verify_canonical_json(paths)
    result = epoch2.verify(require_registered=True)
    return {
        "tracked_work_unit_files": len(paths),
        "completed_count": result["completed_count"],
        "remaining_count": result["remaining_count"],
        "test_split_access": result["test_split_access"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        result = verify()
    except Exception as exc:
        raise SystemExit(f"read-only G8 evidence verification failed: {exc}") from exc
    print("read-only G8 evidence verification PASS: " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
