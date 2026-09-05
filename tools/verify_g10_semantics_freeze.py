#!/usr/bin/env python3
"""Verify the count-only AM-94 freeze without loading data or models."""

from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evaluation.g10_spec_compatibility import G10SemanticsFreezeError, load  # noqa: E402


def main() -> int:
    try:
        value = load(REPO)
    except (G10SemanticsFreezeError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"AM-94 HOLD — {exc}", file=sys.stderr)
        return 1
    print(
        "AM-94 pre-science semantics freeze PASS: "
        f"{value['freeze_id']} source_sha256={value['entries'][0]['current_sha256']} "
        "g10_model_facing_evaluations=0 test=SEALED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
