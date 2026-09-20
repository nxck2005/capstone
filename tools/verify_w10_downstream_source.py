#!/usr/bin/env python3
"""Verify the active W10 successor source manifest and the AM-99 current epoch."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evaluation.downstream_v4 import require, sha256_file  # noqa: E402
from runtime.source_epochs import (  # noqa: E402
    HISTORICAL_SOURCE_PATH,
    active_manifest_path,
    load_w10_manifest,
)


def main() -> int:
    historical = REPO / HISTORICAL_SOURCE_PATH
    require(historical.is_file(), "historical W9 downstream source manifest is missing")
    require(
        sha256_file(historical) == "dac633998a84ebfa5c4d299366ee186e1d1070a1b5e3301ea3d4dc3529684c53",
        "historical W9 downstream source manifest bytes differ",
    )
    active = active_manifest_path(REPO)
    value = load_w10_manifest(REPO, live=True)
    from evaluation.am99_spec_compatibility import load as load_am99_spec_compatibility

    load_am99_spec_compatibility(REPO)
    print(
        f"W10 downstream source manifest PASS: {value['manifest_id']} "
        f"({value['manifest_kind']}); sha256={sha256_file(active)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
