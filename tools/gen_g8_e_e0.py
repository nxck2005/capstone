#!/usr/bin/env python3
"""Generate the deterministic, zero-coverage G8_E E0 opening witness."""

from __future__ import annotations

import sys

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline.g8_e import E0_PATH, build_e0_opening, rendered_json  # noqa: E402


def main() -> int:
    E0_PATH.parent.mkdir(parents=True, exist_ok=True)
    E0_PATH.write_bytes(rendered_json(build_e0_opening()))
    print(f"PASS: wrote {E0_PATH.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
