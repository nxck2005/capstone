#!/usr/bin/env python3
"""Read-only verifier for the active pre-execution W9-A/G-10 authority."""

from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evaluation.g10_protocol import (  # noqa: E402
    AUTHORIZATION_PATH,
    G10ProtocolHold,
    verify_authorization,
)


def main() -> int:
    try:
        value = verify_authorization(REPO / AUTHORIZATION_PATH, root=REPO)
    except (G10ProtocolHold, OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"G-10 authority VERIFY HOLD — {exc}", file=sys.stderr)
        return 1
    print(
        "G-10 pre-execution authority verification PASS: "
        f"{value['authorization_id']} g10_model_facing_evaluations=0 test=SEALED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
