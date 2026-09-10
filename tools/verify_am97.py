#!/usr/bin/env python3
"""Verify the prospective AM-97 semantic freeze in an explicit lifecycle mode."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.am97_spec_compatibility import load


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downstream", action="store_true", help="permit later result namespaces while authenticating the freeze")
    args = parser.parse_args(argv)
    value = load(Path(__file__).resolve().parents[1], allow_downstream=args.downstream)
    mode = "downstream/history" if args.downstream else "pre-science"
    print(f"AM-97 {mode} verification PASS: {value['freeze_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
