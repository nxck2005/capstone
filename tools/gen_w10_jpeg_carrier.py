#!/usr/bin/env python3
"""Build or verify the lossless tracked carrier for JPEG-secondary evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evaluation.w10_jpeg_carrier import (  # noqa: E402
    JPEG_CARRIER_DESCRIPTOR_PATH,
    JPEG_CARRIER_PATH,
    JPEG_SELECTION_PATH,
    JpegCarrierHold,
    build_jpeg_carrier,
    check_jpeg_carrier,
)


def _print_result(label: str, loaded) -> None:
    provenance = loaded.provenance
    print(
        f"{label}: selection_id={provenance['selection_id']} "
        f"raw_bytes={provenance['raw_bytes']} raw_sha256={provenance['raw_sha256']} "
        f"carrier_bytes={provenance['carrier_bytes']} carrier_sha256={provenance['carrier_sha256']} "
        f"descriptor_id={provenance['carrier_descriptor_id']} "
        f"descriptor_sha256={provenance['carrier_descriptor_sha256']}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the tracked carrier without writing")
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=None,
        help=f"exact raw source bytes (default: {JPEG_SELECTION_PATH})",
    )
    args = parser.parse_args(argv)
    try:
        loaded = check_jpeg_carrier(REPO) if args.check else build_jpeg_carrier(REPO, raw_path=args.raw_path)
    except JpegCarrierHold as exc:
        print(f"HOLD — {exc}", file=sys.stderr)
        return 2
    _print_result("JPEG carrier check PASS" if args.check else "JPEG carrier build PASS", loaded)
    print(f"carrier_path={JPEG_CARRIER_PATH} descriptor_path={JPEG_CARRIER_DESCRIPTOR_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
