#!/usr/bin/env python3
"""Verify and compare synthetic OpenJPEG parity reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from config.execution_profiles import canonical_json_bytes  # noqa: E402


class OpenJpegParityError(RuntimeError):
    pass


def read(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise OpenJpegParityError(f"cannot read OpenJPEG report: {exc}") from exc
    if not isinstance(report, dict):
        raise OpenJpegParityError("OpenJPEG report is not an object")
    return report


def verify(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != 1 or report.get("artifact_kind") != "execution_profile_openjpeg_parity":
        raise OpenJpegParityError("unsupported OpenJPEG parity schema")
    if report.get("scientific_status") != "NON-SCIENTIFIC" or report.get("g8_coverage") != 0:
        raise OpenJpegParityError("OpenJPEG parity is not zero-coverage")
    if report.get("openjpeg_version") != "2.5.4":
        raise OpenJpegParityError("OpenJPEG version is not the configured 2.5.4")
    for key in ("test_access", "validation_decoding", "training"):
        if report.get(key) != 0:
            raise OpenJpegParityError(f"OpenJPEG parity {key} is nonzero")
    digest = report.get("report_sha256")
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    if digest != hashlib.sha256(canonical_json_bytes(body)).hexdigest():
        raise OpenJpegParityError("OpenJPEG report digest differs")
    cells = report.get("cells")
    if not isinstance(cells, list) or not cells:
        raise OpenJpegParityError("OpenJPEG report has no cells")
    for cell in cells:
        if not isinstance(cell, Mapping) or int(cell.get("emitted_codestream_bytes", 0)) <= 0:
            raise OpenJpegParityError("OpenJPEG cell is malformed")
        for key in ("codestream_sha256", "decoded_pixels_sha256"):
            if not isinstance(cell.get(key), str) or len(cell[key]) != 64:
                raise OpenJpegParityError(f"OpenJPEG {key} is missing")


def compare(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    verify(left)
    verify(right)
    for key in ("fixture_shape", "fixture_pixels_sha256", "codec_configuration"):
        if left.get(key) != right.get(key):
            raise OpenJpegParityError(f"OpenJPEG fixture/configuration differs: {key}")
    left_cells = left["cells"]
    right_cells = right["cells"]
    if [cell["compression_ratio"] for cell in left_cells] != [cell["compression_ratio"] for cell in right_cells]:
        raise OpenJpegParityError("OpenJPEG compression-ratio order differs")
    cells = []
    for a, b in zip(left_cells, right_cells, strict=True):
        cells.append(
            {
                "compression_ratio": a["compression_ratio"],
                "codestream_bytes_equal": a["emitted_codestream_bytes"] == b["emitted_codestream_bytes"],
                "codestream_sha256_equal": a["codestream_sha256"] == b["codestream_sha256"],
                "decoded_pixels_sha256_equal": a["decoded_pixels_sha256"] == b["decoded_pixels_sha256"],
                "left_bytes": a["emitted_codestream_bytes"],
                "right_bytes": b["emitted_codestream_bytes"],
            }
        )
    passed = all(all(value for key, value in cell.items() if key.endswith("_equal")) for cell in cells)
    return {
        "schema_version": 1,
        "artifact_kind": "execution_profile_openjpeg_parity_comparison",
        "scientific_status": "NON-SCIENTIFIC",
        "left_profile": left["execution_profile_id"],
        "right_profile": right["execution_profile_id"],
        "cells": cells,
        "codestream_and_decoded_pixels_equal": passed,
        "g8_coverage": 0,
        "test_access": 0,
        "validation_decoding": 0,
        "training": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(read(args.left), read(args.right))
    result["comparison_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    args.output.write_bytes(canonical_json_bytes(result))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["codestream_and_decoded_pixels_equal"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OpenJpegParityError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
