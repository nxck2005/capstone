#!/usr/bin/env python3
"""Compare deterministic synthetic JPEG-2000 output across profiles.

The fixture is project-owned synthetic RGB pixels, not a dataset or test
image.  This checks the host-side codec dependency independently of GPU
qualification and records both codestream and decoded-pixel hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

import glymur
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from config.execution_profiles import canonical_json_bytes  # noqa: E402
from config.params import get  # noqa: E402
from env import loaded_openjpeg_version  # noqa: E402

FIXTURE_SHAPE = (64, 64, 3)
COMPRESSION_RATIOS = (8.0, 16.0)


def _fixture() -> np.ndarray:
    values = np.arange(np.prod(FIXTURE_SHAPE), dtype=np.uint32).reshape(FIXTURE_SHAPE)
    return (values % 251).astype(np.uint8)


def _encode(image: np.ndarray, ratio: float) -> tuple[bytes, np.ndarray]:
    with tempfile.TemporaryDirectory(prefix="capstone-openjpeg-parity-") as directory:
        path = Path(directory) / "fixture.j2k"
        glymur.Jp2k(
            path,
            data=image,
            cratios=(ratio,),
            irreversible=True,
            prog=str(get("baseline.j2k_progression_order")),
            numres=int(get("baseline.j2k_resolutions")),
            cbsize=tuple(get("baseline.j2k_code_block_size")),
            tilesize=FIXTURE_SHAPE[:2],
        )
        codestream = path.read_bytes()
        decoded = np.asarray(glymur.Jp2k(path)[:])
    return codestream, decoded


def audit(profile_id: str) -> dict[str, Any]:
    openjpeg = loaded_openjpeg_version(required=True)
    image = _fixture()
    fixture_sha = hashlib.sha256(image.tobytes()).hexdigest()
    cells = []
    for ratio in COMPRESSION_RATIOS:
        codestream, decoded = _encode(image, ratio)
        cells.append(
            {
                "compression_ratio": ratio,
                "requested_payload_budget_bytes": len(codestream),
                "emitted_codestream_bytes": len(codestream),
                "codestream_sha256": hashlib.sha256(codestream).hexdigest(),
                "decoded_shape": list(decoded.shape),
                "decoded_dtype": str(decoded.dtype),
                "decoded_pixels_sha256": hashlib.sha256(decoded.tobytes()).hexdigest(),
            }
        )
    report = {
        "schema_version": 1,
        "artifact_kind": "execution_profile_openjpeg_parity",
        "scientific_status": "NON-SCIENTIFIC",
        "execution_profile_id": profile_id,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "glymur_version": glymur.__version__,
        "openjpeg_version": openjpeg,
        "fixture_shape": list(FIXTURE_SHAPE),
        "fixture_pixels_sha256": fixture_sha,
        "codec_configuration": {
            "progression_order": get("baseline.j2k_progression_order"),
            "resolutions": get("baseline.j2k_resolutions"),
            "code_block_size": list(get("baseline.j2k_code_block_size")),
            "irreversible": True,
            "raw_codestream": True,
        },
        "cells": cells,
        "g8_coverage": 0,
        "test_access": 0,
        "validation_decoding": 0,
        "training": 0,
    }
    report["report_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report))
    print(json.dumps({"profile": args.profile, "report_sha256": report["report_sha256"], "cells": len(report["cells"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
