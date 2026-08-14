#!/usr/bin/env python3
"""Compare the frozen G8 stimulus streams without executing a G8 decoder.

This is a qualification-only audit.  It reads the old required-identity grid
and the preregistered parity plan, generates the exact information/noise arrays
from the frozen contract, and records hashes of their bytes.  No request,
result, state, dataset or test artifact is opened or written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from baseline.g8_bler_contract import (  # noqa: E402
    PURPOSE_AWGN_IMAG,
    PURPOSE_AWGN_REAL,
    PURPOSE_INFORMATION_BITS,
    derive_seed,
    information_bit_stream,
    normal_stream,
)
from baseline.g8_campaign import (  # noqa: E402
    load_campaign_manifest,
    load_required_bler_identities,
    rendered_json,
    sha256_file,
)
from baseline.ldpc.modulation import bits_per_symbol  # noqa: E402
from config.execution_profiles import canonical_json_bytes  # noqa: E402
from config.params import get  # noqa: E402

PARITY_PLAN = REPO / "results/baseline/g8/execution_profile_parity_plan.json"


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def audit(profile_id: str, *, trials: int) -> dict[str, Any]:
    if trials <= 0:
        raise ValueError("trial count must be positive")
    profile = get(f"environment.execution_profiles.{profile_id}")
    if not isinstance(profile, dict):
        raise ValueError(f"unknown execution profile: {profile_id}")
    if str(profile["numpy_version"]) != np.__version__:
        raise RuntimeError(
            f"NumPy {np.__version__} does not match {profile_id} profile {profile['numpy_version']}"
        )
    parity_raw = PARITY_PLAN.read_bytes()
    parity = json.loads(parity_raw)
    if parity_raw != rendered_json(parity) or parity.get("scientific_status") != "NON-SCIENTIFIC":
        raise RuntimeError("parity plan is not canonical non-scientific evidence")
    manifest = load_campaign_manifest()
    required = load_required_bler_identities()
    units = required["required_bler_work_units"]
    bindings = parity["selected_identity_bindings"]
    expected_ordinals = parity["selected_identity_ordinals"]
    if [item["ordinal"] for item in bindings] != expected_ordinals:
        raise RuntimeError("parity plan identity bindings are not the frozen ordered prefix")

    cells: list[dict[str, Any]] = []
    for binding in bindings:
        ordinal = int(binding["ordinal"])
        unit = units[ordinal]
        if binding["work_unit_id"] != unit["work_unit_id"] or binding["identity"] != unit["identity"] or binding["snr_db"] != unit["snr_db"]:
            raise RuntimeError(f"parity plan binding differs from required identity {ordinal}")
        identity = unit["identity"]
        k, n = (int(value) for value in identity["k_and_n"])
        symbols = n // bits_per_symbol(identity["modulation"])
        streams = {}
        for purpose, count, dtype in (
            (PURPOSE_INFORMATION_BITS, trials * k, "uint8"),
            (PURPOSE_AWGN_REAL, trials * symbols, "float64"),
            (PURPOSE_AWGN_IMAG, trials * symbols, "float64"),
        ):
            seed = derive_seed(manifest["campaign_id"], unit["work_unit_id"], purpose)
            if purpose == PURPOSE_INFORMATION_BITS:
                values = information_bit_stream(seed, 0, count)
            else:
                values = normal_stream(seed, count)
            values = np.ascontiguousarray(values)
            streams[purpose] = {
                "seed_uint64": seed,
                "count": count,
                "shape": [trials, k] if purpose == PURPOSE_INFORMATION_BITS else [trials, symbols],
                "dtype": dtype,
                "sha256": _array_sha256(values),
            }
        cells.append(
            {
                "ordinal": ordinal,
                "work_unit_id": unit["work_unit_id"],
                "identity": identity,
                "snr_db": unit["snr_db"],
                "stratum": binding["stratum"],
                "streams": streams,
            }
        )
    report = {
        "schema_version": 1,
        "artifact_kind": "execution_profile_rng_audit",
        "scientific_status": "NON-SCIENTIFIC",
        "execution_profile_id": profile_id,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "campaign_id": manifest["campaign_id"],
        "required_identity_sha256": sha256_file(REPO / "results/baseline/g8/required_bler_identities.json"),
        "parity_plan_sha256": hashlib.sha256(parity_raw).hexdigest(),
        "trials_per_selected_identity": trials,
        "stream_contract": "g8_bler_contract seed_material/derive_seed/information_bit_stream/normal_stream",
        "g8_coverage": 0,
        "test_access": 0,
        "validation_decoding": 0,
        "training": 0,
        "cells": cells,
    }
    report["report_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--trials", type=int, default=512)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.profile, trials=args.trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report))
    print(json.dumps({"profile": args.profile, "report_sha256": report["report_sha256"], "cells": len(report["cells"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
