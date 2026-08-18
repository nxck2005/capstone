#!/usr/bin/env python3
"""Generate or check the additive corrected G8_E E1 pre-data epoch."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline import g8_e_corrected as corrected  # noqa: E402


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write(bundle: dict[str, dict]) -> None:
    corrected.CORRECTED_ROOT.mkdir(parents=True, exist_ok=True)
    names = {
        "measurement_authority": corrected.CORRECTED_AUTHORITY_PATH,
        "logical_measurement_mapping": corrected.CORRECTED_MAPPING_PATH,
        "execution_source_manifest": corrected.CORRECTED_SOURCE_MANIFEST_PATH,
        "correction_provenance": corrected.CORRECTION_PROVENANCE_PATH,
        "measurement_contract": corrected.CORRECTED_CONTRACT_PATH,
    }
    for name, path in names.items():
        path.write_bytes(corrected.rendered_json(bundle[name]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="rebuild in memory and compare exact tracked bytes")
    parser.add_argument("--source-commit", help="source commit to bind; defaults to HEAD")
    args = parser.parse_args()
    try:
        bundle = corrected.build_corrected_bundle(args.source_commit or _head())
        paths = {
            "measurement_authority": corrected.CORRECTED_AUTHORITY_PATH,
            "logical_measurement_mapping": corrected.CORRECTED_MAPPING_PATH,
            "execution_source_manifest": corrected.CORRECTED_SOURCE_MANIFEST_PATH,
            "correction_provenance": corrected.CORRECTION_PROVENANCE_PATH,
            "measurement_contract": corrected.CORRECTED_CONTRACT_PATH,
        }
        if args.check:
            for name, path in paths.items():
                expected = corrected.rendered_json(bundle[name])
                if not path.is_file() or path.read_bytes() != expected:
                    raise corrected.CorrectedG8EError(f"corrected artifact is stale: {path}")
            print("PASS: corrected G8_E E1 artifacts are exact")
        else:
            _write(bundle)
            print({
                "status": "WROTE_PRE_DATA_CORRECTED_E1",
                "campaign_id": bundle["measurement_contract"]["campaign_id"],
                "contract_id": bundle["measurement_contract"]["contract_id"],
                "authority_id": bundle["measurement_authority"]["authority_id"],
                "mapping_id": bundle["logical_measurement_mapping"]["mapping_id"],
                "source_manifest_id": bundle["execution_source_manifest"]["source_manifest_id"],
                "measurement_coverage": 0,
            })
    except (OSError, subprocess.CalledProcessError, corrected.CorrectedG8EError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
