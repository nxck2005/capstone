#!/usr/bin/env python3
"""Generate or check the additive G8_E corrected-v2 pre-data bundle."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline import g8_e_corrected_v2 as v2  # noqa: E402


def _source_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _files(bundle: dict[str, dict[str, object]]) -> dict[Path, dict[str, object]]:
    return {
        v2.V2_SOURCE_MANIFEST_PATH: bundle["source_manifest"],
        v2.V2_AUTHORITY_BINDING_PATH: bundle["authority_binding"],
        v2.V2_MAPPING_BINDING_PATH: bundle["mapping_binding"],
        v2.V2_CORRECTION_PATH: bundle["correction_provenance"],
        v2.V2_STORAGE_PLAN_PATH: bundle["storage_plan"],
        v2.V2_CONTRACT_PATH: bundle["measurement_contract"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    try:
        source_commit = args.source_commit or _source_commit()
        bundle = v2.build_bundle(source_commit)
        files = _files(bundle)
        if args.check:
            for path, value in files.items():
                if not path.is_file() or path.read_bytes() != v2.rendered_json(value):
                    raise v2.G8EV2Error(f"stale or missing v2 artifact: {path.relative_to(REPO)}")
            print("g8_e_corrected_v2: exact bundle check passed")
            return 0
        v2.V2_ROOT.mkdir(parents=True, exist_ok=True)
        for path, value in files.items():
            path.write_bytes(v2.rendered_json(value))
        print({
            "status": "GENERATED_PRE_DATA",
            "root": str(v2.V2_ROOT.relative_to(REPO)),
            "contract_id": bundle["measurement_contract"]["contract_id"],
            "campaign_id": bundle["measurement_contract"]["campaign_id"],
            "source_manifest_id": bundle["source_manifest"]["source_manifest_id"],
        })
    except (OSError, subprocess.SubprocessError, v2.G8EV2Error) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
