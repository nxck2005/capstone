#!/usr/bin/env python3
"""Generate or check the G8_E E1 pre-data validation freeze."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline.g8_e import (  # noqa: E402
    E1_AUTHORITY_PATH,
    E1_CONTRACT_PATH,
    E1_CORPUS_SPEC_PATH,
    E1_SOURCE_MANIFEST_PATH,
    G8EContractError,
    build_e1_bundle,
    rendered_json,
    sha256_bytes,
    verify_e1_contract_file,
    verify_e1_corpus_spec_file,
    verify_e1_authority_file,
    verify_e1_source_manifest_file,
)


def _artifacts() -> tuple[Path, ...]:
    return (E1_AUTHORITY_PATH, E1_SOURCE_MANIFEST_PATH, E1_CORPUS_SPEC_PATH, E1_CONTRACT_PATH)


def _write() -> int:
    existing = [path for path in _artifacts() if path.exists()]
    if existing:
        raise G8EContractError(
            "refusing to overwrite existing E1 artifacts: "
            + ", ".join(str(path.relative_to(REPO)) for path in existing)
        )
    bundle = build_e1_bundle()
    for path, key in (
        (E1_AUTHORITY_PATH, "authority"),
        (E1_SOURCE_MANIFEST_PATH, "source_manifest"),
        (E1_CORPUS_SPEC_PATH, "corpus_spec"),
        (E1_CONTRACT_PATH, "contract"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(rendered_json(bundle[key]))
    # Verify the newly published bytes before reporting the checkpoint green.
    verify_e1_authority_file()
    verify_e1_source_manifest_file(expected_campaign_id=bundle["contract"]["campaign_id"])
    verify_e1_corpus_spec_file()
    verify_e1_contract_file(verify_live_assets=False, verify_live_profile=False)
    print("PASS: G8_E E1 pre-data artifacts written")
    for path in _artifacts():
        print(f"  {path.relative_to(REPO)} sha256={sha256_bytes(path.read_bytes())}")
    print(f"  campaign_id={bundle['contract']['campaign_id']}")
    print(f"  contract_id={bundle['contract']['contract_id']}")
    return 0


def _check() -> int:
    contract = verify_e1_contract_file(verify_live_assets=False, verify_live_profile=False)
    verify_e1_authority_file()
    verify_e1_source_manifest_file(expected_campaign_id=contract["campaign_id"])
    verify_e1_corpus_spec_file()
    print(
        {
            "status": "PASS",
            "contract_id": contract["contract_id"],
            "campaign_id": contract["campaign_id"],
            "measurement_coverage": contract["safety"]["measurement_coverage"],
            "authorization_issued": contract["pass_one_preconditions"]["authorization_issued"],
        }
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write the unopened E1 artifacts")
    mode.add_argument("--check", action="store_true", help="verify existing E1 artifacts")
    args = parser.parse_args()
    try:
        return _write() if args.write else _check()
    except G8EContractError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
