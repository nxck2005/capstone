#!/usr/bin/env python3
"""Owner-gated corrected-v3 worker-successor closeout entry point.

Additive corrective layer for E2-completion verification, E3 and E4: it runs
the complete frozen v3s chain with the full scientific-data-identity FILE
loading rule that the production runner pre-registered.  Production use must
target the frozen epoch paths; synthetic fixtures are driven through the
module API in tests only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from baseline import g8_e_corrected_v3 as v3  # noqa: E402
from baseline import g8_e_corrected_v3s as v3s  # noqa: E402
from baseline import g8_e_v3s_closeout as closeout  # noqa: E402

PROVENANCE_PATH = v3s.V3S_ROOT / "closeout_repair_provenance.json"
REPAIR_FILES = (
    ("src/baseline/g8_e_v3s_closeout.py", "corrected_closeout_lifecycle_module"),
    ("tools/closeout_g8_e_v3s.py", "corrected_closeout_cli_and_provenance_generator"),
    ("tests/test_g8_e_v3s_closeout.py", "closeout_repair_regression_tests"),
)


def _run_verify(args: argparse.Namespace) -> int:
    kwargs = {
        "verify_live_sources": not args.skip_live_sources,
        "verify_live_data": not args.skip_live_data,
    }
    try:
        if args.phase == "e2":
            result = closeout.verify_e2_complete(**kwargs)
        elif args.phase == "e3":
            result = closeout.verify_e3_complete(e3_sha256=args.e3_sha256 or None, **kwargs)
        else:
            result = closeout.verify_e4_complete(e3_sha256=args.e3_sha256 or None, **kwargs)
    except (OSError, v3.G8EV3Error, v3s.G8EV3SError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    contract = result["contract"]
    body = {
        "status": "PASS",
        "phase": result["phase"],
        "contract_id": contract["contract_id"],
        "campaign_id": contract["campaign_id"],
        "profile_id": contract["execution_profile"]["profile_id"],
        "device": contract["execution_profile"]["device"],
        "work_units": contract["transaction"]["production_total_required"],
        "coverage": contract["safety"]["measurement_coverage"],
    }
    for key in ("completion_sha256", "e3_sha256", "e4_sha256"):
        if key in result:
            body[key] = result[key]
    print(json.dumps(body, sort_keys=True))
    return 0


def _run_merge(args: argparse.Namespace) -> int:
    if not args.execute:
        print("REFUSED: E3 requires --execute after exact E2 completion", file=sys.stderr)
        return 2
    try:
        value, path, digest = closeout.publish_e3()
    except (OSError, v3.G8EV3Error, v3s.G8EV3SError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2
    print({
        "status": value["status"],
        "e3_id": value["e3_id"],
        "sha256": digest,
        "required": value["required_work_unit_count"],
        "observed": value["observed_work_unit_count"],
        "missing": value["missing_count"],
        "duplicate": value["duplicate_count"],
        "extra": value["extra_count"],
        "digest": value["ordered_record_sha256"],
        "output": str(path),
    })
    return 0


def _run_aggregate(args: argparse.Namespace) -> int:
    if not args.execute:
        print("REFUSED: E4 requires --execute and an exact E3 SHA-256", file=sys.stderr)
        return 2
    try:
        value, path, digest = closeout.publish_e4(e3_sha256=args.e3_sha256)
    except (OSError, v3.G8EV3Error, v3s.G8EV3SError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2
    print({
        "status": value["status"],
        "e4_id": value["e4_id"],
        "sha256": digest,
        "e3_id": value["e3_id"],
        "object_count": value["object_count"],
        "record_traversal_count": value["record_traversal_count"],
        "output": str(path),
    })
    return 0


def build_repair_provenance() -> dict:
    contract_raw = v3s.V3S_CONTRACT_PATH.read_bytes()
    contract, _ = v3._rendered_object(v3s.V3S_CONTRACT_PATH, "v3s measurement contract")
    authorization, _ = v3._rendered_object(v3s.V3S_AUTHORIZATION_PATH, "v3s owner authorization")
    body = {
        "schema_version": 1,
        "artifact_role": "g8_e_v3s_closeout_repair_provenance",
        "classification": "IMPLEMENTATION_DEFECT_CLOSEOUT_LAYER_ONLY",
        "scientific_defect": "NONE",
        "recomputation_required": False,
        "measurement_records_touched": False,
        "discovered_at": "2026-08-22",
        "symptom": (
            "verify_g8_e_corrected_v3s.py --phase e2 raised KeyError('manifest_bytes') at "
            "g8_e_corrected_v3.frozen_validation_metadata via "
            "g8_e_corrected_v3s.verify_active_e2, which passes the contract's summary block"
        ),
        "defective_sites": [
            {"path": "src/baseline/g8_e_corrected_v3s.py", "symbol": "verify_active_e2", "detail": "summary block passed to frozen_validation_metadata"},
            {"path": "tools/merge_g8_e_corrected_v3s.py", "branch": "production", "detail": "summary block passed to frozen_validation_metadata"},
            {"path": "tools/aggregate_g8_e_corrected_v3s.py", "branch": "production", "detail": "summary block passed to verify_live_validation_identity"},
        ],
        "pre_registered_rule": [
            "tools/run_g8_e_corrected_v3s.py loads results/baseline/g8_e/e1_corrected_v3/scientific_data_identity_manifest.json through the contract-bound path before frozen_validation_metadata/verify_live_validation_identity",
            "tests/test_g8_e_e2_successor.py::test_live_identity_check_rejects_the_contract_summary_block pins that the full data identity file is the authenticator",
        ],
        "repair": {
            "mode": "ADDITIVE_ONLY_NO_BOUND_BYTE_CHANGED",
            "files": [
                {"path": path, "role": role, "bytes": len((v3.REPO_ROOT / path).read_bytes()), "sha256": v3.sha256_file(v3.REPO_ROOT / path)}
                for path, role in REPAIR_FILES
            ],
        },
        "unchanged_frozen_bindings": {
            "contract_path": str(v3s.V3S_CONTRACT_PATH.relative_to(v3.REPO_ROOT)),
            "contract_sha256": v3.sha256_bytes(contract_raw),
            "contract_id": contract["contract_id"],
            "campaign_id": contract["campaign_id"],
            "source_commit": contract["source_manifest"]["source_commit"],
            "authorization_issued_sha256": authorization["issued_sha256"],
        },
        "scope": "E2 completion verification, E3 and E4 only; no E5/pass-one, training, fallback, ratio adjudication or test access is authorized by this repair",
    }
    body["provenance_id"] = v3._id("g8ecloseoutrepair-", {key: child for key, child in body.items() if key != "provenance_id"})
    return body


def _run_provenance(args: argparse.Namespace) -> int:
    try:
        body = build_repair_provenance()
        raw = v3.rendered_json(body)
        if args.check:
            if not PROVENANCE_PATH.is_file():
                print("FAIL: closeout repair provenance artifact is absent", file=sys.stderr)
                return 2
            stored = PROVENANCE_PATH.read_bytes()
            if stored != raw:
                print("FAIL: closeout repair provenance does not reproduce from live inputs", file=sys.stderr)
                return 2
            print({"status": "PASS", "provenance_id": body["provenance_id"], "sha256": v3.sha256_bytes(stored)})
            return 0
        if PROVENANCE_PATH.exists():
            print("REFUSED: closeout repair provenance already exists", file=sys.stderr)
            return 2
        v3._atomic_publish(PROVENANCE_PATH, raw)
        print({"status": "WRITTEN", "provenance_id": body["provenance_id"], "sha256": v3.sha256_bytes(raw), "output": str(PROVENANCE_PATH)})
        return 0
    except (OSError, v3.G8EV3Error, v3s.G8EV3SError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify", help="verify a lifecycle phase with corrected loading")
    verify.add_argument("--phase", choices=("e2", "e3", "e4"), required=True)
    verify.add_argument("--e3-sha256")
    verify.add_argument("--skip-live-sources", action="store_true")
    verify.add_argument("--skip-live-data", action="store_true")
    verify.set_defaults(func=_run_verify)

    merge = subparsers.add_parser("merge", help="publish the E3 exact-set closure")
    merge.add_argument("--execute", action="store_true")
    merge.set_defaults(func=_run_merge)

    aggregate = subparsers.add_parser("aggregate", help="publish E4 count-derived objects from one exact E3 SHA-256")
    aggregate.add_argument("--execute", action="store_true")
    aggregate.add_argument("--e3-sha256", required=True)
    aggregate.set_defaults(func=_run_aggregate)

    provenance = subparsers.add_parser("provenance", help="write or check the closeout repair provenance artifact")
    group = provenance.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    provenance.set_defaults(func=_run_provenance)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
