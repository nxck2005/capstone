#!/usr/bin/env python3
"""Freeze the already-completed Pascal successor G8_C closeout artifacts.

This command is a deterministic artifact writer.  It consumes the committed
successor runtime and never calls a measurement runner.  The independent
checker is ``tools/verify_g8_pascal_closeout.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline.g8_campaign import canonical_json, rendered_json, sha256_bytes  # noqa: E402
from baseline.g8_pascal_merge import (  # noqa: E402
    MERGE_REPORT_PATH,
    MEASUREMENT_SOURCE_COMMIT,
    PROVENANCE_PATH,
    TABLE_PATH,
    SuccessorMergeError,
    build_successor_bler_table,
    build_successor_merge_report,
    closeout_source_digest,
    closeout_source_entries,
    load_required_authority,
)
from baseline.g8_pascal_production import successor_bindings, validate_production_contracts  # noqa: E402
from baseline.g8_pascal_successor import SUCCESSOR_PROFILE_ID, SUCCESSOR_ROOT  # noqa: E402


PROVENANCE_SCHEMA_VERSION = 1
PROVENANCE_ARTIFACT_ROLE = "g8_c_pascal_successor_closeout_provenance"
PROVENANCE_ID_PREFIX = "g8pcloseout-"


def _self_id(payload: dict) -> str:
    body = dict(payload)
    body.pop("closure_id", None)
    return PROVENANCE_ID_PREFIX + sha256_bytes(canonical_json(body))


def _write(path: Path, payload: dict) -> tuple[int, str]:
    body = rendered_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return len(body), sha256_bytes(body)


def build_provenance(merge: dict, merge_raw: bytes, table: dict, table_raw: bytes) -> dict:
    bindings = successor_bindings()
    production = validate_production_contracts()
    authority_units, authority_set_digest, authority_file_digest = load_required_authority()
    source_entries = closeout_source_entries()
    source_digest = closeout_source_digest()
    if merge["closeout_source_digest"] != source_digest or table["closeout_source_digest"] != source_digest:
        raise SuccessorMergeError("closeout source digest changed between artifact construction and provenance closure")
    merge_sha = sha256_bytes(merge_raw)
    table_sha = sha256_bytes(table_raw)
    payload = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "artifact_role": PROVENANCE_ARTIFACT_ROLE,
        "phase": "G8_C",
        "checkpoint": "C6",
        "closure_id": None,
        "campaign_id": bindings["campaign_id"],
        "execution_profile_id": SUCCESSOR_PROFILE_ID,
        "measurement_source": {
            "role": "authenticated_production_measurement_source",
            "commit": MEASUREMENT_SOURCE_COMMIT,
            "production_contract_sha256": production["production_contract_sha256"],
            "production_source_manifest_sha256": production["production_source_manifest_sha256"],
            "production_runner_contract_sha256": production["production_runner_contract_sha256"],
            "execution_profile_id": SUCCESSOR_PROFILE_ID,
            "runtime_relative_path": merge["runtime_relative_path"],
            "runtime_tree_sha256": merge["runtime_tree_sha256"],
            "scientific_execution_performed": True,
        },
        "closeout_source": {
            "role": "deterministic_post_measurement_merge_and_table_consumer",
            "source_digest": source_digest,
            "sources": source_entries,
            "scientific_execution_performed": False,
            "measurement_source_commit_is_not_closeout_commit": True,
            "closeout_commit_is_resolved_by_git_publication": True,
        },
        "authority": {
            "path": "results/baseline/g8/required_bler_identities.json",
            "bytes": len((REPO / "results/baseline/g8/required_bler_identities.json").read_bytes()),
            "sha256": authority_file_digest,
            "identity_count": len(authority_units),
            "ordered_identity_set_sha256": authority_set_digest,
        },
        "artifacts": {
            "merge_report": {
                "path": str(MERGE_REPORT_PATH.relative_to(REPO)),
                "bytes": len(merge_raw),
                "sha256": merge_sha,
                "report_id": merge["report_id"],
            },
            "bler_table": {
                "path": str(TABLE_PATH.relative_to(REPO)),
                "bytes": len(table_raw),
                "sha256": table_sha,
                "table_id": table["table_id"],
            },
        },
        "retry_history": {
            "preserved": True,
            "historical_failed_attempts_contribute": 0,
            "accepted_attempts_contribute": 1,
            "retry_ordinals": merge["retry_history_ordinals"],
            "request_only_attempt_count": merge["request_only_attempt_count"],
            "failed_result_attempt_count": merge["failed_result_attempt_count"],
        },
        "safety": {
            "required_identity_count": merge["required_identity_count"],
            "trials_per_identity": table["trials_per_point"],
            "test_access": 0,
            "protected_counters": merge["protected_counters"],
            "old_result_ingest": False,
            "interpolation_used": False,
            "imputation_used": False,
            "extrapolation_used": False,
        },
        "predecessor_isolation": {
            "predecessor_campaign_id": merge["predecessor_campaign_id"],
            "predecessor_table_contribution": "none",
            "successor_campaign_id": merge["campaign_id"],
            "old_rtx4060_profile_id": "local_4060_cu130",
        },
        "mutation_checks": {
            "source_bytes_rehashed": True,
            "merge_report_id_recomputed": True,
            "table_id_recomputed": True,
            "request_result_state_hashes_rechecked": True,
            "runtime_tree_hash_rechecked": True,
        },
    }
    payload["closure_id"] = _self_id(payload)
    return payload


def freeze(root: Path) -> dict[str, object]:
    merge = build_successor_merge_report(root)
    merge_raw = rendered_json(merge)
    MERGE_REPORT_PATH.write_bytes(merge_raw)
    merge_sha = sha256_bytes(merge_raw)
    table = build_successor_bler_table(merge, merge_report_sha256=merge_sha)
    table_raw = rendered_json(table)
    TABLE_PATH.write_bytes(table_raw)
    provenance = build_provenance(merge, merge_raw, table, table_raw)
    provenance_raw = rendered_json(provenance)
    PROVENANCE_PATH.write_bytes(provenance_raw)
    return {
        "merge_report": str(MERGE_REPORT_PATH.relative_to(REPO)),
        "merge_report_sha256": merge_sha,
        "merge_report_id": merge["report_id"],
        "table": str(TABLE_PATH.relative_to(REPO)),
        "table_sha256": sha256_bytes(table_raw),
        "table_id": table["table_id"],
        "provenance": str(PROVENANCE_PATH.relative_to(REPO)),
        "provenance_sha256": sha256_bytes(provenance_raw),
        "provenance_id": provenance["closure_id"],
        "curves": table["complete_identity_count"],
        "points": table["measured_point_count"],
        "total_trials": table["total_trials"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=SUCCESSOR_ROOT / "runtime")
    parser.add_argument("--write", action="store_true", help="write the three frozen closeout artifacts")
    args = parser.parse_args()
    root = args.root.resolve()
    if not args.write:
        merge = build_successor_merge_report(root)
        print(json.dumps({"status": "PASS", "report_id": merge["report_id"], "records": len(merge["units"])}, sort_keys=True))
        return 0
    print(json.dumps(freeze(root), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SuccessorMergeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
