#!/usr/bin/env python3
"""Authenticate the frozen G8_C successor and open the G8_D D0 gate.

This command is deliberately data-free: it reads only committed G8_C
closeout metadata and runs the strict successor table loader.  It does not
load a validation sample, construct an authorization, or start a worker.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline.g8_campaign import canonical_json, rendered_json  # noqa: E402
from baseline.g8_pascal_merge import load_successor_bler_table  # noqa: E402
from baseline.g8_pascal_successor import SUCCESSOR_ROOT  # noqa: E402


OUT = REPO / "results/baseline/g8_d/d0_open.json"
TABLE = SUCCESSOR_ROOT / "successor_bler_table.json"
MERGE = SUCCESSOR_ROOT / "successor_bler_merge_report.json"
CLOSEOUT = SUCCESSOR_ROOT / "successor_closeout_provenance.json"


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} is not an object")
    return value, raw


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _artifact_id(value: dict[str, Any]) -> str:
    body = dict(value)
    body.pop("artifact_id", None)
    return "g8dopen-" + _sha256(canonical_json(body))


def build_open_artifact() -> dict[str, Any]:
    table, table_raw = _read(TABLE)
    merge, merge_raw = _read(MERGE)
    closeout, closeout_raw = _read(CLOSEOUT)
    loaded = load_successor_bler_table()
    if len(loaded.identities) != 153:
        raise RuntimeError("strict successor loader did not expose 153 curves")
    if table.get("measured_point_count") != 3213:
        raise RuntimeError("successor table is not the frozen 3213-point table")

    artifact: dict[str, Any] = {
        "schema_version": 1,
        "artifact_role": "g8_d_open_verification",
        "phase": "G8_D",
        "checkpoint": "D0",
        "status": "open",
        "successor_table_path": str(TABLE.relative_to(REPO)),
        "g8_c": {
            "campaign_id": table["campaign_id"],
            "execution_profile_id": table["execution_profile_id"],
            "measurement_source_commit": table["measurement_source_commit"],
            "production_contract_sha256": table["production_contract_sha256"],
            "table_id": table["table_id"],
            "table_sha256": _sha256(table_raw),
            "merge_report_id": merge["report_id"],
            "merge_report_sha256": _sha256(merge_raw),
            "closeout_provenance_id": closeout["closure_id"],
            "closeout_provenance_sha256": _sha256(closeout_raw),
            "curves": table["complete_identity_count"],
            "measured_points": table["measured_point_count"],
            "trials_per_point": table["trials_per_point"],
            "interpolation_used": table["interpolation_used"],
            "imputation_used": table["imputation_used"],
            "extrapolation_used": table["extrapolation_used"],
            "predecessor_table_contribution": table["predecessor_table_contribution"],
        },
        "protected_counters": {
            "inference": 0,
            "training": 0,
            "validation_decoding": 0,
            "test_access": 0,
        },
        "test_access": 0,
        "prohibitions": {
            "validation_campaign_started": False,
            "selection_started": False,
            "pass_one_started": False,
            "pass_two_started": False,
            "training_started": False,
            "test_split_accessed": False,
            "g8_e_started": False,
        },
        "next_gate": "G8_D/D1",
    }
    artifact["artifact_id"] = _artifact_id(artifact)
    return artifact


def main() -> int:
    artifact = build_open_artifact()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(rendered_json(artifact))
    print(json.dumps({"status": "PASS", "artifact": str(OUT.relative_to(REPO)), "artifact_id": artifact["artifact_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
