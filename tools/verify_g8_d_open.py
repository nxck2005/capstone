#!/usr/bin/env python3
"""Independently verify the data-free G8_D D0 opening artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from baseline.g8_pascal_merge import load_successor_bler_table  # noqa: E402
from baseline.g8_pascal_successor import SUCCESSOR_ROOT  # noqa: E402


class G8DOpenVerificationError(RuntimeError):
    pass


OUT = REPO / "results/baseline/g8_d/d0_open.json"
TABLE = SUCCESSOR_ROOT / "successor_bler_table.json"
MERGE = SUCCESSOR_ROOT / "successor_bler_merge_report.json"
CLOSEOUT = SUCCESSOR_ROOT / "successor_closeout_provenance.json"


def _rendered(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise G8DOpenVerificationError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict) or raw != _rendered(value):
        raise G8DOpenVerificationError(f"{label} is not canonical rendered JSON")
    return value, raw


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G8DOpenVerificationError(message)


def validate(path: Path = OUT) -> dict[str, Any]:
    artifact, raw = _read(path, "D0 artifact")
    _require(set(artifact) == {
        "schema_version", "artifact_role", "phase", "checkpoint", "status", "artifact_id",
        "successor_table_path", "g8_c", "protected_counters", "test_access", "prohibitions", "next_gate",
    }, "D0 artifact schema differs")
    body = dict(artifact)
    artifact_id = body.pop("artifact_id")
    _require(isinstance(artifact_id, str) and artifact_id == "g8dopen-" + _sha(_canonical(body)), "D0 artifact ID differs")
    _require(artifact["schema_version"] == 1 and artifact["artifact_role"] == "g8_d_open_verification", "D0 artifact role differs")
    _require((artifact["phase"], artifact["checkpoint"], artifact["status"]) == ("G8_D", "D0", "open"), "D0 status differs")
    _require(artifact["successor_table_path"] == str(TABLE.relative_to(REPO)), "D0 table path differs")

    table, table_raw = _read(TABLE, "successor table")
    merge, merge_raw = _read(MERGE, "successor merge")
    closeout, closeout_raw = _read(CLOSEOUT, "successor closeout")
    loaded = load_successor_bler_table()
    _require(len(loaded.identities) == 153, "strict successor loader curve count differs")
    _require(table["measured_point_count"] == 3213, "successor measured-point count differs")
    g8c = artifact["g8_c"]
    expected = {
        "campaign_id": table["campaign_id"],
        "execution_profile_id": table["execution_profile_id"],
        "measurement_source_commit": table["measurement_source_commit"],
        "production_contract_sha256": table["production_contract_sha256"],
        "table_id": table["table_id"],
        "table_sha256": _sha(table_raw),
        "merge_report_id": merge["report_id"],
        "merge_report_sha256": _sha(merge_raw),
        "closeout_provenance_id": closeout["closure_id"],
        "closeout_provenance_sha256": _sha(closeout_raw),
        "curves": table["complete_identity_count"],
        "measured_points": table["measured_point_count"],
        "trials_per_point": table["trials_per_point"],
        "interpolation_used": table["interpolation_used"],
        "imputation_used": table["imputation_used"],
        "extrapolation_used": table["extrapolation_used"],
        "predecessor_table_contribution": table["predecessor_table_contribution"],
    }
    _require(g8c == expected, "D0 G8_C binding differs")
    _require(artifact["protected_counters"] == {"inference": 0, "training": 0, "validation_decoding": 0, "test_access": 0}, "protected counters are nonzero")
    _require(artifact["test_access"] == 0, "D0 test access is nonzero")
    _require(artifact["prohibitions"] == {
        "validation_campaign_started": False,
        "selection_started": False,
        "pass_one_started": False,
        "pass_two_started": False,
        "training_started": False,
        "test_split_accessed": False,
        "g8_e_started": False,
    }, "D0 prohibition state differs")
    _require(artifact["next_gate"] == "G8_D/D1", "D0 did not open D1 as the next gate")
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=OUT)
    args = parser.parse_args()
    try:
        artifact = validate(args.path)
    except G8DOpenVerificationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "PASS", "artifact_id": artifact["artifact_id"], "table_id": artifact["g8_c"]["table_id"], "measured_points": artifact["g8_c"]["measured_points"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
