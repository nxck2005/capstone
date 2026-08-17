#!/usr/bin/env python3
"""Independently verify the final G8_D handoff and G8_E boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from baseline import g8_d  # noqa: E402
from verify_g8_d_smoke import verify as verify_smoke  # noqa: E402


HEX = re.compile(r"^[0-9a-f]{64}$")
HANDOFF = REPO / "results/baseline/g8_d/d7_handoff.json"


class HandoffVerificationError(ValueError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise HandoffVerificationError(f"cannot read D7 handoff: {exc}") from None
    if not isinstance(value, dict) or raw != g8_d.rendered_json(value):
        raise HandoffVerificationError("D7 handoff is not canonical rendered JSON")
    return value, raw


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HandoffVerificationError(message)


def verify(path: Path = HANDOFF) -> dict[str, Any]:
    value, raw = _read(path)
    required = {
        "schema_version", "artifact_role", "phase", "checkpoint", "status", "contract_id", "g8_c", "d0_open",
        "smoke", "verification", "tests", "safety", "next_gate", "g8_e_released", "full_campaign_not_started", "artifact_id",
    }
    _require(set(value) == required, "D7 handoff schema differs")
    _require((value["schema_version"], value["artifact_role"], value["phase"], value["checkpoint"], value["status"]) == (1, "g8_d_handoff", "G8_D", "D7", "GREEN"), "D7 handoff header differs")
    contract = g8_d.build_g8_d_contract(REPO)
    _require(value["contract_id"] == contract["contract_id"] and contract["next_gate"] == "G8_E/E0", "D7 contract or next gate differs")
    body = dict(value)
    artifact_id = body.pop("artifact_id")
    _require(isinstance(artifact_id, str) and artifact_id == "g8dhandoff-" + hashlib.sha256(_canonical(body)).hexdigest(), "D7 handoff ID differs")
    _require(raw == g8_d.rendered_json(value), "D7 handoff bytes are not canonical")

    g8c = value["g8_c"]
    _require(g8c["table_id"] == "g8pblertable-69ecc729f3b7dc3d67c0a3a5d8cf071cab927ad0a1e0cd5b18a6bbe674b9126f", "D7 handoff table ID differs")
    _require(g8c["curves"] == 153 and g8c["measured_points"] == 3213 and g8c["trials_per_point"] == 5000, "D7 handoff table coverage differs")
    _require(g8c["predecessor_table_contribution"] == "none" and HEX.fullmatch(g8c["table_sha256"]) is not None, "D7 handoff table binding differs")
    d0 = value["d0_open"]
    _require(HEX.fullmatch(d0["artifact_sha256"]) is not None and g8_d.sha256_file(REPO / "results/baseline/g8_d/d0_open.json") == d0["artifact_sha256"], "D0 handoff bytes changed")

    smoke_path = REPO / "results/baseline/g8_d/bounded_smoke.json"
    smoke = verify_smoke(smoke_path)
    smoke_binding = value["smoke"]
    _require(smoke_binding["artifact_id"] == smoke["artifact_id"] and smoke_binding["artifact_sha256"] == g8_d.sha256_file(smoke_path), "D6 smoke handoff binding differs")
    _require((smoke_binding["samples"], smoke_binding["candidates"], smoke_binding["cells"], smoke_binding["mutation_cases"]) == (1, 4, 4, 20), "D6 smoke handoff counts differ")
    _require(smoke_binding["merge_eligible"] is False, "D6 smoke was made merge-eligible")

    verification = value["verification"]
    _require(set(verification) == {
        "g8_c_successor", "g8_c_closeout", "exhaustive_frozen_table_lookup", "d0_open", "d1_d6_targeted",
        "d6_smoke", "w4_integration", "g2_adjudication", "packetisation", "documentation", "generated_spec_views",
        "literal_lint", "cpu_runtime_lock", "full_pytest",
    }, "D7 verifier list differs")
    _require(all(result == "PASS" for result in verification.values()), "D7 verifier list contains a failure")
    tests = value["tests"]
    _require(tests["full_pytest_collected"] == 2179 and tests["full_pytest_passed"] == 2179, "full pytest count differs")
    _require(tests["full_pytest_skipped"] == 0 and tests["full_pytest_failures"] == 0, "full pytest is not complete")
    safety = value["safety"]
    _require(all(safety[field] is False for field in ("full_validation_campaign_started", "selection_started", "pass_one_started", "pass_two_started", "training_started", "test_split_accessed", "g8_e_started")), "D7 safety flags are nonzero")
    _require(all(safety[field] == 0 for field in ("inference", "training", "validation_decoding", "test_access")), "D7 protected counters are nonzero")
    _require(value["next_gate"] == "G8_E/E0" and value["g8_e_released"] is True and value["full_campaign_not_started"] is True, "G8_E boundary differs")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=HANDOFF)
    args = parser.parse_args()
    try:
        value = verify(args.path)
    except HandoffVerificationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "PASS", "artifact_id": value["artifact_id"], "next_gate": value["next_gate"], "full_pytest": value["tests"]["full_pytest_passed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
