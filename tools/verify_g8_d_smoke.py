#!/usr/bin/env python3
"""Independently verify the bounded, non-scientific G8_D smoke witness."""

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

from baseline import g8_d  # noqa: E402
from run_g8_d_smoke import MUTATION_CASES, SMOKE_ARTIFACT_ROLE, SMOKE_SCHEMA_VERSION  # noqa: E402


HEX = re.compile(r"^[0-9a-f]{64}$")
ARTIFACT = REPO / "results/baseline/g8_d/bounded_smoke.json"


class SmokeVerificationError(ValueError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise SmokeVerificationError(f"cannot read smoke artifact: {exc}") from None
    if not isinstance(value, dict) or raw != g8_d.rendered_json(value):
        raise SmokeVerificationError("smoke artifact is not canonical rendered JSON")
    return value, raw


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeVerificationError(message)


def verify(path: Path = ARTIFACT) -> dict[str, Any]:
    value, raw = _read(path)
    required = {
        "schema_version", "artifact_role", "contract_id", "campaign_id", "smoke_label",
        "non_scientific", "non_selection", "non_headline", "merge_eligible",
        "validation_campaign_started", "pass_one_started", "training_started", "test_split_accessed",
        "samples", "candidates", "cells", "measurement_work_units", "statuses",
        "codec_backend_calls", "codec_cache_hit", "reconstruction_cache_hit",
        "reconstruction_decoder_calls", "emitted_bytes_authoritative",
        "requested_ratio_is_provenance_only", "over_budget_rejected", "br11",
        "clean_measurement", "resume", "protected_counters", "mutation_case_names", "source", "artifact_id",
    }
    _require(set(value) == required, "smoke artifact schema differs")
    _require(value["schema_version"] == SMOKE_SCHEMA_VERSION and value["artifact_role"] == SMOKE_ARTIFACT_ROLE, "smoke header differs")
    contract = g8_d.build_g8_d_contract(REPO)
    _require(value["contract_id"] == contract["contract_id"] and value["campaign_id"] == contract["campaign_id"], "smoke contract binding differs")
    body = dict(value)
    artifact_id = body.pop("artifact_id")
    _require(isinstance(artifact_id, str) and artifact_id == "g8dsmoke-" + hashlib.sha256(_canonical(body)).hexdigest(), "smoke artifact ID differs")
    _require(raw == g8_d.rendered_json(value), "smoke bytes are not canonical")
    _require(value["smoke_label"] == "NON-SCIENTIFIC BOUNDED SMOKE", "smoke label differs")
    _require(all(value[field] is True for field in ("non_scientific", "non_selection", "non_headline")), "smoke scope flags are not explicit")
    _require(all(value[field] is False for field in ("merge_eligible", "validation_campaign_started", "pass_one_started", "training_started", "test_split_accessed")), "smoke safety flags are nonzero")
    _require((value["samples"], value["candidates"], value["cells"], value["measurement_work_units"]) == (1, 4, 4, 1), "smoke bounds differ")
    _require(value["statuses"] == ["feasible", "structural_infeasibility", "codec_infeasibility", "decode_failure"], "smoke status coverage differs")
    _require(value["codec_backend_calls"] == 3 and value["codec_cache_hit"] is True, "codec cache smoke did not prove one reuse")
    _require(value["reconstruction_cache_hit"] is True and value["reconstruction_decoder_calls"] == 1, "reconstruction cache smoke differs")
    _require(value["emitted_bytes_authoritative"] is True and value["requested_ratio_is_provenance_only"] is True and value["over_budget_rejected"] is True, "emitted-byte smoke differs")
    br11 = value["br11"]
    _require(br11["denominator"] == 2 and br11["verdict_counts"] == {"decode_failure": 1, "delivered": 1}, "BR-11 smoke denominator differs")
    _require(br11["header_bytes"] == 18.0 and br11["payload_bytes"] == 3.0 and br11["payload_filler_bytes"] == 9.0, "BR-11 smoke arithmetic differs")
    clean = value["clean_measurement"]
    _require(clean["correct_count"] == 2 and clean["total_count"] == 3, "clean measurement counts differ")
    _require(clean["accuracy_derivation"] == "correct_count / total_count" and clean["validation_only"] is True, "clean measurement derivation differs")
    _require(clean["test_access"] == 0 and clean["training"] is False and clean["scientific_evidence"] is False and clean["merge_eligible"] is False, "clean measurement safety differs")
    resume = value["resume"]
    _require(resume["completed_count"] == 1 and resume["reused_complete_output"] is True and resume["first_run_count"] == 1, "resume smoke differs")
    _require(resume["in_progress_work_unit_id"] is None and resume["aggregate_record_count"] == 1, "resume terminal state differs")
    _require(value["protected_counters"] == {"inference": 0, "training": 0, "validation_decoding": 0, "test_access": 0}, "smoke protected counters differ")
    _require(value["mutation_case_names"] == list(MUTATION_CASES) and len(MUTATION_CASES) >= 20, "mutation matrix is incomplete")
    _require(value["source"] == "d6-synthetic-non-scientific-smoke", "smoke provenance differs")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=ARTIFACT)
    args = parser.parse_args()
    try:
        value = verify(args.path)
    except SmokeVerificationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "PASS", "artifact_id": value["artifact_id"], "mutation_cases": len(value["mutation_case_names"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
