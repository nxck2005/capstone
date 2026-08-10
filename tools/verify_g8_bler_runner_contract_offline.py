#!/usr/bin/env python3
"""Verify the registered G8 runner contract without requiring CUDA.

The historical B5 verifier is intentionally bound to the primary runtime and
remains unchanged.  This CI verifier checks the same immutable contract
fields, source/configuration bindings, self-hash and registration while
treating the recorded environment version tuple as historical metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))

import verify_g8_bler_runner_contract as strict  # noqa: E402


RUNTIME_FIELDS = frozenset(
    {"numpy_version", "sionna_version", "torch_version", "torch_cuda_version"}
)


def _verify_dependencies(payload: dict[str, Any]) -> None:
    current = strict._dependencies()
    recorded = payload["dependencies"]
    for field, expected in current.items():
        if field in RUNTIME_FIELDS:
            continue
        strict._require(
            recorded.get(field) == expected,
            f"B5 portable dependency binding changed: {field}",
        )
    for field in RUNTIME_FIELDS:
        strict._require(
            isinstance(recorded.get(field), str) and recorded[field],
            f"B5 historical runtime field missing: {field}",
        )


def verify(path: Path = strict.CONTRACT_PATH, *, require_registered: bool = True) -> dict[str, Any]:
    payload, raw = strict._read_json(path, "B5 runner contract")
    strict._require(raw == strict.rendered_json(payload), "B5 runner contract is not canonical rendered JSON")
    expected_top = {
        "schema_version", "artifact_role", "campaign", "phase", "checkpoint",
        "supersedes", "supersession_history", "scientific_execution_performed",
        "characterization_started", "bounded_smoke_started", "contract_sources",
        "authority_bindings", "dependencies", "schemas", "authorization", "rng",
        "physical_layer", "transaction", "publication", "count_semantics",
        "bounded_smoke", "authenticated_hot_path", "no_science_boundary",
        "g8_c_handoff", "contract_id",
    }
    strict._require(set(payload) == expected_top, "B5 runner contract top-level fields changed")
    strict._require(payload["schema_version"] == strict.EXPECTED_SCHEMA_VERSION, "B5 runner contract schema changed")
    strict._require(payload["artifact_role"] == strict.EXPECTED_ROLE, "B4 runner contract role changed")
    strict._require(payload["campaign"] == strict.EXPECTED_CAMPAIGN, "B4 runner contract campaign changed")
    strict._require(
        payload["phase"] == strict.EXPECTED_PHASE and payload["checkpoint"] == strict.EXPECTED_CHECKPOINT,
        "B4 phase/checkpoint changed",
    )
    strict._require(payload["supersedes"] == strict.EXPECTED_SUPERSEDES, "B5 immediate supersession relationship changed")
    strict._require(payload["supersession_history"] == strict.EXPECTED_SUPERSESSION_HISTORY, "B5 supersession history changed")
    strict._require(
        payload["scientific_execution_performed"] is False
        and payload["characterization_started"] is False
        and payload["bounded_smoke_started"] is False,
        "B5 contract claims execution",
    )
    strict._require(payload["contract_sources"] == strict._source_bindings(), "B5 bound source bytes changed")
    strict._require(payload["authority_bindings"] == strict._authority(), "B5 scientific authority bindings changed")
    _verify_dependencies(payload)
    strict._require(payload["schemas"] == strict._schemas(), "B5 request/result/state schema bindings changed")
    strict._require(payload["authorization"] == strict._expected_authorization(), "B5 authorization gates changed")
    strict._require(payload["rng"] == strict._expected_rng(), "B5 RNG rules changed")
    strict._require(payload["physical_layer"] == strict._expected_physical_layer(), "B5 physical-layer pipeline changed")
    strict._require(payload["transaction"] == strict._expected_transaction(), "B5 transaction order changed")
    strict._require(payload["publication"] == strict._expected_publication(), "B5 publication rules changed")
    strict._require(payload["count_semantics"] == strict._expected_counts(), "B5 count semantics changed")
    strict._require(payload["bounded_smoke"] == strict._expected_bounded(), "B5 bounded-smoke rules changed")
    strict._require(payload["authenticated_hot_path"] == strict._expected_hot_path(), "B5 authenticated hot path changed")
    strict._require(payload["no_science_boundary"] == strict._expected_no_science(), "B5 no-science boundary changed")
    strict._require(payload["g8_c_handoff"] == strict._expected_handoff(), "B5 G8_C handoff changed")
    strict._require(payload["contract_id"] == strict._contract_id(payload), "B5 contract ID does not reproduce")
    strict._assert_no_absolute_paths(payload)
    strict._require(strict._sha256(raw) not in raw.decode("utf-8"), "B5 contract binds its own SHA-256")
    if require_registered:
        state, _state_raw = strict._read_json(strict.CAMPAIGN_STATE, "campaign state")
        matches = [
            entry for entry in state["identity"]["produced_artifacts"]
            if entry["path"] == strict.EXPECTED_OUTPUT_PATH
        ]
        strict._require(len(matches) == 1, "B5 runner contract is not registered exactly once")
        strict._require(
            matches[0]["sha256"] == strict._sha256(raw)
            and matches[0]["bytes"] == len(raw),
            "registered B5 bytes do not match",
        )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=strict.CONTRACT_PATH)
    parser.add_argument("--no-require-registered", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload = verify(args.path, require_registered=not args.no_require_registered)
    except strict.RunnerContractVerificationError as exc:
        raise SystemExit(f"offline G8 B5 runner contract verification HOLD: {exc}") from exc
    print(
        "offline G8 B5 runner contract verification PASS: "
        f"contract_id={payload['contract_id']} sha256={strict._sha256(args.path.read_bytes())} bytes={args.path.stat().st_size}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
