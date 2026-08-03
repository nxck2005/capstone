#!/usr/bin/env python3
"""Independently verify the corrected G8_B runner contract.

This verifier intentionally does not import the runner or the generator.  It
reconstructs the expected bindings from the frozen B1C/B2C/B3 artifacts,
repository bytes, and this file's own checkpoint constants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import sionna  # noqa: E402
import torch  # noqa: E402

from baseline import g8_bler_contract as frozen  # noqa: E402
from baseline.g8_campaign import (  # noqa: E402
    CAMPAIGN_STATE,
    load_campaign_state,
    rendered_json,
)
from config.params import get  # noqa: E402


CONTRACT_PATH = REPO_ROOT / "results/baseline/g8/bler_runner_contract.json"
EXPECTED_PHASE = "G8_B"
EXPECTED_CHECKPOINT = "B4"
EXPECTED_SCHEMA_VERSION = 3
EXPECTED_ROLE = "g8_bler_runner_contract"
EXPECTED_ID_PREFIX = "g8runner"
EXPECTED_SOURCE_ROLE = "g8b_b4_runner_contract_source"
EXPECTED_SOURCE_PATHS = (
    "src/baseline/g8_bler_runner.py",
    "tools/run_g8_bler.py",
    "tools/gen_g8_bler_runner_contract.py",
    "tools/verify_g8_bler_runner_contract.py",
    "tools/verify_g8_bounded_smoke.py",
    "tools/migrate_g8_bler_runner_contract.py",
)
EXPECTED_OUTPUT_PATH = "results/baseline/g8/bler_runner_contract.json"
EXPECTED_CAMPAIGN = "G-8"
EXPECTED_CAMPAIGN_ID = "g8-8acd86ad87ef223187b69a2caf6ab8d29de3700dac9d5a60bb421cb228d8900a"
EXPECTED_MANIFEST_SHA256 = "0e9504abdc79e90e07044a12a26aea10d5d3ef2cfc645ee4ee2a2bbe4f0722d1"
EXPECTED_REQUIRED_SHA256 = "b8f7540af2dcc34f3e2f070bbc651ccbd3af99fbbb335dc3988264216cc32b77"
EXPECTED_SELECTION_SHA256 = "6a4ffa98a26ee627f8339f1668f11305e097ca813e246d46a235dbfb2476db0e"
EXPECTED_B1C_ID = "g8bler-fa7b64abd2b20078668b5f251ea75dd3264f9293941657f70d2979bc83907975"
EXPECTED_B1C_SHA256 = "bc0db0a8ffe7b62238fa13e83ac9e82dec257816c98edce9a15cd3b226132866"
EXPECTED_B2C_ID = "g8state-a36b37f3c21d4254a50ffe5e893237ee4738c68c7b3e9d76b473856ca7605deb"
EXPECTED_B2C_SHA256 = "cac1dcf803d435de7b483db04d12afc30bea4180a835d8c0476de65540fbf583"
EXPECTED_B3_ID = "g8resume-7bf4935ed82aa1eea6514208cdd82f9b60f73e4ba58b280642b493444f5364c0"
EXPECTED_B3_SHA256 = "de767deef746e8b5cbc988994a18a72e50dda816ce432781aa95ee18a48acd41"
EXPECTED_COUNT = 3213
EXPECTED_B4_COMMAND = (
    'rg -n "G8_C|characterization_open|full_strength|run_g8_bler|resume_plan|merge_report|'
    'tooling_smoke_complete" src/baseline tools tests instructions'
)
EXPECTED_SUPERSEDES = {
    "contract_id": "g8runner-3e4c870966837d255829dbca6afc4d1e3ce5ccf4754618460c939607d9c1c7e5",
    "contract_sha256": "21ec8ae9c3c0787fa0a43bfdc12b4362bd26534a4774ee682070d94449e11268",
    "contract_bytes": 17597,
    "reason": "complete SR-1 literal compliance for infrastructure-only staging-name entropy and provide recoverable registered-smoke rebinding; no scientific or physical-layer semantics changed",
}
EXPECTED_SUPERSESSION_HISTORY = [
    {
        "schema_version": 2,
        "contract_id": EXPECTED_SUPERSEDES["contract_id"],
        "contract_sha256": EXPECTED_SUPERSEDES["contract_sha256"],
        "contract_bytes": EXPECTED_SUPERSEDES["contract_bytes"],
        "supersedes": {
            "contract_id": "g8runner-f5bd7abab06f88f879f460c33bec03bc76a7e1e5d47fa84bda5c31dc51bc5ec5",
            "contract_sha256": "d35bcce439eef232da58932406531133ac6261eb353722669c1712be89844d40",
            "contract_bytes": 15317,
            "reason": "bounded-smoke verifier referenced fields absent from the closed campaign-state schema",
        },
    },
    {
        "schema_version": 1,
        "contract_id": "g8runner-f5bd7abab06f88f879f460c33bec03bc76a7e1e5d47fa84bda5c31dc51bc5ec5",
        "contract_sha256": "d35bcce439eef232da58932406531133ac6261eb353722669c1712be89844d40",
        "contract_bytes": 15317,
    },
]


class RunnerContractVerificationError(RuntimeError):
    """A B4 runner-contract assertion failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RunnerContractVerificationError(message)


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RunnerContractVerificationError(f"cannot read {label}: {exc}") from exc
    _require(isinstance(payload, dict), f"{label} is not an object")
    return payload, raw


def _source_bindings() -> list[dict[str, Any]]:
    bindings = []
    for path in EXPECTED_SOURCE_PATHS:
        body = (REPO_ROOT / path).read_bytes()
        bindings.append({"path": path, "role": EXPECTED_SOURCE_ROLE, "bytes": len(body), "sha256": _sha256(body)})
    return bindings


def _contract_id(payload: dict[str, Any]) -> str:
    body = dict(payload)
    body.pop("contract_id", None)
    return f"{EXPECTED_ID_PREFIX}-{_sha256(_canonical(body))}"


def _authority() -> dict[str, Any]:
    return {
        "campaign_id": EXPECTED_CAMPAIGN_ID,
        "campaign_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "required_bler_artifact_sha256": EXPECTED_REQUIRED_SHA256,
        "selection_policy_sha256": EXPECTED_SELECTION_SHA256,
        "required_work_unit_count": EXPECTED_COUNT,
        "bler_tooling_contract_id": EXPECTED_B1C_ID,
        "bler_tooling_contract_sha256": EXPECTED_B1C_SHA256,
        "bler_state_contract_id": EXPECTED_B2C_ID,
        "bler_state_contract_sha256": EXPECTED_B2C_SHA256,
        "bler_resume_contract_id": EXPECTED_B3_ID,
        "bler_resume_contract_sha256": EXPECTED_B3_SHA256,
        "request_schema_version": 2,
        "result_schema_version": 2,
        "unit_state_schema_version": 2,
        "tooling_schema_version": 2,
    }


def _schemas() -> dict[str, Any]:
    b1, _ = _read_json(REPO_ROOT / "results/baseline/g8/bler_tooling_contract.json", "B1C contract")
    b2, _ = _read_json(REPO_ROOT / "results/baseline/g8/bler_state_contract.json", "B2C contract")
    b3, _ = _read_json(REPO_ROOT / "results/baseline/g8/bler_resume_contract.json", "B3 contract")
    request = b1["request_schema"]
    result = b1["result_schema"]
    state = b2["unit_state_schema"]
    return {
        "request": {
            "schema_version": request["version"],
            "artifact_role": request["artifact_role"],
            "fields": request["fields"],
            "unknown_fields_rejected": request["unknown_fields_rejected"],
            "request_is_never_merge_eligible": request["request_is_never_merge_eligible"],
            "test_split_access": request["test_split_access"],
        },
        "result": {
            "schema_version": result["version"],
            "artifact_role": result["artifact_role"],
            "fields": result["sections"],
            "identity_fields": result["identity_fields"],
            "measurement_fields": result["measurement_fields"],
            "execution_metadata_fields": result["execution_metadata_fields"],
            "disposition_fields": result["disposition_fields"],
            "implementation_fields": result["implementation_fields"],
            "statuses": result["statuses"],
            "status_rules": result["status_rules"],
            "execution_metadata_rules": result["execution_metadata_rules"],
            "test_split_access": 0,
        },
        "unit_state": {
            "schema_version": state["schema_version"],
            "artifact_role": state["artifact_role"],
            "fields": state["top_level_fields"],
            "identity_fields": state["identity_fields"],
            "runtime_metadata_fields": state["runtime_metadata_fields"],
            "statuses": list(state["statuses"]),
            "test_split_access": 0,
        },
        "resume_plan": {
            "schema_version": b3["schemas"]["resume_plan"]["schema_version"],
            "artifact_role": b3["schemas"]["resume_plan"]["artifact_role"],
            "digest_field": b3["schemas"]["resume_plan"]["digest_field"],
            "digest_rule": b3["schemas"]["resume_plan"]["digest_rule"],
        },
        "merge_report": {
            "schema_version": b3["schemas"]["merge_report"]["schema_version"],
            "artifact_role": b3["schemas"]["merge_report"]["artifact_role"],
            "digest_field": b3["schemas"]["merge_report"]["digest_field"],
            "digest_rule": b3["schemas"]["merge_report"]["digest_rule"],
        },
        "bounded_smoke_record": {
            "schema_version": 2,
            "artifact_role": "g8_bounded_smoke_record",
            "label": "NON-SCIENTIFIC BOUNDED SMOKE",
        },
    }


def _dependencies() -> dict[str, Any]:
    paths = ("src/baseline/ldpc/adapter.py", "src/baseline/ldpc/modulation.py")
    bindings = []
    for path in paths:
        body = (REPO_ROOT / path).read_bytes()
        bindings.append({"path": path, "role": EXPECTED_SOURCE_ROLE, "bytes": len(body), "sha256": _sha256(body)})
    return {
        "numpy_version": str(np.__version__),
        "sionna_version": str(sionna.__version__),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "configured_sionna_version": str(get("baseline.ldpc_impl_version")),
        "configured_decoder": get("baseline.ldpc_decoder"),
        "configured_decoder_impl_spelling": get("baseline.ldpc_decoder_impl_spelling"),
        "configured_decoder_offset": get("baseline.ldpc_decoder_offset"),
        "configured_decoder_vn_update": get("baseline.ldpc_decoder_vn_update"),
        "configured_decoder_schedule": get("baseline.ldpc_decoder_schedule"),
        "configured_max_iters": get("baseline.ldpc_max_iters"),
        "configured_llr_clip": get("baseline.ldpc_decoder_llr_clip"),
        "source_bindings": bindings,
    }


def _expected_authorization() -> dict[str, Any]:
    return {
        "bounded_smoke": {
            "phase": "G8_B",
            "stage": "tooling_open",
            "execution_class": "bounded_smoke",
            "device": "cpu",
            "root": "explicit absolute isolated nonproduction root",
            "max_work_units": 3,
            "max_trials_per_unit": 16,
        },
        "full_strength": {
            "phase": "G8_C",
            "stage": "characterization_open",
            "execution_class": "full_strength",
            "root": "explicit absolute runtime root",
            "rejected_before_root_state_request_adapter_bits_decode_in_g8_b": True,
        },
        "no_implicit_production_root": True,
        "invalid_shard_bounds_rejected": True,
        "unknown_work_unit_ids_rejected": True,
    }


def _expected_rng() -> dict[str, Any]:
    b1, _ = _read_json(REPO_ROOT / "results/baseline/g8/bler_tooling_contract.json", "B1C contract")
    return {
        "information_stream": b1["rng"]["information_bit_api"],
        "normal_stream": b1["rng"]["normal_api"],
        "information_stream_addressed_by_trial_offset": True,
        "persistent_real_and_imaginary_generators": True,
        "real_and_imaginary_streams_independent": True,
        "consumed_sequentially_from_index_zero": True,
        "batch_size_enters_seed_derivation": False,
        "attempt_enters_seed_derivation": False,
        "shard_enters_seed_derivation": False,
        "device_enters_seed_derivation": False,
        "reseed_each_batch": False,
        "philox_bit_generator": b1["rng"]["bit_generator"],
        "library": b1["rng"]["library"],
        "library_version": b1["rng"]["library_version"],
        "stream_chunk_boundary_invariant": b1["rng"]["chunk_boundary_invariant"],
    }


def _expected_physical_layer() -> dict[str, Any]:
    return {
        "identity_source": "authenticated exact required work-unit record",
        "derive_fields": ["K", "N", "base_graph", "lifting_size", "modulation", "rate", "SNR"],
        "information_bits_per_trial": "exactly K",
        "encoder": "baseline.ldpc.adapter.SionnaLDPCAdapter",
        "mapper": "baseline.ldpc.modulation.map_bits",
        "noise_convention": "Es/N0 per symbol; N0 = 10 ** (-SNR_dB / 10)",
        "complex_noise_scale": "sqrt(N0 / 2)",
        "demapper": "baseline.ldpc.modulation.max_log_llr",
        "decoder": "SionnaLDPCAdapter offset-min-sum decoder",
        "decoder_output_shape": "(batch, K)",
        "comparison_domain": "complete K-bit information vector",
        "information_bits_count": "trials_completed * K",
        "bit_errors": "complete K-vector Hamming distance",
        "block_errors": "one for any differing K-bit vector",
        "decoder_exception_policy": "failed attempt, never completed evidence",
        "malformed_output_policy": "failed attempt, never completed evidence",
        "finite_llr_required": True,
        "bounded_memory": True,
    }


def _expected_transaction() -> dict[str, Any]:
    return {
        "global_lock": "B3 shared global parent-directory lease",
        "per_unit_lock": "B2C owns the per-unit lock through state primitives",
        "lock_order": ["global shared parent-directory lock", "B2C per-unit lock"],
        "steps": [
            "build authenticated B3 plan",
            "select assigned remaining unit",
            "claim exact proposed attempt",
            "publish immutable request",
            "execute from trial zero",
            "publish immutable result",
            "transition directly to failed or result_linked",
            "validate complete chain",
            "release shared lease",
        ],
        "mid_work_unit_resume": False,
        "failed_result_recovery": "B3 recovery transitions a published failed result to failed",
        "complete_result_recovery": "B3 recovery links a published complete result",
    }


def _expected_publication() -> dict[str, Any]:
    return {
        "canonical_bytes_before_destination_open": True,
        "staging_same_directory_unique": True,
        "staging_open_flags": ["O_CREAT", "O_EXCL", "O_NOFOLLOW"],
        "staging_mode": "0600",
        "flush_and_file_fsync": True,
        "no_replace": "descriptor-relative renameat2(RENAME_NOREPLACE) with fail-closed availability",
        "directory_fsync": True,
        "final_path_never_opened_for_writing": True,
        "exact_existing_bytes_are_idempotent": True,
        "different_existing_bytes_are_conflict": True,
        "symlink_dangling_symlink_and_hard_link_alias_rejected": True,
        "uncertain_publication": "accept only exact installed canonical bytes and SHA-256",
        "tracked_smoke_record": {
            "canonical_bytes_before_destination_open": True,
            "first_install": "descriptor-relative renameat2(RENAME_NOREPLACE)",
            "guarded_provisional_replacement": "descriptor-relative renameat2(RENAME_EXCHANGE) only for the exact old unregistered record",
            "guarded_corrected_record_rebind": "descriptor-relative renameat2(RENAME_EXCHANGE) only for an exact unregistered schema-2 chain with its prior runner binding",
            "directory_fsync": True,
            "exact_reread_required": True,
            "conflicting_bytes_rejected": True,
        },
    }


def _expected_counts() -> dict[str, Any]:
    return {
        "trials_requested_source": frozen.FULL_STRENGTH_TRIAL_COUNT_SOURCE,
        "authoritative_fields": list(frozen.COUNT_FIELDS_AUTHORITATIVE),
        "invariants": list(frozen.COUNT_CROSS_INVARIANTS),
        "ber_rule": frozen.BER_POINT_ESTIMATE_RULE,
        "bler_rule": frozen.BLER_POINT_ESTIMATE_RULE,
        "confidence_method": frozen.CONFIDENCE_INTERVAL_METHOD,
        "confidence_role": frozen.CONFIDENCE_INTERVAL_ROLE,
        "bounded_smoke_is_not_scientific": True,
    }


def _expected_bounded() -> dict[str, Any]:
    return {
        "label": "NON-SCIENTIFIC BOUNDED SMOKE",
        "selection_rule": "first canonical required work-unit identity per configured modulation in configured order",
        "configured_modulations": list(get("baseline.modulations")),
        "maximum_work_units": 3,
        "trials_per_unit": 16,
        "device": "cpu",
        "terminal_classification": "terminal_nonmergeable",
        "required_coverage_contribution": 0,
        "merge_eligible": False,
        "test_split_access": 0,
        "temporary_root_removed": True,
        "record_artifact_role": "g8_bounded_smoke_record",
        "record_schema_version": 2,
        "official_work_unit_count": len(get("baseline.modulations")),
        "official_max_units_must_equal_count": True,
        "diagnostic_records_may_not_publish_official_artifact": True,
        "record_chain": {
            "work_unit_record": True,
            "request": True,
            "result": True,
            "terminal_state": True,
            "attempt": True,
            "exact_request_result_state_digests": True,
        },
        "record_verifier_must_pass_before_cli_success": True,
        "required_bler_sha_source": "exact closed produced_artifacts binding plus independently read artifact and manifest binding",
        "selection_policy_sha_source": "authenticated campaign manifest selection_policy.selection_policy_sha256",
        "selection_policy_reproduction": "canonical ordered policy-field array from the manifest field list and authenticated frozen W4 selection machinery",
    }


def _expected_hot_path() -> dict[str, Any]:
    return {
        "request_validation": "B3 cached strict fast validator",
        "result_validation": "B3 cached strict fast validator",
        "required_identity_artifact_reads_after_context_construction": 0,
        "b1c_tooling_contract_authentications_after_context_construction": 0,
    }


def _expected_no_science() -> dict[str, Any]:
    return {
        "phase": "G8_B",
        "stage": "tooling_open",
        "full_strength_execution": False,
        "required_bler_work_units_completed": 0,
        "validation_decoding": 0,
        "inference": 0,
        "training": 0,
        "test_access": 0,
        "BlerTable": False,
        "production_runtime_root": False,
    }


def _expected_handoff() -> dict[str, Any]:
    return {
        "first_authorized_phase": "G8_C",
        "first_authorized_stage": "characterization_open",
        "full_strength_is_not_started_in_g8_b": True,
        "runner_contract_is_not_a_bler_table": True,
    }


def _assert_no_absolute_paths(value: Any) -> None:
    if isinstance(value, str):
        _require(not Path(value).is_absolute(), "runner contract contains an absolute path")
    elif isinstance(value, dict):
        for child in value.values():
            _assert_no_absolute_paths(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_absolute_paths(child)


def verify(path: Path = CONTRACT_PATH, *, require_registered: bool = False) -> dict[str, Any]:
    payload, raw = _read_json(path, "B4 runner contract")
    _require(raw == rendered_json(payload), "B4 runner contract is not canonical rendered JSON")
    expected_top = {
        "schema_version", "artifact_role", "campaign", "phase", "checkpoint",
        "supersedes", "supersession_history",
        "scientific_execution_performed", "characterization_started", "bounded_smoke_started",
        "contract_sources", "authority_bindings", "dependencies", "schemas", "authorization",
        "rng", "physical_layer", "transaction", "publication", "count_semantics",
        "bounded_smoke", "authenticated_hot_path", "no_science_boundary", "g8_c_handoff", "contract_id",
    }
    _require(set(payload) == expected_top, "B4 runner contract top-level fields changed")
    _require(payload["schema_version"] == EXPECTED_SCHEMA_VERSION, "B5 runner contract schema changed")
    _require(payload["artifact_role"] == EXPECTED_ROLE, "B4 runner contract role changed")
    _require(payload["campaign"] == EXPECTED_CAMPAIGN, "B4 runner contract campaign changed")
    _require(payload["phase"] == EXPECTED_PHASE and payload["checkpoint"] == EXPECTED_CHECKPOINT, "B4 phase/checkpoint changed")
    _require(payload["supersedes"] == EXPECTED_SUPERSEDES, "B5 immediate supersession relationship changed")
    _require(payload["supersession_history"] == EXPECTED_SUPERSESSION_HISTORY, "B5 supersession history is incomplete or changed")
    _require(payload["scientific_execution_performed"] is False, "B5 contract claims scientific execution")
    _require(payload["characterization_started"] is False, "B5 contract claims characterization")
    _require(payload["bounded_smoke_started"] is False, "B5 contract claims smoke has started")
    _require(payload["contract_sources"] == _source_bindings(), "B5 bound source bytes changed")
    _require([entry["path"] for entry in payload["contract_sources"]] == list(EXPECTED_SOURCE_PATHS), "B5 source order changed")
    _require(all(entry["path"] != EXPECTED_OUTPUT_PATH for entry in payload["contract_sources"]), "B5 binds its own output path")
    _require(payload["authority_bindings"] == _authority(), "B5 scientific authority bindings changed")
    _require(payload["dependencies"] == _dependencies(), "B5 dependency or LDPC source bindings changed")
    _require(payload["schemas"] == _schemas(), "B5 request/result/state schema bindings changed")
    _require(payload["authorization"] == _expected_authorization(), "B5 authorization gates changed")
    _require(payload["rng"] == _expected_rng(), "B5 RNG rules changed")
    _require(payload["physical_layer"] == _expected_physical_layer(), "B5 physical-layer pipeline changed")
    _require(payload["transaction"] == _expected_transaction(), "B5 transaction order changed")
    _require(payload["publication"] == _expected_publication(), "B5 publication rules changed")
    _require(payload["count_semantics"] == _expected_counts(), "B5 count semantics changed")
    _require(payload["bounded_smoke"] == _expected_bounded(), "B5 bounded-smoke rules changed")
    _require(payload["authenticated_hot_path"] == _expected_hot_path(), "B5 authenticated hot path changed")
    _require(payload["no_science_boundary"] == _expected_no_science(), "B5 no-science boundary changed")
    _require(payload["g8_c_handoff"] == _expected_handoff(), "B5 G8_C handoff changed")
    _require(payload["contract_id"] == _contract_id(payload), "B5 contract ID does not reproduce")
    _assert_no_absolute_paths(payload)
    _require(_sha256(raw) not in raw.decode("utf-8"), "B5 contract binds its own SHA-256")

    if require_registered:
        state, _state_raw = _read_json(CAMPAIGN_STATE, "campaign state")
        matches = [entry for entry in state["identity"]["produced_artifacts"] if entry["path"] == EXPECTED_OUTPUT_PATH]
        _require(len(matches) == 1, "B5 runner contract is not registered exactly once")
        _require(matches[0]["sha256"] == _sha256(raw) and matches[0]["bytes"] == len(raw), "registered B5 bytes do not match")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--require-registered", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload = verify(args.path, require_registered=args.require_registered)
    except RunnerContractVerificationError as exc:
        raise SystemExit(f"G8 B5 runner contract verification HOLD: {exc}") from exc
    print(
        "G8 B5 runner contract verification PASS: "
        f"contract_id={payload['contract_id']} sha256={_sha256(args.path.read_bytes())} bytes={args.path.stat().st_size}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
