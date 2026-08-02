#!/usr/bin/env python3
"""Generate the deterministic G8_B runner contract before B4 registration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import sionna  # noqa: E402
import torch  # noqa: E402

from baseline import g8_bler_contract as bler_contract  # noqa: E402
from baseline import g8_bler_resume as resume  # noqa: E402
from baseline import g8_bler_runner as runner  # noqa: E402
from baseline import g8_bler_work_units as work_units  # noqa: E402
from baseline.classical.outage import write_json_atomically  # noqa: E402
from baseline.g8_campaign import canonical_json, rendered_json, sha256_bytes  # noqa: E402
from baseline.g8_bler_resume import AuthenticatedResumeContext  # noqa: E402
from config.params import get  # noqa: E402


CONTRACT_PATH = REPO_ROOT / runner.RUNNER_CONTRACT_REPO_RELATIVE_PATH
CONTRACT_SCHEMA_VERSION = runner.RUNNER_CONTRACT_SCHEMA_VERSION
CONTRACT_ARTIFACT_ROLE = runner.RUNNER_CONTRACT_ARTIFACT_ROLE
CONTRACT_ID_PREFIX = runner.RUNNER_CONTRACT_ID_PREFIX
CONTRACT_SOURCE_PATHS = runner.RUNNER_CONTRACT_SOURCE_PATHS
CONTRACT_SOURCE_ROLE = runner.RUNNER_CONTRACT_SOURCE_ROLE


def contract_identifier(payload: dict[str, Any]) -> str:
    body = dict(payload)
    body.pop("contract_id", None)
    return f"{CONTRACT_ID_PREFIX}-{sha256_bytes(canonical_json(body))}"


def _source_bindings(paths: tuple[str, ...]) -> list[dict[str, Any]]:
    result = []
    for path in paths:
        body = (REPO_ROOT / path).read_bytes()
        result.append(
            {
                "path": path,
                "role": CONTRACT_SOURCE_ROLE,
                "bytes": len(body),
                "sha256": sha256_bytes(body),
            }
        )
    return result


def _dependency_bindings() -> dict[str, Any]:
    source_paths = (
        "src/baseline/ldpc/adapter.py",
        "src/baseline/ldpc/modulation.py",
    )
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
        "source_bindings": _source_bindings(source_paths),
    }


def _authority(context: AuthenticatedResumeContext) -> dict[str, Any]:
    authority = context.authority_binding()
    state = context.state_contract_binding()
    resume_binding = context.require_resume_contract_binding()
    return {
        "campaign_id": authority["campaign_id"],
        "campaign_manifest_sha256": authority["campaign_manifest_sha256"],
        "required_bler_artifact_sha256": authority["required_bler_artifact_sha256"],
        "selection_policy_sha256": authority["selection_policy_sha256"],
        "required_work_unit_count": authority["required_work_unit_count"],
        "bler_tooling_contract_id": authority["bler_tooling_contract_id"],
        "bler_tooling_contract_sha256": authority["bler_tooling_contract_sha256"],
        "bler_state_contract_id": state["bler_state_contract_id"],
        "bler_state_contract_sha256": state["bler_state_contract_sha256"],
        "bler_resume_contract_id": resume_binding["bler_resume_contract_id"],
        "bler_resume_contract_sha256": resume_binding["bler_resume_contract_sha256"],
        "request_schema_version": authority["request_schema_version"],
        "result_schema_version": authority["result_schema_version"],
        "unit_state_schema_version": work_units.UNIT_STATE_SCHEMA_VERSION,
        "tooling_schema_version": authority["tooling_schema_version"],
    }


def _schemas() -> dict[str, Any]:
    return {
        "request": {
            "schema_version": bler_contract.BLER_WORK_UNIT_REQUEST_SCHEMA_VERSION,
            "artifact_role": bler_contract.REQUEST_ARTIFACT_ROLE,
            "fields": list(bler_contract.REQUEST_FIELDS),
            "unknown_fields_rejected": True,
            "request_is_never_merge_eligible": True,
            "test_split_access": 0,
        },
        "result": {
            "schema_version": bler_contract.BLER_WORK_UNIT_RESULT_SCHEMA_VERSION,
            "artifact_role": bler_contract.RESULT_ARTIFACT_ROLE,
            "fields": list(bler_contract.RESULT_FIELDS),
            "identity_fields": list(bler_contract.RESULT_IDENTITY_FIELDS),
            "measurement_fields": list(bler_contract.RESULT_MEASUREMENT_FIELDS),
            "execution_metadata_fields": list(bler_contract.RESULT_EXECUTION_METADATA_FIELDS),
            "disposition_fields": list(bler_contract.RESULT_DISPOSITION_FIELDS),
            "implementation_fields": list(bler_contract.IMPLEMENTATION_FIELDS),
            "statuses": list(bler_contract.RESULT_STATUSES),
            "status_rules": dict(bler_contract.RESULT_STATUS_RULES),
            "execution_metadata_rules": dict(bler_contract.EXECUTION_METADATA_RULES),
            "test_split_access": 0,
        },
        "unit_state": {
            "schema_version": work_units.UNIT_STATE_SCHEMA_VERSION,
            "artifact_role": work_units.UNIT_STATE_ARTIFACT_ROLE,
            "fields": list(work_units.UNIT_STATE_FIELDS),
            "identity_fields": list(work_units.UNIT_STATE_IDENTITY_FIELDS),
            "runtime_metadata_fields": list(work_units.UNIT_STATE_RUNTIME_METADATA_FIELDS),
            "statuses": list(work_units.STATE_STATUSES),
            "test_split_access": 0,
        },
        "resume_plan": {
            "schema_version": resume.RESUME_PLAN_SCHEMA_VERSION,
            "artifact_role": resume.RESUME_PLAN_ARTIFACT_ROLE,
            "digest_field": resume.PLAN_DIGEST_FIELD,
            "digest_rule": "sha256(canonical JSON over the complete plan identity excluding plan_digest)",
        },
        "merge_report": {
            "schema_version": resume.MERGE_REPORT_SCHEMA_VERSION,
            "artifact_role": resume.MERGE_REPORT_ARTIFACT_ROLE,
            "digest_field": resume.MERGE_REPORT_DIGEST_FIELD,
            "digest_rule": "sha256(canonical JSON over the complete report identity excluding report_digest)",
        },
        "bounded_smoke_record": {
            "schema_version": runner.SMOKE_RECORD_SCHEMA_VERSION,
            "artifact_role": runner.SMOKE_RECORD_ARTIFACT_ROLE,
            "label": runner.BOUNDED_SMOKE_LABEL,
        },
    }


def _build_without_id(context: AuthenticatedResumeContext) -> dict[str, Any]:
    authority = _authority(context)
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "artifact_role": CONTRACT_ARTIFACT_ROLE,
        "campaign": "G-8",
        "phase": "G8_B",
        "checkpoint": "B4",
        "scientific_execution_performed": False,
        "characterization_started": False,
        "bounded_smoke_started": False,
        "contract_sources": _source_bindings(CONTRACT_SOURCE_PATHS),
        "authority_bindings": authority,
        "dependencies": _dependency_bindings(),
        "schemas": _schemas(),
        "authorization": {
            "bounded_smoke": {
                "phase": "G8_B",
                "stage": "tooling_open",
                "execution_class": runner.EXECUTION_CLASS_BOUNDED_SMOKE,
                "device": "cpu",
                "root": "explicit absolute isolated nonproduction root",
                "max_work_units": runner.BOUNDED_SMOKE_MAX_WORK_UNITS,
                "max_trials_per_unit": runner.BOUNDED_SMOKE_MAX_TRIALS,
            },
            "full_strength": {
                "phase": "G8_C",
                "stage": "characterization_open",
                "execution_class": runner.EXECUTION_CLASS_FULL_STRENGTH,
                "root": "explicit absolute runtime root",
                "rejected_before_root_state_request_adapter_bits_decode_in_g8_b": True,
            },
            "no_implicit_production_root": True,
            "invalid_shard_bounds_rejected": True,
            "unknown_work_unit_ids_rejected": True,
        },
        "rng": {
            "information_stream": bler_contract.INFORMATION_BIT_API,
            "normal_stream": bler_contract.NORMAL_API,
            "information_stream_addressed_by_trial_offset": True,
            "persistent_real_and_imaginary_generators": True,
            "real_and_imaginary_streams_independent": True,
            "consumed_sequentially_from_index_zero": True,
            "batch_size_enters_seed_derivation": False,
            "attempt_enters_seed_derivation": False,
            "shard_enters_seed_derivation": False,
            "device_enters_seed_derivation": False,
            "reseed_each_batch": False,
            "philox_bit_generator": bler_contract.RNG_BIT_GENERATOR,
            "library": bler_contract.RNG_LIBRARY,
            "library_version": bler_contract.RNG_LIBRARY_VERSION,
            "stream_chunk_boundary_invariant": True,
        },
        "physical_layer": {
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
        },
        "transaction": {
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
        },
        "publication": {
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
        },
        "count_semantics": {
            "trials_requested_source": bler_contract.FULL_STRENGTH_TRIAL_COUNT_SOURCE,
            "authoritative_fields": list(bler_contract.COUNT_FIELDS_AUTHORITATIVE),
            "invariants": list(bler_contract.COUNT_CROSS_INVARIANTS),
            "ber_rule": bler_contract.BER_POINT_ESTIMATE_RULE,
            "bler_rule": bler_contract.BLER_POINT_ESTIMATE_RULE,
            "confidence_method": bler_contract.CONFIDENCE_INTERVAL_METHOD,
            "confidence_role": bler_contract.CONFIDENCE_INTERVAL_ROLE,
            "bounded_smoke_is_not_scientific": True,
        },
        "bounded_smoke": {
            "label": runner.BOUNDED_SMOKE_LABEL,
            "selection_rule": "first canonical required work-unit identity per configured modulation in configured order",
            "configured_modulations": list(get("baseline.modulations")),
            "maximum_work_units": runner.BOUNDED_SMOKE_MAX_WORK_UNITS,
            "trials_per_unit": runner.BOUNDED_SMOKE_MAX_TRIALS,
            "device": "cpu",
            "terminal_classification": resume.CLASSIFICATION_TERMINAL_NONMERGEABLE,
            "required_coverage_contribution": 0,
            "merge_eligible": False,
            "test_split_access": 0,
            "temporary_root_removed": True,
            "record_artifact_role": runner.SMOKE_RECORD_ARTIFACT_ROLE,
        },
        "no_science_boundary": {
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
        },
        "g8_c_handoff": {
            "first_authorized_phase": "G8_C",
            "first_authorized_stage": "characterization_open",
            "full_strength_is_not_started_in_g8_b": True,
            "runner_contract_is_not_a_bler_table": True,
        },
    }


def build() -> dict[str, Any]:
    context = AuthenticatedResumeContext(require_resume_contract=True)
    payload = _build_without_id(context)
    payload["contract_id"] = contract_identifier(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    payload = build()
    expected = rendered_json(payload)
    if args.check:
        try:
            actual = CONTRACT_PATH.read_bytes()
        except OSError as exc:
            raise SystemExit(f"missing {CONTRACT_PATH.relative_to(REPO_ROOT)}: {exc}") from exc
        if actual != expected:
            raise SystemExit("bler_runner_contract.json is stale; regenerate it")
        print(
            "ok: G8 B4 runner contract matches regenerated artifact "
            f"contract_id={payload['contract_id']}"
        )
        return 0
    digest = write_json_atomically(CONTRACT_PATH, payload)
    print(
        f"wrote {CONTRACT_PATH.relative_to(REPO_ROOT)} "
        f"contract_id={payload['contract_id']} sha256={digest} bytes={len(expected)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
