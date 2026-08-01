#!/usr/bin/env python3
"""Generate the frozen G8_B BLER tooling contract; never run science.

The artifact this writes is the machine-readable form of the B1 freeze. It is
generator-owned: never hand-edit `results/baseline/g8/bler_tooling_contract.json`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from baseline.classical.outage import write_json_atomically  # noqa: E402
from baseline.g8_campaign import (  # noqa: E402
    CAMPAIGN,
    REQUIRED_BLER_IDENTITIES,
    canonical_json,
    load_required_bler_identities,
    rendered_json,
    sha256_bytes,
)
from baseline import g8_bler_contract as contract  # noqa: E402
from config.params import REPO_ROOT  # noqa: E402

BLER_TOOLING_CONTRACT = REPO_ROOT / "results/baseline/g8/bler_tooling_contract.json"
PHASE = "G8_B"
CHECKPOINT = "B1"

#: Only sources intended to remain immutable after B1. No runner, shard,
#: checkpoint or merge file is bound here, because none exists yet.
CONTRACT_SOURCES = (
    "src/baseline/g8_bler_contract.py",
    "tools/gen_g8_bler_tooling_contract.py",
    "tools/verify_g8_bler_tooling_contract.py",
)


def _binding(path: str, *, role: str) -> dict[str, Any]:
    body = (REPO_ROOT / path).read_bytes()
    return {"path": path, "role": role, "sha256": sha256_bytes(body), "bytes": len(body)}


def contract_identifier(payload: dict[str, Any]) -> str:
    """Derive the stable ID from every contract field except the ID itself."""

    basis = dict(payload)
    basis.pop("contract_id", None)
    return f"{contract.CONTRACT_ID_PREFIX}-{sha256_bytes(canonical_json(basis))}"


def build() -> dict[str, Any]:
    """Assemble the complete pre-data tooling contract."""

    if not contract.installed_rng_version_matches():
        raise RuntimeError(
            "installed numpy does not match the frozen RNG binding "
            f"{contract.RNG_LIBRARY_VERSION}"
        )
    bindings = contract.campaign_bindings()
    required = load_required_bler_identities(REQUIRED_BLER_IDENTITIES)
    work_units = required["required_bler_work_units"]

    payload: dict[str, Any] = {
        "schema_version": contract.BLER_TOOLING_CONTRACT_SCHEMA_VERSION,
        "campaign": CAMPAIGN,
        "artifact_role": contract.TOOLING_CONTRACT_ARTIFACT_ROLE,
        "phase": PHASE,
        "checkpoint": CHECKPOINT,
        "scientific_execution_performed": False,
        "characterization_started": False,
        "bounded_smoke_started": False,
        "campaign_bindings": {
            "campaign_id": bindings["campaign_id"],
            "campaign_manifest": _binding(
                "results/baseline/g8/campaign_manifest.json", role="g8a_campaign_manifest"
            ),
            "required_bler_identities": _binding(
                str(REQUIRED_BLER_IDENTITIES.relative_to(REPO_ROOT)),
                role="g8a_required_bler_identities",
            ),
            "required_work_unit_count": len(work_units),
            "selection_policy_sha256": bindings["selection_policy_sha256"],
        },
        "trial_count": {
            "full_strength_trials": contract.full_strength_trial_count(),
            "source": contract.FULL_STRENGTH_TRIAL_COUNT_SOURCE,
            "parameter": contract.FULL_STRENGTH_TRIAL_COUNT_PARAMETER,
            "g2_reference_key_not_used": contract.G2_REFERENCE_TRIAL_COUNT_KEY_NOT_USED,
            "adaptive_stopping_permitted": contract.ADAPTIVE_STOPPING_PERMITTED,
            "no_early_stopping_rule": contract.NO_EARLY_STOPPING_RULE,
        },
        "execution_classes": {
            "full_strength": contract.EXECUTION_CLASS_FULL_STRENGTH,
            "bounded_smoke": contract.EXECUTION_CLASS_BOUNDED_SMOKE,
            "bounded_smoke_max_work_units": contract.BOUNDED_SMOKE_MAX_WORK_UNITS,
            "bounded_smoke_max_trials_per_unit": contract.BOUNDED_SMOKE_MAX_TRIALS_PER_UNIT,
            "bounded_smoke_selection_rule": contract.BOUNDED_SMOKE_SELECTION_RULE,
            "bounded_smoke_label": contract.BOUNDED_SMOKE_LABEL,
            "bounded_smoke_trial_count_source": contract.BOUNDED_SMOKE_TRIAL_COUNT_SOURCE,
            "bounded_smoke_is_scientific_evidence": False,
            "bounded_smoke_is_merge_eligible": False,
            "bounded_smoke_required_coverage_contribution": 0,
        },
        "seed": {
            "derivation_identity": contract.SEED_DERIVATION_IDENTITY,
            "domain_separator": contract.SEED_DOMAIN_SEPARATOR,
            "input_encoding": contract.SEED_INPUT_ENCODING,
            "digest": contract.SEED_DIGEST,
            "output_rule": contract.SEED_OUTPUT_RULE,
            "width_bits": contract.SEED_WIDTH_BITS,
            "purposes": list(contract.SEED_PURPOSES),
            "forbidden_inputs": list(contract.SEED_FORBIDDEN_INPUTS),
            "test_vectors": contract.seed_test_vectors(),
        },
        "rng": contract.rng_contract(),
        "resume": {
            "granularity": contract.RESUME_GRANULARITY,
            "policy": contract.RESUME_POLICY,
            "mid_work_unit_resume_permitted": contract.MID_WORK_UNIT_RESUME_PERMITTED,
        },
        "request_schema": {
            "version": contract.BLER_WORK_UNIT_REQUEST_SCHEMA_VERSION,
            "artifact_role": contract.REQUEST_ARTIFACT_ROLE,
            "fields": list(contract.REQUEST_FIELDS),
            "unknown_fields_rejected": True,
            "omitted_field_defaults_permitted": False,
            "request_is_never_merge_eligible": True,
            "test_split_access": contract.TEST_SPLIT_ACCESS,
        },
        "result_schema": {
            "version": contract.BLER_WORK_UNIT_RESULT_SCHEMA_VERSION,
            "artifact_role": contract.RESULT_ARTIFACT_ROLE,
            "sections": list(contract.RESULT_FIELDS),
            "identity_fields": list(contract.RESULT_IDENTITY_FIELDS),
            "measurement_fields": list(contract.RESULT_MEASUREMENT_FIELDS),
            "execution_metadata_fields": list(contract.RESULT_EXECUTION_METADATA_FIELDS),
            "disposition_fields": list(contract.RESULT_DISPOSITION_FIELDS),
            "implementation_fields": list(contract.IMPLEMENTATION_FIELDS),
            "statuses": list(contract.RESULT_STATUSES),
            "non_identity_execution_metadata": list(contract.NON_IDENTITY_EXECUTION_METADATA),
        },
        "count_authority": {
            "authoritative_fields": list(contract.COUNT_FIELDS_AUTHORITATIVE),
            "trial_definition": contract.TRIAL_DEFINITION,
            "comparison_domain": contract.COMPARISON_DOMAIN,
            "bit_error_definition": contract.BIT_ERROR_DEFINITION,
            "block_error_definition": contract.BLOCK_ERROR_DEFINITION,
            "decoder_exception_policy": contract.DECODER_EXCEPTION_POLICY,
            "cross_count_invariants": list(contract.COUNT_CROSS_INVARIANTS),
            "bler_rule": contract.BLER_POINT_ESTIMATE_RULE,
            "ber_rule": contract.BER_POINT_ESTIMATE_RULE,
            "counts_override_stored_floats": contract.COUNTS_OVERRIDE_STORED_FLOATS,
            "information_bits_rule": "trials_completed x K",
            "zero_errors_is_characterized_evidence": True,
            "all_errors_is_characterized_evidence": True,
            "zero_completed_trials_reports_null_not_zero": True,
            "negative_counts_rejected": True,
            "boolean_counts_rejected": True,
            "nan_and_infinity_rejected": True,
            "completed_evidence_requires_positive_trials": True,
        },
        "merge_rules": {
            "incomplete_result_is_merge_eligible": False,
            "failed_result_is_merge_eligible": False,
            "bounded_smoke_is_merge_eligible": False,
            "full_strength_merge_requires_exact_trial_count": True,
            "changed_identity_with_copied_counts_rejected": True,
            "runtime_metadata_cannot_alter_measurement_identity": True,
        },
        "confidence": contract.confidence_policy(),
        "rules": {
            "no_interpolation_or_extrapolation": contract.NO_INTERPOLATION_RULE,
            "test_split_access": contract.TEST_SPLIT_ACCESS,
            "one_work_unit_matches_exactly_one_required_entry": True,
            "snr_never_rounded_or_coerced": True,
        },
        "contract_sources": [
            _binding(path, role="g8b_b1_contract_source") for path in CONTRACT_SOURCES
        ],
    }
    payload["contract_id"] = contract_identifier(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    payload = build()
    if args.check:
        if not BLER_TOOLING_CONTRACT.exists():
            raise SystemExit(f"missing {BLER_TOOLING_CONTRACT.relative_to(REPO_ROOT)}")
        if BLER_TOOLING_CONTRACT.read_bytes() != rendered_json(payload):
            raise SystemExit("bler_tooling_contract.json is stale")
        print(
            "ok: BLER tooling contract matches the regenerated B1 freeze "
            f"contract_id={payload['contract_id']}"
        )
        return 0
    digest = write_json_atomically(BLER_TOOLING_CONTRACT, payload)
    print(
        f"wrote {BLER_TOOLING_CONTRACT.relative_to(REPO_ROOT)} "
        f"contract_id={payload['contract_id']} sha256={digest}; "
        f"full_strength_trials={payload['trial_count']['full_strength_trials']} "
        f"required_work_units={payload['campaign_bindings']['required_work_unit_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
