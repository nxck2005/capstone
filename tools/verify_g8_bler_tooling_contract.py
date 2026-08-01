#!/usr/bin/env python3
"""Independently verify the frozen G8_B BLER tooling contract.

This verifier recomputes every bound quantity from primary sources rather than
trusting the artifact. It performs no characterization, no smoke, and no
scientific execution of any kind.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import gen_g8_bler_tooling_contract as generator  # noqa: E402
from baseline.g8_campaign import (  # noqa: E402
    CAMPAIGN,
    CAMPAIGN_MANIFEST,
    REQUIRED_BLER_IDENTITIES,
    G8ContractError,
    load_campaign_manifest,
    load_required_bler_identities,
    rendered_json,
    sha256_bytes,
    sha256_file,
)
from baseline import g8_bler_contract as contract  # noqa: E402
from config.params import REPO_ROOT, get  # noqa: E402


class G8BlerToolingError(RuntimeError):
    """An independently checked B1 contract invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G8BlerToolingError(message)


EXPECTED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "campaign",
    "artifact_role",
    "phase",
    "checkpoint",
    "supersedes_contract_id",
    "supersedes_contract_sha256",
    "supersession_reason",
    "scientific_execution_performed",
    "characterization_started",
    "bounded_smoke_started",
    "campaign_bindings",
    "trial_count",
    "execution_classes",
    "seed",
    "rng",
    "resume",
    "request_schema",
    "result_schema",
    "count_authority",
    "merge_rules",
    "confidence",
    "rules",
    "contract_sources",
    "contract_id",
}


def _independent_seed(campaign_id: str, work_unit_id: str, purpose: str) -> tuple[str, int]:
    """Re-derive a seed without calling the contract module's helpers."""

    material = (
        "["
        + ",".join(
            json.dumps(part, ensure_ascii=True)
            for part in (contract.SEED_DOMAIN_SEPARATOR, campaign_id, work_unit_id, purpose)
        )
        + "]"
    ).encode("utf-8")
    digest = hashlib.sha256(material)
    return digest.hexdigest(), int.from_bytes(digest.digest()[:8], "big")


def _independent_wilson(errors: int, trials: int, percent: float) -> tuple[float, float]:
    z = NormalDist().inv_cdf(0.5 + percent / 200.0)
    p = errors / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _verify_bindings(payload: dict[str, Any]) -> None:
    for entry in payload["contract_sources"]:
        path = entry["path"]
        _require(not Path(path).is_absolute(), f"contract source path is absolute: {path}")
        _require(path in generator.CONTRACT_SOURCES, f"unexpected bound contract source: {path}")
        _require(entry["role"] == "g8b_b1c_contract_source",
                 f"unexpected role for bound contract source: {path}")
        body = (REPO_ROOT / path).read_bytes()
        _require(entry["bytes"] == len(body), f"bound byte length changed: {path}")
        _require(entry["sha256"] == sha256_bytes(body), f"bound SHA-256 changed: {path}")
    bound = [entry["path"] for entry in payload["contract_sources"]]
    _require(bound == list(generator.CONTRACT_SOURCES), "bound contract-source set or order changed")
    output = str(generator.BLER_TOOLING_CONTRACT.relative_to(REPO_ROOT))
    _require(output not in bound, "the contract binds its own output hash")


def _verify_campaign(payload: dict[str, Any]) -> None:
    bindings = payload["campaign_bindings"]
    manifest = load_campaign_manifest(CAMPAIGN_MANIFEST)
    _require(bindings["campaign_id"] == manifest["campaign_id"], "bound campaign ID changed")
    _require(
        bindings["campaign_manifest"]["sha256"] == sha256_file(CAMPAIGN_MANIFEST),
        "bound campaign manifest hash changed",
    )
    _require(
        bindings["campaign_manifest"]["bytes"] == len(CAMPAIGN_MANIFEST.read_bytes()),
        "bound campaign manifest byte length changed",
    )
    _require(
        bindings["required_bler_identities"]["sha256"] == sha256_file(REQUIRED_BLER_IDENTITIES),
        "bound required-identity hash changed",
    )
    _require(
        bindings["required_bler_identities"]["bytes"] == len(REQUIRED_BLER_IDENTITIES.read_bytes()),
        "bound required-identity byte length changed",
    )
    required = load_required_bler_identities(REQUIRED_BLER_IDENTITIES)
    _require(
        bindings["required_work_unit_count"] == len(required["required_bler_work_units"]),
        "bound required work-unit count is false",
    )
    policy = (manifest.get("selection_policy") or {}).get("selection_policy_sha256")
    _require(bindings["selection_policy_sha256"] == policy, "bound selection-policy hash changed")


def _verify_trial_count(payload: dict[str, Any]) -> None:
    trial = payload["trial_count"]
    _require(
        trial["parameter"] == "baseline.bler_characterisation_trials",
        "the G-8 trial count is not owned by baseline.bler_characterisation_trials",
    )
    _require(
        trial["source"] == "params.baseline.bler_characterisation_trials",
        "the declared trial-count source changed",
    )
    _require(
        trial["g2_reference_key_not_used"] == "params.baseline.ldpc_bler_reference.blocks_per_snr",
        "the excluded G-2 trial-count key changed",
    )
    _require(
        trial["full_strength_trials"] == int(get("baseline.bler_characterisation_trials")),
        "the bound full-strength trial count does not match its own parameter",
    )
    _require(trial["adaptive_stopping_permitted"] is False, "adaptive stopping is permitted")
    _require(
        trial["no_early_stopping_rule"] == contract.NO_EARLY_STOPPING_RULE,
        "the no-early-stopping rule changed",
    )


def _verify_seeds(payload: dict[str, Any]) -> None:
    seed = payload["seed"]
    _require(
        seed["derivation_identity"] == contract.SEED_DERIVATION_IDENTITY,
        "seed derivation identity drifted from the live campaign state string",
    )
    _require(seed["domain_separator"] == contract.SEED_DOMAIN_SEPARATOR, "seed domain separator changed")
    _require(seed["digest"] == "sha256", "seed digest changed")
    _require(seed["input_encoding"] == contract.SEED_INPUT_ENCODING, "seed input encoding changed")
    _require(seed["output_rule"] == contract.SEED_OUTPUT_RULE, "seed output rule changed")
    _require(seed["purposes"] == list(contract.SEED_PURPOSES), "the allowed random purposes changed")
    _require(seed["width_bits"] == 64, "the seed width changed")  # literal-ok: uint64 seed width
    _require(seed["forbidden_inputs"] == list(contract.SEED_FORBIDDEN_INPUTS),
             "the forbidden seed inputs changed")

    vectors = seed["test_vectors"]
    campaign_id = vectors["fixture_campaign_id"]
    work_unit_id = vectors["fixture_work_unit_id"]
    seen: set[int] = set()
    for purpose in contract.SEED_PURPOSES:
        entry = vectors["seeds"][purpose]
        material_sha, value = _independent_seed(campaign_id, work_unit_id, purpose)
        _require(entry["material_sha256"] == material_sha,
                 f"seed material hash for {purpose} does not reproduce")
        _require(entry["seed_uint64"] == value, f"seed vector for {purpose} does not reproduce")
        _require(0 <= value < 2**64, f"seed for {purpose} left the unsigned 64-bit range")
        _require(value not in seen, "two random purposes share a stream seed")
        seen.add(value)
        words = [int(word) for word in np.random.Philox(key=value).random_raw(4)]
        _require(entry["first_raw_words"] == words,
                 f"bound Philox words for {purpose} do not reproduce")
        if purpose == contract.PURPOSE_INFORMATION_BITS:
            _require(entry["bits_0_to_8"] == [(words[0] >> i) & 1 for i in range(8)],
                     "bound information bits 0..7 do not reproduce")
            _require(
                entry["bits_60_to_68"] == [(words[i // 64] >> (i % 64)) & 1 for i in range(60, 68)],
                "bound information bits 60..67 do not reproduce",
            )
        else:
            drawn = [
                float(draw)
                for draw in np.random.Generator(np.random.Philox(key=value)).standard_normal(4)
            ]
            _require(entry["first_normals"] == drawn,
                     f"bound normal draws for {purpose} do not reproduce")


def _verify_confidence(payload: dict[str, Any]) -> None:
    confidence = payload["confidence"]
    _require(confidence["method"] == "wilson_score", "the confidence method changed")
    percent = confidence["percent"]
    _require(percent == 95, "the diagnostic confidence level changed")  # literal-ok: G8-owned diagnostic level
    _require(confidence["adaptive_stopping_permitted"] is False,
             "the confidence policy permits adaptive stopping")
    _require(
        "diagnostic only" in confidence["role"]
        and "not used in BR-4 ranking or eligibility" in confidence["role"]
        and "not a stopping rule" in confidence["role"],
        "the confidence interval is no longer declared diagnostic only",
    )
    vectors = payload["seed"]["test_vectors"]["wilson"]
    for name, (errors, trials) in {
        "zero_errors_16_trials": (0, 16),
        "one_error_16_trials": (1, 16),
        "all_errors_16_trials": (16, 16),
    }.items():
        low, high = _independent_wilson(errors, trials, percent)
        _require(vectors[name] == [low, high], f"bound Wilson vector {name} does not reproduce")


def _verify_schemas(payload: dict[str, Any]) -> None:
    request = payload["request_schema"]
    result = payload["result_schema"]
    _require(
        request["version"] == contract.BLER_WORK_UNIT_REQUEST_SCHEMA_VERSION == 2,
        "the request schema version is not v2",
    )
    _require(
        result["version"] == contract.BLER_WORK_UNIT_RESULT_SCHEMA_VERSION == 2,
        "the result schema version is not v2",
    )
    _require(request["artifact_role"] == contract.REQUEST_ARTIFACT_ROLE,
             "the request artifact role changed")
    _require(result["artifact_role"] == contract.RESULT_ARTIFACT_ROLE,
             "the result artifact role changed")
    _require(request["fields"] == list(contract.REQUEST_FIELDS), "the request schema changed")
    _require(request["unknown_fields_rejected"] is True, "unknown request fields are tolerated")
    _require(request["request_is_never_merge_eligible"] is True,
             "a request may claim merge eligibility")
    _require(request["test_split_access"] == 0, "the request schema permits test-split access")
    _require(result["sections"] == list(contract.RESULT_FIELDS), "the result section set changed")
    _require(result["identity_fields"] == list(contract.RESULT_IDENTITY_FIELDS),
             "the result identity schema changed")
    _require(result["measurement_fields"] == list(contract.RESULT_MEASUREMENT_FIELDS),
             "the result measurement schema changed")
    _require(result["execution_metadata_fields"] == list(contract.RESULT_EXECUTION_METADATA_FIELDS),
             "the result execution-metadata schema changed")
    _require(result["disposition_fields"] == list(contract.RESULT_DISPOSITION_FIELDS),
             "the result disposition schema changed")
    _require(result["implementation_fields"] == list(contract.IMPLEMENTATION_FIELDS),
             "the result implementation schema changed")
    _require(result["statuses"] == list(contract.RESULT_STATUSES), "the result status enum changed")
    _require(result["status_rules"] == contract.RESULT_STATUS_RULES,
             "the result status rules changed")
    _require(
        result["non_identity_execution_metadata"] == list(contract.NON_IDENTITY_EXECUTION_METADATA),
        "runtime provenance fields leaked into or out of measurement identity",
    )
    _require(result["execution_metadata_rules"] == contract.EXECUTION_METADATA_RULES,
             "the execution metadata rules changed")

    counts = payload["count_authority"]
    _require(counts["authoritative_fields"] == list(contract.COUNT_FIELDS_AUTHORITATIVE),
             "the authoritative count fields changed")
    _require(
        counts["trial_definition"] == contract.TRIAL_DEFINITION,
        "the trial definition changed",
    )
    _require(
        counts["comparison_domain"] == contract.COMPARISON_DOMAIN,
        "the count comparison domain changed",
    )
    _require(
        counts["bit_error_definition"] == contract.BIT_ERROR_DEFINITION,
        "the bit-error definition changed",
    )
    _require(
        counts["block_error_definition"] == contract.BLOCK_ERROR_DEFINITION,
        "the block-error definition changed",
    )
    _require(
        counts["decoder_exception_policy"] == contract.DECODER_EXCEPTION_POLICY,
        "the decoder-exception policy changed",
    )
    _require(
        counts["cross_count_invariants"] == list(contract.COUNT_CROSS_INVARIANTS),
        "the cross-count invariants changed",
    )
    _require(counts["bler_rule"] == "block_errors / trials_completed", "the BLER estimate rule changed")
    _require(counts["ber_rule"] == "bit_errors / information_bits", "the BER estimate rule changed")
    _require(counts["information_bits_rule"] == "trials_completed x K",
             "the information-bit count rule changed")
    _require(counts["counts_override_stored_floats"] is True, "stored floats may override counts")
    _require(counts["zero_errors_is_characterized_evidence"] is True,
             "zero observed errors is no longer characterized evidence")
    _require(counts["all_errors_is_characterized_evidence"] is True,
             "an all-error result is no longer characterized evidence")
    _require(counts["zero_completed_trials_reports_null_not_zero"] is True,
             "zero completed trials may report a zero rate")
    for name in (
        "negative_counts_rejected",
        "boolean_counts_rejected",
        "nan_and_infinity_rejected",
        "completed_evidence_requires_positive_trials",
    ):
        _require(counts[name] is True, f"count rule {name} was weakened")

    merge = payload["merge_rules"]
    for name in (
        "incomplete_result_is_merge_eligible",
        "failed_result_is_merge_eligible",
        "bounded_smoke_is_merge_eligible",
    ):
        _require(merge[name] is False, f"merge rule {name} weakened")
    _require(merge["full_strength_merge_requires_exact_trial_count"] is True,
             "full-strength merge no longer requires the exact trial count")
    _require(merge["runtime_metadata_cannot_alter_measurement_identity"] is True,
             "runtime metadata may alter measurement identity")

    classes = payload["execution_classes"]
    _require(classes["full_strength"] == contract.EXECUTION_CLASS_FULL_STRENGTH,
             "full-strength execution class changed")
    _require(classes["bounded_smoke"] == contract.EXECUTION_CLASS_BOUNDED_SMOKE,
             "bounded-smoke execution class changed")
    _require(classes["bounded_smoke_trial_count_source"] == contract.BOUNDED_SMOKE_TRIAL_COUNT_SOURCE,
             "bounded-smoke trial-count source changed")
    _require(classes["bounded_smoke_selection_rule"] == contract.BOUNDED_SMOKE_SELECTION_RULE,
             "bounded-smoke selection rule changed")
    _require(classes["bounded_smoke_label"] == contract.BOUNDED_SMOKE_LABEL,
             "bounded-smoke label changed")
    _require(classes["bounded_smoke_max_work_units"] == contract.BOUNDED_SMOKE_MAX_WORK_UNITS,
             "bounded-smoke work-unit ceiling changed")
    _require(classes["bounded_smoke_max_trials_per_unit"] == contract.BOUNDED_SMOKE_MAX_TRIALS_PER_UNIT,
             "bounded-smoke trial ceiling changed")
    _require(classes["bounded_smoke_is_scientific_evidence"] is False,
             "bounded smoke claims scientific evidence")
    _require(classes["bounded_smoke_is_merge_eligible"] is False, "bounded smoke claims merge eligibility")
    _require(classes["bounded_smoke_required_coverage_contribution"] == 0,
             "bounded smoke claims required coverage")
    _require(
        classes["bounded_smoke_max_trials_per_unit"] < payload["trial_count"]["full_strength_trials"],
        "the bounded-smoke ceiling is not below the full-strength trial count",
    )
    _require(classes["bounded_smoke_max_work_units"] > 0, "the bounded-smoke work-unit ceiling is unset")


def _verify_rng(payload: dict[str, Any]) -> None:
    rng = payload["rng"]
    _require(rng["library"] == "numpy", "the bound RNG library changed")
    _require(rng["library_version"] == str(np.__version__),
             "the bound RNG library version does not match the locked environment")
    _require(rng["bit_generator"] == "Philox", "the bound bit generator changed")
    _require(rng["chunk_boundary_invariant"] is True, "chunk-boundary invariance was dropped")
    _require(rng["purposes"] == list(contract.SEED_PURPOSES), "the RNG purposes changed")
    _require(
        rng["information_bit_extraction"] == "bit_i = (word[i // 64] >> (i % 64)) & 1",
        "the information-bit extraction rule changed",
    )
    # Prove the declared invariance rather than restating it.
    seed = 1234567890123456789
    whole = [int(bit) for bit in contract.information_bit_stream(seed, 0, 200)]
    pieces: list[int] = []
    start = 0
    for size in (3, 61, 1, 70, 65):
        pieces.extend(int(bit) for bit in contract.information_bit_stream(seed, start, size))
        start += size
    _require(whole == pieces, "the information-bit stream is not chunk-boundary invariant")
    reference = list(np.random.Generator(np.random.Philox(key=seed)).standard_normal(200))
    chunked: list[float] = []
    generator_ = np.random.Generator(np.random.Philox(key=seed))
    for size in (3, 61, 1, 70, 65):
        chunked.extend(generator_.standard_normal(size))
    _require(reference == chunked, "the Gaussian stream is not chunk-boundary invariant")


def _verify_resume(payload: dict[str, Any]) -> None:
    resume = payload["resume"]
    _require(resume["granularity"] == contract.RESUME_GRANULARITY, "the resume granularity changed")
    _require(resume["policy"] == contract.RESUME_POLICY, "the resume policy changed")
    _require(resume["mid_work_unit_resume_permitted"] is contract.MID_WORK_UNIT_RESUME_PERMITTED,
             "mid-work-unit resume was permitted")


def _verify_rules(payload: dict[str, Any]) -> None:
    rules = payload["rules"]
    _require(rules["no_interpolation_or_extrapolation"] == contract.NO_INTERPOLATION_RULE,
             "the no-interpolation rule changed")
    _require(rules["test_split_access"] == contract.TEST_SPLIT_ACCESS,
             "the contract permits test-split access")
    _require(rules["one_work_unit_matches_exactly_one_required_entry"] is True,
             "one work unit may match more than one required entry")
    _require(rules["snr_never_rounded_or_coerced"] is True,
             "SNR rounding or coercion is permitted")


def verify(path: Path = generator.BLER_TOOLING_CONTRACT) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise G8BlerToolingError(f"cannot read BLER tooling contract {path}: {exc}") from exc
    _require(isinstance(payload, dict), "BLER tooling contract is not a JSON object")
    _require(set(payload) == EXPECTED_TOP_LEVEL_FIELDS,
             "BLER tooling contract has unknown or missing top-level keys")
    _require(raw == rendered_json(payload), "BLER tooling contract is not canonical rendered JSON")
    _require(sha256_file(path) == sha256_bytes(raw),
             "current BLER tooling contract SHA-256 cannot be reproduced")
    _require(payload.get("campaign") == CAMPAIGN, "the contract names the wrong campaign")
    _require(payload.get("artifact_role") == contract.TOOLING_CONTRACT_ARTIFACT_ROLE,
             "the contract artifact role changed")
    _require(payload.get("schema_version") == contract.BLER_TOOLING_CONTRACT_SCHEMA_VERSION == 2,
             "unsupported tooling contract schema_version")
    _require(payload.get("phase") == contract.TOOLING_CONTRACT_PHASE
             and payload.get("checkpoint") == contract.TOOLING_CONTRACT_CHECKPOINT,
             "the contract phase or checkpoint changed")
    _require(payload.get("supersedes_contract_id") == contract.SUPERSEDES_CONTRACT_ID,
             "the superseded contract ID changed")
    _require(payload.get("supersedes_contract_sha256") == contract.SUPERSEDES_CONTRACT_SHA256,
             "the superseded contract SHA-256 changed")
    _require(payload.get("supersession_reason") == contract.SUPERSESSION_REASON,
             "the supersession reason changed")
    _require(payload.get("scientific_execution_performed") is False,
             "the tooling contract claims scientific execution")
    _require(payload.get("characterization_started") is False,
             "the tooling contract claims characterization started")
    _require(payload.get("bounded_smoke_started") is False,
             "the tooling contract claims bounded smoke started")

    _require(
        payload.get("contract_id") == generator.contract_identifier(payload),
        "contract_id does not reproduce from the contract content",
    )
    _verify_bindings(payload)
    _verify_campaign(payload)
    _verify_trial_count(payload)
    _verify_seeds(payload)
    _verify_confidence(payload)
    _verify_schemas(payload)
    _verify_rng(payload)
    _verify_resume(payload)
    _verify_rules(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=generator.BLER_TOOLING_CONTRACT)
    args = parser.parse_args(argv)
    try:
        payload = verify(args.contract)
    except (G8BlerToolingError, G8ContractError) as exc:
        raise SystemExit(f"G8 BLER tooling contract HOLD: {exc}") from exc
    print(
        "G8 BLER tooling contract PASS: "
        f"contract_id={payload['contract_id']}, "
        f"full_strength_trials={payload['trial_count']['full_strength_trials']} "
        f"from {payload['trial_count']['source']}, "
        f"required_work_units={payload['campaign_bindings']['required_work_unit_count']}, "
        f"purposes={len(payload['seed']['purposes'])}, "
        "characterization=false, smoke=false, test_split_access=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
