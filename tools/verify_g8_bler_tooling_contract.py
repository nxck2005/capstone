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
        bindings["required_bler_identities"]["sha256"] == sha256_file(REQUIRED_BLER_IDENTITIES),
        "bound required-identity hash changed",
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


def _verify_seeds(payload: dict[str, Any]) -> None:
    seed = payload["seed"]
    _require(
        seed["derivation_identity"] == contract.SEED_DERIVATION_IDENTITY,
        "seed derivation identity drifted from the live campaign state string",
    )
    _require(seed["domain_separator"] == contract.SEED_DOMAIN_SEPARATOR, "seed domain separator changed")
    _require(seed["digest"] == "sha256", "seed digest changed")
    _require(seed["purposes"] == list(contract.SEED_PURPOSES), "the allowed random purposes changed")
    _require(seed["width_bits"] == 64, "the seed width changed")  # literal-ok: uint64 seed width

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
    _require(result["statuses"] == list(contract.RESULT_STATUSES), "the result status enum changed")
    _require(
        result["non_identity_execution_metadata"] == list(contract.NON_IDENTITY_EXECUTION_METADATA),
        "runtime provenance fields leaked into or out of measurement identity",
    )

    counts = payload["count_authority"]
    _require(counts["bler_rule"] == "block_errors / trials_completed", "the BLER estimate rule changed")
    _require(counts["ber_rule"] == "bit_errors / information_bits", "the BER estimate rule changed")
    _require(counts["counts_override_stored_floats"] is True, "stored floats may override counts")
    _require(counts["zero_errors_is_characterized_evidence"] is True,
             "zero observed errors is no longer characterized evidence")
    _require(counts["all_errors_is_characterized_evidence"] is True,
             "an all-error result is no longer characterized evidence")
    _require(counts["zero_completed_trials_reports_null_not_zero"] is True,
             "zero completed trials may report a zero rate")

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
    _require(resume["granularity"] == "work_unit_atomic", "the resume granularity changed")
    _require(resume["mid_work_unit_resume_permitted"] is False, "mid-work-unit resume was permitted")


def verify(path: Path = generator.BLER_TOOLING_CONTRACT) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise G8BlerToolingError(f"cannot read BLER tooling contract {path}: {exc}") from exc
    _require(isinstance(payload, dict), "BLER tooling contract is not a JSON object")
    _require(raw == rendered_json(payload), "BLER tooling contract is not canonical rendered JSON")
    _require(payload.get("campaign") == CAMPAIGN, "the contract names the wrong campaign")
    _require(payload.get("schema_version") == 1, "unsupported tooling contract schema_version")
    _require(payload.get("phase") == "G8_B" and payload.get("checkpoint") == "B1",
             "the contract phase or checkpoint changed")
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
