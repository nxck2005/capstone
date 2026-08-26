#!/usr/bin/env python3
"""Fail-closed verifier and compact completion builder for W5."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
W5 = REPO / "results/learned/w5"
SCHEMA_PATH = REPO / "spec/schemas/w5_training_artifacts.schema.json"
CONTRACT_PATH = REPO / "instructions/W5.txt"
SOURCE_MANIFEST_PATH = W5 / "w5_source_manifest_v2.json"
SMOKE_PATH = W5 / "w5_smoke_result.json"
COMPLETION_PATH = W5 / "w5_completion.json"
G8_CLOSEOUT_PATH = REPO / "results/baseline/g8/g8_closeout.json"
G8_CORRECTION_PATH = REPO / "results/baseline/g8/g8_terminal_binding_metadata_correction.json"

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from config.params import get  # noqa: E402
from training.djscc import ELIGIBILITY, PROTECTED_COUNTERS  # noqa: E402
from gen_w5_source_manifest import verify as verify_source_manifest  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError(f"invalid/missing JSON: {path.relative_to(REPO)}") from None
    if not isinstance(value, dict):
        raise ValueError(f"JSON is not an object: {path.relative_to(REPO)}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_schema() -> dict[str, Any]:
    schema = _load(SCHEMA_PATH)
    _require(schema.get("schema_version") == 1, "W5 schema version differs")
    _require(schema.get("eligibility_constants") == ELIGIBILITY, "W5 schema eligibility constants differ")
    definitions = schema.get("$defs")
    _require(isinstance(definitions, dict) and set(definitions) == {
        "sha256", "git_commit", "eligibility", "lineage", "protected_counters",
        "checkpoint", "checkpoint_sidecar", "smoke_result", "completion",
    }, "W5 schema definitions differ")
    _require(definitions["protected_counters"]["required"] == list(PROTECTED_COUNTERS), "W5 schema protected-counter order/set differs")
    _require(all(definitions["protected_counters"]["properties"][key] == {"const": 0} for key in PROTECTED_COUNTERS), "W5 schema protected counters are not exact zero")
    return schema


def verify_contract() -> dict[str, str]:
    raw = CONTRACT_PATH.read_bytes()
    _require(b"W5 DOES NOT" not in raw or b"W7" in raw, "W5 contract scope boundary is malformed")
    for token in (
        b"W5_NON_SCIENTIFIC_PLUMBING_SMOKE",
        b"NOT_ELIGIBLE_FOR_W7_G4",
        b"NOT_ELIGIBLE_FOR_W8",
        b"NOT_ELIGIBLE_FOR_TEST",
        b"W6 requires separate owner authorization",
    ):
        _require(token in raw, f"W5 contract missing {token.decode()}")
    digest = hashlib.sha256(raw).hexdigest()
    return {"path": str(CONTRACT_PATH.relative_to(REPO)), "contract_id": "w5contract-" + digest, "sha256": digest}


def verify_g8_lineage() -> dict[str, Any]:
    closeout = _load(G8_CLOSEOUT_PATH)
    correction = _load(G8_CORRECTION_PATH)
    _require(closeout.get("closeout_id") == "g8closeout-07526958639a3b0040c45264d0ec10e51ee3269755b5d3f8aac48c4c2f3ef2a7", "G8 closeout ID differs")
    _require(correction.get("correction_id") == "g8bindingcorrection-1bff458ee803b41599d969016da6b04d393b2a425df3b5fc9fa0e9e823523610", "G8 metadata correction ID differs")
    _require(all(value == 0 for value in correction["scientific_boundary"].values()), "G8 metadata correction scientific boundary is nonzero")
    return {
        "closeout_id": closeout["closeout_id"],
        "closeout_path": str(G8_CLOSEOUT_PATH.relative_to(REPO)),
        "closeout_sha256": _sha(G8_CLOSEOUT_PATH),
        "metadata_correction_id": correction["correction_id"],
        "metadata_correction_path": str(G8_CORRECTION_PATH.relative_to(REPO)),
        "metadata_correction_sha256": _sha(G8_CORRECTION_PATH),
    }


def verify_source() -> dict[str, Any]:
    manifest = _load(SOURCE_MANIFEST_PATH)
    verify_source_manifest(manifest, current=True)
    return {
        "path": str(SOURCE_MANIFEST_PATH.relative_to(REPO)),
        "manifest_id": manifest["manifest_id"],
        "file_sha256": _sha(SOURCE_MANIFEST_PATH),
        "source_commit": manifest["source_commit"],
        "entries": len(manifest["entries"]),
    }


def _identity(prefix: str, value: dict[str, Any], identity_field: str) -> str:
    body = dict(value)
    identity = body.pop(identity_field)
    expected = prefix + hashlib.sha256(_canonical(body)).hexdigest()
    _require(identity == expected, f"{identity_field} does not match canonical content")
    return identity


def verify_smoke(runtime_root: Path | None = None) -> dict[str, Any]:
    smoke = _load(SMOKE_PATH)
    required = {
        "schema_version", "artifact_role", "smoke_id", "eligibility", "lineage",
        "scope", "environment", "training", "gradients", "checkpoint_resume",
        "selected_ratio_plumbing", "data_isolation", "protected_counters",
    }
    _require(set(smoke) == required and smoke["schema_version"] == 1 and smoke["artifact_role"] == "w5_djscc_smoke_result", "W5 smoke schema/role differs")
    _identity("w5smoke-", smoke, "smoke_id")
    _require(smoke["eligibility"] == ELIGIBILITY, "W5 smoke eligibility differs")
    _require(smoke["protected_counters"] == PROTECTED_COUNTERS, "W5 smoke protected counters differ")
    _require(smoke["scope"] == {
        "role": "NON_SCIENTIFIC_W5_PLUMBING_ONLY",
        "dataset": "cifar10",
        "lambda": smoke["scope"]["lambda"],
        "lambda_status": "provisional_until_G-4",
        "smoke_only_max_microbatches_per_epoch": 1,
        "accuracy_recorded": False,
        "selection_performed": False,
    }, "W5 smoke scope differs")
    _require(smoke["scope"]["lambda"] in (0, get("learned_system.lambda_core")), "W5 smoke lambda is outside authorized plumbing inputs")
    _require(smoke["checkpoint_resume"]["process_boundary"] is True and smoke["checkpoint_resume"]["fresh_process_resume"] is True and smoke["checkpoint_resume"]["exact"] is True, "W5 fresh-process resume proof differs")
    _require(all(smoke["checkpoint_resume"]["comparison"].values()), "W5 uninterrupted/resumed comparison differs")
    selected = smoke["selected_ratio_plumbing"]
    expected_ratios = {get("bandwidth.crossover_ratio"), get("bandwidth.efficiency_ratio"), get("bandwidth.low_ratio_operating_point")}
    _require(set(selected) == expected_ratios == {"r_1_6", "r_1_24"}, "W5 selected-ratio set differs")
    for ratio, value in selected.items():
        _require(value["dataset"] == "imagenette160" and value["k"] == get(f"bandwidth.k_symbols.imagenette160.{ratio}"), f"W5 selected-ratio k differs for {ratio}")
        _require(value["steps"] > 0 and value["samples"] > 0, f"W5 selected-ratio coverage is zero for {ratio}")
        _require(all(status["finite"] and status["nonzero"] for status in value["gradient_checks"].values()), f"W5 selected-ratio gradient differs for {ratio}")
    _require(all(value == 0 for value in smoke["data_isolation"].values()), "W5 data-isolation counter is nonzero")
    _require(smoke["training"]["w5_non_scientific_optimizer_steps"] > 0, "W5 smoke optimizer steps were not counted")
    _require(smoke["training"]["finite_total_ce_mse"] is True, "W5 smoke loss is non-finite")
    if runtime_root is not None:
        for pointer in sorted(runtime_root.glob("*/latest.json")):
            sidecar = _load(pointer)
            checkpoint = pointer.parent / sidecar["checkpoint_path"]
            _require(checkpoint.is_file() and not checkpoint.is_symlink(), f"W5 checkpoint missing/unsafe: {checkpoint}")
            _require(checkpoint.stat().st_size == sidecar["checkpoint_bytes"] and _sha(checkpoint) == sidecar["checkpoint_id"], f"W5 checkpoint bytes differ: {checkpoint}")
            _require(sidecar["eligibility"] == ELIGIBILITY, f"W5 checkpoint eligibility differs: {checkpoint}")
    return {
        "path": str(SMOKE_PATH.relative_to(REPO)),
        "smoke_id": smoke["smoke_id"],
        "file_sha256": _sha(SMOKE_PATH),
        "optimizer_steps": smoke["training"]["w5_non_scientific_optimizer_steps"],
        "samples": smoke["training"]["samples_across_all_physical_smoke_trajectories"],
    }


def build_completion(verification_record: Path) -> dict[str, Any]:
    schema = verify_schema()
    contract = verify_contract()
    g8 = verify_g8_lineage()
    source = verify_source()
    smoke = verify_smoke(W5 / "runtime_attempt_2")
    verification = _load(verification_record)
    body = {
        "schema_version": 1,
        "artifact_role": "w5_training_infrastructure_completion",
        "w5_contract": contract,
        "w5_schema": {
            "version": schema["schema_version"],
            "path": str(SCHEMA_PATH.relative_to(REPO)),
            "schema_id": "w5schema-" + _sha(SCHEMA_PATH),
            "sha256": _sha(SCHEMA_PATH),
        },
        "g8_lineage": g8,
        "source_lineage": source,
        "recipe": {
            "amendment": "AM-91",
            "lambda_status": get("learned_system.lambda_status"),
            "optimizer": get("learned_system.optimizer_implementation"),
            "lr": get("learned_system.lr"),
            "schedule": get("learned_system.lr_schedule_equation"),
            "warmup_epochs": get("learned_system.lr_warmup_epochs"),
            "amp": get("learned_system.amp"),
            "amp_dtype": get("learned_system.amp_dtype"),
            "checkpoint_every_epochs": get("learned_system.checkpoint_every_epochs"),
            "selection": get("learned_system.w5_checkpoint_selection"),
        },
        "rng_policy": {
            "stream": get("artifacts.rng_stream"),
            "training_purpose": "training_channel_noise",
            "identity_fields": list(get("artifacts.rng_identity_fields.training_channel_noise")),
            "sequential_channel_rng_state": None,
        },
        "smoke_lineage": smoke,
        "verification": verification,
        "protected_counters": dict(PROTECTED_COUNTERS),
        "next_gate": "SEPARATE_W6_OWNER_AUTHORIZATION_REQUIRED",
    }
    body["completion_id"] = "w5completion-" + hashlib.sha256(_canonical(body)).hexdigest()
    return body


def verify_completion() -> dict[str, Any]:
    completion = _load(COMPLETION_PATH)
    required = {
        "schema_version", "artifact_role", "completion_id", "w5_contract", "w5_schema",
        "g8_lineage", "source_lineage", "recipe", "rng_policy", "smoke_lineage",
        "verification", "protected_counters", "next_gate",
    }
    _require(set(completion) == required and completion["schema_version"] == 1 and completion["artifact_role"] == "w5_training_infrastructure_completion", "W5 completion schema/role differs")
    _identity("w5completion-", completion, "completion_id")
    _require(completion["w5_contract"] == verify_contract(), "W5 completion contract binding differs")
    _require(completion["w5_schema"]["sha256"] == _sha(SCHEMA_PATH), "W5 completion schema binding differs")
    _require(completion["g8_lineage"] == verify_g8_lineage(), "W5 completion G8 binding differs")
    _require(completion["source_lineage"] == verify_source(), "W5 completion source binding differs")
    _require(
        completion["smoke_lineage"] == verify_smoke(W5 / "runtime_attempt_2"),
        "W5 completion smoke binding differs",
    )
    _require(completion["protected_counters"] == PROTECTED_COUNTERS, "W5 completion protected counters differ")
    _require(completion["next_gate"] == "SEPARATE_W6_OWNER_AUTHORIZATION_REQUIRED", "W5 completion next gate differs")
    return completion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-source", action="store_true")
    parser.add_argument("--generate-completion", action="store_true")
    parser.add_argument("--verification-record", type=Path)
    args = parser.parse_args()
    verify_schema()
    contract = verify_contract()
    verify_g8_lineage()
    if args.pre_source:
        print(f"W5 pre-source contract/schema PASS: {contract['contract_id']}")
        return 0
    if args.generate_completion:
        if args.verification_record is None:
            parser.error("--generate-completion requires --verification-record")
        value = build_completion(args.verification_record)
        COMPLETION_PATH.parent.mkdir(parents=True, exist_ok=True)
        COMPLETION_PATH.write_bytes(_canonical(value))
        print(f"wrote {COMPLETION_PATH.relative_to(REPO)}: {value['completion_id']}")
        return 0
    completion = verify_completion()
    print(f"W5 GREEN verifier PASS: {completion['completion_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
