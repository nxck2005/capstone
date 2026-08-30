#!/usr/bin/env python3
"""Fail-closed verifier and compact completion builder for W5."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
W5 = REPO / "results/learned/w5"
SCHEMA_PATH = REPO / "spec/schemas/w5_training_artifacts.schema.json"
CONTRACT_PATH = REPO / "instructions/W5.txt"
HISTORICAL_SOURCE_MANIFEST_PATH = W5 / "w5_source_manifest_v3.json"
HISTORICAL_SMOKE_PATH = W5 / "w5_smoke_result.json"
HISTORICAL_COMPLETION_PATH = W5 / "w5_completion.json"
SOURCE_MANIFEST_PATH = W5 / "w5_source_manifest_v4.json"
SMOKE_PATH = W5 / "w5_smoke_result_attempt_4_schema1.json"
COMPLETION_PATH = W5 / "w5_gradscaler_accounting_repair_completion.json"
HISTORICAL_COMPLETION_ID = "w5completion-680b2688dc761a30a7a68aee91c021fe057bbb726b44b614bdffd19712c5fc70"
HISTORICAL_COMPLETION_SHA256 = "cb5afdf0f0d85742b27a82ee27265c592b598f522a1f6bb3b5ef30c78ca5a539"
HISTORICAL_SMOKE_ID = "w5smoke-9868ecda4c29b61e21b055b78ca315fea2eb51d4bbea80414a70ae17b606e67a"
HISTORICAL_SMOKE_SHA256 = "2dc04add556614dba643bff9848232c1e9de3aee5da07e8d259e70bc72da463a"
W7C_CURRENT_VERIFIER_PROJECTION_SHA256 = "b0c737391df046bbce3e3ba09b18ae7ca5d656019660a9c2d80064a2c80229f3"
W5_RECORDED_VERIFIER_SHA256 = "a4f8dbd34d9b76b600d9fc384856275b0570329c979eb10191f8e5a2a6f96dab"
W5_HISTORICAL_LAMBDA = 1.0  # literal-ok: immutable pre-G4 W5 smoke scope
G8_CLOSEOUT_PATH = REPO / "results/baseline/g8/g8_closeout.json"
G8_CORRECTION_PATH = REPO / "results/baseline/g8/g8_terminal_binding_metadata_correction.json"

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from config.params import get  # noqa: E402
from baseline.w7c_source_compatibility import load as load_w7c_source_compatibility  # noqa: E402
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


def _w7c_verifier_projection(source: bytes) -> bytes:
    projected, count = re.subn(
        rb'(?m)^W7C_CURRENT_VERIFIER_PROJECTION_SHA256\s*=\s*["\'][0-9a-f]{64}["\']',
        b'W7C_CURRENT_VERIFIER_PROJECTION_SHA256 = "<exact-compatibility-binding>"',
        source,
        count=1,
    )
    _require(count == 1, "W7-C verifier compatibility binding is missing")
    return projected


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


def verify_historical_completion() -> dict[str, Any]:
    completion = _load(HISTORICAL_COMPLETION_PATH)
    _require(_sha(HISTORICAL_COMPLETION_PATH) == HISTORICAL_COMPLETION_SHA256, "historical W5 completion bytes differ")
    _require(completion.get("completion_id") == HISTORICAL_COMPLETION_ID, "historical W5 completion ID differs")
    _identity("w5completion-", completion, "completion_id")
    _require(completion.get("artifact_role") == "w5_training_infrastructure_completion", "historical W5 completion role differs")
    _require(completion.get("protected_counters") == PROTECTED_COUNTERS, "historical W5 completion protected counters differ")
    smoke = _load(HISTORICAL_SMOKE_PATH)
    _require(_sha(HISTORICAL_SMOKE_PATH) == HISTORICAL_SMOKE_SHA256, "historical W5 attempt-3 smoke bytes differ")
    _require(smoke.get("smoke_id") == HISTORICAL_SMOKE_ID, "historical W5 attempt-3 smoke ID differs")
    _require(completion.get("smoke_lineage", {}).get("smoke_id") == HISTORICAL_SMOKE_ID, "historical W5 completion smoke binding differs")
    manifest = _load(HISTORICAL_SOURCE_MANIFEST_PATH)
    _require(
        _sha(HISTORICAL_SOURCE_MANIFEST_PATH)
        == "2a734bb2f07aa8a506050d25d068197d8a9209a82bdbea1d099208cd797b9841",
        "historical W5 source-manifest bytes differ",
    )
    _require(
        manifest.get("manifest_id")
        == "w5source-170e34c0bc1f7bfe4b2ec2c48b12dbf856bd59b48e9dae23570917b14b9b0015",
        "historical W5 source-manifest ID differs",
    )
    _require(completion.get("source_lineage", {}).get("manifest_id") == manifest["manifest_id"], "historical W5 completion source binding differs")
    return {
        "path": str(HISTORICAL_COMPLETION_PATH.relative_to(REPO)),
        "completion_id": HISTORICAL_COMPLETION_ID,
        "file_sha256": HISTORICAL_COMPLETION_SHA256,
        "status": "PRESERVED_HISTORICAL_NON_SCIENTIFIC_EVIDENCE",
    }


def verify_source() -> dict[str, Any]:
    manifest = _load(SOURCE_MANIFEST_PATH)
    verify_source_manifest(manifest, current=False)
    # Attempt 4 remains bound to every exact source byte in v4. Only this
    # post-execution terminal verifier may advance additively; its current bytes
    # are bound separately below rather than misrepresented as execution bytes.
    post_execution_roles = {
        "w5_verifier", "w5_verifier_mutation_regression",
        "fresh_process_smoke_orchestrator",
    }
    w7c = None
    for entry in manifest["entries"]:
        if entry["role"] in post_execution_roles:
            continue
        path = REPO / entry["path"]
        current_matches = (
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == entry["bytes"]
            and _sha(path) == entry["sha256"]
        )
        if not current_matches and entry["path"] in {"spec/SPEC.md", "spec/params.generated.yaml"}:
            if w7c is None:
                try:
                    w7c = load_w7c_source_compatibility(REPO)
                except Exception as exc:
                    raise ValueError(f"W7-C normative source compatibility differs: {exc}") from None
            successor = next(
                item for item in w7c["entries"] if item["path"] == entry["path"]
            )
            current_matches = (
                successor["archived_bytes"] == entry["bytes"]
                and successor["archived_sha256"] == entry["sha256"]
                and successor["current_bytes"] == path.stat().st_size
                and successor["current_sha256"] == _sha(path)
            )
        _require(current_matches, f"W5 current execution source byte drift: {entry['path']}")
    return {
        "path": str(SOURCE_MANIFEST_PATH.relative_to(REPO)),
        "manifest_id": manifest["manifest_id"],
        "file_sha256": _sha(SOURCE_MANIFEST_PATH),
        "source_commit": manifest["source_commit"],
        "entries": len(manifest["entries"]),
        "post_execution_verification_sources": [
            {
                "path": str(path.relative_to(REPO)),
                "sha256": _sha(path),
            }
            for path in (
                Path(__file__).resolve(),
                REPO / "tests/test_w5_training_system.py",
                REPO / "tools/run_w5_training_smoke.py",
            )
        ],
        "post_execution_change_reason": "additive schema-v1 projection and successor closeout verification only; not execution source",
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
    smoke_schema = _load(SCHEMA_PATH)["$defs"]["smoke_result"]
    _require(
        smoke_schema.get("additionalProperties") is False
        and set(smoke_schema.get("required", [])) == required
        and set(smoke_schema.get("properties", {})) == required,
        "W5 committed smoke schema differs from verifier",
    )
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
    _require(
        smoke["scope"]["lambda"] in (0, W5_HISTORICAL_LAMBDA, get("learned_system.lambda_core")),
        "W5 smoke lambda is outside authorized plumbing inputs",
    )
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
    accounting = smoke["training"]["optimizer_step_accounting"]
    _require(
        isinstance(accounting, dict)
        and accounting.get("all_optimizer_owned_gradients_covered") is True
        and accounting.get("actual_applied_optimizer_steps")
        == smoke["training"]["w5_non_scientific_optimizer_steps"],
        "W5 optimizer-step accounting summary differs",
    )
    _require(set(accounting.get("trajectories", {})) == {
        "cifar_uninterrupted", "cifar_resumed", "imagenette_r_1_6", "imagenette_r_1_24",
    }, "W5 optimizer-step accounting trajectory set differs")
    for name, proof in accounting["trajectories"].items():
        _require(
            proof.get("global_optimizer_step_matches_trace") is True
            and proof.get("optimizer_wide_finiteness_matches_applied_markers") is True
            and proof.get("actual_applied_optimizer_steps") >= 0
            and proof.get("grad_scaler_skips") >= 0
            and proof.get("optimizer_parameter_counts"),
            f"W5 optimizer-step accounting proof differs for {name}",
        )
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


def _scientific_boundary() -> dict[str, int]:
    return {
        "g8_scientific_change": 0,
        "f2_optimizer_steps": 0,
        "f3_reruns": 0,
        "pass_two_reruns": 0,
        "pass_three": 0,
        "scientific_learned_training_runs": 0,
        "w7_lambda_pilots": 0,
        "w8_final_runs": 0,
        "learned_validation_selection": 0,
        "learned_test_inference": 0,
        "test_access": 0,
    }


def build_completion(verification_record: Path) -> dict[str, Any]:
    historical = verify_historical_completion()
    source = verify_source()
    smoke = verify_smoke(W5 / "runtime_attempt_4")
    verification = _load(verification_record)
    _require(
        verification.get("artifact_role") == "w5_gradscaler_accounting_repair_verification"
        and verification.get("verdict") == "PASS",
        "W5 repair verification record differs",
    )
    body = {
        "schema_version": 1,
        "artifact_role": "w5_gradscaler_accounting_repair_completion",
        "supersedes": historical,
        "defect": {
            "classification": "IMPLEMENTATION DEFECT",
            "summary": "named-region finiteness could disagree with GradScaler optimizer-wide skip semantics",
            "repair": "all optimizer-owned gradients now determine applied-step accounting",
            "am91_recipe_changed": False,
            "amendment_added": False,
        },
        "g8_lineage": verify_g8_lineage(),
        "source_lineage": source,
        "regression_tests": verification["regression_tests"],
        "pre_smoke_ci": verification["pre_smoke_ci"],
        "attempt_4": smoke,
        "optimizer_step_accounting": {
            "attempt_1_optimizer_steps": 0,
            "attempt_2_optimizer_steps": 2,
            "attempt_3_historically_recorded_optimizer_steps": 4,
            "attempt_3_applied_step_certainty_not_strengthened": True,
            "attempt_4_verified_applied_optimizer_steps": smoke["optimizer_steps"],
            "scientific_learned_optimizer_steps": 0,
        },
        "kill_resume": verification["kill_resume"],
        "selected_ratio_plumbing": verification["selected_ratio_plumbing"],
        "scientific_boundary": _scientific_boundary(),
        "protected_counters": dict(PROTECTED_COUNTERS),
        "verification": verification,
        "supersession_reason": "attempt 3 remains historical non-scientific evidence from the pre-repair accounting implementation; attempt 4 verifies the repaired successor source epoch",
        "next_gate": "SEPARATE_W6_OWNER_AUTHORIZATION_REQUIRED",
    }
    body["repair_id"] = "w5repaircompletion-" + hashlib.sha256(_canonical(body)).hexdigest()
    return body


def verify_completion() -> dict[str, Any]:
    completion = _load(COMPLETION_PATH)
    required = {
        "schema_version", "artifact_role", "repair_id", "supersedes", "defect",
        "g8_lineage", "source_lineage", "regression_tests", "pre_smoke_ci",
        "attempt_4", "optimizer_step_accounting", "kill_resume",
        "selected_ratio_plumbing", "scientific_boundary", "protected_counters",
        "verification", "supersession_reason", "next_gate",
    }
    _require(
        set(completion) == required
        and completion["schema_version"] == 1
        and completion["artifact_role"] == "w5_gradscaler_accounting_repair_completion",
        "W5 repair completion schema/role differs",
    )
    _identity("w5repaircompletion-", completion, "repair_id")
    _require(completion["supersedes"] == verify_historical_completion(), "W5 historical completion binding differs")
    _require(completion["defect"] == {
        "classification": "IMPLEMENTATION DEFECT",
        "summary": "named-region finiteness could disagree with GradScaler optimizer-wide skip semantics",
        "repair": "all optimizer-owned gradients now determine applied-step accounting",
        "am91_recipe_changed": False,
        "amendment_added": False,
    }, "W5 repair defect semantics differ")
    _require(completion["g8_lineage"] == verify_g8_lineage(), "W5 repair G8 binding differs")
    source = verify_source()
    recorded_source = completion["source_lineage"]
    if source != recorded_source:
        source_without_post_execution = dict(source)
        recorded_without_post_execution = dict(recorded_source)
        current_post_execution = source_without_post_execution.pop("post_execution_verification_sources", None)
        recorded_post_execution = recorded_without_post_execution.pop("post_execution_verification_sources", None)
        _require(
            source_without_post_execution == recorded_without_post_execution
            and isinstance(current_post_execution, list)
            and isinstance(recorded_post_execution, list)
            and hashlib.sha256(
                _w7c_verifier_projection(
                    (REPO / "tools/verify_w5_training_system.py").read_bytes()
                )
            ).hexdigest()
            == W7C_CURRENT_VERIFIER_PROJECTION_SHA256
            and current_post_execution[0] == {
                "path": "tools/verify_w5_training_system.py",
                "sha256": _sha(REPO / "tools/verify_w5_training_system.py"),
            }
            and recorded_post_execution[0] == {
                "path": "tools/verify_w5_training_system.py",
                "sha256": W5_RECORDED_VERIFIER_SHA256,
            }
            and current_post_execution[1:] == recorded_post_execution[1:],
            "W5 repair source binding differs",
        )
    _require(completion["attempt_4"] == verify_smoke(W5 / "runtime_attempt_4"), "W5 attempt-4 binding differs")
    _require(completion["scientific_boundary"] == _scientific_boundary(), "W5 repair scientific boundary differs")
    _require(completion["protected_counters"] == PROTECTED_COUNTERS, "W5 repair protected counters differ")
    accounting = completion["optimizer_step_accounting"]
    _require(
        accounting.get("attempt_3_historically_recorded_optimizer_steps") == 4
        and accounting.get("attempt_3_applied_step_certainty_not_strengthened") is True
        and accounting.get("attempt_4_verified_applied_optimizer_steps") == completion["attempt_4"]["optimizer_steps"]
        and accounting.get("scientific_learned_optimizer_steps") == 0,
        "W5 repair optimizer-step custody differs",
    )
    _require(
        completion["verification"].get("artifact_role") == "w5_gradscaler_accounting_repair_verification"
        and completion["verification"].get("verdict") == "PASS",
        "W5 repair verification differs",
    )
    _require(completion["next_gate"] == "SEPARATE_W6_OWNER_AUTHORIZATION_REQUIRED", "W5 repair next gate differs")
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
        historical = verify_historical_completion()
        print(
            "W5 repair pre-source contract/schema/historical evidence PASS: "
            f"{contract['contract_id']} {historical['completion_id']}"
        )
        return 0
    if args.generate_completion:
        if args.verification_record is None:
            parser.error("--generate-completion requires --verification-record")
        value = build_completion(args.verification_record)
        COMPLETION_PATH.parent.mkdir(parents=True, exist_ok=True)
        COMPLETION_PATH.write_bytes(_canonical(value))
        print(f"wrote {COMPLETION_PATH.relative_to(REPO)}: {value['repair_id']}")
        return 0
    completion = verify_completion()
    print(f"W5 GREEN repaired verifier PASS: {completion['repair_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
