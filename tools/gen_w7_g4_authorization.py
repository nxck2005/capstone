#!/usr/bin/env python3
"""Create or verify the additive, procedural W7-C G-4 authorization.

This artifact authorizes exactly one deterministic application of the already
frozen G-4 rule to the compact W7-B2R evidence.  It is deliberately not a
pre-result preregistration and contains no training or inference entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
AUTHORIZATION_PATH = REPO / "results/learned/w7/w7_g4_procedural_authorization.json"
AUTHORIZATION_PREFIX = "w7c4auth-"
SCHEMA_VERSION = 1

STARTING_MAIN_SHA = "94518919cc2f7603eb8dd35f41b8aef9a4c49e9d"
SCIENTIFIC_SOURCE_COMMIT = "cc704fcacec706719bc2791ae14a6c9d71dd4032"
ADJUDICATOR_PATH = "src/adjudication/w7_g4.py"
ADJUDICATOR_BLOB = "f1071971bce8dc6a48ddf504e5743e3faea5edfa"

B2R_EVIDENCE = (
    {
        "role": "B2R_COMPLETION",
        "path": "results/learned/w7/w7_b2_completion.json",
        "content_id": "w7b2rcompletion-172842c61df0231efd451d3d66b7857b5a67e79af887ff2d2bd8bcd9c801bee3",
        "file_sha256": "569b53bc41852d20936d11c3f0df3f5089ab783e2e6dcc534a5f8e69a7087395",
        "git_blob_sha1": "2ea930cd2206af38cf6a3541fdf8bf0f3884aaf5",
    },
    {
        "role": "B2R_CANDIDATE_INDEX",
        "path": "results/learned/w7/w7_b2_reconciliation_index.json",
        "content_id": "w7b2rindex-171ac64c8c56a00bff1e943aacfbe9fb92d1ecfc23686d9f6f55054c1ffbf3f0",
        "file_sha256": "d1019e85bb4da8a5ba3e21b016e663b09bcb2f911abfbc3814025dcb75fe834a",
        "git_blob_sha1": "81578bd39685945e2c3ea32eaba1699506398146",
    },
    {
        "role": "B2R_COMMON_NOISE_AUDIT",
        "path": "results/learned/w7/w7_b2_common_noise_audit.json",
        "content_id": "w7b2rnoise-f7f162de8664c8b03e15983e436f66918827cce8bcd1444ed2c53ff3d72662e5",
        "file_sha256": "fd7f59e8f91c5139ffd3cfae49f03c4fda492233f3aa53517ba1fd801cd8bf89",
        "git_blob_sha1": "7bd9399844bd656dedf00479217baf013555e870",
    },
    {
        "role": "B2R_CHECKPOINT_CUSTODY",
        "path": "results/learned/w7/w7_b2_checkpoint_custody.json",
        "content_id": "w7b2rcustody-1bbe4b083907b5debdb8104a3818ee9a5161d2bbccf963bc1ce1e2a0c53eb9ec",
        "file_sha256": "79725464fd666299bf8f71cf733db9d9948b7fa8b78fc16d0cdb4c4be2d8d634",
        "git_blob_sha1": "225de028ff39ac22a0d7f53b85717bac1d39e996",
    },
    {
        "role": "B2R_RECONCILIATION",
        "path": "results/learned/w7/w7_b2_reconciliation.json",
        "content_id": "w7b2rreconciliation-981bce14b3d851dd68a8304823fec86d5b6bcf9a948b1f1914ca7e4cd4cf168e",
        "file_sha256": "6d41ff5c6d852eb892132eb7772c03fc22c107d41c612218ab297f9065c031de",
        "git_blob_sha1": "81219aa29fcc6bceac17cb19ca93d2abeefc3e9f",
    },
)

FROZEN_PROTOCOL = {
    "dataset": "imagenette160",
    "validation_split": "validation",
    "calibration_ratio": "r_1_6",
    "calibration_ratio_parameter": "params.learned_system.lambda_calibration_ratio",
    "lambda_grid": [0.0, 0.1, 0.3, 1.0, 3.0],
    "lambda_order": "exact_configured_lambda_grid_order",
    "accuracy_tolerance_pp": 1.0,
    "primary_psnr_floor_db": 20.0,
    "relaxed_psnr_floor_db": 16.0,
    "calibration_snr_db": 7,
    "psnr_evaluation_snr_db": 15,
    "validation_denominator": 1000,
    "validation_noise_policy": "keyed_channel_noise_same_per_image_across_lambda",
    "selection": "smallest_numeric_lambda_in_the_applicable_qualifying_tier",
    "papr_role": "factual_evidence_only_not_a_selection_input",
}

HOMOGENEITY_AUTHORITY = {
    "source_commit": SCIENTIFIC_SOURCE_COMMIT,
    "source_manifest_id": "w7b1source-ef005dc427ea83a2ff38904362c6a85612ff133a253e880fc671f2654a4aeb3f",
    "source_manifest_sha256": "163d83e25e2dbeb2d0dfd77610634fc2c975a218a4d3a6f1cb189fe24352e392",
    "execution_profile_id": "confessor_pascal_cu126",
    "execution_image_family": "pascal-cu126-requirements-pascal-lock-v1",
    "profile_freeze_id": "w7profilefreeze-fab8a6960a6124de7276599c8b6e9971e93266fa23f58cc2a65e3498b41573b9",
    "profile_freeze_sha256": "d0eb628a910d93b350a6e9b542f845b5ec418211850899868f16278364ff2301",
    "gpu_name": "NVIDIA GeForce GTX 1080 Ti",
    "gpu_uuid": "GPU-00214b86-48e7-fcf0-bf46-575fa7f85b6b",
    "architecture": "djscc_residual_v1",
    "dataset": "imagenette160",
    "ratio": "r_1_6",
    "k": 12800,
    "train_seed": 0,
    "channel_seed": 0,
    "training_snr_db": 7,
    "calibration_snr_db": 7,
    "psnr_evaluation_snr_db": 15,
    "candidate_count": 5,
    "only_intended_candidate_field": "lambda",
    "unexpected_differences": [],
    "common_validation_noise": True,
}

PROTECTED_PRE_STATE = {
    "g4_adjudication_run": 0,
    "g4_adjudications": 0,
    "lambda_decision": "NOT_PERFORMED",
    "lambda_core": 1.0,
    "lambda_core_updated": False,
    "lambda_status": "provisional_until_G-4",
    "w8_final_training_runs": 0,
    "w8_state": "UNOPENED",
    "test_model_facing_access": 0,
    "learned_test_inference": 0,
    "test_state": "SEALED",
}

PROHIBITED = [
    "training",
    "model_inference",
    "validation_inference",
    "PSNR_inference",
    "PAPR_inference",
    "checkpoint_resume_or_checkpoint_open",
    "W8_initialization_or_final_training",
    "test_access_or_learned_test_inference",
    "modification_of_the_frozen_G4_adjudicator_or_selection_rule",
]


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def rendered_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("ascii")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _full_sha(value: object, width: int) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(rf"[0-9a-f]{{{width}}}", value))


def build_authorization(issued_at_utc: str) -> dict[str, Any]:
    if _git("rev-parse", "HEAD") != STARTING_MAIN_SHA:
        raise RuntimeError("procedural authorization must be created from the authenticated B2R main SHA")
    if _git("rev-parse", "origin/main") != STARTING_MAIN_SHA:
        raise RuntimeError("origin/main moved before procedural authorization")

    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "W7_C_G4_PROCEDURAL_AUTHORIZATION",
        "status": "AUTHORIZED",
        "authorized_by": "repository owner/operator through the explicit W7-C final G-4 adjudication instruction",
        "issued_at_utc": issued_at_utc,
        "pre_result_preregistration": False,
        "scientific_rule_binding": "G-4 selection rule was frozen in W7-A before scientific results existed; this artifact only authorizes its deterministic application to the already-authenticated five-candidate evidence",
        "terminal_b2r_main_sha": STARTING_MAIN_SHA,
        "scientific_source_commit": SCIENTIFIC_SOURCE_COMMIT,
        "frozen_adjudicator": {
            "path": ADJUDICATOR_PATH,
            "required_git_blob_sha1": ADJUDICATOR_BLOB,
            "terminal_main_git_blob_sha1": ADJUDICATOR_BLOB,
            "scientific_source_git_blob_sha1": ADJUDICATOR_BLOB,
        },
        "b2r_evidence": [dict(item) for item in B2R_EVIDENCE],
        "frozen_protocol": dict(FROZEN_PROTOCOL),
        "homogeneity_authority": dict(HOMOGENEITY_AUTHORITY),
        "protected_pre_state": dict(PROTECTED_PRE_STATE),
        "authorization_scope": {
            "scope": "EXACTLY_ONE_DETERMINISTIC_G4_ADJUDICATION",
            "input": "authenticated_compact_W7_B2R_evidence_only",
            "adjudication_call_count": 1,
            "writes_lambda_core": False,
            "opens_W8": False,
            "opens_test": False,
        },
        "prohibited_operations": list(PROHIBITED),
        "immutability": {
            "content_addressed": True,
            "refuse_second_authorization_or_second_adjudication": True,
            "evidence_bytes_are_read_only": True,
        },
    }
    body["authorization_id"] = AUTHORIZATION_PREFIX + canonical_sha256(body)
    return body


def verify_authorization(path: Path = AUTHORIZATION_PATH) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read authorization: {exc}") from exc
    expected_keys = {
        "schema_version", "artifact_role", "status", "authorized_by", "issued_at_utc",
        "pre_result_preregistration", "scientific_rule_binding", "terminal_b2r_main_sha",
        "scientific_source_commit", "frozen_adjudicator", "b2r_evidence", "frozen_protocol",
        "homogeneity_authority", "protected_pre_state", "authorization_scope",
        "prohibited_operations", "immutability", "authorization_id",
    }
    if set(value) != expected_keys:
        raise RuntimeError("W7-C authorization schema differs")
    identifier = value["authorization_id"]
    body = {key: item for key, item in value.items() if key != "authorization_id"}
    if identifier != AUTHORIZATION_PREFIX + canonical_sha256(body):
        raise RuntimeError("W7-C authorization ID does not authenticate its body")
    if value["schema_version"] != SCHEMA_VERSION or value["artifact_role"] != "W7_C_G4_PROCEDURAL_AUTHORIZATION" or value["status"] != "AUTHORIZED":
        raise RuntimeError("W7-C authorization role/status differs")
    if value["pre_result_preregistration"] is not False:
        raise RuntimeError("W7-C authorization incorrectly claims pre-result preregistration")
    if value["terminal_b2r_main_sha"] != STARTING_MAIN_SHA or value["scientific_source_commit"] != SCIENTIFIC_SOURCE_COMMIT:
        raise RuntimeError("W7-C authorization source SHA differs")
    adjudicator = value["frozen_adjudicator"]
    if adjudicator != {
        "path": ADJUDICATOR_PATH,
        "required_git_blob_sha1": ADJUDICATOR_BLOB,
        "terminal_main_git_blob_sha1": ADJUDICATOR_BLOB,
        "scientific_source_git_blob_sha1": ADJUDICATOR_BLOB,
    }:
        raise RuntimeError("W7-C authorization adjudicator binding differs")
    if value["b2r_evidence"] != [dict(item) for item in B2R_EVIDENCE]:
        raise RuntimeError("W7-C authorization B2R binding differs")
    if value["frozen_protocol"] != FROZEN_PROTOCOL:
        raise RuntimeError("W7-C authorization frozen protocol differs")
    if value["homogeneity_authority"] != HOMOGENEITY_AUTHORITY:
        raise RuntimeError("W7-C authorization homogeneity binding differs")
    if value["protected_pre_state"] != PROTECTED_PRE_STATE:
        raise RuntimeError("W7-C authorization protected pre-state differs")
    if value["authorization_scope"] != {
        "scope": "EXACTLY_ONE_DETERMINISTIC_G4_ADJUDICATION",
        "input": "authenticated_compact_W7_B2R_evidence_only",
        "adjudication_call_count": 1,
        "writes_lambda_core": False,
        "opens_W8": False,
        "opens_test": False,
    }:
        raise RuntimeError("W7-C authorization scope differs")
    if value["prohibited_operations"] != PROHIBITED or value["immutability"] != {
        "content_addressed": True,
        "refuse_second_authorization_or_second_adjudication": True,
        "evidence_bytes_are_read_only": True,
    }:
        raise RuntimeError("W7-C authorization prohibitions/immutability differ")
    if not isinstance(value["issued_at_utc"], str) or not re.fullmatch(r"20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", value["issued_at_utc"]):
        raise RuntimeError("W7-C authorization timestamp is invalid")
    for item in value["b2r_evidence"]:
        path_value = REPO / item["path"]
        if _sha256_file(path_value) != item["file_sha256"] or _git("hash-object", item["path"]) != item["git_blob_sha1"]:
            raise RuntimeError(f"W7-C authorization B2R bytes differ: {item['path']}")
    if _git("hash-object", ADJUDICATOR_PATH) != ADJUDICATOR_BLOB:
        raise RuntimeError("current frozen G-4 adjudicator blob differs")
    if _git("rev-parse", f"{SCIENTIFIC_SOURCE_COMMIT}:{ADJUDICATOR_PATH}") != ADJUDICATOR_BLOB:
        raise RuntimeError("scientific-source frozen G-4 adjudicator blob differs")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--issued-at", help="UTC timestamp used only when building the immutable artifact")
    parser.add_argument("--output", type=Path, default=AUTHORIZATION_PATH)
    args = parser.parse_args(argv)
    if args.command == "build":
        if not args.issued_at:
            parser.error("build requires --issued-at")
        if args.output.exists():
            raise SystemExit(f"refusing to replace existing authorization: {args.output}")
        value = build_authorization(args.issued_at)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(rendered_json(value))
        verify_authorization(args.output)
        print(json.dumps({"status": "FROZEN", "authorization_id": value["authorization_id"], "path": str(args.output.relative_to(REPO))}, sort_keys=True))
        return 0
    value = verify_authorization(args.output)
    print(json.dumps({"status": "PASS", "authorization_id": value["authorization_id"], "file_sha256": _sha256_file(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
