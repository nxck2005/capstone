#!/usr/bin/env python3
"""Verify the W8-A pre-execution freeze without opening scientific state."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from gen_w8_data_verification import verify_data_verification  # noqa: E402
from gen_w8_runtime_estimate import verify_runtime_estimate  # noqa: E402
from gen_w8_execution_authorization import (  # noqa: E402
    AUTHORIZATION_ROLE,
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    verify_authorization,
)
from gen_w8_source_manifest import CRITICAL_SOURCES, verify_manifest  # noqa: E402
from run_w8_campaign import (  # noqa: E402
    _expected_boundary,
    _expected_zero_counters,
    _run_w7_g4_verifier,
    _validate_authorization_contract,
    _verify_dataset,
    _verify_test_boundary,
    load_authority,
)
from training.deterministic_core import canonical_sha256  # noqa: E402
from config.run_config import config_hash as run_config_hash  # noqa: E402
from training.w8_protocol import (  # noqa: E402
    W8_EXPECTED_K,
    W8_EXPECTED_RATIOS,
    W8_SMOKE_ROLE,
    W8_SELECTED_GPU_NAME,
    W8_SELECTED_GPU_UUID,
    eligibility_for_role,
    fresh_initialization_identity,
    load_w8_config,
    protocol_config_hash,
    run_cells,
)


COMPLETION_PATH = REPO / "results/learned/w8/w8_a_completion.json"
SMOKE_PATH = REPO / "results/learned/w8/w8_a_smoke.json"
AUTH_PATH = REPO / "results/learned/w8/w8_execution_authorization.json"
MANIFEST_PATH = REPO / "results/learned/w8/w8_source_manifest.json"
DATA_PATH = REPO / "results/learned/w8/w8_data_verification.json"
RUNTIME_ESTIMATE_PATH = REPO / "results/learned/w8/w8_runtime_estimate.json"
COMPLETION_ROLE = "W8_A_PRE_EXECUTION_COMPLETION"


class W8AVerificationHold(RuntimeError):
    """An immutable W8-A authority or zero-coverage boundary is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise W8AVerificationHold(message)


def _sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise W8AVerificationHold(f"cannot hash {path}: {exc}") from None


def _read(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise W8AVerificationHold(f"{label} is unreadable: {exc}") from None
    _require(isinstance(value, dict), f"{label} is not a JSON object")
    return value


def _verify_current_sources(manifest: dict[str, Any]) -> None:
    for entry in manifest["entries"]:
        path = REPO / str(entry["path"])
        _require(path.is_file() and not path.is_symlink(), f"W8 source entry is missing: {entry['path']}")
        _require(path.stat().st_size == entry["bytes"] and _sha(path) == entry["sha256"], f"W8 carrier changed source entry: {entry['path']}")


def _verify_smoke(path: Path, source_commit: str) -> dict[str, Any]:
    value = _read(path, "W8 smoke record")
    required = {
        "schema_version", "artifact_role", "scientific_status", "status", "issued_at_utc",
        "source_commit", "protocol_checks", "configurations", "fresh_initialization_checks",
        "boundary", "focused_test_command", "focused_test_result",
        "papr_secondary_protocol_item", "eligibility", "smoke_id",
    }
    _require(set(value) == required, "W8 smoke schema differs")
    body = dict(value)
    identifier = body.pop("smoke_id")
    _require(identifier == "w8smoke-" + canonical_sha256(body), "W8 smoke ID differs")
    _require(value["schema_version"] == 1 and value["artifact_role"] == "W8_NON_SCIENTIFIC_SMOKE" and value["scientific_status"] == "NON_SCIENTIFIC" and value["status"] == "PASS_NON_SCIENTIFIC_ONLY", "W8 smoke role/status differs")
    _require(value["source_commit"] == source_commit, "W8 smoke source commit differs")
    _require(
        value["protocol_checks"] == {
            "six_configurations_constructed": True,
            "ratios_and_k_paths": dict(W8_EXPECTED_K),
            "lambda_exact": 3.0,
            "seed_zipper": "PASS",
            "run_order": [cell.to_dict() for cell in run_cells()],
        },
        "W8 smoke protocol result differs",
    )
    expected_eligibility = eligibility_for_role(W8_SMOKE_ROLE)
    _require(
        len(value["configurations"]) == len(run_cells())
        and [item.get("run_index") for item in value["configurations"]]
        == [cell.run_index for cell in run_cells()],
        "W8 smoke does not cover the six frozen cells",
    )
    configuration_keys = {
        "run_index", "ratio", "k", "train_seed", "channel_seed", "config_hash",
        "protocol_config_hash", "artifact_role", "eligibility",
    }
    for item, cell in zip(value["configurations"], run_cells()):
        _require(set(item) == configuration_keys, "W8 smoke configuration schema differs")
        config = load_w8_config(
            cell.ratio, cell.train_seed, cell.channel_seed, role=W8_SMOKE_ROLE
        )
        expected = {
            "run_index": cell.run_index,
            "ratio": cell.ratio,
            "k": cell.k,
            "train_seed": cell.train_seed,
            "channel_seed": cell.channel_seed,
            "config_hash": run_config_hash(config),
            "protocol_config_hash": protocol_config_hash(config),
            "artifact_role": W8_SMOKE_ROLE,
            "eligibility": expected_eligibility,
        }
        _require(item == expected, "W8 smoke cell/configuration differs")
    _require(
        value["fresh_initialization_checks"] == {
            "identity_fields": ["train_seed", "component_path"],
            "same_seed_same_identity": fresh_initialization_identity(0),
            "different_seed_identity": fresh_initialization_identity(1),
            "w7_checkpoint_initialization": "REJECTED_BY_W8_TRAINER_CONTRACT",
            "foreign_w8_checkpoint_initialization": "REJECTED_BY_W8_TRAINER_CONTRACT",
            "same_run_resume_only": "AUTHENTICATED_CHECKPOINT_LINEAGE_REQUIRED",
        },
        "W8 smoke initialization proof differs",
    )
    _require(
        value["boundary"] == {
            "optimizer_steps": 0,
            "scientific_checkpoints": 0,
            "w8_result_eligibility": "NOT_ELIGIBLE_FOR_W8_RESULT",
            "g10_eligibility": "NOT_ELIGIBLE_FOR_G10",
            "test_eligibility": "NOT_ELIGIBLE_FOR_TEST",
            "test_model_facing_access": 0,
            "learned_test_inference": 0,
            "er2_randomized_training": 0,
            "papr_constrained_training": 0,
            "er9_training": 0,
            "g10_adjudications": 0,
        },
        "W8 smoke boundary counters differ",
    )
    _require(value["eligibility"] == expected_eligibility, "W8 smoke eligibility differs")
    return value


def _verify_completion(
    path: Path,
    *,
    source_manifest: dict[str, Any],
    source_manifest_path: Path,
    authorization: dict[str, Any],
    authorization_path: Path,
    smoke: dict[str, Any],
    smoke_path: Path,
    data: dict[str, Any],
    runtime_estimate: dict[str, Any],
) -> dict[str, Any]:
    value = _read(path, "W8-A completion")
    required = {
        "schema_version", "artifact_role", "status", "issued_at_utc", "upstream",
        "scientific_source", "campaign", "profile", "smoke", "data_verification",
        "runtime_estimate", "pre_execution_counters", "boundary",
        "smoke_optimizer_steps_are_scientific_zero", "completion_id",
    }
    _require(set(value) == required, "W8-A completion schema differs")
    body = dict(value)
    identifier = body.pop("completion_id")
    _require(identifier == "w8acompletion-" + canonical_sha256(body), "W8-A completion ID differs")
    _require(value["schema_version"] == 1 and value["artifact_role"] == COMPLETION_ROLE and value["status"] == "GREEN_PRE_EXECUTION", "W8-A completion role/status differs")
    _require(value["upstream"] == {
        "w7_terminal_completion_id": "w7completion-fcd91d565ec3c98e1aff6c69a71b86af398971e7f8e898efa0499dc6e5c3dc1f",
        "g4_adjudication_id": "w7g4adjudication-2136277dbb5e4d3f8a467c6e4137e959e5b1ffc947777c70b487aa3e884e3ec0",
        "lambda_core": 3.0,
        "lambda_status": "selected_at_G-4",
    }, "W8-A upstream authority differs")
    source = value["scientific_source"]
    _require(source["source_commit"] == source_manifest["source_commit"] and source["source_manifest_id"] == source_manifest["manifest_id"] and source["source_manifest_sha256"] == _sha(source_manifest_path), "W8-A source manifest binding differs")
    _require(source["authorization_id"] == authorization["authorization_id"] and source["authorization_sha256"] == _sha(authorization_path), "W8-A authorization binding differs")
    _require(value["data_verification"] == data, "W8-A data-verification binding differs")
    _require(value["runtime_estimate"] == runtime_estimate, "W8-A runtime-estimate binding differs")
    _require(value["campaign"]["campaign_id"] == CAMPAIGN_ID and value["campaign"]["campaign_root"] == CAMPAIGN_ROOT and value["campaign"]["run_count"] == 6 and value["campaign"]["run_cells"] == [cell.to_dict() for cell in run_cells()] and value["campaign"]["unique_ratios"] == list(W8_EXPECTED_RATIOS) and value["campaign"]["seed_pairing"] == "zipped_not_cross_product", "W8-A campaign binding differs")
    profile = value["profile"]
    _require(profile == {
        "execution_profile_id": "confessor_pascal_cu126",
        "scientific_writer_host": "confessor",
        "gpu_name": W8_SELECTED_GPU_NAME,
        "gpu_uuid": W8_SELECTED_GPU_UUID,
        "requirements_lock_sha256": "d3561c8e930797d328cf45df1bdc5085842833665f9b5c9e5617d4c891e31a82",
        "physical_batch_size": 32,
        "accumulation_factor": 1,
        "effective_batch_size": 32,
        "validation_batch_size": 32,
    }, "W8-A profile binding differs")
    smoke_ref = value["smoke"]
    _require(smoke_ref == {
        "path": str(smoke_path.relative_to(REPO)) if smoke_path.is_relative_to(REPO) else str(smoke_path),
        "file_sha256": _sha(smoke_path),
        "smoke_id": smoke["smoke_id"],
        "scientific_status": "NON_SCIENTIFIC",
        "w8_result_eligibility": "NOT_ELIGIBLE_FOR_W8_RESULT",
        "g10_eligibility": "NOT_ELIGIBLE_FOR_G10",
        "test_eligibility": "NOT_ELIGIBLE_FOR_TEST",
    }, "W8-A smoke binding differs")
    _require(value["pre_execution_counters"] == {
        "w8_scientific_optimizer_steps": 0,
        "w8_final_training_runs": 0,
        "w8_completed_runs": 0,
        "w8_scientific_checkpoints": 0,
        "g10_adjudications": 0,
        "er2_randomized_training": 0,
        "papr_constrained_training": 0,
        "er9_training": 0,
        "test_model_facing_access": 0,
        "learned_test_inference": 0,
    }, "W8-A pre-execution counters differ")
    _require(value["boundary"] == {
        "scientific_execution": "REQUIRES_SEPARATE_W8_B_OWNER_AUTHORIZATION",
        "w8_b_launch_authorization_present": False,
        "source_contains_scientific_w8_results": False,
        "campaign_root_created_at_freeze": False,
        "test": "SEALED",
        "g10": "NOT_AUTHORIZED",
        "er2_randomized_training": "NOT_AUTHORIZED",
        "papr_constrained_training": "NOT_AUTHORIZED_BOUND_REQUIRES_DOWNSTREAM_PROTOCOL_ITEM",
        "er9": "NOT_AUTHORIZED",
    }, "W8-A boundary differs")
    _require(value["smoke_optimizer_steps_are_scientific_zero"] is True, "W8-A smoke boundary marker differs")
    return value


def verify(*, require_data: bool = True) -> dict[str, Any]:
    _require(
        all(
            path.is_file() and not path.is_symlink()
            for path in (
                MANIFEST_PATH, AUTH_PATH, SMOKE_PATH, DATA_PATH,
                RUNTIME_ESTIMATE_PATH, COMPLETION_PATH,
            )
        ),
        "W8-A immutable authority/completion files are incomplete",
    )
    try:
        authorization, manifest = load_authority(AUTH_PATH, MANIFEST_PATH)
    except Exception as exc:
        raise W8AVerificationHold(str(exc)) from None
    _verify_current_sources(manifest)
    _run_w7_g4_verifier(REPO)
    _verify_test_boundary(REPO)
    data = _verify_dataset(REPO, require_extracted=require_data)
    _require(
        data["status"] == "VERIFIED" or (
            not require_data
            and data["status"] == "ARCHIVE_NOT_PRESENT_PREPARATION_REQUIRED"
        ),
        "W8-A dataset provenance is not verified",
    )
    smoke = _verify_smoke(SMOKE_PATH, manifest["source_commit"])
    try:
        data_record = json.loads(DATA_PATH.read_bytes())
        _require(isinstance(data_record, dict), "W8 data-verification record is not an object")
        verify_data_verification(
            data_record, repo=REPO, require_local_data=require_data
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise W8AVerificationHold(f"W8 data-verification record is invalid: {exc}") from None
    try:
        runtime_record = json.loads(RUNTIME_ESTIMATE_PATH.read_bytes())
        _require(isinstance(runtime_record, dict), "W8 runtime-estimate record is not an object")
        verify_runtime_estimate(runtime_record)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise W8AVerificationHold(f"W8 runtime-estimate record is invalid: {exc}") from None
    completion = _verify_completion(
        COMPLETION_PATH,
        source_manifest=manifest,
        source_manifest_path=MANIFEST_PATH,
        authorization=authorization,
        authorization_path=AUTH_PATH,
        smoke=smoke,
        smoke_path=SMOKE_PATH,
        data=data_record,
        runtime_estimate=runtime_record,
    )
    root = Path(authorization["campaign"]["campaign_root"])
    _require(not root.exists() or (root.is_dir() and not any(root.iterdir())), "W8-A campaign root contains scientific state")
    return {
        "status": "PASS",
        "verdict": "W8-A GREEN_PRE_EXECUTION",
        "completion_id": completion["completion_id"],
        "authorization_id": authorization["authorization_id"],
        "source_manifest_id": manifest["manifest_id"],
        "source_commit": manifest["source_commit"],
        "w8_scientific_optimizer_steps": 0,
        "w8_final_training_runs": 0,
        "g10_adjudications": 0,
        "test_model_facing_access": 0,
        "learned_test_inference": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-data", action="store_true", help="only for source-only CI where the ignored archive is unavailable")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(verify(require_data=not args.skip_data), sort_keys=True))
    except (W8AVerificationHold, ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "HOLD", "reason": str(exc)}, sort_keys=True))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
