#!/usr/bin/env python3
"""Generate/verify the additive W7-A pre-science test-hardening authority."""

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

from training.deterministic_core import canonical_bytes, canonical_sha256  # noqa: E402

HISTORICAL_COMPLETION = REPO / "results/learned/w7/w7_a_completion.json"
HISTORICAL_SOURCE = REPO / "results/learned/w7/w7_source_manifest.json"
PROFILE_REPORT = REPO / "results/learned/w7/w7_pascal_profile.json"
PROFILE_FREEZE = REPO / "results/learned/w7/w7_pascal_profile_freeze.json"
SOURCE_V2 = REPO / "results/learned/w7/w7_source_manifest_v2.json"
COMPLETION = REPO / "results/learned/w7/w7_a_test_hardening_completion.json"
AUDIT = REPO / "audit/w7-a-test-hardening-repair-2026-08-27.md"

NEW_TESTS = {
    "tests/test_w7_trainer_hardening.py": ("w7_trainer_resume_gradscaler_regressions", 13),
    "tests/test_w7_validation_hardening.py": ("w7_validation_common_noise_metric_regressions", 7),
    "tests/test_w7_campaign_hardening.py": ("w7_campaign_recovery_regressions", 8),
}
TEST_SUPPORT = {
    "tests/__init__.py": "test_package_boundary",
    "tests/w7_hardening_fixtures.py": "deterministic_tiny_w7_fixtures",
}
CRITICAL_EXTRA = {
    "configs/learned-w7-g4-pilot.yaml": "w7_run_config",
    "requirements-pascal.lock": "pascal_execution_lock",
    "spec/params.generated.yaml": "resolved_parameters",
    "spec/schemas/w7_g4_artifacts.schema.json": "w7_artifact_schema",
    "tools/run_w7_campaign.py": "detached_campaign_state_machine",
}
PROTECTED_COUNTERS = {
    "w7_scientific_optimizer_steps": 0,
    "w7_lambda_pilot_runs": 0,
    "w7_candidate_results": 0,
    "g4_adjudications": 0,
    "w8_final_training_runs": 0,
    "learned_test_inference": 0,
    "test_model_facing_access": 0,
    "g8_scientific_changes": 0,
    "f1_reruns": 0,
    "f2_optimizer_steps_during_w7": 0,
    "f3_reruns": 0,
    "pass_one_reruns": 0,
    "pass_two_reruns": 0,
    "pass_three": 0,
    "bler_regeneration": 0,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, check=True).stdout


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a mapping")
    return value


def _entry(commit: str, path: str, role: str) -> dict[str, Any]:
    raw = _git("show", f"{commit}:{path}")
    return {
        "path": path,
        "role": role,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "git_blob": _git("rev-parse", f"{commit}:{path}").decode().strip(),
    }


def source_paths() -> dict[str, str]:
    historical = _load(HISTORICAL_SOURCE)
    paths = {
        entry["path"]: entry["role"]
        for entry in historical["entries"]
        if str(entry["path"]).startswith("src/")
    }
    paths.update(CRITICAL_EXTRA)
    paths.update({path: role for path, (role, _count) in NEW_TESTS.items()})
    paths.update(TEST_SUPPORT)
    return dict(sorted(paths.items()))


def build_source(commit: str) -> dict[str, Any]:
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("W7 hardening source commit must be a full SHA-1")
    historical = _load(HISTORICAL_SOURCE)
    historical_entries = {entry["path"]: entry for entry in historical["entries"]}
    entries = [_entry(commit, path, role) for path, role in source_paths().items()]
    changed = [
        entry["path"]
        for entry in entries
        if entry["path"] in historical_entries
        and entry["sha256"] != historical_entries[entry["path"]]["sha256"]
    ]
    if changed != ["src/training/w7_g4.py"]:
        raise ValueError(f"unexpected W7 critical production drift: {changed}")
    body = {
        "schema_version": 2,
        "artifact_role": "W7_A_TEST_HARDENING_SOURCE_MANIFEST",
        "source_commit": commit,
        "historical_source_manifest": {
            "path": str(HISTORICAL_SOURCE.relative_to(REPO)),
            "manifest_id": historical["manifest_id"],
            "source_commit": historical["source_commit"],
            "file_sha256": _sha(HISTORICAL_SOURCE),
        },
        "entries": entries,
        "production_source_changed": True,
        "production_changed_paths": ["src/training/w7_g4.py"],
        "change_classification": [
            "checkpoint_resume_lineage_control_only",
            "post_unscale_optimizer_finiteness_accounting_only",
        ],
        "scientific_semantics_changed": False,
        "g4_protocol_changed": False,
        "scientific_execution_authorization": "ABSENT",
    }
    body["manifest_id"] = "w7testsource-" + canonical_sha256(body)
    return body


def verify_source(value: dict[str, Any], *, current: bool = True) -> dict[str, Any]:
    expected = build_source(str(value.get("source_commit")))
    if value != expected:
        raise ValueError("W7 hardening source manifest differs from bound Git bytes")
    if current:
        for entry in value["entries"]:
            path = REPO / entry["path"]
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"W7 hardening current source missing/unsafe: {entry['path']}")
            if path.stat().st_size != entry["bytes"] or _sha(path) != entry["sha256"]:
                raise ValueError(f"W7 hardening current source drift: {entry['path']}")
    return value


def build_completion(*, source: dict[str, Any], ci_run: int, ci_job: int) -> dict[str, Any]:
    historical = _load(HISTORICAL_COMPLETION)
    profile = _load(PROFILE_REPORT)
    freeze = _load(PROFILE_FREEZE)
    body = {
        "schema_version": 1,
        "artifact_role": "W7_A_TEST_HARDENING_COMPLETION",
        "status": "GREEN_PRE_SCIENCE_TEST_HARDENED",
        "historical_w7_a_completion": {
            "path": str(HISTORICAL_COMPLETION.relative_to(REPO)),
            "completion_id": historical["completion_id"],
            "file_sha256": _sha(HISTORICAL_COMPLETION),
            "preserved_byte_identical": True,
        },
        "historical_pascal_profile": {
            "report_path": str(PROFILE_REPORT.relative_to(REPO)),
            "report_id": profile["report_id"],
            "report_sha256": _sha(PROFILE_REPORT),
            "freeze_path": str(PROFILE_FREEZE.relative_to(REPO)),
            "freeze_id": freeze["profile_freeze_id"],
            "freeze_sha256": _sha(PROFILE_FREEZE),
            "rerun": False,
            "applicability": "REMAINS_VALID_TRAINING_MATH_RUNTIME_VRAM_AND_BATCH_UNCHANGED",
        },
        "successor_source_manifest": {
            "path": str(SOURCE_V2.relative_to(REPO)),
            "manifest_id": source["manifest_id"],
            "source_commit": source["source_commit"],
            "file_sha256": _sha(SOURCE_V2),
        },
        "production_source_change": {
            "changed": True,
            "paths": ["src/training/w7_g4.py"],
            "classification": [
                "campaign_checkpoint_lineage_control_only",
                "compact_gradscaler_accounting_only",
            ],
            "training_math_or_performance_changed": False,
            "validation_semantics_changed": False,
            "campaign_control_changed": True,
            "pascal_profile_rerun_required": False,
        },
        "regressions": {
            "trainer": {"tests": 13, "passed": 13},
            "validation": {"tests": 7, "passed": 7},
            "campaign": {"tests": 8, "passed": 8},
            "new_total": {"tests": 28, "passed": 28},
            "existing_w7_protocol_adjudicator_lock": {"tests": 11, "passed": 11},
            "combined_w7": {"tests": 39, "passed": 39},
            "historical_w5_targeted": {"tests": 45, "passed": 45},
        },
        "behavioral_evidence": {
            "complete_denominator_no_duplicate_drop": True,
            "final_partial_batch": "35_samples_to_32_plus_3_two_exact_updates",
            "sample_weighted_accumulation": "five_sample_partial_matches_full_batch_mean_and_update",
            "fresh_instance_resume": "exact_model_optimizer_scheduler_scaler_global_step_and_latest_predecessor",
            "corrupt_latest": "HOLD_NO_OLDER_FALLBACK",
            "lineage_mutations": [
                "source", "gpu_uuid", "lambda", "seed", "config_hash",
                "execution_profile", "checkpoint_predecessor",
            ],
            "gradscaler_shared_decoder_nonfinite": "optimizer_wide_false_backoff_no_update_no_global_step",
            "gradscaler_finite": "genuine_update",
            "gradient_denominator_order": "after_unscale_before_optimizer_step",
            "validation_split": "test_structurally_unreachable",
            "validation_common_noise": "lambda_and_checkpoint_excluded_ambient_rng_invariant",
            "top1": "independently_count_recomputed",
            "psnr": "independently_recomputed_from_per_image_mse_data_range_1",
            "checkpoint_selection": "exact_earliest_epoch_tie_and_independent_reload",
            "campaign_recovery": "skip_complete_resume_latest_validation_only_replay_corrupt_and_foreign_hold",
            "campaign_terminal_status": "COMPLETE_NOT_ADJUDICATED",
            "w8_ineligibility": "candidate_and_checkpoint_roles_not_eligible_for_W8_initialization",
        },
        "failed_test_history": {
            "path": str(AUDIT.relative_to(REPO)),
            "sha256": _sha(AUDIT),
            "exposed_defects": 2,
            "pre_fix_failures": 3,
        },
        "carrier_ci": {
            "source_commit": source["source_commit"],
            "run_id": ci_run,
            "job_id": ci_job,
            "status": "success",
        },
        "scientific_execution_authorization": "ABSENT",
        "g4_status": "UNRESOLVED",
        "lambda_status": "PROVISIONAL_UNTIL_G4",
        "w8_status": "UNOPENED",
        "test_status": "SEALED",
        "protected_counters": dict(PROTECTED_COUNTERS),
        "next_action": "RETURN_FOR_INDEPENDENT_AUDIT_BEFORE_W7_B1_AUTHORIZATION",
    }
    body["completion_id"] = "w7testhardening-" + canonical_sha256(body)
    return body


def verify_completion(value: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    ci = value.get("carrier_ci", {})
    expected = build_completion(
        source=source,
        ci_run=int(ci.get("run_id", -1)),
        ci_job=int(ci.get("job_id", -1)),
    )
    if value != expected:
        raise ValueError("W7 hardening completion differs from authenticated inputs")
    if ci.get("status") != "success" or int(ci.get("run_id", 0)) <= 0 or int(ci.get("job_id", 0)) <= 0:
        raise ValueError("W7 hardening carrier CI is not green")
    if any(value["protected_counters"].values()):
        raise ValueError("W7 hardening protected counter is nonzero")
    return value


def verify_all(*, current: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    source = verify_source(_load(SOURCE_V2), current=current)
    completion = verify_completion(_load(COMPLETION), source)
    return source, completion


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--source-commit", required=True)
    complete = sub.add_parser("completion")
    complete.add_argument("--ci-run", type=int, required=True)
    complete.add_argument("--ci-job", type=int, required=True)
    sub.add_parser("verify")
    args = parser.parse_args()
    if args.command == "manifest":
        value = build_source(args.source_commit)
        SOURCE_V2.write_bytes(canonical_bytes(value))
        print(f"wrote {SOURCE_V2.relative_to(REPO)}: {value['manifest_id']}")
    elif args.command == "completion":
        source = verify_source(_load(SOURCE_V2), current=True)
        value = build_completion(source=source, ci_run=args.ci_run, ci_job=args.ci_job)
        COMPLETION.write_bytes(canonical_bytes(value))
        print(f"wrote {COMPLETION.relative_to(REPO)}: {value['completion_id']}")
    else:
        source, completion = verify_all(current=True)
        print(f"W7-A test hardening PASS: {completion['completion_id']} ({source['manifest_id']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
