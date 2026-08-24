"""G8_F/F0 authorization construction and fail-closed authentication.

F0 opens only the deterministic F1 contract.  It does not launch F1, decode an
image, invoke JPEG 2000, run a classifier, train, select, or access test.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from baseline.g8_f_materializer import (
    CODEC_CONFIGURATION_HASH,
    CODEC_CONFIGURATION_ID,
    MANIFEST_SHA256,
    ORDERED_PAIR_SHA256,
    PAIR_SET_SHA256,
    SAMPLER_PLAN_ID,
    SAMPLER_PLAN_SHA256,
    load_frozen_assignments,
)
from baseline.g8_f_sampler_plan import (
    AM87_PLAN_FILE_SHA256,
    AM87_PLAN_ID,
    EXPECTED_ATTEMPTS,
    EXPECTED_QUALITY_COUNT,
    EXPECTED_TRAINING_COUNT,
    EXPECTED_VARIANTS,
    canonical_json,
)
from config.execution_profiles import authenticate_execution_profile, profile_definition
from config.params import REPO_ROOT, get

SCHEMA_VERSION = 1
ARTIFACT_ROLE = "g8_f_f0_execution_authorization"
AUTHORIZATION_PREFIX = "g8ff0auth-"
AUTHORIZATION_PATH = REPO_ROOT / "results/baseline/g8_f/f0_execution_authorization.json"
RUNTIME_ROOT = REPO_ROOT / "results/baseline/g8_f/runtime"
PROFILE_ID = "local_4060_cu130"
DEVICE = "cuda:0"
SPEC_PATH = REPO_ROOT / "spec/SPEC.md"
PARAMS_PATH = REPO_ROOT / "spec/params.generated.yaml"
MANIFEST_PATH = REPO_ROOT / "data/manifests/imagenette160.csv"
AM87_PATH = REPO_ROOT / "results/baseline/g8_f/corpus_plan.json"
AM88_PATH = REPO_ROOT / "results/baseline/g8_f/am88_sampler_plan.json"
G8D_CONTRACT_PATH = REPO_ROOT / "results/baseline/g8_d/measurement_contract.json"
G1_PATH = REPO_ROOT / "results/reference_classifier/g1_adjudication.json"
E7_PATH = REPO_ROOT / "results/baseline/g8_e/e2_confessor_successor/e7_handoff.json"
PASS_ONE_PATH = REPO_ROOT / "results/baseline/g8_e/pass_one_state.json"

# Exact transitive scientific boundaries used by the later F1 entry point.
# Documentation, old campaign implementations, and training code are excluded.
F1_SOURCE_PATHS = (
    "src/baseline/g8_f_f0.py",
    "src/baseline/g8_f_materializer.py",
    "src/baseline/g8_f_sampler_plan.py",
    "src/baseline/g8_f_corpus_plan.py",
    "src/baseline/j2k.py",
    "src/config/params.py",
    "src/config/execution_profiles.py",
    "src/data/adapters.py",
    "src/data/identity.py",
    "src/data/manifests.py",
    "src/data/preprocessing.py",
    "src/data/provenance.py",
    "src/data/registry.py",
    "src/env.py",
    "tools/run_g8_f_f1.py",
    "tools/verify_g8_f_f0.py",
)

GIT_HEX_LENGTH = 40  # literal-ok: full SHA-1 Git object identity width
GIT_TIMEOUT_SECONDS = 30  # literal-ok: bounded local Git metadata query

ZERO_COUNTERS = {
    "materialized_artifact_objects": 0,
    "real_f1_jpeg2000_invocations": 0,
    "image_payloads_decoded": 0,
    "artifact_classifier_inference": 0,
    "artifact_classifier_optimizer_steps": 0,
    "pass_two": 0,
    "fallback_invoked": 0,
    "ratio_adjudicated": 0,
    "test_access": 0,
    "learned_system_training": 0,
    "prior_science_reruns": 0,
}


class G8FF0Error(RuntimeError):
    """A prerequisite or frozen F0 identity differs."""


def rendered_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G8FF0Error(message)


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise G8FF0Error(f"cannot read {path}: {exc}") from None
    _require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def _binding(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"path": str(path.relative_to(REPO_ROOT)), "bytes": len(raw), "sha256": sha256_bytes(raw)}


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise G8FF0Error(f"git {' '.join(args)} failed: {exc}") from None


def _source_closure(source_commit: str) -> list[dict[str, Any]]:
    _require(len(source_commit) == GIT_HEX_LENGTH, "F1 source commit is not full length")
    _require(_git("cat-file", "-t", source_commit) == "commit", "F1 source commit does not resolve")
    entries: list[dict[str, Any]] = []
    for relative in F1_SOURCE_PATHS:
        path = REPO_ROOT / relative
        raw = path.read_bytes()
        try:
            committed = subprocess.run(
                ["git", "show", f"{source_commit}:{relative}"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                timeout=GIT_TIMEOUT_SECONDS,
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            raise G8FF0Error(f"cannot read {relative} at F1 source commit: {exc}") from None
        _require(raw == committed, f"current {relative} differs from intended F1 source commit")
        entries.append({"path": relative, "bytes": len(raw), "sha256": sha256_bytes(raw)})
    return entries


def _protocol_hash(source_commit: str) -> str:
    value = {
        "source_commit": source_commit,
        "am87_plan_id": AM87_PLAN_ID,
        "am87_plan_sha256": AM87_PLAN_FILE_SHA256,
        "am88_plan_id": SAMPLER_PLAN_ID,
        "am88_plan_sha256": SAMPLER_PLAN_SHA256,
        "ordered_pair_sha256": ORDERED_PAIR_SHA256,
        "pair_set_sha256": PAIR_SET_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "codec_configuration_id": CODEC_CONFIGURATION_ID,
        "codec_configuration_hash": CODEC_CONFIGURATION_HASH,
        "execution_profile_id": PROFILE_ID,
    }
    return sha256_bytes(canonical_json(value))


def build_f0_authorization(*, source_commit: str, authorization_date: str) -> dict[str, Any]:
    """Construct F0 only after authenticating the live destination and zero state."""

    _require(authorization_date == "2026-08-24", "F0 authorization date differs from owner action")
    _require(_git("rev-parse", "HEAD") == source_commit, "generation HEAD is not the intended F1 source commit")
    _require(not _git("status", "--porcelain", "--untracked-files=all"), "F0 authorization requires a clean source checkout")
    _require(not RUNTIME_ROOT.exists(), "G8_F runtime already exists; F1 may have started")

    am87_raw = AM87_PATH.read_bytes()
    am88_raw = AM88_PATH.read_bytes()
    _require(sha256_bytes(am87_raw) == AM87_PLAN_FILE_SHA256 and _read(AM87_PATH).get("plan_id") == AM87_PLAN_ID, "AM-87 support binding differs")
    _require(sha256_bytes(am88_raw) == SAMPLER_PLAN_SHA256 and _read(AM88_PATH).get("plan_id") == SAMPLER_PLAN_ID, "AM-88 sampler binding differs")
    assignments = load_frozen_assignments()
    _require(len(assignments) == EXPECTED_ATTEMPTS, "AM-88 nominal assignment count differs")
    _require(sha256_bytes(MANIFEST_PATH.read_bytes()) == MANIFEST_SHA256, "training manifest differs")

    e7 = _read(E7_PATH)
    pass_one = _read(PASS_ONE_PATH)
    _require(e7.get("g8_f", {}).get("authorized") is False and e7.get("g8_f", {}).get("execution_count") == 0, "pre-F0 E7 boundary is not unopened")
    _require(e7.get("counters", {}).get("pass_one_executed_count") == 1, "pass one was not executed exactly once")
    for name in ("training", "pass_two", "pass_three", "fallback_invoked", "ratio_adjudicated", "test_access", "learned_system_training", "g8_f_execution"):
        _require(e7["counters"].get(name) == 0, f"protected E7 counter is nonzero: {name}")

    protocol_hash = _protocol_hash(source_commit)
    live_runtime = authenticate_execution_profile(
        PROFILE_ID,
        device=DEVICE,
        config_hash=protocol_hash,
        require_openjpeg=True,
    )
    _require(live_runtime["git_commit"] == source_commit and live_runtime["git_dirty"] is False, "live profile source state differs")
    usage = shutil.disk_usage(REPO_ROOT)
    reserve = int(_read(AM88_PATH)["compute_consequence"]["maximum_with_25_percent_safety_bytes"])
    _require(usage.free >= reserve, "execution destination lacks AM-88 storage reserve")
    profile = profile_definition(PROFILE_ID)
    dataset = get("datasets.imagenette160")
    g8d = _read(G8D_CONTRACT_PATH)
    codec = g8d["codec_binding"]
    _require(codec["configuration_hash"] == CODEC_CONFIGURATION_HASH, "G8_D codec configuration hash differs")
    _require("g8dcodec-" + sha256_bytes(canonical_json(codec)) == CODEC_CONFIGURATION_ID, "G8_D codec identity differs")

    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": ARTIFACT_ROLE,
        "phase": "G8_F",
        "checkpoint": "F0",
        "status": "F0_GREEN_EXECUTION_CONTRACT_AUTHORIZED_AND_FROZEN",
        "authorization_date": authorization_date,
        "owner_authorization": {
            "scope": "G8_F_F0_ONLY",
            "basis": "owner_explicit_API_prompt_after_independent_AM88_audit",
            "permitted": [
                "authenticate_and_freeze_production_identities_and_prerequisites",
                "establish_the_F1_execution_contract_and_handoff",
                "mark_G8_F_ready_for_a_separate_F1_launch",
            ],
            "not_permitted": [
                "materialize_any_production_training_pair",
                "invoke_real_F1_JPEG2000",
                "train_or_fine_tune_or_optimizer_step",
                "artifact_classifier_inference_or_validation_scoring",
                "pass_two_or_fallback_or_ratio_adjudication",
                "learned_system_training",
                "test_access",
                "launch_F1_or_later_scientific_stages",
            ],
            "f1_launch_authorized": False,
            "separate_owner_operator_action_required": True,
        },
        "source": {
            "intended_f1_source_commit": source_commit,
            "protocol_config_sha256": protocol_hash,
            "closure_rule": "exact_list_of_F1_output_or_admission_affecting_sources_at_intended_commit",
            "closure": _source_closure(source_commit),
        },
        "protocol": {
            "specification": _binding(SPEC_PATH),
            "generated_parameters": _binding(PARAMS_PATH),
            "am87_support_plan": {**_binding(AM87_PATH), "plan_id": AM87_PLAN_ID},
            "am88_sampler_plan": {**_binding(AM88_PATH), "plan_id": SAMPLER_PLAN_ID},
            "sampler": {
                "version": "g8_f_balanced_sampler_v1",
                "seed": "am88-g8f-balanced-sampler-20260824-v1",
                "algorithm": "sha256_keyed_stable_id_order_global_quality_permutation_class_chunks_cyclic_v1",
                "variants_per_training_image": EXPECTED_VARIANTS,
            },
            "quality_count": EXPECTED_QUALITY_COUNT,
            "training_stable_id_count": EXPECTED_TRAINING_COUNT,
            "nominal_attempt_count": EXPECTED_ATTEMPTS,
            "ordered_pair_sha256": ORDERED_PAIR_SHA256,
            "pair_set_sha256": PAIR_SET_SHA256,
            "typed_codec_infeasibility": "record_omitted_assigned_pair_no_replacement_no_resampling",
            "unexpected_failure": "HOLD",
            "dynamic_assignment_or_cartesian_fallback": False,
        },
        "data": {
            "dataset": "imagenette160",
            "published_archive_filename": dataset["archive_filename"],
            "published_archive_bytes": dataset["archive_bytes"],
            "published_archive_sha256": dataset["archive_sha256"],
            "training_manifest": _binding(MANIFEST_PATH),
            "training_stable_id_set_sha256": "20df375a9915d26e950ef817dfa1b6ef847f304c4dad4db100f41cfc8511cfaa",
            "stable_id_class_mapping_sha256": "3aa96b86be7685370bf26221a21edf29ca6f378ab90c65a789b95c94a327dd54",
            "split": "train",
            "validation_ids": 0,
            "test_ids": 0,
            "test_sealed": True,
        },
        "codec": {
            "configuration_contract": _binding(G8D_CONTRACT_PATH),
            "codec_configuration_id": CODEC_CONFIGURATION_ID,
            "configuration_hash": CODEC_CONFIGURATION_HASH,
            "snapshot": codec["snapshot"],
            "openjpeg_version": codec["runtime_version"],
            "glymur_version": get("environment.glymur"),
            "canonicalization": {
                "source": _binding(REPO_ROOT / "src/data/preprocessing.py"),
                "canonical_image": get("preprocessing.canonical_image"),
                "codec_input": get("preprocessing.codec_input"),
                "downsample_interpolation": get("preprocessing.codec_downsample_interpolation"),
                "upsample_interpolation": get("preprocessing.codec_upsample_interpolation"),
            },
        },
        "execution": {
            "execution_profile_id": PROFILE_ID,
            "device": DEVICE,
            "lock_file": profile["lock_file"],
            "lock_file_sha256": profile["lock_file_sha256"],
            "live_runtime_authentication": live_runtime,
            "sole_writer_host": profile["scientific_writer_host"],
            "runtime_root": str(RUNTIME_ROOT.relative_to(REPO_ROOT)),
            "runtime_root_existed_at_f0": False,
        },
        "storage_preflight": {
            "destination": str(REPO_ROOT),
            "filesystem_total_bytes": usage.total,
            "filesystem_used_bytes": usage.used,
            "filesystem_available_bytes": usage.free,
            "validation_incidence_estimated_bytes": _read(AM88_PATH)["compute_consequence"]["validation_incidence_estimated_bytes"],
            "all_assignment_estimated_maximum_bytes": _read(AM88_PATH)["compute_consequence"]["maximum_estimated_bytes"],
            "required_with_25_percent_reserve_bytes": reserve,
            "sufficient": True,
            "planning_only_not_scientific_measurement": True,
        },
        "prerequisites": {
            "g8_c": "GREEN_CLOSED_REAUTHENTICATED_NO_RECOMPUTATION",
            "g8_d": "GREEN_CLOSED_REAUTHENTICATED_NO_RECOMPUTATION",
            "g8_e": "GREEN_THROUGH_E7_REAUTHENTICATED_NO_RECOMPUTATION",
            "g1_clean_classifier": {**_binding(G1_PATH), "status": "FROZEN_UNCHANGED_REAUTHENTICATED_NO_RETRAINING"},
            "pass_one": {**_binding(PASS_ONE_PATH), "state_id": pass_one["state_id"], "executed_count": 1, "rerun": False},
            "e7_handoff": {**_binding(E7_PATH), "handoff_id": e7["handoff_id"]},
            "am87_verification": "PASS_EXACT_120_SUPPORT_8469_TRAIN_IDS",
            "am88_verification": "PASS_EXACT_6_PER_IMAGE_50814_PAIRS_BALANCED",
        },
        "protected_starting_state": {
            "f0_previously_authorized": False,
            "f0_authorized_by_this_artifact": True,
            "f1_started": False,
            "g8_f_execution_count": 0,
            **ZERO_COUNTERS,
            "production_worker_running": False,
            "confessor_started": False,
        },
        "smoke": {
            "permitted_kind": "synthetic_non_scientific_only",
            "production_training_image_used": False,
            "real_f1_jpeg2000_invocations": 0,
            "scientific_coverage": 0,
        },
        "next_action": "OWNER/OPERATOR LAUNCH OF F1 USING THIS EXACT F0 AUTHORIZATION",
        "later_launch_command": ".venv/bin/python tools/run_g8_f_f1.py --start --f0-authorization results/baseline/g8_f/f0_execution_authorization.json --f1-launch-authorization <OWNER_ISSUED_F1_LAUNCH_AUTHORIZATION.json> --runtime-root results/baseline/g8_f/runtime",
        "terminal_statement": "F0 GREEN - G8_F EXECUTION CONTRACT/AUTHORIZATION FROZEN; F1 NOT STARTED; SEPARATE OWNER/OPERATOR LAUNCH REQUIRED.",
    }
    body["authorization_id"] = AUTHORIZATION_PREFIX + sha256_bytes(canonical_json(body))
    return body


def verify_f0_authorization(
    path: Path = AUTHORIZATION_PATH,
    *,
    live_runtime: bool = False,
    require_zero_prefix: bool = True,
) -> dict[str, Any]:
    """Authenticate frozen bytes; optionally re-authenticate live device/storage."""

    raw = path.read_bytes()
    value = json.loads(raw)
    _require(raw == rendered_json(value), "F0 authorization is not canonical rendered JSON")
    body = dict(value)
    authorization_id = body.pop("authorization_id", None)
    _require(authorization_id == AUTHORIZATION_PREFIX + sha256_bytes(canonical_json(body)), "F0 authorization ID differs")
    _require(value.get("schema_version") == SCHEMA_VERSION and value.get("artifact_role") == ARTIFACT_ROLE, "F0 authorization header differs")
    _require(value.get("status") == "F0_GREEN_EXECUTION_CONTRACT_AUTHORIZED_AND_FROZEN", "F0 is not green/frozen")
    _require(value["owner_authorization"]["scope"] == "G8_F_F0_ONLY", "owner authorization scope differs")
    _require(value["owner_authorization"]["f1_launch_authorized"] is False, "F0 improperly authorizes F1 launch")
    _require(value["next_action"] == "OWNER/OPERATOR LAUNCH OF F1 USING THIS EXACT F0 AUTHORIZATION", "F0 next action differs")

    protocol = value["protocol"]
    _require(protocol["am87_support_plan"]["plan_id"] == AM87_PLAN_ID and protocol["am87_support_plan"]["sha256"] == AM87_PLAN_FILE_SHA256, "F0 AM-87 binding differs")
    _require(protocol["am88_sampler_plan"]["plan_id"] == SAMPLER_PLAN_ID and protocol["am88_sampler_plan"]["sha256"] == SAMPLER_PLAN_SHA256, "F0 AM-88 binding differs")
    _require(
        (
            protocol["quality_count"],
            protocol["training_stable_id_count"],
            protocol["sampler"]["variants_per_training_image"],
            protocol["nominal_attempt_count"],
        )
        == (EXPECTED_QUALITY_COUNT, EXPECTED_TRAINING_COUNT, EXPECTED_VARIANTS, EXPECTED_ATTEMPTS),
        "F0 support/multiplicity differs",
    )
    _require(protocol["ordered_pair_sha256"] == ORDERED_PAIR_SHA256 and protocol["pair_set_sha256"] == PAIR_SET_SHA256, "F0 pair digest differs")
    _require(protocol["dynamic_assignment_or_cartesian_fallback"] is False, "F0 permits dynamic/Cartesian assignments")
    _require(protocol["typed_codec_infeasibility"] == "record_omitted_assigned_pair_no_replacement_no_resampling" and protocol["unexpected_failure"] == "HOLD", "F0 outcome semantics differ")
    _require(value["data"]["training_manifest"]["sha256"] == MANIFEST_SHA256 and value["data"]["validation_ids"] == value["data"]["test_ids"] == 0, "F0 data membership differs")
    _require(value["codec"]["codec_configuration_id"] == CODEC_CONFIGURATION_ID and value["codec"]["configuration_hash"] == CODEC_CONFIGURATION_HASH, "F0 codec identity differs")
    _require(value["execution"]["execution_profile_id"] == PROFILE_ID and value["execution"]["lock_file_sha256"] == profile_definition(PROFILE_ID)["lock_file_sha256"], "F0 execution profile/lock differs")
    _require(value["protected_starting_state"] == {"f0_previously_authorized": False, "f0_authorized_by_this_artifact": True, "f1_started": False, "g8_f_execution_count": 0, **ZERO_COUNTERS, "production_worker_running": False, "confessor_started": False}, "F0 protected starting state differs")

    for section_name in ("protocol", "data", "codec", "prerequisites"):
        section = value[section_name]
        for binding in section.values() if isinstance(section, Mapping) else ():
            if isinstance(binding, Mapping) and {"path", "bytes", "sha256"} <= set(binding):
                bound_path = REPO_ROOT / binding["path"]
                current = bound_path.read_bytes()
                _require(len(current) == binding["bytes"] and sha256_bytes(current) == binding["sha256"], f"F0 bound bytes differ: {binding['path']}")
    canonical_binding = value["codec"]["canonicalization"]["source"]
    current = (REPO_ROOT / canonical_binding["path"]).read_bytes()
    _require(len(current) == canonical_binding["bytes"] and sha256_bytes(current) == canonical_binding["sha256"], "F0 canonicalization source differs")

    source_commit = value["source"]["intended_f1_source_commit"]
    closure = value["source"]["closure"]
    _require([entry["path"] for entry in closure] == list(F1_SOURCE_PATHS), "F1 source closure path set/order differs")
    for entry in closure:
        current = (REPO_ROOT / entry["path"]).read_bytes()
        _require(len(current) == entry["bytes"] and sha256_bytes(current) == entry["sha256"], f"current F1 source differs: {entry['path']}")
        committed = subprocess.run(["git", "show", f"{source_commit}:{entry['path']}"], cwd=REPO_ROOT, check=True, capture_output=True).stdout
        _require(current == committed, f"F1 source commit bytes differ: {entry['path']}")
    _require(value["source"]["protocol_config_sha256"] == _protocol_hash(source_commit), "F0 protocol config identity differs")
    lock_path = REPO_ROOT / value["execution"]["lock_file"]
    _require(sha256_bytes(lock_path.read_bytes()) == value["execution"]["lock_file_sha256"], "F0 lock bytes differ")
    reserve = value["storage_preflight"]["required_with_25_percent_reserve_bytes"]
    _require(value["storage_preflight"]["sufficient"] is True and value["storage_preflight"]["filesystem_available_bytes"] >= reserve, "recorded F0 storage preflight is insufficient")
    if require_zero_prefix:
        _require(not (REPO_ROOT / value["execution"]["runtime_root"]).exists(), "F1 runtime exists; F0 opening is no longer zero-prefix")

    if live_runtime:
        actual = authenticate_execution_profile(PROFILE_ID, device=DEVICE, config_hash=value["source"]["protocol_config_sha256"], require_openjpeg=True)
        recorded = value["execution"]["live_runtime_authentication"]
        for key in ("execution_profile_id", "lock_file_sha256", "python_version", "torch_version", "torch_cuda_build", "torchvision_version", "numpy_version", "sionna_version", "openjpeg_version", "gpu_name", "gpu_uuid", "gpu_compute_capability"):
            _require(actual[key] == recorded[key], f"live F0 runtime differs: {key}")
        _require(shutil.disk_usage(REPO_ROOT).free >= reserve, "live F1 destination no longer has required storage reserve")
    return value
