"""Canonical pre-optimizer authorization for G8_F/F2 BR-12 only."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from config.params import REPO_ROOT, get
from training.g8_f_f2 import (
    EXPECTED_ASSIGNMENTS,
    EXPECTED_BATCH_SIZE,
    EXPECTED_EPOCHS,
    EXPECTED_MATERIALIZED,
    EXPECTED_OMISSIONS,
    EXPECTED_OPTIMIZER_STEPS,
    EXPECTED_STEPS_PER_EPOCH,
    F1_COMPLETION_ID,
    F1_COMPLETION_SHA256,
    F1_CORPUS_ID,
    F1_MANIFEST_SHA256,
    F2_SCOPE,
    F2_VARIANT,
    G1_ADJUDICATION_ID,
    G1_ADJUDICATION_SHA256,
    G1_CHECKPOINT_BYTES,
    G1_CHECKPOINT_ID,
    G1_CHECKPOINT_SHA256,
    canonical_json,
    f2_recipe,
    f2_recipe_sha256,
    sha256_bytes,
)

AUTHORIZATION_PATH = REPO_ROOT / "results/baseline/g8_f/f2_execution_authorization.json"
AUTHORIZATION_PREFIX = "g8ff2auth-"
PROFILE_ID = "confessor_pascal_cu126"
EXPECTED_DEVICE = "cuda:0"
EXPECTED_HOST = "confessor"
EXPECTED_GPU_NAME = "NVIDIA TITAN Xp"
EXPECTED_GPU_UUID = "GPU-46acd0f2-2ff5-1a43-cac9-2ae20e56dc9a"
SOURCE_PATHS = (
    "src/training/g8_f_f2.py",
    "src/training/g8_f_f2_authorization.py",
    "tools/run_g8_f_f2.py",
    "src/data/preprocessing.py",
    "src/data/classifier.py",
    "src/models/reference_classifier.py",
    "src/models/frozen_reference_classifier.py",
    "src/training/reference_classifier.py",
    "src/baseline/g8_f_closeout.py",
    "src/baseline/g8_f_f0.py",
    "src/baseline/g8_f_materializer.py",
    "src/baseline/g8_f_sampler_plan.py",
    "src/baseline/g8_f_corpus_plan.py",
    "src/baseline/g8_campaign.py",
    "src/baseline/g8_pascal_production.py",
    "src/baseline/g8_d.py",
    "src/baseline/g8_e.py",
    "src/baseline/g8_e_corrected_v3s.py",
    "src/baseline/g8_f_sampler_plan.py",
    "tools/verify_g8_f_sampler_plan.py",
    "tools/verify_w4_baseline_integration.py",
    "src/baseline/j2k.py",
    "src/config/params.py",
    "src/config/execution_profiles.py",
    "src/data/adapters.py",
    "src/data/identity.py",
    "src/data/manifests.py",
    "src/data/provenance.py",
    "src/data/registry.py",
    "src/env.py",
    "tools/run_g8_f_f1.py",
    "spec/params.generated.yaml",
    "results/baseline/g8_f/am89_f2_source_compatibility.json",
    "requirements-pascal.lock",
)


class F2AuthorizationHold(RuntimeError):
    """The frozen F2 authorization or source closure differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise F2AuthorizationHold(message)


def rendered_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


def _git(*args: str, cwd: Path = REPO_ROOT) -> bytes:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=False)
    _require(result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def source_closure(source_commit: str, *, repo_root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    _require(len(source_commit) == 40, "F2 source commit must be full length")  # literal-ok: Git SHA-1 width
    entries: list[dict[str, Any]] = []
    for relative in SOURCE_PATHS:
        raw = _git("show", f"{source_commit}:{relative}", cwd=repo_root)
        blob = _git("rev-parse", f"{source_commit}:{relative}", cwd=repo_root).decode("ascii").strip()
        entries.append({"path": relative, "bytes": len(raw), "sha256": sha256_bytes(raw), "git_blob": blob})
    return entries


def build_authorization(
    *,
    source_commit: str,
    issued_at: str,
    preflight: Mapping[str, Any],
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    profile = dict(get(f"environment.execution_profiles.{PROFILE_ID}"))
    body: dict[str, Any] = {
        "schema_version": 1,
        "artifact_role": "g8_f_f2_br12_execution_authorization",
        "status": "FROZEN_OWNER_AUTHORIZED_BEFORE_OPTIMIZER_STEP_1",
        "scope": F2_SCOPE,
        "authorized_operations": ["artifact_classifier_optimizer_steps", "f2_checkpoint_selection_validation_inference", "epoch_checkpoint_writes", "best_checkpoint_tracking"],
        "prohibited_operations": ["f1_regeneration", "jpeg2000_encoding", "f3_cached_sweep_rescoring", "pass_two", "pass_three", "ratio_adjudication", "fallback", "learned_training", "test_access"],
        "owner_authorization": {"basis": "owner prompt explicitly authorizing G8_F/F2/BR-12 artifact-classifier fine-tuning only", "issued_at": issued_at, "single_scientific_run": True},
        "source_commit": source_commit,
        "source_closure": source_closure(source_commit, repo_root=repo_root),
        "f1": {
            "completion_path": "results/baseline/g8_f/f1_completion.json",
            "completion_id": F1_COMPLETION_ID,
            "completion_file_sha256": F1_COMPLETION_SHA256,
            "corpus_manifest_path": "results/baseline/g8_f/f1_corpus_manifest.csv",
            "corpus_id": F1_CORPUS_ID,
            "corpus_manifest_sha256": F1_MANIFEST_SHA256,
            "assignment_rows": EXPECTED_ASSIGNMENTS,
            "materialized_training_rows": EXPECTED_MATERIALIZED,
            "typed_omissions": EXPECTED_OMISSIONS,
            "unexpected_outcomes": 0,
            "multiplicity": "one_logical_training_item_per_materialized_assignment_row_no_reconstruction_deduplication",
            "validation_ids": 0,
            "test_ids": 0,
        },
        "g1_parent": {
            "adjudication_path": "results/reference_classifier/g1_adjudication.json",
            "adjudication_id": G1_ADJUDICATION_ID,
            "adjudication_file_sha256": G1_ADJUDICATION_SHA256,
            "classifier_variant": "clean",
            "architecture": "resnet18",
            "train_seed": 0,
            "checkpoint_path": "checkpoints/reference_classifier/epoch-99.pt",
            "checkpoint_id": G1_CHECKPOINT_ID,
            "checkpoint_file_sha256": G1_CHECKPOINT_SHA256,
            "checkpoint_bytes": G1_CHECKPOINT_BYTES,
            "checkpoint_schema_version": int(get("reference_classifier.checkpoint_schema_version")),
            "class_mapping": {"n01440764": 0, "n02102040": 1, "n02979186": 2, "n03000684": 3, "n03028079": 4, "n03394916": 5, "n03417042": 6, "n03425413": 7, "n03445777": 8, "n03888257": 9},  # literal-ok: frozen Imagenette class mapping
            "class_mapping_source": "results/baseline/g8_e/e1_corrected_v3/scientific_data_identity_manifest.json",
            "descendant_variant": F2_VARIANT,
        },
        "training": {
            "recipe": f2_recipe(),
            "recipe_sha256": f2_recipe_sha256(),
            "logical_dataset_length": EXPECTED_MATERIALIZED,
            "batch_size": EXPECTED_BATCH_SIZE,
            "steps_per_epoch": EXPECTED_STEPS_PER_EPOCH,
            "epochs": EXPECTED_EPOCHS,
            "expected_optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
            "sample_order_rule": "keyed_philox_batch_order(train_seed=0,epoch)_over_manifest_ordinal_filtered_materialized_assignment_rows",
            "validation_rule": "clean_canonical_imagenette160_validation_1000_every_epoch_top1_max_earliest_epoch_tie",
            "validation_role": "F2_checkpoint_selection_only_not_F3_cached_sweep",
            "resume_rule": "latest_authenticated_completed_epoch_only_corrupt_latest_holds_no_older_fallback",
        },
        "execution_profile": {
            "execution_profile_id": PROFILE_ID,
            "host": EXPECTED_HOST,
            "device": EXPECTED_DEVICE,
            "gpu_name": EXPECTED_GPU_NAME,
            "gpu_uuid": EXPECTED_GPU_UUID,
            "lock_file": profile["lock_file"],
            "lock_file_sha256": profile["lock_file_sha256"],
            "amp": False,
        },
        "paths": {
            "f1_runtime_read_only": "results/baseline/g8_f/runtime",
            "f2_runtime": "results/baseline/g8_f/f2_runtime",
            "host_ops": "/home/nick/g8-f-f2-ops",
            "session": "g8f-f2",
        },
        "preflight": dict(preflight),
        "zero_at_authorization_freeze": {
            "artifact_classifier_optimizer_steps": 0,
            "f2_checkpoint_selection_validation_inference": 0,
            "f3_cached_sweep_inference": 0,
            "pass_two": 0,
            "pass_three": 0,
            "fallback": 0,
            "learned_training": 0,
            "test_access": 0,
        },
    }
    body["authorization_id"] = AUTHORIZATION_PREFIX + sha256_bytes(canonical_json(body))
    return body


def verify_authorization(path: Path = AUTHORIZATION_PATH, *, repo_root: Path = REPO_ROOT, require_head_closure: bool = True) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    _require(sha256_bytes(raw) == sha256_bytes(rendered_json(json.loads(raw))), "F2 authorization is not canonical rendered JSON")
    value = json.loads(raw)
    _require(isinstance(value, dict), "F2 authorization must be an object")
    body = dict(value)
    authorization_id = body.pop("authorization_id", None)
    _require(authorization_id == AUTHORIZATION_PREFIX + sha256_bytes(canonical_json(body)), "F2 authorization content identity differs")
    _require(value.get("schema_version") == 1 and value.get("scope") == F2_SCOPE, "F2 authorization scope/schema differs")
    _require(value.get("status") == "FROZEN_OWNER_AUTHORIZED_BEFORE_OPTIMIZER_STEP_1", "F2 authorization is not frozen")
    try:
        expected = build_authorization(
            source_commit=str(value["source_commit"]),
            issued_at=str(value["owner_authorization"]["issued_at"]),
            preflight=value["preflight"],
            repo_root=repo_root,
        )
    except (KeyError, TypeError):
        raise F2AuthorizationHold("F2 authorization structure differs") from None
    _require(value == expected, "F2 authorization differs from the exact source/config-derived contract")
    _require(value.get("zero_at_authorization_freeze") == {"artifact_classifier_optimizer_steps": 0, "f2_checkpoint_selection_validation_inference": 0, "f3_cached_sweep_inference": 0, "pass_two": 0, "pass_three": 0, "fallback": 0, "learned_training": 0, "test_access": 0}, "F2 authorization zero boundary differs")
    _require(value.get("source_closure") == source_closure(str(value.get("source_commit")), repo_root=repo_root), "F2 source closure differs from source commit")
    _require(value["f1"]["completion_id"] == F1_COMPLETION_ID and value["f1"]["corpus_id"] == F1_CORPUS_ID, "F2 authorization F1 identity differs")
    _require(sha256_bytes((repo_root / value["f1"]["completion_path"]).read_bytes()) == F1_COMPLETION_SHA256, "F2 authorization live F1 completion bytes differ")
    _require(sha256_bytes((repo_root / value["f1"]["corpus_manifest_path"]).read_bytes()) == F1_MANIFEST_SHA256, "F2 authorization live F1 manifest bytes differ")
    _require(value["g1_parent"]["checkpoint_id"] == G1_CHECKPOINT_ID and value["g1_parent"]["checkpoint_file_sha256"] == G1_CHECKPOINT_SHA256, "F2 authorization G1 parent differs")
    _require(sha256_bytes((repo_root / value["g1_parent"]["adjudication_path"]).read_bytes()) == G1_ADJUDICATION_SHA256, "F2 authorization live G1 adjudication bytes differ")
    _require(value["training"]["recipe"] == f2_recipe() and value["training"]["recipe_sha256"] == f2_recipe_sha256(), "F2 authorization recipe differs")
    if require_head_closure:
        head = _git("rev-parse", "HEAD", cwd=repo_root).decode("ascii").strip()
        result = subprocess.run(["git", "merge-base", "--is-ancestor", str(value["source_commit"]), head], cwd=repo_root, check=False)
        _require(result.returncode == 0, "F2 scientific source is not an ancestor of launch HEAD")
        for entry in value["source_closure"]:
            current = repo_root / entry["path"]
            _require(current.is_file() and not current.is_symlink(), f"F2 current source is missing/unsafe: {entry['path']}")
            current_raw = current.read_bytes()
            _require(len(current_raw) == entry["bytes"] and sha256_bytes(current_raw) == entry["sha256"], f"F2 current source differs: {entry['path']}")
    return value
