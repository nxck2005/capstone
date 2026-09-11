#!/usr/bin/env python3
"""Execute the prospective W9 v4 ER-9/ER-2 lifecycle.

Only the explicitly authorized future commands are exposed here.  The
historical campaign implementations and their evidence readers live in their
own custody modules; this active entry point has one v4 source, authority,
live-authentication, model, and transactional-runtime path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config.run_config import config_hash, load_experiment  # noqa: E402
from evaluation.er9_search import (  # noqa: E402
    all_configured_pairs,
    feasible_pairs,
    packetisation_floor,
    select_stage1,
    stage1_candidates,
)
from runtime.source_guard import (  # noqa: E402
    SourceGuardHold,
    assert_clean_source_closure,
    assert_v4_manifest_contract,
)
from runtime.w9_authority import (  # noqa: E402
    authenticate_live_w9_pascal,
    load_authority,
    resolve_runtime_root,
)
from training.deterministic_core import canonical_bytes, canonical_sha256  # noqa: E402
from training.er9_v4 import ER9V4CandidateTrainer  # noqa: E402


RESULT_ROOT = REPO / "results/learned/er9"
SOURCE_MANIFEST = RESULT_ROOT / "er_execution_source_manifest_v4.json"
STAGE1_AUTHORITY = RESULT_ROOT / "er9_stage1_execution_authorization_v4.json"
STAGE1_INDEX = RESULT_ROOT / "stage1_v4_terminal_index.json"
STAGE1_EVALUATION_ROOT = RESULT_ROOT / "stage1_v4_evaluations"
STAGE1_SELECTION = RESULT_ROOT / "er9_stage1_selection_v4.json"
ER2_AUTHORITY = REPO / "results/learned/er2_randomized/er2_execution_authorization_v4.json"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"v4 artifact is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"v4 artifact is corrupt: {path}: {exc}") from None
    if not isinstance(value, dict):
        raise RuntimeError(f"v4 artifact is not an object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_immutable(path: Path, value: dict[str, Any]) -> None:
    raw = canonical_bytes(value)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.read_bytes() != raw:
            raise RuntimeError(f"immutable v4 artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = path.open("xb")
    try:
        descriptor.write(raw)
        descriptor.flush()
        os.fsync(descriptor.fileno())
    finally:
        descriptor.close()
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _assert_parity() -> str:
    head = _git("rev-parse", "HEAD")
    origin = _git("rev-parse", "origin/main")
    if head != origin:
        raise RuntimeError(f"local/remote parity differs: {head} != {origin}")
    return head


def _load_manifest() -> dict[str, Any]:
    manifest = _read(SOURCE_MANIFEST)
    body = dict(manifest)
    identifier = body.pop("manifest_id", None)
    if identifier != "er9sourcev4-" + canonical_sha256(body):
        raise RuntimeError("v4 source manifest ID differs")
    try:
        assert_v4_manifest_contract(manifest)
    except SourceGuardHold as exc:
        raise RuntimeError(f"v4 source manifest contract differs: {exc}") from None
    try:
        assert_clean_source_closure(REPO, manifest)
    except SourceGuardHold as exc:
        raise RuntimeError(f"v4 source closure is not authenticated: {exc}") from None
    return manifest


def _validate_stage1_scope(authority: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    """Validate semantic authority fields before any future model allocation."""

    if authority.get("schema_version") != 1:
        raise RuntimeError("v4 Stage-1 authority schema differs")
    if authority.get("status") != "FROZEN_STAGE1_ONLY_PRE_SCIENCE":
        raise RuntimeError("v4 Stage-1 authority is not frozen pre-science")
    if authority.get("authorization_scope") != "W9_ER9_STAGE1_ONLY":
        raise RuntimeError("v4 Stage-1 authority scope differs")
    if authority.get("source_commit") != manifest.get("source_commit"):
        raise RuntimeError("v4 Stage-1 source commit differs")
    if authority.get("config_path") != "configs/er9-digital-pascal-v4.yaml":
        raise RuntimeError("v4 Stage-1 config path differs")
    if authority.get("runtime_root") != "checkpoints/er9_pascal_v4":
        raise RuntimeError("v4 Stage-1 runtime root differs")
    if authority.get("execution_profile_id") != "confessor_pascal_cu126":
        raise RuntimeError("v4 Stage-1 execution profile differs")
    if authority.get("host") != "confessor":
        raise RuntimeError("v4 Stage-1 host differs")
    if authority.get("device") != "cuda:0":
        raise RuntimeError("v4 Stage-1 logical device differs")
    gpu_uuid = authority.get("gpu_uuid")
    gpu_name = authority.get("gpu_name")
    compute = authority.get("compute_capability")
    if (
        not isinstance(gpu_uuid, str)
        or not gpu_uuid.startswith("GPU-")
        or not isinstance(gpu_name, str)
        or not isinstance(compute, str)
        or authority.get("cuda_visible_devices") != gpu_uuid
    ):
        raise RuntimeError("v4 Stage-1 exact GPU binding differs")
    mapping = authority.get("cuda_mapping")
    expected_mapping = {
        "cuda_visible_devices": gpu_uuid,
        "logical_device": "cuda:0",
        "cuda0_gpu_uuid": gpu_uuid,
        "cuda0_gpu_name": gpu_name,
        "cuda0_compute_capability": compute,
        "device_count": 1,
    }
    if mapping != expected_mapping:
        raise RuntimeError("v4 Stage-1 CUDA mapping authority differs")
    if (
        authority.get("sole_writer") is not True
        or authority.get("live_authentication_required_immediately_before_model_and_data") is not True
        or authority.get("source_working_tree_guard_required") is not True
        or authority.get("fresh_initialization_required") is not True
    ):
        raise RuntimeError("v4 Stage-1 execution safety policy differs")
    if any(
        authority.get(key) is not False
        for key in ("stage2_authorized", "production_authorized", "randomized_er2_authorized")
    ):
        raise RuntimeError("v4 Stage-1 authority scope is wider than Stage-1")
    if authority.get("v1_v2_v3_checkpoints_ineligible") is not True or authority.get("smoke_checkpoints_ineligible") is not True:
        raise RuntimeError("v4 Stage-1 historical/smoke custody policy differs")
    if authority.get("test") != "SEALED" or authority.get("test_access") != 0:
        raise RuntimeError("v4 Stage-1 test boundary differs")
    expected_counters = {
        "new_er9_v4_training": 0,
        "randomized_er2_scientific_training": 0,
        "g11": 0,
        "w10": 0,
        "learned_test_inference": 0,
        "model_facing_test_access": 0,
    }
    if authority.get("pre_execution_counters") != expected_counters:
        raise RuntimeError("v4 Stage-1 pre-execution counters differ")


def _validate_er2_scope(authority: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    """Validate the future ER-2 authority before importing its data path."""

    if authority.get("schema_version") != 1:
        raise RuntimeError("v4 ER-2 authority schema differs")
    if authority.get("status") != "FROZEN_ER2_ONLY_PRE_SCIENCE":
        raise RuntimeError("v4 ER-2 authority is not frozen pre-science")
    if authority.get("authorization_scope") != "W9_ER2_RANDOMIZED_ONLY":
        raise RuntimeError("v4 ER-2 authority scope differs")
    if authority.get("source_commit") != manifest.get("source_commit") or authority.get("source_binding") != manifest:
        raise RuntimeError("v4 ER-2 source binding differs")
    if authority.get("config_path") != "configs/learned-er2-randomized-pascal-v4.yaml":
        raise RuntimeError("v4 ER-2 config path differs")
    if authority.get("runtime_root") != "checkpoints/er2_randomized_pascal_v4":
        raise RuntimeError("v4 ER-2 runtime root differs")
    if authority.get("execution_profile_id") != "confessor_pascal_cu126" or authority.get("host") != "confessor":
        raise RuntimeError("v4 ER-2 execution profile differs")
    if authority.get("device") != "cuda:0" or authority.get("cuda_visible_devices") != authority.get("gpu_uuid"):
        raise RuntimeError("v4 ER-2 CUDA mapping authority differs")
    gpu_uuid = authority.get("gpu_uuid")
    mapping = authority.get("cuda_mapping")
    expected_mapping = {
        "cuda_visible_devices": gpu_uuid,
        "logical_device": "cuda:0",
        "cuda0_gpu_uuid": gpu_uuid,
        "cuda0_gpu_name": authority.get("gpu_name"),
        "cuda0_compute_capability": authority.get("compute_capability"),
        "device_count": 1,
    }
    if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-") or mapping != expected_mapping:
        raise RuntimeError("v4 ER-2 exact CUDA identity differs")
    if (
        authority.get("randomized_er2_authorized") is not True
        or authority.get("sole_writer") is not True
        or authority.get("live_authentication_required_immediately_before_model_and_data") is not True
        or authority.get("source_working_tree_guard_required") is not True
        or authority.get("fresh_initialization_required") is not True
    ):
        raise RuntimeError("v4 ER-2 execution safety policy differs")
    if authority.get("test") != "SEALED" or authority.get("test_access") != 0:
        raise RuntimeError("v4 ER-2 test boundary differs")
    expected_counters = {
        "new_er9_v4_training": 0,
        "randomized_er2_scientific_training": 0,
        "g11": 0,
        "w10": 0,
        "learned_test_inference": 0,
        "model_facing_test_access": 0,
    }
    if authority.get("pre_execution_counters") != expected_counters:
        raise RuntimeError("v4 ER-2 pre-execution counters differ")


def _solver_binding(config: Any) -> dict[str, Any]:
    floor = packetisation_floor(int(config.resolved["k"]))
    configured = all_configured_pairs()
    admissible = feasible_pairs(floor.payload_bits)
    candidates = stage1_candidates(floor.payload_bits)
    return {
        "k_symbols": int(config.resolved["k"]),
        "metadata_bits": 1,
        "packet_floor": floor.as_dict(),
        "configured_pairs": [item.as_dict() for item in configured],
        "admissible_pairs": [item.as_dict() for item in admissible],
        "rejected_pairs": [
            {**item.as_dict(), "reason": "raw_bound_exceeds_A_floor"}
            for item in configured
            if item not in admissible
        ],
        "stage1_candidates": [item.as_dict() for item in candidates],
    }


def _load_stage1() -> tuple[dict[str, Any], dict[str, Any], Any, dict[str, Any]]:
    manifest = _load_manifest()
    authority = load_authority(STAGE1_AUTHORITY, kind="W9_ER9_STAGE1_EXECUTION_AUTHORITY_V4")
    authority_body = dict(authority)
    authority_id = authority_body.pop("authority_id", None)
    if authority_id != "w9er9stage1v4auth-" + canonical_sha256(authority_body):
        raise RuntimeError("v4 Stage-1 authority ID differs")
    _validate_stage1_scope(authority, manifest)
    source_record = authority.get("source_manifest")
    if not isinstance(source_record, dict):
        raise RuntimeError("v4 Stage-1 source record is missing")
    if (
        source_record.get("manifest_id") != manifest.get("manifest_id")
        or source_record.get("path") != str(SOURCE_MANIFEST.relative_to(REPO))
        or source_record.get("sha256") != _sha(SOURCE_MANIFEST)
        or authority.get("source_commit") != manifest.get("source_commit")
        or authority.get("source_binding") != manifest
    ):
        raise RuntimeError("v4 Stage-1 source binding differs")
    if authority.get("config_path") != "configs/er9-digital-pascal-v4.yaml":
        raise RuntimeError("v4 Stage-1 config path differs")
    config = load_experiment(authority["config_path"], train_seed=0, channel_seed=0)
    if authority.get("config_hash") != config_hash(config):
        raise RuntimeError("v4 Stage-1 config hash differs")
    if authority.get("config_source_blob_sha256") != manifest["relevant_config_sha256"][authority["config_path"]]:
        raise RuntimeError("v4 Stage-1 config source identity differs")
    solver = _solver_binding(config)
    proof = authority.get("packet_budget_admissibility_proof")
    if not isinstance(proof, dict):
        raise RuntimeError("v4 Stage-1 solver proof is missing")
    if (
        proof.get("k_symbols") != solver["k_symbols"]
        or proof.get("metadata_bits") != solver["metadata_bits"]
        or proof.get("packet_floor") != solver["packet_floor"]
        or proof.get("configured_pairs") != solver["configured_pairs"]
        or proof.get("admissible_pairs") != solver["admissible_pairs"]
        or proof.get("rejected_pairs") != solver["rejected_pairs"]
        or proof.get("stage1_candidates") != solver["stage1_candidates"]
        or proof.get("configured_pair_count") != len(solver["configured_pairs"])
        or proof.get("admissible_pair_count") != len(solver["admissible_pairs"])
    ):
        raise RuntimeError("v4 Stage-1 authority arithmetic differs from the solver")
    if authority.get("stage1_candidates") != solver["stage1_candidates"]:
        raise RuntimeError("v4 Stage-1 candidate order differs from the solver")
    if (
        authority.get("k_symbols") != solver["k_symbols"]
        or authority.get("metadata_bits") != solver["metadata_bits"]
        or authority.get("packet_floor") != solver["packet_floor"]
        or authority.get("configured_pair_count") != len(solver["configured_pairs"])
        or authority.get("admissible_pair_count") != len(solver["admissible_pairs"])
        or authority.get("admissible_pairs") != solver["admissible_pairs"]
        or authority.get("rejected_pairs") != solver["rejected_pairs"]
        or authority.get("stage1_candidate_count") != len(solver["stage1_candidates"])
        or authority.get("stage1_training_count") != len(solver["stage1_candidates"])
    ):
        raise RuntimeError("v4 Stage-1 top-level solver binding differs")
    return manifest, authority, config, solver


def _source_binding(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return dict(manifest)


def _candidate_slug(candidate: Mapping[str, Any]) -> str:
    return f"D{int(candidate['transmit_dim'])}_b{int(candidate['quantiser_bits'])}"


def _candidate_runtime(authority: Mapping[str, Any], candidate: Mapping[str, Any]) -> Path:
    return resolve_runtime_root(REPO, authority) / "stage1" / _candidate_slug(candidate)


def _live_pascal(authority: Mapping[str, Any], config: Any) -> dict[str, Any]:
    """Authenticate the source and exact process-visible Pascal immediately."""

    return authenticate_live_w9_pascal(
        REPO,
        authority,
        config_hash=config_hash(config),
    )


def _candidate_identity_fields(candidate: Mapping[str, Any]) -> dict[str, int]:
    if not isinstance(candidate, Mapping):
        raise RuntimeError("v4 candidate identity is not a mapping")
    try:
        transmit_dim = int(candidate["transmit_dim"])
        quantiser_bits = int(candidate["quantiser_bits"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"v4 candidate identity is malformed: {exc}") from None
    if transmit_dim <= 0 or quantiser_bits <= 0:
        raise RuntimeError("v4 candidate identity is not positive")
    return {
        "transmit_dim": transmit_dim,
        "quantiser_bits": quantiser_bits,
    }


def run_stage1_train(*, resume: bool = False) -> None:
    _assert_parity()
    manifest, authority, config, _solver = _load_stage1()
    if authority.get("fresh_initialization_required") is not True:
        raise RuntimeError("v4 Stage-1 is not fresh-initialization bound")
    authority_root = resolve_runtime_root(REPO, authority)
    if not resume and (authority_root.exists() or authority_root.is_symlink()):
        raise RuntimeError("v4 Stage-1 runtime root already exists; fresh initialization is not safe")
    entries: list[dict[str, Any]] = []
    for candidate in authority["stage1_candidates"]:
        candidate = _candidate_identity_fields(candidate)
        runtime = _candidate_runtime(authority, candidate)
        # Live authentication is deliberately the last operation before the
        # trainer constructs the model, optimizer, scaler, or any dataset.
        live = _live_pascal(authority, config)
        trainer = ER9V4CandidateTrainer(
            config,
            transmit_dim=candidate["transmit_dim"],
            quantiser_bits=candidate["quantiser_bits"],
            device=str(authority["device"]),
            runtime_root=runtime,
            source_binding=_source_binding(manifest),
            campaign_id="er9_stage1_v4",
            run_id=f"er9-stage1-v4-{_candidate_slug(candidate)}",
            live_authentication=live,
            resume=resume,
        )
        terminal = trainer.run()
        if terminal is None:
            raise RuntimeError(f"ER-9 v4 candidate did not terminalize: {runtime}")
        entries.append(
            {
                "candidate": candidate,
                "runtime_root": str(runtime.relative_to(REPO)),
                "terminal": terminal,
                "terminal_sha256": _sha(runtime / "run_terminal.json"),
                "selected_epoch": terminal["selected_epoch"],
                "selected_checkpoint_sha256": terminal["selected_checkpoint_sha256"],
                "test": "SEALED",
                "test_access": 0,
            }
        )
    body = {
        "schema_version": 1,
        "artifact_role": "ER9_STAGE1_V4_TERMINAL_INDEX",
        "source_commit": manifest["source_commit"],
        "source_manifest_id": manifest["manifest_id"],
        "authority_id": authority["authority_id"],
        "runtime_root": authority["runtime_root"],
        "candidate_order": entries,
        "training_count": len(entries),
        "selection_performed": False,
        "test": "SEALED",
        "test_access": 0,
    }
    body["index_id"] = "er9stage1v4index-" + canonical_sha256(body)
    _write_immutable(STAGE1_INDEX, body)
    print(f"ER-9 v4 Stage-1 training complete: {len(entries)} terminal candidates")


def _load_terminal(runtime: Path) -> dict[str, Any]:
    terminal = _read(runtime / "run_terminal.json")
    if terminal.get("test") != "SEALED" or terminal.get("test_access") != 0:
        raise RuntimeError(f"v4 terminal test boundary differs: {runtime}")
    return terminal


def run_stage1_evaluate() -> None:
    """Evaluate selected v4 checkpoints only after a complete Stage-1 run."""

    _assert_parity()
    manifest, authority, config, _solver = _load_stage1()
    index = _read(STAGE1_INDEX)
    expected_candidates = [
        _candidate_identity_fields(candidate) for candidate in authority["stage1_candidates"]
    ]
    if (
        index.get("schema_version") != 1
        or index.get("artifact_role") != "ER9_STAGE1_V4_TERMINAL_INDEX"
        or index.get("source_commit") != manifest["source_commit"]
        or index.get("source_manifest_id") != manifest["manifest_id"]
        or index.get("selection_performed") is not False
        or index.get("authority_id") != authority["authority_id"]
        or index.get("training_count") != len(expected_candidates)
        or index.get("test") != "SEALED"
        or index.get("test_access") != 0
    ):
        raise RuntimeError("v4 Stage-1 terminal index is not an unselected complete index")
    entries = index.get("candidate_order")
    if not isinstance(entries, list) or len(entries) != len(expected_candidates):
        raise RuntimeError("v4 Stage-1 terminal index candidate count differs")
    for item, candidate in zip(entries, expected_candidates, strict=True):
        if not isinstance(item, Mapping) or _candidate_identity_fields(item.get("candidate", {})) != candidate:
            raise RuntimeError("v4 Stage-1 terminal index candidate order differs")
        expected_runtime = _candidate_runtime(authority, candidate)
        if item.get("runtime_root") != str(expected_runtime.relative_to(REPO)):
            raise RuntimeError("v4 Stage-1 terminal index runtime binding differs")
    from evaluation.er9_campaign import (  # noqa: PLC0415
        collect_validation_features,
        evaluate_candidate_at_snr,
        fit_entropy_model,
    )

    for item, candidate in zip(entries, expected_candidates, strict=True):
        runtime = _candidate_runtime(authority, candidate)
        terminal = _load_terminal(runtime)
        if item.get("terminal_sha256") != _sha(runtime / "run_terminal.json"):
            raise RuntimeError("v4 Stage-1 terminal index terminal hash differs")
        live = _live_pascal(authority, config)
        # Trainer construction is the canonical v4 model/optimizer/scaler
        # construction.  It also authenticates the transactional prefix.
        trainer = ER9V4CandidateTrainer(
            config,
            transmit_dim=candidate["transmit_dim"],
            quantiser_bits=candidate["quantiser_bits"],
            device=str(authority["device"]),
            runtime_root=runtime,
            source_binding=_source_binding(manifest),
            campaign_id="er9_stage1_v4",
            run_id=f"er9-stage1-v4-{_candidate_slug(candidate)}",
            live_authentication=live,
            resume=True,
        )
        selected_epoch = int(terminal["selected_epoch"])
        trainer.runtime.restore_epoch(selected_epoch, trainer.model, trainer.optimizer, trainer.scaler)
        model = trainer.model
        entropy, entropy_record = fit_entropy_model(model, config, device=authority["device"], num_workers=trainer.num_workers)
        features = collect_validation_features(model, config, device=authority["device"], num_workers=trainer.num_workers)
        result = evaluate_candidate_at_snr(
            model,
            config,
            entropy=entropy,
            dimension=candidate["transmit_dim"],
            quantiser_bits=candidate["quantiser_bits"],
            snr_db=int(authority["evaluation_snr_db"]),
            device=authority["device"],
            num_workers=trainer.num_workers,
            validation_features=features,
            include_per_image=False,
        )
        body = {
            "schema_version": 1,
            "artifact_role": "ER9_STAGE1_V4_CANDIDATE_EVALUATION",
            "source_commit": manifest["source_commit"],
            "source_manifest_id": manifest["manifest_id"],
            "authority_id": authority["authority_id"],
            "candidate": candidate,
            "runtime_root": str(runtime.relative_to(REPO)),
            "terminal_sha256": _sha(runtime / "run_terminal.json"),
            "selected_epoch": selected_epoch,
            "entropy_model": entropy_record,
            "real_chain_validation": result,
            "test": "SEALED",
            "test_access": 0,
        }
        body["evaluation_id"] = "er9stage1v4eval-" + canonical_sha256(body)
        _write_immutable(STAGE1_EVALUATION_ROOT / f"{_candidate_slug(candidate)}.json", body)
    print("ER-9 v4 Stage-1 validation evaluations complete")


def run_stage1_select() -> None:
    _assert_parity()
    manifest, authority, _config, _solver = _load_stage1()
    rows: list[dict[str, Any]] = []
    files: list[str] = []
    expected_candidates = [_candidate_identity_fields(candidate) for candidate in authority["stage1_candidates"]]
    for candidate in expected_candidates:
        path = STAGE1_EVALUATION_ROOT / f"{_candidate_slug(candidate)}.json"
        value = _read(path)
        expected_runtime = _candidate_runtime(authority, candidate)
        if (
            value.get("schema_version") != 1
            or value.get("artifact_role") != "ER9_STAGE1_V4_CANDIDATE_EVALUATION"
            or value.get("source_commit") != manifest["source_commit"]
            or value.get("source_manifest_id") != manifest["manifest_id"]
            or value.get("authority_id") != authority["authority_id"]
            or value.get("candidate") != candidate
            or value.get("runtime_root") != str(expected_runtime.relative_to(REPO))
            or value.get("terminal_sha256") != _sha(expected_runtime / "run_terminal.json")
            or value.get("test") != "SEALED"
            or value.get("test_access") != 0
        ):
            raise RuntimeError("v4 Stage-1 evaluation binding differs")
        result = value.get("real_chain_validation")
        if not isinstance(result, dict):
            raise RuntimeError("v4 Stage-1 real-chain result is missing")
        selected_phy = result.get("selected")
        if not isinstance(selected_phy, dict) or not isinstance(result.get("validation_n_correct"), int):
            raise RuntimeError("v4 Stage-1 evaluation count is not exact")
        rows.append(
            {
                **candidate,
                "n_correct": int(result["validation_n_correct"]),
                "n_total": int(result["validation_total"]),
                "selected_phy": selected_phy,
                "evaluation_id": value["evaluation_id"],
                "runtime_root": value["runtime_root"],
                "selected_epoch": value["selected_epoch"],
                "test": "SEALED",
                "test_access": 0,
            }
        )
        files.append(str(path.relative_to(REPO)))
    selected = dict(select_stage1(rows))
    body = {
        "schema_version": 1,
        "artifact_role": "ER9_STAGE1_V4_SELECTION",
        "source_commit": manifest["source_commit"],
        "source_manifest_id": manifest["manifest_id"],
        "authority_id": authority["authority_id"],
        "candidate_order": [
            _candidate_identity_fields(candidate) for candidate in authority["stage1_candidates"]
        ],
        "candidate_evaluation_files": files,
        "rows": rows,
        "selected": selected,
        "selection_metric": "validation_n_correct",
        "tie_break": "smallest_transmit_dim",
        "test": "SEALED",
        "test_access": 0,
    }
    body["selection_id"] = "er9stage1v4selection-" + canonical_sha256(body)
    _write_immutable(STAGE1_SELECTION, body)
    print(f"ER-9 v4 Stage-1 selection complete: D{selected['transmit_dim']}_b{selected['quantiser_bits']}")


def _load_er2_authority(path: Path) -> tuple[dict[str, Any], dict[str, Any], Any]:
    manifest = _load_manifest()
    authority = load_authority(path, kind="W9_ER2_RANDOMIZED_EXECUTION_AUTHORITY_V4")
    body = dict(authority)
    identifier = body.pop("authority_id", None)
    if identifier != "w9er2randomizedv4auth-" + canonical_sha256(body):
        raise RuntimeError("v4 ER-2 authority ID differs")
    _validate_er2_scope(authority, manifest)
    source_record = authority.get("source_manifest")
    if (
        not isinstance(source_record, Mapping)
        or source_record.get("path") != str(SOURCE_MANIFEST.relative_to(REPO))
        or source_record.get("manifest_id") != manifest["manifest_id"]
        or source_record.get("sha256") != _sha(SOURCE_MANIFEST)
    ):
        raise RuntimeError("v4 ER-2 source manifest record differs")
    config = load_experiment(authority["config_path"], train_seed=0, channel_seed=0)
    if authority.get("config_hash") != config_hash(config):
        raise RuntimeError("v4 ER-2 config hash differs")
    if authority.get("config_source_blob_sha256") != manifest["relevant_config_sha256"][authority["config_path"]]:
        raise RuntimeError("v4 ER-2 config source identity differs")
    return manifest, authority, config


def run_er2_train(authority_path: Path = ER2_AUTHORITY, *, resume: bool = False) -> None:
    _assert_parity()
    manifest, authority, config = _load_er2_authority(authority_path)
    if authority.get("randomized_er2_authorized") is not True:
        raise RuntimeError("v4 ER-2 authority is not scoped for ER-2")
    runtime = resolve_runtime_root(REPO, authority)
    if not resume and (runtime.exists() or runtime.is_symlink()):
        raise RuntimeError("v4 ER-2 runtime root already exists; fresh initialization is not safe")
    live = _live_pascal(authority, config)
    from training.er2_v4 import ER2V4RandomizedTrainer  # noqa: PLC0415

    trainer = ER2V4RandomizedTrainer(
        config,
        device=str(authority["device"]),
        runtime_root=runtime,
        source_binding=_source_binding(manifest),
        campaign_id="er2_randomized_v4",
        run_id="er2-randomized-v4-train0-channel0",
        live_authentication=live,
        resume=resume,
    )
    terminal = trainer.run()
    if terminal is None:
        raise RuntimeError("v4 ER-2 run did not terminalize")
    print("randomized ER-2 v4 training complete")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage1-train", "stage1-evaluate", "stage1-select", "er2-train"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--er2-authority", type=Path, default=ER2_AUTHORITY)
    args = parser.parse_args(argv)
    if args.action == "stage1-train":
        run_stage1_train(resume=args.resume)
    elif args.action == "stage1-evaluate":
        run_stage1_evaluate()
    elif args.action == "stage1-select":
        run_stage1_select()
    else:
        run_er2_train(args.er2_authority, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
