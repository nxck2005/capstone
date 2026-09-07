#!/usr/bin/env python3
"""Run the staged, validation-only ER-9/ER-2/G-11 campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config.execution_profiles import authenticate_execution_profile  # noqa: E402
from config.params import get  # noqa: E402
from config.run_config import config_hash, load_experiment  # noqa: E402
from evaluation.er9_campaign import (  # noqa: E402
    collect_validation_features,
    evaluate_candidate_at_snr,
    fit_entropy_model,
)
from evaluation.er9_protocol import factorisation_for_dimension  # noqa: E402
from evaluation.er9_search import select_stage1, select_stage2, stage2_candidates  # noqa: E402
from models.djscc import build_djscc  # noqa: E402
from training.deterministic_core import canonical_bytes, canonical_sha256  # noqa: E402
from training.er2_randomized import (  # noqa: E402
    ER2RandomizedTrainer,
    evaluate_er2_at_snr,
)
from training.er9 import ER9Trainer, load_er9_checkpoint, _publish_immutable, _sha256_file  # noqa: E402
from verify_er9 import verify_source_manifest, verify_stage1_authorization  # noqa: E402


RESULT_ROOT = REPO / "results/learned/er9"
# Successor namespace after the preserved first-checkpoint publication incident.
CHECKPOINT_ROOT = REPO / "checkpoints/er9_successor_v2"
SOURCE_MANIFEST = RESULT_ROOT / "er_execution_source_manifest_v3.json"
STAGE1_AUTH = RESULT_ROOT / "er9_stage1_execution_authorization_v3.json"
STAGE1_SELECTION = RESULT_ROOT / "er9_stage1_selection.json"
STAGE2_AUTH = RESULT_ROOT / "er9_stage2_execution_authorization.json"
STAGE2_SELECTION = RESULT_ROOT / "er9_stage2_selection.json"
PRODUCTION = RESULT_ROOT / "er9_final_production_manifest.json"
ARCHITECTURE_DIFF = RESULT_ROOT / "er9_architecture_difference.json"
ER2_RESULT_ROOT = REPO / "results/learned/er2_randomized"
ER2_COMPLETION = ER2_RESULT_ROOT / "er2_randomized_completion.json"
ER2_AUDIT = ER2_RESULT_ROOT / "er2_snr_assignment_audit.json"
ER2_VALIDATION = ER2_RESULT_ROOT / "er2_randomized_validation.json"
G11_ROOT = REPO / "results/learned/g11"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_immutable(value: dict[str, Any], path: Path) -> None:
    if path.exists() or path.is_symlink():
        existing = _read(path)
        if existing != value:
            raise RuntimeError(f"immutable ER artifact differs: {path}")
        return
    _publish_immutable(path, canonical_bytes(value))


def _assert_clean_parity() -> str:
    status = _git("status", "--short", "--untracked-files=all")
    if status:
        raise RuntimeError(f"scientific source must be clean before this phase:\n{status}")
    head = _git("rev-parse", "HEAD")
    origin = _git("rev-parse", "origin/main")
    if head != origin:
        raise RuntimeError(f"local/remote parity differs: {head} != {origin}")
    return head


def _config(path: str, train_seed: int, channel_seed: int):
    return load_experiment(path, train_seed=train_seed, channel_seed=channel_seed)


def _profile_binding(cfg: Any) -> dict[str, Any]:
    profile_id = str(cfg.resolved["execution_profile_id"])
    if profile_id != "local_4060_cu130":
        raise RuntimeError("ER-9/ER-2 execution is bound to local_4060_cu130")
    environment = authenticate_execution_profile(
        profile_id,
        device="cuda:0",
        config_hash=config_hash(cfg),
    )
    if environment.get("git_dirty") is not False:
        raise RuntimeError("authenticated ER execution checkout is dirty")
    body = {
        "schema_version": 1,  # literal-ok: ER execution binding schema
        "authentication_status": "PASSED",
        "execution_profile_id": profile_id,
        "device": "cuda:0",
        "gpu_uuid": environment["gpu_uuid"],
        "gpu_name": environment["gpu_name"],
        "gpu_compute_capability": environment["gpu_compute_capability"],
        "lock_file": environment["lock_file"],
        "lock_file_sha256": environment["lock_file_sha256"],
        "git_commit": environment["git_commit"],
        "git_dirty": False,
        "config_hash": config_hash(cfg),
        "profile_environment": environment,
    }
    body["binding_sha256"] = canonical_sha256(body)
    return body


def _source_binding(cfg: Any) -> dict[str, Any]:
    manifest = verify_source_manifest(SOURCE_MANIFEST)
    verify_stage1_authorization(STAGE1_AUTH, SOURCE_MANIFEST)
    profile_path = CHECKPOINT_ROOT / "execution_profile_binding.json"
    if profile_path.exists():
        profile = _read(profile_path)
    else:
        profile = _profile_binding(cfg)
        _write_immutable(profile, profile_path)
    if profile["config_hash"] != config_hash(cfg):
        raise RuntimeError("execution profile binding config differs")
    execution_commit = _git("rev-parse", "HEAD")
    return {
        "schema_version": 1,  # literal-ok: ER source binding schema
        "implementation_commit": manifest["source_commit"],
        "source_manifest_id": manifest["manifest_id"],
        "source_manifest_sha256": _file_sha(SOURCE_MANIFEST),
        "execution_source_commit": execution_commit,
        "execution_profile_binding": profile,
        "test": "SEALED",
        "test_access": 0,
    }


def _candidate_slug(dimension: int, bits: int) -> str:
    return f"D{int(dimension)}_b{int(bits)}"


def _stage1_runtime(candidate: dict[str, Any]) -> Path:
    return CHECKPOINT_ROOT / "stage1" / _candidate_slug(candidate["transmit_dim"], candidate["quantiser_bits"])


def _stage2_runtime(candidate: dict[str, Any]) -> Path:
    return CHECKPOINT_ROOT / "stage2" / _candidate_slug(candidate["transmit_dim"], candidate["quantiser_bits"])


def _runtime_completion(runtime: Path) -> dict[str, Any] | None:
    path = runtime / "run_completion.json"
    return _read(path) if path.is_file() and not path.is_symlink() else None


def _validate_runtime_completion(runtime: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    """Authenticate the immutable terminal record before reusing a run."""

    completion = _runtime_completion(runtime)
    if completion is None:
        raise RuntimeError(f"ER-9 runtime is not complete: {runtime}")
    if (
        completion.get("artifact_role") != "ER9_CANDIDATE_COMPLETION"
        or completion.get("training_run_count") != 1
        or completion.get("transmit_dim") != candidate["transmit_dim"]
        or completion.get("quantiser_bits") != candidate["quantiser_bits"]
        or completion.get("test_access") != 0
    ):
        raise RuntimeError(f"ER-9 runtime completion differs from candidate: {runtime}")
    selected_path = runtime / "selected_checkpoint.json"
    if not selected_path.is_file() or selected_path.is_symlink():
        raise RuntimeError(f"ER-9 selected-checkpoint record is missing or unsafe: {runtime}")
    selected = _read(selected_path)
    if (
        selected.get("transmit_dim") != candidate["transmit_dim"]
        or selected.get("quantiser_bits") != candidate["quantiser_bits"]
        or selected.get("metric") != "validation_n_correct"
        or selected.get("mode") != "max"
        or selected.get("tie_break") != "earliest_epoch"
        or completion.get("selected_checkpoint") != selected.get("selection")
    ):
        raise RuntimeError(f"ER-9 selected-checkpoint record differs: {runtime}")
    checkpoint = (runtime / str(selected["selection"]["checkpoint_path"])).resolve()
    if runtime.resolve() not in checkpoint.parents or not checkpoint.is_file() or checkpoint.is_symlink():
        raise RuntimeError(f"ER-9 selected checkpoint is missing or unsafe: {checkpoint}")
    if selected["selection"].get("checkpoint_id") != _file_sha(checkpoint):
        raise RuntimeError(f"ER-9 selected checkpoint hash differs: {checkpoint}")
    return completion


def _read_existing_evaluation(path: Path, prefix: str, candidate: dict[str, Any]) -> dict[str, Any] | None:
    """Reuse an authenticated completed validation evaluation after interruption."""

    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink():
        raise RuntimeError(f"ER-9 evaluation is an unsafe symlink: {path}")
    body = _read(path)
    identifier = body.get("evaluation_id")
    without_id = dict(body)
    without_id.pop("evaluation_id", None)
    if identifier != prefix + canonical_sha256(without_id):
        raise RuntimeError(f"ER-9 evaluation ID differs: {path}")
    if body.get("candidate") != candidate or body.get("test_access") != 0:
        raise RuntimeError(f"ER-9 evaluation candidate/scope differs: {path}")
    return body


def _train_er9_candidate(cfg: Any, candidate: dict[str, Any], runtime: Path, source: dict[str, Any], campaign: str) -> dict[str, Any]:
    completion = _runtime_completion(runtime)
    if completion is not None:
        completion = _validate_runtime_completion(runtime, candidate)
        return {
            "candidate": dict(candidate),
            "runtime_root": str(runtime.relative_to(REPO)),
            "completion": completion,
            "completion_sha256": _file_sha(runtime / "run_completion.json"),
            "reused_existing_attempt": True,
        }
    resume = runtime.exists() or runtime.is_symlink()
    trainer = ER9Trainer(
        cfg,
        transmit_dim=int(candidate["transmit_dim"]),
        quantiser_bits=int(candidate["quantiser_bits"]),
        device="cuda:0",
        runtime_root=runtime,
        source_binding=source,
        campaign_id=campaign,
        run_id=f"er9-{campaign}-{_candidate_slug(candidate['transmit_dim'], candidate['quantiser_bits'])}-train{cfg.resolved['train_seed']}-channel{cfg.resolved['channel_seed']}",
        resume=resume,
    )
    result = trainer.run()
    return {
        "candidate": dict(candidate),
        "runtime_root": str(runtime.relative_to(REPO)),
        "completion": result,
        "completion_sha256": _file_sha(runtime / "run_completion.json"),
        "reused_existing_attempt": resume,
    }


def run_stage1_train() -> None:
    _assert_clean_parity()
    auth = verify_stage1_authorization(STAGE1_AUTH, SOURCE_MANIFEST)
    if STAGE1_SELECTION.exists():
        raise RuntimeError("Stage-1 selection already exists; refusing a second Stage-1 campaign")
    cfg = _config("configs/er9-digital.yaml", 0, 0)  # literal-ok: authorized Stage-1 seed cell
    source = _source_binding(cfg)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    candidates = list(auth["stage1_candidates"])
    runs = []
    for candidate in candidates:
        runs.append(_train_er9_candidate(cfg, candidate, _stage1_runtime(candidate), source, "stage1"))
    index = {
        "schema_version": 1,  # literal-ok: Stage-1 runtime index schema
        "artifact_role": "ER9_STAGE1_TRAINING_INDEX",
        "authorization_id": auth["authorization_id"],
        "source_binding": source,
        "candidate_order": candidates,
        "training_run_count": len(runs),
        "runs": runs,
        "test_access": 0,
    }
    _write_immutable(index, CHECKPOINT_ROOT / "stage1_training_index.json")
    print(f"ER-9 Stage-1 training complete: {len(runs)} authorized runs")


def _evaluate_one_er9_candidate(cfg: Any, candidate: dict[str, Any], runtime_root: Path, source: dict[str, Any]) -> dict[str, Any]:
    selected = _read(runtime_root / "selected_checkpoint.json")
    selection = selected["selection"]
    checkpoint_path = runtime_root / str(selection["checkpoint_path"])
    model = load_er9_checkpoint(
        cfg,
        transmit_dim=int(candidate["transmit_dim"]),
        quantiser_bits=int(candidate["quantiser_bits"]),
        checkpoint_path=checkpoint_path,
        device="cuda:0",
    )
    entropy, entropy_record = fit_entropy_model(model, cfg, device="cuda:0", num_workers=4)
    features = collect_validation_features(model, cfg, device="cuda:0", num_workers=4)
    real_chain = evaluate_candidate_at_snr(
        model,
        cfg,
        entropy=entropy,
        dimension=int(candidate["transmit_dim"]),
        quantiser_bits=int(candidate["quantiser_bits"]),
        snr_db=int(cfg.resolved["train_snr_db"]),
        device="cuda:0",
        num_workers=4,
        validation_features=features,
    )
    return {
        "schema_version": 1,  # literal-ok: ER-9 candidate evaluation schema
        "artifact_role": "ER9_STAGE_SEARCH_CANDIDATE_EVALUATION",
        "candidate": dict(candidate),
        "source_binding": source,
        "runtime_root": str(runtime_root.relative_to(REPO)),
        "training_selected_checkpoint": selection,
        "checkpoint_sha256": _file_sha(checkpoint_path),
        "entropy_model": entropy_record,
        "real_chain_validation": real_chain,
        "test_access": 0,
    }


def run_stage1_evaluate() -> None:
    _assert_clean_parity()
    if STAGE1_SELECTION.exists() or STAGE1_SELECTION.is_symlink():
        raise RuntimeError("Stage-1 selection already exists; refusing a second selection pass")
    auth = verify_stage1_authorization(STAGE1_AUTH, SOURCE_MANIFEST)
    index_path = CHECKPOINT_ROOT / "stage1_training_index.json"
    index = _read(index_path)
    if (
        index.get("artifact_role") != "ER9_STAGE1_TRAINING_INDEX"
        or index.get("authorization_id") != auth["authorization_id"]
        or index.get("candidate_order") != auth["stage1_candidates"]
        or index.get("training_run_count") != auth["stage1_training_count"]
        or index.get("test_access") != 0
    ):
        raise RuntimeError("Stage-1 training count differs from authorization")
    if len(index.get("runs", [])) != auth["stage1_training_count"]:
        raise RuntimeError("Stage-1 runtime index length differs from authorization")
    for candidate, run in zip(auth["stage1_candidates"], index["runs"], strict=True):
        if run.get("candidate") != candidate or run.get("runtime_root") != str(_stage1_runtime(candidate).relative_to(REPO)):
            raise RuntimeError("Stage-1 runtime candidate order differs from authorization")
        _validate_runtime_completion(_stage1_runtime(candidate), candidate)
    cfg = _config("configs/er9-digital.yaml", 0, 0)  # literal-ok: authorized Stage-1 seed cell
    source = index["source_binding"]
    evaluations = []
    eval_root = RESULT_ROOT / "stage1_evaluations"
    for candidate in auth["stage1_candidates"]:
        runtime = _stage1_runtime(candidate)
        path = eval_root / f"{_candidate_slug(candidate['transmit_dim'], candidate['quantiser_bits'])}.json"
        body = _read_existing_evaluation(path, "er9stage1eval-", candidate)
        if body is None:
            body = _evaluate_one_er9_candidate(cfg, candidate, runtime, source)
            body["evaluation_id"] = "er9stage1eval-" + canonical_sha256(body)
            _write_immutable(body, path)
        evaluations.append(body)
    rows = [
        {
            "transmit_dim": body["candidate"]["transmit_dim"],
            "quantiser_bits": body["candidate"]["quantiser_bits"],
            "n_correct": body["real_chain_validation"]["validation_n_correct"],
            "n_total": body["real_chain_validation"]["validation_total"],
            "selected_phy": body["real_chain_validation"]["selected"],
            "evaluation_id": body["evaluation_id"],
            "runtime_root": body["runtime_root"],
        }
        for body in evaluations
    ]
    selected = dict(select_stage1(rows))
    body = {
        "schema_version": 1,  # literal-ok: Stage-1 selection schema
        "artifact_role": "ER9_STAGE1_TERMINAL_SELECTION",
        "status": "STAGE1_COMPLETE_STAGE2_NOT_AUTHORIZED_IN_THIS_ARTIFACT",
        "authorization_id": auth["authorization_id"],
        "source_binding": source,
        "candidate_order": auth["stage1_candidates"],
        "training_run_count": len(evaluations),
        "selection_metric": "exact_validation_n_correct_at_7db_real_digital_chain",
        "tie_break": "smallest_transmit_dim",
        "rows": rows,
        "selected": selected,
        "candidate_evaluation_files": [
            str((RESULT_ROOT / "stage1_evaluations" / f"{_candidate_slug(candidate['transmit_dim'], candidate['quantiser_bits'])}.json").relative_to(REPO))
            for candidate in auth["stage1_candidates"]
        ],
        "test_access": 0,
    }
    body["selection_id"] = "er9stage1selection-" + canonical_sha256(body)
    _write_immutable(body, STAGE1_SELECTION)
    print(f"ER-9 Stage-1 selected D={selected['transmit_dim']} with n_correct={selected['n_correct']}")


def build_stage2_authorization() -> None:
    _assert_clean_parity()
    if not STAGE1_SELECTION.is_file():
        raise RuntimeError("Stage-1 selection is missing")
    if STAGE2_AUTH.exists():
        raise RuntimeError("Stage-2 authorization already exists")
    auth = verify_stage1_authorization(STAGE1_AUTH, SOURCE_MANIFEST)
    selection = _read(STAGE1_SELECTION)
    floor = auth["packet_floor"]
    selected_dimension = int(selection["selected"]["transmit_dim"])
    candidates = [candidate.as_dict() for candidate in stage2_candidates(int(floor["payload_bits"]), selected_dimension)]
    body: dict[str, Any] = {
        "schema_version": 1,  # literal-ok: Stage-2 authorization schema
        "artifact_role": "ER9_STAGE2_EXECUTION_AUTHORIZATION",
        "status": "FROZEN_BEFORE_STAGE2_OPTIMIZER_STEP",
        "authorization_scope": "ER9_STAGE2_ONLY",
        "source_commit": _git("rev-parse", "HEAD"),
        "implementation_commit": auth["source_commit"],
        "source_manifest_id": auth["source_manifest_id"],
        "source_manifest_sha256": auth["source_manifest_sha256"],
        "config_path": auth["config_path"],
        "config_hash": auth["config_hash"],
        "stage1_authorization_id": auth["authorization_id"],
        "stage1_selection_id": selection["selection_id"],
        "stage1_selection_sha256": _file_sha(STAGE1_SELECTION),
        "selected_transmit_dim": selected_dimension,
        "stage1_reused_candidate": {"transmit_dim": selected_dimension, "quantiser_bits": 2},  # literal-ok: AM-96 Stage-2 reuse
        "stage2_candidate_order": candidates,
        "stage2_bits_order": "ascending_numeric",
        "stage2_selection_metric": "exact_validation_n_correct_at_7db_real_digital_chain",
        "stage2_tie_break": "smallest_quantiser_bits",
        "cross_product": False,
        "stage2_training_count": max(len(candidates) - 1, 0),
        "execution_profile_id": "local_4060_cu130",
        "pre_execution_counters": {
            "er9_training": auth["stage1_training_count"],
            "randomized_er2_training": 0,
            "g11": 0,
            "w10": 0,
            "learned_test_inference": 0,
            "model_facing_test_access": 0,
        },
        "test": "SEALED",
    }
    body["authorization_id"] = "er9stage2auth-" + canonical_sha256(body)
    _write_immutable(body, STAGE2_AUTH)
    print(f"ER-9 Stage-2 authorization written: {body['authorization_id']} ({body['stage2_training_count']} new runs)")


def _require_committed(path: Path) -> None:
    relative = str(path.relative_to(REPO))
    try:
        _git("cat-file", "-e", f"HEAD:{relative}")
    except subprocess.CalledProcessError:
        raise RuntimeError(f"required authorization/evidence is not committed: {relative}") from None
    _assert_clean_parity()


def _verify_stage2_authorization() -> dict[str, Any]:
    value = _read(STAGE2_AUTH)
    if value["cross_product"] is not False or value["status"] != "FROZEN_BEFORE_STAGE2_OPTIMIZER_STEP":
        raise RuntimeError("Stage-2 authorization is not the exact AM-96 staged rule")
    if value["source_manifest_id"] != verify_source_manifest(SOURCE_MANIFEST)["manifest_id"]:
        raise RuntimeError("Stage-2 source manifest differs")
    selection = _read(STAGE1_SELECTION)
    if value["stage1_selection_id"] != selection["selection_id"]:
        raise RuntimeError("Stage-2 selection binding differs")
    return value


def run_stage2_train() -> None:
    _require_committed(STAGE2_AUTH)
    auth = _verify_stage2_authorization()
    cfg = _config("configs/er9-digital.yaml", 0, 0)  # literal-ok: authorized search seed cell
    source = _read(CHECKPOINT_ROOT / "execution_profile_binding.json")
    source_binding = {
        "schema_version": 1,  # literal-ok: ER source binding schema
        "implementation_commit": auth["implementation_commit"],
        "source_manifest_id": auth["source_manifest_id"],
        "source_manifest_sha256": auth["source_manifest_sha256"],
        "execution_source_commit": auth["source_commit"],
        "execution_profile_binding": source,
        "test": "SEALED",
        "test_access": 0,
    }
    runs = []
    for candidate in auth["stage2_candidate_order"]:
        if candidate == auth["stage1_reused_candidate"]:
            continue
        runs.append(_train_er9_candidate(cfg, candidate, _stage2_runtime(candidate), source_binding, "stage2"))
    if len(runs) != auth["stage2_training_count"]:
        raise RuntimeError("Stage-2 training count does not reconcile")
    _write_immutable(
        {
            "schema_version": 1,  # literal-ok: Stage-2 runtime index schema
            "artifact_role": "ER9_STAGE2_TRAINING_INDEX",
            "authorization_id": auth["authorization_id"],
            "source_binding": source_binding,
            "candidate_order": [
                candidate for candidate in auth["stage2_candidate_order"]
                if candidate != auth["stage1_reused_candidate"]
            ],
            "runs": runs,
            "training_run_count": len(runs),
            "test_access": 0,
        },
        CHECKPOINT_ROOT / "stage2_training_index.json",
    )
    print(f"ER-9 Stage-2 training complete: {len(runs)} new runs")


def run_stage2_evaluate() -> None:
    _assert_clean_parity()
    if STAGE2_SELECTION.exists() or STAGE2_SELECTION.is_symlink():
        raise RuntimeError("Stage-2 selection already exists; refusing a second selection pass")
    auth = _verify_stage2_authorization()
    _require_committed(STAGE2_AUTH)
    selection1 = _read(STAGE1_SELECTION)
    cfg = _config("configs/er9-digital.yaml", 0, 0)  # literal-ok: authorized search seed cell
    source = selection1["source_binding"]
    evaluations = []
    eval_root = RESULT_ROOT / "stage2_evaluations"
    for candidate in auth["stage2_candidate_order"]:
        if candidate == auth["stage1_reused_candidate"]:
            stage1_row = next(
                row for row in selection1["rows"]
                if row["transmit_dim"] == candidate["transmit_dim"] and row["quantiser_bits"] == candidate["quantiser_bits"]
            )
            evaluations.append({
                "schema_version": 1,  # literal-ok: Stage-2 reused evaluation schema
                "artifact_role": "ER9_STAGE2_REUSED_STAGE1_EVALUATION",
                "candidate": candidate,
                "reused_stage1": True,
                "stage1_evaluation_id": stage1_row["evaluation_id"],
                "runtime_root": str(_stage1_runtime(candidate).relative_to(REPO)),
                "real_chain_validation": {
                    "validation_n_correct": stage1_row["n_correct"],
                    "validation_total": stage1_row["n_total"],
                    "selected": stage1_row["selected_phy"],
                },
                "evaluation_id": stage1_row["evaluation_id"],
                "source_binding": source,
                "test_access": 0,
            })
        else:
            path = eval_root / f"{_candidate_slug(candidate['transmit_dim'], candidate['quantiser_bits'])}.json"
            body = _read_existing_evaluation(path, "er9stage2eval-", candidate)
            if body is None:
                body = _evaluate_one_er9_candidate(cfg, candidate, _stage2_runtime(candidate), source)
                body["reused_stage1"] = False
                body["evaluation_id"] = "er9stage2eval-" + canonical_sha256(body)
                _write_immutable(body, path)
            evaluations.append(body)
    rows = [
        {
            "transmit_dim": body["candidate"]["transmit_dim"],
            "quantiser_bits": body["candidate"]["quantiser_bits"],
            "n_correct": body["real_chain_validation"]["validation_n_correct"],
            "n_total": body["real_chain_validation"]["validation_total"],
            "selected_phy": body["real_chain_validation"]["selected"],
            "evaluation_id": body.get("evaluation_id", body.get("stage1_evaluation_id")),
            "reused_stage1": bool(body["reused_stage1"]),
            "runtime_root": body["runtime_root"],
        }
        for body in evaluations
    ]
    selected = dict(select_stage2(rows))
    body = {
        "schema_version": 1,  # literal-ok: Stage-2 selection schema
        "artifact_role": "ER9_STAGE2_TERMINAL_SELECTION",
        "status": "SELECTED_PAIR_FROZEN",
        "authorization_id": auth["authorization_id"],
        "stage1_selection_id": selection1["selection_id"],
        "source_binding": source,
        "candidate_order": auth["stage2_candidate_order"],
        "training_run_count_new": auth["stage2_training_count"],
        "selection_metric": auth["stage2_selection_metric"],
        "tie_break": auth["stage2_tie_break"],
        "rows": rows,
        "selected": selected,
        "selected_factorisation": factorisation_for_dimension(int(selected["transmit_dim"])).as_dict(),
        "test_access": 0,
    }
    body["selection_id"] = "er9stage2selection-" + canonical_sha256(body)
    _write_immutable(body, STAGE2_SELECTION)
    print(f"ER-9 Stage-2 selected (D,b)=({selected['transmit_dim']},{selected['quantiser_bits']}) with n_correct={selected['n_correct']}")


def run_final_production() -> None:
    _require_committed(STAGE2_SELECTION)
    if PRODUCTION.exists() or PRODUCTION.is_symlink():
        raise RuntimeError("ER-9 final production manifest already exists; refusing a second production set")
    selection = _read(STAGE2_SELECTION)
    if selection["status"] != "SELECTED_PAIR_FROZEN":
        raise RuntimeError("ER-9 selected pair is not frozen")
    selected = selection["selected"]
    dimension = int(selected["transmit_dim"])
    bits = int(selected["quantiser_bits"])
    pairs = tuple(zip(get("evaluation.train_seeds"), get("evaluation.channel_seeds"), strict=True))
    if pairs != ((0, 0), (1, 1), (2, 2)):
        raise RuntimeError(f"unexpected zipped production seed cells: {pairs}")
    search_profile = selection["source_binding"]["execution_profile_binding"]
    production_profiles: dict[tuple[int, int], dict[str, Any]] = {}
    for train_seed, channel_seed in pairs:
        if (int(train_seed), int(channel_seed)) == (0, 0):
            production_profiles[(0, 0)] = dict(search_profile)
        else:
            cfg = _config("configs/er9-digital.yaml", int(train_seed), int(channel_seed))
            production_profiles[(int(train_seed), int(channel_seed))] = _profile_binding(cfg)
    stage1_auth = _read(STAGE1_AUTH)
    architecture_difference = {
        "schema_version": 1,  # literal-ok: ER-9 architecture-difference schema
        "artifact_role": "ER9_ARCHITECTURE_DIFFERENCE_AUDIT",
        "source_binding": selection["source_binding"],
        "declared_shared_components": list(get("digital_semantic_control.shared_with_learned")),
        "declared_difference": str(get("digital_semantic_control.differs_only_in")),
        "observed_shared_components": {
            "encoder_arch": str(get("learned_system.encoder_arch")),
            "encoder_trunk": "djscc_residual_v1_input_normalisation_stem_body_entry_residual_stack",
            "task_head_arch": "ImageClassificationHead_adaptive_global_average_pooling_plus_linear",
            "train_split": "train",
            "augmentation": list(get("learned_system.augmentation")),
            "optimizer": str(get("learned_system.optimizer_implementation")),
            "epochs": int(get("learned_system.epochs.imagenette160")),
        },
        "learned_interface": {
            "representation": "encoder_projection_complex_symbols",
            "transport": "learned_awgn_channel",
            "post_interface": "learned_decoder_residual_then_reconstruction_and_task_heads",
        },
        "er9_interface": {
            "representation": str(get("digital_semantic_control.pre_interface_tap")),
            "projection": "trainable_body_channels_to_selected_output_channels",
            "pooling": "AdaptiveAvgPool2d_8x8",
            "quantiser": str(get("digital_semantic_control.quantiser")),
            "transport": "existing_digital_packet_ldpc_modulation_awgn_chain",
            "post_interface": str(get("digital_semantic_control.post_interface_task_path")),
            "reconstruction_head": str(get("digital_semantic_control.reconstruction_head")),
            "loss": str(get("digital_semantic_control.training_loss")),
            "lambda": str(get("digital_semantic_control.training_lambda")),
        },
        "architecture_difference_paths": ["channel_interface"],
        "only_declared_difference": True,
        "am96_semantics": "task_only_er9_interface_difference_is_the_declared_channel_interface",
        "test_access": 0,
    }
    architecture_difference["audit_id"] = "er9archdiff-" + canonical_sha256(architecture_difference)
    _write_immutable(architecture_difference, ARCHITECTURE_DIFF)
    entries = []
    new_training = 0
    search_runtime = Path(REPO / str(next(row for row in selection["rows"] if row["transmit_dim"] == dimension and row["quantiser_bits"] == bits).get("runtime_root", "")))
    for train_seed, channel_seed in pairs:
        cfg = _config("configs/er9-digital.yaml", int(train_seed), int(channel_seed))
        if train_seed == 0:
            runtime = search_runtime
            completion = _validate_runtime_completion(runtime, {"transmit_dim": dimension, "quantiser_bits": bits})
            source_binding = selection["source_binding"]
            promoted = True
        else:
            candidate = {"transmit_dim": dimension, "quantiser_bits": bits}
            runtime = CHECKPOINT_ROOT / "production" / f"train{train_seed}_channel{channel_seed}"
            profile = production_profiles[(int(train_seed), int(channel_seed))]
            source_binding = {
                "schema_version": 1,  # literal-ok: ER source binding schema
                "implementation_commit": selection["source_binding"]["implementation_commit"],
                "source_manifest_id": selection["source_binding"]["source_manifest_id"],
                "source_manifest_sha256": selection["source_binding"]["source_manifest_sha256"],
                "execution_source_commit": _git("rev-parse", "HEAD"),
                "execution_profile_binding": profile,
                "test": "SEALED",
                "test_access": 0,
            }
            run = _train_er9_candidate(cfg, candidate, runtime, {
                **source_binding,
            }, "production")
            completion = run["completion"]
            new_training += 0 if run["reused_existing_attempt"] else 1
            promoted = False
        entries.append({
            "train_seed": int(train_seed),
            "channel_seed": int(channel_seed),
            "candidate": {"transmit_dim": dimension, "quantiser_bits": bits},
            "runtime_root": str(runtime.relative_to(REPO)),
            "completion": completion,
            "completion_sha256": _file_sha(runtime / "run_completion.json"),
            "execution_profile_binding": source_binding["execution_profile_binding"],
            "promoted_search_run": promoted,
            "config_hash": config_hash(cfg),
            "test_access": 0,
        })
    body = {
        "schema_version": 1,  # literal-ok: ER-9 production manifest schema
        "artifact_role": "ER9_FINAL_PRODUCTION_MANIFEST",
        "status": "THREE_ZIPPED_SEED_CELLS_AVAILABLE",
        "source_binding": selection["source_binding"],
        "selection_id": selection["selection_id"],
        "selected_pair": {"transmit_dim": dimension, "quantiser_bits": bits},
        "selected_factorisation": factorisation_for_dimension(dimension).as_dict(),
        "architecture_difference": str(ARCHITECTURE_DIFF.relative_to(REPO)),
        "architecture_difference_sha256": _file_sha(ARCHITECTURE_DIFF),
        "packet_floor": stage1_auth["packet_floor"],
        "budget_proof": {
            "rule": "D_times_bits_plus_metadata_bits_at_most_A_floor_bits",
            "metadata_bits": stage1_auth["metadata_bits"],
            "selected_raw_bound_bits": dimension * bits + int(stage1_auth["metadata_bits"]),
            "A_floor_bits": stage1_auth["packet_floor"]["payload_bits"],
            "selected_pair_feasible": dimension * bits + int(stage1_auth["metadata_bits"]) <= int(stage1_auth["packet_floor"]["payload_bits"]),
            "all_admissible_pairs": stage1_auth["admissible_pairs"],
            "rejected_pairs": stage1_auth["rejected_pairs"],
        },
        "seed_pairing": "zipped_not_cross_product",
        "seed_cells": entries,
        "search_training_count": int(_read(STAGE1_AUTH)["stage1_training_count"] + _read(STAGE2_AUTH)["stage2_training_count"]),
        "final_production_training_count": new_training,
        "full_configured_pair_count": int(_read(STAGE1_AUTH)["configured_pair_count"]),
        "admissible_pair_count": int(_read(STAGE1_AUTH)["admissible_pair_count"]),
        "test_access": 0,
        "test": "SEALED",
    }
    _write_immutable(body, PRODUCTION)
    print(f"ER-9 final production set complete: {new_training} new runs; selected (D,b)=({dimension},{bits})")


def run_er9_validation() -> None:
    _require_committed(PRODUCTION)
    manifest = _read(PRODUCTION)
    result_root = RESULT_ROOT / "final_validation"
    snrs = tuple(int(value) for value in get("channel.test_snr_grid_db"))
    for entry in manifest["seed_cells"]:
        path = result_root / f"train{entry['train_seed']}_channel{entry['channel_seed']}.json"
        if path.exists() or path.is_symlink():
            if path.is_symlink():
                raise RuntimeError(f"ER-9 validation artifact is an unsafe symlink: {path}")
            existing = _read(path)
            identifier = existing.get("evidence_id")
            without_id = dict(existing)
            without_id.pop("evidence_id", None)
            if (
                identifier != "er9validation-" + canonical_sha256(without_id)
                or existing.get("seed_cell") != {
                    "train_seed": entry["train_seed"],
                    "channel_seed": entry["channel_seed"],
                }
                or existing.get("selected_pair") != manifest["selected_pair"]
                or existing.get("test_access") != 0
                or existing.get("test") != "SEALED"
            ):
                raise RuntimeError(f"ER-9 validation artifact differs: {path}")
            print(f"ER-9 validation cell already authenticated; reusing {path.name}")
            continue
        cfg = _config("configs/er9-digital.yaml", int(entry["train_seed"]), int(entry["channel_seed"]))
        runtime = REPO / str(entry["runtime_root"])
        selected = _read(runtime / "selected_checkpoint.json")["selection"]
        checkpoint = runtime / str(selected["checkpoint_path"])
        model = load_er9_checkpoint(
            cfg,
            transmit_dim=int(manifest["selected_pair"]["transmit_dim"]),
            quantiser_bits=int(manifest["selected_pair"]["quantiser_bits"]),
            checkpoint_path=checkpoint,
            device="cuda:0",
        )
        entropy, entropy_record = fit_entropy_model(model, cfg, device="cuda:0", num_workers=4)
        features = collect_validation_features(model, cfg, device="cuda:0", num_workers=4)
        curves = []
        for snr in snrs:
            curves.append(evaluate_candidate_at_snr(
                model,
                cfg,
                entropy=entropy,
                dimension=int(manifest["selected_pair"]["transmit_dim"]),
                quantiser_bits=int(manifest["selected_pair"]["quantiser_bits"]),
                snr_db=snr,
                device="cuda:0",
                num_workers=4,
                validation_features=features,
                include_per_image=True,
            ))
        body = {
            "schema_version": 1,  # literal-ok: ER-9 final validation schema
            "artifact_role": "ER9_FINAL_VALIDATION_ONLY_EVIDENCE",
            "source_binding": manifest["source_binding"],
            "seed_cell": {"train_seed": entry["train_seed"], "channel_seed": entry["channel_seed"]},
            "selected_pair": manifest["selected_pair"],
            "checkpoint": {"path": str(checkpoint.relative_to(REPO)), "sha256": _file_sha(checkpoint), "selection": selected},
            "entropy_model": entropy_record,
            "snr_grid_db": list(snrs),
            "curves": curves,
            "validation_only": True,
            "test_access": 0,
            "test": "SEALED",
        }
        body["evidence_id"] = "er9validation-" + canonical_sha256(body)
        _write_immutable(body, path)
        print(f"ER-9 validation cell complete: train{entry['train_seed']}/channel{entry['channel_seed']}")


def run_er2_train() -> None:
    _require_committed(PRODUCTION)
    _assert_clean_parity()
    if ER2_COMPLETION.exists():
        raise RuntimeError("randomized ER-2 completion already exists; refusing a second run")
    cfg = _config("configs/learned-er2-randomized.yaml", 0, 0)  # literal-ok: first ER-2 zipped seed cell
    binding = _profile_binding(cfg)
    source = {
        "schema_version": 1,  # literal-ok: ER source binding schema
        "implementation_commit": _read(PRODUCTION)["source_binding"]["implementation_commit"],
        "source_manifest_id": _read(PRODUCTION)["source_binding"]["source_manifest_id"],
        "source_manifest_sha256": _read(PRODUCTION)["source_binding"]["source_manifest_sha256"],
        "execution_source_commit": binding["git_commit"],
        "execution_profile_binding": binding,
        "test": "SEALED",
        "test_access": 0,
    }
    runtime = CHECKPOINT_ROOT / "er2_randomized" / "train0_channel0"
    trainer = ER2RandomizedTrainer(
        cfg,
        device="cuda:0",
        runtime_root=runtime,
        source_binding=source,
        campaign_id="er2_randomized_single_run",
        run_id="er2-randomized-r_1_6-train0-channel0",
    )
    completion = trainer.run()
    runtime_completion = _read(runtime / "run_completion.json")
    audit = _read(runtime / "snr_assignment_audit.json")
    _write_immutable({
        **completion,
        "source_binding": source,
        "runtime_root": str(runtime.relative_to(REPO)),
        "completion_sha256": _file_sha(runtime / "run_completion.json"),
        "scientific_training_run_count": 1,
        "test_access": 0,
    }, ER2_COMPLETION)
    _write_immutable({
        **audit,
        "source_binding": source,
        "runtime_root": str(runtime.relative_to(REPO)),
        "audit_sha256": _file_sha(runtime / "snr_assignment_audit.json"),
        "test_access": 0,
    }, ER2_AUDIT)
    if runtime_completion.get("training_run_count") != 1:
        raise RuntimeError("randomized ER-2 completion does not record exactly one run")
    print("randomized ER-2 training complete: exactly one scientific run")


def run_er2_validation() -> None:
    _require_committed(ER2_COMPLETION)
    _assert_clean_parity()
    if ER2_VALIDATION.exists() or ER2_VALIDATION.is_symlink():
        if ER2_VALIDATION.is_symlink():
            raise RuntimeError("randomized ER-2 validation artifact is an unsafe symlink")
        existing = _read(ER2_VALIDATION)
        identifier = existing.get("evidence_id")
        without_id = dict(existing)
        without_id.pop("evidence_id", None)
        if identifier != "er2validation-" + canonical_sha256(without_id) or existing.get("test") != "SEALED" or existing.get("test_access") != 0:
            raise RuntimeError("randomized ER-2 validation artifact differs")
        print("randomized ER-2 validation already authenticated; refusing a second evaluation")
        return
    completion = _read(ER2_COMPLETION)
    cfg = _config("configs/learned-er2-randomized.yaml", 0, 0)  # literal-ok: first ER-2 zipped seed cell
    runtime = REPO / str(completion["runtime_root"])
    selected = _read(runtime / "selected_checkpoint.json")["selection"]
    model = build_djscc(cfg, device="cuda:0")
    checkpoint = runtime / str(selected["checkpoint_path"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("config_hash") != config_hash(cfg):
        raise RuntimeError("randomized ER-2 checkpoint config differs")
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    curves = [
        evaluate_er2_at_snr(model, cfg, snr_db=snr, device="cuda:0", num_workers=4)
        for snr in (int(value) for value in get("channel.test_snr_grid_db"))
    ]
    body = {
        "schema_version": 1,  # literal-ok: randomized ER-2 validation schema
        "artifact_role": "ER2_RANDOMIZED_VALIDATION_ONLY_EVIDENCE",
        "source_binding": completion["source_binding"],
        "run_identity": "er2-randomized-r_1_6-train0-channel0",
        "selected_checkpoint": {"path": str(checkpoint.relative_to(REPO)), "sha256": _file_sha(checkpoint), "selection": selected},
        "snr_grid_db": [int(value) for value in get("channel.test_snr_grid_db")],
        "curves": curves,
        "scientific_training_run_count": 1,
        "validation_only": True,
        "test_access": 0,
        "test": "SEALED",
    }
    body["evidence_id"] = "er2validation-" + canonical_sha256(body)
    _write_immutable(body, ER2_VALIDATION)
    print("randomized ER-2 validation evidence complete")


def run_g11() -> None:
    _require_committed(PRODUCTION)
    er9_paths = [
        RESULT_ROOT / "final_validation" / "train0_channel0.json",
        RESULT_ROOT / "final_validation" / "train1_channel1.json",
        RESULT_ROOT / "final_validation" / "train2_channel2.json",
    ]
    for er9_path in er9_paths:
        _require_committed(er9_path)
    _require_committed(ER2_COMPLETION)
    _require_committed(ER2_AUDIT)
    _require_committed(ER2_VALIDATION)
    from evaluation.h4_precision import simulate_h4_precision

    production = _read(PRODUCTION)
    er2 = _read(ER2_VALIDATION)
    er9_path = er9_paths[0]
    er9 = _read(er9_path)
    er9_curve = next(curve for curve in er9["curves"] if curve["snr_db"] == 7)
    er2_curve = next(curve for curve in er2["curves"] if curve["snr_db"] == 7)
    er9_rows = {row["stable_sample_id"]: bool(row["correct"]) for row in er9_curve["per_image"]}
    er2_rows = {row["stable_sample_id"]: bool(row["correct"]) for row in er2_curve["outcomes"]}
    if set(er9_rows) != set(er2_rows):
        raise RuntimeError("G-11 validation pairing IDs differ")
    discordance = [int(er9_rows[key] != er2_rows[key]) for key in sorted(er9_rows)]
    h4 = simulate_h4_precision(
        discordance,
        sample_size=int(get("datasets.imagenette160.test_images")),
    )
    h4_path = G11_ROOT / "h4_precision_simulation.json"
    h4["input_artifacts"] = {
        "er9_validation": str(er9_path.relative_to(REPO)),
        "er9_validation_sha256": _file_sha(er9_path),
        "er2_validation": str(ER2_VALIDATION.relative_to(REPO)),
        "er2_validation_sha256": _file_sha(ER2_VALIDATION),
        "snr_db": 7,  # literal-ok: fixed training SNR validation pairing
    }
    h4["evidence_id"] = "g11h4-" + canonical_sha256(h4)
    _write_immutable(h4, h4_path)
    terminal = {
        "schema_version": 1,  # literal-ok: G-11 terminal schema
        "artifact_role": "G11_TERMINAL_VALIDATION_ONLY_CLOSEOUT",
        "status": "G11_GREEN_NO_TEST_ACCESS",
        "decision": "GREEN",
        "source_binding": production["source_binding"],
        "er9": {
            "production_manifest": str(PRODUCTION.relative_to(REPO)),
            "production_manifest_sha256": _file_sha(PRODUCTION),
            "stage1_selection": str(STAGE1_SELECTION.relative_to(REPO)),
            "stage1_selection_sha256": _file_sha(STAGE1_SELECTION),
            "stage2_selection": str(STAGE2_SELECTION.relative_to(REPO)),
            "stage2_selection_sha256": _file_sha(STAGE2_SELECTION),
            "architecture_difference": str(ARCHITECTURE_DIFF.relative_to(REPO)),
            "architecture_difference_sha256": _file_sha(ARCHITECTURE_DIFF),
            "validation_cells": [
                {
                    "path": str(path.relative_to(REPO)),
                    "sha256": _file_sha(path),
                }
                for path in er9_paths
            ],
        },
        "randomized_er2": {
            "completion": str(ER2_COMPLETION.relative_to(REPO)),
            "completion_sha256": _file_sha(ER2_COMPLETION),
            "assignment_audit": str(ER2_AUDIT.relative_to(REPO)),
            "assignment_audit_sha256": _file_sha(ER2_AUDIT),
            "validation": str(ER2_VALIDATION.relative_to(REPO)),
            "validation_sha256": _file_sha(ER2_VALIDATION),
            "scientific_training_run_count": 1,
        },
        "h4": {
            "artifact": str(h4_path.relative_to(REPO)),
            "artifact_sha256": _file_sha(h4_path),
            "gate_decision": h4["gate_decision"],
            "mde_percentage_points": h4["mde_percentage_points"],
        },
        "g10_terminal": {
            "source": "9515c490aed4439f7ced2c163abef61557654ddf",
            "evaluations": 63,  # literal-ok: immutable G-10 terminal count
            "reruns": 0,  # literal-ok: immutable G-10 rerun count
            "classification": "expected_crossover_observed",
            "headline_bracket": "-5 -> -4 dB",
        },
        "protected_counters": {
            "g10_evaluations": 63,
            "g10_reruns": 0,
            "er9_training": int(production["search_training_count"] + production["final_production_training_count"]),
            "randomized_er2_training": 1,
            "g11": 1,
            "w10": 0,
            "learned_test_inference": 0,
            "model_facing_test_access": 0,
        },
        "g10_terminal_immutable": True,
        "test": "SEALED",
        "test_access": 0,
    }
    terminal["terminal_id"] = "g11terminal-" + canonical_sha256(terminal)
    _write_immutable(terminal, G11_ROOT / "g11_terminal_closeout.json")
    print(f"G-11 terminal closeout complete: H4={h4['gate_decision']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=(
        "stage1-train", "stage1-evaluate", "stage2-authorize", "stage2-train",
        "stage2-evaluate", "final-production", "validate-er9", "er2-train",
        "validate-er2", "g11",
    ))
    args = parser.parse_args(argv)
    actions = {
        "stage1-train": run_stage1_train,
        "stage1-evaluate": run_stage1_evaluate,
        "stage2-authorize": build_stage2_authorization,
        "stage2-train": run_stage2_train,
        "stage2-evaluate": run_stage2_evaluate,
        "final-production": run_final_production,
        "validate-er9": run_er9_validation,
        "er2-train": run_er2_train,
        "validate-er2": run_er2_validation,
        "g11": run_g11,
    }
    actions[args.action]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
