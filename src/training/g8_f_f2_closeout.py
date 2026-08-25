"""Read-only G8_F/F2 closeout and artifact-classifier freeze (BR-12, AM-89)."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from config.params import REPO_ROOT, get
from data.classifier import epoch_permutation
from models.frozen_reference_classifier import load_frozen_reference_classifier
from training.g8_f_f2 import (
    EXPECTED_ASSIGNMENTS,
    EXPECTED_BATCH_SIZE,
    EXPECTED_EPOCHS,
    EXPECTED_MATERIALIZED,
    EXPECTED_OMISSIONS,
    EXPECTED_OPTIMIZER_STEPS,
    EXPECTED_STEPS_PER_EPOCH,
    EXPECTED_UNIQUE_RECONSTRUCTIONS,
    F1_COMPLETION_ID,
    F1_COMPLETION_SHA256,
    F1_CORPUS_ID,
    F1_MANIFEST_SHA256,
    F2_SCHEMA_VERSION,
    F2_SCOPE,
    F2_VARIANT,
    G1_ADJUDICATION_ID,
    G1_ADJUDICATION_SHA256,
    G1_CHECKPOINT_BYTES,
    G1_CHECKPOINT_ID,
    G1_CHECKPOINT_SHA256,
    F2ArtifactDataset,
    canonical_json,
    f2_recipe,
    f2_recipe_sha256,
    learning_rate_for_epoch,
    sha256_bytes,
)
from training.g8_f_f2_authorization import AUTHORIZATION_PATH, verify_authorization

COMPLETION_PATH = REPO_ROOT / "results/baseline/g8_f/f2_completion.json"
FREEZE_PATH = REPO_ROOT / "results/baseline/g8_f/artifact_classifier_freeze.json"
MONITOR_PATH = REPO_ROOT / "results/baseline/g8_f/f2_monitor_closeout.json"
COMPLETION_PREFIX = "g8ff2completion-"
FREEZE_PREFIX = "g8fclassifierfreeze-"
MONITOR_PREFIX = "g8fmonitorcloseout-"
BEST_RELEASE = {
    "provider": "github_release",
    "repository": "nxck2005/capstone",
    "release_tag": "g8-f-f2-artifact-classifier-2026-08-25",
    "asset_name": "artifact-finetuned-imagenette160-epoch17-468710ba5e6426d2daeaba50af331b498d5d079726476538d69e2fd3b6355ca1.pt",
    "bytes": 89_555_403,
    "sha256": "468710ba5e6426d2daeaba50af331b498d5d079726476538d69e2fd3b6355ca1",
}
BEST_EPOCH = 17  # literal-ok: observed preregistered F2 best epoch, zero-based
BEST_TOP1 = 0.89  # literal-ok: observed count-derived F2 validation top-1
LAUNCH_HEAD = "3b6068891772ba016448a4c978cfdd8de56bbbeb"
AUTHORIZATION_SHA256 = "fbe1252cc2b20cbd46007f9e534468b697abca260c1f74bfee0ea904af14cb3a"
LOGICAL_ROWS_SHA256 = "6f089ac46c62290b2ac709e5419ba2976ba23eefe74e43e63b7af2703878defd"
ASSIGNMENT_IDS_SHA256 = "6c149d26c03ca1655dc3ec3e4d467e9e96941382b0791b93f925516d606b3977"
F1_ORDINALS_SHA256 = "278b63d8e551bd102e2dd16f42226c16f1d97a2317eb3197f1a13b1ac07a3275"
TRAINING_HISTORY_SHA256 = "4d246fa83f8cfae86f4aebe4e425df61da8127116faf4b7ce5570bbda909566f"
VALIDATION_HISTORY_SHA256 = "ee5ad0861456aaf0a1d66752cf858bfd0fefcdd9b0b5047ac8eb57920cc11b00"
CHECKPOINT_MANIFEST_SHA256 = "0508d311eadb50f62ddb1b991af081576e3bea165e9ae8e2251935bf396d5b96"
CLASS_MAPPING = {
    "n01440764": 0, "n02102040": 1, "n02979186": 2, "n03000684": 3,  # literal-ok: frozen Imagenette class mapping
    "n03028079": 4, "n03394916": 5, "n03417042": 6, "n03425413": 7,  # literal-ok: frozen Imagenette class mapping
    "n03445777": 8, "n03888257": 9,  # literal-ok: frozen Imagenette class mapping
}
VALIDATION_IMAGES = int(get("datasets.imagenette160.val_images"))
CLASS_COUNT = int(get("datasets.imagenette160.classes"))
CANONICAL_AXIS = int(get("datasets.imagenette160.image_size")[0])
IO_CHUNK_BYTES = 1024 * 1024  # literal-ok: operational one-MiB streaming chunk
CHECKPOINT_FIELDS = {
    "schema_version", "artifact_role", "lineage", "completed_epoch", "next_epoch",
    "total_optimizer_steps", "model_state", "optimizer_state", "scheduler_state",
    "training_history", "validation_history", "checkpoint_history", "best_epoch",
    "best_validation_top1", "expected_epochs", "expected_optimizer_steps", "protected_counters",
}
PROTECTED = {
    "f3_cached_sweep_rescoring": 0,
    "pass_two": 0,
    "pass_three": 0,
    "fallback": 0,
    "ratio_adjudication": 0,
    "learned_training": 0,
    "test_access": 0,
}


class F2CloseoutHold(RuntimeError):
    """The stopped worker or its compact completion evidence did not authenticate."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise F2CloseoutHold(message)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise F2CloseoutHold(f"cannot read {path}: {exc}") from None
    _require(isinstance(value, dict), f"{path} must contain an object")
    return value


def rendered_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


def identified(value: Mapping[str, Any], *, field: str, prefix: str) -> dict[str, Any]:
    body = dict(value)
    body.pop(field, None)
    body[field] = prefix + sha256_bytes(canonical_json(body))
    return body


def _verify_id(value: Mapping[str, Any], *, field: str, prefix: str) -> None:
    identifier = value.get(field)
    _require(isinstance(identifier, str) and identifier.startswith(prefix), f"{field} differs")
    body = dict(value)
    body.pop(field)
    _require(identifier == prefix + sha256_bytes(canonical_json(body)), f"{field} content identity differs")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(IO_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lineage(authorization: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scope": F2_SCOPE,
        "authorization_id": authorization["authorization_id"],
        "authorization_sha256": AUTHORIZATION_SHA256,
        "source_commit": authorization["source_commit"],
        "execution_profile_id": authorization["execution_profile"]["execution_profile_id"],
        "device": authorization["execution_profile"]["device"],
        "f1_completion_id": F1_COMPLETION_ID,
        "f1_completion_sha256": F1_COMPLETION_SHA256,
        "f1_corpus_id": F1_CORPUS_ID,
        "f1_manifest_sha256": F1_MANIFEST_SHA256,
        "g1_parent_checkpoint_id": G1_CHECKPOINT_ID,
        "g1_parent_checkpoint_sha256": G1_CHECKPOINT_SHA256,
        "recipe_sha256": f2_recipe_sha256(),
        "train_seed": 0,
        "classifier_variant": F2_VARIANT,
    }


def select_best(validation: Sequence[Mapping[str, Any]]) -> tuple[int, float]:
    _require(len(validation) == EXPECTED_EPOCHS, "F2 validation history length differs")
    best_epoch = max(range(len(validation)), key=lambda epoch: float(validation[epoch]["top1_accuracy"]))
    return best_epoch, float(validation[best_epoch]["top1_accuracy"])


def _validate_epoch_record(epoch: int, training: Mapping[str, Any], validation: Mapping[str, Any]) -> None:
    _require(set(training) == {"epoch", "lr", "loss", "examples", "steps", "duration_seconds", "sample_order_sha256"}, "F2 training record schema differs")
    _require(training["epoch"] == epoch and training["examples"] == EXPECTED_MATERIALIZED and training["steps"] == EXPECTED_STEPS_PER_EPOCH, "F2 training epoch arithmetic differs")
    _require(training["lr"] == learning_rate_for_epoch(epoch), "F2 learning-rate history differs")
    _require(isinstance(training["loss"], (int, float)) and math.isfinite(float(training["loss"])) and float(training["loss"]) >= 0, "F2 training loss is invalid")
    _require(isinstance(training["duration_seconds"], (int, float)) and float(training["duration_seconds"]) > 0, "F2 epoch duration is invalid")
    order = epoch_permutation(EXPECTED_MATERIALIZED, 0, epoch)
    _require(len(order) == EXPECTED_MATERIALIZED and len(set(order)) == EXPECTED_MATERIALIZED and set(order) == set(range(EXPECTED_MATERIALIZED)), "F2 sample order is not an exact permutation")
    _require(training["sample_order_sha256"] == sha256_bytes(canonical_json(order)), "F2 sample-order digest differs")
    _require(set(validation) == {"epoch", "n_correct", "n_total", "top1_accuracy", "role"}, "F2 validation record schema differs")
    _require(validation["epoch"] == epoch and validation["n_total"] == VALIDATION_IMAGES and 0 <= validation["n_correct"] <= VALIDATION_IMAGES, "F2 validation counts differ")
    _require(validation["top1_accuracy"] == validation["n_correct"] / validation["n_total"], "F2 validation top-1 is not count-derived")
    _require(validation["role"] == "f2_checkpoint_selection_validation_not_f3_cached_sweep", "F2 validation role differs")


def _state_shape_digest(state: Mapping[str, Any]) -> str:
    rows = []
    for name, tensor in sorted(state.items()):
        _require(isinstance(name, str) and isinstance(tensor, torch.Tensor), "F2 model state is invalid")
        rows.append({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)})
    return sha256_bytes(canonical_json(rows))


def _optimizer_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    _require(set(value) == {"state", "param_groups"} and isinstance(value["state"], Mapping), "F2 optimizer state schema differs")
    groups = value["param_groups"]
    _require(isinstance(groups, list) and len(groups) == 1, "F2 optimizer param groups differ")
    group = groups[0]
    required = {"lr", "momentum", "dampening", "weight_decay", "nesterov", "maximize", "foreach", "differentiable", "fused", "params"}
    _require(set(group) == required, "F2 optimizer group schema differs")
    recipe = f2_recipe()
    _require(group["momentum"] == recipe["momentum"] and group["dampening"] == 0 and group["weight_decay"] == recipe["weight_decay"] and group["nesterov"] is recipe["nesterov"] and group["maximize"] is False and group["differentiable"] is False, "F2 optimizer recipe differs")
    momentum = []
    for key in sorted(value["state"]):
        item = value["state"][key]
        _require(isinstance(item, Mapping) and set(item) == {"momentum_buffer"} and isinstance(item["momentum_buffer"], torch.Tensor), "F2 SGD momentum state differs")
        tensor = item["momentum_buffer"]
        momentum.append({"parameter": int(key), "shape": list(tensor.shape), "dtype": str(tensor.dtype)})
    return {
        "implementation": "torch.optim.SGD",
        "state_entries": len(value["state"]),
        "parameter_references": len(group["params"]),
        "momentum_shape_digest": sha256_bytes(canonical_json(momentum)),
    }


def audit_runtime(*, runtime_root: Path, f1_runtime: Path, ops_root: Path, authorization_path: Path = AUTHORIZATION_PATH, authenticate_objects: bool = True) -> dict[str, Any]:
    authorization = verify_authorization(authorization_path)
    _require(_sha(authorization_path) == AUTHORIZATION_SHA256, "F2 authorization file SHA-256 differs")
    launch = _json(runtime_root / "launch.json")
    progress = _json(runtime_root / "progress.json")
    latest = _json(runtime_root / "latest.json")
    _require(launch.get("status") == "AUTHENTICATED_BEFORE_OPTIMIZER" and launch.get("mode") == "start", "F2 launch/resume lineage differs")
    _require(launch.get("launch_head") == LAUNCH_HEAD and launch.get("source_commit") == authorization["source_commit"], "F2 launch source lineage differs")
    _require(launch.get("authorization_id") == authorization["authorization_id"] and launch.get("authorization_sha256") == AUTHORIZATION_SHA256, "F2 launch authorization differs")
    _require(launch.get("profile") == {**launch["profile"], "git_commit": LAUNCH_HEAD}, "F2 launch profile is invalid")
    profile = launch["profile"]
    _require(profile.get("execution_profile_id") == "confessor_pascal_cu126" and profile.get("gpu_uuid") == authorization["execution_profile"]["gpu_uuid"] and profile.get("gpu_index") == 0 and profile.get("git_dirty") is False, "F2 execution profile differs")
    _require(profile.get("lock_file_sha256") == authorization["execution_profile"]["lock_file_sha256"], "F2 environment lock differs")
    dataset = F2ArtifactDataset.production(epoch=0, runtime_root=f1_runtime, authenticate_objects=authenticate_objects)
    logical_rows = [asdict(dataset.trace(index)) for index in range(len(dataset))]
    logical_sha = sha256_bytes(canonical_json(logical_rows))
    assignment_sha = sha256_bytes(canonical_json([row["assignment_id"] for row in logical_rows]))
    ordinals_sha = sha256_bytes(canonical_json([row["f1_ordinal"] for row in logical_rows]))
    _require(logical_sha == LOGICAL_ROWS_SHA256 and assignment_sha == ASSIGNMENT_IDS_SHA256 and ordinals_sha == F1_ORDINALS_SHA256, "F2 logical training-row identity differs")
    expected_summary = {
        "assignment_rows": EXPECTED_ASSIGNMENTS, "materialized_rows": EXPECTED_MATERIALIZED,
        "omitted_rows": EXPECTED_OMISSIONS, "unexpected_rows": 0,
        "distinct_materialized_assignments": EXPECTED_MATERIALIZED,
        "unique_reconstruction_sha256": EXPECTED_UNIQUE_RECONSTRUCTIONS,
        "validation_ids": 0, "test_ids": 0,
    }
    _require(asdict(dataset.summary) == expected_summary == launch.get("dataset_summary"), "F2 dataset summary differs")

    paths = sorted((runtime_root / "checkpoints").glob("epoch-*.pt"))
    _require([path.name for path in paths] == [f"epoch-{epoch:02d}.pt" for epoch in range(EXPECTED_EPOCHS)], "F2 checkpoint cadence/prefix differs")
    before = [(path.stat().st_size, path.stat().st_mtime_ns) for path in paths]
    checkpoint_rows: list[dict[str, Any]] = []
    prior_training: list[dict[str, Any]] = []
    prior_validation: list[dict[str, Any]] = []
    final_optimizer: dict[str, Any] | None = None
    model_shape_digest: str | None = None
    for epoch, path in enumerate(paths):
        digest = _sha(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _require(isinstance(payload, Mapping) and set(payload) == CHECKPOINT_FIELDS, "F2 checkpoint schema differs")
        _require(payload["schema_version"] == F2_SCHEMA_VERSION and payload["artifact_role"] == "g8_f_f2_epoch_checkpoint" and payload["lineage"] == _lineage(authorization), "F2 checkpoint lineage differs")
        _require(payload["completed_epoch"] == epoch and payload["next_epoch"] == epoch + 1 and payload["total_optimizer_steps"] == (epoch + 1) * EXPECTED_STEPS_PER_EPOCH, "F2 checkpoint step/epoch arithmetic differs")
        _require(payload["scheduler_state"] == {"completed_epoch": epoch} and payload["expected_epochs"] == EXPECTED_EPOCHS and payload["expected_optimizer_steps"] == EXPECTED_OPTIMIZER_STEPS, "F2 scheduler/expected arithmetic differs")
        _require(payload["protected_counters"] == {"f3_cached_sweep_inference": 0, "pass_two": 0, "fallback": 0, "learned_training": 0, "test_access": 0}, "F2 checkpoint protected state differs")
        training = payload["training_history"]
        validation = payload["validation_history"]
        history = payload["checkpoint_history"]
        _require(isinstance(training, list) and isinstance(validation, list) and isinstance(history, list), "F2 checkpoint histories differ")
        _require(training[:-1] == prior_training and validation[:-1] == prior_validation and len(training) == len(validation) == epoch + 1, "F2 epoch history is not an exact cumulative prefix")
        _validate_epoch_record(epoch, training[-1], validation[-1])
        _require(len(history) == epoch, "F2 checkpoint history cadence differs")
        for prior_epoch, record in enumerate(history):
            _require(record == checkpoint_rows[prior_epoch], "F2 prior checkpoint binding differs")
        derived_epoch, derived_top1 = max(range(epoch + 1), key=lambda index: float(validation[index]["top1_accuracy"])), max(float(item["top1_accuracy"]) for item in validation)
        _require(payload["best_epoch"] == derived_epoch and payload["best_validation_top1"] == derived_top1, "F2 running best rule differs")
        optimizer_summary = _optimizer_summary(payload["optimizer_state"])
        _require(payload["optimizer_state"]["param_groups"][0]["lr"] == learning_rate_for_epoch(epoch), "F2 checkpoint optimizer LR differs")
        shape_digest = _state_shape_digest(payload["model_state"])
        if model_shape_digest is None:
            model_shape_digest = shape_digest
        _require(shape_digest == model_shape_digest, "F2 model parameter shapes changed")
        checkpoint_rows.append({"completed_epoch": epoch, "path": f"checkpoints/epoch-{epoch:02d}.pt", "checkpoint_id": digest, "bytes": path.stat().st_size})
        prior_training = list(training)
        prior_validation = list(validation)
        final_optimizer = optimizer_summary
    after = [(path.stat().st_size, path.stat().st_mtime_ns) for path in paths]
    _require(before == after, "F2 checkpoints changed during closeout authentication")
    _require(sha256_bytes(canonical_json(checkpoint_rows)) == CHECKPOINT_MANIFEST_SHA256, "F2 checkpoint manifest digest differs")
    _require(sha256_bytes(canonical_json(prior_training)) == TRAINING_HISTORY_SHA256 and sha256_bytes(canonical_json(prior_validation)) == VALIDATION_HISTORY_SHA256, "F2 training/validation history digest differs")
    best_epoch, best_top1 = select_best(prior_validation)
    _require(best_epoch == BEST_EPOCH and best_top1 == BEST_TOP1 and checkpoint_rows[best_epoch]["checkpoint_id"] == BEST_RELEASE["sha256"] and checkpoint_rows[best_epoch]["bytes"] == BEST_RELEASE["bytes"], "F2 selected checkpoint differs from frozen rule")
    _require(progress.get("status") == "COMPLETED" and progress.get("completed_epoch") == EXPECTED_EPOCHS - 1 and progress.get("total_optimizer_steps") == EXPECTED_OPTIMIZER_STEPS, "F2 terminal progress differs")
    _require(latest.get("completed_epoch") == EXPECTED_EPOCHS - 1 and latest.get("next_epoch") == EXPECTED_EPOCHS and latest.get("checkpoint_id") == checkpoint_rows[-1]["checkpoint_id"], "F2 latest checkpoint pointer differs")

    selected_payload = torch.load(paths[best_epoch], map_location="cpu", weights_only=False)
    model = load_frozen_reference_classifier(torch.device("cpu"), allow_download=False)
    incompatible = model.load_state_dict(selected_payload["model_state"], strict=True)
    _require(not incompatible.missing_keys and not incompatible.unexpected_keys, "F2 selected model weights differ")
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros((1, 3, CANONICAL_AXIS, CANONICAL_AXIS), dtype=torch.float32))
    _require(tuple(output.shape) == (1, CLASS_COUNT) and bool(torch.isfinite(output).all()), "F2 selected model synthetic forward failed")

    start_path = ops_root / "start.timestamp"
    exit_path = ops_root / "exit.status"
    _require(start_path.read_text(encoding="ascii").strip() == "2026-08-24T23:58:03Z", "F2 start timestamp differs")
    _require(exit_path.read_text(encoding="ascii").strip() == "0", "F2 detached exit status differs")
    _require((ops_root / "f2.stderr.log").stat().st_size == 0 and (ops_root / "f2.stdout.log").stat().st_size == 0, "F2 detached logs contain unresolved output")
    for name in ("worker.pid", "wrapper.pid"):
        pid = int((ops_root / name).read_text(encoding="ascii").strip())
        try:
            os.kill(pid, 0)
        except OSError:
            pass
        else:
            raise F2CloseoutHold(f"F2 {name} is still alive")
    trainer_processes = []
    for proc in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = proc.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if "run_g8_f_f2.py" in command:
            trainer_processes.append({"pid": int(proc.parent.name), "command": command})
    _require(not trainer_processes, "a second F2 trainer is running")
    tmux = subprocess.run(["tmux", "has-session", "-t", "g8f-f2"], capture_output=True, check=False)
    _require(tmux.returncode != 0, "F2 detached tmux session is still active")
    end = dt.datetime.fromtimestamp(exit_path.stat().st_mtime, tz=dt.timezone.utc)
    start = dt.datetime.fromisoformat("2026-08-24T23:58:03+00:00")

    completion = identified({
        "schema_version": 1,
        "artifact_role": "g8_f_f2_completion",
        "checkpoint": "F2",
        "status": "GREEN_AUTHENTICATED_TRAINING_CLOSED",
        "authorization": {"path": str(authorization_path.relative_to(REPO_ROOT)), "authorization_id": authorization["authorization_id"], "file_sha256": AUTHORIZATION_SHA256},
        "scientific_source_commit": authorization["source_commit"],
        "launch_head": LAUNCH_HEAD,
        "source_closure": authorization["source_closure"],
        "source_closure_sha256": sha256_bytes(canonical_json(authorization["source_closure"])),
        "execution_profile": authorization["execution_profile"],
        "f1_corpus": {"completion_id": F1_COMPLETION_ID, "completion_sha256": F1_COMPLETION_SHA256, "corpus_id": F1_CORPUS_ID, "manifest_sha256": F1_MANIFEST_SHA256},
        "training_dataset": {**expected_summary, "logical_rows_sha256": logical_sha, "assignment_ids_sha256": assignment_sha, "materialized_f1_ordinals_sha256": ordinals_sha, "multiplicity": "one_logical_item_per_materialized_f1_assignment_row_per_epoch_no_reconstruction_deduplication", "clean_substitutes": 0, "resampling": 0},
        "g1_parent": {"adjudication_id": G1_ADJUDICATION_ID, "adjudication_sha256": G1_ADJUDICATION_SHA256, "checkpoint_id": G1_CHECKPOINT_ID, "checkpoint_sha256": G1_CHECKPOINT_SHA256, "checkpoint_bytes": G1_CHECKPOINT_BYTES, "classifier_variant": "clean", "altered": False},
        "recipe": f2_recipe(),
        "recipe_sha256": f2_recipe_sha256(),
        "execution": {"started_at": "2026-08-24T23:58:03Z", "ended_at": end.isoformat(timespec="microseconds").replace("+00:00", "Z"), "elapsed_seconds": (end - start).total_seconds(), "epochs_completed": EXPECTED_EPOCHS, "steps_per_epoch": EXPECTED_STEPS_PER_EPOCH, "optimizer_steps": EXPECTED_OPTIMIZER_STEPS, "extra_optimizer_steps": 0, "resume_events": [], "detached_exit_status": 0, "worker_processes_at_closeout": 0, "tmux_active_at_closeout": False, "stderr_bytes": 0, "stdout_bytes": 0},
        "training_history": prior_training,
        "training_history_sha256": TRAINING_HISTORY_SHA256,
        "validation_history": prior_validation,
        "validation_history_sha256": VALIDATION_HISTORY_SHA256,
        "checkpoint_manifest": checkpoint_rows,
        "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
        "optimizer_state": final_optimizer,
        "model_state_shape_sha256": model_shape_digest,
        "selection": {"metric": "validation_top1_accuracy", "mode": "max", "tie_break": "earliest_epoch", "best_epoch": best_epoch, "best_validation_top1": best_top1, "best_n_correct": prior_validation[best_epoch]["n_correct"], "best_n_total": prior_validation[best_epoch]["n_total"], "checkpoint_path": checkpoint_rows[best_epoch]["path"], "checkpoint_id": checkpoint_rows[best_epoch]["checkpoint_id"], "checkpoint_file_sha256": checkpoint_rows[best_epoch]["checkpoint_id"], "checkpoint_bytes": checkpoint_rows[best_epoch]["bytes"], "external_artifact": BEST_RELEASE},
        "architecture": {"name": "resnet18", "class_count": CLASS_COUNT, "class_mapping": CLASS_MAPPING, "parameter_shape_sha256": model_shape_digest, "strict_load": True, "synthetic_forward_shape": [1, CLASS_COUNT]},
        "training_time_validation": {"role": "F2_checkpoint_selection_only_not_F3_cached_sweep", "epochs": EXPECTED_EPOCHS, "examples_per_epoch": VALIDATION_IMAGES, "total_inferences": EXPECTED_EPOCHS * VALIDATION_IMAGES},
        "acceptance_gate": {"additional_metric_gate": None, "rule": "successful_exact_recipe_completion_and_preregistered_best_checkpoint_selection", "fallback_authorized": False},
        "ops_bindings": {name: {"bytes": (ops_root / name).stat().st_size, "sha256": _sha(ops_root / name)} for name in ("launch.sh", "start.timestamp", "worker.pid", "wrapper.pid", "exit.status", "f2.stdout.log", "f2.stderr.log")},
        "runtime_bindings": {name: {"bytes": (runtime_root / name).stat().st_size, "sha256": _sha(runtime_root / name)} for name in ("launch.json", "progress.json", "latest.json")},
        "checkpoint_loadability": "PASS_STRICT_SCHEMA_SHAPES_CLASS_MAPPING_NO_MISSING_OR_UNEXPECTED_WEIGHTS",
        "protected_state": PROTECTED,
        "terminal_statement": "F2 GREEN - ARTIFACT-FINETUNED REFERENCE CLASSIFIER AUTHENTICATED AND FROZEN; TRAINING CLOSED; F3/PASS TWO REQUIRE SEPARATE OWNER AUTHORIZATION.",
    }, field="completion_id", prefix=COMPLETION_PREFIX)
    return completion


def build_freeze(completion: Mapping[str, Any], completion_sha256: str) -> dict[str, Any]:
    _verify_completion_object(completion)
    selection = completion["selection"]
    return identified({
        "schema_version": 1,
        "artifact_role": "artifact_finetuned_br12_reference_classifier_freeze",
        "classifier_variant": "artifact_finetuned",
        "scorer_identity": "br12_artifact_finetuned_reference_classifier",
        "distinct_scorers": [
            {"scorer_identity": "g1_clean_trained_reference_classifier", "classifier_variant": "clean", "checkpoint_id": G1_CHECKPOINT_ID, "checkpoint_sha256": G1_CHECKPOINT_SHA256},
            {"scorer_identity": "br12_artifact_finetuned_reference_classifier", "classifier_variant": F2_VARIANT, "checkpoint_id": selection["checkpoint_id"], "checkpoint_sha256": selection["checkpoint_file_sha256"]},
        ],
        "f2_completion": {"path": str(COMPLETION_PATH.relative_to(REPO_ROOT)), "completion_id": completion["completion_id"], "file_sha256": completion_sha256},
        "parent_g1_checkpoint_id": G1_CHECKPOINT_ID,
        "parent_g1_checkpoint_sha256": G1_CHECKPOINT_SHA256,
        "f1_corpus_id": F1_CORPUS_ID,
        "f1_corpus_sha256": F1_MANIFEST_SHA256,
        "authorization_id": completion["authorization"]["authorization_id"],
        "authorization_sha256": AUTHORIZATION_SHA256,
        "scientific_source_commit": completion["scientific_source_commit"],
        "recipe_sha256": f2_recipe_sha256(),
        "train_seed": 0,
        "training_dataset_logical_rows_sha256": LOGICAL_ROWS_SHA256,
        "epochs": EXPECTED_EPOCHS,
        "optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
        "validation_history_sha256": VALIDATION_HISTORY_SHA256,
        "best_epoch": BEST_EPOCH,
        "best_validation_top1": BEST_TOP1,
        "checkpoint_repository_path": "checkpoints/artifact_classifier/epoch-17.pt",
        "checkpoint_id": selection["checkpoint_id"],
        "checkpoint_file_sha256": selection["checkpoint_file_sha256"],
        "checkpoint_bytes": selection["checkpoint_bytes"],
        "checkpoint_external_artifact": BEST_RELEASE,
        "architecture": completion["architecture"],
        "execution_profile": completion["execution_profile"],
        "started_at": completion["execution"]["started_at"],
        "ended_at": completion["execution"]["ended_at"],
        "run_exit_state": "EXITED_0_COMPLETE_NO_RESUME",
        "checkpoint_loadability": "PASS",
        "test_access": 0,
        "protected_state": PROTECTED,
        "status": "FROZEN_TRAINING_CLOSED_F3_CLOSED",
    }, field="freeze_id", prefix=FREEZE_PREFIX)


def _verify_completion_object(value: Mapping[str, Any]) -> None:
    authorization = verify_authorization()
    _require(_sha(AUTHORIZATION_PATH) == AUTHORIZATION_SHA256, "F2 authorization file SHA-256 differs")
    _verify_id(value, field="completion_id", prefix=COMPLETION_PREFIX)
    _require(value.get("artifact_role") == "g8_f_f2_completion" and value.get("status") == "GREEN_AUTHENTICATED_TRAINING_CLOSED", "F2 completion role/status differs")
    _require(
        value.get("authorization") == {
            "path": str(AUTHORIZATION_PATH.relative_to(REPO_ROOT)),
            "authorization_id": authorization["authorization_id"],
            "file_sha256": AUTHORIZATION_SHA256,
        },
        "F2 completion authorization differs",
    )
    _require(value.get("scientific_source_commit") == authorization["source_commit"], "F2 completion scientific source differs")
    _require(value.get("source_closure") == authorization["source_closure"], "F2 completion source closure differs")
    _require(value.get("source_closure_sha256") == sha256_bytes(canonical_json(authorization["source_closure"])), "F2 completion source-closure digest differs")
    _require(value.get("execution_profile") == authorization["execution_profile"], "F2 completion execution profile differs")
    _require(
        value.get("f1_corpus") == {
            "completion_id": F1_COMPLETION_ID,
            "completion_sha256": F1_COMPLETION_SHA256,
            "corpus_id": F1_CORPUS_ID,
            "manifest_sha256": F1_MANIFEST_SHA256,
        },
        "F2 completion F1 corpus lineage differs",
    )
    _require(
        value.get("g1_parent") == {
            "adjudication_id": G1_ADJUDICATION_ID,
            "adjudication_sha256": G1_ADJUDICATION_SHA256,
            "checkpoint_id": G1_CHECKPOINT_ID,
            "checkpoint_sha256": G1_CHECKPOINT_SHA256,
            "checkpoint_bytes": G1_CHECKPOINT_BYTES,
            "classifier_variant": "clean",
            "altered": False,
        },
        "F2 completion G1 parent lineage differs",
    )
    _require(value.get("recipe") == f2_recipe() and value.get("recipe_sha256") == f2_recipe_sha256(), "F2 completion recipe differs")
    dataset = value.get("training_dataset", {})
    _require(dataset.get("assignment_rows") == EXPECTED_ASSIGNMENTS and dataset.get("materialized_rows") == EXPECTED_MATERIALIZED and dataset.get("omitted_rows") == EXPECTED_OMISSIONS and dataset.get("unique_reconstruction_sha256") == EXPECTED_UNIQUE_RECONSTRUCTIONS, "F2 completion data counts differ")
    _require(dataset.get("logical_rows_sha256") == LOGICAL_ROWS_SHA256 and dataset.get("clean_substitutes") == dataset.get("resampling") == 0, "F2 completion logical data identity differs")
    training = value.get("training_history")
    validation = value.get("validation_history")
    _require(isinstance(training, list) and isinstance(validation, list) and len(training) == len(validation) == EXPECTED_EPOCHS, "F2 completion histories differ")
    for epoch, (train, valid) in enumerate(zip(training, validation, strict=True)):
        _validate_epoch_record(epoch, train, valid)
    _require(sha256_bytes(canonical_json(training)) == value.get("training_history_sha256") == TRAINING_HISTORY_SHA256, "F2 completion training digest differs")
    _require(sha256_bytes(canonical_json(validation)) == value.get("validation_history_sha256") == VALIDATION_HISTORY_SHA256, "F2 completion validation digest differs")
    checkpoints = value.get("checkpoint_manifest")
    _require(isinstance(checkpoints, list) and len(checkpoints) == EXPECTED_EPOCHS and [item.get("completed_epoch") for item in checkpoints] == list(range(EXPECTED_EPOCHS)), "F2 completion checkpoint manifest differs")
    _require(sha256_bytes(canonical_json(checkpoints)) == value.get("checkpoint_manifest_sha256") == CHECKPOINT_MANIFEST_SHA256, "F2 completion checkpoint digest differs")
    best_epoch, best_top1 = select_best(validation)
    selection = value.get("selection", {})
    _require(best_epoch == selection.get("best_epoch") == BEST_EPOCH and best_top1 == selection.get("best_validation_top1") == BEST_TOP1, "F2 completion selection differs")
    _require(selection.get("checkpoint_id") == selection.get("checkpoint_file_sha256") == BEST_RELEASE["sha256"] and selection.get("external_artifact") == BEST_RELEASE, "F2 completion selected bytes differ")
    execution = value.get("execution", {})
    _require(execution.get("epochs_completed") == EXPECTED_EPOCHS and execution.get("steps_per_epoch") == EXPECTED_STEPS_PER_EPOCH and execution.get("optimizer_steps") == EXPECTED_OPTIMIZER_STEPS and execution.get("extra_optimizer_steps") == 0 and execution.get("resume_events") == [] and execution.get("detached_exit_status") == 0, "F2 completion execution differs")
    _require(value.get("training_time_validation", {}).get("total_inferences") == EXPECTED_EPOCHS * VALIDATION_IMAGES, "F2 validation inference accounting differs")
    _require(value.get("protected_state") == PROTECTED, "F2 completion protected state differs")


def verify_compact(completion_path: Path = COMPLETION_PATH, freeze_path: Path = FREEZE_PATH, monitor_path: Path | None = MONITOR_PATH) -> tuple[dict[str, Any], dict[str, Any]]:
    completion_raw = completion_path.read_bytes()
    completion = _json(completion_path)
    _require(completion_raw == rendered_json(completion), "F2 completion is not canonical rendered JSON")
    _verify_completion_object(completion)
    freeze_raw = freeze_path.read_bytes()
    freeze = _json(freeze_path)
    _require(freeze_raw == rendered_json(freeze), "artifact classifier freeze is not canonical rendered JSON")
    _verify_id(freeze, field="freeze_id", prefix=FREEZE_PREFIX)
    _require(freeze == build_freeze(completion, sha256_bytes(completion_raw)), "artifact classifier freeze differs from completion")
    if monitor_path is not None and monitor_path.exists():
        monitor_raw = monitor_path.read_bytes()
        monitor = _json(monitor_path)
        _require(monitor_raw == rendered_json(monitor), "F2 monitor closeout is not canonical rendered JSON")
        _verify_id(monitor, field="monitor_closeout_id", prefix=MONITOR_PREFIX)
        _require(monitor.get("completion_id") == completion["completion_id"] and monitor.get("freeze_id") == freeze["freeze_id"] and monitor.get("delivery", {}).get("http_status") == 204 and monitor.get("transition", {}).get("active_f2_polling") is False and monitor.get("transition", {}).get("webhook_configuration_deleted") is False, "F2 monitor closeout differs")
    return completion, freeze


def verify_checkpoint(checkpoint_path: Path, freeze_path: Path = FREEZE_PATH) -> None:
    completion_path = freeze_path.parent / COMPLETION_PATH.name
    _, freeze = verify_compact(completion_path, freeze_path, monitor_path=None)
    _require(checkpoint_path.stat().st_size == freeze["checkpoint_bytes"] and _sha(checkpoint_path) == freeze["checkpoint_file_sha256"], "artifact classifier checkpoint bytes differ")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _require(isinstance(payload, Mapping) and set(payload) == CHECKPOINT_FIELDS and payload.get("completed_epoch") == BEST_EPOCH, "artifact classifier checkpoint schema/epoch differs")
    model = load_frozen_reference_classifier(torch.device("cpu"), allow_download=False)
    incompatible = model.load_state_dict(payload["model_state"], strict=True)
    _require(not incompatible.missing_keys and not incompatible.unexpected_keys, "artifact classifier state dict differs")
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros((1, 3, CANONICAL_AXIS, CANONICAL_AXIS), dtype=torch.float32))
    _require(tuple(output.shape) == (1, CLASS_COUNT) and bool(torch.isfinite(output).all()), "artifact classifier synthetic forward failed")


def build_monitor_closeout(*, completion: Mapping[str, Any], freeze: Mapping[str, Any], delivered_at: str, http_status: int, message_sha256: str, source_binding: Mapping[str, Any], state_binding: Mapping[str, Any], log_binding: Mapping[str, Any], service_binding: Mapping[str, Any], timer_binding: Mapping[str, Any]) -> dict[str, Any]:
    return identified({
        "schema_version": 1,
        "artifact_role": "g8_f_f2_discord_monitor_closeout",
        "status": "COMPLETE",
        "completion_id": completion["completion_id"],
        "freeze_id": freeze["freeze_id"],
        "delivery": {"delivered_at": delivered_at, "http_status": http_status, "message_sha256": message_sha256, "contains": {"status": "F2 COMPLETE", "epochs": "20/20", "optimizer_steps": 6900, "best_epoch_zero_based": BEST_EPOCH, "best_validation_top1": BEST_TOP1, "selected_checkpoint_id": BEST_RELEASE["sha256"], "elapsed_seconds": completion["execution"]["elapsed_seconds"], "f3_pass_two_closed": True}},
        "bindings": {"closeout_source": dict(source_binding), "state": dict(state_binding), "log": dict(log_binding), "service_unit": dict(service_binding), "timer_unit": dict(timer_binding)},
        "transition": {"active_f2_polling": False, "service_active": "inactive", "timer_active": "inactive", "timer_enabled": "disabled", "webhook_configuration_deleted": False, "automatic_f3_launch": False},
        "secret_disclosure": False,
    }, field="monitor_closeout_id", prefix=MONITOR_PREFIX)
