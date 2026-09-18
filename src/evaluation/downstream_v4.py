"""Fail-closed contracts shared by the final W9 successor lifecycle."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from config.params import get
from config.run_config import config_hash, load_experiment
from runtime.source_guard import assert_clean_source_closure, assert_downstream_manifest_contract
from training.deterministic_core import canonical_bytes, canonical_sha256


FINAL_PAIR = {"transmit_dim": 2048, "quantiser_bits": 2}  # literal-ok: final closed Stage-2 selection
PRODUCTION_CELLS = ((0, 0), (1, 1), (2, 2))
FROZEN_SNR_GRID = tuple(int(value) for value in get("channel.test_snr_grid_db"))
EXPECTED_SNR_GRID = FROZEN_SNR_GRID
STAGE2_PATH = "results/learned/er9/er9_stage2_selection_v4.json"
STAGE2_ID = "er9stage2v4selection-cfa70ee970b142434c46c84acdc932a41d6f97436ba522e7793e3898fe84db14"
STAGE2_SHA256 = "1603fa734dc790646676d831df315f3becc47b4e4fead1f68fe13581e7ba9f4b"
SOURCE_PATH = "results/learned/w9/downstream_source_manifest_v4.json"
PRODUCTION_AUTHORITY_PATH = "results/learned/er9/er9_production_execution_authorization_v4.json"
ER2_AUTHORITY_PATH = "results/learned/er2_randomized/er2_execution_authorization_v4.json"
G11_AUTHORITY_PATH = "results/learned/g11/g11_execution_authorization_v4.json"
PRODUCTION_RUNTIME_ROOT = "checkpoints/er9_production_pascal_v4"
ER2_RUNTIME_ROOT = "checkpoints/er2_randomized_pascal_v4"
ER9_CONFIG = "configs/er9-digital-pascal-v4.yaml"
ER2_CONFIG = "configs/learned-er2-randomized-pascal-v4.yaml"
TITAN_XP_NAME = "NVIDIA TITAN Xp"
TITAN_XP_UUID = "GPU-46acd0f2-2ff5-1a43-cac9-2ae20e56dc9a"


class DownstreamHold(RuntimeError):
    """The final downstream source, authority, or result differs."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DownstreamHold(message)


def read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{label} is missing or unsafe")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DownstreamHold(f"{label} is corrupt: {exc}") from None
    require(isinstance(value, dict), f"{label} is not an object")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_stage2(root: Path) -> dict[str, Any]:
    path = root / STAGE2_PATH
    require(sha256_file(path) == STAGE2_SHA256, "closed Stage-2 bytes differ")
    value = read_json(path, "closed Stage-2 selection")
    require(value.get("selection_id") == STAGE2_ID, "closed Stage-2 ID differs")
    require(value.get("status") == "STAGE2_CLOSED_ZERO_WORK", "Stage-2 is not closed zero-work")
    require(value.get("selected_pair") == FINAL_PAIR, "final ER-9 pair differs")
    require(value.get("new_training_count") == 0 and value.get("new_evaluation_count") == 0, "Stage-2 work counters differ")
    require(value.get("test") == "SEALED" and value.get("test_access") == 0, "Stage-2 test boundary differs")
    return value


def load_source(root: Path, *, live: bool = True) -> dict[str, Any]:
    path = root / SOURCE_PATH
    value = read_json(path, "downstream source manifest")
    body = dict(value)
    identifier = body.pop("manifest_id", None)
    require(identifier == "w9downstreamsource-" + canonical_sha256(body), "downstream source manifest ID differs")
    assert_downstream_manifest_contract(value)
    if live:
        from runtime.source_epochs import assert_active_epoch_closure

        assert_active_epoch_closure(root, value)
    load_stage2(root)
    return value


def source_record(root: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    path = root / SOURCE_PATH
    return {"path": SOURCE_PATH, "manifest_id": source["manifest_id"], "sha256": sha256_file(path)}


def cell_config_records(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = []
    for train_seed, channel_seed in PRODUCTION_CELLS:
        config = load_experiment(ER9_CONFIG, train_seed=train_seed, channel_seed=channel_seed)
        records.append({
            "train_seed": train_seed,
            "channel_seed": channel_seed,
            "config_hash": config_hash(config),
            "runtime_root": f"{PRODUCTION_RUNTIME_ROOT}/train{train_seed}_channel{channel_seed}",
        })
    return records


def _profile(authority: Mapping[str, Any]) -> None:
    require(authority.get("execution_profile_id") == "confessor_pascal_cu126", "downstream profile differs")
    require(authority.get("host") == "confessor" and authority.get("device") == "cuda:0", "downstream host/device differs")
    uuid = authority.get("gpu_uuid")
    require(authority.get("gpu_name") == TITAN_XP_NAME and uuid == TITAN_XP_UUID, "downstream worker is not the exact Confessor TITAN Xp")
    require(isinstance(uuid, str) and uuid.startswith("GPU-") and authority.get("cuda_visible_devices") == uuid, "downstream GPU binding differs")
    require(authority.get("cuda_mapping") == {
        "cuda_visible_devices": uuid,
        "logical_device": "cuda:0",
        "cuda0_gpu_uuid": uuid,
        "cuda0_gpu_name": authority.get("gpu_name"),
        "cuda0_compute_capability": authority.get("compute_capability"),
        "device_count": 1,
    }, "downstream CUDA mapping differs")


def verify_production_authority(root: Path, path: Path | None = None, *, live_source: bool = True) -> dict[str, Any]:
    source = load_source(root, live=live_source)
    path = path or root / PRODUCTION_AUTHORITY_PATH
    value = read_json(path, "ER-9 production authority")
    body = dict(value)
    identifier = body.pop("authority_id", None)
    require(identifier == "w9er9productionv4auth-" + canonical_sha256(body), "production authority ID differs")
    require(value.get("schema_version") == 1 and value.get("authority_kind") == "W9_ER9_PRODUCTION_EXECUTION_AUTHORITY_V4", "production authority role differs")
    require(value.get("status") == "FROZEN_PRODUCTION_ONLY_PRE_SCIENCE" and value.get("authorization_scope") == "W9_ER9_PRODUCTION_ONLY", "production authority scope differs")
    require(value.get("source_manifest") == source_record(root, source) and value.get("source_binding") == source and value.get("source_commit") == source["source_commit"], "production source binding differs")
    require(value.get("stage2_closeout") == {"path": STAGE2_PATH, "selection_id": STAGE2_ID, "sha256": STAGE2_SHA256}, "production Stage-2 binding differs")
    require(value.get("selected_pair") == FINAL_PAIR, "production pair differs")
    require(value.get("seed_cells") == cell_config_records(source), "production cells/configs/runtime roots differ")
    require(value.get("training_count") == 3 and value.get("no_best_seed_selection") is True, "production count/selection policy differs")
    require(value.get("stage1_promotion") is False and value.get("fresh_initialization_required") is True, "production fresh-start policy differs")
    require(value.get("runtime_root") == PRODUCTION_RUNTIME_ROOT and value.get("epochs") == 100, "production runtime/epoch scope differs")  # literal-ok: AM-96 frozen epoch count
    require(value.get("checkpoint_selection") == {"metric": "validation_n_correct", "mode": "max", "tie_break": "earliest_epoch"}, "production checkpoint rule differs")
    require(value.get("resume_semantics") == "exact_authenticated_completed_epoch_model_optimizer_scaler", "production resume semantics differ")
    for key in ("er2_authorized", "g11_authorized", "w10_authorized", "test_authorized"):
        require(value.get(key) is False, f"production authority improperly authorizes {key}")
    require(value.get("test") == "SEALED" and value.get("test_access") == 0, "production test boundary differs")
    _profile(value)
    return value


def verify_er2_authority(root: Path, path: Path | None = None, *, live_source: bool = True) -> dict[str, Any]:
    source = load_source(root, live=live_source)
    path = path or root / ER2_AUTHORITY_PATH
    value = read_json(path, "randomized ER-2 authority")
    body = dict(value)
    identifier = body.pop("authority_id", None)
    require(identifier == "w9er2randomizedv4auth-" + canonical_sha256(body), "ER-2 authority ID differs")
    require(value.get("authority_kind") == "W9_ER2_RANDOMIZED_EXECUTION_AUTHORITY_V4" and value.get("authorization_scope") == "W9_ER2_RANDOMIZED_ONLY", "ER-2 authority scope differs")
    require(value.get("source_manifest") == source_record(root, source) and value.get("source_binding") == source, "ER-2 source binding differs")
    require(value.get("runtime_root") == ER2_RUNTIME_ROOT and value.get("seed_cell") == {"train_seed": 0, "channel_seed": 0}, "ER-2 single-run identity differs")
    require(value.get("scientific_training_run_count") == 1 and value.get("randomized_er2_authorized") is True, "ER-2 run count differs")
    require(value.get("fresh_initialization_required") is True, "ER-2 fresh-start policy differs")
    require(value.get("randomized_snr_domain") == [1, 4, 7, 13, 19], "ER-2 AM-95 domain differs")  # literal-ok: AM-95 exact domain
    for key in ("stage1_authorized", "stage2_authorized", "production_authorized", "g11_authorized", "w10_authorized", "test_authorized"):
        require(value.get(key) is False, f"ER-2 authority improperly authorizes {key}")
    require(value.get("test") == "SEALED" and value.get("test_access") == 0, "ER-2 test boundary differs")
    _profile(value)
    return value


def task_head_identity(model: Any) -> str:
    """Hash only the ER-9 task head, independently of the checkpoint file."""

    from training.deterministic_core import state_tree_sha256

    head = getattr(model, "task_head", None)
    if head is None:
        head = getattr(getattr(model, "decoder", None), "task_head", None)
    require(head is not None, "ER-9 own task head is missing")
    return "er9taskhead-" + state_tree_sha256(head.state_dict())


def validate_final_er9_cell(value: Mapping[str, Any], *, cell: tuple[int, int]) -> None:
    require(value.get("artifact_role") == "ER9_FINAL_PRODUCTION_VALIDATION_CELL_V4", "ER-9 final-validation role differs")
    require((value.get("train_seed"), value.get("channel_seed")) == cell, "ER-9 final-validation seed differs")
    require(value.get("selected_pair") == FINAL_PAIR, "ER-9 final-validation pair differs")
    checkpoint = value.get("checkpoint")
    require(isinstance(checkpoint, Mapping), "ER-9 final-validation checkpoint binding is missing")
    checkpoint_id = checkpoint.get("sha256")
    require(isinstance(checkpoint_id, str) and len(checkpoint_id) == 64 and checkpoint_id != "0" * 64, "ER-9 checkpoint identity is absent/sentinel")  # literal-ok: SHA-256 width/sentinel
    task_head = value.get("task_head")
    require(isinstance(task_head, Mapping) and task_head.get("kind") == "own_task_head" and len(str(task_head.get("identity", ""))) == len("er9taskhead-") + 64 and str(task_head.get("identity", "")).startswith("er9taskhead-"), "ER-9 own-task-head identity differs")  # literal-ok: SHA-256 identity width
    require(value.get("snr_grid_db") == list(EXPECTED_SNR_GRID), "ER-9 final-validation grid differs")
    points = value.get("points")
    require(isinstance(points, list) and [point.get("snr_db") for point in points] == list(EXPECTED_SNR_GRID), "ER-9 validation point order differs")
    stable_order: list[str] | None = None
    for point in points:
        require(point.get("row_binding") == {
            "checkpoint_id": checkpoint_id,
            "train_seed": cell[0],
            "channel_seed": cell[1],
            "transmit_dim": FINAL_PAIR["transmit_dim"],
            "quantiser_bits": FINAL_PAIR["quantiser_bits"],
            "task_head_identity": task_head.get("identity"),
            "test_access": 0,
        }, "ER-9 per-image row binding differs")
        require(point.get("checkpoint_id") == checkpoint_id, "ER-9 point checkpoint binding differs")
        require(point.get("task_head_identity") == task_head.get("identity"), "ER-9 point task-head binding differs")
        require(point.get("validation_total") == 1000 and point.get("test_access") == 0, "ER-9 validation denominator/test boundary differs")  # literal-ok: frozen validation denominator
        rows = point.get("per_image")
        require(isinstance(rows, list) and len(rows) == 1000, "ER-9 per-image denominator differs")  # literal-ok: frozen validation denominator
        ids = [str(row.get("stable_sample_id")) for row in rows]
        require(len(set(ids)) == 1000 and ids == sorted(ids), "ER-9 stable sample IDs differ")  # literal-ok: frozen validation denominator
        if stable_order is None:
            stable_order = ids
        require(ids == stable_order, "ER-9 stable sample trajectories are incomplete")
        require(all(row.get("split") == "val" and row.get("test_snr_db") == point.get("snr_db") for row in rows), "ER-9 per-image SNR/split differs")
        require(all(row.get("run_id") == point.get("run_id") for row in rows), "ER-9 rows are not checkpoint/task-head bound")
        require(sum(bool(row.get("correct")) for row in rows) == point.get("validation_n_correct"), "ER-9 correctness is not row-derived")
    require(value.get("test") == "SEALED" and value.get("test_access") == 0, "ER-9 final-validation test boundary differs")


def immutable_write(path: Path, value: Mapping[str, Any]) -> None:
    raw = canonical_bytes(dict(value))
    if path.exists() or path.is_symlink():
        require(not path.is_symlink() and path.read_bytes() == raw, f"immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def exact_cells(values: Sequence[Mapping[str, Any]]) -> None:
    require([(item.get("train_seed"), item.get("channel_seed")) for item in values] == list(PRODUCTION_CELLS), "cell list is not the exact three zipped cells")


__all__ = [
    "DownstreamHold", "ER2_AUTHORITY_PATH", "ER2_CONFIG", "ER2_RUNTIME_ROOT", "G11_AUTHORITY_PATH",
    "EXPECTED_SNR_GRID", "FINAL_PAIR", "FROZEN_SNR_GRID", "PRODUCTION_AUTHORITY_PATH",
    "PRODUCTION_CELLS", "PRODUCTION_RUNTIME_ROOT", "SOURCE_PATH", "STAGE2_ID",
    "STAGE2_PATH", "STAGE2_SHA256", "TITAN_XP_NAME", "TITAN_XP_UUID", "cell_config_records", "exact_cells", "immutable_write",
    "load_source", "load_stage2", "read_json", "require", "sha256_file", "source_record",
    "task_head_identity", "validate_final_er9_cell", "verify_er2_authority", "verify_production_authority",
]
