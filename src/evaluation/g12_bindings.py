"""Frozen G-12 bindings: every checkpoint, scorer and operating point, per cell.

Cell (0, 0) reuses the frozen W10 authority bindings unchanged (the W10 v11
continuation authority is checked to carry the same binding identities).  The
ER-1 learned and ER-9 arms add cells (1, 1) and (2, 2) from the same closed
W8 reconciliation and ER-9 production closeout W10 drew cell (0, 0) from.
Classical arms have no training seed: their operating points are seed-free and
only the channel seed changes between cells.  AM-100's variant is bound to its
closed Stage-1 checkpoint and its committed validation record.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from config.params import get
from evaluation.g12_scope import SCOPE, G12Arm
from training.deterministic_core import canonical_sha256

W10_AUTHORITY = "results/learned/w10/w10_rehearsal_authorization.json"
W10_CONTINUATION_AUTHORITY = "results/learned/w10/w10_continuation_authorization_v11.json"
W8_RECONCILIATION = "results/learned/w8/w8_c_reconciliation.json"
W8_CAMPAIGN_ROOT = "/home/nick/w8-final-pascal-20260901-r1"
ER9_PRODUCTION_CLOSEOUT = "results/learned/er9/er9_production_closeout_v4.json"
LOW_RATE_VALIDATION = "results/learned/er9_low_rate/validation_summary.json"
LOW_RATE_RUNTIME_ROOT = "checkpoints/er9_pascal_v4/stage1/D1024_b2"


class G12BindingError(RuntimeError):
    """A G-12 binding is missing or differs from its closed source."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G12BindingError(message)


def _read(root: Path, relative: str) -> Any:
    path = Path(root) / relative
    _require(path.is_file() and not path.is_symlink(), f"G-12 binding source is missing: {relative}")
    return json.loads(path.read_bytes())


def _record(root: Path, relative: str) -> dict[str, Any]:
    raw = (Path(root) / relative).read_bytes()
    return {"path": relative, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def w10_bindings(root: Path) -> dict[str, dict[str, Any]]:
    authority = _read(root, W10_AUTHORITY)
    continuation = _read(root, W10_CONTINUATION_AUTHORITY)
    bindings = {str(item["scope"]["role"]): item for item in authority["bindings"]}
    later = {str(item["scope"]["role"]): item["binding_id"] for item in continuation["bindings"]}
    _require(
        {role: item["binding_id"] for role, item in bindings.items()} == later,
        "W10 continuation authority binds different identities from the original authority",
    )
    return bindings


def _w8_cell(root: Path, ratio: str, cell: int) -> dict[str, Any]:
    reconciliation = _read(root, W8_RECONCILIATION)
    selection = {str(item["run_id"]): item for item in reconciliation["selection"]["per_run"]}
    for run in reconciliation["run_identities"]:
        if run["ratio"] == ratio and int(run["train_seed"]) == cell and int(run["channel_seed"]) == cell:
            recovered = selection[str(run["run_id"])]["independently_reconstructed"]
            return {
                "state": "FROZEN",
                "kind": "w8_selected_checkpoint",
                "reconciliation": _record(root, W8_RECONCILIATION),
                "run_id": str(run["run_id"]),
                "ratio": ratio,
                "train_seed": cell,
                "channel_seed": cell,
                "w8_config_hash": str(run["config_hash"]),
                "checkpoint_id": str(recovered["checkpoint_id"]),
                "epoch": int(recovered["epoch"]),
                "checkpoint_path": f"{W8_CAMPAIGN_ROOT}/{run['run_directory']}/checkpoints/epoch-{int(recovered['epoch']):04d}.pt",
                "validation_selection_correct": int(recovered["n_correct"]),
            }
    raise G12BindingError(f"W8 selected checkpoint is missing for {ratio} cell {cell}")


def _er9_cell(root: Path, cell: int) -> dict[str, Any]:
    closeout = _read(root, ER9_PRODUCTION_CLOSEOUT)
    records = [item for item in closeout["seed_cells"] if (int(item["train_seed"]), int(item["channel_seed"])) == (cell, cell)]
    _require(len(records) == 1, f"ER-9 production cell {cell} is missing")
    record = records[0]
    validation_path = str(record["validation_path"])
    _require(_record(root, validation_path)["sha256"] == record["validation_sha256"], "ER-9 final-validation bytes differ")
    validation = _read(root, validation_path)
    per_snr_phy = [
        {
            "snr_db": float(point["snr_db"]),
            "modulation": str(point["selected"]["modulation"]),
            "ldpc_rate": str(point["selected"]["ldpc_rate"]),
        }
        for point in validation["points"]
    ]
    return {
        "state": "FROZEN",
        "kind": "er9_production_cell",
        "closeout": _record(root, ER9_PRODUCTION_CLOSEOUT),
        "train_seed": cell,
        "channel_seed": cell,
        "runtime_root": str(record["runtime_root"]),
        "selected_epoch": int(record["selected_epoch"]),
        "selected_checkpoint_sha256": str(record["selected_checkpoint_sha256"]),
        "validation_path": validation_path,
        "validation_sha256": str(record["validation_sha256"]),
        "entropy_table": validation["entropy_model"]["table"],
        "per_snr_phy": per_snr_phy,
    }


def _low_rate(root: Path) -> dict[str, Any]:
    summary = _read(root, LOW_RATE_VALIDATION)
    keys = ("transmit_dim", "quantiser_bits", "checkpoint_epoch", "checkpoint_sha256", "modulation", "ldpc_rate", "payload_bits")
    variant = {key: get(f"digital_semantic_control.low_rate_variant_{key}") for key in keys}
    _require(all(summary["variant"][key] == value for key, value in variant.items()), "AM-100 validation record differs from its parameters")
    _require(summary["git_dirty"] is False and summary["test_access"] == 0, "AM-100 validation record is not clean and sealed")
    return {
        "state": "FROZEN",
        "kind": "am100_low_rate_variant",
        "validation": _record(root, LOW_RATE_VALIDATION),
        "runtime_root": LOW_RATE_RUNTIME_ROOT,
        **variant,
    }


def checkpoint_for(root: Path, arm: G12Arm, cell: int, w10: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if arm.backend == "er9_low_rate":
        return _low_rate(root)
    bound = w10[str(arm.w10_role)]
    if arm.backend == "classical" or arm.backend == "jpeg_secondary":
        return dict(bound["checkpoint"])
    if cell == 0:
        return dict(bound["checkpoint"])
    if arm.backend == "learned" and arm.system == "learned":
        return _w8_cell(root, arm.bw_ratio, cell)
    if arm.backend == "er9_digital":
        return _er9_cell(root, cell)
    raise G12BindingError(f"{arm.role} has no binding for cell {cell}")


def resolve(root: Path) -> list[dict[str, Any]]:
    """Every arm × cell binding, content-addressed, in scope order."""

    root = Path(root)
    w10 = w10_bindings(root)
    resolved = []
    for arm in SCOPE:
        selection = None if arm.w10_role is None else w10[arm.w10_role].get("selection")
        for train_seed, _channel_seed in arm.cells:
            checkpoint = checkpoint_for(root, arm, train_seed, w10)
            if arm.backend == "er9_digital" and train_seed == 0:
                # Cell (0, 0) also carries its compression table for the same refit check.
                checkpoint = {**checkpoint, "entropy_table": _er9_cell(root, 0)["entropy_table"]}
            body = {
                "arm": arm.identity(),
                "cell": [train_seed, train_seed],
                "w10_binding_id": None if arm.w10_role is None else w10[arm.w10_role]["binding_id"],
                "checkpoint": checkpoint,
                "selection": selection,
            }
            resolved.append({**body, "binding_id": "g12binding-" + canonical_sha256(body)})
    return resolved


def binding_for(bindings: list[Mapping[str, Any]], role: str, cell: int) -> Mapping[str, Any]:
    for item in bindings:
        if item["arm"]["role"] == role and int(item["cell"][0]) == cell:
            return item
    raise G12BindingError(f"no G-12 binding for {role} cell {cell}")


__all__ = ["G12BindingError", "binding_for", "checkpoint_for", "resolve", "w10_bindings"]
