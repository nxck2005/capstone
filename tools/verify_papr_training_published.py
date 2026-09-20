#!/usr/bin/env python3
"""Authenticate the committed PAPR terminal evidence in a clean clone.

This is intentionally weaker than ``verify_papr_terminal`` in one precise
way: it does not read or recompute the worker-local epoch/checkpoint runtime.
It does authenticate every immutable terminal record and all cross-bindings
that are published for hosted CI.  The Confessor verifier remains the source
of the stronger runtime-chain proof.
"""

from __future__ import annotations

import hashlib
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config.params import get  # noqa: E402
from evaluation.downstream_v4 import read_json, require  # noqa: E402
from training.deterministic_core import canonical_sha256  # noqa: E402
from training.papr_constrained import (  # noqa: E402
    PAPR_AUTHORITY_PATH,
    PAPR_COMPLETION_PATH,
    PAPR_COMPLETION_PREFIX,
    PAPR_COMPLETION_ROLE,
    PAPR_EPOCHS,
    PAPR_RATIO,
    PAPR_RUN_ID,
    PAPR_SELECTED_CHECKPOINT_PATH,
    PAPR_SELECTED_PREFIX,
    PAPR_SELECTED_ROLE,
    PAPR_TRAIN_SEED,
    PAPR_CHANNEL_SEED,
    active_protected_counters,
    papr_cap_db,
    pre_execution_protected_counters,
)
from training.papr_lifecycle import verify_papr_authority  # noqa: E402
from training.w8_final import W8_PAPR_BOUND_TOLERANCE_DB, W8_PAPR_DOMAIN  # noqa: E402

AUTHORITY = REPO / PAPR_AUTHORITY_PATH
SELECTED = REPO / PAPR_SELECTED_CHECKPOINT_PATH
COMPLETION = REPO / PAPR_COMPLETION_PATH


class PublishedPaprHold(RuntimeError):
    """Published PAPR evidence is incomplete or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublishedPaprHold(message)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _full_sha(value: object, width: int = 64) -> str:
    _require(
        isinstance(value, str)
        and len(value) == width
        and value != "0" * width
        and all(character in "0123456789abcdef" for character in value),
        "PAPR published SHA-256 is malformed",
    )
    return value


def _content_address(value: Mapping[str, Any], field: str, prefix: str, label: str) -> None:
    body = dict(value)
    identifier = body.pop(field, None)
    _require(identifier == prefix + canonical_sha256(body), f"{label} ID differs")


def _selection(value: Mapping[str, Any], authority: Mapping[str, Any], authority_sha256: str) -> None:
    _content_address(value, "selection_id", PAPR_SELECTED_PREFIX, "PAPR selected checkpoint")
    _require(value.get("schema_version") == 1 and value.get("artifact_role") == PAPR_SELECTED_ROLE, "PAPR selected role/schema differs")
    _require(value.get("status") == "SELECTED_VALIDATION_ONLY", "PAPR selected status differs")
    _require(value.get("authority_id") == authority["authority_id"], "PAPR selected authority differs")
    _require(value.get("authority_path") == PAPR_AUTHORITY_PATH, "PAPR selected authority path differs")
    _require(value.get("authority_sha256") == authority_sha256, "PAPR selected authority SHA differs")
    _require(value.get("source_manifest") == authority.get("source_manifest"), "PAPR selected source manifest differs")
    _require(value.get("scientific_source_commit") == authority.get("source_commit"), "PAPR selected source commit differs")
    _require(isinstance(value.get("execution_commit"), str) and len(value["execution_commit"]) == 40, "PAPR selected execution commit is malformed")  # literal-ok: Git SHA-1 width
    _require(value.get("config_hash") == authority.get("config_hash"), "PAPR selected config hash differs")
    _require(value.get("protocol_config_hash") == authority.get("protocol_config_hash"), "PAPR selected protocol hash differs")
    _require(value.get("papr_cap_db") == papr_cap_db() and value.get("papr_domain") == W8_PAPR_DOMAIN, "PAPR selected cap/domain differs")
    _require(value.get("train_seed") == PAPR_TRAIN_SEED and value.get("channel_seed") == PAPR_CHANNEL_SEED, "PAPR selected cell differs")
    _require(value.get("epochs_completed") == PAPR_EPOCHS, "PAPR selected epoch-chain length differs")
    _full_sha(value.get("epoch_chain_digest"))
    _full_sha(value.get("validation_trajectory_digest"))
    selection = value.get("selection")
    _require(isinstance(selection, Mapping), "PAPR selected checkpoint rule is missing")
    _require(selection.get("selection_id") == canonical_sha256({key: item for key, item in selection.items() if key != "selection_id"}), "PAPR selected selection rule ID differs")
    _require(
        selection.get("artifact_role") == "PAPR_CONSTRAINED_SELECTED_CHECKPOINT"
        and selection.get("metric") == "validation_top1_accuracy"
        and selection.get("mode") == "max"
        and selection.get("tie_break") == "earliest_epoch"
        and selection.get("cross_seed_selection") is False
        and selection.get("psnr_selected") is False
        and selection.get("papr_selected") is False
        and selection.get("reconstruction_loss_selected") is False,
        "PAPR selected checkpoint rule differs",
    )
    epoch = value.get("selected_epoch")
    _require(isinstance(epoch, int) and not isinstance(epoch, bool) and 0 <= epoch < PAPR_EPOCHS, "PAPR selected epoch is invalid")
    _require(selection.get("selected_epoch") == epoch and selection.get("selected_checkpoint_id") == value.get("checkpoint_id"), "PAPR selected checkpoint/selection differs")
    _full_sha(value.get("checkpoint_id"))
    _require(value.get("checkpoint_path") == f"checkpoints/papr_constrained_pascal_v4/train0_channel0/checkpoints/epoch-{epoch:04d}.pt", "PAPR selected checkpoint path differs")
    _require(isinstance(value.get("checkpoint_bytes"), int) and not isinstance(value["checkpoint_bytes"], bool) and value["checkpoint_bytes"] > 0, "PAPR selected checkpoint byte metadata differs")
    _require(value.get("n_total") == int(get("datasets.imagenette160.val_images")) and 0 <= int(value.get("n_correct", -1)) <= value["n_total"], "PAPR selected validation count differs")
    _require(value.get("training_runs") == 1 and value.get("protected_counters") == active_protected_counters(), "PAPR selected counters differ")
    _require(value.get("test") == "SEALED" and value.get("test_access") == 0, "PAPR selected evidence crossed the test boundary")


def _completion(value: Mapping[str, Any], authority: Mapping[str, Any], selected: Mapping[str, Any], authority_sha256: str) -> None:
    _content_address(value, "completion_id", PAPR_COMPLETION_PREFIX, "PAPR completion")
    _require(value.get("schema_version") == 1 and value.get("artifact_role") == PAPR_COMPLETION_ROLE, "PAPR completion role/schema differs")
    _require(value.get("status") == "COMPLETE_VALIDATION_ONLY", "PAPR completion status differs")
    _require(value.get("authority_id") == authority["authority_id"] and value.get("authority_path") == PAPR_AUTHORITY_PATH, "PAPR completion authority differs")
    _require(value.get("authority_sha256") == authority_sha256, "PAPR completion authority SHA differs")
    _require(value.get("source_manifest") == authority.get("source_manifest") and value.get("scientific_source_commit") == authority.get("source_commit"), "PAPR completion source differs")
    _require(isinstance(value.get("execution_commit"), str) and len(value["execution_commit"]) == 40, "PAPR completion execution commit is malformed")  # literal-ok: Git SHA-1 width
    _require(value.get("runtime_root") == "checkpoints/papr_constrained_pascal_v4", "PAPR completion runtime identity differs")
    _require(value.get("campaign_id") and value.get("run_id") == PAPR_RUN_ID and value.get("ratio") in (None, PAPR_RATIO), "PAPR completion run identity differs")
    _require(value.get("config_hash") == authority.get("config_hash") and value.get("protocol_config_hash") == authority.get("protocol_config_hash"), "PAPR completion config/protocol differs")
    _require(value.get("papr_cap_db") == papr_cap_db() and value.get("papr_domain") == W8_PAPR_DOMAIN, "PAPR completion cap/domain differs")
    observed = value.get("papr_max_observed_db")
    _require(isinstance(observed, int | float) and not isinstance(observed, bool) and math.isfinite(float(observed)), "PAPR completion observed cap is invalid")
    _require(value.get("papr_cap_compliant") is True and float(observed) <= papr_cap_db() + W8_PAPR_BOUND_TOLERANCE_DB, "PAPR completion cap compliance differs")
    _require(value.get("epochs") == PAPR_EPOCHS, "PAPR completion epoch count differs")
    _full_sha(value.get("epoch_chain_digest"))
    _full_sha(value.get("validation_trajectory_digest"))
    for field in ("optimizer_step_opportunities", "optimizer_steps", "grad_scaler_skips", "global_optimizer_step"):
        _require(isinstance(value.get(field), int) and not isinstance(value[field], bool) and value[field] >= 0, f"PAPR completion {field} is invalid")
    _require(value["optimizer_steps"] + value["grad_scaler_skips"] == value["optimizer_step_opportunities"], "PAPR completion optimizer accounting differs")
    _require(value["global_optimizer_step"] == value["optimizer_steps"], "PAPR completion global step differs")
    _require(value.get("selection_id") == selected.get("selection_id") and value.get("selected_epoch") == selected.get("selected_epoch"), "PAPR completion selection differs")
    _require(value.get("checkpoint_id") == selected.get("checkpoint_id") and value.get("checkpoint_path") == selected.get("checkpoint_path") and value.get("checkpoint_bytes") == selected.get("checkpoint_bytes"), "PAPR completion checkpoint differs")
    _require(value.get("n_correct") == selected.get("n_correct") and value.get("n_total") == selected.get("n_total"), "PAPR completion validation count differs")
    _require(value.get("training_runs") == 1 and value.get("train_seed") == PAPR_TRAIN_SEED and value.get("channel_seed") == PAPR_CHANNEL_SEED, "PAPR completion training identity differs")
    _require(value.get("fresh_initialization") is True and value.get("transfer_initialization") is False and _full_sha(value.get("initial_model_state_sha256")), "PAPR completion initialization differs")
    _require(value.get("protected_counters") == active_protected_counters(), "PAPR completion counters differ")
    _require(value.get("test") == "SEALED" and value.get("test_access") == 0, "PAPR completion crossed the test boundary")


def verify_published() -> dict[str, str | int]:
    """Verify immutable terminal evidence, explicitly without worker runtime."""

    authority = verify_papr_authority(REPO, authority_path=AUTHORITY, require_live_source=True)
    _require(SELECTED.is_file() and not SELECTED.is_symlink(), "PAPR selected checkpoint evidence is missing or unsafe")
    _require(COMPLETION.is_file() and not COMPLETION.is_symlink(), "PAPR completion evidence is missing or unsafe")
    authority_sha256 = _sha256_file(AUTHORITY)
    selected = read_json(SELECTED, "PAPR selected checkpoint")
    completion = read_json(COMPLETION, "PAPR completion")
    _selection(selected, authority, authority_sha256)
    _completion(completion, authority, selected, authority_sha256)
    return {"authority_id": str(authority["authority_id"]), "selection_id": str(selected["selection_id"]), "completion_id": str(completion["completion_id"]), "test_access": 0}


if __name__ == "__main__":
    value = verify_published()
    print(f"PAPR published-evidence verifier PASS: {value['completion_id']}; worker runtime not recomputed")
