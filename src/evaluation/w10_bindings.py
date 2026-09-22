"""Resolve and authenticate the exact W10 scientific bindings (AM-98).

Every scope entry names where its model, scorer and operating-point identities
come from.  This module turns those names into content-addressed binding
records over immutable repository artifacts.  Nothing here loads a dataset or a
model; execution layers verify the referenced bytes themselves.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from evaluation.downstream_v4 import read_json, require
from evaluation.w10_jpeg_carrier import (
    JPEG_CARRIER_DESCRIPTOR_PATH,
    JPEG_CARRIER_PATH,
    JPEG_SELECTION_PATH as JPEG_LOGICAL_SELECTION_PATH,
    JpegCarrierHold,
    load_jpeg_selection_artifact,
)
from evaluation.w10_scope import SCOPE, W10ScopeEntry
from training.deterministic_core import canonical_sha256

W8_RECONCILIATION = "results/learned/w8/w8_c_reconciliation.json"
ER2_SELECTED_CHECKPOINT = "results/learned/er2_randomized/er2_selected_checkpoint_v4.json"
ER2_COMPLETION = "results/learned/er2_randomized/er2_randomized_completion_v4.json"
ER9_PRODUCTION_CLOSEOUT = "results/learned/er9/er9_production_closeout_v4.json"
ER9_STAGE2_SELECTION = "results/learned/er9/er9_stage2_selection_v4.json"
PASS_TWO_STATE = "results/baseline/g8_f/pass_two_state.json"
CANDIDATE_AUTHORITY = "results/baseline/g8_e/candidate_authority.json"
G8_CLOSEOUT = "results/baseline/g8/g8_closeout.json"
BR12_FREEZE = "results/baseline/g8_f/artifact_classifier_freeze.json"
G1_BEST_CHECKPOINT = "results/reference_classifier/best_checkpoint.json"
JPEG_SELECTION = JPEG_LOGICAL_SELECTION_PATH
ER12_SELECTION = "results/learned/w10/er12_validation_selection.json"
PAPR_SELECTED_CHECKPOINT = "results/learned/w10/papr_selected_checkpoint.json"
PAPR_COMPLETION = "results/learned/w10/papr_training_completion.json"
PAPR_AUTHORITY = "results/learned/w10/papr_training_authorization.json"

PENDING_PREFIX = "pending:"

# The W8 campaign root is Confessor custody; the repository records the exact
# checkpoint identity and relative path, and execution verifies the bytes.
W8_CAMPAIGN_ROOT = "/home/nick/w8-final-pascal-20260901-r1"


def file_sha256(root: Path, relative: str, *, required: bool = True) -> str:
    path = Path(root) / relative
    if not path.is_file() or path.is_symlink():
        if required:
            raise RuntimeError(f"W10 binding artifact is missing or unsafe: {relative}")
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_record(root: Path, relative: str, *, required: bool = True) -> dict[str, Any]:
    path = Path(root) / relative
    if not path.is_file() or path.is_symlink():
        if required:
            raise RuntimeError(f"W10 binding artifact is missing or unsafe: {relative}")
        return {"path": relative, "sha256": "", "present": False}
    value = read_json(path, relative)
    return {
        "path": relative,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "present": True,
        "artifact_role": value.get("artifact_role"),
        "artifact_id": value.get("manifest_id") or value.get("selection_id") or value.get("closeout_id") or value.get("freeze_id") or value.get("completion_id"),
    }


def _pending(kind: str, relative: str, requirement: str) -> dict[str, Any]:
    return {
        "state": PENDING_PREFIX + kind,
        "required_artifact": relative,
        "requirement": requirement,
    }


def _w8_checkpoint(root: Path, ratio: str) -> dict[str, Any]:
    reconciliation = read_json(Path(root) / W8_RECONCILIATION, "W8 reconciliation")
    selection = {
        str(item["run_id"]): item
        for item in reconciliation.get("selection", {}).get("per_run", [])
    }
    match = None
    identity = None
    for run in reconciliation.get("run_identities", []):
        if run.get("ratio") == ratio and run.get("train_seed") == 0 and run.get("channel_seed") == 0:
            match = selection.get(str(run["run_id"]))
            identity = run
            break
    require(match is not None and identity is not None, f"W8 selected checkpoint is missing for {ratio}")
    recovered = match["independently_reconstructed"]
    published = match["runner_published"]
    return {
        "state": "FROZEN",
        "kind": "w8_selected_checkpoint",
        "reconciliation": artifact_record(root, W8_RECONCILIATION),
        "run_id": str(identity["run_id"]),
        "run_directory": str(identity["run_directory"]),
        "ratio": ratio,
        "train_seed": 0,
        "channel_seed": 0,
        "k": int(identity["k"]),
        "w8_config_hash": str(identity["config_hash"]),
        "w8_protocol_config_hash": str(identity["protocol_config_hash"]),
        "checkpoint_id": str(recovered["checkpoint_id"]),
        "epoch": int(recovered["epoch"]),
        "w8_result_id": str(published["result_id"]),
        "w8_result_file_sha256": str(published["result_file_sha256"]),
        "checkpoint_path": f"{W8_CAMPAIGN_ROOT}/{identity['run_directory']}/checkpoints/epoch-{int(recovered['epoch']):04d}.pt",
        "validation_selection_correct": int(recovered["n_correct"]),
        "validation_selection_total": int(recovered["n_total"]),
    }


def _er2_checkpoint(root: Path) -> dict[str, Any]:
    selected = read_json(Path(root) / ER2_SELECTED_CHECKPOINT, "ER-2 selected checkpoint")
    return {
        "state": "FROZEN",
        "kind": "er2_selected_checkpoint",
        "selection": artifact_record(root, ER2_SELECTED_CHECKPOINT),
        "completion": artifact_record(root, ER2_COMPLETION, required=False),
        "authority_id": str(selected["authority_id"]),
        "selection_id": str(selected["selection_id"]),
        "checkpoint_path": str(selected["checkpoint_path"]),
        "checkpoint_sha256": str(selected["checkpoint_sha256"]),
        "selected_epoch": int(selected["selected_epoch"]),
        "task_head_identity": str(selected["task_head_identity"]),
        "train_seed": 0,
        "channel_seed": 0,
    }


def _er9_cell(root: Path) -> dict[str, Any]:
    closeout = read_json(Path(root) / ER9_PRODUCTION_CLOSEOUT, "ER-9 production closeout")
    cell = None
    for record in closeout["seed_cells"]:
        if (record["train_seed"], record["channel_seed"]) == (0, 0):
            cell = record
    require(cell is not None, "ER-9 production cell 0/0 is missing")
    stage2 = read_json(Path(root) / ER9_STAGE2_SELECTION, "ER-9 Stage-2 selection")
    validation_path = Path(root) / str(cell["validation_path"])
    require(
        validation_path.is_file()
        and hashlib.sha256(validation_path.read_bytes()).hexdigest() == str(cell["validation_sha256"]),
        "ER-9 final-validation bytes differ from the closeout binding",
    )
    validation = read_json(validation_path, "ER-9 final validation")
    per_snr_phy = [
        {
            "snr_db": float(point["snr_db"]),
            "modulation": str(point["selected"]["modulation"]),
            "ldpc_rate": str(point["selected"]["ldpc_rate"]),
            "n_correct": int(point["selected"]["n_correct"]),
            "n_total": int(point["selected"]["n_total"]),
        }
        for point in validation["points"]
    ]
    require(len(per_snr_phy) == 21, "ER-9 final validation does not cover 21 SNRs")  # literal-ok: frozen SNR grid cardinality
    return {
        "state": "FROZEN",
        "kind": "er9_production_cell",
        "closeout": artifact_record(root, ER9_PRODUCTION_CLOSEOUT),
        "stage2_selection": artifact_record(root, ER9_STAGE2_SELECTION),
        "validation": artifact_record(root, str(cell["validation_path"])),
        "selected_pair": dict(stage2["selected_pair"]),
        "train_seed": 0,
        "channel_seed": 0,
        "runtime_root": str(cell["runtime_root"]),
        "terminal_path": str(cell["terminal_path"]),
        "terminal_sha256": str(cell["terminal_sha256"]),
        "selected_epoch": int(cell["selected_epoch"]),
        "selected_checkpoint_sha256": str(cell["selected_checkpoint_sha256"]),
        "validation_path": str(cell["validation_path"]),
        "validation_sha256": str(cell["validation_sha256"]),
        "per_snr_phy": per_snr_phy,
        "per_snr_phy_digest": canonical_sha256({"per_snr_phy": per_snr_phy}),
    }


def _papr_checkpoint(root: Path, *, verify_runtime: bool = True) -> dict[str, Any]:
    path = Path(root) / PAPR_SELECTED_CHECKPOINT
    completion_path = Path(root) / PAPR_COMPLETION
    authority_path = Path(root) / PAPR_AUTHORITY
    if not (
        path.is_file()
        and not path.is_symlink()
        and completion_path.is_file()
        and not completion_path.is_symlink()
        and authority_path.is_file()
        and not authority_path.is_symlink()
    ):
        return _pending(
            "papr_training",
            PAPR_SELECTED_CHECKPOINT,
            "one authorized fresh-initialization r_1_6/cell(0,0) PAPR-constrained training lifecycle",
        )
    selected = read_json(path, "PAPR selected checkpoint")
    completion = read_json(completion_path, "PAPR training completion")
    authority = read_json(authority_path, "PAPR training authority")
    # This is deliberately the same result-independent authority boundary used
    # by the PAPR launcher.  W10 binding resolution must not admit a malformed
    # projected config/protocol and wait for a worker terminal chain to expose
    # it later.  ``verify_papr_authority`` does not require the worker runtime.
    from training.papr_lifecycle import verify_papr_authority

    verify_papr_authority(root, authority_path=authority_path)
    authority_body = dict(authority)
    authority_id = authority_body.pop("authority_id", None)
    require(
        authority_id == "paprtrainingauth-" + canonical_sha256(authority_body),
        "PAPR authority ID differs",
    )
    selected_body = dict(selected)
    selected_id = selected_body.pop("selection_id", None)
    require(
        selected_id == "paprselected-" + canonical_sha256(selected_body),
        "PAPR selected checkpoint ID differs",
    )
    completion_body = dict(completion)
    completion_id = completion_body.pop("completion_id", None)
    require(
        completion_id == "paprcompletion-" + canonical_sha256(completion_body),
        "PAPR completion ID differs",
    )
    require(
        selected.get("authority_id") == authority.get("authority_id")
        and completion.get("authority_id") == authority.get("authority_id")
        and completion.get("selection_id") == selected.get("selection_id")
        and completion.get("checkpoint_id") == selected.get("checkpoint_id")
        and completion.get("checkpoint_path") == selected.get("checkpoint_path"),
        "PAPR committed evidence cross-binding differs",
    )
    require(
        selected.get("config_hash") == authority.get("config_hash")
        and completion.get("config_hash") == authority.get("config_hash")
        and selected.get("protocol_config_hash") == authority.get("protocol_config_hash")
        and completion.get("protocol_config_hash") == authority.get("protocol_config_hash"),
        "PAPR committed configuration binding differs",
    )
    require(
        selected.get("authority_sha256") == file_sha256(root, PAPR_AUTHORITY, required=True)
        and completion.get("authority_sha256") == file_sha256(root, PAPR_AUTHORITY, required=True),
        "PAPR committed authority digest differs",
    )
    terminal = None
    if verify_runtime:
        from training.papr_lifecycle import verify_papr_terminal

        terminal = verify_papr_terminal(root, authority_path=authority_path)
    require(
        float(selected["papr_cap_db"]) == float(_parameter("w10_papr_cap_db")),
        "PAPR selected cap differs from the frozen cap",
    )
    require(completion["papr_cap_db"] == selected["papr_cap_db"], "PAPR completion cap differs")
    require(completion["papr_cap_compliant"] is True, "PAPR completion does not certify cap compliance")
    require(completion["training_runs"] == 1 and completion["test_access"] == 0, "PAPR completion boundary differs")
    if terminal is not None:
        require(selected["selection_id"] == terminal["selection_id"], "PAPR binding selection differs from the terminal")
        require(selected["checkpoint_id"] == terminal["checkpoint_id"], "PAPR binding checkpoint differs from the terminal")
        require(int(selected["selected_epoch"]) == int(terminal["selected_epoch"]), "PAPR binding epoch differs from the terminal")
    return {
        "state": "FROZEN",
        "kind": "papr_selected_checkpoint",
        "selection": artifact_record(root, PAPR_SELECTED_CHECKPOINT),
        "completion": artifact_record(root, PAPR_COMPLETION),
        "authority_id": str(selected["authority_id"]),
        "authority_sha256": file_sha256(root, PAPR_AUTHORITY, required=True),
        "selection_id": str(selected["selection_id"]),
        "completion_id": str(completion["completion_id"]),
        "config_hash": str(selected["config_hash"]),
        "protocol_config_hash": str(selected["protocol_config_hash"]),
        # One explicit schema: ``checkpoint_id`` is the SHA-256 of the checkpoint
        # bytes and ``checkpoint_path`` the repository-relative location, exactly
        # what the learned dispatch route consumes.
        "checkpoint_id": str(selected["checkpoint_id"]),
        "checkpoint_path": str(selected["checkpoint_path"]),
        "checkpoint_bytes": int(selected["checkpoint_bytes"]),
        "epoch": int(selected["selected_epoch"]),
        "selected_epoch": int(selected["selected_epoch"]),
        "papr_cap_db": float(selected["papr_cap_db"]),
        "papr_cap_compliant": True,
        "papr_max_observed_db": float(completion["papr_max_observed_db"]),
        "train_seed": int(selected["train_seed"]),
        "channel_seed": int(selected["channel_seed"]),
    }


def _parameter(name: str) -> Any:
    from config.params import get

    return get(f"evaluation.{name}")


def _classical_selection(root: Path, mode: str, ratio: str) -> dict[str, Any]:
    state = read_json(Path(root) / PASS_TWO_STATE, "G8 pass-two state")
    require(state.get("selection_passes") == [1, 2], "pass-two state is not the two-pass closeout")
    calls = [call for call in state["calls"] if call.get("mode") == mode and call.get("ratio") == ratio and call.get("dataset") == "imagenette160"]
    require(len(calls) == 1, f"pass-two selection call is missing or duplicated for {mode}/{ratio}")
    call = calls[0]
    selections = [
        {
            "snr_db": float(point["snr_db"]),
            "authority_candidate_id": str(point["authority_candidate_id"]),
            "tie_break_applied": bool(point["tie_break_applied"]),
        }
        for point in call["per_snr"]
    ]
    require(len(selections) == 21, f"pass-two selection does not cover 21 SNRs for {mode}/{ratio}")  # literal-ok: frozen SNR grid cardinality
    require(all(item["authority_candidate_id"] and str(item["authority_candidate_id"]).startswith("cand-") for item in selections), "pass-two selection carries an empty candidate")
    return {
        "state": "FROZEN",
        "kind": "classical_pass_two",
        "pass_two_state": artifact_record(root, PASS_TWO_STATE),
        "candidate_authority": artifact_record(root, CANDIDATE_AUTHORITY),
        "mode": mode,
        "ratio": ratio,
        "selections": selections,
        "selections_digest": canonical_sha256({"selections": selections}),
    }


def _br16_selection(root: Path) -> dict[str, Any]:
    closeout = read_json(Path(root) / G8_CLOSEOUT, "G8 closeout")
    freeze = closeout["br16_h2_validation_freeze"]
    require(freeze["system"] == "classical_fixed_mcs" and freeze["ratio"] == "r_1_6", "BR-16 freeze differs")
    return {
        "state": "FROZEN",
        "kind": "br16_fixed_mcs",
        "closeout": artifact_record(root, G8_CLOSEOUT),
        "fixed_configuration": dict(freeze["fixed_configuration"]),
        "window": {
            "low_snr_db": float(freeze["low_snr_db"]),
            "high_snr_db": float(freeze["high_snr_db"]),
            "classical_drop_pp": float(freeze["classical_drop_pp"]),
        },
        "ratio": "r_1_6",
    }


def _pending_selection(root: Path, relative: str, kind: str, requirement: str) -> dict[str, Any]:
    path = Path(root) / relative
    from evaluation.w10_selections import verify_selection_artifact

    selection_kind = "jpeg_secondary" if kind == "w10_jpeg_validation_selection" else "er12_label_bound"
    provenance: dict[str, Any] | None = None
    if kind == "w10_jpeg_validation_selection":
        raw_present = path.exists() or path.is_symlink()
        carrier_present = (Path(root) / JPEG_CARRIER_PATH).exists() or (Path(root) / JPEG_CARRIER_PATH).is_symlink()
        descriptor_present = (Path(root) / JPEG_CARRIER_DESCRIPTOR_PATH).exists() or (Path(root) / JPEG_CARRIER_DESCRIPTOR_PATH).is_symlink()
        if not raw_present and not carrier_present and not descriptor_present:
            return _pending(kind, relative, requirement)
        try:
            loaded = load_jpeg_selection_artifact(root)
        except JpegCarrierHold as exc:
            raise RuntimeError(str(exc)) from None
        value = verify_selection_artifact(root, selection_kind, loaded.value, path=path)
        provenance = dict(loaded.provenance)
    else:
        if not path.is_file() or path.is_symlink():
            return _pending(kind, relative, requirement)
        value = verify_selection_artifact(root, selection_kind, read_json(path, relative), path=path)
    selections = value["selections"]
    require(isinstance(selections, list) and len(selections) == 21, f"{relative} does not select 21 SNRs")  # literal-ok: frozen SNR grid cardinality
    if provenance is None:
        artifact = artifact_record(root, relative)
    else:
        artifact = {
            "path": relative,
            "sha256": provenance["raw_sha256"],
            "present": True,
            "artifact_role": value.get("artifact_role"),
            "artifact_id": value.get("selection_id"),
            **provenance,
        }
    return {
        "state": "FROZEN",
        "kind": kind,
        "artifact": artifact,
        "selection_id": str(value["selection_id"]),
        "contract_sha256": str(value["contract_sha256"]),
        "source_epoch": dict(value["source_epoch"]),
        "raw_sha256": provenance["raw_sha256"] if provenance is not None else artifact["sha256"],
        "carrier_path": provenance["carrier_path"] if provenance is not None else None,
        "carrier_sha256": provenance["carrier_sha256"] if provenance is not None else None,
        "carrier_descriptor_path": provenance["carrier_descriptor_path"] if provenance is not None else None,
        "carrier_descriptor_id": provenance["carrier_descriptor_id"] if provenance is not None else None,
        "carrier_descriptor_sha256": provenance["carrier_descriptor_sha256"] if provenance is not None else None,
        "selection_digest": provenance["selection_digest"] if provenance is not None else canonical_sha256(value),
        "selections": selections,
        "selections_digest": canonical_sha256({"selections": [dict(item) for item in selections]}),
    }


def _scorer(root: Path, name: str) -> dict[str, Any]:
    if name == "br12_artifact_finetuned_reference_classifier":
        freeze = read_json(Path(root) / BR12_FREEZE, "BR-12 classifier freeze")
        return {
            "state": "FROZEN",
            "kind": "br12_artifact_finetuned_reference_classifier",
            "freeze": artifact_record(root, BR12_FREEZE),
            "checkpoint_id": str(freeze["checkpoint_id"]),
            "checkpoint_file_sha256": str(freeze["checkpoint_file_sha256"]),
            "checkpoint_external_artifact": freeze.get("checkpoint_external_artifact"),
            "classifier_variant": str(freeze["classifier_variant"]),
            "scorer_identity": str(freeze["scorer_identity"]),
            "variant": "artifact_finetuned",
        }
    if name == "frozen_g1_reference_classifier":
        best = read_json(Path(root) / G1_BEST_CHECKPOINT, "G-1 best checkpoint")
        return {
            "state": "FROZEN",
            "kind": "frozen_g1_reference_classifier",
            "best_checkpoint": artifact_record(root, G1_BEST_CHECKPOINT),
            "checkpoint_id": str(best["best_checkpoint_id"]),
            "variant": "clean",
        }
    return {
        "state": "FROZEN",
        "kind": name,
        "variant": "own_task_head" if name == "own_task_head" else "predicted_label",
    }


def _checkpoint(root: Path, source: str, *, verify_runtime: bool = True) -> dict[str, Any]:
    if source.startswith("w8_reconciliation:"):
        _, ratio, _cell = source.split(":")
        return _w8_checkpoint(root, ratio)
    if source == "er2_selected_checkpoint_v4":
        return _er2_checkpoint(root)
    if source.startswith("er9_production_closeout:"):
        return _er9_cell(root)
    if source == "pending_papr_training":
        return _papr_checkpoint(root, verify_runtime=verify_runtime)
    if source == "not_applicable_untrained_codec":
        return {"state": "FROZEN", "kind": "not_applicable_untrained_codec"}
    raise RuntimeError(f"unknown W10 checkpoint source: {source}")


def _selection(root: Path, source: str) -> dict[str, Any]:
    if source == "not_applicable_no_selection":
        return {"state": "FROZEN", "kind": "not_applicable_no_selection"}
    if source.startswith("pass_two_state:"):
        _, mode, ratio = source.split(":")
        return _classical_selection(root, mode, ratio)
    if source.startswith("br16_fixed_configuration:"):
        return _br16_selection(root)
    if source == "w10_jpeg_validation_selection":
        return _pending_selection(
            root,
            JPEG_SELECTION,
            "w10_jpeg_validation_selection",
            "validation-only JPEG quality x LDPC x modulation selection frozen before W10",
        )
    if source == "w10_er12_validation_selection":
        return _pending_selection(
            root,
            ER12_SELECTION,
            "w10_er12_validation_selection",
            "validation-only label-bound PHY selection frozen before W10",
        )
    if source.startswith("er9_stage2_selection_v4"):
        return {"state": "FROZEN", "kind": "er9_stage2_selection_v4"}
    raise RuntimeError(f"unknown W10 selection source: {source}")


def resolve_binding(
    root: Path, entry: W10ScopeEntry, *, verify_runtime: bool = True
) -> dict[str, Any]:
    """The complete resolved scientific binding for one scope entry."""

    binding = {
        "scope": entry.identity(),
        "checkpoint": _checkpoint(root, entry.checkpoint_source, verify_runtime=verify_runtime),
        "scorer": _scorer(root, entry.scorer_source),
        "selection": _selection(root, entry.selection_source),
    }
    binding["binding_id"] = "w10binding-" + canonical_sha256(binding)
    return binding


def resolve_scope_bindings(root: Path, *, verify_runtime: bool = True) -> list[dict[str, Any]]:
    return [resolve_binding(root, entry, verify_runtime=verify_runtime) for entry in SCOPE]


def pending_states(bindings: list[Mapping[str, Any]]) -> list[str]:
    pending: list[str] = []
    for binding in bindings:
        for section in ("checkpoint", "selection"):
            state = str(binding[section].get("state", ""))
            if state.startswith(PENDING_PREFIX):
                pending.append(f"{binding['scope']['system']}/{binding['scope']['bw_ratio']}/{section}:{state}")
    return pending


def binding_for(root: Path, system: str, bw_ratio: str) -> dict[str, Any]:
    for entry in SCOPE:
        if entry.system == system and entry.bw_ratio == bw_ratio:
            return resolve_binding(root, entry)
    raise KeyError(f"no W10 scope entry for ({system!r}, {bw_ratio!r})")


__all__ = [
    "BR12_FREEZE",
    "CANDIDATE_AUTHORITY",
    "ER12_SELECTION",
    "ER2_COMPLETION",
    "ER2_SELECTED_CHECKPOINT",
    "ER9_PRODUCTION_CLOSEOUT",
    "ER9_STAGE2_SELECTION",
    "G1_BEST_CHECKPOINT",
    "G8_CLOSEOUT",
    "JPEG_SELECTION",
    "PAPR_AUTHORITY",
    "PAPR_COMPLETION",
    "PAPR_SELECTED_CHECKPOINT",
    "PASS_TWO_STATE",
    "W8_CAMPAIGN_ROOT",
    "W8_RECONCILIATION",
    "artifact_record",
    "binding_for",
    "file_sha256",
    "pending_states",
    "resolve_binding",
    "resolve_scope_bindings",
]
