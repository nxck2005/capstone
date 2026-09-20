#!/usr/bin/env python3
"""Run CI without pretending worker-local transactional runtimes are portable.

The scientific terminal verifiers remain the worker-side authority.  Hosted CI
replaces only their exact runtime-consuming commands with checks over the
content-addressed evidence that is actually committed to Git.  Every other
quality-gate command is preserved verbatim.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

import run_quality_gate as gate  # noqa: E402
from config.params import get  # noqa: E402
from evaluation.downstream_v4 import (  # noqa: E402
    ER2_RUNTIME_ROOT,
    FINAL_PAIR,
    PRODUCTION_CELLS,
    read_json,
    sha256_file,
    validate_final_er9_cell,
    verify_er2_authority,
    verify_production_authority,
)
from evaluation.g10_protocol import verify_identified  # noqa: E402
from training.deterministic_core import canonical_sha256  # noqa: E402
from verify_er2_randomized import verify_audit, verify_validation  # noqa: E402
from verify_g11 import verify_authority as verify_g11_authority  # noqa: E402
from training.papr_constrained import (  # noqa: E402
    PAPR_AUTHORITY_PATH,
    PAPR_COMPLETION_PATH,
    PAPR_SELECTED_CHECKPOINT_PATH,
)


ER9_CLOSEOUT = REPO / "results/learned/er9/er9_production_closeout_v4.json"
ER2_ROOT = REPO / "results/learned/er2_randomized"
ER2_COMPLETION = ER2_ROOT / "er2_randomized_completion_v4.json"
G11_ROOT = REPO / "results/learned/g11"
G11_AUTHORITY = G11_ROOT / "g11_execution_authorization_v4.json"
G11_TERMINAL = G11_ROOT / "g11_terminal_closeout.json"
G10_MANIFEST = REPO / "results/learned/w9/g10_runtime_manifest.json"
W10_AUTHORITY = REPO / "results/learned/w10/w10_rehearsal_authorization.json"
W10_CLOSEOUT = REPO / "results/learned/w10/w10_rehearsal_closeout.json"
PAPR_AUTHORITY = REPO / PAPR_AUTHORITY_PATH
PAPR_SELECTED = REPO / PAPR_SELECTED_CHECKPOINT_PATH
PAPR_COMPLETION = REPO / PAPR_COMPLETION_PATH


class HostedEvidenceHold(RuntimeError):
    """Committed evidence or hosted routing differs from the frozen contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HostedEvidenceHold(message)


def full_sha256(value: Any, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and value != "0" * 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} is not a full non-sentinel SHA-256",
    )
    return value


def content_address(value: Mapping[str, Any], field: str, prefix: str, label: str) -> None:
    body = dict(value)
    identifier = body.pop(field, None)
    require(identifier == prefix + canonical_sha256(body), f"{label} content address differs")


def verify_er9_published() -> None:
    """Authenticate only immutable ER-9 evidence available in a clean clone."""

    authority = verify_production_authority(REPO)
    closeout = read_json(ER9_CLOSEOUT, "ER-9 production closeout")
    content_address(closeout, "closeout_id", "er9productionv4closeout-", "ER-9 closeout")
    require(closeout.get("schema_version") == 1, "ER-9 closeout schema differs")
    require(closeout.get("artifact_role") == "ER9_PRODUCTION_V4_CLOSEOUT", "ER-9 closeout role differs")
    require(closeout.get("status") == "THREE_REQUIRED_CELLS_COMPLETE_VALIDATION_ONLY", "ER-9 closeout status differs")
    require(closeout.get("authority_id") == authority["authority_id"], "ER-9 closeout authority differs")
    require(closeout.get("source_commit") == authority["source_commit"], "ER-9 closeout source differs")
    require(closeout.get("source_manifest_id") == authority["source_binding"]["manifest_id"], "ER-9 closeout manifest differs")
    require(closeout.get("selected_pair") == FINAL_PAIR, "ER-9 closeout pair differs")
    require(closeout.get("training_count") == 3, "ER-9 closeout training count differs")
    require(closeout.get("stage1_promotion_count") == 0, "ER-9 closeout promoted Stage 1")
    require(closeout.get("best_seed_selection") is False, "ER-9 closeout selected a best seed")
    records = closeout.get("seed_cells")
    require(
        isinstance(records, list)
        and [(item.get("train_seed"), item.get("channel_seed")) for item in records] == list(PRODUCTION_CELLS),
        "ER-9 closeout cell set/order differs",
    )
    cross_cell_ids: list[list[str]] = []
    for cell, record in zip(PRODUCTION_CELLS, records, strict=True):
        runtime_root = f"checkpoints/er9_production_pascal_v4/train{cell[0]}_channel{cell[1]}"
        terminal_path = runtime_root + "/run_terminal.json"
        validation_path = f"results/learned/er9/final_validation/train{cell[0]}_channel{cell[1]}.json"
        require(record.get("candidate") == FINAL_PAIR, f"ER-9 closeout candidate differs at {cell}")
        require(record.get("runtime_root") == runtime_root, f"ER-9 runtime identity differs at {cell}")
        require(record.get("terminal_path") == terminal_path, f"ER-9 terminal path differs at {cell}")
        full_sha256(record.get("terminal_sha256"), f"ER-9 terminal {cell}")
        require(record.get("promoted_stage1") is False, f"ER-9 Stage-1 promotion differs at {cell}")
        require(record.get("test_access") == 0, f"ER-9 closeout test access differs at {cell}")
        epoch = record.get("selected_epoch")
        require(isinstance(epoch, int) and not isinstance(epoch, bool) and 0 <= epoch < 100, f"ER-9 selected epoch differs at {cell}")
        checkpoint_sha = full_sha256(record.get("selected_checkpoint_sha256"), f"ER-9 checkpoint {cell}")
        require(record.get("validation_path") == validation_path, f"ER-9 validation path differs at {cell}")
        validation_file = REPO / validation_path
        require(sha256_file(validation_file) == record.get("validation_sha256"), f"ER-9 validation SHA differs at {cell}")
        validation = read_json(validation_file, f"ER-9 validation {cell}")
        content_address(validation, "validation_id", "er9productionvalidation-", f"ER-9 validation {cell}")
        validate_final_er9_cell(validation, cell=cell)
        require(validation.get("checkpoint") == {
            "epoch": epoch,
            "path": f"{runtime_root}/epochs/epoch-{epoch:04d}/checkpoint.pt",
            "sha256": checkpoint_sha,
        }, f"ER-9 selected validation checkpoint differs at {cell}")
        ids = [str(row["stable_sample_id"]) for row in validation["points"][0]["per_image"]]
        cross_cell_ids.append(ids)
    require(all(ids == cross_cell_ids[0] for ids in cross_cell_ids), "ER-9 cross-cell stable-ID order differs")
    require(closeout.get("test") == "SEALED" and closeout.get("test_access") == 0, "ER-9 closeout crossed test boundary")
    print("ER-9 production v4 published-evidence verifier PASS: three closeout cells and full validation; worker runtime not inspected")


def verify_er2_published() -> None:
    """Authenticate immutable ER-2 evidence without worker checkpoint bytes."""

    authority = verify_er2_authority(REPO)
    selected_path = ER2_ROOT / "er2_selected_checkpoint_v4.json"
    audit_path = ER2_ROOT / "er2_snr_assignment_audit_v4.json"
    validation_path = ER2_ROOT / "er2_randomized_validation_v4.json"
    selected = read_json(selected_path, "ER-2 selected checkpoint")
    content_address(selected, "selection_id", "er2selectedv4-", "ER-2 selection")
    require(selected.get("artifact_role") == "ER2_RANDOMIZED_SELECTED_CHECKPOINT_V4", "ER-2 selection role differs")
    require(selected.get("authority_id") == authority["authority_id"] and selected.get("source_commit") == authority["source_commit"], "ER-2 selection authority/source differs")
    require((selected.get("train_seed"), selected.get("channel_seed")) == (0, 0), "ER-2 selection seed differs")
    epoch = selected.get("selected_epoch")
    require(isinstance(epoch, int) and not isinstance(epoch, bool) and 0 <= epoch < 100, "ER-2 selected epoch differs")
    checkpoint_sha = full_sha256(selected.get("checkpoint_sha256"), "ER-2 selected checkpoint")
    require(selected.get("checkpoint_path") == f"{ER2_RUNTIME_ROOT}/epochs/epoch-{epoch:04d}/checkpoint.pt", "ER-2 selected checkpoint path differs")
    metric = selected.get("selection_metric")
    metric_value = metric.get("value") if isinstance(metric, Mapping) else None
    require(
        isinstance(metric, Mapping)
        and metric.get("metric") == "validation_n_correct"
        and metric.get("mode") == "max"
        and isinstance(metric_value, int)
        and not isinstance(metric_value, bool)
        and 0 <= metric_value <= int(get("datasets.imagenette160.val_images"))
        and selected.get("tie_break") == "earliest_epoch",
        "ER-2 checkpoint-selection rule differs",
    )
    require(str(selected.get("task_head_identity", "")).startswith("er9taskhead-") and len(str(selected["task_head_identity"])) == 76, "ER-2 task-head identity differs")
    require(selected.get("test") == "SEALED" and selected.get("test_access") == 0, "ER-2 selection crossed test boundary")
    # Rebuilding the keyed assignment digest requires the worker-local
    # Imagenette training corpus.  The unchanged terminal verifier performs
    # that strong recomputation on Confessor; hosted CI authenticates the
    # committed audit itself and all arithmetic that can be derived from it.
    audit = verify_audit(audit_path, authority, recompute=False)
    require(audit.get("schema_version") == 2, "ER-2 assignment-audit schema differs")
    require(audit.get("system") == "learned_snr_randomised", "ER-2 assignment-audit system differs")
    require(audit.get("rng_purpose") == "er2_snr_randomised_v1", "ER-2 assignment-audit RNG purpose differs")
    require(audit.get("identity_fields") == [
        "dataset_version", "split_manifest_hash", "stable_sample_id", "train_seed", "epoch",
    ], "ER-2 assignment-audit identity fields differ")
    require(audit.get("config_hash") == authority.get("config_hash"), "ER-2 assignment-audit config differs")
    full_sha256(audit.get("assignment_digest"), "ER-2 assignment digest")
    sample_count = audit.get("sample_count")
    epoch_count = audit.get("epoch_count")
    domain_keys = [str(value) for value in authority["randomized_snr_domain"]]
    require(sample_count == 8469 and epoch_count == 100, "ER-2 assignment-audit dimensions differ")
    rows = audit.get("per_epoch_counts")
    require(isinstance(rows, list) and [row.get("epoch") for row in rows] == list(range(epoch_count)), "ER-2 per-epoch assignment order differs")
    aggregate = {key: 0 for key in domain_keys}
    for row in rows:
        counts = row.get("counts")
        require(isinstance(counts, Mapping) and set(counts) == set(domain_keys), "ER-2 per-epoch assignment domain differs")
        require(all(isinstance(counts[key], int) and not isinstance(counts[key], bool) and counts[key] >= 0 for key in domain_keys), "ER-2 per-epoch assignment count is invalid")
        require(sum(counts[key] for key in domain_keys) == sample_count, "ER-2 per-epoch assignment total differs")
        for key in domain_keys:
            aggregate[key] += counts[key]
    require(audit.get("global_counts") == aggregate, "ER-2 global assignment counts differ")
    require(sum(aggregate.values()) == audit.get("assignment_count"), "ER-2 global assignment total differs")
    require(audit.get("runtime_root") == ER2_RUNTIME_ROOT, "ER-2 assignment-audit runtime differs")
    validation = verify_validation(validation_path, authority, selected)
    require(all(row.get("checkpoint_id") == checkpoint_sha for curve in validation["curves"] for row in curve["outcomes"]), "ER-2 validation checkpoint binding differs")
    completion = read_json(ER2_COMPLETION, "ER-2 completion")
    content_address(completion, "completion_id", "er2completionv4-", "ER-2 completion")
    require(completion.get("schema_version") == 2 and completion.get("artifact_role") == "ER2_RANDOMIZED_V4_COMPLETION", "ER-2 completion role/schema differs")
    require(completion.get("status") == "ONE_RANDOMIZED_RUN_COMPLETE_VALIDATION_ONLY", "ER-2 completion status differs")
    require(completion.get("authority_id") == authority["authority_id"] and completion.get("source_commit") == authority["source_commit"], "ER-2 completion authority/source differs")
    require(completion.get("runtime_root") == ER2_RUNTIME_ROOT and completion.get("scientific_training_run_count") == 1, "ER-2 completion scope differs")
    expected_records = {
        "terminal": (f"{ER2_RUNTIME_ROOT}/run_terminal.json", None),
        "selected_checkpoint": (str(selected_path.relative_to(REPO)), sha256_file(selected_path)),
        "assignment_audit": (str(audit_path.relative_to(REPO)), sha256_file(audit_path)),
        "validation": (str(validation_path.relative_to(REPO)), sha256_file(validation_path)),
    }
    for field, (path, digest) in expected_records.items():
        record = completion.get(field)
        require(isinstance(record, Mapping) and record.get("path") == path, f"ER-2 completion {field} path differs")
        full_sha256(record.get("sha256"), f"ER-2 completion {field}")
        if digest is not None:
            require(record["sha256"] == digest, f"ER-2 completion {field} SHA differs")
    require(completion.get("test") == "SEALED" and completion.get("test_access") == 0, "ER-2 completion crossed test boundary")
    print("randomized ER-2 v4 published-evidence verifier PASS: one run and full validation; worker runtime not inspected")


def _verify_g11_published_legacy() -> None:
    """Authenticate committed G11/H4 evidence without worker checkpoint bytes.

    Hosted CI authenticates the frozen G11 authority, the content-addressed
    H4/architecture/terminal artifacts, the three committed ER-9 validation
    bindings, and the 63 learned bindings against the committed G10 runtime
    manifest.  It deliberately does NOT recompute H4 from the worker-local
    per-image G10 cells under the Confessor campaign root; the unchanged
    Confessor verifier remains the strong recomputation authority.
    """

    authority = verify_g11_authority()
    h4_path = G11_ROOT / "h4_pointwise_precision_v4.json"
    architecture_path = G11_ROOT / "er9_architecture_difference_v4.json"

    h4 = read_json(h4_path, "G11 H4 diagnostic")
    content_address(h4, "diagnostic_id", "g11h4v4-", "G11 H4 diagnostic")
    require(h4.get("artifact_role") == "G11_H4_POINTWISE_PRECISION_DIAGNOSTIC" and h4.get("schema_version") == 2, "G11 H4 role differs")
    require(h4.get("diagnostic_role") == "pointwise_precision_diagnostic_only", "G11 H4 diagnostic role differs")
    require(h4.get("learned_arm") == "ordinary_learned_w8_g10" and h4.get("comparator") == "er9_digital", "G11 H4 arms differ")
    require(h4.get("randomized_er2_as_h4_arm") is False, "randomized ER-2 substituted into H4")
    require(h4.get("cells") == [list(cell) for cell in PRODUCTION_CELLS], "G11 H4 cells differ")
    require(h4.get("bootstrap_unit") == "stable_image_complete_three_cell_trajectory" and h4.get("bootstrap_resamples") == 10000, "G11 bootstrap contract differs")  # literal-ok: AM-97 count
    require(h4.get("reference_pp") == 2.0, "G11 reference differs")  # literal-ok: AM-97 reference
    interpretation = h4.get("interpretation", {})
    require(interpretation.get("pointwise_only") is True and interpretation.get("full_h4_decision_procedure_power_certified") is False, "G11 H4 overclaims power")
    require(interpretation.get("correlated_snr_pointwise_probabilities_may_be_multiplied") is False and interpretation.get("negative_h4_result_excludes_meaningful_advantage") is False, "G11 H4 interpretation differs")
    require(h4.get("test_data_read") is False and h4.get("test_access") == 0, "G11 H4 accessed test")

    learned_bindings = h4.get("input_bindings", {}).get("ordinary_learned_w8_g10")
    require(isinstance(learned_bindings, list) and len(learned_bindings) == 63, "G11 learned binding count differs")  # literal-ok: 3x21 G10 cells
    runtime = read_json(G10_MANIFEST, "G10 runtime manifest")
    verify_identified(runtime, field="runtime_manifest_id", prefix="g10runtime-", label="G10 runtime manifest")
    require(runtime.get("status") == "COMPLETE_MATRIX_READY_FOR_AGGREGATION" and runtime.get("matrix_shape") == {"checkpoints": 3, "snr_points": 21, "cells": 63}, "G10 runtime manifest differs")  # literal-ok: frozen 3x21 matrix
    cells = runtime.get("cells")
    require(isinstance(cells, list) and len(cells) == 63, "G10 runtime manifest cell count differs")  # literal-ok: 3x21 G10 cells
    seen_artifact_ids: set[str] = set()
    for index, (binding, cell) in enumerate(zip(learned_bindings, cells, strict=True)):
        require(isinstance(binding, Mapping) and isinstance(cell, Mapping), "G11 learned binding is malformed")
        full_sha256(binding.get("sha256"), f"G11 learned binding {index}")
        require(binding.get("path") == cell.get("file_path"), f"G11 learned binding path differs at {index}")
        require(binding.get("sha256") == cell.get("file_sha256"), f"G11 learned binding hash differs at {index}")
        require(binding.get("artifact_id") == cell.get("artifact_id"), f"G11 learned binding artifact differs at {index}")
        require(cell.get("cell_index") == index, f"G10 manifest cell order differs at {index}")
        artifact_id = str(cell.get("artifact_id"))
        require(bool(artifact_id) and artifact_id not in seen_artifact_ids, f"G10 manifest cell repeats at {index}")
        seen_artifact_ids.add(artifact_id)

    er9_bindings = h4.get("input_bindings", {}).get("final_production_er9")
    require(isinstance(er9_bindings, list) and len(er9_bindings) == len(PRODUCTION_CELLS), "G11 ER-9 binding count differs")
    for cell, binding in zip(PRODUCTION_CELLS, er9_bindings, strict=True):
        require(isinstance(binding, Mapping), "G11 ER-9 binding is malformed")
        relative = f"results/learned/er9/final_validation/train{cell[0]}_channel{cell[1]}.json"
        require(binding.get("path") == relative, f"G11 ER-9 binding path differs at {cell}")
        full_sha256(binding.get("sha256"), f"G11 ER-9 binding {cell}")
        require(sha256_file(REPO / relative) == binding["sha256"], f"G11 ER-9 binding hash differs at {cell}")
        value = read_json(REPO / relative, f"ER-9 final validation {cell}")
        validate_final_er9_cell(value, cell=cell)
        content_address(value, "validation_id", "er9productionvalidation-", f"ER-9 validation {cell}")
        require(value.get("validation_id") == binding.get("validation_id"), f"G11 ER-9 validation id differs at {cell}")

    architecture = read_json(architecture_path, "G11 architecture audit")
    content_address(architecture, "audit_id", "er9archdiffv2-", "G11 architecture audit")
    require(architecture.get("only_declared_difference") is True and architecture.get("unexpected_difference_paths") == [], "G11 architecture audit is not closed")
    require(architecture.get("observed_interface_differences") == ["channel_interface"], "G11 interface difference differs")
    require(architecture.get("test_access") == 0, "G11 architecture audit accessed test")

    terminal = read_json(G11_TERMINAL, "G11 terminal")
    content_address(terminal, "terminal_id", "g11terminal-", "G11 terminal")
    require(terminal.get("artifact_role") == "G11_TERMINAL_VALIDATION_ONLY_CLOSEOUT" and terminal.get("status") == "G11_GREEN_NO_TEST_ACCESS", "G11 terminal role/status differs")
    require(terminal.get("decision") == "GREEN", "G11 terminal decision differs")
    require(terminal.get("authority_id") == authority["authority_id"], "G11 terminal authority differs")
    require(terminal.get("source_commit") == authority["source_commit"], "G11 terminal source commit differs")
    require(terminal.get("source_manifest_id") == authority["source_manifest"]["manifest_id"], "G11 terminal source manifest differs")
    require(terminal.get("cells") == [list(cell) for cell in PRODUCTION_CELLS], "G11 terminal cells differ")
    require(terminal.get("input_sources") == {"learned": "ordinary_learned_w8_g10", "comparator": "final_production_er9", "randomized_er2": False}, "G11 input arms differ")
    require(terminal.get("am97_semantics") == {"signed_correctness_difference": True, "within_image_three_cell_mean": True, "stable_image_complete_trajectory_bootstrap": True, "bootstrap_resamples": 10000, "quantiles": ["q50", "q95", "q99"], "reference_pp": 2, "pointwise_only": True, "full_h4_power_claim": False}, "G11 AM-97 semantics differ")  # literal-ok: AM-97 constants
    h4_record = terminal.get("h4", {})
    require(h4_record.get("artifact") == str(h4_path.relative_to(REPO)) and h4_record.get("artifact_sha256") == sha256_file(h4_path), "G11 terminal H4 binding differs")
    er9_record = terminal.get("er9", {})
    require(er9_record.get("production_closeout") == str(ER9_CLOSEOUT.relative_to(REPO)) and er9_record.get("production_closeout_sha256") == sha256_file(ER9_CLOSEOUT), "G11 terminal ER-9 closeout binding differs")
    require(er9_record.get("architecture_difference") == str(architecture_path.relative_to(REPO)) and er9_record.get("architecture_difference_sha256") == sha256_file(architecture_path), "G11 terminal architecture binding differs")
    validation_cells = er9_record.get("validation_cells")
    require(isinstance(validation_cells, list) and len(validation_cells) == len(PRODUCTION_CELLS), "G11 terminal ER-9 cell count differs")
    for cell, record in zip(PRODUCTION_CELLS, validation_cells, strict=True):
        relative = f"results/learned/er9/final_validation/train{cell[0]}_channel{cell[1]}.json"
        require(isinstance(record, Mapping) and record.get("path") == relative and sha256_file(REPO / relative) == record.get("sha256"), f"G11 terminal ER-9 cell binding differs at {cell}")
    require(terminal.get("protected_counters") == {"production_er9_training": 3, "randomized_er2_training": 1, "g11": 1, "w10": 0, "learned_test_inference": 0, "model_facing_test_access": 0}, "G11 protected counters differ")
    require(terminal.get("test") == "SEALED" and terminal.get("test_access") == 0, "G11 terminal crossed test boundary")
    print("G11/H4 published-evidence verifier PASS: authority, H4, architecture, terminal, 3 ER-9 and 63 G10 manifest bindings; worker-local G10 per-image bytes not recomputed")


def verify_g11_published() -> None:
    """Use the explicit clean-clone G11 published-evidence verifier."""

    from verify_g11_published import verify_published

    verify_published()


def verify_w10_published() -> None:
    from verify_w10_rehearsal import verify_published

    value = verify_published()
    print(
        "W10 published-evidence verifier PASS: "
        f"authority={value['authority_id']} closeout={value['closeout_id']} "
        f"units={value['unit_count']}; worker-local per-image bytes not recomputed"
    )


def verify_papr_authority_published() -> None:
    """Authenticate a PAPR authority without requiring its worker runtime."""

    from training.papr_lifecycle import verify_papr_authority

    value = verify_papr_authority(REPO, authority_path=PAPR_AUTHORITY, require_live_source=True)
    print(f"PAPR published authority verifier PASS: {value['authority_id']}; worker runtime not inspected")


def verify_papr_published() -> None:
    """Authenticate completed PAPR evidence without recomputing worker facts."""

    from verify_papr_training_published import verify_published

    value = verify_published()
    print(f"PAPR published-evidence verifier PASS: {value['completion_id']}; worker runtime not recomputed")


def verify_w10_authority_published() -> None:
    """Authenticate W10 authority bindings through hosted G11/PAPR paths."""

    verify_papr_published()
    from verify_w10_rehearsal import verify_authority

    value = verify_authority(verify_bindings_runtime=False, verify_g11_runtime=False)
    print(f"W10 published authority verifier PASS: {value['authority_id']}; worker runtime not inspected")


def hosted_commands(profile: str) -> tuple[list[str], ...]:
    """Replace only exact worker-runtime terminal commands, failing closed."""

    commands = gate.profile_commands(profile)
    replaced_er9 = 0
    replaced_er2 = 0
    replaced_g11 = 0
    replaced_w10 = 0
    replaced_papr_authority = 0
    replaced_papr_terminal = 0
    replaced_w10_authority = 0
    result: list[list[str]] = []
    for command in commands:
        tool = Path(command[1]).name if len(command) >= 2 else ""
        arguments = command[2:]
        if tool == "verify_er9_production_v4.py" and arguments == ["--terminal"]:
            require(ER9_CLOSEOUT.is_file() and not ER9_CLOSEOUT.is_symlink(), "ER-9 terminal routing lacks a safe closeout")
            result.append([sys.executable, str(Path(__file__).resolve()), "verify-er9-published"])
            replaced_er9 += 1
        elif tool == "verify_er2_randomized.py" and not arguments:
            require(ER2_COMPLETION.is_file() and not ER2_COMPLETION.is_symlink(), "ER-2 terminal routing lacks a safe completion")
            result.append([sys.executable, str(Path(__file__).resolve()), "verify-er2-published"])
            replaced_er2 += 1
        elif tool == "verify_g11.py" and not arguments:
            require(
                G11_AUTHORITY.is_file() and not G11_AUTHORITY.is_symlink()
                and G11_TERMINAL.is_file() and not G11_TERMINAL.is_symlink(),
                "G11 terminal routing lacks a safe authority/closeout",
            )
            result.append([sys.executable, str(Path(__file__).resolve()), "verify-g11-published"])
            replaced_g11 += 1
        elif tool == "verify_w10_rehearsal.py" and arguments == ["--terminal"]:
            require(
                W10_AUTHORITY.is_file() and not W10_AUTHORITY.is_symlink()
                and W10_CLOSEOUT.is_file() and not W10_CLOSEOUT.is_symlink(),
                "W10 terminal routing lacks a safe authority/closeout",
            )
            result.append([sys.executable, str(Path(__file__).resolve()), "verify-w10-published"])
            replaced_w10 += 1
        elif tool == "verify_w10_rehearsal.py" and not arguments:
            require(
                W10_AUTHORITY.is_file() and not W10_AUTHORITY.is_symlink()
                and PAPR_AUTHORITY.is_file() and not PAPR_AUTHORITY.is_symlink()
                and PAPR_SELECTED.is_file() and not PAPR_SELECTED.is_symlink()
                and PAPR_COMPLETION.is_file() and not PAPR_COMPLETION.is_symlink(),
                "W10 authority routing lacks a completed PAPR binding",
            )
            result.append([sys.executable, str(Path(__file__).resolve()), "verify-w10-authority-published"])
            replaced_w10_authority += 1
        elif tool == "verify_papr_training_authorization.py" and arguments == ["--terminal"]:
            require(
                PAPR_AUTHORITY.is_file() and not PAPR_AUTHORITY.is_symlink()
                and PAPR_SELECTED.is_file() and not PAPR_SELECTED.is_symlink()
                and PAPR_COMPLETION.is_file() and not PAPR_COMPLETION.is_symlink(),
                "PAPR terminal routing lacks a safe selected/completion pair",
            )
            result.append([sys.executable, str(Path(__file__).resolve()), "verify-papr-published"])
            replaced_papr_terminal += 1
        elif tool == "verify_papr_training_authorization.py" and not arguments:
            require(PAPR_AUTHORITY.is_file() and not PAPR_AUTHORITY.is_symlink(), "PAPR authority routing lacks a safe authority")
            result.append([sys.executable, str(Path(__file__).resolve()), "verify-papr-authority-published"])
            replaced_papr_authority += 1
        else:
            result.append(command)
    expected_er9 = int(ER9_CLOSEOUT.is_file() and not ER9_CLOSEOUT.is_symlink())
    expected_er2 = int(ER2_COMPLETION.is_file() and not ER2_COMPLETION.is_symlink())
    expected_g11 = int(
        G11_AUTHORITY.is_file() and not G11_AUTHORITY.is_symlink()
        and G11_TERMINAL.is_file() and not G11_TERMINAL.is_symlink()
    )
    expected_w10 = int(
        W10_AUTHORITY.is_file() and not W10_AUTHORITY.is_symlink()
        and W10_CLOSEOUT.is_file() and not W10_CLOSEOUT.is_symlink()
    )
    papr_authority_present = PAPR_AUTHORITY.is_file() and not PAPR_AUTHORITY.is_symlink()
    papr_terminal_present = int(
        PAPR_AUTHORITY.is_file() and not PAPR_AUTHORITY.is_symlink()
        and PAPR_SELECTED.is_file() and not PAPR_SELECTED.is_symlink()
        and PAPR_COMPLETION.is_file() and not PAPR_COMPLETION.is_symlink()
    )
    expected_papr_authority = int(papr_authority_present and not papr_terminal_present)
    expected_papr_terminal = papr_terminal_present
    expected_w10_authority = int(
        W10_AUTHORITY.is_file() and not W10_AUTHORITY.is_symlink()
        and PAPR_AUTHORITY.is_file() and not PAPR_AUTHORITY.is_symlink()
        and PAPR_SELECTED.is_file() and not PAPR_SELECTED.is_symlink()
        and PAPR_COMPLETION.is_file() and not PAPR_COMPLETION.is_symlink()
        and not (W10_CLOSEOUT.is_file() and not W10_CLOSEOUT.is_symlink())
    )
    require(replaced_er9 == expected_er9, "hosted ER-9 runtime-command replacement count differs")
    require(replaced_er2 == expected_er2, "hosted ER-2 runtime-command replacement count differs")
    require(replaced_g11 == expected_g11, "hosted G-11 runtime-command replacement count differs")
    require(replaced_w10 == expected_w10, "hosted W10 runtime-command replacement count differs")
    require(replaced_papr_authority == expected_papr_authority, "hosted PAPR authority-command replacement count differs")
    require(replaced_papr_terminal == expected_papr_terminal, "hosted PAPR terminal-command replacement count differs")
    require(replaced_w10_authority == expected_w10_authority, "hosted W10 authority-command replacement count differs")
    return tuple(result)


def self_test() -> None:
    original = gate.profile_commands("ci-cpu")
    hosted = hosted_commands("ci-cpu")
    require(len(original) == len(hosted), "hosted routing changed command count")
    differences = [(before, after) for before, after in zip(original, hosted, strict=True) if before != after]
    g11_present = G11_AUTHORITY.is_file() and not G11_AUTHORITY.is_symlink() and G11_TERMINAL.is_file() and not G11_TERMINAL.is_symlink()
    w10_present = W10_AUTHORITY.is_file() and not W10_AUTHORITY.is_symlink() and W10_CLOSEOUT.is_file() and not W10_CLOSEOUT.is_symlink()
    papr_present = PAPR_AUTHORITY.is_file() and not PAPR_AUTHORITY.is_symlink()
    papr_terminal = papr_present and PAPR_SELECTED.is_file() and not PAPR_SELECTED.is_symlink() and PAPR_COMPLETION.is_file() and not PAPR_COMPLETION.is_symlink()
    w10_authority = W10_AUTHORITY.is_file() and not W10_AUTHORITY.is_symlink() and not w10_present and papr_terminal
    expected = (
        int(ER9_CLOSEOUT.is_file() and not ER9_CLOSEOUT.is_symlink())
        + int(ER2_COMPLETION.is_file() and not ER2_COMPLETION.is_symlink())
        + int(g11_present)
        + int(w10_present)
        + int(papr_present and not papr_terminal)
        + int(papr_terminal)
        + int(w10_authority)
    )
    require(len(differences) == expected, "hosted routing changed an unexpected command")
    for before, after in differences:
        require(Path(before[1]).name in {"verify_er9_production_v4.py", "verify_er2_randomized.py", "verify_g11.py", "verify_w10_rehearsal.py", "verify_papr_training_authorization.py"}, "hosted routing replaced a non-runtime command")
        require(Path(after[1]).resolve() == Path(__file__).resolve(), "hosted routing replacement is not this audited adapter")
    verify_er9_published()
    if ER2_COMPLETION.is_file() and not ER2_COMPLETION.is_symlink():
        verify_er2_published()
    if g11_present:
        verify_g11_published()
    if w10_present:
        verify_w10_published()
    if papr_present and not papr_terminal:
        verify_papr_authority_published()
    if papr_terminal:
        verify_papr_published()
    if w10_authority:
        verify_w10_authority_published()
    print(f"hosted quality-gate routing self-test PASS: replacements={len(differences)}")


def run(profile: str) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), str(REPO / "tools"), environment.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    for command in hosted_commands(profile):
        print("$ " + " ".join(command), flush=True)
        subprocess.run(command, cwd=REPO, env=environment, check=True)
    gate._check_clean_checkout()
    print(f"hosted quality gate PASS: {profile}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("self-test", "verify-er9-published", "verify-er2-published", "verify-g11-published", "verify-papr-authority-published", "verify-papr-published", "verify-w10-authority-published", "verify-w10-published", "static", "ci-cpu"))
    args = parser.parse_args(argv)
    if args.action == "self-test":
        self_test()
    elif args.action == "verify-er9-published":
        verify_er9_published()
    elif args.action == "verify-er2-published":
        verify_er2_published()
    elif args.action == "verify-g11-published":
        verify_g11_published()
    elif args.action == "verify-papr-authority-published":
        verify_papr_authority_published()
    elif args.action == "verify-papr-published":
        verify_papr_published()
    elif args.action == "verify-w10-authority-published":
        verify_w10_authority_published()
    elif args.action == "verify-w10-published":
        verify_w10_published()
    else:
        run(args.action)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
