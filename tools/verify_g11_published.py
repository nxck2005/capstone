#!/usr/bin/env python3
"""Verify the committed G11/H4 evidence without the Confessor runtime.

This is deliberately a separate path from ``verify_g11.py``'s worker terminal
entry point.  It authenticates the immutable authority, H4/architecture/
terminal records, G10 manifest bindings and ER-9 validation bindings that are
published in a clean clone; it does not recompute worker-local G10 per-image
trajectories.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evaluation.downstream_v4 import (  # noqa: E402
    PRODUCTION_CELLS,
    read_json,
    sha256_file,
    validate_final_er9_cell,
)
from evaluation.g10_protocol import verify_identified  # noqa: E402
from training.deterministic_core import canonical_sha256  # noqa: E402
from verify_g11 import verify_authority  # noqa: E402

G11_ROOT = REPO / "results/learned/g11"
G11_TERMINAL = G11_ROOT / "g11_terminal_closeout.json"
ER9_CLOSEOUT = REPO / "results/learned/er9/er9_production_closeout_v4.json"
G10_MANIFEST = REPO / "results/learned/w9/g10_runtime_manifest.json"


class PublishedG11Hold(RuntimeError):
    """Committed G11 evidence is not a valid hosted closeout."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PublishedG11Hold(message)


def content_address(value: Mapping[str, object], field: str, prefix: str, label: str) -> None:
    body = dict(value)
    identifier = body.pop(field, None)
    require(identifier == prefix + canonical_sha256(body), f"{label} content address differs")


def full_sha(value: object, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and value != "0" * 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} is not a full SHA-256",
    )
    return value


def verify_published() -> dict[str, str | int]:
    authority = verify_authority()
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
    require(isinstance(cells, list) and len(cells) == 63, "G10 manifest cell count differs")  # literal-ok: 3x21 G10 cells
    seen: set[str] = set()
    for index, (binding, cell) in enumerate(zip(learned_bindings, cells, strict=True)):
        require(isinstance(binding, Mapping) and isinstance(cell, Mapping), "G11 learned binding is malformed")
        full_sha(binding.get("sha256"), f"G11 learned binding {index}")
        require(binding.get("path") == cell.get("file_path") and binding.get("sha256") == cell.get("file_sha256"), f"G11 learned binding differs at {index}")
        require(binding.get("artifact_id") == cell.get("artifact_id") and cell.get("cell_index") == index, f"G10 manifest cell binding differs at {index}")
        artifact_id = str(cell.get("artifact_id"))
        require(bool(artifact_id) and artifact_id not in seen, f"G10 manifest cell repeats at {index}")
        seen.add(artifact_id)

    er9_bindings = h4.get("input_bindings", {}).get("final_production_er9")
    require(isinstance(er9_bindings, list) and len(er9_bindings) == len(PRODUCTION_CELLS), "G11 ER-9 binding count differs")
    for cell, binding in zip(PRODUCTION_CELLS, er9_bindings, strict=True):
        require(isinstance(binding, Mapping), "G11 ER-9 binding is malformed")
        relative = f"results/learned/er9/final_validation/train{cell[0]}_channel{cell[1]}.json"
        require(binding.get("path") == relative, f"G11 ER-9 binding path differs at {cell}")
        full_sha(binding.get("sha256"), f"G11 ER-9 binding {cell}")
        require(sha256_file(REPO / relative) == binding["sha256"], f"G11 ER-9 binding hash differs at {cell}")
        validation = read_json(REPO / relative, f"ER-9 final validation {cell}")
        validate_final_er9_cell(validation, cell=cell)
        content_address(validation, "validation_id", "er9productionvalidation-", f"ER-9 validation {cell}")
        require(validation.get("validation_id") == binding.get("validation_id"), f"G11 ER-9 validation ID differs at {cell}")

    architecture = read_json(architecture_path, "G11 architecture audit")
    content_address(architecture, "audit_id", "er9archdiffv2-", "G11 architecture audit")
    require(architecture.get("only_declared_difference") is True and architecture.get("unexpected_difference_paths") == [], "G11 architecture audit is not closed")
    require(architecture.get("observed_interface_differences") == ["channel_interface"] and architecture.get("test_access") == 0, "G11 architecture evidence differs")

    terminal = read_json(G11_TERMINAL, "G11 terminal")
    content_address(terminal, "terminal_id", "g11terminal-", "G11 terminal")
    require(terminal.get("artifact_role") == "G11_TERMINAL_VALIDATION_ONLY_CLOSEOUT" and terminal.get("status") == "G11_GREEN_NO_TEST_ACCESS", "G11 terminal role/status differs")
    require(terminal.get("decision") == "GREEN" and terminal.get("authority_id") == authority["authority_id"], "G11 terminal decision/authority differs")
    require(terminal.get("source_commit") == authority["source_commit"] and terminal.get("source_manifest_id") == authority["source_manifest"]["manifest_id"], "G11 terminal source differs")
    require(terminal.get("cells") == [list(cell) for cell in PRODUCTION_CELLS], "G11 terminal cells differ")
    require(terminal.get("input_sources") == {"learned": "ordinary_learned_w8_g10", "comparator": "final_production_er9", "randomized_er2": False}, "G11 terminal input arms differ")
    require(terminal.get("test") == "SEALED" and terminal.get("test_access") == 0, "G11 terminal crossed test boundary")
    h4_record = terminal.get("h4", {})
    require(h4_record.get("artifact") == str(h4_path.relative_to(REPO)) and h4_record.get("artifact_sha256") == sha256_file(h4_path), "G11 terminal H4 binding differs")
    er9_record = terminal.get("er9", {})
    require(er9_record.get("production_closeout") == str(ER9_CLOSEOUT.relative_to(REPO)) and er9_record.get("production_closeout_sha256") == sha256_file(ER9_CLOSEOUT), "G11 terminal ER-9 binding differs")
    require(terminal.get("protected_counters", {}).get("w10") == 0 and terminal.get("protected_counters", {}).get("model_facing_test_access") == 0, "G11 terminal protected counters differ")
    print("G11/H4 published-evidence verifier PASS: worker runtime not inspected")
    return {"authority_id": str(authority["authority_id"]), "terminal_id": str(terminal["terminal_id"]), "test_access": 0}


if __name__ == "__main__":
    verify_published()
