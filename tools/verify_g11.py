#!/usr/bin/env python3
"""Verify the validation-only G-11 H4 and terminal closeout artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.deterministic_core import canonical_sha256


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "results/learned/g11"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _artifact(relative: str) -> Path:
    path = (REPO / relative).resolve()
    _require(REPO.resolve() in path.parents, f"G-11 artifact escapes repository: {relative}")
    return path


def verify_h4(path: Path) -> dict[str, Any]:
    value = _read(path)
    _require(value.get("artifact_role") == "G11_H4_VALIDATION_PRECISION_SIMULATION", "H4 artifact role differs")
    _require(value.get("method") == "prospective_paired_precision_on_validation_discordance", "H4 method differs")
    _require(value.get("gate") == "G-11", "H4 gate differs")
    _require(value.get("test_access") == 0, "H4 records test access")
    _require(value.get("assumptions", {}).get("test_data_read") is False, "H4 test-data assumption differs")
    _require(int(value.get("repetitions", 0)) > 0, "H4 repetition count is invalid")
    quantiles = value.get("quantiles")
    _require(isinstance(quantiles, dict) and set(quantiles) == {"q50", "q95", "q99"}, "H4 quantiles are incomplete")
    for key, percentage in value.get("mde_percentage_points", {}).items():
        _require(key in quantiles and float(percentage) == float(quantiles[key]) * 100.0, "H4 percentage conversion differs")  # literal-ok: percentage-point conversion
    _require(value.get("gate_decision") in {"full_strength", "narrowed"}, "H4 gate decision is invalid")
    _require(
        value.get("gate_rule") == "full_strength_if_validation_derived_p95_mde_at_future_sample_size_is_at_most_2pp",
        "H4 gate rule differs",
    )
    inputs = value.get("input_artifacts", {})
    for field in ("er9_validation", "er2_validation"):
        input_path = _artifact(str(inputs[field]))
        _require(_sha(input_path) == inputs[f"{field}_sha256"], f"H4 input artifact hash differs: {field}")
    _require(inputs.get("snr_db") == 7, "H4 pairing SNR differs")  # literal-ok: fixed validation pairing SNR
    return value


def verify_architecture_difference(path: Path) -> dict[str, Any]:
    value = _read(path)
    _require(value.get("artifact_role") == "ER9_ARCHITECTURE_DIFFERENCE_AUDIT", "ER-9 architecture-difference role differs")
    _require(value.get("declared_difference") == "channel_interface", "ER-9 declared difference differs")
    _require(value.get("architecture_difference_paths") == ["channel_interface"], "ER-9 observed difference paths differ")
    _require(value.get("only_declared_difference") is True, "ER-9 architecture audit is not closed")
    _require(value.get("er9_interface", {}).get("reconstruction_head") == "none", "ER-9 reconstruction head differs")
    _require(value.get("er9_interface", {}).get("loss") == "cross_entropy", "ER-9 loss differs")
    _require(value.get("er9_interface", {}).get("lambda") == "not_applicable", "ER-9 lambda differs")
    _require(value.get("test_access") == 0, "ER-9 architecture audit records test access")
    return value


def verify_terminal(path: Path) -> dict[str, Any]:
    value = _read(path)
    identifier = value.get("terminal_id")
    body = dict(value)
    body.pop("terminal_id", None)
    _require(identifier == "g11terminal-" + canonical_sha256(body), "G-11 terminal ID differs")
    _require(value.get("artifact_role") == "G11_TERMINAL_VALIDATION_ONLY_CLOSEOUT", "G-11 terminal role differs")
    _require(value.get("status") == "G11_GREEN_NO_TEST_ACCESS" and value.get("decision") == "GREEN", "G-11 decision differs")
    counters = value.get("protected_counters", {})
    expected_counters = {
        "g10_evaluations": 63,  # literal-ok: immutable G-10 terminal count
        "g10_reruns": 0,  # literal-ok: immutable G-10 rerun count
        "er9_training": int(counters.get("er9_training", -1)),
        "randomized_er2_training": 1,  # literal-ok: one authorized randomized run
        "g11": 1,  # literal-ok: one terminal G-11 adjudication
        "w10": 0,  # literal-ok: W10 remains unopened
        "learned_test_inference": 0,  # literal-ok: test remains sealed
        "model_facing_test_access": 0,  # literal-ok: test remains sealed
    }
    _require(counters == expected_counters, "G-11 protected counters differ")
    g10 = value.get("g10_terminal")
    _require(g10 == {
        "source": "9515c490aed4439f7ced2c163abef61557654ddf",
        "evaluations": 63,  # literal-ok: immutable G-10 terminal count
        "reruns": 0,  # literal-ok: immutable G-10 rerun count
        "classification": "expected_crossover_observed",
        "headline_bracket": "-5 -> -4 dB",
    }, "G-10 terminal reference differs")
    _require(value.get("g10_terminal_immutable") is True, "G-10 immutability flag differs")
    _require(value.get("test") == "SEALED" and value.get("test_access") == 0, "G-11 test boundary differs")
    h4_record = value.get("h4", {})
    h4_path = _artifact(str(h4_record["artifact"]))
    _require(_sha(h4_path) == h4_record["artifact_sha256"], "G-11 H4 artifact hash differs")
    verify_h4(h4_path)
    architecture_record = value.get("er9", {})
    architecture_path = _artifact(str(architecture_record["architecture_difference"]))
    _require(
        _sha(architecture_path) == architecture_record["architecture_difference_sha256"],
        "ER-9 architecture audit hash differs",
    )
    verify_architecture_difference(architecture_path)
    validation_cells = architecture_record.get("validation_cells")
    expected_validation_paths = [
        "results/learned/er9/final_validation/train0_channel0.json",
        "results/learned/er9/final_validation/train1_channel1.json",
        "results/learned/er9/final_validation/train2_channel2.json",
    ]
    _require(isinstance(validation_cells, list) and len(validation_cells) == len(expected_validation_paths), "G-11 ER-9 validation-cell closure differs")
    for cell, expected_path in zip(validation_cells, expected_validation_paths, strict=True):
        _require(cell.get("path") == expected_path, "G-11 ER-9 validation-cell order differs")
        cell_path = _artifact(expected_path)
        _require(_sha(cell_path) == cell.get("sha256"), "G-11 ER-9 validation-cell hash differs")
    for section in (value.get("er9", {}), value.get("randomized_er2", {})):
        for field in (
            "production_manifest", "stage1_selection", "stage2_selection",
            "validation_train0_channel0", "architecture_difference", "completion", "validation",
        ):
            if field in section:
                artifact = _artifact(str(section[field]))
                _require(_sha(artifact) == section[f"{field}_sha256"], f"G-11 artifact hash differs: {field}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    root = args.root if args.root.is_absolute() else REPO / args.root
    value = verify_terminal(root / "g11_terminal_closeout.json")
    print(f"G-11 verifier PASS: {value['terminal_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
