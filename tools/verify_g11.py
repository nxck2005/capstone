#!/usr/bin/env python3
"""Verify the terminal AM-97 G11 artifact from ordinary W8/G10 and ER-9."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evaluation.downstream_v4 import (  # noqa: E402
    G11_AUTHORITY_PATH,
    PRODUCTION_CELLS,
    load_source,
    read_json,
    require,
    sha256_file,
    source_record,
)
from training.deterministic_core import canonical_sha256  # noqa: E402

DEFAULT_ROOT = REPO / "results/learned/g11"


def verify_authority(path: Path | None = None) -> dict:
    source = load_source(REPO)
    value = read_json(path or REPO / G11_AUTHORITY_PATH, "G11 authority")
    body = dict(value)
    identifier = body.pop("authority_id", None)
    require(identifier == "g11h4auth-" + canonical_sha256(body), "G11 authority ID differs")
    require(value.get("authority_kind") == "G11_H4_VALIDATION_ONLY_EXECUTION_AUTHORITY", "G11 authority role differs")
    require(value.get("status") == "FROZEN_G11_ONLY_PRE_EXECUTION", "G11 authority status differs")
    require(value.get("source_manifest") == source_record(REPO, source) and value.get("source_binding") == source, "G11 source binding differs")
    require(value.get("cells") == [list(cell) for cell in PRODUCTION_CELLS], "G11 authority cells differ")
    require(value.get("learned_arm") == "ordinary_learned_w8_g10" and value.get("comparator") == "final_production_er9", "G11 authority arms differ")
    require(value.get("randomized_er2_as_h4_arm") is False, "G11 authority substitutes randomized ER-2")
    require(value.get("bootstrap_resamples") == 10000 and value.get("quantiles") == ["q50", "q95", "q99"] and value.get("reference_pp") == 2, "G11 authority AM-97 constants differ")  # literal-ok: AM-97 constants
    require(value.get("pointwise_only") is True and value.get("full_h4_power_claim") is False, "G11 authority overclaims H4")
    require(value.get("g11_execution_count") == 1 and value.get("training_authorized") is False and value.get("w10_authorized") is False and value.get("test_authorized") is False, "G11 authority scope differs")
    for record in value.get("input_bindings", {}).values():
        require(isinstance(record, dict), "G11 authority input binding is malformed")
        require(sha256_file(REPO / record["path"]) == record["sha256"], "G11 authority input hash differs")
    require(value.get("validation_only") is True and value.get("test") == "SEALED" and value.get("test_access") == 0, "G11 authority crossed test boundary")
    return value


def verify_h4(path: Path) -> dict:
    value = read_json(path, "G11 H4 diagnostic")
    body = dict(value); identifier = body.pop("diagnostic_id", None)
    require(identifier == "g11h4v4-" + canonical_sha256(body), "G11 H4 diagnostic ID differs")
    require(value.get("artifact_role") == "G11_H4_POINTWISE_PRECISION_DIAGNOSTIC" and value.get("schema_version") == 2, "G11 H4 role differs")
    require(value.get("learned_arm") == "ordinary_learned_w8_g10" and value.get("comparator") == "er9_digital", "G11 H4 arms differ")
    require(value.get("randomized_er2_as_h4_arm") is False, "randomized ER-2 substituted into H4")
    require(value.get("cells") == [list(cell) for cell in PRODUCTION_CELLS], "G11 H4 cells differ")
    require(value.get("bootstrap_unit") == "stable_image_complete_three_cell_trajectory" and value.get("bootstrap_resamples") == 10000, "G11 bootstrap contract differs")  # literal-ok: AM-97 count
    require(value.get("reference_pp") == 2.0 and value.get("diagnostic_role") == "pointwise_precision_diagnostic_only", "G11 pointwise reference differs")  # literal-ok: AM-97 reference
    require(value.get("interpretation", {}).get("full_h4_decision_procedure_power_certified") is False, "G11 overclaims H4 power")
    require(value.get("test_data_read") is False and value.get("test_access") == 0, "G11 H4 accessed test")
    for arm in ("ordinary_learned_w8_g10", "final_production_er9"):
        bindings = value.get("input_bindings", {}).get(arm)
        expected_count = 63 if arm.startswith("ordinary") else 3  # literal-ok: 3x21 G10 / three ER9 files
        require(isinstance(bindings, list) and len(bindings) == expected_count, f"G11 {arm} binding count differs")
        for record in bindings:
            target = Path(record["path"])
            if not target.is_absolute():
                target = REPO / target
            require(sha256_file(target) == record["sha256"], f"G11 input hash differs: {target}")
    return value


def verify_architecture(path: Path) -> dict:
    value = read_json(path, "G11 architecture audit")
    body = dict(value); identifier = body.pop("audit_id", None)
    require(identifier == "er9archdiffv2-" + canonical_sha256(body), "G11 architecture audit ID differs")
    require(value.get("only_declared_difference") is True and value.get("unexpected_difference_paths") == [], "G11 architecture audit is not closed")
    require(value.get("observed_interface_differences") == ["channel_interface"], "G11 interface difference differs")
    require(value.get("test_access") == 0, "G11 architecture audit accessed test")
    return value


def verify_terminal(path: Path) -> dict:
    authority = verify_authority()
    value = read_json(path, "G11 terminal")
    body = dict(value); identifier = body.pop("terminal_id", None)
    require(identifier == "g11terminal-" + canonical_sha256(body), "G11 terminal ID differs")
    require(value.get("artifact_role") == "G11_TERMINAL_VALIDATION_ONLY_CLOSEOUT" and value.get("status") == "G11_GREEN_NO_TEST_ACCESS", "G11 terminal role/status differs")
    require(value.get("authority_id") == authority["authority_id"], "G11 terminal authority differs")
    require(value.get("cells") == [list(cell) for cell in PRODUCTION_CELLS], "G11 terminal cells differ")
    require(value.get("input_sources") == {"learned": "ordinary_learned_w8_g10", "comparator": "final_production_er9", "randomized_er2": False}, "G11 input arms differ")
    semantics = value.get("am97_semantics", {})
    require(semantics == {"signed_correctness_difference": True, "within_image_three_cell_mean": True, "stable_image_complete_trajectory_bootstrap": True, "bootstrap_resamples": 10000, "quantiles": ["q50", "q95", "q99"], "reference_pp": 2, "pointwise_only": True, "full_h4_power_claim": False}, "G11 AM-97 semantics differ")  # literal-ok: AM-97 constants
    h4 = value["h4"]; h4_path = REPO / h4["artifact"]
    require(sha256_file(h4_path) == h4["artifact_sha256"], "G11 H4 hash differs")
    verify_h4(h4_path)
    er9 = value["er9"]; architecture_path = REPO / er9["architecture_difference"]
    require(sha256_file(architecture_path) == er9["architecture_difference_sha256"], "G11 architecture hash differs")
    verify_architecture(architecture_path)
    require(len(er9.get("validation_cells", [])) == 3, "G11 ER-9 validation closure differs")  # literal-ok: exact cells
    require(value.get("test") == "SEALED" and value.get("test_access") == 0, "G11 test boundary differs")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--authority-only", action="store_true")
    args = parser.parse_args(argv)
    if args.authority_only:
        value = verify_authority()
        print(f"G-11 verifier PASS: authority only; {value['authority_id']}")
        return 0
    root = args.root if args.root.is_absolute() else REPO / args.root
    value = verify_terminal(root / "g11_terminal_closeout.json")
    print(f"G-11 verifier PASS: {value['terminal_id']}; AM-97 pointwise diagnostic only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
