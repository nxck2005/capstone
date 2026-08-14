#!/usr/bin/env python3
"""Compare per-trial paired parity reports and apply the frozen diagnostic rule."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from config.execution_profiles import canonical_json_bytes  # noqa: E402


class ParityVerificationError(RuntimeError):
    pass


def read_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ParityVerificationError(f"cannot read parity report {path}: {exc}") from exc
    if not isinstance(report, dict):
        raise ParityVerificationError("parity report is not an object")
    return report


def verify_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != 1 or report.get("artifact_kind") != "execution_profile_paired_numerical_parity":
        raise ParityVerificationError("unsupported parity report schema")
    if report.get("scientific_status") != "NON-SCIENTIFIC" or report.get("diagnostic_only") is not True:
        raise ParityVerificationError("parity report is not diagnostic-only")
    for key in ("g8_coverage", "test_access", "validation_decoding", "training"):
        if report.get(key) != 0:
            raise ParityVerificationError(f"parity report {key} is nonzero")
    digest = report.get("report_sha256")
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    if digest != hashlib.sha256(canonical_json_bytes(body)).hexdigest():
        raise ParityVerificationError("parity report digest differs")
    cells = report.get("cells")
    if not isinstance(cells, list) or len(cells) != report.get("selected_cell_count"):
        raise ParityVerificationError("parity cell count differs")
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise ParityVerificationError("parity cell is not an object")
        indicators = cell.get("block_error_indicators")
        trials = cell.get("trials")
        if not isinstance(indicators, list) or not isinstance(trials, int) or len(indicators) != trials or any(item not in (0, 1) for item in indicators):
            raise ParityVerificationError("per-trial block-error indicator vector differs")
        if cell.get("block_errors") != sum(indicators):
            raise ParityVerificationError("block-error count does not match indicators")
        k = int(cell["identity"]["k_and_n"][0])
        if cell.get("bit_errors", -1) < 0 or cell.get("information_bits", 0) != trials * k:
            raise ParityVerificationError("parity count arithmetic differs")
        if abs(float(cell.get("bler", -1.0)) - sum(indicators) / trials) > 1e-12 or abs(float(cell.get("ber", -1.0)) - float(cell["bit_errors"]) / (trials * k)) > 1e-12:
            raise ParityVerificationError("parity rates do not reproduce counts")


def _crossing(points: list[tuple[float, float]], threshold: float = 0.5) -> float | None:
    points = sorted(points)
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
        if (y0 - threshold) == 0:
            return x0
        if (y0 - threshold) * (y1 - threshold) <= 0 and y0 != y1:
            return x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
    if points and points[-1][1] == threshold:
        return points[-1][0]
    return None


def compare(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    verify_report(left)
    verify_report(right)
    for key in ("campaign_id", "parity_plan_sha256", "paired_trial_count_per_cell", "selected_cell_count"):
        if left.get(key) != right.get(key):
            raise ParityVerificationError(f"parity reports bind different {key}")
    a_cells = left["cells"]
    b_cells = right["cells"]
    if [cell["ordinal"] for cell in a_cells] != [cell["ordinal"] for cell in b_cells]:
        raise ParityVerificationError("parity cell order differs")
    cell_results: list[dict[str, Any]] = []
    total_disagreements = 0
    total_trials = 0
    for a, b in zip(a_cells, b_cells, strict=True):
        if a["work_unit_id"] != b["work_unit_id"] or a["stream_seeds"] != b["stream_seeds"]:
            raise ParityVerificationError(f"paired stimuli differ at ordinal {a['ordinal']}")
        disagreements = sum(x != y for x, y in zip(a["block_error_indicators"], b["block_error_indicators"], strict=True))
        total_disagreements += disagreements
        total_trials += int(a["trials"])
        cell_results.append(
            {
                "ordinal": a["ordinal"],
                "work_unit_id": a["work_unit_id"],
                "snr_db": a["snr_db"],
                "trials": a["trials"],
                "disagreement_count": disagreements,
                "disagreement_rate": disagreements / a["trials"],
                "left_bler": a["bler"],
                "right_bler": b["bler"],
                "absolute_bler_delta": abs(a["bler"] - b["bler"]),
                "left_ber": a["ber"],
                "right_ber": b["ber"],
                "absolute_ber_delta": abs(a["ber"] - b["ber"]),
            }
        )

    curves: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for a, b in zip(a_cells, b_cells, strict=True):
        key = json.dumps({key: a["identity"][key] for key in ("base_graph", "k_and_n", "lifting_size", "modulation", "rate")}, sort_keys=True)
        curves[key].append((float(a["snr_db"]), float(a["bler"]), float(b["bler"])))
    displacement_values: list[float] = []
    displacement_details: list[dict[str, Any]] = []
    for key, points in curves.items():
        left_cross = _crossing([(x, y) for x, y, _ in points])
        right_cross = _crossing([(x, y) for x, _, y in points])
        if left_cross is not None and right_cross is not None:
            displacement_values.append(abs(left_cross - right_cross))
            displacement_details.append({"identity": json.loads(key), "left_crossing_db": left_cross, "right_crossing_db": right_cross, "absolute_displacement_db": abs(left_cross - right_cross)})
    criterion = left["criterion"]
    max_cell = max((item["disagreement_rate"] for item in cell_results), default=0.0)
    aggregate = total_disagreements / total_trials if total_trials else 0.0
    max_displacement = max(displacement_values, default=None)
    holds = (
        max_cell <= float(criterion["per_cell_disagreement_rate_max"])
        and aggregate <= float(criterion["aggregate_disagreement_rate_max"])
        and (max_displacement is None or max_displacement <= float(criterion["waterfall_displacement_db_max"]))
    )
    return {
        "schema_version": 1,
        "artifact_kind": "execution_profile_paired_numerical_parity_comparison",
        "scientific_status": "NON-SCIENTIFIC",
        "left_profile": left["execution_profile_id"],
        "left_device": left["device"],
        "right_profile": right["execution_profile_id"],
        "right_device": right["device"],
        "campaign_id": left["campaign_id"],
        "parity_plan_sha256": left["parity_plan_sha256"],
        "selected_cell_count": len(cell_results),
        "paired_trial_count": total_trials,
        "cell_results": cell_results,
        "aggregate_disagreement_count": total_disagreements,
        "aggregate_disagreement_rate": aggregate,
        "max_cell_disagreement_rate": max_cell,
        "bler_delta_max": max((item["absolute_bler_delta"] for item in cell_results), default=0.0),
        "ber_delta_max": max((item["absolute_ber_delta"] for item in cell_results), default=0.0),
        "waterfall_displacement_assessed": bool(displacement_values),
        "waterfall_displacement_db": max_displacement,
        "waterfall_displacement_details": displacement_details,
        "criterion": criterion,
        "criterion_pass": holds,
        "g8_coverage": 0,
        "test_access": 0,
        "validation_decoding": 0,
        "training": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(read_report(args.left), read_report(args.right))
    result["comparison_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    if args.output:
        args.output.write_bytes(canonical_json_bytes(result))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["criterion_pass"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ParityVerificationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
