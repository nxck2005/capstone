"""Read-only access to the published G-12 test-split curves shown by the Streamlit demo.

The figure and table read the same exported tables as the paper's figures
(presentation-results/data/g12_test_*.csv, written by tools/export_g12_tables.py).
Before anything is shown, every plotted point is checked against the committed
G-12 closeout (results/g12/results.csv): the cell-mean accuracy and delivered
fraction must agree exactly. Nothing here trains, evaluates or re-scores a model.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURVES_CSV = "presentation-results/data/g12_test_curves.csv"
DIFFERENCES_CSV = "presentation-results/data/g12_test_differences.csv"
CLOSEOUT_CSV = "results/g12/results.csv"
TEST_IMAGES = 3925

# (system, scorer) pairs drawn at r = 1/6, in the paper's headline-figure order.
PLOTTED = (
    ("learned", "own_task_head"),
    ("classical_adaptive", "artifact_finetuned"),
    ("er9_digital", "own_task_head"),
    ("er9_digital_low_rate", "own_task_head"),
)
COMPARISONS = ("learned-classical_adaptive", "learned-er9_digital")


@dataclass(frozen=True)
class Point:
    snr_db: float
    accuracy: float
    ci_low: float
    ci_high: float
    coverage: float
    cells: int


@dataclass(frozen=True)
class Difference:
    snr_db: float
    difference: float
    ci_low: float
    ci_high: float
    cells: int


@dataclass(frozen=True)
class TestResults:
    grid: list[float]
    curves: dict[str, list[Point]]
    differences: dict[str, list[Difference]]

    def point(self, system: str, snr_db: float) -> Point:
        return next(p for p in self.curves[system] if p.snr_db == snr_db)

    def difference(self, comparison: str, snr_db: float) -> Difference:
        return next(d for d in self.differences[comparison] if d.snr_db == snr_db)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


Key = tuple[str, str, str]  # (system, scorer, bw_ratio)


def load_curves(keys: tuple[Key, ...], root: Path = ROOT) -> tuple[list[float], dict[Key, list[Point]]]:
    """The exported test curves for `keys`, each point checked against the G-12 closeout."""
    closeout: dict[Key, dict[float, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in _rows(root / CLOSEOUT_CSV):
        if row["split"] != "test" or int(row["n_test"]) != TEST_IMAGES:
            raise RuntimeError(f"G-12 closeout row {row['run_id']} is not a full test-split row")
        closeout[(row["system"], row["classifier_variant"], row["bw_ratio"])][float(row["test_snr_db"])].append(row)

    exported: dict[Key, list[dict[str, str]]] = defaultdict(list)
    for row in _rows(root / CURVES_CSV):
        exported[(row["system"], row["classifier_variant"], row["bw_ratio"])].append(row)

    curves: dict[Key, list[Point]] = {}
    grid: list[float] | None = None
    for key in keys:
        system, _scorer, ratio = key
        rows = sorted(exported[key], key=lambda r: float(r["snr_db"]))
        snrs = [float(r["snr_db"]) for r in rows]
        if len(rows) != 21 or (grid is not None and snrs != grid):
            raise RuntimeError(f"{system} {ratio}: exported curve does not cover the 21-point SNR grid")
        grid = snrs
        points = []
        for row in rows:
            snr = float(row["snr_db"])
            cells = closeout[key][snr]
            if int(row["n_images"]) != TEST_IMAGES or len(cells) != int(row["cells"]):
                raise RuntimeError(f"{system} {ratio} at {snr:g} dB: cell count differs from the closeout")
            accuracy = sum(int(c["n_correct"]) / int(c["n_test"]) for c in cells) / len(cells)
            coverage = sum(float(c["coverage_rate"]) for c in cells) / len(cells)
            if not (math.isclose(accuracy, float(row["accuracy"]), abs_tol=1e-12)
                    and math.isclose(coverage, float(row["coverage"]), abs_tol=1e-12)):
                raise RuntimeError(f"{system} {ratio} at {snr:g} dB: exported value differs from the closeout")
            points.append(Point(snr, float(row["accuracy"]), float(row["ci_low"]), float(row["ci_high"]),
                                float(row["coverage"]), int(row["cells"])))
        curves[key] = points
    if grid is None:
        raise ValueError("no curves requested")
    return grid, curves


def load(root: Path = ROOT) -> TestResults:
    grid, verified = load_curves(tuple((system, scorer, "r_1_6") for system, scorer in PLOTTED), root)
    curves = {system: points for (system, _scorer, _ratio), points in verified.items()}

    differences: dict[str, list[Difference]] = defaultdict(list)
    for row in _rows(root / DIFFERENCES_CSV):
        if row["bw_ratio"] == "r_1_6" and row["comparison"] in COMPARISONS:
            differences[row["comparison"]].append(Difference(
                float(row["snr_db"]), float(row["difference"]), float(row["ci_low"]),
                float(row["ci_high"]), int(row["cells"])))
    for comparison in COMPARISONS:
        differences[comparison].sort(key=lambda d: d.snr_db)
        if [d.snr_db for d in differences[comparison]] != grid:
            raise RuntimeError(f"{comparison}: paired differences do not cover the SNR grid")
        # A paired difference of cell means equals the difference of the curve means.
        for d in differences[comparison]:
            learned, other = comparison.split("-")
            gap = next(p.accuracy for p in curves[learned] if p.snr_db == d.snr_db) - next(
                p.accuracy for p in curves[other] if p.snr_db == d.snr_db)
            if not math.isclose(gap, d.difference, abs_tol=1e-9):
                raise RuntimeError(f"{comparison} at {d.snr_db:g} dB: difference disagrees with the curves")
    return TestResults(grid, curves, dict(differences))
