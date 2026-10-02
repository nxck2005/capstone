#!/usr/bin/env python3
"""Export the G-12 test results as plotting tables for the paper and deck.

Reads the committed closeout (``results/g12/results.csv``,
``results/per_image_manifest.csv``) and the per-image streams it binds
(``results/per_image/g12/``, each checked against its manifest digest). For
every system, ratio and scorer it writes the accuracy at each SNR averaged over
the seed cells that were run, with a 95% percentile interval from the same
shared image bootstrap the ER-10 analysis uses; and for the two preregistered
comparisons, the paired cell-averaged difference with its interval.

    python tools/export_g12_tables.py
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config.params import get  # noqa: E402
from evaluation.g12_analysis import SharedBootstrap, _interval  # noqa: E402

OUT = REPO / "presentation-results" / "data"
PRIMARY = {
    "learned": "own_task_head",
    "learned_snr_randomised": "own_task_head",
    "learned_papr_constrained": "own_task_head",
    "semantic_recon_ablation": "clean",
    "er9_digital": "own_task_head",
    "er9_digital_low_rate": "own_task_head",
    "label_transmission_bound": "predicted_label",
    "classical_adaptive": "artifact_finetuned",
    "classical_fixed_mcs": "artifact_finetuned",
    "classical_fixed_mod": "artifact_finetuned",
    "classical_jpeg_secondary": "artifact_finetuned",
}
COMPARISONS = (("learned", "classical_adaptive"), ("learned", "er9_digital"))


def _streams() -> dict[tuple[str, str, str], dict[tuple[int, float], dict[str, int]]]:
    """Per-image correctness keyed by (system, ratio, scorer) → (cell, SNR) → stable ID."""

    streams: dict[tuple[str, str, str], dict[tuple[int, float], dict[str, int]]] = defaultdict(dict)
    with open(REPO / "results/per_image_manifest.csv", newline="") as handle:
        for entry in csv.DictReader(handle):
            raw = gzip.open(REPO / "results/per_image" / entry["file"]).read()
            if hashlib.sha256(raw).hexdigest() != entry["rows_sha256"]:
                raise SystemExit(f"per-image stream differs from its manifest: {entry['file']}")
            rows = [json.loads(line) for line in raw.splitlines()]
            key = (entry["system"], entry["bw_ratio"], entry["classifier_variant"])
            streams[key][(int(entry["train_seed"]), float(entry["test_snr_db"]))] = {row["stable_sample_id"]: int(bool(row["correct"])) for row in rows}
    return streams


def _cube(cells: dict[tuple[int, float], dict[str, int]], ids: list[str], seeds: list[int], snrs: list[float]) -> np.ndarray:
    """images × cells × SNRs of 0/1 correctness."""

    return np.array([[[cells[(seed, snr)][stable_id] for snr in snrs] for seed in seeds] for stable_id in ids], dtype=np.int8)


def main() -> int:
    streams = _streams()
    coverage = {}
    with open(REPO / "results/g12/results.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            coverage[(row["system"], row["bw_ratio"], row["classifier_variant"], int(row["train_seed"]), float(row["test_snr_db"]))] = float(row["coverage_rate"])
    any_key = next(iter(streams))
    ids = sorted(next(iter(streams[any_key].values())))
    snrs = sorted({snr for cells in streams.values() for _seed, snr in cells})
    boot = SharedBootstrap(len(ids), int(get("evaluation.bootstrap_resamples")))

    curves, cubes = [], {}
    for (system, ratio, scorer), cells in sorted(streams.items()):
        seeds = sorted({seed for seed, _snr in cells})
        cube = _cube(cells, ids, seeds, snrs)
        cubes[(system, ratio, scorer)] = cube
        per_image = cube.mean(axis=1)  # images × SNRs, averaged over the cells run
        resampled = boot.means(per_image)
        for s, snr in enumerate(snrs):
            low, high = _interval(resampled[:, s])
            curves.append({
                "system": system, "bw_ratio": ratio, "classifier_variant": scorer,
                "primary_scorer": PRIMARY.get(system) == scorer,
                "cells": len(seeds), "snr_db": snr, "n_images": len(ids),
                "accuracy": float(per_image[:, s].mean()), "ci_low": low, "ci_high": high,
                **{f"accuracy_cell{seed}": float(cube[:, j, s].mean()) for j, seed in enumerate(seeds)},
                "coverage": float(np.mean([coverage[(system, ratio, scorer, seed, snr)] for seed in seeds])),
            })

    differences = []
    for learned, comparator in COMPARISONS:
        d = cubes[(learned, "r_1_6", PRIMARY[learned])].astype(np.float64) - cubes[(comparator, "r_1_6", PRIMARY[comparator])]
        per_image = d.mean(axis=1)
        resampled = boot.means(per_image)
        for s, snr in enumerate(snrs):
            low, high = _interval(resampled[:, s])
            differences.append({"comparison": f"{learned}-{comparator}", "bw_ratio": "r_1_6", "cells": d.shape[1], "snr_db": snr, "difference": float(per_image[:, s].mean()), "ci_low": low, "ci_high": high})

    OUT.mkdir(parents=True, exist_ok=True)
    for name, table in (("g12_test_curves.csv", curves), ("g12_test_differences.csv", differences)):
        fields = list(dict.fromkeys(field for row in table for field in row))
        with open(OUT / name, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in fields} for row in table)
        print(f"wrote presentation-results/data/{name}: {len(table)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
