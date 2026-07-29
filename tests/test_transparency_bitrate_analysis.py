"""Frozen-grid and selection-aware probe analysis tests."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from config.params import REPO_ROOT
from probes.transparency_bitrate import (
    ProbeDesignError,
    aggregate,
    load_design,
    selection_aware_bootstrap,
    threshold_forecast,
)


def test_committed_probe_design_has_exact_union_grid_and_axis_order():
    design = load_design(
        REPO_ROOT / "configs/transparency-bitrate-probe.yaml",
        require_implementation_commit=False,
    )

    assert [item["budget_bytes"] for item in design["budget_grid"]] == [
        663,
        800,
        1330,
        1600,
        2400,
        2661,
        3200,
        4000,
        4800,
        5328,
        5344,
        6400,
        8000,
        9600,
        10656,
        12800,
        15997,
    ]
    assert design["encode_axes_px"] == [160, 128, 96, 64]
    assert design["budget_grid"][-3]["sources"] == [
        "packetisation:qam16:5/6:r_1_3",
        "direct_bpp:3.33",
    ]


def test_design_rejects_post_observation_budget_change(tmp_path: Path):
    source = REPO_ROOT / "configs/transparency-bitrate-probe.yaml"
    design = yaml.safe_load(source.read_text(encoding="utf-8"))
    design["budget_grid"][0]["budget_bytes"] += 1
    path = tmp_path / source.name
    path.write_text(yaml.safe_dump(design, sort_keys=False), encoding="utf-8")

    with pytest.raises(ProbeDesignError, match="budget grid differs"):
        load_design(path, require_implementation_commit=False)


def _analysis_design() -> dict:
    return {
        "encode_axes_px": [160, 128],
        "budget_grid": [
            {"budget_bytes": 100, "requested_bpp": 1.0, "sources": ["fixture:a"]},
            {"budget_bytes": 200, "requested_bpp": 2.0, "sources": ["fixture:b"]},
        ],
        "bootstrap": {
            "method": "stable_id_trajectory_selection_aware_paired_bootstrap",
            "resamples": 200,
            "seed": 17,
            "identity": "fixture",
            "one_sided_confidence": 0.95,
            "quantile_method": "lower",
        },
    }


def _analysis_rows() -> list[dict]:
    rows: list[dict] = []
    axis_outcomes = {
        100: {
            160: [True, True, False, False],
            128: [False, False, True, True],
        },
        200: {
            160: [True, True, True, False],
            128: [True, True, True, False],
        },
    }
    for budget, axes in axis_outcomes.items():
        for axis, outcomes in axes.items():
            for index, correct in enumerate(outcomes):
                rows.append(
                    {
                        "stable_sample_id": f"id-{index}",
                        "clean_correct": index != 3,
                        "correct": correct,
                        "budget_bytes": budget,
                        "encode_axis": axis,
                        "feasible": True,
                        "decode_success": True,
                        "emitted_bytes": budget,
                        "realized_bpp": budget / 100,
                        "psnr": 20.0,
                        "ssim": 0.8,
                    }
                )
    return rows


def test_point_selection_tie_breaks_by_committed_axis_order():
    aggregate_rows, selected = aggregate(_analysis_rows(), _analysis_design())

    assert selected == {100: 160, 200: 160}
    assert len(aggregate_rows) == 4
    chosen = [row for row in aggregate_rows if row["selected_point_estimate"]]
    assert [row["encode_axis"] for row in chosen] == [160, 160]


def test_selection_aware_bootstrap_is_exactly_reproducible():
    design = _analysis_design()
    rows = _analysis_rows()

    first = selection_aware_bootstrap(rows, design)
    second = selection_aware_bootstrap(copy.deepcopy(rows), copy.deepcopy(design))

    assert first == second
    assert first["resamples"] == 200
    assert first["budgets"][0]["selected_encode_axis"] == 160


def test_threshold_forecast_reports_left_and_right_censoring():
    result = {
        "budgets": [
            {"meets": True, "budget_bytes": 100},
            {"meets": True, "budget_bytes": 200},
        ]
    }
    left = threshold_forecast(
        result,
        key="meets",
        label="probe_efficiency_threshold",
    )
    result["budgets"][0]["meets"] = False
    result["budgets"][1]["meets"] = False
    right = threshold_forecast(
        result,
        key="meets",
        label="probe_crossover_threshold",
    )

    assert left["left_censored"] and not left["right_censored"]
    assert right["right_censored"] and right["result"] is None
