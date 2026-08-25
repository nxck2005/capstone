"""Focused deterministic G8_G rule and scope guards."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from baseline import g8_g_closeout as closeout


def _inputs(*, efficiency: set[str], crossover: set[str]) -> dict:
    return {
        "ratio_ceilings": [
            {
                "ratio": ratio,
                "meets_efficiency_threshold": ratio in efficiency,
                "meets_crossover_threshold": ratio in crossover,
            }
            for ratio in closeout.get("bandwidth.ratios")
        ]
    }


def test_ratio_rule_selects_smallest_matching_rungs() -> None:
    value = closeout._selected_ratios(
        _inputs(
            efficiency={"r_1_2", "r_1_3", "r_1_6", "r_1_12"},
            crossover={"r_1_2", "r_1_3"},
        )
    )
    assert value["efficiency_ratio"] == "r_1_12"
    assert value["crossover_ratio"] == "r_1_3"
    assert value["headline_ratio"] == "r_1_3"
    assert value["low_ratio_operating_point"] == "r_1_12"
    assert not value["asymmetric_fallback_applied"]


def test_ratio_rule_applies_preregistered_asymmetric_fallback() -> None:
    value = closeout._selected_ratios(
        _inputs(efficiency={"r_1_2", "r_1_3", "r_1_6", "r_1_12", "r_1_24"}, crossover=set())
    )
    assert value["efficiency_ratio"] == "r_1_24"
    assert value["crossover_ratio"] == "r_1_2"
    assert value["headline_ratio_selector"] == "efficiency_ratio"
    assert value["headline_ratio"] == "r_1_24"
    assert value["low_ratio_operating_point"] == "r_1_48"
    assert value["low_ratio_boundary_rule_applied"]


def test_ratio_rule_holds_on_bottom_saturation() -> None:
    with pytest.raises(closeout.G8CloseoutHold, match="smallest ladder rung"):
        closeout._selected_ratios(_inputs(efficiency={"r_1_48"}, crossover=set()))


def test_closeout_source_has_no_training_test_or_pass_three_entrypoint() -> None:
    source = Path(closeout.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert "data.test_access" not in imported
    assert not any(name.startswith("training") for name in imported)
    assert "def run_pass_three" not in source
    assert '"pass_three": 0' in source
    assert '"test_access": 0' in source
