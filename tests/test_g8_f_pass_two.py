"""Synthetic and structural guards for exact-once BR-4 pass two."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from baseline import g8_f_pass_two as pass_two
from baseline.classical import composition


def test_pass_two_identified_record_rejects_mutation() -> None:
    value = pass_two._identified({"schema_version": 1, "scope": pass_two.SCOPE}, field="authorization_id", prefix=pass_two.AUTH_PREFIX)
    pass_two._verify_identified(value, field="authorization_id", prefix=pass_two.AUTH_PREFIX)
    value["scope"] = "WIDENED"
    with pytest.raises(pass_two.PassTwoHold, match="content digest"):
        pass_two._verify_identified(value, field="authorization_id", prefix=pass_two.AUTH_PREFIX)


def test_normative_selection_has_exactly_two_passes_and_no_third() -> None:
    assert composition.selection_passes() == (1, 2)
    campaign = composition.SelectionCampaign(composition.CLASSICAL_ADAPTIVE)
    with pytest.raises(composition.SelectionPassError, match="unknown selection pass 3"):
        campaign.run_pass(3, lambda _context: (), scorer="forbidden")


def test_pass_two_runner_checks_completion_before_authorization(tmp_path: Path) -> None:
    output = tmp_path / "pass_two_state.json"
    output.write_text("immutable", encoding="ascii")
    with pytest.raises(pass_two.PassTwoHold, match="rerun is forbidden"):
        pass_two.run_pass_two(output_path=output)


def test_pass_two_source_has_no_training_test_or_pass_three_entrypoint() -> None:
    source = Path(pass_two.__file__).read_text(encoding="utf-8")
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
    assert '"pass_two": 1' in source
