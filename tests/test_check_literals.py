"""Mutation tests for the SR-1 numeric-literal checker."""

from __future__ import annotations

import sys

import pytest

import check_literals as literals


@pytest.fixture
def mutant_repo(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    monkeypatch.setattr(literals, "REPO", tmp_path)

    def _write(body: str):
        path = src / "mutant.py"
        path.write_text(body)
        return path

    return _write


def test_positive_parameter_literal_is_reported(mutant_repo):
    mutant_repo("snr_db = 7\n")

    result = literals.check()

    assert len(result.findings) == 1
    assert "literal 7" in result.findings[0].message
    assert "params.channel.train_snr_db_fixed" in result.findings[0].message


def test_negative_literal_is_reported_as_unary_operation(mutant_repo):
    mutant_repo("snr_db = -8\n")

    result = literals.check()

    matches = [finding for finding in result.findings if "literal -8" in finding.message]
    assert len(matches) == 1, (
        "negative SNR escaped the AST walk or its Constant operand was counted "
        f"separately: {result.findings}"
    )
    assert "params.channel.test_snr_grid_db.0" in matches[0].message


def test_annotation_without_reason_does_not_suppress(mutant_repo):
    mutant_repo("snr_db = 7  # literal-ok:\n")

    result = literals.check()

    assert any("requires a non-empty reason" in item.message for item in result.findings)
    assert any("literal 7" in item.message for item in result.findings)
    assert result.annotations == 0


def test_reasoned_annotation_suppresses_one_line_and_is_counted(mutant_repo):
    mutant_repo("snr_db = 7  # literal-ok: protocol fixture deliberately injects 7 dB\n")

    result = literals.check()

    assert result.findings == ()
    assert result.annotations == 1


def test_annotation_text_inside_a_string_does_not_suppress(mutant_repo):
    mutant_repo('snr_db = 7\nnote = "# literal-ok: not a comment"\n')

    result = literals.check()

    assert any("literal 7" in item.message for item in result.findings)
    assert result.annotations == 0


def test_main_fails_on_mutant_and_reports_summary(mutant_repo, monkeypatch, capsys):
    mutant_repo("snr_db = -8\n")
    monkeypatch.setattr(sys, "argv", ["check_literals.py"])

    assert literals.main() == 1
    output = capsys.readouterr().out
    assert "literal configuration finding" in output
    assert "literal -8" in output


def test_repository_source_passes():
    result = literals.check()

    assert result.findings == ()
