"""Tests for `tools/check_doc_consistency.py` -- the checker that guards the docs.

AM-62 built that tool and tested it by hand, by injecting drift and watching it
fire. That immediately found two real bugs in it:

  * line-by-line checking flagged correctly-labelled history, because an
    amendment's back-reference routinely sits a line or two below the value it
    explains (fixed by checking blocks);
  * an amendment number that does not exist still granted an exemption, so the
    back-reference rule could be defeated by citing anything at all -- `AM-999`
    worked as well as `AM-52` (fixed by intersecting cited numbers with the real
    AM set).

A tool whose only test is "someone ran it once and it looked right" is exactly
what this project keeps catching in its own audits: AM-58's finding was a passing
evidence script that violated four of the rules it claimed to enforce. So the
hand-run is written down here instead.

The third case is the one that matters. Cases 1 and 2 would pass against a
checker with the second bug still in it; only case 3 distinguishes them.
"""

from __future__ import annotations

import sys

import pytest

import check_doc_consistency as cdc

# A value the spec really did supersede, and the amendment that superseded it:
# AM-52 enlarged the SNR grid, so "18 SNR points" is stale prose everywhere it
# appears without a back-reference. Taken from the tool's own `stale` table so
# this test breaks loudly if that rule is ever removed rather than silently
# testing nothing.
STALE_TEXT = "18 SNR points"
FIXED_BY = "AM-52"
NONEXISTENT = "AM-999"


@pytest.fixture
def run_checker(tmp_path, monkeypatch, capsys):
    """Run `main()` over a synthetic document tree, returning (exit_code, stdout).

    The real `params.generated.yaml` and `SPEC.md` are used -- the point is to
    test the checker against the actual spec, not against a fixture of one that
    would drift. Only the *documents* it scans are synthetic.

    `PACKET_RECORD` is pointed at a nonexistent path so the evidence-record check
    is skipped; it is covered by `check_packetisation.py` itself.
    """

    def _run(doc_text: str) -> tuple[int, str]:
        doc = tmp_path / "doc.md"
        doc.write_text(doc_text)
        monkeypatch.setattr(cdc, "REPO", tmp_path)
        monkeypatch.setattr(cdc, "DOCS", ["doc.md"])
        monkeypatch.setattr(cdc, "PACKET_RECORD", tmp_path / "no-such-record.json")
        monkeypatch.setattr(sys, "argv", ["check_doc_consistency.py"])
        code = cdc.main()
        return code, capsys.readouterr().out

    return _run


def test_stale_value_without_back_reference_fails(run_checker):
    """Case 1: the superseded value, unlabelled. Must be reported."""
    code, out = run_checker(f"The evaluation sweeps {STALE_TEXT} in total.\n")
    assert code == 1
    assert "superseded" in out
    assert "grid size" in out


def test_stale_value_with_back_reference_passes(run_checker):
    """Case 2: the same value, carrying the amendment that superseded it.

    This is append-only history, which the repository's convention explicitly
    allows -- superseded entries stay wrong in place, on purpose, so long as they
    say so. A checker that flagged this would be unusable.
    """
    code, out = run_checker(
        f"The evaluation swept {STALE_TEXT} before the grid was enlarged.\n"
        f"That figure is superseded by {FIXED_BY} and is kept for the record.\n"
    )
    assert code == 0, out


def test_nonexistent_amendment_grants_no_exemption(run_checker):
    """Case 3: the regression test for the bug AM-62 found in itself.

    Citing an amendment that does not exist must NOT excuse a stale value. Before
    the cited set was intersected with the real AM set, this passed -- which meant
    the back-reference rule could be defeated by typing any number at all.

    Asserting on the *reason* rather than just the exit code matters here: the
    checker also reports the dangling reference separately, so a bare
    `code == 1` would pass even with the exemption bug still present.
    """
    code, out = run_checker(
        f"The evaluation sweeps {STALE_TEXT} in total.\n"
        f"Superseded by {NONEXISTENT}, allegedly.\n"
    )
    assert code == 1
    assert "superseded" in out, (
        "stale value was excused by a nonexistent amendment -- the AM-62 "
        f"exemption bug is back:\n{out}"
    )
    assert f"references {NONEXISTENT}" in out


def test_bold_whole_amendment_round_phrase_is_a_current_claim(run_checker):
    """The README's emphasis shape must not hide a stale live AM count.

    The checker originally recognised ``records **ten** amendment rounds`` but
    not ``records **ten amendment rounds**``. The latter is the exact form that
    let README.md continue claiming 64 entries after AM-65..AM-67 existed.
    """
    code, out = run_checker(
        "The specification records **ten amendment rounds** across 64 `AM` entries.\n"
    )

    assert code == 1
    assert "claims 64 AM entries as current" in out


def _historical_banner(target: str) -> str:
    return (
        f"{cdc.HISTORICAL_MARKER}\n"
        "> **Historical snapshot:** This plan is retained for provenance and may "
        "contain superseded commands or status. See "
        f"[`NEXT.md`]({target}) for current repository state.\n\n"
        f"The evaluation swept {STALE_TEXT}; retained as provenance.\n"
    )


def test_only_exact_resolving_historical_plan_banner_is_excluded(
    tmp_path, monkeypatch, capsys
):
    plan = tmp_path / "docs" / "plans" / "old.md"
    plan.parent.mkdir(parents=True)
    plan.write_text(_historical_banner("../../NEXT.md"))
    (tmp_path / "NEXT.md").write_text("# Current handoff\n")
    monkeypatch.setattr(cdc, "REPO", tmp_path)
    monkeypatch.setattr(cdc, "DOCS", None)
    monkeypatch.setattr(cdc, "PACKET_RECORD", tmp_path / "missing.json")
    monkeypatch.setattr(sys, "argv", ["check_doc_consistency.py"])

    code = cdc.main()
    out = capsys.readouterr().out

    assert code == 0, out
    assert "1 current hand-written documentation files" in out


@pytest.mark.parametrize(
    "body, expected",
    [
        (
            f"The evaluation swept {STALE_TEXT}.\n",
            "missing exact opening historical-plan marker",
        ),
        (
            _historical_banner("../NEXT.md"),
            "does not resolve to repository root",
        ),
    ],
)
def test_invalid_historical_plan_banner_is_reported_and_scanned(
    tmp_path, monkeypatch, capsys, body, expected
):
    plan = tmp_path / "docs" / "plans" / "old.md"
    plan.parent.mkdir(parents=True)
    plan.write_text(body)
    (tmp_path / "NEXT.md").write_text("# Current handoff\n")
    monkeypatch.setattr(cdc, "REPO", tmp_path)
    monkeypatch.setattr(cdc, "DOCS", None)
    monkeypatch.setattr(cdc, "PACKET_RECORD", tmp_path / "missing.json")
    monkeypatch.setattr(sys, "argv", ["check_doc_consistency.py"])

    code = cdc.main()
    out = capsys.readouterr().out

    assert code == 1
    assert "historical-plan banner finding" in out
    assert expected in out
    assert "superseded grid size" in out


def test_repo_docs_are_consistent(monkeypatch, capsys):
    """The real documents pass. This is the check the commit hook cares about."""
    monkeypatch.setattr(sys, "argv", ["check_doc_consistency.py"])
    code = cdc.main()
    assert code == 0, capsys.readouterr().out
