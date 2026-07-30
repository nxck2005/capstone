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


# The historical-plan cases below need a root `NEXT.md` only so the banner's link
# resolves, but NEXT.md is required to declare its current phase (see the
# current-phase block further down), so the stub declares one rather than being
# exempted -- an exemption would be a hole in the rule that matters.
MINIMAL_NEXT = """# Current handoff

## Single next task

| bounded W4 integration | **next** |

### Cold-start: the first thing to do in a fresh session

Begin bounded W4 integration only.
"""


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
    (tmp_path / "NEXT.md").write_text(MINIMAL_NEXT)
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
    (tmp_path / "NEXT.md").write_text(MINIMAL_NEXT)
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


# --- NEXT.md current-phase agreement ----------------------------------------------
#
# The defect: NEXT.md declared bounded W4 integration as the single next task, and
# then two other live sections said "Do not begin W4" and "begin the
# transparency-bitrate probe only" -- work that had already finished. Three live
# next steps in one file, none behind a banner, surviving several sessions.
#
# The synthetic file below is deliberately minimal but structurally real: a
# declaration table, the three sections a cold start reads, and a session-log entry
# that says exactly the wrong thing on purpose. Each test substitutes ONE live
# sentence, so a failure names the rule that broke.

NEXT_TEMPLATE = """# Very Next Steps

**Last updated:** 2026-07-30 · **Phase:** **W3 complete; G-2 PASS.**

## Single next task

| | |
|---|---|
| W3 | complete |
| transparency-bitrate probe | complete |
| bounded W4 integration | **next** |
| G-8 | unresolved |

Begin only the bounded **W4 classical-baseline integration required before G-8**.
Do not run the full BR-4 sweep, calibrate lambda, or open G-8.

### Cold-start: the first thing to do in a fresh session

The single next engineering task is bounded W4 classical-baseline integration.
{live}

### The short version, in order

Bounded W4 integration is the single next engineering task.

#### ~~Batch 1 — the old plan~~ **DONE 2026-07-28**

Confirm nothing drifted, then begin the transparency-bitrate probe only.

## Session log

- **2026-07-29** — Ran the probe. Next: begin the transparency-bitrate probe, then W3.
"""


@pytest.fixture
def run_next_checker(tmp_path, monkeypatch, capsys):
    """Run `main()` over a synthetic `NEXT.md`, returning (exit_code, stdout)."""

    def _run(live: str, template: str = NEXT_TEMPLATE) -> tuple[int, str]:
        (tmp_path / "NEXT.md").write_text(template.format(live=live))
        monkeypatch.setattr(cdc, "REPO", tmp_path)
        monkeypatch.setattr(cdc, "DOCS", ["NEXT.md"])
        monkeypatch.setattr(cdc, "PACKET_RECORD", tmp_path / "no-such-record.json")
        monkeypatch.setattr(sys, "argv", ["check_doc_consistency.py"])
        code = cdc.main()
        return code, capsys.readouterr().out

    return _run


def test_agreeing_handoff_passes(run_next_checker):
    """The baseline: one declared phase, every live section agreeing with it.

    Without this case the four below could all pass against a checker that simply
    fails everything.
    """
    code, out = run_next_checker(
        "Confirm nothing drifted, then begin bounded W4 integration only."
    )
    assert code == 0, out


def test_prohibiting_the_declared_next_task_fails(run_next_checker):
    """The live NEXT.md:91 defect: a live section forbidding the declared frontier."""
    code, out = run_next_checker(
        "Do not begin W4, G-8, or the reference-classifier fallback ladder."
    )
    assert code == 1
    assert "prohibits the declared next task" in out


def test_prohibiting_part_of_the_next_task_is_allowed(run_next_checker):
    """The distinction that makes the rule usable rather than merely noisy.

    Bounded W4 integration being live does NOT license the full BR-4 sweep, so a
    live section must still be able to forbid the sweep by name. A checker that
    fired on this would force the hand-off to drop its real scope boundary.
    """
    code, out = run_next_checker("Do not begin W4's full BR-4 validation sweep.")
    assert code == 0, out


def test_directing_completed_work_as_next_fails(run_next_checker):
    """The live NEXT.md:312 defect: a live section sending a cold start backwards."""
    code, out = run_next_checker(
        "Confirm nothing drifted, then begin the transparency-bitrate probe only:"
    )
    assert code == 1
    assert "directs completed work" in out
    assert "transparency-bitrate probe" in out


def test_historical_sections_may_still_say_the_wrong_thing(run_next_checker):
    """History is exempt, and the template proves it on two markers at once.

    Every case above runs against a file that already contains the stale directive
    twice -- once under a struck-through `**DONE**` heading and once as a dated
    session-log entry. Both must stay silent, or the only way to pass the check
    would be to rewrite the record, which is the opposite of the convention.
    """
    code, out = run_next_checker("Nothing further.")
    assert code == 0, out


def test_live_section_that_never_names_the_frontier_fails(run_next_checker):
    """Agreement is positive, not just the absence of contradiction.

    A cold start that reads only the Cold-start section must find the frontier
    there. Deleting the mention is not a contradiction any negative rule can see.
    """
    code, out = run_next_checker("Nothing further.", NEXT_TEMPLATE.replace(
        "The single next engineering task is bounded W4 classical-baseline integration.",
        "The single next engineering task is named at the top of this file.",
    ))
    assert code == 1
    assert "never names the declared frontier W4" in out


def test_missing_declaration_is_reported(run_next_checker):
    """No authoritative declaration means the whole check is vacuous -- say so."""
    code, out = run_next_checker(
        "Nothing further.",
        NEXT_TEMPLATE.replace("| bounded W4 integration | **next** |\n", ""),
    )
    assert code == 1
    assert "no declared next task" in out


# --- preflight ordering -----------------------------------------------------------


PREFLIGHT = """# Commands

```bash
.venv/bin/python tools/gen_spec_views.py --check
{fetch_before}.venv/bin/python -m pytest
{fetch_after}```
"""


@pytest.mark.parametrize(
    "fetch_before, fetch_after, expected",
    [
        (".venv/bin/python tools/fetch_ldpc_golden_vectors.py\n", "", None),
        ("", "", "without first running"),
        ("", ".venv/bin/python tools/fetch_ldpc_golden_vectors.py\n", "after pytest"),
    ],
    ids=["fetch-first", "no-fetch", "fetch-last"],
)
def test_preflight_block_must_fetch_the_ignored_fixture_first(
    fetch_before, fetch_after, expected
):
    """A fresh clone that runs the documented preflight in order must not fail.

    `tests/fixtures/ldpc_ts38212_golden.npz` is git-ignored (AM-25) and the srsRAN
    fixture test hard-asserts it rather than skipping, so a preflight block that
    reaches pytest first fails the suite for a provenance reason that reads like a
    scientific one. Ordering, not just presence, is the thing checked -- a fetch
    line below pytest documents the fix while still breaking the first run.
    """
    body = PREFLIGHT.format(fetch_before=fetch_before, fetch_after=fetch_after)
    findings = cdc.preflight_order_findings("doc.md", body)
    if expected is None:
        assert findings == []
    else:
        assert len(findings) == 1
        assert expected in findings[0]
