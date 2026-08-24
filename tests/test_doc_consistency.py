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


# --- Pascal G8_C operational-cursor agreement ------------------------------------


PASCAL_COMPLETED_STATE = {
    "execution": "complete",
    "coverage": "3213/3213",
    "evidence": "published",
    "next": "g8-f-f1-separate-owner-operator-launch",
    "bler_table": "frozen",
    "g8_d": "d7-complete",
    "g8_e_e2e4": "complete-verified",
    "g8_e_e5e7": "complete-green-pass-one-frozen",
    "readiness_state": "f0-green-f1-zero",
    "runtime_state": "completed-production-state",
    "rerun": "forbidden",
    "old_local": "immutable-zero-successor-coverage",
}


def _pascal_cursor(**changes: str) -> str:
    state = dict(PASCAL_COMPLETED_STATE)
    state.update(changes)
    fields = "; ".join(f"{key}={value}" for key, value in state.items())
    return f"<!-- capstone-current-pascal-state: {fields} -->"


def _pascal_documents(**changes: str) -> dict[str, str]:
    return {
        doc: _pascal_cursor(**changes)
        for doc in cdc.PASCAL_CURSOR_DOCS
    }


def test_pascal_cursor_catches_stale_current_zero_coverage_guidance() -> None:
    """A live cursor cannot leave Pascal at pre-launch 0/3213 after NEXT advances."""
    documents = _pascal_documents()
    documents["AGENTS.md"] = _pascal_cursor(
        execution="pre-launch",
        coverage="0/3213",
        evidence="not-published",
        next="owner-launch-authorization",
    )
    documents["instructions/RESUME.md"] = _pascal_cursor(
        execution="pre-launch",
        coverage="0/3213",
        evidence="not-published",
        next="owner-launch-authorization",
    )

    findings = cdc.pascal_cursor_findings(documents)

    assert any("AGENTS.md" in finding and "coverage" in finding for finding in findings)
    assert any("instructions/RESUME.md" in finding and "coverage" in finding for finding in findings)


def test_pascal_cursor_accepts_completed_execution_and_d1_next_gate() -> None:
    documents = _pascal_documents()

    assert cdc.pascal_cursor_findings(documents) == []


def test_pascal_cursor_ignores_historical_prelaunch_records() -> None:
    """Historical zero-coverage facts remain evidence, not current guidance."""
    documents = {
        doc: (
            _pascal_cursor()
            + "\n\n## Historical pre-launch snapshot\n"
            + "The successor was 0/3213 before the owner opened the campaign."
        )
        for doc in cdc.PASCAL_CURSOR_DOCS
    }

    assert cdc.pascal_cursor_findings(documents) == []


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


# --- sub-phase agreement, one level below the milestone ---------------------------
#
# The defect check 6 could not see: NEXT.md declared `W4 · PB_3` next, and a lower
# live section still read "PB_1 ... is complete; `instructions/PB_2.txt` is the next
# step", with a third saying "bounded W4 integration is the live task". All three
# contain "W4", so the frontier-naming rule passed; "is the next step" has none of
# `DIRECTIVE_RE`'s verbs, so the directive rule never looked. It survived a full
# corrective phase.
#
# The template below declares PB_3 next with PB_1/PB_2 complete, and carries the two
# stale sentences twice on purpose -- once under a struck `**DONE**` heading and once
# in the session log. Both must stay silent, exactly as in the check-6 block above:
# the exemption is the marker, not the wording.

NEXT_PB_TEMPLATE = """# Very Next Steps

**Last updated:** 2026-08-01 · **Phase:** **W4 PB_2 corrected; PB_3 next.**

## Single next task

| | |
|---|---|
| W3 | complete |
| bounded W4 integration | PA, PB_1 (incl. PB_1C) and PB_2 (incl. PB_2C) complete; PB_3 remains |
| W4 · PB_1 (incl. the PB_1C correction) | complete |
| W4 · PB_2 (incl. the PB_2C correction) | complete |
| W4 · PB_3 | **next engineering phase — not started** |
| G-8 | unresolved |

Run `instructions/PB_3.txt` from B3.0. Do not open G-8.

### Cold-start: the first thing to do in a fresh session

The single next engineering task is W4 PB_3, the BR-4 selection infrastructure.
{live}

### The short version, in order

W4 PB_3 is the single next engineering task.

#### ~~Batch 1 — the old plan~~ **DONE 2026-07-29**

Run `instructions/PB_2.txt` from B2.0. PB_2 is the next step.

## Session log

- **2026-07-30** — Landed PB_1. Next: bounded W4 integration is the live task.

Then confirm nothing drifted and begin bounded W4 integration.
"""


@pytest.fixture
def run_pb_checker(run_next_checker):
    """`run_next_checker`, bound to the PB_3-frontier template."""

    def _run(live: str, template: str = NEXT_PB_TEMPLATE) -> tuple[int, str]:
        return run_next_checker(live, template)

    return _run


def test_live_section_naming_a_completed_subphase_as_next_fails(run_pb_checker):
    """Case 1: the exact NEXT.md:203 defect, one level below the milestone."""
    code, out = run_pb_checker("`instructions/PB_2.txt` is the next step.")
    assert code == 1
    assert "names completed PB_2 as the next or live task" in out


def test_historical_snapshot_may_still_say_pb_2_was_next(run_pb_checker):
    """Case 2 and 6: the same two sentences, behind the markers, stay silent.

    The template already carries "PB_2 is the next step" under a struck `**DONE**`
    heading and "bounded W4 integration is the live task" in the session log, so
    this passing is the whole historical exemption, tested on both markers at once
    rather than asserted in a comment.
    """
    code, out = run_pb_checker("Nothing further.")
    assert code == 0, out


def test_active_section_naming_the_frontier_passes(run_pb_checker):
    """Case 3: the correct wording must not fire, or the rule is unusable."""
    code, out = run_pb_checker(
        "The single next engineering task is W4 PB_3, run from B3.0."
    )
    assert code == 0, out


def test_coarse_parent_nomination_fails(run_pb_checker):
    """Case 4: "bounded W4 integration is the live task".

    Reported as *imprecise*, not as backwards. The parent phase is partially
    complete -- PA, PB_1 and PB_2 are done and PB_3 remains -- so calling it the
    live task is not directing finished work, it is failing to say which part.
    Asserting on the reason keeps the two apart.
    """
    code, out = run_pb_checker("Right now, bounded W4 integration is the live task.")
    assert code == 1
    assert "nominates only the coarse parent phase 'bounded w4 integration'" in out
    assert "names completed" not in out


def test_stale_instruction_path_as_the_next_action_fails(run_pb_checker):
    """Case 5: a completed phase's instruction file, written as a live command."""
    code, out = run_pb_checker("Run `instructions/PB_2.txt` from B2.0.")
    assert code == 1
    assert "sends the reader back to PB_2" in out


def test_prohibiting_a_completed_phase_is_not_a_directive(run_pb_checker):
    """The negation guard: "do not run PB_2C again" is advice, not a next action."""
    code, out = run_pb_checker("Do not run `instructions/PB_2C.txt` again.")
    assert code == 0, out


def test_descriptive_mention_of_a_completed_phase_passes(run_pb_checker):
    """Without a nomination cue this is prose, and prose is not the tool's business.

    `AGENTS.md` really does need to say what `PB_2.txt` got wrong. A rule that
    fired on every mention of a finished phase would be deleted within a week.
    """
    code, out = run_pb_checker(
        "Note that `instructions/PB_2.txt` calls these `analysis.*`, which is stale."
    )
    assert code == 0, out


def test_finding_names_frontier_file_section_and_wording(run_pb_checker):
    """Case 7: a finding must be actionable without opening the file first."""
    code, out = run_pb_checker("`instructions/PB_2.txt` is the next step.")
    assert code == 1
    assert "NEXT.md:" in out                                  # file and line
    assert "Cold-start: the first thing to do in a fresh session" in out  # section
    assert "PB_3" in out                                      # declared frontier
    assert "PB_2" in out                                      # stale phase


def test_subphase_rule_reaches_documents_beyond_next_md(tmp_path, monkeypatch, capsys):
    """One declaration, every document. README.md must not contradict it either."""
    (tmp_path / "NEXT.md").write_text(NEXT_PB_TEMPLATE.format(live="Nothing further."))
    (tmp_path / "README.md").write_text(
        "# Project\n\nThe single next engineering task is W4 PB_2.\n"
    )
    monkeypatch.setattr(cdc, "REPO", tmp_path)
    monkeypatch.setattr(cdc, "DOCS", ["NEXT.md", "README.md"])
    monkeypatch.setattr(cdc, "PACKET_RECORD", tmp_path / "missing.json")
    monkeypatch.setattr(sys, "argv", ["check_doc_consistency.py"])

    code = cdc.main()
    out = capsys.readouterr().out

    assert code == 1
    assert "README.md:3" in out
    assert "names completed PB_2" in out


def test_declared_phase_signature_is_unchanged():
    """The addendum's compatibility rule, pinned.

    `declared_subphase` is a separate reader precisely so this three-value unpack
    keeps working for every existing caller and test.
    """
    frontier, token, done = cdc.declared_phase(
        NEXT_PB_TEMPLATE.format(live="Nothing further.")
    )
    assert token == "w4"
    assert "pb3" in frontier
    assert cdc.declared_subphase(NEXT_PB_TEMPLATE.format(live="x")) == "pb_3"
    assert "w3" in done


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


# ---------------------------------------------------------------------------
# Check 8 must survive the frontier ceasing to be a PB_n sub-phase (PB_3)
# ---------------------------------------------------------------------------

_GATE_DECLARATION = """# Very Next Steps

## Single next task

| | |
|---|---|
| W4 · PB_2 | complete |
| W4 · PB_3 | complete |
| G-8 classical validation work | **next engineering task — not started** |

The single next engineering task is the G-8 classical validation work.
"""


def test_a_gate_frontier_declares_no_sub_phase() -> None:
    assert cdc.declared_subphase(_GATE_DECLARATION) is None
    frontier, token, _ = cdc.declared_phase(_GATE_DECLARATION)
    assert token == "g-8"
    assert "g-8" in frontier


def test_completed_sub_phases_survive_a_gate_frontier() -> None:
    """Nothing is discarded as "the frontier's own number" when there is none."""

    completed = cdc.completed_subphases(_GATE_DECLARATION)
    assert "pb_2" in completed
    assert "pb_3" in completed


def test_a_gate_frontier_still_catches_a_stale_sub_phase_nomination() -> None:
    """The check must not go silent the moment W4 closes.

    Before PB_3 this returned no findings whenever the frontier named no
    `PB_n` -- which is exactly the handoff where a live section is most likely
    to still nominate the sub-phase that just finished.
    """

    body = _GATE_DECLARATION + "\nThe single next engineering task is W4 PB_3.\n"
    findings = cdc.subphase_findings("NEXT.md", body, _GATE_DECLARATION)
    assert any("names completed PB_3" in finding for finding in findings)


def test_a_gate_frontier_still_catches_a_backward_directive() -> None:
    body = _GATE_DECLARATION + "\nConfirm nothing drifted, then run instructions/PB_3.txt.\n"
    findings = cdc.subphase_findings("NEXT.md", body, _GATE_DECLARATION)
    assert any("sends the reader back to PB_3" in finding for finding in findings)


def test_a_gate_frontier_stays_silent_on_a_correct_nomination() -> None:
    body = _GATE_DECLARATION + "\nPB_3 is complete; see the worklog for what it built.\n"
    assert cdc.subphase_findings("NEXT.md", body, _GATE_DECLARATION) == []
