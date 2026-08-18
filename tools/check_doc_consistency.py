#!/usr/bin/env python3
"""Cross-document consistency check: do the hand-written docs still agree with the spec?

`gen_spec_views.py --check` validates `SPEC.md` against itself and regenerates the
derived views. Nothing validated the *current hand-written documentation*
against it. Three
separate drift incidents came out of that gap:

  * AM-52 enlarged the SNR grid and nothing that *counts* the grid was updated,
    so §2's family-wise arithmetic, §16's cost prose and the ER-1 projections
    stayed stale for three amendment rounds (found by AM-59);
  * AM-58 and AM-59 each moved a piece of the test-access rule and left them
    pointing at each other, reopening the leak they had just closed (AM-60);
  * an hour later the same G-10 -> G-12 change failed to reach the prose that
    summarises it, in the very session that recorded the lesson.

The rule this tool enforces is the repository's own convention, mechanised: a
superseded value may appear **only on a line that cites the amendment which
superseded it** (or a later one). That is exactly what "corrections carry an
`(AM-n)` back-reference" already requires, so append-only history passes and
unlabelled staleness fails -- without needing a human to eyeball which is which.

Usage:
    python tools/check_doc_consistency.py          # exit 1 on any finding
    python tools/check_doc_consistency.py -v       # also list what passed
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
PARAMS = REPO / "spec" / "params.generated.yaml"
SPEC = REPO / "spec" / "SPEC.md"
PACKET_RECORD = REPO / "spec" / "evidence" / "packetisation_record.json"

# Tests may replace this with an explicit fixture list. The real run discovers
# current hand-written documentation so a newly added file cannot evade checks.
DOCS: list[str] | None = None

HISTORICAL_MARKER = "<!-- capstone-doc-status: historical-plan -->"
HISTORICAL_BANNER_RE = re.compile(
    r"^> \*\*Historical snapshot:\*\* This plan is retained for provenance and "
    r"may contain superseded commands or status\. See "
    r"\[`NEXT\.md`\]\((?P<target>[^)]+)\) for current repository state\.$"
)

REQ_RE = re.compile(r"^- \*\*([A-Z]+)-(\d+)\*\* — ")
TOMBSTONE_RE = re.compile(r"^- ~~\*\*([A-Z]+)-(\d+)\*\*~~ — ")
CITE_RE = re.compile(r"`(params\.[A-Za-z0-9_.]+)`")
AM_REF_RE = re.compile(r"\bAM-(\d+)\b")
GATE_REF_RE = re.compile(r"\bG-(\d+)\b")
# Lines that ARE the append-only record, rather than prose about it.
HISTORY_LINE_RE = re.compile(r"^\s*[-*]\s+~?~?\*\*(AM|G|SR|BR|ER|DR|HR|PR|OPT|FW|DEC)-\d+\*\*|"
                             r"^\*\*Amendment round|^\s*[-*]\s+\*\*20\d\d-\d\d-\d\d")

# --- current-phase agreement inside NEXT.md --------------------------------------
#
# NEXT.md declares its phase once, in the table under `## Single next task`, and
# then explains it over a thousand lines of accreted prose. Three of those live
# sections drifted into stating three different next steps: one said to begin
# bounded W4 integration, one said "Do not begin W4", and the live Cold-start
# section said to begin the transparency-bitrate probe -- which had already
# finished. None of the three was behind a historical banner, so nothing caught it.
#
# What is checked, in one sentence: a live section may not prohibit the declared
# next task, may not direct already-completed work as the next step, and the
# sections that a cold start actually reads must each name the declared frontier.
#
# Deliberately NOT attempted: parsing the whole file as current state. Most of
# NEXT.md is accreted record, and a checker that read it as instructions would
# fire on every sentence. Historical content is exempt where it sits behind a
# struck-through heading, a DONE/Complete/PASS marker, a struck line, or the
# session log -- the markers this file already uses.
NEXT_DOC = "NEXT.md"
DECLARATION_HEADING = "## Single next task"
DECL_ROW_RE = re.compile(r"^\|\s*(?P<subject>[^|]*?)\s*\|\s*(?P<state>[^|]*?)\s*\|\s*$")
MILESTONE_RE = re.compile(r"\b(W\d+|G-\d+)\b", re.IGNORECASE)
HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.*)$")
HISTORICAL_HEADINGS = ("session log",)
HISTORICAL_HEADING_RE = re.compile(r"~~|\*\*DONE|\*\*Complete|PASS\*\*")
# Verbs that start work. `started`/`begins` are excluded by the word boundary on
# purpose: "G-8 has not started" is a status report, not a directive.
DIRECTIVE_RE = re.compile(r"\b(?:begin|start|open|run)\s+(?P<object>[^.:;]*)", re.IGNORECASE)
NEGATION_RE = re.compile(r"\b(?:do not|don't|never|cannot|not yet|no longer)\b", re.IGNORECASE)
# A prohibition naming the frontier is legitimate when it names a *part* of the
# frontier that really is out of scope. This table is judgement, and like the
# `stale` table above it is meant to grow.
NARROWING = ("'s", "sweep", "selection", "select", "full ", "lambda", "λ",
             "training", "train", "test split", "er-9", "g-8")
# The sections a cold start actually reads. Every one of these that is PRESENT must
# name the declared frontier, and at least one live section outside the declaration
# must name it -- which is what makes "the top status line, the cold-start section
# and the task table must all agree" mechanical rather than aspirational. Presence
# is not required: NEXT.md is rewritten constantly and pinning three exact heading
# strings forever would fail a rename rather than a drift, while the declaration
# itself stays mandatory so the check cannot be disarmed by deleting it.
FRONTIER_SECTIONS = (
    "## Single next task",
    "### Cold-start: the first thing to do in a fresh session",
    "### The short version, in order",
)

# These are the three operational cursors that must agree about the completed
# Pascal execution and the current G8_D gate. `instructions/` is intentionally not part of the general
# Markdown discovery scope, so RESUME.md is loaded explicitly for this narrow
# high-value check below.
PASCAL_CURSOR_DOCS = ("NEXT.md", "AGENTS.md", "instructions/RESUME.md")
PASCAL_CURSOR_RE = re.compile(
    r"<!--\s*capstone-current-pascal-state:\s*(?P<body>[^>]+?)\s*-->"
)
PASCAL_CURSOR_EXPECTED = {
    "execution": "complete",
    "coverage": "3213/3213",
    "evidence": "published",
    "next": "g8-e-e2",
    "bler_table": "frozen",
    "g8_d": "d7-complete",
    "readiness_state": "immutable-zero-coverage-history",
    "runtime_state": "completed-production-state",
    "rerun": "forbidden",
    "old_local": "immutable-zero-successor-coverage",
}
# --- sub-phase agreement, one level below the milestone --------------------------
#
# Check 6 reduces the declared frontier to a milestone token: `W4`. That is one
# level too coarse to see the defect it was written for. NEXT.md declared
# `W4 · PB_3` next while a lower live section still read "PB_1 ... is complete;
# `instructions/PB_2.txt` is the next step", and a third said "bounded W4
# integration is the live task". Every one of those sentences contains "W4", so
# the frontier-naming rule was satisfied, and "is the next step" carries none of
# `DIRECTIVE_RE`'s verbs, so nothing fired. The whole contradiction was invisible.
#
# What is checked: a live sentence that *nominates* a next or live task must name
# the declared sub-phase. Four cases fall out, and the distinction between the
# last two is the reason this is not one string match:
#
#   1. backward directive  -- "run `instructions/PB_2.txt`" when PB_2 is complete;
#   2. stale nomination    -- "PB_2 is the next step";
#   3. coarse nomination   -- "bounded W4 integration is the live task", which is
#      not *completed* work (PB_3 remains) but is the parent phase, so it does not
#      tell a cold start what to do. Reported as imprecise, not as backwards;
#   4. valid              -- names PB_3. Silent.
#
# Deliberately NOT flagged: descriptive mentions. "Where bounded W4 stands" and
# "`PB_2.txt` calls these `analysis.*`" carry no nomination cue and must stay
# silent, or the rule becomes a prose linter. Table rows are skipped for the same
# reason: the declaration is a status matrix, and check 6 already owns it.
PHASE_RE = re.compile(r"\bPB[_\- ]?(\d+)([A-D]?)\b", re.I)
PHASE_LETTERS = ("", "a", "b", "c", "d")
# Both word orders, because the repository writes it both ways: "PB_2 is the next
# step" and "the single next engineering task is PB_3".
NOMINATION_RE = re.compile(
    r"(?:is|are|remains?)\s+the\s+(?:single\s+)?(?:next|live|current)\b[^.;:]{0,40}?"
    r"\b(?:step|task|action|phase|work)\b"
    r"|the\s+(?:single\s+)?(?:next|live|current)\s+(?:\w+\s+){0,3}?"
    r"(?:step|task|action|phase|work)\s+(?:is|are|remains?)\b",
    re.I,
)

# A fenced block that runs the whole preflight must materialize the git-ignored
# rung-2 LDPC fixture before pytest, or a fresh clone fails the suite for a
# provenance reason that reads like a scientific one.
PREFLIGHT_ANCHOR = "gen_spec_views.py --check"
PREFLIGHT_FETCH = "fetch_ldpc_golden_vectors.py"
PREFLIGHT_TESTS = "-m pytest"


def _norm(text: str) -> str:
    """Strip markdown emphasis and collapse whitespace, for comparing prose."""
    return re.sub(r"\s+", " ", re.sub(r"[`*_]", "", text)).strip().lower()


def _milestone(subject: str) -> str | None:
    match = MILESTONE_RE.search(subject)
    return match.group(1).lower() if match else None


def declared_phase(body: str) -> tuple[str | None, str | None, list[str]]:
    """Read NEXT.md's own phase declaration: (frontier subject, its milestone, done).

    The declaration is the table under `## Single next task` and nothing else. One
    designated place has to be authoritative or "contradicts the declared phase"
    has no meaning; this is that place.
    """
    frontier: str | None = None
    done: list[str] = []
    inside = False
    for line in body.splitlines():
        if line.startswith("#"):
            inside = line.strip() == DECLARATION_HEADING
            continue
        if not inside:
            continue
        row = DECL_ROW_RE.match(line)
        if row is None:
            continue
        subject, state = _norm(row.group("subject")), _norm(row.group("state"))
        if not subject or set(subject) <= set("- "):
            continue
        if "next" in state:
            frontier = subject
        elif "complete" in state or "pass" in state:
            done.append(subject)
    return frontier, _milestone(frontier) if frontier else None, done


def live_lines_with_sections(body: str) -> list[tuple[int, str, str]]:
    """`live_lines`, plus the heading each live line sits under.

    The section is carried so a finding can say *where* the contradiction is. A
    line number alone sends a reader to a thousand-line file with no idea which
    of its accreted sections went stale.
    """
    out: list[tuple[int, str, str]] = []
    historical_at: int | None = None
    section = ""
    for number, line in enumerate(body.splitlines(), 1):
        heading = HEADING_RE.match(line)
        if heading:
            level = len(heading.group("hashes"))
            title = heading.group("title")
            if historical_at is not None and level <= historical_at:
                historical_at = None
            if (HISTORICAL_HEADING_RE.search(title)
                    or any(marker in title.lower() for marker in HISTORICAL_HEADINGS)):
                historical_at = level
            section = title.strip()
            continue
        if historical_at is not None or "~~" in line or HISTORY_LINE_RE.match(line):
            continue
        out.append((number, line, section))
    return out


def live_lines(body: str) -> list[tuple[int, str]]:
    """Lines of NEXT.md that read as current instructions.

    A heading is historical if it is struck through, carries a DONE/Complete/PASS
    marker, or is the session log; everything under it stays historical until a
    heading at the same or a higher level takes over. Struck lines are dropped
    wherever they appear, which covers the completed rows of the task table.
    """
    return [(number, line) for number, line, _ in live_lines_with_sections(body)]


def _parse_pascal_cursor(doc: str, body: str) -> tuple[dict[str, str] | None, list[str]]:
    """Read one explicit current Pascal cursor, leaving append-only history alone."""
    matches = list(PASCAL_CURSOR_RE.finditer(body))
    if not matches:
        return None, [f"{doc}: missing current Pascal cursor declaration"]
    if len(matches) != 1:
        return None, [f"{doc}: expected one current Pascal cursor declaration, found {len(matches)}"]

    match = matches[0]
    line = body.count("\n", 0, match.start()) + 1
    fields: dict[str, str] = {}
    malformed: list[str] = []
    for item in match.group("body").split(";"):
        item = item.strip()
        if not item:
            continue
        key, separator, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not key or not value or key in fields:
            malformed.append(item)
            continue
        fields[key] = value

    missing = PASCAL_CURSOR_EXPECTED.keys() - fields.keys()
    unexpected = fields.keys() - PASCAL_CURSOR_EXPECTED.keys()
    findings: list[str] = []
    if malformed:
        findings.append(f"{doc}:{line}: malformed current Pascal cursor field(s): {', '.join(malformed)}")
    if missing:
        findings.append(
            f"{doc}:{line}: current Pascal cursor is missing field(s): {', '.join(sorted(missing))}"
        )
    if unexpected:
        findings.append(
            f"{doc}:{line}: current Pascal cursor has unexpected field(s): {', '.join(sorted(unexpected))}"
        )
    return (fields if not findings else None), findings


def pascal_cursor_findings(documents: dict[str, str]) -> list[str]:
    """Require the live Pascal execution and next gate to agree across cursors.

    Only the explicit cursor declarations are read. Dated checkpoint paragraphs,
    struck historical sections and old campaign records may continue to describe
    their then-current zero-coverage state without being rewritten.
    """
    parsed: dict[str, dict[str, str]] = {}
    findings: list[str] = []
    for doc in PASCAL_CURSOR_DOCS:
        body = documents.get(doc)
        if body is None:
            findings.append(f"{doc}: missing operational cursor document")
            continue
        state, state_findings = _parse_pascal_cursor(doc, body)
        findings.extend(state_findings)
        if state is not None:
            parsed[doc] = state

    authoritative = parsed.get("NEXT.md")
    if authoritative is None:
        return findings
    for field, expected in PASCAL_CURSOR_EXPECTED.items():
        actual = authoritative.get(field)
        if actual != expected:
            findings.append(
                f"NEXT.md: current Pascal cursor field {field!r} is {actual!r}; expected {expected!r}"
            )
    for doc, state in parsed.items():
        for field in PASCAL_CURSOR_EXPECTED:
            if state.get(field) != authoritative.get(field):
                findings.append(
                    f"{doc}: current Pascal cursor field {field!r} is {state.get(field)!r}, "
                    f"but NEXT.md says {authoritative.get(field)!r}"
                )
    return findings


def _objects(text: str) -> list[str]:
    """Split a directive's object list into individual objects."""
    parts = re.split(r",|\bor\b|\band\b", _norm(text))
    out = []
    for part in parts:
        part = re.sub(r"^(?:only\s+)?(?:the|a|an|any|its|another)\s+", "", part.strip())
        if part:
            out.append(part)
    return out


def next_phase_findings(body: str) -> list[str]:
    """Findings where a live NEXT.md section contradicts NEXT.md's declared phase."""
    frontier, token, done = declared_phase(body)
    if frontier is None or token is None:
        return [f"{NEXT_DOC}:1: no declared next task under '{DECLARATION_HEADING}'"]
    findings = []
    for number, line in live_lines(body):
        for sentence in re.split(r"(?<=[.:;])\s+", line):
            negated = bool(NEGATION_RE.search(sentence))
            for match in DIRECTIVE_RE.finditer(sentence):
                for obj in _objects(match.group("object")):
                    if negated:
                        rest = obj[len(token):] if obj.startswith(token) else None
                        if (obj == token or (rest is not None
                                             and not any(n in obj for n in NARROWING))):
                            findings.append(
                                f"{NEXT_DOC}:{number}: prohibits the declared next task "
                                f"({frontier!r}): {sentence.strip()[:100]}")
                    else:
                        for finished in done:
                            if re.search(rf"\b{re.escape(finished)}\b", obj):
                                findings.append(
                                    f"{NEXT_DOC}:{number}: directs completed work "
                                    f"({finished!r}) as the next step, but the declared "
                                    f"next task is {frontier!r}: {sentence.strip()[:100]}")
    live = {number for number, _ in live_lines(body)}
    section: str | None = None
    seen: dict[str, bool] = {}
    elsewhere = False
    for number, line in enumerate(body.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            section = stripped if stripped in FRONTIER_SECTIONS else None
            if section:
                seen.setdefault(section, False)
            continue
        if number not in live or not re.search(rf"\b{token}\b", line, re.IGNORECASE):
            continue
        if section:
            seen[section] = True
        if section != DECLARATION_HEADING:
            elsewhere = True
    for wanted, named in seen.items():
        if not named:
            findings.append(
                f"{NEXT_DOC}:1: section '{wanted}' never names the declared frontier "
                f"{token.upper()}, so a cold start reading only that section is misdirected")
    if not elsewhere:
        findings.append(
            f"{NEXT_DOC}:1: the declared frontier {token.upper()} is named only inside "
            f"'{DECLARATION_HEADING}', so no live section tells a cold start to do it")
    return findings


def _subphases(text: str) -> set[str]:
    """Every `PB_n[X]` sub-phase named in a fragment, normalised to `pb_2c` form."""
    return {f"pb_{m.group(1)}{m.group(2).lower()}" for m in PHASE_RE.finditer(text)}


def declared_subphase(body: str) -> str | None:
    """The sub-phase of the declared frontier, e.g. `pb_3`, or None if it has none.

    Deliberately a separate reader rather than a fourth return value from
    `declared_phase`: every caller and test unpacks that function's three values,
    and a signature change would be a silent breakage for a cosmetic gain.
    """
    frontier, _, _ = declared_phase(body)
    if frontier is None:
        return None
    match = PHASE_RE.search(frontier)
    if match is None:
        return None
    return f"pb_{match.group(1)}{match.group(2).lower()}"


def completed_subphases(body: str) -> set[str]:
    """Sub-phases the declaration marks complete, with their correction letters.

    Narrow on purpose. Only phase *numbers* that a declaration row explicitly
    calls complete are included, and the frontier's own number never is. A parent
    row such as "bounded W4 integration" names no `PB_n`, so it contributes
    nothing here -- it is a partially completed parent, not a completed phase, and
    treating it as one would report a coarse nomination as a backwards one.
    """
    frontier_sub = declared_subphase(body)
    frontier_number = frontier_sub.removeprefix("pb_").rstrip("abcd") if frontier_sub else None
    _, _, done = declared_phase(body)
    numbers = {p.removeprefix("pb_").rstrip("abcd")
               for subject in done for p in _subphases(subject)}
    numbers.discard(frontier_number)
    return {f"pb_{n}{letter}" for n in numbers for letter in PHASE_LETTERS}


def parent_phrases(body: str) -> set[str]:
    """Declaration subjects that name the milestone but no sub-phase.

    "bounded W4 integration" is the real one: the phrase a live section reached
    for when it meant PB_3 and said something one level too vague.
    """
    frontier, token, done = declared_phase(body)
    if token is None:
        return set()
    subjects = [s for s in done + ([frontier] if frontier else []) if s]
    return {s for s in subjects if token in s and not _subphases(s)}


def subphase_findings(doc: str, body: str, declaration: str) -> list[str]:
    """Findings where live text nominates the wrong -- or too vague -- next phase.

    `declaration` is NEXT.md's body; `body` is the document being scanned, so the
    same rule reaches README.md and AGENTS.md while one file stays authoritative.
    """
    frontier, token, _ = declared_phase(declaration)
    sub = declared_subphase(declaration)
    if frontier is None or token is None:
        return []
    # A frontier may legitimately stop being a `PB_n` sub-phase -- once W4 closed,
    # the frontier became the gate `G-8`, which names none. The check must not go
    # silent at exactly that moment: a live section may still nominate a completed
    # sub-phase, and that is still the defect this was written for. Only the
    # coarse-parent branch needs a sub-phase to compare against.
    completed = completed_subphases(declaration)
    parents = parent_phrases(declaration)
    lines = live_lines_with_sections(body)
    findings: list[str] = []

    def where(number: int, section: str) -> str:
        place = f"section {section!r}" if section else "no enclosing section"
        label = sub.upper() if sub else token.upper()
        return (f"{doc}:{number}: {place}, declared frontier {frontier!r} ({label})")

    for index, (number, line, section) in enumerate(lines):
        # Table rows are status matrices, not instructions; check 6 owns the
        # declaration table itself.
        if line.lstrip().startswith("|"):
            continue
        # A nomination routinely wraps onto the next line, so the frontier may be
        # named just below its cue. Read the continuation before calling it vague.
        following = lines[index + 1][1] if index + 1 < len(lines) else ""
        for sentence in re.split(r"(?<=[.:;])\s+", line):
            named = _subphases(sentence)
            for match in DIRECTIVE_RE.finditer(sentence):
                if NEGATION_RE.search(sentence):
                    continue
                target = _subphases(match.group("object")) & completed
                if target and (sub is None or sub not in _subphases(match.group("object"))):
                    findings.append(
                        f"{where(number, section)}: live directive sends the reader back to "
                        f"{'/'.join(sorted(t.upper() for t in target))}, which the declaration "
                        f"records as complete: {sentence.strip()[:100]}")
            if not NOMINATION_RE.search(sentence):
                continue
            if sub is not None and (sub in named or sub in _subphases(following)):
                continue
            stale = named & completed
            if stale:
                findings.append(
                    f"{where(number, section)}: names completed "
                    f"{'/'.join(sorted(s.upper() for s in stale))} as the next or live task: "
                    f"{sentence.strip()[:100]}")
            elif sub is not None and not named and (parent := next(
                    (p for p in parents if p in _norm(sentence)), None)):
                findings.append(
                    f"{where(number, section)}: nominates only the coarse parent phase "
                    f"{parent!r}, which is partially complete, so a cold start is not told to "
                    f"do {sub.upper()}: {sentence.strip()[:100]}")
    return findings


def preflight_order_findings(doc: str, body: str) -> list[str]:
    """Findings where a full-preflight command block runs pytest before the fetch."""
    findings = []
    for block in re.findall(r"^```(?:bash|sh)?\n(.*?)^```", body, re.M | re.S):
        if PREFLIGHT_ANCHOR not in block or PREFLIGHT_TESTS not in block:
            continue
        if PREFLIGHT_FETCH not in block:
            findings.append(f"{doc}: preflight block runs pytest without first running "
                            f"{PREFLIGHT_FETCH}; a fresh clone fails on the ignored fixture")
        elif block.index(PREFLIGHT_FETCH) > block.index(PREFLIGHT_TESTS):
            findings.append(f"{doc}: preflight block runs {PREFLIGHT_FETCH} after pytest, "
                            "so the first run still fails on the absent fixture")
    return findings


def resolve(params: dict, path: str):
    node = params
    for part in path.removeprefix("params.").split("."):
        node = node[part]
    return node


def spec_ids(text: str) -> tuple[dict[str, set[int]], int]:
    """All declared IDs (live + retired), and the live count `gen_spec_views` reports."""
    ids: dict[str, set[int]] = {}
    live = 0
    for line in text.splitlines():
        if m := REQ_RE.match(line):
            live += 1
        elif not (m := TOMBSTONE_RE.match(line)):
            continue
        ids.setdefault(m.group(1), set()).add(int(m.group(2)))
    return ids, live


def blocks(body: str) -> list[tuple[int, str]]:
    """Split a document into (first-line-number, text) blocks.

    Line-by-line checking is wrong here: an amendment's back-reference routinely
    sits two lines below the superseded value it explains, and a dated session-log
    entry wraps over several lines. Both would read as unlabelled staleness. A
    block is a paragraph or a list item -- the unit a human actually reads -- so
    the back-reference only has to be *near* the value, not on the same line.
    """
    out: list[tuple[int, str]] = []
    cur: list[str] = []
    start = 1
    for i, line in enumerate(body.splitlines(), 1):
        starts_item = bool(re.match(r"^\s*[-*|]\s|^#|^\|", line))
        if (not line.strip() or starts_item) and cur:
            out.append((start, "\n".join(cur)))
            cur, start = [], i
        if line.strip():
            if not cur:
                start = i
            cur.append(line)
    if cur:
        out.append((start, "\n".join(cur)))
    return out


def discover_documents() -> list[str]:
    """Return the complete AM-76 current-document scope."""

    if DOCS is not None:
        return list(DOCS)
    paths = set(REPO.glob("*.md"))
    paths.update((REPO / "configs").rglob("*.md"))
    paths.update((REPO / "docs").rglob("*.md"))
    paths.update((REPO / "spec" / "evidence").rglob("*.md"))
    paths.add(REPO / "spec" / "SPEC.md")
    paths.add(REPO / "requirements.txt")
    paths.update(REPO.glob("requirements*.in"))
    return sorted(
        str(path.relative_to(REPO))
        for path in paths
        if path.is_file()
    )


def _is_plan(path: str) -> bool:
    parts = Path(path).parts
    return "plans" in parts and Path(path).suffix == ".md"


def historical_plan_error(path: str, body: str) -> str | None:
    """Validate the exact marker, visible banner and root-NEXT resolving link."""

    lines = body.splitlines()
    if not lines or lines[0] != HISTORICAL_MARKER:
        return "missing exact opening historical-plan marker"
    if len(lines) < 2:
        return "missing exact visible historical-snapshot banner"
    match = HISTORICAL_BANNER_RE.fullmatch(lines[1])
    if match is None:
        return "historical-plan banner text is not exact"
    target = (REPO / path).parent / match.group("target")
    if target.resolve() != (REPO / "NEXT.md").resolve():
        return "historical-plan NEXT.md link does not resolve to repository root"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true", help="list passing checks too")
    args = ap.parse_args()

    params = yaml.safe_load(PARAMS.read_text())
    spec_text = SPEC.read_text()
    ids, n_live = spec_ids(spec_text)
    documents = discover_documents()
    findings: list[str] = []
    text: dict[str, str] = {}
    excluded_historical = 0
    for doc in documents:
        path = REPO / doc
        if not path.exists():
            continue
        body = path.read_text()
        if _is_plan(doc):
            banner_error = historical_plan_error(doc, body)
            if banner_error is None:
                excluded_historical += 1
                continue
            findings.append(f"{doc}:1: historical-plan banner finding: {banner_error}")
        text[doc] = body
    doc_blocks = {d: blocks(b) for d, b in text.items()}
    n_am = len(ids.get("AM", set()))
    rec = json.loads(PACKET_RECORD.read_text()) if PACKET_RECORD.exists() else None

    passed: list[str] = []
    passed.append(
        f"current-document discovery ({len(text)} scanned, "
        f"{excluded_historical} valid historical plans excluded)"
    )

    # The operational cursor check is deliberately separate from general document
    # discovery: instructions/ is mostly append-only phase history, but RESUME.md
    # is too important to omit from this one live-state invariant.
    cursor_documents: dict[str, str] | None = None
    if DOCS is None:
        discovered_cursors = {
            doc: (REPO / doc).read_text()
            for doc in PASCAL_CURSOR_DOCS
            if (REPO / doc).is_file()
        }
        # Small synthetic trees used by the older document-rule tests contain
        # only NEXT.md; the real/default tree and any cursor-focused fixture
        # contain the high-value operational files and therefore enter this
        # check (including a missing-file finding when one is absent).
        if len(discovered_cursors) > 1:
            cursor_documents = discovered_cursors
    elif DOCS is not None and all(doc in text for doc in PASCAL_CURSOR_DOCS):
        cursor_documents = {doc: text[doc] for doc in PASCAL_CURSOR_DOCS}
    if cursor_documents is not None:
        findings.extend(pascal_cursor_findings(cursor_documents))
        passed.append("Pascal G8_C current execution/next-gate cursors agree")

    # --- 1. Superseded values must carry the amendment that superseded them ----
    #
    # (pattern, what it is, the AM that made it stale). A match fails unless the
    # line cites AM-<n> for some n >= that amendment, or is itself a history line.
    grid = params["channel"]["test_snr_grid_db"]
    tsnr = params["channel"]["train_snr_db_fixed"]
    low = len([x for x in grid if x <= tsnr])
    # `fixed_by` is the amendment that made the value WRONG, not the one that later
    # corrected the prose -- because that is what an honest back-reference cites.
    # AM-52 enlarged the grid, so everything derived from the grid became stale
    # there, even though AM-57 and AM-59 are what propagated the corrections.
    stale = [
        (r"\b13 points\b", f"low-region point count (now {low})", 52),
        (r"\b11 candidate runs\b", f"H1 candidate runs (now {low - 2})", 52),
        (r"1\.7 ?[×x] ?10⁻⁴", "H1 union bound (now 2.19e-4)", 52),
        (r"\b18 aggregated\b", f"aggregated SNR points (now {len(grid)})", 52),
        (r"\b18 SNR points\b", f"grid size (now {len(grid)})", 52),
        (r"\b424k\b", "classifier forwards (now 494,550)", 52),
        (r"\b212k\b", "JPEG 2000 decodes (now 247,275)", 52),
        (r"\b2\.07\b", "ER-1 one-ratio projection (now 2.42 h)", 52),
        (r"\b4\.14\b", "ER-1 two-ratio projection (now 4.83 h)", 52),
        (r"3\.9 h to 4\.1 h", "ER-1 projection range", 52),
        (r"\b42,666\b", "canonical payload A (now 42,624)", 58),
        (r"K_r = 7135", "canonical K' (now 7,132)", 58),
        (r"\b609 filler\b", "canonical filler (now 612/block, 3,672 total)", 58),
        (r"(?:same )?six clamps|all six clamped", "min-rate clamp count (now three)", 58),
        (r"smallest_ratio_at_least_two_rungs", "low_ratio_rule value", 59),
        (r"uniformly at random from the dataset's classes", "outage policy", 58),
        (r"not in torchvision", "false torchvision claim", 58),
        (r"the test split is touched once", "DEC-12 wording", 58),
        (r"third: 16\b|W4 / W10 / W16", "third review week (now 17)", 59),
        (r"until G-10 closes|G-10 at W10|test_access_gate.*G-10", "test-release gate (now G-12)", 60),
        (r"registration status is unverified|registration.*unverified",
         "proposal registration (confirmed complete)", 63),
        (
            r"config_hash.*(?:canonical JSON of|SHA-256 over).*resolved configuration",
            "partial run fingerprint",
            72,
        ),
        (
            r"torch==2\.13\.0(?!\+)|torchvision==0\.28\.0(?!\+)|"
            r"CPU-only.*(?:PyPI|public index)",
            "false CPU-only lock claim",
            73,
        ),
    ]
    for doc, bl in doc_blocks.items():
        for ln, block in bl:
            if HISTORY_LINE_RE.match(block):
                continue
            # Intersected with the real AM set: a nonexistent or mistyped AM-n must
            # not grant an exemption, or the back-reference rule can be defeated by
            # citing anything at all.
            cited = {int(n) for n in AM_REF_RE.findall(block)} & ids.get("AM", set())
            for pat, what, fixed_by in stale:
                if re.search(pat, block) and not any(c >= fixed_by for c in cited):
                    hit = next(l.strip() for l in block.splitlines() if re.search(pat, l))
                    findings.append(
                        f"{doc}:{ln}: superseded {what} with no AM-{fixed_by}+ back-reference\n"
                        f"      {hit[:120]}")
    passed.append(f"superseded-value back-references ({len(stale)} rules)")

    # --- 2. Requirement and amendment counts agree across the summaries -------
    # Counts appear far more often as history ("122 -> 158", "took it to 97",
    # "closed at 159") than as current claims, so this whitelists *currency* rather
    # than blacklisting past tense -- a blacklist lets a stale live count hide
    # behind any stray verb, which is the failure this whole tool exists to catch.
    # Judged on a LOCAL window, not the whole block: AGENTS.md's status paragraph is
    # a single block carrying both "it now carries 169" and "it was revised (97)",
    # so block-level currency cannot tell them apart.
    CURRENT = re.compile(
        r"now carries|passes at|currently|expect:|--check|"
        r"records \*\*(?:\w+\*\* amendment rounds|\w+ amendment rounds\*\*)"
    )
    HISTORY = re.compile(r"\bwas\b|\bwere\b|closed at|previously|took it to|→|->|~~|Result:|"
                         r"round closed|Before that")
    COUNTS = [(re.compile(r"\b(\d+)\s+requirements\b"), n_live, "requirements"),
              (re.compile(r"\b(\d+)\s+(?:are\s+)?`?AM`?\s+(?:amendment )?(?:records|entries)"),
               n_am, "AM entries")]
    for doc, bl in doc_blocks.items():
        for ln, block in bl:
            for rx, truth, label in COUNTS:
                for m in rx.finditer(block):
                    if int(m.group(1)) == truth:
                        continue
                    window = block[max(0, m.start() - 110):m.end() + 60]
                    if HISTORY.search(window) or not CURRENT.search(window):
                        continue
                    findings.append(f"{doc}:{ln}: claims {m.group(1)} {label} as current, "
                                    f"SPEC.md has {truth}")
    passed.append(f"requirement count ({n_live}) and AM count ({n_am})")

    # --- 3. Every referenced ID exists ---------------------------------------
    for doc, body in text.items():
        for i, line in enumerate(body.splitlines(), 1):
            for n in AM_REF_RE.findall(line):
                if int(n) not in ids.get("AM", set()):
                    findings.append(f"{doc}:{i}: references AM-{n}, which does not exist")
            for n in GATE_REF_RE.findall(line):
                if int(n) not in ids.get("G", set()):
                    findings.append(f"{doc}:{i}: references G-{n}, which is not defined in §13")
    passed.append("every AM-n and G-n reference resolves")

    # --- 4. params citations in hand-written docs resolve ---------------------
    # SPEC.md's own citations are checked by gen_spec_views.py; this covers the
    # files it does not look at, where a renamed key would otherwise rot quietly.
    # `params.x.y` is the placeholder AGENTS.md uses when explaining the citation
    # syntax itself, so it is not a citation.
    PLACEHOLDERS = {"params.x.y", "params.x", "params.foo.bar", "params.generated.yaml"}
    for doc, body in text.items():
        if doc == "spec/SPEC.md":
            continue
        for i, line in enumerate(body.splitlines(), 1):
            for path in CITE_RE.findall(line):
                if path in PLACEHOLDERS:
                    continue
                try:
                    resolve(params, path)
                except (KeyError, TypeError):
                    findings.append(f"{doc}:{i}: cites undefined `{path}`")
    passed.append("params citations outside SPEC.md resolve")

    # --- 5. Evidence record agrees with the prose that quotes it --------------
    if rec:
        feas = [r for r in rec["configurations"] if r["feasible"]]
        clamps = [r for r in feas if r["clamped"]]
        canon = next((r for r in rec["configurations"]
                      if r["tag"] == "imagenette160/r_1_3/qpsk/5/6"), None)
        if rec.get("failures"):
            findings.append(f"packetisation record carries {len(rec['failures'])} failures")
        if canon and canon["A"] % 8:
            findings.append(f"packetisation record canonical A={canon['A']} is not byte-aligned")
        for doc, body in text.items():
            for i, line in enumerate(body.splitlines(), 1):
                if HISTORY_LINE_RE.match(line):
                    continue
                m = re.search(r"min-rate clamp\w*\s*(?:from six to |now )?(\w+)", line)
                if m and m.group(1) in ("six", "6") and len(clamps) != 6:
                    findings.append(f"{doc}:{i}: says six clamps, record has {len(clamps)}")
        passed.append(f"evidence record: {len(feas)} feasible, {len(clamps)} clamped, "
                      f"{len(rec.get('failures', []))} failures")

    # --- 6. NEXT.md's live sections agree with its declared phase -------------
    if NEXT_DOC in text:
        phase_findings = next_phase_findings(text[NEXT_DOC])
        findings.extend(phase_findings)
        frontier, token, done = declared_phase(text[NEXT_DOC])
        passed.append(
            f"NEXT.md current-phase agreement (frontier {token.upper() if token else '?'}, "
            f"{len(done)} completed subjects, {len(live_lines(text[NEXT_DOC]))} live lines)"
        )

    # --- 8. No live section nominates a completed or coarse phase -------------
    # Runs over every scanned document, with NEXT.md as the one declaration, so a
    # README paragraph cannot quietly contradict the hand-off either.
    if NEXT_DOC in text:
        for doc, body in text.items():
            findings.extend(subphase_findings(doc, body, text[NEXT_DOC]))
        sub = declared_subphase(text[NEXT_DOC])
        passed.append(
            f"live sub-phase nominations (frontier {sub.upper() if sub else 'none declared'}, "
            f"completed {', '.join(sorted(p.upper() for p in completed_subphases(text[NEXT_DOC]))) or 'none'})"
        )

    # --- 7. Full-preflight blocks materialize the fixture before pytest -------
    for doc, body in text.items():
        findings.extend(preflight_order_findings(doc, body))
    passed.append("preflight blocks fetch the ignored LDPC fixture before pytest")

    # --- report ---------------------------------------------------------------
    if args.verbose:
        for p in passed:
            print(f"  ok: {p}")
    if findings:
        print(f"{len(findings)} documentation consistency finding(s):")
        for f in findings:
            print(f"  - {f}")
        return 1
    print(f"ok: {len(text)} current hand-written documentation files consistent "
          f"with the spec "
          f"({n_live} requirements, {n_am} amendments)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
