#!/usr/bin/env python3
"""Cross-document consistency check: do the hand-written docs still agree with the spec?

`gen_spec_views.py --check` validates `SPEC.md` against itself and regenerates the
derived views. Nothing validated the *hand-written* files -- `README.md`,
`AGENTS.md`, `NEXT.md`, `docs/`, `spec/evidence/README.md` -- against it. Three
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

# Hand-written files. The generated views under spec/ are excluded: they are
# reproduced from SPEC.md and `gen_spec_views.py --check` already guards them.
DOCS = ["README.md", "AGENTS.md", "NEXT.md", "CLAUDE.md", "requirements.txt",
        "spec/SPEC.md", "spec/evidence/README.md", "docs/crossover-explained.md"]

REQ_RE = re.compile(r"^- \*\*([A-Z]+)-(\d+)\*\* — ")
TOMBSTONE_RE = re.compile(r"^- ~~\*\*([A-Z]+)-(\d+)\*\*~~ — ")
CITE_RE = re.compile(r"`(params\.[A-Za-z0-9_.]+)`")
AM_REF_RE = re.compile(r"\bAM-(\d+)\b")
GATE_REF_RE = re.compile(r"\bG-(\d+)\b")
# Lines that ARE the append-only record, rather than prose about it.
HISTORY_LINE_RE = re.compile(r"^\s*[-*]\s+~?~?\*\*(AM|G|SR|BR|ER|DR|HR|PR|OPT|FW|DEC)-\d+\*\*|"
                             r"^\*\*Amendment round|^\s*[-*]\s+\*\*20\d\d-\d\d-\d\d")


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true", help="list passing checks too")
    args = ap.parse_args()

    params = yaml.safe_load(PARAMS.read_text())
    spec_text = SPEC.read_text()
    ids, n_live = spec_ids(spec_text)
    text = {d: (REPO / d).read_text() for d in DOCS if (REPO / d).exists()}
    doc_blocks = {d: blocks(b) for d, b in text.items()}
    n_am = len(ids.get("AM", set()))
    rec = json.loads(PACKET_RECORD.read_text()) if PACKET_RECORD.exists() else None

    findings: list[str] = []
    passed: list[str] = []

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

    # --- report ---------------------------------------------------------------
    if args.verbose:
        for p in passed:
            print(f"  ok: {p}")
    if findings:
        print(f"{len(findings)} documentation consistency finding(s):")
        for f in findings:
            print(f"  - {f}")
        return 1
    print(f"ok: {len(DOCS)} hand-written docs consistent with the spec "
          f"({n_live} requirements, {n_am} amendments)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
