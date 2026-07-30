# Resume ledger — PA / PB_1 / PB_2 / PB_3

**This file is the single source of truth for where the four-phase sequence stands.**
It is committed, so it survives a session dying mid-step. Prose in `NEXT.md` is a hand-off summary;
this file is the operational cursor. If they disagree, this file is right about progress and
`NEXT.md` needs updating.

Read this before anything else. Update it in the same commit as the work it describes.

## Rules

1. **Commit at every checkpoint.** Never let more than one checkpoint's worth of work sit
   uncommitted. Sessions end abruptly and without warning.
2. **Mark a step `in-progress` and commit that *before* starting it** if the step is long or
   expensive (a campaign, a sweep, a multi-file refactor). A crash then leaves evidence of where
   you were, not silence.
3. **Work-in-progress commits go on `main` with a `wip(<phase>):` prefix.** They are expected to be
   non-green. Push every one — an unpushed commit is not durable. Do not rewrite pushed history.
4. **A phase ends with one green commit** using its real conventional-commit message, after which
   every step below is `done` and the full verification block passes.
5. **Never `git reset`, `git checkout --`, `git clean`, or stash to "tidy up" on cold start.**
   Uncommitted changes are the previous session's unfinished work. Inspect and finish or commit
   them; do not discard them.
6. **Record observed facts here** (test counts, selected classes, SHAs) so the next session does not
   pay to re-derive them. Mark anything not yet re-verified.

## Cold-start protocol

    git status --short
    git log -12 --oneline
    git rev-parse HEAD && git fetch origin && git rev-parse origin/main

Then:

* If HEAD ≠ origin/main, or the worktree is dirty, reconcile that first. Uncommitted work belongs
  to the step marked `in-progress` below — read the diff before touching it.
* If the top commit is a `wip(...)` commit, the phase is mid-flight. Resume at the first step below
  that is not `done`.
* If a step is marked `in-progress`, do **not** assume its outputs are correct. Re-verify that one
  step's outputs from scratch, then continue.
* Only steps marked `done` may be trusted without re-checking.

---

## Status

**Current phase:** PA — in progress (fresh run, started at A1)
**Last green commit:** `82f6c569f792bf17ff28acd80ed1d516adfc06fa` (`fix(ldpc): make G-2 tools directly executable`)
**Next action:** run `instructions/PA.txt` step A3 (full preflight) — mark it `in-progress` and commit that first

---

## PA — recover and harden post-G-2 state

| Step | State | Notes |
|---|---|---|
| A1 establish exact state | done | fresh run; no prior `wip(handoff)` commits, clean worktree, HEAD = origin/main |
| A2 fixture workflow + docs | done | fetch-tool audit (do NOT re-audit): **already present** — pinned-asset-only fetch, complete-asset SHA-256 vs `baseline.ldpc_golden_vector_asset_sha256`, produces the ignored `.npz`, records source rung 2, never writes third-party bytes anywhere tracked (`.gitignore:157` + `/data/*`). **Added** — (a) `.npz`-absent guard: was gated on the *asset tarball*, now a network-free no-op when the fixture exists, `--force` to re-materialize; (b) inner-archive verification widened from encoder-only to *every* pinned archive in `baseline.ldpc_golden_vector_sha256` (encoder + rate_matcher + segmenter). Docs: fetch+verify_g2 lines added to `AGENTS.md`, fetch line added to `README.md`, both with the fresh-clone rationale. Offline floor left ungated. |
| A3 complete preflight | in-progress | 14-command block + `tests/test_test_access.py`; results land in the facts table as each command returns. Do not trust partial results — re-run any command whose row is absent. |
| A4 repair hand-off + consistency check | not-started | |
| A5 G-2 source provenance manifest | not-started | |
| A5b G-2 mutation tests | not-started | |
| A6 green commit + push | not-started | |

## PB_1 — classical transport path

| Step | State | Notes |
|---|---|---|
| B1.0 confirm PA green | not-started | |
| B1.1 `channel_transport.py` | not-started | |
| B1.2 `pipeline.py` | not-started | |
| B1.3 accounting + failure taxonomy | not-started | |
| B1.4 required tests | not-started | |
| B1.5 mutation tests | not-started | |
| B1.6 bounded executions | not-started | |
| B1.7 green commit + push | not-started | |

## PB_2 — outage, records, smoke evidence

| Step | State | Notes |
|---|---|---|
| B2.0 confirm PB_1 green | not-started | |
| B2.1 `outage.py` + validation selection | not-started | |
| B2.2 `records.py` + identities | not-started | |
| B2.3 smoke runner + configs | not-started | |
| B2.4 `verify_w4_baseline_integration.py` v1 | not-started | |
| B2.5 required + mutation tests | not-started | |
| B2.6 bounded executions + evidence | not-started | |
| B2.7 green commit + push | not-started | |

## PB_3 — BR-4 selection infrastructure + W4 adjudication

| Step | State | Notes |
|---|---|---|
| B3.0 confirm PB_2 green | not-started | |
| B3.1 `composition.py` arithmetic | not-started | |
| B3.2 BLER lookup + support guard | not-started | |
| B3.3 candidate cache + tie-break | not-started | |
| B3.4 system modes + two-pass limit | not-started | |
| B3.5 sweep budget guard | not-started | |
| B3.6 required + mutation tests | not-started | |
| B3.7 adjudication evidence | not-started | |
| B3.8 spec bookkeeping judgment | not-started | |
| B3.9 green commit + push | not-started | |

---

## Observed facts (carry forward; re-verify anything marked stale)

| Fact | Value | Observed at | Verified |
|---|---|---|---|
| A1 HEAD | `174cf19bfa2b10cb89d85211ab330e5cd8251de0` | A1 | yes |
| A1 origin/main | `174cf19bfa2b10cb89d85211ab330e5cd8251de0` | A1 | yes |
| A1 worktree | clean; `git diff --check` clean | A1 | yes |
| adjudicated fixture SHA-256 (local, matches `results/baseline/g2/golden_vector_summary.json`) | `55754b508ab1b6eb6625eae301d2d0a3fefcdf7b03e98038264b76b71e26aae0` | A2 | yes |
| fixture regeneration reproducibility | re-materializing from the pinned asset reproduces all 6 arrays and the same `.npz` SHA-256 | A2 | yes |
| A1 divergence from PA.txt expected HEAD | none material — `174cf19` is docs-only (`chore: add instructions for post W3`, adds `instructions/` only, 5 files / +1443 lines); last code commit is still `82f6c56` | A1 | yes |
| total tests | — | — | |
| `tests/test_test_access.py` count | — | — | |
| CUDA available | — | — | |
| selected outage class | — | — | |
| outage class measured val accuracy | — | — | |
| W4 implementation commits | — | — | |
