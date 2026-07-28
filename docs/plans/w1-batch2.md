# W1 Batch 2 — config plumbing and the SR-1 literal lint

**Status:** approved, not started · **Written:** 2026-07-28, immediately after batch 1 landed as
`e90a1e0` · **Owner:** agent, with two author-only items noted at the end.

**This is a point-in-time plan, not a maintained document.** It records what was intended and why,
against the repository as it stood on 2026-07-28 at 175 requirements. It is deliberately **not** in
`tools/check_doc_consistency.py`'s watched list — a plan that had to be kept current with every
later amendment would either rot or stop being a record of intent. If a number here disagrees with
`spec/SPEC.md`, **the spec wins and this file is simply older**. Once batch 2 is done, mark it done
rather than editing it into agreement.

---

## Context

### Where the project is

A solo graded capstone on **semantic communication / deep joint source-channel coding (DJSCC)**:
train an image encoder → noisy channel → decoder end-to-end, optimising *classification accuracy
after noise* at a fixed bandwidth budget, and compare it against a properly-tuned classical
JPEG 2000 + LDPC pipeline on the identical task, SNR and bandwidth budget.

`spec/SPEC.md` is normative and self-sufficient — **175 requirements**, decisions `DEC-1`..`DEC-16`,
and an append-only amendment record in §17 (`AM-1`..`AM-67`). Read §17 before re-litigating anything.
W0 is closed (gate G-9 passed). The current front is **W1**, which ends at gate **G-1**.

**Batch 1 landed on 2026-07-28 as commit `e90a1e0`** (GPG-signed by the author): the environment lock
(`requirements.in`/`.lock` plus a CPU-only pair), `pyproject.toml`, `.gitignore` entries,
`src/config/params.py`, `src/env.py`, `tests/test_env.py`, `tests/test_doc_consistency.py`, and
amendments `AM-65`..`AM-67`. torch 2.13.0+cu130 is installed and asserted on an RTX 4060 Laptop.

### What batch 2 is for

**SR-1** — *"Every run MUST be fully determined by a configuration file derived from
`params.generated.yaml`; no experiment-affecting constant may be hard-coded in source."*
Verify clause: *"a unit test asserting config round-trip, plus a lint rule flagging numeric SNR/k
literals outside `src/config/` and tests."*

Both halves are unbuilt. This batch builds them, and it comes second because **everything later
imports it** — the preprocessing contract, the dataset registry, the split manifests, the reference
classifier and every training run all read their constants through this layer.

It is also the last cheap moment. Once results exist, a constant that was hard-coded rather than
configured is no longer a lint finding; it is a number in the thesis whose provenance cannot be
reconstructed, and **ER-7 requires every thesis number to resolve to a committed artifact**.

### Why this is not busywork

`SPEC.md` §16 records that several `params` values are still *provisional* and will move at a gate:
`crossover_ratio`, `efficiency_ratio`, `low_ratio_operating_point`, `lambda_core`, and the
`clean_acc_floor`s. When `crossover_ratio` moves at G-8, every downstream run must move with it.
That is automatic if code reads `params.bandwidth.crossover_ratio`, and it is a silent
inconsistency — the worst kind, because everything still runs — if a `1/3` was typed into a source
file. The lint exists specifically to make that impossible rather than merely discouraged.

---

## Cold-start orientation (read this if you have no prior context)

```bash
cd /home/nick/projects/capstone

# the four checks. All four must pass before you start and before you commit.
.venv/bin/python tools/gen_spec_views.py --check        # expect: 175 requirements (2 retired)
.venv/bin/python tools/check_doc_consistency.py         # expect: 8 hand-written docs consistent
.venv/bin/python spec/evidence/check_packetisation.py   # expect: 215 feasible, 0 failures
.venv/bin/python -m pytest                              # expect: 18 passed
git status --short                                      # expect: clean
```

**Conventions that are not optional here:**

- `spec/SPEC.md` is hand-written and authoritative. `spec/DATASHEET.md`, `spec/concerns/*.md` and
  `spec/params.generated.yaml` are **generated from it** — edit `SPEC.md`, then run
  `tools/gen_spec_views.py`. Never edit a generated file.
- **Every spec change gets an `AM-n` entry in §17** stating what changed and why, plus an `(AM-n)`
  back-reference on the item changed. §17 is **append-only**: an amendment that is later revisited
  gets a *new* entry citing the old one; superseded entries stay wrong in place, on purpose.
- Two `AM` formatting traps, both already paid for once: an entry must be a **single line** however
  long (the parser is anchored and single-line; a wrapped entry fails silently and surfaces as a
  confusing ID-contiguity error), and a `` `params.x.y` `` written inside an entry **is parsed as a
  real citation and must resolve** — describe a rename in plain text, not in backticked form.
- Adding a new `params` section requires some requirement to cite it, or `--check` fails on "every
  parameter section cited by some requirement".
- `check_doc_consistency.py` enforces that a **superseded value may appear only in a block citing the
  amendment that superseded it**. If an amendment here supersedes a value that appears in prose, add
  a rule to that script's `stale` table — the table is the tool's memory and is meant to grow.
- Commit messages cite requirement IDs. The author commits with a GPG key; **do not commit for
  them** unless asked — batch 1's automated commit failed on a pinentry timeout and the author
  completed it themselves.

**Existing code to reuse, not reimplement:**

| Thing | Where | What it gives you |
|---|---|---|
| `load_params()`, `get("environment.lock_file")`, `REPO_ROOT`, `PARAMS_PATH` | `src/config/params.py` | Cached params tree and a dotted-path accessor that tolerates a `params.` prefix and raises `KeyError` naming the *full* path |
| `assert_cuda()`, `set_deterministic_backend()`, `environment_record()` | `src/env.py` | The AM-23 CUDA trap, SR-12 determinism, and the six SR-21 run-metadata fields |
| Checker skeleton — `argparse`, module-level `REPO` constant, `findings`/`passed` lists, `main() -> int`, exit 1 on any finding, `-v` to list passes | `tools/check_doc_consistency.py` | Copy this shape for `tools/check_literals.py`; `tests/test_doc_consistency.py` shows how to test such a tool by monkeypatching its module globals |

---

## Findings from planning — read before designing anything

**`params.artifacts.run_id_key` names five fields that no parameter defines.** Verified by walking
the whole generated params tree: `config_hash`, `checkpoint_id`, `dataset_version`,
`classifier_variant` and `analysis_version` appear only as *strings inside the key list*.

- `config_hash` and `checkpoint_id` are runtime-computed by design — correct as-is.
- `classifier_variant` is implied by `params.reference_classifier.artifact_finetuned_variant_required`
  (BR-12's two variants) but is never enumerated.
- **`dataset_version` and `analysis_version` have no defined source at all.** Batch 3 builds `run_id`
  from this key list (SR-18) and cannot do so while two components are undefined. Settle them here,
  in the config layer that will supply them.

The clean resolution for `dataset_version` reuses machinery that already exists: SR-20 already
requires each dataset's `archive_sha256` to be recorded at fetch, and that checksum is *precisely*
what identifies the data. So `dataset_version` is a rule, not a new value.

**Do not treat this as licence to open a spec audit.** `NEXT.md` records a deliberate decision to
stop auditing the specification — every fatal finding landed in amendment rounds 0–3, and rounds 6
and 8 found only damage the fixing itself caused. These two gaps are in scope only because SR-1 and
SR-18 cannot be built around them.

---

## Decisions taken with the author (do not reopen)

1. **Experiment configs plus sweep axes.** A handful of committed YAML files under
   `params.config.dir`, each naming only the *choices* for one experiment. Everything else is derived
   from params at load time, and the fully-resolved per-run config is archived beside that run's
   results. Rejected: one committed file per run (thousands of unreviewable files, since the project
   spans 6 ratios × 3 train seeds × 3 channel seeds × ~11 systems × 21 SNR points), and
   code-only construction with no committed file (SR-1 says "a configuration file", and there would
   be nothing to review *before* a run).
2. **The lint hard-fails, with per-site annotated exceptions that must carry a reason.**
   `# literal-ok: <reason>` suppresses exactly one line and a reason-less annotation does not
   suppress. The checker prints the running total so exceptions cannot accumulate unnoticed.
   Rejected: no exceptions at all — the pressure would become "remove that value from the watched
   list", which weakens the check *globally* to fix a *local* collision. Note this is a different
   risk class from AM-67's rejected env-var escape hatch: that would have been set once and forgotten
   for the whole process; this is visible at the site it applies to and counted in the summary.
3. **This plan lives here**, with a pointer from `NEXT.md`'s "Do next".

---

## Files

### New — the config layer

| Path | What |
|---|---|
| `src/config/run_config.py` | `RunConfig` (frozen dataclass), `load_experiment(path, **overrides) -> RunConfig`, `config_hash(cfg) -> str`, `to_dict`/`from_dict`. |
| `src/config/experiments.py` *(only if needed)* | Resolution of symbolic choices — `bw_ratio: crossover_ratio` → the concrete rung, `lambda: lambda_core`. Fold into `run_config.py` if it stays small; do not create an empty module. |
| `configs/README.md` | What an experiment config may and may not contain: choices only, never derived values. |
| `configs/*.yaml` | Two to start — one learned, one classical — enough to exercise the loader. Full set arrives with the experiments that need them. |
| `tools/check_literals.py` | The SR-1 lint. |
| `tests/test_run_config.py` | Round-trip, hash stability, hash sensitivity, symbolic resolution. |
| `tests/test_check_literals.py` | Inject-and-assert, in the shape of `tests/test_doc_consistency.py`. |

### Design — `RunConfig`

A **frozen dataclass** holding the fully-resolved, experiment-affecting settings for one run.
Everything in `params.artifacts.run_id_key` that is a *setting* rather than an identity hash should
be reachable from it, because batch 3 builds `run_id` from exactly that list.

- **Resolution is explicit.** `bw_ratio: crossover_ratio` in the YAML resolves through
  `params.bandwidth.crossover_ratio` to a concrete rung at load. Store **both** — the symbolic choice
  and the resolved value — so that when G-8 moves `crossover_ratio`, the archived config of an old
  run still says what it meant *and* what it was.
- **`config_hash` = SHA-256 over the canonical JSON of the resolved config**, keys sorted, compact
  separators. This mirrors `params.artifacts.run_id_form`
  (`content_addressed_sha256_over_sorted_key_value_pairs`) rather than inventing a second convention.
  Hash the **resolved** values, so that a params change that alters a run alters its hash.
- **No I/O, no torch import.** Keep this layer importable by analysis and demo code on the CPU-only
  path (`params.environment.cpu_install_path_required_for`).
- Reuse `config.params.get()`; do not re-open the YAML.

### Design — `tools/check_literals.py`

**Watched values:** every numeric leaf in the params tree (walk `dict`s and `list`s alike), minus
`params.config.literal_lint_exempt_values`. Report the parameter path that a literal collides with,
not just the number — `literal 128 matches params.learned_system.encoder_body_channels` tells you
what to do; `literal 128 is magic` does not.

**Scope:** `src/`, excluding `src/config/`. Not `tests/`, not `tools/` — those are not experiment
code, and SR-1's verify clause names exactly this scope.

**Implementation:** walk each file with `ast`, collecting `ast.Constant` nodes with `int`/`float`
values.

> ⚠️ **The one implementation trap.** A negative literal is not a `Constant` — `-8` parses as
> `UnaryOp(USub, Constant(8))`. Handle `UnaryOp` explicitly or **every negative SNR in the grid
> slips through silently**, which is the largest single block of watched values
> (`params.channel.test_snr_grid_db` runs from −8 to +18). A lint that passes while missing the SNR
> grid is precisely the AM-58 failure mode: a passing check is not evidence that the thing checked
> is right.

**Annotations:** `# literal-ok: <reason>` on the offending line suppresses that site only. An empty
or missing reason does **not** suppress. Print the count of active annotations in the summary.

**Its own parameters must come from params** — the exempt set, the scope and the annotation token.
A lint against hard-coded constants that hard-codes its own constants would be self-refuting.

**Expect noise on first contact and tune deliberately.** Suggested starting exempt set is
`[-1, 0, 1, 2, 3]`: `-1` is idiomatic in reshapes, and `3` is both `image_size[2]` and the RGB
channel count that appears in every convolution. Any change to this set is a spec change with an
`AM` entry, not a quiet edit — that is the whole point of putting it in params.

### Spec amendments

Two entries, `AM-68` and `AM-69`, taking the count **175 → 177**. Add a new `config:` parameter
section cited by SR-1, and extend SR-18's or SR-2's citations for the identity fields.

- **`AM-68`** — SR-1 required a configuration file and named no location, no format, no hash rule and
  no lint parameters, so neither half of its own verify clause was executable. Adds
  `params.config.dir`, `run_config_hash_form`, `literal_lint_exempt_values`, `literal_lint_scope` and
  `literal_lint_annotation`; records the experiment-config-plus-sweep-axes shape and why one file per
  run was rejected.
- **`AM-69`** — `params.artifacts.run_id_key` names `dataset_version` and `analysis_version`, neither
  of which any parameter defines, so SR-18's `run_id` is not constructible as written. Adds
  `params.config.analysis_version` with a bump rule and a dataset-version rule derived from the
  SR-20 `archive_sha256`, which is already required to be recorded and is exactly what identifies the
  data. Note in the entry that `config_hash` and `checkpoint_id` are runtime-computed **by design**
  and are not gaps.

Also owed, and easy to forget: **`AGENTS.md`'s requirement count moves 175 → 177**, and
`check_doc_consistency.py` checks that claim, so it must change in the same commit.

---

## Order of work

1. `SPEC.md`: `AM-68`, `AM-69`, the `config:` section, SR-1 and SR-18 back-references. Regenerate
   views. Run `--check` and `check_doc_consistency.py` **before** writing any code — a params key
   that does not exist yet will fail the code that reads it, and the failure will look like a code
   bug.
2. `src/config/run_config.py`, then `tests/test_run_config.py`.
3. `configs/` — the README and two example configs.
4. `tools/check_literals.py`, then `tests/test_check_literals.py`.
5. Run the lint against the existing `src/` and tune the exempt set if it is unusable. Record any
   change as part of `AM-68` rather than as a silent edit.
6. Full verification; hand the commit to the author.
7. Update `NEXT.md` (session log, batch status) and mark this plan done.

---

## Verification

```bash
.venv/bin/python -m pytest                              # round-trip, hash, lint tests
.venv/bin/python tools/check_literals.py                # expect: 0 findings against current src/
.venv/bin/python tools/check_literals.py -v             # lists what it scanned and any annotations
.venv/bin/python tools/gen_spec_views.py --check        # expect: 177 requirements (2 retired)
.venv/bin/python tools/check_doc_consistency.py         # expect: 8 docs consistent
.venv/bin/python spec/evidence/check_packetisation.py   # unchanged: 215 feasible, 0 failures
git status --short
```

**Prove both tools bite; do not assume from a pass.** This repository has twice found a *passing*
check that was not checking what it claimed (AM-58's evidence script violated four of its own rules;
AM-62's doc checker granted exemptions for amendment numbers that did not exist). Batch 1 established
the habit — `assert_cuda` was run against a simulated CPU build, and the doc-consistency test was run
against a mutant checker with the AM-62 bug reintroduced, where **only** the exemption case failed —
and batch 2 should keep it:

- Add a temporary file `src/_scratch.py` containing a bare `snr = 7` and confirm the lint reports it
  against `params.channel.train_snr_db_fixed`; then add `# literal-ok:` with no reason and confirm it
  **still** reports; then add a reason and confirm it passes and is counted. Delete the file.
- Add a negative literal — `snr = -8` — and confirm it is caught. This is the `UnaryOp` trap above,
  and it is the single most likely defect in the whole batch.
- Mutate one field of a `RunConfig` and confirm `config_hash` changes; reorder the keys of the
  serialised form and confirm it does **not**.

---

## Risks

| Risk | Handling |
|---|---|
| The lint is unusably noisy against real model code | Expected on first contact. Tune `literal_lint_exempt_values`, and prefer *fixing* a collision (read the value from config) over widening the exemption. Widening is a spec change with an `AM` entry. |
| Negative literals slip through the `ast` walk | Explicitly handle `UnaryOp(USub, Constant)`. There is a dedicated verification step above; do not skip it. |
| `config_hash` is unstable across runs or Python versions | Canonical JSON with `sort_keys=True` and compact separators; no `hash()`, no `pickle`, no `repr` of floats. A hash-stability test is in scope. |
| Scope creep into batch 3 | `run_id`, `pair_id`, `noise_id`, the keyed RNG and the test-access guard are **batch 3** (SR-18, SR-22). Batch 2 supplies `config_hash` and the settings they key on, and stops there. |
| An amendment breaks `--check` in a way that looks like a code bug | Do the spec work first (step 1) and confirm all four checks pass before writing code that reads new params. |

## Explicitly out of scope

Batch 3 (`src/artifacts/ids.py`, `rng.py`, `src/data/test_access.py` — SR-18, SR-22, the two things
that cannot be retrofitted); the preprocessing contract (SR-19); dataset registry, splits and
manifests (SR-2, SR-17, SR-20); the reference classifier and G-1 itself (BR-8, DEC-15);
`sionna-no-rt` and anything LDPC (W3).

Non-code items that need the **author**, not an agent, and are tracked in `NEXT.md`: **PR-9**'s
hardware-alternative conversation with the guide (due before W4), and **PR-1** (literature review,
≥25 refs) and **PR-2** (Gantt with the real W4 / W10 / **W17** dates) — which together are 10 of the
First Review's 30 sub-marks and gate the review package alongside G-1.
