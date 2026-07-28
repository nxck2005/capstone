# Very Next Steps

**Working file for hand-off between sessions.** Short-lived, frequently rewritten, deliberately
scrappy. Read it first at the start of a session; update it before finishing one.

Not normative — `spec/SPEC.md` governs. If something here contradicts the spec, the spec wins and
this file is wrong. Anything here that turns out to be a durable decision belongs in `SPEC.md`
(as a `DEC`), a durable risk belongs in `SPEC.md` §16, and an explanation belongs in `docs/`.

**Last updated:** 2026-07-28 · **Phase:** **W1 — batches 1–3 done, preprocessing (SR-19) next** · spike executed,
three rounds of external review adjudicated and applied (AM-26..AM-55 in `8e65329`, AM-56 and the
docs sweep in `7b7c70a`, AM-57..AM-64 in `9d46d6d`), and **the first commit of project code this
session (AM-65..AM-67)**, followed by batch 2's AM-68..AM-70. `requirements.txt` is still tooling-only by design; the runtime stack now
lives in the hashed `requirements.lock` that SR-21 asked for, and it is installed. All checks pass:
`gen_spec_views.py --check` at **178 requirements**, `check_doc_consistency.py` across 8 hand-written
docs, and `check_packetisation.py` with **zero failures**. Pytest passes outside the two deliberately
unskipped GPU-runtime assertions; this agent environment blocks NVML/CUDA device access despite the
correct CUDA build, so those two require the primary-device run before the signed commit.

---

## Just landed — the Codex gate audit adjudicated, AM-57..AM-64 applied

**`audit/JUDGE_codex` (`EXT-6`)** returned **project GO, W1 NOGO — temporary hold**: 11 P0 findings,
10 P1s, 3 P2s. It is the most accurate review this project has received. **Every checkable numeric
claim reproduced exactly** when re-derived — the packetisation defect counts, the corrected canonical
case, the grid arithmetic, the runtime figures, and an H4 power calculation no earlier review
attempted. Verified independently rather than adopted: the packetisation against a separately written
solver, the base-graph rate floor and the modulation interleaver against the **pinned Sionna source**
rather than its docs, the review dates by rendering a scanned PDF that has no extractable text, and
the torchvision loader against current upstream docs.

**Exactly one claim was rejected, and one sharpened:**
- **Rejected — "`run_id` cannot pair systems".** It is a structured tuple key; pairing on every field
  except `system` is well defined. The *real* defect is the opposite: the key omitted `split`, the
  config and checkpoint hashes, the classifier variant and every system-specific setting, so distinct
  runs **collided** — validation with test, and the two BR-12 classifier variants with each other.
- **Sharpened — the ER-1 projection.** The audit was right that a 21/18 rescaling is not arithmetic,
  but the direction is the reverse of the obvious guess: the added points are at the **noisy** end
  where BR-4 picks BPSK, whose channel-bit budget is a quarter of 16-QAM's and which needs 2 code
  blocks rather than 11. The worst-case figure **overstates** there. Both ends are now recorded.

**Three decisions you made this session, don't reopen:** H3 keeps the **full-grid** slope and gains a
magnitude-contraction clause (the audit's high-SNR refit was rejected — it abandons the half of the
grid where the effect is largest); W10's rehearsal and the Second Review figure move to
**validation**, test stays sealed behind a freeze manifest until **G-12 at W11** (AM-60 — it briefly
read G-10, which after AM-59 sits at the *start of W9* and would have released the split three weeks
early); H4's power floor is declared and simulated **before the test split opens at G-12** rather
than fixed by more training runs (AM-70 — ER-9 does not exist until W9).

**The packetisation script reported zero failures while breaking four of its own rules** — that is
the finding worth carrying forward. 92/215 rows had a non-byte-aligned `A` under a solver whose own
parameter promises byte alignment; 21 had a non-integral `B'/C` silently rescued with `ceil`; 47/103
BG2 rows computed filler from the selection `K_b` instead of the encoded `K = 10Z`; and the rate
floor was tested *strictly* against the **smallest** `E_r` — the block least likely to fail — when the
library raises only on `r < floor` and BG1's mother code is exactly 1/3. Fixing all four keeps 215
configs feasible and **every** headline-dataset config feasible, moves capacity in 18 rows by −8 to
+1 byte, and cuts the clamps from **six to three**. Canonical case is now **A = 42,624 b (5,328 B),
B' = 42,792, K' = 7,132, Z = 352, filler 612/block and 3,672 total**. A repair, not a collapse.

**The proof obligation is now all 144 headline-dataset configs, not the 72 that today's three
provisional ratios name** — ER-3 can select any rung, so the old scope proved nothing about the
configuration actually used. That change immediately found something: **ER-9's admissible (dim, bits)
pairs fall to 1 at STL-10's `r_1_48`**, so at the ladder bottom its "two-stage validation search" has
one candidate. Recorded as a carried risk; G-11 must report the count.

**The circular was readable all along.** It is scanned images with no extractable text — which is not
the same as unreadable. Rendering its two pages resolves all four dates: First Review **18–22 Aug
(W4)**, Second **29 Sep–3 Oct (W10)**, Final **17–21 Nov (W17, not the W16 the spec carried)**, report
due **20 Nov — inside Final Review week**. W16 is now deliberately allocated contingency and W15
carries an internal freeze. Clause 5 also expects hardware or "significant design aspects with an
application to real world problems", so PR-9 is a full deployment dossier and the guide must be asked
before W4. ~~⚠️ **The repo's proposal DOCX is a blank template — PR-10 exists because registration
status is unverified.**~~ **Registration confirmed complete 2026-07-28 (AM-63)** — the blank
attachment proved nothing either way, which is why it was checked rather than assumed. That was the
only carried risk with no graceful degradation; PR-10 is closed.

---

## Previously — two external reviews adjudicated, 30 amendments applied, committed as `8e65329`

**Two independent full-spec reviews** (`EXT-4`, Claude; `EXT-5`, a second model) were adjudicated
against the spec and the W0 evidence, not deferred to. **Neither verdict was adopted as given.**
EXT-4 said "commit after seven edits" but its one external claim was false and it missed the three
worst defects; EXT-5 said "NO-GO/HOLD on the whole spec" but only *one* of its findings touched W1.
Result: 122 → **158 requirements**, `AM-26`..`AM-55`, split into two rounds in §17. A third round
followed on 2026-07-28 — `AM-56`, from a **self-audit of those two rounds**, which found that AM-53
had left H2 able to select its comparison window on a different curve from the one it evaluates.
That round closed at 159 requirements; the count is **172** after AM-57..AM-64 above. Worth noting as
a pattern rather than an embarrassment: every audit round so far — including the audit of the audit,
and including the round that audited a *passing* evidence script — has found something real.

**The three that mattered, all confirmed by recomputation, none of which either review found alone:**

1. **ER-9 was arithmetically impossible** (AM-55). Sharing `transmit_feature_dim` pinned it at 2k
   real values while the digital budget gives `Qm·R/2` bits per value — **0.167 at BPSK 1/3**
   against a 2-bit floor. Every config except 16-QAM 5/6 was infeasible, and that one only decodes
   above ~11 dB, *outside H1's region entirely*. So the H4 control would have sat at chance across
   exactly the range it is tested in — unfalsifiable, and flattering. Same class as AM-15.
   **Fixed:** ER-9 keeps the identical encoder and chooses its own `transmit_dim` on validation.
2. **TS 38.212 packetisation was non-conformant three ways** (AM-49): CRC24A applied
   unconditionally where 17 of 72 live configs are entitled to CRC16; `code_block_max_bits` 8448
   applied where 14 select BG2 (3840); and the base graph derived *after* segmentation from the
   per-CB rate, where the standard selects it once per TB from (A,R) *before*. **Blast radius is
   small** — segmentation changes in zero configs, so BR-3 and BR-10's zero-slack result survive and
   the cost is one byte in 17 cells. It is a defect in DEC-13's *claim*, not in any number.
3. **ER-10 promised a variance decomposition AM-17 had already made impossible** (AM-31) — zipped
   seeds alias training luck and channel luck. Now compound replicates.

Plus **G-8 was required to decide a crossover it cannot observe** (AM-33) — W6's sweep is
classical-only and no learned model exists until W7. New **G-10** decides it — placed at W10 then,
moved to the **start of W9** by AM-59.

**Three decisions you made this session, don't reopen:** ER-9 gets the *same encoder with its own
output width* (not a bolted-on layer, not fixed pooling); seeds are *compound replicates* (don't pay
3× ER-1 to restore the decomposition); BR-4 selection is *two passes then stop*.

**What was rejected, with reasons, so it doesn't come back:**
- **EXT-4's OCUDU claim is false** (AM-30). It said the successor ships the same vectors under
  BSD-3, so the archived-upstream risk could be closed. OCUDU publishes **no pre-generated vectors** —
  they moved to a MATLAB-companion plugin whose docs require a licensed MATLAB + 5G Toolbox. The
  risk is *worse* than recorded, and §16's old mitigation line ("take them from OCUDU") was also
  wrong and is deleted. ⚠️ **Action: the pinned srsRAN asset is still live (verified HTTP 200) —
  fetch and archive it locally, outside git, before W3.**
- **H1's run rule stays** (AM-32) — but AM-4's *bound* was wrong, and that is now recorded. Positive
  dependence does **not** sandwich the answer: four independent blocks of three, each all-or-nothing
  Bernoulli(0.025), gives ≈0.096, four times the "perfect dependence" figure. The conclusion holds
  under AR-1/exchangeable dependence, so ER-10 now *measures* it by sign-flip permutation instead of
  arguing it. EXT-5's proposed replacement already exists as `h1_effect_size` (AM-3).
- **BR-13's random outage draw stays** — it is reproducible from `channel_seed` and therefore
  identical across systems, which is what ER-10's pairing needs. Only its *reporting* changed.
- **EXT-4's rubric denominator was wrong**: Third Review is **60** sub-marks, not 55. Its
  conclusions still hold — `Objectives Met` really is 10, and PR-8 now forbids stating objectives as
  outcomes (AM-46).

**New evidence artifact:** [`spec/evidence/check_packetisation.py`](spec/evidence/check_packetisation.py)
— pure arithmetic, no GPU/network, runs in under a second, emits the per-config record BR-10 now
requires. Confirms zero slack across 215 feasible configs, **the same six BG1 clamps** (so AM-24
stands), BR-10's canonical case exact, and **ER-9 feasible at all 72 live configs** (7 options even
at the tightest, 94 bytes).

## In flight — nothing

**G-9 is closed. The LDPC spike passed all seven checks** (AM-24, AM-25). The environment lives in
`~/capstone-w0-spike/` and is reusable: `./run_spike.sh run` re-runs in seconds and regenerates
`g9_spike_record.json`. Python 3.14.6 · torch 2.13.0+cu130 · sionna-no-rt 2.0.1 · RTX 4060 Laptop 8 GB.

**Measured, now in the spec:** exact `E_r` across **all 180 configurations** (72 live), so BR-3 holds
against the library and not just on paper · 625.2 code-block decodes/s at 50 iterations, batch 32
(corrected from 634 by AM-29 — the old figure was faster than the committed evidence) ·
ER-1 projects to **~2.0 h at one ratio, ~4.1 h at two**, worst-case modulation — so **G-8's
one-ratio-or-two decision is not compute-constrained**, which was the open question AM-20 deferred ·
smallest workable payload 16 bits.

**Three defects the spike caught, which is what it was for:**

1. **LLR sign.** The library reads LLRs as log(p(x=1)/p(x=0)) — the *opposite* sign to `x = 1−2c`.
   Getting it backwards is **totally silent**: decoder runs, returns exactly k bits, raises nothing,
   BER 0.77 where the correct sign gives 0.00. At W3 this would have looked like "the classical
   baseline is weak" — i.e. it would have manufactured the result ER-8 forbids. Fixed at the BR-14
   seam as `params.baseline.ldpc_llr_convention`.
2. **Rate 1/3 did not exist at three live operating points.** `floor(G × rate)` lands at 0.333281
   against BG1's floor of 0.333333, and BG1 cannot go lower without repetition coding. One-bit clamp,
   applied *after* segmentation. In BR-10 as `params.baseline.ldpc_bg1_min_coderate`.
3. **Spelling.** Spec says `offset_min_sum`; the library only accepts `offset-minsum`. Mapped at the
   seam.

**Golden vectors — solved, and better than expected.** Sionna now agrees **bit-exactly with the
MATLAB-generated srsRAN vectors, zero mismatches**, across lifting sizes 2–288 on both base graphs.
BR-2 is no longer a plan, it is a demonstrated result. The alignment recipe is in BR-2 and is not
obvious — three wrong attempts agreed at 0.50, which is chance and looks exactly like a library bug.

**Evidence is now in the repo: [`spec/evidence/`](spec/evidence/).** The spike record, the scripts
that produced it, the golden-vector check and its log, and the fetch script plus checksums. Read its
`README.md` first. Two things it deliberately does *not* contain: the third-party `.dat` vectors and
srsRAN's `ldpc_encoder_test_data.h`, both AGPLv3 and both `.gitignore`d — run
`spec/evidence/fetch_srsran_vectors.sh` to obtain them. Note the README's variance caveat: re-running
the spike gives 625–663 cb/s, and the spec records the figure from the **committed run** (625.2)
because ER-7 requires every number to resolve to an artifact in the repo — it previously read 634,
which was faster than the evidence beside it (AM-29). The old scratch copies at `~/capstone-w0-spike/` are now redundant.

---

## Do next

### ⏰ Standing trigger — First Review package (check this every session)

**Fires when all three are true: PR-1 committed · PR-2 committed · G-1 passed.**
(`params.deliverables.review_1_ready_when`. As of 2026-07-28: **none of the three**.)

When it fires, do two things and do them in this order:

1. **Cut the snapshot tag** — a tag, deliberately not a branch (AM-64):
   ```bash
   git tag -a review-1-basis -m "State the First Review package is derived from"
   ```
2. **Tell Nick to start the review package.** He asked to be reminded, and it is not something he
   should have to remember on the right week. It lives in `deliverables/review-1/`.

**Why those three gate it.** The rubric scores the First Review as six criteria × 5 sub-marks = 30,
scaled to 10: Motivation · Objectives · Hypothesis · Problem Survey · Subject Knowledge · Time Plan.
**Problem Survey *is* PR-1 and Time Plan *is* PR-2** — a third of the review, with nothing behind it
today. G-1 is in the list because Subject Knowledge is assessed by general viva and Hypothesis by
whether a real proposal exists; a measured clean-accuracy number makes both concrete.

⚠️ **The one way to lose marks at W17 by what you say at W4.** PR-8 requires objectives stated in
§2's **completion** terms — build, validate, bandwidth-match, bit-account, evaluate at the
learned-blind operating point, report with paired inference — and **never** as an outcome such as
"show the learned system beats the classical one". The Third Review scores `Objectives Met` against
whatever is set here, so an outcome-phrased objective turns DEC-16's perfectly valid dominance
fallback into a visible failure. Check the objectives slide against §2 before it is shown.

Two smaller facts: the slot is **15 minutes**, so ~10–12 slides; and **Presentation is not a First
Review criterion** — it first appears at the Second — so these marks are for substance, not delivery.

---

### Cold-start: the first thing to do in a fresh session

**Nothing blocks W1.** The specification side is finished; what remains is code. Registration is
confirmed (AM-63), which was the only carried risk with no graceful degradation — every other failure
mode has a fallback ladder, a recorded alternative outcome, or a gate that catches it. **Do not open
another full-spec audit round.** The record says why: all fatal findings landed in rounds 0–3, the
last one being AM-55 (ER-9 infeasible); rounds 4–9 have been correctness, defensibility and process,
and rounds 6 and 8 found only damage the fixing itself caused. The two worst defects this project ever
had — the silent LLR sign at BER 0.77 and rate 1/3 not existing at three operating points — were found
by **running** the W0 spike, not by reading. G-1 and G-2 are the audits with teeth now.

**State on 2026-07-28, verified:** `src/`, `tests/` and both lockfiles exist, carrying batches 1–3
(`e90a1e0`, `2b23c1e`, `72be2af`); no `data/`, no `results/`, no `checkpoints/` yet.
`.venv` now holds the full runtime stack, installed from `requirements.lock`.
Machine: Python 3.14.6 · `uv` 0.11.32 at `/usr/sbin/uv` · RTX 4060 Laptop 8 GB · driver 592.82.
srsRAN vectors **already fetched** to `spec/evidence/srsran_vectors/` (276 files, gitignored).
Only IRL item still open: PR-9's hardware-alternative acknowledgement, which is not blocking — its
acceptance criterion fires when the dossier is delivered, and the conversation sits before W4.

Confirm nothing drifted, then start the preprocessing contract (SR-19):

```bash
.venv/bin/python tools/gen_spec_views.py --check           # expect: 178 requirements (2 retired)
.venv/bin/python tools/check_doc_consistency.py            # expect: 8 hand-written docs consistent
.venv/bin/python tools/check_literals.py                   # expect: 0 findings
.venv/bin/python spec/evidence/check_packetisation.py      # expect: 215 feasible, 144 obligation, 0 failures
.venv/bin/python -m pytest                                  # expect: 54 passed
git status --short                                          # expect: clean
```

#### GPU access — probe it, do not assume it either way

This is **WSL2**, so the GPU arrives through `/dev/dxg` and a Microsoft-supplied driver shim, not
through `/dev/nvidia*`. An agent that checks for the wrong device concludes there is no GPU on a
machine that has one. Run all three and report them verbatim:

```bash
ls -l /dev/dxg                                      # the WSL GPU device
/usr/lib/wsl/lib/nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
.venv/bin/python -c "import torch; print(torch.version.cuda, torch.cuda.is_available(), \
    torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'NONE')"
```

**Measured 2026-07-28:** `/dev/dxg` present as `crw-rw-rw- root 10,125`; the WSL `nvidia-smi` reports
`NVIDIA GeForce RTX 4060 Laptop GPU, 592.82`; torch reports `13.0 True`, and a real device matmul
succeeds. **The first two also succeed inside the Codex sandbox**, which earlier failed NVML — so
that restriction has lifted, at least for the device node.

**The third command is the only one that settles it.** A visible `/dev/dxg` and a working
`nvidia-smi` do **not** guarantee torch can initialise CUDA; that additionally needs `libcuda`
resolvable through the WSL shim. Report `torch.cuda.is_available()` explicitly rather than inferring
it from the other two. Two notes: the plain `nvidia-smi` on `PATH` and the one at
`/usr/lib/wsl/lib/` are **not always the same binary**, so prefer the explicit path; and
`nvidia-smi` says `CUDA Version: 13.1` while torch is built for `13.0` — **normal minor-version
compatibility, not a mismatch, and not a reason to move the pin.**

⚠️ **If `torch.cuda.is_available()` is `False`, `pytest` reads `52 passed, 2 failed`.** The two are
`test_cuda_is_available` and `test_environment_record_is_fully_populated`, they need real device
access, and they **must not be skipped, weakened, or given a skip marker** — that is the AM-23 alarm
working, and an escape hatch would disarm the only check that catches a CPU build on the machine
that trains. Report `52 passed, 2 failed`, say which two, and continue; it is not a regression.

Network is a **separate** permission from device access, and it matters later: an unsandboxed shell
here gets **HTTP 200** from `params.datasets.imagenette160.source_url`. If yours does not, the
**SR-20 dataset fetch and the BR-8 classifier training must run unsandboxed** — everything up to and
including the preprocessing contract does not.

`check_doc_consistency.py` is new (AM-62) and exists because the same propagation failure happened
three rounds running. It enforces the convention this repo already had — **a superseded value may
appear only in a block that cites the amendment which superseded it** — across the hand-written files
that `--check` never looked at. When an amendment supersedes a value that appears in prose, add a
rule to its `stale` table; that table is the tool's memory and is meant to grow.

#### ~~Batch 1 — scaffold + environment (W1a)~~ **DONE 2026-07-28.** What it cost, and what it taught

Landed: `requirements.in`/`requirements.lock`, `requirements-cpu.in`/`requirements-cpu.lock`,
`pyproject.toml`, `.gitignore` entries, `src/config/params.py`, `src/env.py`, `tests/test_env.py`,
`tests/test_doc_consistency.py`, and AM-65..AM-67. The assertion passes on the RTX 4060:
torch 2.13.0+cu130, CUDA 13.0, `torch.cuda.is_available()` True.

**Three findings, all from running the commands rather than reading about them (AM-65..AM-67).**
The hand-off called the cu130 *extra index* "the one real unknown"; it was the smaller half of the
problem, and the resolve was fine once `--index-strategy unsafe-best-match` was passed.

1. **`uv pip sync` would have uninstalled PyYAML and broken all three spec checks.** `sync` makes the
   environment *exactly* the lockfile — anything absent is removed, not left alone — and the batch as
   originally scoped had a runtime-only `requirements.in`. Fixed by making the lock a **superset**
   that pins PyYAML (AM-65). Caught by reading what `sync` does; it would have fired on the first
   command of W1 and stayed silent until the next spec edit.
2. **`--emit-index-url` is not optional, and this one did fire.** A lockfile compiled without it
   records **no index at all**, so nothing can install it — not `uv`, and not the plain
   `pip install --require-hashes` that SR-21's portability clause requires, which means that clause
   was untestable as written. The error names only the version (`no version of torch==2.13.0+cu130`)
   and never mentions the missing index, so it reads exactly like a bad pin (AM-66).
3. **The CPU-only lock is built** rather than promised (AM-67).

**Two operational notes for anyone re-running the install.** `pypi.nvidia.com` timed out twice under
uv's default concurrency on wheels in the 200–350 MB range; `UV_HTTP_TIMEOUT=900
UV_CONCURRENT_DOWNLOADS=2` got it through, and the cache persists so retries resume rather than
restart. And **do not pipe the install through `tail`** — `$?` then reports the pipe's status and a
failed sync looks like a success.

#### ~~Batch 2 — config plumbing + literal lint~~ **DONE 2026-07-28, committed as `2b23c1e`**

The point-in-time plan is [`docs/plans/w1-batch2.md`](docs/plans/w1-batch2.md). Batch 2 added the
resolved frozen `RunConfig`, canonical SHA-256 config hashing, two committed experiment-choice YAML
files, and `tools/check_literals.py` with mutation tests for positive and negative literals and
reasoned exceptions. AM-68..AM-70 take the spec to 178 requirements.

Three things settled with the author and recorded there:
run configs are **committed experiment files naming choices only**, with sweep axes and the resolved
per-run config archived beside results (one file per run would be thousands of unreviewable files);
the literal lint **hard-fails with per-site `# literal-ok: <reason>` annotations** that must carry a
reason and are counted in the summary; and the plan lives in the repo rather than inline here.

**The finding from planning that batch 3 depended on is closed by AM-69:** `dataset_version` and
`analysis_version`, previously undefined, now resolve through `params.config`; `config_hash` and
`checkpoint_id` remain runtime-computed by design and were never gaps.

#### ~~Batch 3 — identity, keyed RNG, test guard~~ **DONE 2026-07-28** (SR-18, SR-22)

`src/artifacts/ids.py`, `src/artifacts/rng.py` and `src/data/test_access.py` — the two things that
could not be retrofitted. **No amendment was needed**, which is itself the signal that AM-69 had
already closed the gap this batch would otherwise have hit. 54 tests pass.

Verified by re-derivation rather than by reading the diff. The RNG is genuinely keyed: a fresh
Philox generator per `(purpose, identity)` with no shared state, so drawing for image 500 **cold**
gives the same values as drawing for it after 499 prior draws, reversed iteration order changes
nothing, and — the case SR-18 actually exists for — **a system that outages and skips images still
sees identical draws for the images it does process.** All 18 `run_id_key` fields are covered, and a
missing field raises rather than hashing a partial key; both arms of a comparison share `pair_id`
while holding distinct `run_id`s; the guard fails closed with no manifest *and* with one field
short; and nothing outside `tests/` imports the guarded module, enforced by an AST-walking test.
`config_hash` is byte-identical after the hash helper was generalised, so batch 2's committed
configs did not silently move.

⚠️ **One SR-22 clause is deferred and must not be forgotten at the freeze.** Its verify clause wants
*an archived freeze manifest whose hashes resolve to the committed code, config, manifests and
checkpoints*. The guard currently checks that every field in
`params.evaluation.freeze_manifest_covers` is **present and non-empty** — not that the hashes
resolve to anything real. That is correct today, because no checkpoints or split manifests exist to
resolve against. It becomes a **G-12 obligation at W11** and is the kind of thing that looks done
because the guard is green.

**Also fixed here, and it was a defect in batch 1:** `.gitignore` carried `data/`, which matches at
any depth and would therefore have silently untracked `src/data/`. It is now `/data/`, anchored to
the repository root. Worth remembering as a class — an unanchored ignore pattern is invisible until
the file it swallowed is needed.

---

### The short version, in order

Everything in the W1 release checklist is now satisfied except the two items only the author can do,
so the hold is cleared on the specification side.

| # | Do | Owner | Why now | Blocks |
|---|---|---|---|---|
| ~~0~~ | ~~**Commit `AM-57`..`AM-60`**~~ **DONE** — `37a02dd` | — | — | — |
| ~~3~~ | ~~**Fetch/archive the srsRAN vectors**~~ **DONE 2026-07-28** — 276 files, 7.2 MB, 3 checksums OK | — | Upstream archived; window now closed in our favour | ~~G-2 at W3~~ |
| ~~1~~ | ~~**Verify proposal registration** (PR-10)~~ **DONE 2026-07-28** — confirmed complete (AM-63) | — | Was the only risk with no graceful degradation | — |
| 2 | **Hardware-alternative acknowledgement** (PR-9) | **author** | Circular clause 5. The *decision* is due before W4; the recorded acknowledgement is PR-9's acceptance criterion, so it lands with the dossier | |
| 4 | **Continue W1(c)–(f) below** — batches 1–3 done; next is the preprocessing contract (SR-19), then registry, splits, classifier | agent, **except the two sandboxed steps below** | G-1 is now wide: environment, provenance, manifests, preprocessing, guard, classifier | all of W2+ |
| 5 | **PR-1 literature review, in parallel** | either | Due W4, ≥25 refs, needs no code — and it **is** the First Review's `Problem Survey` criterion, 5 of its 30 sub-marks | First Review; DEC-13's novelty claim (AM-10 makes it *conditional* on PR-1) |
| 6 | **PR-2 Gantt, with the real dates** | either | The First Review's `Time Plan` criterion, another 5 sub-marks. Must use `params.deliverables.review_dates` — W4 / W10 / **W17** — not the spreadsheet's 2023 template | First Review; §13's schedule is its source |

**Two things to carry into W1 that are new this round and easy to get wrong:**

- **The identity/pairing keys (SR-18) and the test guard (SR-22) are W1 work, not W10 work.** Both are
  cheap now and near-impossible to retrofit once results exist. `run_id` alone used to *collide*
  between validation and test.
- **G-1 is validation-only and must prove zero test reads.** Build the guard before the loaders, not
  after.

**And one habit worth keeping.** This round found a passing evidence script that violated four rules
it claimed to enforce, and then a follow-up audit found that the round's *own* schedule edits had
reopened the leak they closed (AM-60). Both were caught by re-deriving rather than re-reading. When
you change a rule and the schedule that obeys it in the same sitting, read one against the other
afterwards — AM-47 exists for exactly this and still did not catch it.

---

0. ~~**§2 sign-off · MATLAB licence · the LDPC spike.**~~ **All closed 2026-07-25/27** (AM-19,
   AM-21..AM-25). Two consequences worth keeping in view rather than re-deriving: the hypotheses were
   **delegated**, so this repo's git history is the *sole* preregistration record for H1–H4 — never
   edit a hypothesis in place, always a new `AM` citing the old one; and "try your best for a
   crossover" authorises strengthening the baseline, never weakening the learned system.

1. ~~⚠️ **Fetch the srsRAN vectors.**~~ **DONE 2026-07-28.** `spec/evidence/srsran_vectors/` holds
   **276 `.dat` files (7.2 MB)** plus `ldpc_encoder_test_data.h`, all three
   `params.baseline.ldpc_golden_vector_sha256` checksums verified OK, and `.gitignore` keeps them
   out of git as designed. **The §16 risk is now materially smaller**: if upstream disappears, the
   fixture still builds here — but the data lives only on this machine, so it is not in a backup and
   a clean checkout elsewhere still needs the network or rung 4. Re-run
   `spec/evidence/fetch_srsran_vectors.sh` to reproduce; original text below for the reasoning.

   The script is committed and pins an immutable release; the asset returned HTTP 200 on 2026-07-27
   and again on 2026-07-28.
   The upstream repo is **archived** and AM-30 established that **OCUDU publishes no replacement** —
   its vectors moved to a MATLAB-companion plugin that needs a licensed 5G Toolbox. So if this asset
   is withdrawn, rung 2 is gone permanently and G-2 degrades to the single hand-derived floor case.
   Pull it now, keep it outside git (`spec/evidence/.gitignore` already handles that), and W3 stops
   depending on someone else's hosting decision.

2. **W1 — repo scaffold through G-1.** Build strictly in this order; each step is the input to the
   next, and three of them contaminate everything downstream if retrofitted. **G-1 is now much wider
   than an accuracy number** (AM-58): it also accepts the dataset checksums, the split manifests, the
   registry, the config round-trip, the canonical-pixel identity test, the clean-install smoke run
   and the classifier's provenance — and it is **validation-only**, with SR-22's guard in place to
   prove zero test reads. Everything below is a G-1 acceptance item, not just step (f).

   **(a) Environment lock (SR-21, AM-61) — see the cold-start block above for the commands.**
   `requirements.txt` stays tooling-only by design; the runtime stack is `requirements.in` →
   `requirements.lock` (hashed, committed), resolved by **`uv`** and installed from
   `params.environment.torch_index_url`. Also owed here, and easy to forget because the install
   succeeding feels like done: `params.environment.deterministic_backend` set; driver/device captured
   into run metadata per `params.environment.record_in_run_metadata`; a **CPU-only install path** for
   analysis and demo; and the `pip --require-hashes` portability check.

   **All pins resolved 2026-07-28 — every one has a `cp314` wheel, nothing is guesswork:**
   `torch==2.13.0+cu130` · `torchvision==0.28.0+cu130` (from the cu130 index; the wheel is
   `torchvision-0.28.0+cu130-cp314-cp314-manylinux_2_28_x86_64.whl`) · `sionna-no-rt==2.0.1`
   (W3, not W1) · `numpy 2.5.1` · `pillow 12.3.0` · `scikit-image 0.26.0` · `pytest 9.1.1`.
   The first four are already proven working together in `~/capstone-w0-spike/venv`, where
   `torch.cuda.is_available()` is True on the RTX 4060. `scikit-image` is only needed from W2 for
   `params.preprocessing.ssim_impl`, but it resolves cleanly, so there is no reason to defer it.

   **(b) Config plumbing (SR-1)** — `src/config/`. Code reads `spec/params.generated.yaml`, never
   markdown and never literals. Needs a run-config that *derives* from params and carries the
   `config_hash` SR-13 wants. Two verify clauses: a round-trip test, and **a lint rule flagging
   numeric SNR/k literals outside `src/config/` and tests**. The workable form of that lint is to
   pull the experiment-affecting values *out of* params (SNR grid, `k_symbols`, thresholds, lr,
   epochs) and flag source literals matching them, excluding trivia like 0/1/2 — a blanket
   "no magic numbers" scan is unusably noisy.

   **(c) Preprocessing contract (SR-19)** — `src/data/preprocessing.py`. **Build this before
   anything touches a pixel.** Define the canonical image as **uint8 RGB HWC**, with the `[0,1]`
   float tensor a pure function of it; then "the codec compresses the same pixels the encoder
   receives" is true by construction and SR-19's bit-identical test is trivial rather than a
   promise. `params.preprocessing.channel_normalisation` is `inside_model_never_in_the_pipeline`, so
   the classifier owns its own normalisation layer.

   **(d) Splits (SR-17)** — deterministic val carve from the *published train* split using
   `params.evaluation.split_seed` (1337). The arithmetic lines up with the real datasets, which is
   worth knowing before you debug a count: Imagenette v2-160 ships 9469 train / 3925 val, so
   9469 − 1000 = 8469 train, 1000 val, and the published val becomes the 3925-image **test** split;
   STL-10 is 5000 labelled + 8000 test → 4500 + 500 + 8000; CIFAR-10 is 50000 + 10000 →
   45000 + 5000 + 10000. Owed: a disjointness test **and** an audit that no selection code path can
   reach the test loader — the audit is the harder half and is easiest as a structural rule (test
   access lives in one module nothing else imports) rather than a convention.

   **(d2) Identity and pairing keys (SR-18) — get this right in W1 or pay for it in W11.**
   Four keys, not one: `run_id` (content-addressed over the full `params.artifacts.run_id_key`,
   including `split`, config and checkpoint hashes and the classifier variant — the old key omitted
   all of them and **collided** between validation and test), `noise_id`, `analysis_cell_id`, and a
   system-independent `pair_id` that ER-10 joins on. RNG must be **counter-based and keyed** over
   `params.artifacts.rng_purposes`, never a sequential stream consumed on demand — systems outage on
   different images, so a shared seed desynchronises exactly when it matters. Per-image rows carry
   every join column and a **stable sample ID**, not a positional index.

   **(e) Dataset registry (SR-2, SR-20) — and the note that used to sit here was wrong.**
   ⚠️ **Imagenette IS in torchvision**: `torchvision.datasets.Imagenette(root, split=..., size="160px",
   download=True)`, checked against current upstream docs. This file previously asserted the
   opposite and sent you to build a bespoke fetcher. Use the library loader (`params.datasets.
   imagenette160.loader`), and *also* record `archive_sha256` yourself — torchvision's docs do not
   state that it verifies integrity, and SR-20 fails G-1 while any checksum is still `pending`.
   STL-10 and CIFAR-10 likewise come from torchvision. All three go through one code path with no
   dataset-specific branching; CIFAR-10 is a plumbing path only (DEC-1) but must still instantiate,
   because SR-2's verify clause instantiates every dataset. Headroom is not a concern: 891 GB free.

   **(e2) Split manifests (SR-17) — a seed is not a split.** Loader ordering and library behaviour
   change between versions, so materialise the carve as a **committed manifest** of stable sample IDs
   under `params.datasets.manifest_dir`, hashed into run metadata. Stratified, ordered by stable ID
   before shuffling, drawn with the named RNG. Class indices from sorted directory names.

   **(e3) Test-access guard (SR-22) — build it in W1, not when you need it.** Loading a test sample
   must **fail** without a committed freeze manifest. The release point is **G-12 at W11** — not
   G-10, which now sits at the start of W9 (AM-60 caught that: the guard would have opened three
   weeks before anything was frozen). G-1 and the W10 rehearsal must demonstrate **zero** test-loader
   reads, which is far easier as a structural rule now than as a retrofit later.

   **(f) Reference classifier (BR-8, DEC-15) → G-1.** ResNet-18 **from scratch**, and the recipe is
   now fully specified in `params.reference_classifier` — SGD+momentum, lr 0.1, momentum 0.9, weight
   decay 5e-4, cosine with 5 warmup epochs, 100 epochs, batch 128, label smoothing 0.1, and
   `[random_resized_crop, horizontal_flip]`. **Read it from config; do not improvise one** — the
   absence of that recipe was a straight SR-1 violation on the artifact that gates G-1 *and* defines
   the denominator of ER-3's whole selection rule (AM-27). Owed: measured clean accuracy archived, a
   training log showing no pretrained initialisation, and a test that every element of the recipe is
   config-derived.

   **G-1 is `clean_acc_floor` = 0.88 on Imagenette, clean variant only, measured on validation.**
   STL-10's and CIFAR-10's floors are reported but advisory (AM-13). If 0.88 does not come, the
   fallback is now an **ordered ladder** rather than a licence (AM-58):
   `params.reference_classifier.fallback_ladder` — extend to 150 epochs, then ResNet-34, then
   ResNet-50 — selected on validation, stopping at the first rung that clears the floor, and still
   bound by SR-14's cap that the learned arm may not exceed the network scoring the classical arm.
   It is not "lower the floor"; §16 says to move a floor *at G-1* as a recorded spec change if it
   turns out wrong, not to quietly miss it later.

   Also needed at some point in W1, cheaply: `.gitignore` entries for `data/`, `checkpoints/` and
   `results/per_image/` — none exist yet. Aggregate `results/*.csv` stays **tracked**, because ER-7
   requires every thesis number to resolve to a committed CSV, and so do
   `params.artifacts.inference_summary_file` and `params.artifacts.per_image_manifest` — the
   inference summary is new (AM-57) and exists because the aggregate schema cannot hold an interval
   bound, a p-value or a verdict, i.e. exactly the numbers §2 turns on.

   **Ordering constraint waiting on this:** the transparency-bitrate probe (item 4) needs a trained
   classifier, so it slots in immediately after G-1. Don't re-derive that dependency.

3. **Build BR-2's fixture when W3 approaches — the design is settled, the work is not done.** The
   spec now specifies a committed fetch-and-convert script that pins release `release_25_10`, verifies
   `params.baseline.ldpc_golden_vector_sha256`, and leaves the `.npz` untracked, plus a committed
   hand-derived floor case that always runs. Two things to carry over from the W0 probe:
   - **Pin the lifting size.** 85 of the 102 upstream cases were skipped in the probe only because
     Sionna infers Z from (k, n) and picked a different one. That is a probe limitation, not a
     disagreement — every structurally valid comparison matched exactly. Pinning Z unlocks the rest.
   - **The comparison can only cover rates above each base graph's minimum**, since Sionna refuses to
     encode below them. Say so in the fixture rather than quietly truncating.

4. **Transparency bitrate — needs a classifier first.** `r ≈ 1/5` rests on the estimate that JPEG 2000
   goes task-transparent around 1.5–2.0 bpp at 160 px. It is the number that most determines how much
   airtime the headline comparison needs. **Dependency:** scoring needs a classifier, and the
   reference classifier is not trained until W1/G-1. Either slot this immediately after G-1, or get a
   rough early read with an ImageNet-pretrained proxy — legitimate for locating the knee in
   the curve, but *spike only*, never reported, since DEC-15 bans pretrained weights for the
   reference classifier and Imagenette is an ImageNet subset. ⚠️ **AM-30 sharpened why this matters:**
   §16 now records the 1.5–2.0 bpp figure as the weakest number in the spec — it is a *visual*
   transparency threshold applied to an *accuracy* criterion, and classification tolerates several
   times more compression, so ER-3's rule may bite much further down the ladder than the provisional
   ratios suggest. `r_1_48` and `params.bandwidth.ladder_bottom_saturation_rule` exist to catch that.

5. **Re-check that W3 still fits.** DEC-16 added 2–3 days of 16-QAM soft-demapping to a week that
   already holds LDPC integration, BER validation and bit accounting. May need resequencing. The
   16-QAM demapper is now the *only* place the AM-24 LLR-sign trap can bite again — it is the same
   convention, one level harder.

## Open questions for the user

- ~~**srsRAN golden-vector licensing.**~~ **Delegated and decided 2026-07-27 (AM-25):** "do what's
  best, I don't have a preference". Chosen: **don't vendor.** The premise turned out to be wrong —
  srsRAN never committed the vector data, it ships as a per-release asset — so the fixture fetches
  from a pinned immutable release, verifies SHA-256, and keeps the `.npz` out of git. No AGPL data in
  the submitted artifact, byte-exactness still provable. The offline-reproducibility cost that made
  this look like the lossy option is paid off by promoting rung 4 from *fallback* to *always-run
  floor*. If challenged, the argument is: checksums are facts about a file, not copies of one.
- **MATLAB licence** — still pending an outcome, and now fully off the critical path: rung 2 is not
  merely available, it is demonstrated working (AM-25). OPT-1/OPT-3 remain provisional upside.
- ~~**The actual 2026-27 review dates.**~~ **Closed 2026-07-28 (AM-59)** — and it did not need you.
  The circular is scanned images with *no extractable text*, which this file had recorded as if it
  meant unreadable; rendering the two pages to PNG and reading them resolves all four dates directly.
  Lesson worth keeping: "no extractable text" is a statement about `pdftotext`, not about the
  document. The guess was half wrong — Final Review is **W17**, not W16.
- ⚠️ **Proposal registration status — this one really does need you.** The circular makes submitting
  the proposal to your guide part of **registration**, and the only copy in this repo is a blank
  template. That does not prove nothing was submitted, which is exactly why it must be checked. PR-10
  exists for it. Nothing in the specification recovers from an unregistered project.
- ⚠️ **The hardware-alternative decision, due before W4 (PR-9).** Circular clause 5: projects are
  *expected* to have a hardware implementation, "if not, at least they should have significant design
  aspects with an application to real world problems". Tier 1 can satisfy that, but only if the design
  work is written down as design work — hence the deployment dossier. Ask the guide early enough that
  the schedule can absorb the answer, and record the acknowledgement. This must **not** be allowed to
  turn into a promise of Tier 2/3 scope (DEC-14, HR-5).
- **Name a real BLER reference before G-2.** `params.baseline.ldpc_bler_reference_source` is pending.
  TS 38.212 is a specification and contains no curve to match, so the old wording named nothing
  obtainable. Needs a downloadable, checksummed dataset agreeing on (K,N), base graph, lifting size,
  modulation, decoder algorithm, offset, iterations and SNR convention.

## Recently settled — don't reopen

- **LDPC = Sionna 2.0.1** (DEC-10). Sionna dropped TensorFlow in 2.0.0; the DEC-3 objection is dead.
  Segmentation is ours, behind the BR-14 seam.
- **Crossover strategy = adaptive modulation** (DEC-16), with dominance-everywhere as an explicitly
  last-resort fallback. Never cap modulation back to QPSK to "simplify".
- **§2 is not pass/fail on a crossover.** Completion is running the protocol properly — and the
  supervisor has now ratified exactly that (AM-19), so this is settled externally, not just
  internally. Pursue the crossover hard; report dominance if it doesn't appear.
- **Two operating points, not one** (AM-20). ER-3's rule runs at 5 pp → `efficiency_ratio` and at
  2 pp → `crossover_ratio`; ER-1's headline sits at the crossover point. Both are lookups on the same
  classical sweep table, so the second costs nothing to select. They may coincide — that's a clean
  outcome, not a failure. Don't collapse this back to one threshold: 5 pp alone can select a ratio
  where no crossover exists, which is the whole reason the second threshold is there.
- **H1's run rule stays** (AM-4). The "three consecutive points is a multiple-comparisons problem"
  objection is backwards — the run requirement is the multiplicity control, and the bound is now
  written into §2. Expect this one to come back; the arithmetic is there so you don't re-derive it.
- **ER-9 keeps entropy coding** (AM-5), bounded to a static offline-fitted coder. Dropping it would
  weaken the control that exists to *deny* joint-coding credit, which makes H4 easier to pass — the
  wrong direction to be wrong in.
- **Read `SPEC.md` §17 before acting on any future review.** Twenty-five amendments now record what
  changed and why, including four things an external review recommended that were already settled.
  AM-24 and AM-25 are the W0 spike; note that AM-25 corrects a *factual premise* of AM-22 and AM-23
  (srsRAN's vectors were never committed to git), which is why it is a new entry and not an edit —
  §17 is append-only and superseded entries stay wrong in place, on purpose.

## Session log

- **2026-07-28 (W1 batch 3, `72be2af`)** — **Identity keys, keyed RNG and the test-access guard;
  SR-18 and SR-22 implemented, no amendment needed, 54 tests passing.** The cleanest batch of the
  three, and the one that mattered most: these are the pieces that cannot be retrofitted once results
  exist. Every claim was re-derived rather than read — the RNG's control-flow invariance, all 18
  `run_id_key` fields, the pairing join's rejection of duplicate and missing trajectories, the
  guard's three failure modes, the import-graph isolation, and a `config_hash` regression check
  confirming batch 2's committed configs did not move when the hash helper was generalised. Details
  in the batch 3 block above, including the **deferred SR-22 hash-resolution clause** that becomes a
  G-12 obligation at W11.

  **Two environment facts worth carrying, because they shape who can do what.** The parallel agent
  (Codex, same WSL2 machine) reported `torch.cuda.is_available()` **False** with *"Failed to
  initialize NVML: GPU access blocked by the operating system"*, and `curl` to the Imagenette URL
  returning nothing — **both device and network blocked at the time** — while an unsandboxed shell on
  the same box sees the RTX 4060 (driver 592.82) and gets **HTTP 200** from
  `params.datasets.imagenette160.source_url`; both measured, not assumed, and an earlier draft of this
  line inferred the S3 reachability from a `download.pytorch.org` fetch, which is a different host and
  proved nothing. **This is a sandbox policy, not hardware, and it can change between sessions — so
  probe it with the three commands in the cold-start block rather than trusting this paragraph.**
  Because the box is WSL2, the GPU appears as `/dev/dxg` with the driver shim at
  `/usr/lib/wsl/lib/nvidia-smi`; an agent looking for `/dev/nvidia*` will wrongly conclude the machine
  has no GPU. Consequence, while it holds: preprocessing and split
  logic can be done sandboxed, but **the SR-20 dataset fetch and the BR-8 classifier training
  cannot**, and those are the two hard dependencies for G-1. The two GPU-bound tests
  (`test_cuda_is_available`, `test_environment_record_is_fully_populated`) will keep failing in that
  environment and must keep not being skipped — they are a correct signal, not noise.
- **2026-07-28 (W1 batch 2, `2b23c1e`)** — **Config plumbing and SR-1 literal lint;
  AM-68..AM-70, 175 → 178 requirements.** Added a deeply frozen resolved `RunConfig`, canonical
  SHA-256 hashing, learned/classical experiment-choice YAMLs, and a parameter-driven AST lint that
  catches negative SNRs as `UnaryOp(USub, Constant)` and requires a reason on every exception.
  Mutation tests prove bare `7`, bare `-8`, and an empty `# literal-ok:` fail. The README status and
  its doc-consistency regression were repaired, PR-9 was removed from the First Review readiness
  column, and AM-70 corrected §16's H4 deadline to before G-12 without moving any gate or schedule
  row. Spec, documentation, literal and packetisation checks pass. In this agent environment the
  CUDA wheel is correct (`torch 2.13.0+cu130`, built for CUDA 13.0), but OS policy blocks NVML/device
  access, so the two deliberately unskipped GPU-runtime tests remain expected failures here and must
  be rerun on the primary device before the author signs the commit. **Confirmed 36 passed on the
  primary device**, so the two failures were environmental exactly as reported. **Adjudication then
  found one real defect, and it is the lesson worth keeping:** `_resolve_choice` silently returned
  the raw string when a symbolic name failed to resolve, so `train_snr_db: train_snr_db_fixedd` — a
  one-character typo — was *accepted*, resolved to the literal string instead of `7`, and flowed
  into `config_hash`. `bw_ratio` was protected by `_validate_named_choices`; `train_snr_db` and
  `lambda` were not, and the test suite covered only the happy path. Fixed by requiring resolution
  whenever a symbolically-namespaced choice carries a **string**, while numeric values still pass
  through untouched, so a config that hard-codes a number keeps working and the classical file — which
  has no `lambda` at all — still loads. Both the defect and the fix were verified by *running* the
  typo, not by reading the diff, which is the same habit that caught the AM-24 LLR sign and AM-58's
  passing-but-wrong evidence script.
- **2026-07-28 (W1 batch 1)** — **First commit of project code; AM-65..AM-67, 172 → 175
  requirements.** The environment is locked, installed and asserted: torch 2.13.0+cu130 on CUDA 13.0,
  torchvision 0.28.0+cu130, driver 592.82, `torch.cuda.is_available()` True, 18 tests passing. All
  three spec checks still pass. **The hand-off named the wrong unknown.** It flagged the cu130 extra
  index under `--generate-hashes` as "the one real unknown"; that resolved cleanly once
  `--index-strategy unsafe-best-match` was passed. The actual defects were both in what the lockfile
  *contains*, and neither was visible from reading: a runtime-only lock would have had `uv pip sync`
  **uninstall PyYAML** and break every check that guards the spec (AM-65 — caught before it fired),
  and a lock compiled without `--emit-index-url` records **no index at all**, so nothing can install
  it — including the plain `pip install --require-hashes` that SR-21's portability clause requires,
  which means that clause had never been testable (AM-66 — caught by the install failing, with an
  error naming only the version and never the index). Both flags are now parameters rather than shell
  history, because the lockfile is regenerated whenever a pin moves. Also corrected two claims this
  file carried: `.pytest_cache/` was already in `.gitignore`, and the pins were **not** "proven
  working together" in the W0 spike venv — that venv has no torchvision, no scikit-image, no glymur
  and no pytest, so the torch/torchvision co-resolution was untested until this session. Both tests
  were checked for *bite* rather than assumed: `assert_cuda` was run against a simulated CPU build,
  and `test_doc_consistency.py` was run against a mutant checker with the AM-62 exemption bug
  reintroduced — only the exemption case failed, which is the point, since the other two cases cannot
  tell the buggy checker from the correct one. Operational notes for re-running the install:
  `pypi.nvidia.com` timed out twice under default concurrency, `UV_HTTP_TIMEOUT=900
  UV_CONCURRENT_DOWNLOADS=2` got it through, and **never pipe the install through `tail`** — `$?`
  then reports the pipe and a failed sync reads as a success, which happened once here.
- **2026-07-28 (end of session)** — **PR-10 closed (AM-63); First Review package specified (AM-64);
  batch 1 written up for a cold start.** Registration confirmed complete, which closes the last item
  no audit could resolve and the only one with no graceful degradation. Reading the rubric
  spreadsheet against the repo then found that **a third of the First Review has no artifact behind
  it**: it scores six criteria at 5 sub-marks each, and *Problem Survey* is PR-1 while *Time Plan* is
  PR-2 — neither of which exists. AM-64 adds `params.deliverables.review_package_dir`, the snapshot
  mechanism (an annotated **tag**, not a branch — a branch diverges, reads as N commits behind and
  invites a merge that must not happen), and `review_1_ready_when` = PR-1 + PR-2 + G-1 as a checkable
  trigger. The ⏰ standing trigger at the top of this file fires on it and is the reminder Nick asked
  for. Also settled, after working through the amendment record: **stop
  auditing the specification.** The trajectory is 25 → 23 → 7 → 1 → 3 → 1 → 1 → 1 entries per round;
  every fatal finding landed in rounds 0–3 and the last was AM-55; rounds 6 and 8 found only
  self-inflicted damage. A stopping rule of "audit until two agents return GO" was considered and
  **rejected as re-rollable** — with enough draws two GOs appear regardless of the spec's state, which
  is preregistration discipline applied to the science but not to the process gating it. Three of
  EXT-6's own 22 release-checklist items (#7 tested IDs, #8 committed manifests, #11 the lockfile)
  are W1 deliverables, so its exit criterion is partly circular and cannot be closed by reading.
- **2026-07-28 (earlier)** — **`check_doc_consistency.py` committed (AM-62); vectors archived;
  `uv` chosen; hand-off written for a cold start.** The tool exists because the same propagation
  failure happened three rounds running and `gen_spec_views.py --check` could never have seen any of
  them — it validates `SPEC.md` against itself, while all three failures were in the hand-written
  files. Its rule is this repo's own convention mechanised: a superseded value may appear only in a
  block that cites the amendment that superseded it. **It was tested by injecting drift, not by
  trusting that it passed** — which immediately found two bugs in it: line-by-line checking flagged
  correctly-labelled history (back-references sit a line or two below the value, so it works on
  blocks now), and an amendment number that does not exist was still granting exemptions, so the
  back-reference rule could be defeated by citing anything at all (cited AM numbers are now
  intersected with the real set). That is the AM-58 lesson applied to the checker itself — and the
  tool then caught the invented number this very log entry originally used to describe the test. `fetch_srsran_vectors.sh` run: 276 files, 7.2 MB, all three checksums OK, gitignored — so
  §16's "the upstream repo is archived and could vanish" risk is now materially smaller, with the
  residual being that the data lives on this machine only. `params.environment.lock_tool` = **`uv`**
  (AM-61), the author's choice over `pip-tools`; both emit the required format, `uv` handles the
  separate CUDA wheel index more cleanly, and that index is the exact mechanism AM-23 showed can
  silently produce a CPU build. **No implementation was started** — deliberately, so the next session
  begins clean rather than inheriting half a scaffold. The cold-start block at the top of "Do next"
  is written to be executable without reading this log: it carries the verified machine state, the
  three drift checks to run first, the install commands, and the two traps in them.

- **2026-07-28 (later still)** — **Cross-document consistency audit (`INT-5`); AM-60, 166 → 168.**
  Ran every hand-written file against the amended spec after round 5, which was large and touched the
  schedule. Found one substantive defect **of round 5's own making**: AM-58 moved W10's rehearsal onto
  validation and AM-59 moved G-10 to the start of W9, but `test_access_gate` still pointed at G-10 —
  so SR-22's guard would have released the test split at W9, three weeks before anything was frozen,
  and ER-11/ER-12 were still scheduled at W10 reading test at sweep strength. New **G-12** (test
  release, W11) closes it, and opening the test split now has a gate for the first time — it had been
  the only irreversible act in the project without one. Also swept: `docs/crossover-explained.md`'s
  hypothesis table, multiplicity arithmetic, operating-point rule and G-10 week were all stale;
  `README.md` and `AGENTS.md` counted external reviews differently, now resolved by stating in §17
  that `EXT-n` are labels rather than an ordinal count. The lesson, recorded in AM-60: a rule and the
  schedule that obeys it were edited in the same round without being read against each other, which
  is precisely what AM-47's process rule exists to prevent and precisely what it failed to catch.

- **2026-07-28 (later)** — **Codex gate audit (`EXT-6`) adjudicated; AM-57..AM-59 applied, 159 → 166
  requirements.** Verdict was project GO / W1 NOGO and it was not disputed: the defects are contract
  and evidence defects, cheap now and expensive once config and results encode them. Unlike every
  earlier round, **every checkable numeric claim reproduced exactly** — packetisation defect counts,
  the corrected canonical case, the grid arithmetic, the runtime figures, and an H4 power calculation
  (MDE ≈ 1.4 pp at 10% discordance, 3.2 pp at 50%) that no previous review attempted. One framing
  claim rejected (`run_id` "cannot pair" — it is a tuple key; the real defect is that it *collided*),
  one sharpened (the added grid points are BPSK at the noisy end, needing **fewer** code blocks, so
  the worst-case projection overstates rather than understates there). Verified against primary
  sources rather than adopted: the Sionna encoder source, for both the exact-1/3 BG1 floor and the
  fact that TS 38.212 §5.4.2.2's interleaver is applied **only** when `num_bits_per_symbol` is passed
  — which the W0 probe never did; the circular, by rendering a text-free scanned PDF; torchvision's
  current docs, which do ship an Imagenette loader this file wrongly said was absent.
  `check_packetisation.py` rewritten: four defects, zero of which its own "0 failures" surfaced.
  Estimand, H1 calibration binding, H2 intersection-union, H3 magnitude clause, `run_id`/`pair_id`/
  `noise_id` split, dataset provenance, environment lock, test-access guard, deterministic outage
  fallback, PHY seam pins, narrowed standards claim, W17 schedule, G-10 moved to W9. **Nothing now
  blocks W1.**

- **2026-07-28** — **Docs swept for staleness; AM-56 from a self-audit.** Three hand-written files
  were stale and none had been touched by the amendment rounds: `docs/crossover-explained.md` still
  claimed the cliff H2 depends on was untouched by adaptive modulation (superseded by AM-53), still
  routed the crossover fallback through G-8 rather than G-10, and still pointed at the retired
  OPT-4; `README.md` said work starts at W0; `AGENTS.md` listed the removed `core_ratio` as a
  provisional value and described supervisor sign-off as outstanding. All corrected in place with
  the supersession shown. The self-audit then found a real defect in AM-53's own work — H2's window
  selection clause still said "the classical system" when three classical curves now exist — fixed
  as AM-56, with ER-9's unspecified `transmit_dim_realised_by` factorisation recorded in §16 as a
  carried gap due before W9. All dependency pins resolved with cp314 wheels (torch 2.13.0+cu130,
  torchvision 0.28.0+cu130, scikit-image 0.26.0, pytest 9.1.1); Imagenette source verified live.
  **Nothing blocks W1.**

- **2026-07-27 (later)** — **Two external reviews adjudicated; 30 amendments applied (AM-26..AM-55),
  122 → 158 requirements.** Neither review's verdict was adopted: EXT-4's "commit after seven edits"
  understated the problem and EXT-5's blanket HOLD overstated it, since only one finding touched W1.
  Three serious defects, all found by recomputation rather than by reading: ER-9 was arithmetically
  infeasible at every operating point but one and would have sat at chance across all of H1's region
  (AM-55); TS 38.212 packetisation was non-conformant in the CRC, the code-block cap and the
  base-graph selection *order*, though segmentation changes in zero configurations so no measured
  number moves (AM-49); and ER-10 promised a variance decomposition that AM-17's zipped seeds had
  already made impossible (AM-31). Also caught: G-8 was required to decide a crossover at W6 when no
  learned model exists until W7 (AM-33, new G-10 at W10), and W9 built an entire third system with
  no gate on it (new G-11). Refuted two claims against primary sources — EXT-4's OCUDU vector
  recommendation is false and made §16's existing mitigation line false too (AM-30), and its Third
  Review rubric denominator is 60 rather than 55 (AM-46). Corrected a throughput figure that was
  faster than the evidence committed to support it (AM-29). New evidence artifact
  `spec/evidence/check_packetisation.py`; validator hardened so the dangling-vocabulary bug that
  hid `augmentation` for a whole round now fails `--check`.

- **2026-07-27** — **G-9 passed; W1 open.** Spike run to completion: all 180 configurations hit an
  exact `E_r`, 634 cb/s at 50 iterations (later corrected to the committed 625.2 by AM-29), ER-1
  projected at ~2.1 h / ~4.1 h for one and two ratios,
  smallest payload 16 bits (AM-24). Three defects found by running it rather than reading it: the
  library's LLR sign is inverted relative to `x = 1−2c` and fails *silently* at BER 0.77; nominal
  rate 1/3 was unrealizable at three live operating points against BG1's coderate floor; and the
  decoder spelling in the spec is not the one the library accepts. Golden vectors resolved beyond
  what was asked (AM-25) — Sionna matches the MATLAB-generated srsRAN vectors **bit-exactly, zero
  mismatches**, lifting sizes 2–288, both base graphs. The licensing question dissolved rather than
  being decided: the data was never committed upstream, it is a release asset, so the fixture fetches
  and verifies instead of vendoring. New carried risk: the upstream repo is archived — srsRAN became
  OCUDU in Dec 2025 — mitigated by the always-run rung-4 floor. Spec self-consistency swept
  afterwards: DEC-10 still said the fixture was a *committed* `.npz`, the ladder's rung 2 was still
  named `..._committed_testvectors`, §16's pending block was still pending, and AGENTS.md still said
  "W0 has not started" — all corrected, the rename recorded in AM-25 rather than made silently.
  Evidence folder `spec/evidence/` created so the measured claims can be checked, not trusted.
- **2026-07-25 (latest+4)** — W0 spike started; documentary half recorded as AM-23 while the torch
  install ran. TS 38.212 pinned (V17.13.0, closing AM-9); srsRAN rate-matcher and segmenter vector
  generators confirmed; §16's Python 3.14 risk rewritten after finding it was aimed at declared
  minimums rather than at CUDA wheel availability, where the actual trap is a bare `pip install
  torch` silently yielding a CPU build. BR-10's segmentation and rate-matching arithmetic verified
  at zero slack across all 72 live configurations — BR-3's equal-channel-uses claim now holds as
  arithmetic, independent of any library. New open question: srsRAN's vector data is AGPLv3, so
  vendoring it is a licensing decision.
- **2026-07-25 (latest+3)** — MATLAB licence answered: will be attempted, contingency requested.
  Added a four-rung golden-vector source ladder (AM-22) so BR-2 no longer sits behind a licence the
  project does not control, and narrowed DEC-10's AFF3CT rejection to runtime use only, since a
  one-shot fixture generator never touches CI. Rung 2 (srsRAN's committed MATLAB-derived vectors) is
  the one to confirm at W0. G-9 is now down to the LDPC spike alone.
- **2026-07-25 (latest+2)** — §2 fully dispositioned (AM-21). Completion criterion approved outright
  with "try your best for a crossover" and an explicit "if things go south, you'll be good"; the four
  hypotheses and the statistical method delegated to the author. G-9's §2 clause closed. Consequence
  worth remembering: nobody outside is checking H1–H4, so the commit history is the preregistration.
  Only the MATLAB licence and the LDPC spike remain in G-9.
- **2026-07-25 (latest+1)** — Split ER-3's operating-point selection into two thresholds (AM-20)
  after the audit found the 5 pp rule could select a ratio where no crossover exists — while ER-1,
  the only full-strength experiment, runs at exactly one ratio. Now 5 pp → efficiency point, 2 pp →
  crossover point, headline at the crossover point. Selection costs nothing extra (same sweep table);
  training costs at most three more runs; whether ER-1 doubles is deferred to G-8, because the decode
  throughput that decides it doesn't exist until the W0 spike.
- **2026-07-25 (latest)** — Supervisor ratified §2's success criterion: pursue a crossover, fall back
  to dominance-everywhere if it does not appear (AM-19). Closes the longest-lead-time G-9 item. Also
  recorded what the instruction does *not* authorise, because "try your best for a crossover" is the
  sentence most likely to be misread later as permission to weaken the learned system. Residual: was
  the sign-off on §2 whole, or the crossover clause only? The LDPC spike is now the front of the queue.
- **2026-07-25 (later)** — Adjudicated a fifth external review (ChatGPT) and applied 18 amendments
  (97 → 115 requirements). Added the §17 amendment record and the `AM` prefix, so spec changes now
  carry what-changed-and-why rather than landing silently. Adopted: §2's H2 window rule, H3 slope
  test and H1 effect size (AM-1..AM-3); ER-9 rebuilt as a shared-front-end control after the audit
  found it could not be scored at all as written (AM-5); BR-12 and λ both moved behind the gates that
  produce their inputs (AM-6, AM-7); JPEG 2000 exact-byte rate control replaced after checking the
  OpenJPEG manual (AM-8); novelty position narrowed and §16's false "no prior art" line deleted
  (AM-10). Rejected: the H1 multiplicity objection (AM-4). The internal audit found five more,
  notably ER-3 asking for 2000 validation images from splits that hold 1000 and 500 (AM-14) and ER-9
  being modulation-capped while the baseline was not (AM-15). Schedule W6–W11 resequenced.
- **2026-07-26** — Reordered "Do next": the LDPC spike moved ahead of the transparency-bitrate probe,
  which turns out to depend on a classifier that does not exist until G-1.
- **2026-07-25** — Spec revised against four external reviews (66 → 97 requirements). Rewrote §2 as
  completion-plus-hypotheses; JPEG → JPEG 2000 (DEC-9); validation-split selection (DEC-12); paired
  inference (ER-10); attribution control (ER-9); PR-1..PR-8 for rubric deliverables; fixed the
  validator's ID-contiguity bug and added tombstones. Settled DEC-10 (Sionna) and DEC-16 (adaptive
  modulation + fallback). Added `docs/crossover-explained.md`.
