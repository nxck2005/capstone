# Very Next Steps

**Working file for hand-off between sessions.** Short-lived, frequently rewritten, deliberately
scrappy. Read it first at the start of a session; update it before finishing one.

Not normative — `spec/SPEC.md` governs. If something here contradicts the spec, the spec wins and
this file is wrong. Anything here that turns out to be a durable decision belongs in `SPEC.md`
(as a `DEC`), a durable risk belongs in `SPEC.md` §16, and an explanation belongs in `docs/`.

**Last updated:** 2026-07-27 · **Phase:** **G-9 passed — W1 is open** · spike executed, two external
reviews adjudicated and applied (AM-26..AM-55), nothing installed into the repo yet

---

## Just landed — two external reviews adjudicated, 30 amendments applied

**Two independent full-spec reviews** (`EXT-4`, Claude; `EXT-5`, a second model) were adjudicated
against the spec and the W0 evidence, not deferred to. **Neither verdict was adopted as given.**
EXT-4 said "commit after seven edits" but its one external claim was false and it missed the three
worst defects; EXT-5 said "NO-GO/HOLD on the whole spec" but only *one* of its findings touched W1.
Result: 122 → **158 requirements**, `AM-26`..`AM-55`, split into two rounds in §17.

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
classical-only and no learned model exists until W7. New **G-10** at W10 decides it.

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

0. ~~**§2 sign-off.**~~ **Done, fully — 2026-07-25 (AM-19, AM-21).** The completion criterion is
   approved directly: finishing means running the protocol properly, whichever way the result falls,
   with the instruction to *try hard for a crossover* and an explicit "if things go south, you'll be
   good". The four hypotheses, paired inference, the no-weakening rule and the learned-blind
   operating-point rule were **delegated** — "up to you". That closes G-9's §2 clause.

   **Two things that follow from the delegation, and matter more than they look:**
   - **This repo is now the only preregistration record for H1–H4.** Nobody outside checked them, so
     the git history *is* the evidence that they were fixed before any data existed. Never edit a
     hypothesis in place — new `AM` citing the old one, always, per §17. A post-hoc change made
     honestly and recorded still reads as rigour; the same change made silently reads as fraud, and
     the diff cannot tell them apart on your behalf.
   - **"Try your best for a crossover" does not license anything on the learned system's side.**
     Strengthen the baseline, preregister the lever, or do neither. See DEC-16's guardrail before any
     G-8 decision.

1. ~~**MATLAB licence.**~~ **Answered 2026-07-25: he will try, and asked for a contingency.** So the
   licence is attempted but not assured, and OPT-1/OPT-3 stay provisional. The contingency is now a
   four-rung ladder in `params.baseline.ldpc_golden_vector_source_ladder` (AM-22), and the useful
   find is **rung 2**: srsRAN's vectors are generated from MATLAB's 5G Toolbox as their own trusted
   reference, which gives MATLAB-provenance vectors *without* a MATLAB licence — all BR-2 needs, since
   it only demands independence from Sionna. **Both open questions are now answered (AM-25):** the
   vectors do cover the rate-matched output, and the licence question dissolved because the data is
   *not* committed upstream at all — it ships as a release asset, so we fetch and verify rather than
   vendor. ⚠️ Earlier wording here and in AM-22/AM-23 said srsRAN "ships committed" vectors; that was
   wrong, and AM-25 supersedes it. Rung 4 is now an always-run floor, not a fallback.

2. ~~**The LDPC spike.**~~ **Done, all seven checks — 2026-07-27 (AM-24, AM-25).** See above.

3. **Start W1 — this is the front of the queue, and it is now unblocked in both senses.** G-9 passed
   2026-07-27 and the amendment rounds are applied, so nothing is outstanding. W1: repo scaffold,
   config plumbing (SR-1), data loaders and validation splits (SR-2, SR-17), reference classifier
   trained from scratch on clean images (BR-8). Its gate is **G-1** — the classifier clears its
   `clean_acc_floor`.

   **Two things the amendments added that land in W1 specifically.** BR-8's classifier now has a
   full recipe in `params.reference_classifier` — optimizer, lr, schedule, warmup, epochs, batch
   size, label smoothing, augmentation — because it had none at all, which was a straight SR-1
   violation on the artifact that gates G-1 *and* defines the denominator of ER-3's selection rule.
   Use it; don't improvise one. And **SR-19 is new**: one canonical image defined in
   `params.preprocessing` before either pipeline sees data, with the codec compressing the *same*
   pixels the encoder receives. Build that first — it is cheap now and contaminates everything if
   retrofitted.

   **Concrete first action:** `requirements.txt` with the pins W0 verified — Python 3.14.6,
   `torch 2.13.0+cu130`, `sionna-no-rt 2.0.1`. The `--index-url https://download.pytorch.org/whl/cu130`
   is **mandatory**: a bare `pip install torch` silently gives the CPU build, and the assertion that
   catches it is `torch.version.cuda is not None`, not a successful import (AM-23).

   **Ordering constraint that has been waiting on this:** the transparency-bitrate probe (item 5)
   needs a trained classifier, so it slots in immediately after G-1. That dependency is why it got
   pushed behind the spike in the first place — don't re-derive it.

4. **Build BR-2's fixture when W3 approaches — the design is settled, the work is not done.** The
   spec now specifies a committed fetch-and-convert script that pins release `release_25_10`, verifies
   `params.baseline.ldpc_golden_vector_sha256`, and leaves the `.npz` untracked, plus a committed
   hand-derived floor case that always runs. Two things to carry over from the W0 probe:
   - **Pin the lifting size.** 85 of the 102 upstream cases were skipped in the probe only because
     Sionna infers Z from (k, n) and picked a different one. That is a probe limitation, not a
     disagreement — every structurally valid comparison matched exactly. Pinning Z unlocks the rest.
   - **The comparison can only cover rates above each base graph's minimum**, since Sionna refuses to
     encode below them. Say so in the fixture rather than quietly truncating.

5. **Transparency bitrate — needs a classifier first.** `r ≈ 1/5` rests on the estimate that JPEG 2000
   goes task-transparent around 1.5–2.0 bpp at 160 px. It is the number that most determines how much
   airtime the headline comparison needs. **Dependency:** scoring needs a classifier, and the
   reference classifier is not trained until W1/G-1. Either slot this immediately after G-1, or get a
   rough early read tomorrow with an ImageNet-pretrained proxy — legitimate for locating the knee in
   the curve, but *spike only*, never reported, since DEC-15 bans pretrained weights for the
   reference classifier and Imagenette is an ImageNet subset.

6. **Re-check that W3 still fits.** DEC-16 added 2–3 days of 16-QAM soft-demapping to a week that
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
