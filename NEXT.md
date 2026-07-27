# Very Next Steps

**Working file for hand-off between sessions.** Short-lived, frequently rewritten, deliberately
scrappy. Read it first at the start of a session; update it before finishing one.

Not normative — `spec/SPEC.md` governs. If something here contradicts the spec, the spec wins and
this file is wrong. Anything here that turns out to be a durable decision belongs in `SPEC.md`
(as a `DEC`), a durable risk belongs in `SPEC.md` §16, and an explanation belongs in `docs/`.

**Last updated:** 2026-07-28 · **Phase:** **W1 — implementation starts here** · spike executed, two
external reviews adjudicated and applied (AM-26..AM-55, committed as `8e65329`). **No project code
exists yet:** no `src/`, no `tests/`, and `requirements.txt` is still tooling-only (PyYAML).

---

## Just landed — two external reviews adjudicated, 30 amendments applied, committed as `8e65329`

**Two independent full-spec reviews** (`EXT-4`, Claude; `EXT-5`, a second model) were adjudicated
against the spec and the W0 evidence, not deferred to. **Neither verdict was adopted as given.**
EXT-4 said "commit after seven edits" but its one external claim was false and it missed the three
worst defects; EXT-5 said "NO-GO/HOLD on the whole spec" but only *one* of its findings touched W1.
Result: 122 → **158 requirements**, `AM-26`..`AM-55`, split into two rounds in §17. A third round
followed on 2026-07-28 — `AM-56`, from a **self-audit of those two rounds**, which found that AM-53
had left H2 able to select its comparison window on a different curve from the one it evaluates.
159 requirements now. Worth noting as a pattern rather than an embarrassment: every audit round so
far, including the audit of the audit, has found something real.

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

0. ~~**§2 sign-off · MATLAB licence · the LDPC spike.**~~ **All closed 2026-07-25/27** (AM-19,
   AM-21..AM-25). Two consequences worth keeping in view rather than re-deriving: the hypotheses were
   **delegated**, so this repo's git history is the *sole* preregistration record for H1–H4 — never
   edit a hypothesis in place, always a new `AM` citing the old one; and "try your best for a
   crossover" authorises strengthening the baseline, never weakening the learned system.

1. ⚠️ **Fetch the srsRAN vectors — do this first, it takes a minute and the window is not
   guaranteed.**

   ```bash
   spec/evidence/fetch_srsran_vectors.sh
   ```

   The script is committed and pins an immutable release; the asset returned HTTP 200 on 2026-07-27.
   The upstream repo is **archived** and AM-30 established that **OCUDU publishes no replacement** —
   its vectors moved to a MATLAB-companion plugin that needs a licensed 5G Toolbox. So if this asset
   is withdrawn, rung 2 is gone permanently and G-2 degrades to the single hand-derived floor case.
   Pull it now, keep it outside git (`spec/evidence/.gitignore` already handles that), and W3 stops
   depending on someone else's hosting decision.

2. **W1 — repo scaffold through G-1.** Build strictly in this order; each step is the input to the
   next, and two of them contaminate everything downstream if retrofitted.

   **(a) Dependencies and the CUDA assertion.** `requirements.txt` is still `PyYAML>=6.0` only.
   Split it: project deps from PyPI, torch from the cu130 index, because a bare `pip install torch`
   silently resolves to the **CPU build** and the check that catches it is
   `torch.version.cuda is not None`, not a successful import (AM-23).

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

   **(e) Dataset registry (SR-2)** — all three selectable by name through one code path, no
   dataset-specific branching. Imagenette is **not** in torchvision and needs its own fetch:
   `https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz`, verified live 2026-07-28,
   99 MB, and it unpacks to an `ImageFolder` layout. STL-10 and CIFAR-10 come from torchvision.
   CIFAR-10 is a plumbing path only (DEC-1) but must still instantiate, because SR-2's verify
   clause instantiates every dataset. Headroom is not a concern: 891 GB free, 8 GB VRAM idle.

   **(f) Reference classifier (BR-8, DEC-15) → G-1.** ResNet-18 **from scratch**, and the recipe is
   now fully specified in `params.reference_classifier` — SGD+momentum, lr 0.1, momentum 0.9, weight
   decay 5e-4, cosine with 5 warmup epochs, 100 epochs, batch 128, label smoothing 0.1, and
   `[random_resized_crop, horizontal_flip]`. **Read it from config; do not improvise one** — the
   absence of that recipe was a straight SR-1 violation on the artifact that gates G-1 *and* defines
   the denominator of ER-3's whole selection rule (AM-27). Owed: measured clean accuracy archived, a
   training log showing no pretrained initialisation, and a test that every element of the recipe is
   config-derived.

   **G-1 is `clean_acc_floor` = 0.88 on Imagenette, clean variant only.** STL-10's and CIFAR-10's
   floors are reported but advisory (AM-13). If 0.88 does not come, the preregistered fallback is
   **switch backbone or extend training** — it is not "lower the floor", and §16 says to move a
   floor *at G-1* as a recorded spec change if it turns out wrong, not to quietly miss it later.

   Also needed at some point in W1, cheaply: `.gitignore` entries for `data/`, `checkpoints/` and
   `results/per_image/` — none exist yet. Aggregate `results/*.csv` stays **tracked**, because ER-7
   requires every thesis number to resolve to a committed CSV.

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
- **The actual 2026-27 review dates.** ⚠️ New, and it needs you rather than me — the dates on
  `Capstone Project Rubrics.xlsx` are a **2023 template** (its own submission deadline reads "10th
  Dec. 2023") and MUST NOT be used. `params.deliverables.review_dates_status` records this as
  pending against the 2026-27 circular, which is scanned images with no extractable text. The review
  *weeks* are fixed (4 / 10 / 16); W1 opening 2026-07-27 puts them at the weeks of 17 Aug, 28 Sep
  and 9 Nov. **Why it is worth ten minutes:** if the real third review is late November, W16 lands
  early and there is genuine unallocated slack — which §16 says to assign deliberately to **W9 and
  W11**, the two weeks carrying the most work and the least protection. Slack discovered late gets
  absorbed; slack known now gets used. Settle it before PR-2's Gantt is committed (AM-46).

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
