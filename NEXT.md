# Very Next Steps

**Working file for hand-off between sessions.** Short-lived, frequently rewritten, deliberately
scrappy. Read it first at the start of a session; update it before finishing one.

Not normative — `spec/SPEC.md` governs. If something here contradicts the spec, the spec wins and
this file is wrong. Anything here that turns out to be a durable decision belongs in `SPEC.md`
(as a `DEC`), a durable risk belongs in `SPEC.md` §16, and an explanation belongs in `docs/`.

**Last updated:** 2026-07-25 · **Phase:** spec revision, pre-W0 · nothing executed, nothing committed

---

## Do next

0. **Two emails. Send both today — they are the only things here with external lead time,
   and both gate G-9.** Neither takes a minute to write; both can sit unanswered for days.

   **(a) §2 scope, to the supervisor.** The crossover criterion is ratified (2026-07-25): try hard
   for a crossover, fall back to the always-better graph if it genuinely does not appear. That is
   DEC-16 as already written, now externally agreed (AM-19), so the proposal's "curves must cross"
   wording is superseded rather than quietly redefined. **What is still open is narrower:** did the
   sign-off cover §2 *as a whole* — completion-not-outcome, all four hypotheses, paired inference —
   or the crossover clause alone? G-9 needs the former. Ask it plainly, do not reopen the crossover
   question, and say explicitly that you want it on record before any data exists, because the
   timing is the whole defence. **And do not let "try your best" drift:** it licenses strengthening
   the baseline, nothing on the learned system's side. See DEC-16's guardrail before any G-8 decision.

   **(b) MATLAB licence, to whoever administers it.** Determines whether OPT-1 and OPT-3 are live
   and — the part that actually matters — where BR-2's golden vectors come from. A "no" is fine but
   needs lead time, because then the golden vectors have to be sourced from some other implementation
   independent of Sionna, and BR-2 gates G-2 which gates every reported comparison. Ask early enough
   that a no is survivable.

1. **The LDPC spike.** Half a day, no dependencies, and the last substantive G-9 item — W1 cannot
   start without it. Do it while the two emails are in flight. Throwaway venv in the scratchpad —
   nothing installs into the repo until W1, per `requirements.txt`. Establish these, in order of how
   much they would hurt if wrong:
   - `LDPC5GEncoder(k, n)` hits an **exact** `n`. BR-3's equal-channel-uses claim rests entirely on
     this. If it cannot, the whole bandwidth-matching design needs rework — find out now, not at W3.
   - `sionna==2.0.1` installs and runs on Python 3.14 + torch 2.13 (it resolves on paper; nobody
     tests that combination). Fallback is pinning a 3.12/3.13 interpreter.
   - `cn_update="offset-minsum"` is accepted.
   - Encode → QPSK → AWGN → decode at each of the four LDPC rates: clean at high SNR, failing at low.
   - Batched decode throughput at 50 iterations, and the smallest workable payload size.
   - Pin the exact TS 38.212 document version within Release 17 (AM-9) — it is a G-9 item now, and
     it costs one line while you are already reading the standard.

   Findings go into DEC-10 and the G-9 record. Record the throughput number — BR-4's compute plan
   needs it, and G-9 now also requires you to turn it into a **projected ER-1 evaluation wall clock —
   for one operating ratio and for two** (AM-18, AM-20). That projection is the point: three systems
   × 18 SNR points × the full test split × three seed cells is the one cost in the project with no
   slack in it, ER-6 forbids subsetting it, and it lands at W11 with a single week behind it. The
   two-ratio number is what G-8 uses to decide whether the efficiency point gets full-strength
   intervals or sweep-strength ones. Cheap to compute now, expensive to discover then.

2. **Transparency bitrate — needs a classifier first.** `r ≈ 1/5` rests on the estimate that JPEG 2000
   goes task-transparent around 1.5–2.0 bpp at 160 px. It is the number that most determines how much
   airtime the headline comparison needs. **Dependency:** scoring needs a classifier, and the
   reference classifier is not trained until W1/G-1. Either slot this immediately after G-1, or get a
   rough early read tomorrow with an ImageNet-pretrained proxy — legitimate for locating the knee in
   the curve, but *spike only*, never reported, since DEC-15 bans pretrained weights for the
   reference classifier and Imagenette is an ImageNet subset.

3. **Re-check that W3 still fits.** DEC-16 added 2–3 days of 16-QAM soft-demapping to a week that
   already holds LDPC integration, BER validation and bit accounting. May need resequencing.

## Open questions for the user

- **MATLAB licence** — asked, awaiting reply (item 0b). If the answer is no, OPT-1 and OPT-3 drop
  without prejudice and BR-2's golden vectors need another source independent of Sionna — name it
  before G-9 closes rather than after.
- **§2 sign-off scope** — asked, awaiting reply (item 0a). Crossover clause is ratified; whether the
  rest of §2 is covered is the open part.

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
- **Read `SPEC.md` §17 before acting on any future review.** Eighteen amendments now record what
  changed and why, including four things an external review recommended that were already settled.

## Session log

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
