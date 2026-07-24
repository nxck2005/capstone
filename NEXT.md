# Very Next Steps

**Working file for hand-off between sessions.** Short-lived, frequently rewritten, deliberately
scrappy. Read it first at the start of a session; update it before finishing one.

Not normative — `spec/SPEC.md` governs. If something here contradicts the spec, the spec wins and
this file is wrong. Anything here that turns out to be a durable decision belongs in `SPEC.md`
(as a `DEC`), a durable risk belongs in `SPEC.md` §16, and an explanation belongs in `docs/`.

**Last updated:** 2026-07-25 · **Phase:** spec revision, pre-W0 · nothing executed, nothing committed

---

## Do next

0. **Message the supervisor about §2** — five minutes, do it before anything else because it is
   async and has the longest lead time. Hand them `docs/crossover-explained.md`; Part 4 is written
   as the argument to make. See `SPEC.md` §16 for why the timing matters.

1. **The LDPC spike.** Half a day, no dependencies, and it is a G-9 requirement so W1 cannot start
   without it. Throwaway venv in the scratchpad — nothing installs into the repo until W1, per
   `requirements.txt`. Five things to establish, in order of how much they would hurt if wrong:
   - `LDPC5GEncoder(k, n)` hits an **exact** `n`. BR-3's equal-channel-uses claim rests entirely on
     this. If it cannot, the whole bandwidth-matching design needs rework — find out now, not at W3.
   - `sionna==2.0.1` installs and runs on Python 3.14 + torch 2.13 (it resolves on paper; nobody
     tests that combination). Fallback is pinning a 3.12/3.13 interpreter.
   - `cn_update="offset-minsum"` is accepted.
   - Encode → QPSK → AWGN → decode at each of the four LDPC rates: clean at high SNR, failing at low.
   - Batched decode throughput at 50 iterations, and the smallest workable payload size.

   Findings go into DEC-10 and the G-9 record. Record the throughput number — BR-4's compute plan
   needs it.

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

- **MATLAB licence** — unresolved. Determines whether OPT-1/OPT-3 are live and where BR-2's golden
  vectors come from. Needs an answer by G-9.

## Recently settled — don't reopen

- **LDPC = Sionna 2.0.1** (DEC-10). Sionna dropped TensorFlow in 2.0.0; the DEC-3 objection is dead.
  Segmentation is ours, behind the BR-14 seam.
- **Crossover strategy = adaptive modulation** (DEC-16), with dominance-everywhere as an explicitly
  last-resort fallback. Never cap modulation back to QPSK to "simplify".
- **§2 is not pass/fail on a crossover.** Completion is running the protocol properly.

## Session log

- **2026-07-26** — Reordered "Do next": the LDPC spike moved ahead of the transparency-bitrate probe,
  which turns out to depend on a classifier that does not exist until G-1.
- **2026-07-25** — Spec revised against four external reviews (66 → 97 requirements). Rewrote §2 as
  completion-plus-hypotheses; JPEG → JPEG 2000 (DEC-9); validation-split selection (DEC-12); paired
  inference (ER-10); attribution control (ER-9); PR-1..PR-8 for rubric deliverables; fixed the
  validator's ID-contiguity bug and added tombstones. Settled DEC-10 (Sionna) and DEC-16 (adaptive
  modulation + fallback). Added `docs/crossover-explained.md`.
