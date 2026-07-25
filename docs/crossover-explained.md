# What a "crossover" is, and why its absence is not a failure

Background note for the thesis discussion chapter and viva preparation. Explains the change to
`spec/SPEC.md` §2 in plain language first, then technically. Not normative — the spec governs.

---

## The short answer

The original success criterion required the two accuracy-vs-SNR curves to **cross**. At the operating
point the spec originally named, a crossover is arithmetically impossible — the learned system is
expected to be ahead at *every* noise level, and the two curves flatten into parallel lines rather
than swapping places.

That is a **good** outcome for the method and a **bad** outcome for the criterion, which would have
scored a working result as a failure. §2 was rewritten to describe what is actually being claimed.

But dominance everywhere is a *stronger claim on weaker evidence* than a crossover, and the fix for
that is to choose the operating point where the classical baseline is healthy — which is what gate
G-8 now does.

---

## Part 1 — In plain language

### The setup

Picture a graph. Left to right is **how good the radio link is**: far left is a terrible connection
full of static, far right is crystal clear. Bottom to top is **how often the receiving computer
correctly identifies what is in the photo** that was sent.

Two lines are drawn on it — one for the ordinary way of sending photos, one for the neural network
way. A **crossover** simply means the two lines cross each other. They swap places: one method is
better on the noisy side, the other is better on the clean side.

### The two methods fail differently

The **ordinary method** is like spelling a message out letter by letter with a checksum at the end.
If the link is good, it arrives perfectly — you get the exact photo, pixel for pixel. If the link is
bad enough that even a little gets garbled, the checksum fails, the whole thing is thrown away, and
you get *nothing at all*. That abrupt drop from "perfect" to "nothing" is the **cliff**.

The **neural method** is like describing the gist of the picture instead. Static makes the
description vaguer, but the listener still gets the general idea. It never delivers a perfect photo,
and it never delivers nothing either. It just degrades smoothly. That is **graceful degradation**,
and it is the phenomenon this project is about.

### Why anyone expected a crossover

From those two failure modes, the natural expectation follows:

- On a **clean** link, precise beats approximate — the ordinary method should win, because it
  delivers a perfect photo while the neural method only ever delivers an approximation.
- On a **noisy** link, vague-but-present beats nothing-at-all — the neural method should win.
- Somewhere in the middle, they swap. That swap is the crossover.

```
     GOOD |                        ,,,--- ordinary way
          |                  ,,,--/
accuracy  |            ,,,--X          <- the crossover
          |      ,,---/    \___ neural way
          |  ,--/
     BAD  |_/
          +----------------------------------------
            noisy  ---- link quality ---->  clean
```

### What will actually happen

The bandwidth budget originally written into the spec is **brutally tight**. So tight that the
ordinary method never gets enough room to make a good photo — even on a perfect link with zero
static, what arrives is a blocky, smeared mess.

The analogy: you are told to describe a painting in twenty words. A shrunk-down *photograph* worth
twenty words of data is unrecognisable no matter how quiet the room is. But someone who knows the
listener only needs to answer "is it a dog or a cat" can spend those twenty words well.

So the ordinary method's line flattens out at a low level, the neural method's line flattens out
higher, and the neural one is on top the whole way:

```
     GOOD |
          |        _______________________ neural way
accuracy  |    ___/
          |   /        ___________________ ordinary way
          |  /    ____/
     BAD  |_/____/
          +----------------------------------------
            noisy  ---- link quality ---->  clean
```

The lines get closer together as the link improves. They never cross.

### Why that was a problem

The second picture is a **good result**. The method wins everywhere, at every link quality. But the
original rulebook said *"success = the lines cross."* By that rule, winning everywhere counts as a
failure.

That is the trap: do the work perfectly, get a genuinely strong result, and be obliged to write it up
as a failed project because of one word in the success criterion.

---

## Part 2 — So isn't winning everywhere better news?

Yes and no, and the distinction matters more than it first appears.

**Yes:** it is a stronger claim. "Better at every noise level" beats "better below some threshold."

**No:** it is weaker *evidence*, for two reasons.

### It invites the strangled-baseline objection

An examiner looks at "I beat the ordinary method everywhere" and asks the obvious question: *did you
beat it, or did you starve it?* At a budget where the classical baseline cannot produce a usable
image even on a perfect link, winning is not informative — it is close to arithmetic. This is exactly
the unfair-comparison charge that the baseline-fairness requirements (`BR-1`..`BR-14`) and the
negative-result commitment (`ER-8`) exist to defend against, and dominance-everywhere is *more*
exposed to it than a crossover would be.

### It cannot demonstrate the trade-off

A crossover shows something dominance cannot: that the trade-off is **real and locatable**. It says
"the ordinary method is genuinely better when conditions are good — and here is precisely the point
where that stops being true." That is a complete story, and it is much harder to argue with.

Dominance everywhere can only *assert* the trade-off. It shows one regime, not two.

### The fix

This is why the operating point is now chosen at gate **G-8**, before the headline experiment, using a
rule that inspects **only the classical system** — the smallest bandwidth ratio at which the classical
baseline's best-case accuracy comes within 5 percentage points of its clean-image accuracy. Because
the rule never looks at the learned system, it cannot be accused of being chosen to flatter the
hypothesis. It simply places the comparison where the baseline is healthy.

Beating a healthy baseline is the result worth having. Beating a strangled one is a result spent
defending.

The low-budget regime is still reported — as the *dominance* regime, supporting evidence rather than
the headline.

---

## Part 2b — How a crossover is deliberately made possible (DEC-16)

### The artificial cap that was hiding in the spec

Both systems get the same number of "beeps" down the link. What differs is how much is packed into
each one.

Think of it as tones. Agree on **4 clearly separated tones** and each beep carries a little
information, but static barely troubles it. Agree on **16 tones packed closer together** and each beep
carries twice as much — at the cost of being far easier for static to confuse. Real radios switch
between these settings constantly; it is why wifi speed changes as you move around the house.

The original spec locked the classical baseline to the safe, slow setting at every noise level. It was
a person whispering slowly whether they stood in a nightclub or a library. That cap — not the noise —
is what flattened the classical line on the clean side of the graph.

```
  BEFORE (classical locked to the slow setting)

     GOOD |        _______________________ neural
          |    ___/
accuracy  |   /        ___________________ classical  <- flat because it is
          |  /    ____/                                  speed-capped, not
     BAD  |_/____/                                       noise-limited
          +-------------------------------------------
            noisy  ---- link quality ---->  clean


  AFTER (classical allowed to speed up on clean links)

     GOOD |                      ______--- classical
          |        __________,,,X                     <- crossover
          |    ___/           \__________ neural
accuracy  |   /
          |  /    ____/
     BAD  |_/____/
          +-------------------------------------------
            noisy  ---- link quality ---->  clean
```

### Why removing the cap is legitimate

The governing rule, recorded in DEC-16:

> Every lever used to obtain a crossover must **strengthen the baseline** or be **preregistered**.
> Handicapping the learned system is prohibited.

Adding 16-QAM helps the *opponent*. Because BR-4 always reports the best feasible configuration at
each SNR, the classical curve is the upper envelope over all (modulation, code rate) pairs — so this
raises it on the clean side and cannot lower it on the noisy side. The cliff that H2 depends on is
untouched.

Numerically, at r = 1/3 on Imagenette (k = 25,600 channel uses):

| Configuration | bits/symbol | Payload | bits per pixel |
| --- | --- | --- | --- |
| QPSK, rate 1/3 (noisy end) | 2 | 17,066 b ≈ 2.1 kB | 0.67 |
| QPSK, rate 5/6 (the old cap) | 2 | 42,666 b ≈ 5.3 kB | 1.67 |
| **16-QAM, rate 5/6 (clean end)** | 4 | 85,333 b ≈ 10.7 kB | **3.33** |

At 3.3 bpp, JPEG 2000 is effectively transparent to the classifier, so the classical ceiling
approaches clean accuracy (~0.88) while a learned system trained at a fixed 7 dB plausibly stalls
around 0.85–0.87. That is a crossover.

### The intuition for why extra room helps classical more

Two people packing suitcases: one packs carefully and brings only essentials, the other throws in
everything. Enlarge both suitcases and the careless packer gains enormously; the careful one was
already fine.

The learned system is the careful packer — it only ever sends what the task needs, so it is close to
its ceiling even on a tiny budget. The classical system insists on sending a whole picture, so every
additional bit helps it disproportionately. **That asymmetry in the returns is the mechanism that makes
a crossover possible at all.**

### The thing that must be said out loud (BR-15)

The classical system is **re-tuned at every noise level** — fresh choice of tone count and
error-correction strength. The learned system is **trained once and evaluated frozen** (DEC-11).

That is realistic, since a deployed sensor cannot retrain itself mid-flight, but it is genuinely an
advantage handed to the opponent, and it is part of why a crossover appears. BR-15 therefore requires
it to be stated in the methods section and in every headline figure caption, with a fixed-modulation
classical curve reported alongside the adaptive one so the contribution of adaptivity is visible
rather than implicit. OPT-4's SNR-randomised learned variant is the natural counterpart.

Disclosed in advance, this reads as rigour. Discovered by an examiner, it reads as a thumb on the
scale. Same fact, opposite reception.

### What it costs

- **The headline claim shifts** from "does more with dramatically less airtime" toward "survives
  conditions where the standard approach collapses entirely." The thesis is already built on the
  second claim, so it survives, but §1 needs to stay honest about it.
- **Two to three days of baseline engineering**, and it must not be skimped: 16-QAM needs genuine
  soft-demapping to log-likelihood ratios, and a subtly wrong demapper degrades the baseline
  *invisibly* — defeating the whole purpose. BR-2 now requires per-modulation validation for exactly
  this reason.
- **The test range had to extend to 18 dB**, because 16-QAM at rate 5/6 does not decode until roughly
  11–12 dB. Truncating earlier would engineer a crossover and then fail to measure it.

### The last-resort fallback

If the adaptive baseline still does not cross by G-8, the recorded fallback is to report learned
dominance across the whole grid, promote the reconstruction-quality (PSNR) crossover to the secondary
figure, and rely on §2 — which already makes that a complete Tier 1.

This exists to keep the demo and the thesis intact, **not** as a preferred outcome. Both systems are
racing toward the same wall: the classifier's accuracy on a perfect image. Nobody beats perfect. If
the learned system reaches that wall too, the curves meet instead of crossing, and that is simply what
the experiment found. Taking the fallback must be recorded in DEC-16 alongside the G-8 evidence that
the adaptive baseline was genuinely attempted first.

The one response that is never permitted: weakening the learned system to manufacture a crossover.

---

## Part 3 — The technical version

### Why the classical curve flattens

Two separate saturation effects stack up.

**Channel coding saturates.** For QPSK with 5G NR LDPC at these blocklengths, the decoding waterfall
sits at roughly −1 dB Es/N0 at rate 1/3 and 4–5 dB at rate 5/6. Above about 7–9 dB every configuration
decodes with block error rate ≈ 0. Beyond that point, more SNR buys nothing: the file arrives
bit-exact, and accuracy is pinned at whatever the *source coding* allows.

**Source coding is the binding constraint.** At the original core ratio:

| Quantity | Value |
| --- | --- |
| Image (Imagenette160) | 160 × 160 × 3 → n = 76,800 |
| Bandwidth ratio r = 1/12 | k = 6,400 complex channel uses |
| QPSK, 2 bits/symbol | 12,800 channel bits |
| Best LDPC rate 5/6 | ⌊12,800 × 5/6⌋ = 10,666 information bits = 1,333 bytes |
| Over 25,600 pixels | **≈ 0.42 bits per pixel** |
| Less JPEG's ~250–290 B container floor | **≈ 0.22 bpp of actual image data** |

0.2–0.4 bpp on a 160 px image is roughly JPEG quality 5–15 — heavily artifacted. A frozen ResNet-18
trained on clean images scores such reconstructions well below its ~0.88 clean-accuracy floor.

### The asymptotes cannot cross, as originally specified

Under the original parameters — r = 1/12, QPSK only:

- Classical ceiling ≈ accuracy on ~0.4 bpp imagery ≈ **0.65–0.75**
- Learned ceiling ≈ near the clean floor, since 6,400 complex symbols is ample for a 10-class
  decision ≈ **0.85–0.90**

Both flat above ~7 dB, learned above classical. Two parallel flat lines. No crossing point exists.

### Where a crossover would live

A crossover requires the classical ceiling to *exceed* the learned ceiling, which requires the
classical reconstruction to be near-transparent to the classifier — roughly 1.5–2.0 bpp, so call it
51,200 bits over 25,600 pixels. The ratio needed depends entirely on how much each channel use
carries:

```
QPSK only (the original cap, 2 bits/symbol):
    k = 51,200 / (2 × 5/6) = 30,720 symbols  ->  r = 30,720 / 76,800  ≈ 2/5

With 16-QAM available (DEC-16, 4 bits/symbol):
    k = 51,200 / (4 × 5/6) = 15,360 symbols  ->  r = 15,360 / 76,800  ≈ 1/5
```

So the modulation cap was costing a factor of two in required bandwidth. Without it, a crossover
needed r ≈ 1/3 to 1/2 — four to eight times the originally specified core ratio, which is why `r_1_3`
and `r_1_2` were added to `params.bandwidth.ratios`. **With DEC-16's adaptive modulation the
requirement halves to r ≈ 1/5**, putting `r_1_6` (1.67 bpp at 16-QAM rate 5/6) within reach and making
`r_1_3` (3.33 bpp) the conservative choice.

That halving is the substantive argument for DEC-16 over simply granting more airtime: the crossover
is bought by using the budget better rather than by enlarging it, which preserves far more of the
bandwidth-efficiency story. G-8 makes the actual selection; these are estimates, not measurements.

### The theory underneath

Shannon's separation theorem makes compress-then-protect optimal only asymptotically, for infinitely
long messages. At short blocklengths, separation incurs finite-blocklength penalties, so joint
source-channel coding *may* gain (Kostina & Verdú, arXiv:1209.1317). Note "may": this is the
hypothesis under test, not a theorem being applied — which is why §1 was reworded away from "precisely
where joint source-channel coding wins."

### What replaced the criterion

Completion and outcome are now separated. Tier 1 is **complete** when the preregistered protocol has
been run properly, regardless of result. Four hypotheses are then reported either way:

| | Claim | Test |
| --- | --- | --- |
| **H1** | Low-SNR separation (primary) | Paired 95% interval on the accuracy *difference* above zero at ≥3 consecutive low-SNR points, plus a preregistered mean paired difference across the whole low-SNR region as the effect size |
| **H2** | Cliff versus graceful | The 4 dB window is chosen **on validation**, where the *classical* curve drops most, then frozen; over those same endpoints on test, classical loses ≥30 pp and learned ≤15 pp |
| **H3** | Convergence | The paired gap trends to zero: negative weighted-least-squares slope against SNR, bootstrap interval excluding zero. **A crossover is reported if observed but is not required** |
| **H4** | Attribution | Learned must also beat the task-aware digital control (ER-9) — which shares the learned system's front end and differs only in the channel interface — or the gain is credited to task-awareness rather than joint coding |

Three of those four rows were tightened on 2026-07-25 (`SPEC.md` §17, AM-1 through AM-5) after an
external review found H2's window and H3's pass condition were not defined tightly enough to be
decided without a judgement call. The changes make each hypothesis *harder* to support, not easier —
which is the direction a preregistration should move in when it moves at all.

One objection worth knowing about, because it will recur: H1's "three consecutive points" rule is
sometimes read as a multiple-comparisons problem — many candidate runs, therefore inflated false
positives. The arithmetic runs the other way. Under the null, the chance that any run of three clears
a one-sided 0.025 interval is at most 11 × 0.025³ ≈ 0.00017 if the points were independent, and 0.025
if they were perfectly correlated. Both are stricter than the 0.05 a single point carries. Requiring
a run *is* the multiplicity control; the cost is statistical power, not validity. §2 now records this
in writing so it does not have to be argued from scratch.

The statistical test also changed, and this matters independently of the crossover question. The
original criterion asked for two independent confidence intervals not to overlap, computed over three
seeds — a Student-t interval with two degrees of freedom, where t₀.₉₇₅,₂ = 4.303 means a ~5 percentage
point gap would be needed to register. Because both systems see identical images and identical noise
draws, a **paired** analysis over per-image outcomes is both the natural and the far more powerful
choice, detecting differences of 1–2 pp.

---

## Part 4 — Why this needed sign-off before starting

The "curves must cross" wording did not originate in the spec. It came from the capstone proposal
that the supervisor already approved, and it appeared in `README.md` and `AGENTS.md` until the
revision.

A success criterion cannot be quietly replaced by its author partway through — even when the author
is demonstrably right. From the outside, changing the target after seeing where the arrow landed is
indistinguishable from changing it *because* of where the arrow landed.

Timing is the entire defence:

- Raised **before any data exists** → preregistration, and evidence of good experimental judgement.
- Raised **after results are in** → rationalisation, and the hardest question in the viva.

The argument to make is that this is a rigour *upgrade*, not a relaxation:

1. Crossover is a property of the **operating point**, not of the method — and the operating point is
   now chosen by a rule that cannot see the learned system.
2. The replacement test is **harder**: paired per-image inference rather than non-overlap of two
   weak intervals.
3. A crossover is **still reported** if it appears (H3).
4. The project **pre-commits to publishing a negative result** (ER-8), which most published work in
   this area does not do.

If the supervisor prefers to keep crossover language, there is a fallback that costs almost nothing:
keep it as the headline criterion but run the headline comparison at r ≈ 1/3–1/2, where a crossover
genuinely can occur. That is already what G-8 selects.

---

## Glossary

| Term | Plain meaning |
| --- | --- |
| **SNR** | How clean the radio link is. Higher = less static. Measured in dB. |
| **Crossover** | The point on the graph where the two methods swap which one is better. |
| **Graceful degradation** | Getting steadily vaguer as conditions worsen, instead of abruptly failing. |
| **Cliff** | The abrupt drop from "perfect" to "nothing" when error correction gives up. |
| **Bandwidth ratio (r)** | How much radio airtime you get per unit of image. Smaller = tighter budget. |
| **bpp** | Bits per pixel — how much data each pixel is allowed. Lower = more compressed. |
| **Ceiling** | The best a method can do even on a perfect link, limited by compression alone. |
| **Paired test** | Comparing both methods on the *same* images and *same* noise, image by image. Much more sensitive than comparing two averages. |
| **Preregistration** | Committing to how you will judge the result before you have the result. |
| **Modulation (QPSK, 16-QAM)** | How much is packed into each "beep" sent down the link. QPSK uses 4 well-separated tones (2 bits per beep, robust); 16-QAM uses 16 closer-together tones (4 bits per beep, fragile). |
| **Adaptive modulation and coding** | Switching to a faster, more fragile setting when the link is clean and back to a slow, robust one when it is noisy. Why wifi speed changes as you move around. |
| **Soft-demapping / LLR** | Working out, for each received beep, *how confident* you are in each bit rather than just guessing 0 or 1. Straightforward for QPSK, genuinely fiddly for 16-QAM. |
