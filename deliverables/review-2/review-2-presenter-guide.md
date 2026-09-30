# Review 2 presenter guide

Second Review · 29 September – 3 October 2026 · 25 minutes per student

Companion to `second-review-package.md`. That file says what each slide is for;
this one says what to say on it.

## General delivery rules

- Lead with the number, not the concept. "Learned 72.8%, digital 10% at −8 dB"
  lands faster than any framing sentence.
- Say "validation" out loud on the first results slide and every time you change
  figure. It is the single most repeated caveat in the deck and it should be
  repeated out loud too.
- If a question goes to something you have not measured, say so and say what
  would settle it. Do not estimate.
- Never say "statistically significant", "significant improvement", or "proven".
  One seed cell does not support any of those. "Observed" is the word.
- Never say "the learned system beats the classical system" without a ratio and
  an SNR. The honest version is always arm-specific and SNR-specific.
- If you are running over, compress slides 11 and 12. Never compress 15.

## Slide 1 — Title

Ten seconds. Thesis, then the caveat that frames everything: validation only,
test never opened. Do not read the pipeline diagram.

## Slide 2 — What changed since August

Forty seconds. One sentence per row. The point of this slide is that the
classical arm went from nonexistent to fully measured while the learned models
were trained, and that G-10 has already been decided.

Likely prompt: "So is the project finished?" No. Test campaign, hypotheses, demo
and report are all ahead, and slide 17 lays them out.

## Slide 3 — Objectives

Ninety seconds, and the most important non-results slide. Go through the five
numbered criteria quickly, then stop on the right-hand panel and say it plainly:

> We did not set out to show the learned system beats the classical one. We set
> out to build both, match the budget, charge the overhead honestly, evaluate at
> an operating point chosen without looking at the learned curve, and report
> paired per-image outcomes. Both a crossover and learned dominance finish that
> list. The rubric scores whether the list was done, not which way the curve went.

If the panel pushes on this, that is a good sign — it is the trap AM-46 exists
to close, and you should name that.

## Slide 4 — What was measured

Sixty seconds. Get the denominators out: 252 units, 12 arms, 21 SNR points,
1,000 images per point, zero test images. Then the scorer difference, because it
governs how every later slide should be read:

> The learned systems are scored by their own task head. The classical systems
> are scored by the classifier we fine-tuned on codec artifacts. So the gap on
> the next six slides is a whole-system gap, not a measurement of channel coding
> in isolation.

## Slide 5 — Headline

Ninety seconds. Walk it left to right.

> At −8 dB the learned system is at 72.8% and the digital chain cannot deliver
> a single image, so it scores 10.0% under the fallback rule. At −4 dB the
> digital chain starts delivering and jumps to 83.4%, which is above the
> learned system's 79.4%. From there it stays above all the way to +18 dB, where
> it reaches 89.3% against 83.4%.

Then the interpretation, once:

> The learned advantage lives entirely below the digital delivery threshold.
> Above it, the classical chain is simply the better system.

## Slide 6 — The delivery transition

Sixty seconds. The point is the shape, not the exact values: digital coverage is
0% through −5 dB and 100% at −4 dB. Say the bracket, not a threshold — nothing
was measured in between. Mention the two coverage dips, because if you do not,
someone will find them.

## Slide 7 — Bandwidth

Ninety seconds. Two panels, compare within each.

> Cutting from 12,800 channel uses to 3,200 costs the learned system 23 points
> at −8 dB but almost nothing at +18 dB. The digital arm loses far more at the
> noisy end and does not begin substantial delivery until −2 dB.

Then the honest complication, which is on the slide:

> At 1/24 the digital arm first passes the learned arm at the measured +4 dB
> point, the learned arm retakes the lead at +9 dB where digital coverage dips,
> and the digital arm goes ahead again after that. So I am not going to claim a
> single clean crossover at this ratio.

## Slide 8 — SNR-randomised training

Sixty seconds. Two rows, the gap is widest at the bottom.

> Drawing the training SNR uniformly across the grid lifts the learned system at
> most measured points, by 4.2 points at −8 dB and 3.0 at −4 dB.

Then the caveat, because it is the one that gets over-claimed: two separately
trained checkpoints in one seed cell. This is not a seed-variance estimate.

## Slide 9 — The ER-9 control

Ninety seconds. The most important slide after the headline, because it is the
one that stops the result being overstated.

> The obvious objection to our comparison is that we are comparing learned
> features against JPEG, and of course that favours the learned system. ER-9 is
> the control. It takes learned task-aware features, puts them through the same
> LDPC chain and the same modulation at the same channel-use budget, and sends
> digital.

> It holds 82.0% at every SNR from −4 dB up. So digital transmission of task
> features is strong once it delivers. What the low-SNR gap is made of is outage
> behaviour as much as representation quality.

If asked whether this isolates the LDPC code: no, and do not claim it does. The
feature encoders and task heads differ.

## Slide 10 — PAPR

Sixty seconds.

> The unconstrained learned symbols reach 20.06 dB PAPR, which no power
> amplifier would survive. We trained a separate checkpoint under a hard 3 dB
> cap. It measured 3.000002 dB, inside the frozen 0.0001 dB tolerance, and it
> scores 3.1 points better at −8 dB and 0.2 points worse at +18 dB.

Two caveats, both stated: symbol-domain, not waveform or amplifier. And a
separately trained checkpoint, so it is not a clean causal penalty.

## Slide 11 — Secondary controls

Sixty seconds, and only if time allows. One line per panel. Do not apologise for
the JPEG result — it is unflattering and you kept it:

> JPEG secondary reaches 88.6% at +18 dB, and it has a real zero-coverage point
> at +9 dB. That point is in the figure because removing it would have been the
> dishonest choice.

## Slide 12 — Delivery coverage

Ninety seconds. This is the mechanism slide and it answers the most technical
question in the room.

> Coverage and accuracy are reported side by side on every row. When the digital
> chain drops, accuracy drops with it, and where it comes back, accuracy comes
> back. The 10.0% floor is the fallback rule: on an exactly stratified
> validation split, a single fixed class covers 100 of 1,000 images.

If asked why the learned system has no floor: its contract returns a label at
every SNR. That is a real design choice and it is worth saying out loud.

## Slide 13 — The table

Sixty seconds. Do not read it out. Point at the crossover row and say the G-10
decision, then use the bottom box: 21 points on one grid is not 21 replications.

## Slide 14 — Limits

Ninety seconds, unhurried. Read the five headings, spend the time on the
scorer one, and let the rest go quickly. The purpose of this slide is that the
panel stops finding holes and starts asking real questions.

## Slide 15 — Course correction

Two minutes. This is the slide `Methodology` is scoring, and the one an examiner
is most likely to probe.

> Four times we changed the approach because of what we found.

- **AM-34.** We had capped the baseline at QPSK, which made a crossover
  arithmetically impossible. We added 16-QAM and made modulation adaptive per
  SNR. We strengthened the baseline on purpose, and the rule we adopted is that
  every lever either strengthens the baseline or is preregistered. We have never
  weakened the learned system.
- **AM-52.** Our SNR grid was sampled every 2 dB exactly where the BPSK
  waterfall sits, so the cliff we were trying to measure could have been missed
  or smeared. We added three points before it mattered.
- **AM-58.** The packetisation checker reported zero failures while breaking four
  of its own rules. We fixed the checker. Every scientific outcome survived
  unchanged.
- **AM-60.** Test access was set to release at G-10, three weeks before the
  freeze manifest existed. We moved it to G-12. It has been sealed ever since.

## Slide 16 — Novelty

Ninety seconds. Two claims, then the negative list, which is the part that
earns the marks:

> We are not claiming graceful degradation, we are not claiming the cliff, and
> we are not claiming a task-aware digital semantic system. All of those have
> substantial prior art. The two claims are both about experimental design —
> the ER-9 control, and charging format overhead to the baseline so it is tuned
> on the payload it can actually use.

## Slide 17 — Timeline

Sixty seconds. W11 is the one that matters: one guarded opening of the test split
at G-12. Mention that the report deadline falls inside Final Review week, which
is why W15 is an internal freeze and W16 carries no experimental scope.

## Slide 18 — Close

Thirty seconds. One sentence for the result, one for what is next, then the
sealed test split. Do not add anything after that line.

## Short answers for common questions

### Is the crossover an artifact of the outage fallback?

Partly, and the mechanism is real rather than a measurement trick. The digital
arm has a floor because its contract has one: no delivery means every image
falls back to one class. The learned arm has no floor because its contract
returns a label at every SNR. Below the threshold the comparison is partly
"which system fails more gracefully". Above it, both are delivering and the
classical system is simply more accurate. We present both regimes.

### Different classifiers on each side — does that break the comparison?

It makes it a whole-system comparison. It does not isolate channel coding. Both
scorer streams are published in the same CSV, so anyone can recompute the
classical arms under the clean classifier if they want to see the size of the
effect.

### Why not just send the class label?

That is in the deck, as the predicted-label control. It reaches 82.0% once it
delivers. So whatever receiver-side intelligence is worth here, it is worth
something real, and the learned system's low-SNR advantage is not simply "it
knows the task".

### Why 16-QAM and not QPSK?

Because a QPSK cap fixes the classical payload no matter how clean the link gets,
which makes a crossover impossible by construction rather than by measurement.
DEC-16 removed the cap for exactly that reason. Capping the baseline would have
handed us a result.

### Why AWGN and not a radio?

Tier 1 is a simulated channel and the project is meant to succeed without
hardware. The deployment dossier in `docs/deployment-dossier.md` maps the
finite-blocklength link onto a real one. Tier 2 SDR replay is a stretch goal with
a pre-recorded fallback, and nothing in this package depends on it.

### PAPR is symbol-domain. Does that mean anything about a real transmitter?

No, and we do not claim it does. Symbol-domain PAPR is a useful design
constraint and a reasonable proxy, but amplifier behaviour, pulse shaping and
filtering all change it. That is future work under a separate authorization.

### One seed cell. Is that enough?

Not for a significance claim, and the deck says so. It is enough to close the
crossover gate, because that gate was defined in advance on exact rational
arithmetic over frozen validation curves. More seed cells and the test campaign
are scheduled.

### Has the test set been touched?

No. `test_access` is 0 across every gate, run and artifact in the project, and
the split stays sealed until G-12 in week 11.

### What if the test results come out differently?

SPEC §2 treats both a crossover and learned dominance as complete Tier 1
outcomes, and completion is defined by running the protocol properly. We would
report whatever came out. The third review is where the final numbers land.