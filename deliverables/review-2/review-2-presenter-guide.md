# Review 2 presenter guide

Second Review · 29 September – 3 October 2026 · 25 minutes per student

Companion to `second-review-package.md`. That file says what each slide is for;
this one says what to say on it.

## General delivery rules

- Start plain. The panel has not read the spec, and some of them were not in the
  room in August. Say "the digital system", not "the adaptive classical chain",
  until slide 4 has given you permission to use the shorter terms.
- Lead with the number, not the concept. "728 of 1,000" lands faster than 72.8%.
- Say "held-back pictures" out loud on the first results slide and every time you
  change figure. It is the caveat that matters most.
- If a question goes to something you have not measured, say so and say what
  would settle it. Do not estimate.
- Never say "statistically significant", "significant improvement", or "proven".
  One training run does not support any of those. "Observed" is the word.
- Never say "the learned system beats the digital one" without a rate and a
  signal strength. The honest version is always specific.
- Do not rush slide 4 and do not skip it. Everything after it depends on it.
- If you are running over, compress slides 12 and 13. Never compress 4 or 16.

## Slide 1 — Title

Ten seconds. Thesis, then the caveat that frames everything: held-back
pictures only, exam set never opened. Do not read the pipeline diagram.

## Slide 2 — What has changed since August

Forty seconds. One sentence per row. The point is that the digital system went
from nothing to fully measured while the learned models were trained, and that
the crossing question has already been decided.

Likely prompt: "So is the project finished?" No. Exam campaign, hypotheses,
demo and report are all ahead, and slide 18 lays them out.

## Slide 3 — Objectives

Ninety seconds, and the most important slide that is not a result. Go through the
five numbered items quickly, then stop on the right-hand panel and say it
plainly:

> We did not set out to show the learned system beats the digital one. We set out
> to build both, give them the same amount of radio time, charge the overhead
> honestly, pick the operating point without looking at the learned curve, and
> report what happened to every picture. Both a crossing and the learned system
> winning everywhere finish that list. The rubric scores whether the list was
> done, not which way the curve went.

If the panel pushes on this, that is a good sign. It is the exact trap this
requirement exists to close, and you should say so.

## Slide 4 — Before the numbers, one picture

Two minutes. This is the slide that makes the next fifteen understandable.

Take the four rows in order, in the plain words of the left column. Then the
example on the right, slowly:

> At our weakest signal, the learned system labelled 728 of the 1,000 pictures
> correctly. The digital system got almost nothing through, so every picture
> fell back to one fixed answer, and that answer happens to be right 100 times
> out of 1,000. So 10%.

Then close the loop on the flat line, because it is the single most
misunderstood thing in the whole deck:

> So when you see that flat 10% on every chart that follows, it is not a bad
> prediction. It is 1,000 pictures that never arrived.

## Slide 5 — What we measured

Sixty seconds. Get the denominators out: 252 measurements, twelve versions of the
system, twenty-one signal strengths, 1,000 held-back pictures each, zero exam
pictures. Then the grader difference, because it governs how every later slide
should be read:

> The learned systems are marked by the classifier built into them. The digital
> systems are marked by one we trained specifically on compressed-looking images.
> So the gap on the next eight slides is a whole-system gap, not a measurement of
> error-correction coding on its own.

## Slide 6 — Headline

Ninety seconds. Walk it left to right.

> At the weakest signal the learned system is at 72.8% and the digital system
> cannot deliver a single picture, so it scores 10%. At −4 dB the digital system
> starts delivering and jumps to 83.4%, which is above the learned system's
> 79.4%. From there it stays above all the way to +18 dB, where it reaches 89.3%
> against 83.4%.

Then the interpretation, once:

> The learned advantage lives entirely below the digital delivery threshold.
> Above it, the digital system is simply the better system.

## Slide 7 — The delivery transition

Sixty seconds. The point is the shape, not the exact values: the digital system
gets nothing through until −5 dB and everything at −4 dB. Say the bracket, not a
threshold — nothing was measured in between. Mention the two dips, because if
you do not, someone will find them.

## Slide 8 — Bandwidth

Ninety seconds. Two panels, compare within each.

> Cutting from 12,800 uses of the radio to 3,200 costs the learned system 23
> points at −8 dB but almost nothing at +18 dB. The digital system loses much
> more at the weak end and does not begin substantial delivery until −2 dB.

Then the honest complication, which is on the slide:

> At the quarter rate the digital system first passes the learned one at the +4 dB
> point, the learned one retakes the lead at +9 dB where the digital system drops
> some pictures, and the digital system goes ahead again after that. So I am not
> going to claim one clean crossing at this rate.

## Slide 9 — Randomised training

Sixty seconds. Two rows, the gap is widest at the bottom.

> Picking the training signal strength at random each time lifts the learned
> system at most measured points, by 4.2 points at −8 dB and 3.0 at −4 dB.

Then the caveat, because it is the one that gets over-claimed: two separately
trained models from one run. This is not a measure of how much results vary.

## Slide 10 — The task-aware digital control

Ninety seconds. The most important slide after the headline, because it is the
one that stops the result being overstated.

> The obvious objection is that we are comparing learned features against a normal
> compressed image, and of course that favours the learned system. This is the
> control. Instead of a JPEG image it sends the learned features themselves,
> through the same error-correction code and the same radio settings, using the
> same 12,800 uses.

> It holds 82.0% at every strength from −4 dB up. So sending learned features
> digitally is strong once pictures get through. What the weak-signal gap is made
> of is failed deliveries as much as representation quality.

If asked whether this isolates the error-correction code: no, and do not claim it
does. The feature encoders differ.

## Slide 11 — Peak power

Sixty seconds.

> The ordinary learned model sends symbols whose tallest spike is 20.06 dB above
> the average. No real amplifier would survive that. So we trained a second model
> under a hard 3 dB limit. It measured 3.000002 dB, inside the frozen tolerance,
> and it scores 3.1 points better at −8 dB and 0.2 points worse at +18 dB.

Two caveats, both stated: this is the numbers we put on the radio, not a real
amplifier. And it is a separately trained model, so it is not a clean penalty.

## Slide 12 — Secondary controls

Sixty seconds, and only if time allows. One line per panel. Do not apologise for
the JPEG result — it is unflattering and you kept it:

> JPEG reaches 88.6% at +18 dB, and it has a real zero-delivery point at +9 dB. That
> point is in the figure because removing it would have been the dishonest choice.

## Slide 13 — Why the digital line breaks

Ninety seconds. This is the mechanism slide and it answers the most technical
question in the room.

> We report how many pictures arrived next to how many we got right. When the
> digital system stops delivering, the score drops to match, and when it starts
> again, the score comes back. The flat 10% is the fallback rule: on an evenly
> split set, one fixed answer covers 100 pictures in 1,000.

If asked why the learned system has no flat line: its rules promise a label at
every signal strength. That is a real design choice and it is worth saying out
loud.

## Slide 14 — The table

Sixty seconds. Do not read it out. Point at where the lines cross and state the
decision, then use the bottom box: 21 signal strengths is not 21 independent
experiments.

## Slide 15 — Limits

Ninety seconds, unhurried. Read the five headings, spend the time on the grader
one, and let the rest go quickly. The purpose of this slide is that the panel
stops finding holes and starts asking real questions.

## Slide 16 — Course correction

Two minutes. This is the slide `Methodology` is scoring, and the one an examiner
is most likely to probe.

> Four times we changed the approach because of what we found.

- **AM-34.** We had pinned the digital system to the weakest radio setting, which
  made the lines crossing impossible by arithmetic. We let it pick a stronger
  setting as the signal improves. The rule we adopted is that every change either
  strengthens the digital system or is decided in advance. We have never weakened
  the learned system.
- **AM-52.** Our signal strengths were spaced 2 dB apart exactly where the sharpest
  drop happens, so we could have missed it. We added three more before it mattered.
- **AM-58.** The script that checks our packet sizes reported no problems while
  breaking four of its own rules. We fixed the script. Every result came out the
  same.
- **AM-60.** The exam set was set to unlock in week 9, about three weeks before the
  manifest that locks it existed. We moved it to week 11. It has been locked ever
  since.

## Slide 17 — Novelty

Ninety seconds. Two claims, then the negative list, which is the part that earns
the marks:

> We are not claiming that systems get steadily worse as links worsen. We are not
> claiming the sudden drop-off. We are not claiming a digital system that
> understands the task. People have already done all of those. The two claims are
> both about experimental design — the control that sends learned features
> digitally, and charging the file and framing overhead to the digital system so
> it is tuned on the payload it can actually use.

## Slide 18 — Timeline

Sixty seconds. W11 is the one that matters: one guarded opening of the exam set.
Mention that the report deadline falls inside Final Review week, which is why W15
is an internal freeze and W16 carries no experimental work.

## Slide 19 — Close

Thirty seconds. One sentence for the result, one for what is next, then the locked
exam set. Do not add anything after that line.

## Short answers for common questions

### Is the crossing an artifact of the fallback?

Partly, and the mechanism is real rather than a measurement trick. The digital
system has a floor because its rules have one: no delivery means every picture
falls back to one answer. The learned system has no floor because its rules
return a label every time. Below the threshold the comparison is partly "which
system fails more gracefully". Above it, both are delivering and the digital
system is simply more accurate. We present both regimes.

### A different grader on each side — does that break the comparison?

It makes it a whole-system comparison. It does not isolate error-correction
coding. Both grader streams are in the same published data file, so anyone can
recompute the digital system under the other grader and see the size of the
effect.

### Why not just send the answer?

That is in the deck, as the predicted-answer control. It reaches 82.0% once it
arrives. So whatever receiver-side intelligence is worth here, it is worth
something real, and the learned system's weak-signal advantage is not simply
"it knows the task".

### Why let it change radio settings instead of pinning one?

Because pinning the weakest setting fixes how much the digital system can send no
matter how clean the link gets, which makes a crossing impossible by construction
rather than by measurement. We removed the pin for exactly that reason. Keeping it
would have handed us a result.

### Why a simulated channel and not a real radio?

The project is meant to succeed without hardware, and the first tier is a
simulated link. The deployment dossier in `docs/deployment-dossier.md` maps it
onto a real one. Real-radio replay is a stretch goal with a pre-recorded
fallback, and nothing in this package depends on it.

### The spike measurement is on the numbers we send. Does that mean anything about
a real transmitter?

No, and we do not claim it does. It is a useful design constraint and a
reasonable stand-in, but amplifier behaviour, pulse shaping and filtering all
change it. That is future work under a separate authorization.

### One training run. Is that enough?

Not for a significance claim, and the deck says so. It is enough to decide the
crossing question, because those rules were fixed in advance and applied to
exact arithmetic over frozen held-back curves. More runs and the exam campaign
are scheduled.

### Has the exam set been touched?

No. It stays locked until week 11, and no gate, run or record in the project has
read it.

### What if the exam results come out differently?

Our specification treats both a crossing and the learned system winning everywhere
as complete, successful outcomes, and finishing means running the agreed protocol
properly. We would report whatever came out. The final review is where the real
numbers land.