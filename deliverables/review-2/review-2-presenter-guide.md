# Review 2 presenter guide

Second Review · 29 September – 3 October 2026 · 25 minutes per student

Companion to `second-review-package.md`. That file says what each slide is for;
this one says what to say on it.

## How to talk

- Start plain. Say "the digital system" until slide 4 has introduced the shorter
  terms, then use them.
- Lead with the number. "728 of 1,000" lands faster than 72.8%.
- Say "held-back pictures" out loud on the first results slide and each time you
  change figure.
- Describe what happened. "The digital system got nothing through" is a fact.
  "We should be careful not to overstate this" is a tic — cut it.
- When a question goes past what we measured, say where the answer would come
  from. That is a complete answer.
- "Observed" is the word. Every figure here is one measurement from one run, so
  say "observed" rather than "improved" or "significant".
- Give the rate and the signal strength whenever you compare the two systems.
  "The learned system wins" on its own is not a sentence anyone should say.
- Give slide 4 the two minutes it needs. Everything after it depends on it.
- Running long? Compress 12 and 13. Keep 4 and 16.

## Slide 1 — Title

Ten seconds. Thesis, then the caveat that frames everything. Skip the diagram.

## Slide 2 — What has changed since August

Forty seconds. One sentence per row. The point is that the digital system went
from prototype to fully measured while the learned models were trained, and that
the crossing question is already settled.

Likely prompt: "Is the project finished?" The exam campaign, hypotheses, demo
and report are all ahead. Slide 18 lays them out.

## Slide 3 — Objectives

Ninety seconds, and the most important slide that carries no results. Take the
five numbered items quickly, then stop on the right-hand panel:

> We did not set out to show the learned system beats the digital one. We set out
> to build both systems, give them the same amount of radio time, charge the
> overhead honestly, pick the operating point without looking at the learned
> curve, and report what happened to every picture.

Then the scoring point, which is the reason this slide exists:

> Completion is independent of which way the curve went. A crossing and the
> learned system winning everywhere both finish this list, and the rubric scores
> whether the list was completed.

If the panel pushes on this, that is the discussion we want.

## Slide 4 — Before the numbers, one picture

Two minutes. This slide makes the next fifteen understandable.

Take the four rows in the plain words of the left column. Then the worked example
on the right, slowly:

> At our weakest signal, the learned system labelled 728 of the 1,000 pictures
> correctly. The digital system got almost nothing through, so every picture
> fell back to one fixed answer, and that answer happens to be right 100 times
> out of 1,000. So 10%.

Then close the loop on the flat line, because it is the most misread number in
the deck:

> That flat 10% is the fallback rule. Those 1,000 pictures never arrived.

## Slide 5 — What we measured

Sixty seconds. Walk the table: 252 measurements, twelve versions of the system,
twenty-one signal strengths, 1,000 held-back pictures each, zero exam pictures.
Then the two rows underneath, because they govern how every later slide reads:

> Within one rate both systems get exactly the same amount of radio time. And the
> two sides are marked by different classifiers — the learned systems by the one
> built into them, the digital systems by one we trained on compressed-looking
> images. So the gap on the next eight slides is a whole-system gap.

## Slide 6 — The two models behind every number

Ninety seconds, and skip it if the panel already knows the architecture.

> Two models are involved. A ResNet-18 classifier reads the pictures, and a
> residual convolutional link decides what to transmit. Both trained from
> scratch on this dataset, no pre-trained weights.

> The link has an encoder and a mirrored decoder with two heads: class scores,
> and a rebuilt picture as an auxiliary task. It sends 8 complex symbols per
> channel use at half rate. It is 1.64 million parameters at its widest.

> Training is Adam at 0.001, cosine decay, batch 32, 100 epochs, random crop and
> horizontal flip, mixed 16-bit. The objective is class scores plus three times
> the reconstruction error, and that weight of three was chosen on the held-back
> set rather than by hand.

The last line is the one worth saying out loud: the weight was calibrated on
validation, so a panel question about it has a preregistered answer behind it.


## Slide 7 — Headline

Ninety seconds. Walk it left to right.

> At the weakest signal the learned system is right on 728 of 1,000 and the
> digital system delivers nothing. From −4 dB the digital system delivers and
> goes ahead — 83.4% against the learned 79.4%. It stays ahead all the way to
> +18 dB, where it reaches 89.3% against 83.4%.

Then the interpretation, once:

> The learned advantage lives entirely below the digital delivery threshold.
> Above it, the digital system is the better system.

## Slide 8 — The moment the digital system starts working

Sixty seconds. The shape is the point: nothing gets through until −5 dB,
everything gets through at −4 dB. The transition lies somewhere between the two.
Mention the two dips at −2 and +9 dB yourself, so nobody has to find them.

## Slide 9 — Why the digital line breaks

Ninety seconds. This answers the most technical question in the room.

> We report how many pictures arrived next to how many we got right. When the
> digital system stops delivering, the score drops to match; when it starts again,
> the score comes back. The flat 10% is the fallback rule — on an evenly split
> set, one fixed answer covers 100 pictures in 1,000.

If asked why the learned system has no flat line: its rules promise a label at
every signal strength.

## Slide 10 — Quartering the radio time

Ninety seconds. Two panels, compare within each.

> Cutting from 12,800 uses of the radio to 3,200 costs the learned system 23
> points at −8 dB and almost nothing at +18 dB. The digital system loses much
> more at the weak end and doesn't begin substantial delivery until −2 dB.

Then the complication, which the slide already states:

> At this rate the digital system passes the learned one at +4 dB, falls back at
> +9 dB, and goes ahead again after that. The crossing happens twice here.

## Slide 11 — Training across signal strengths

Sixty seconds. Two rows, the gap is widest at the bottom.

> Drawing the training signal strength at random each time lifts the learned
> system at most measured points, by 4.2 points at −8 dB and 3.0 at −4 dB.

Both rows are separately trained models from one run. Say that and move on.

## Slide 12 — The task-aware digital control

Ninety seconds. The most important slide after the headline, because it stops the
result being overstated.

> The obvious objection is that we are comparing learned features against a normal
> compressed image. This is the control. Instead of a JPEG image it sends the
> learned features themselves, through the same error-correction code and the same
> radio settings, on the same 12,800 uses.

> It holds 82.0% at every strength from −4 dB up. So sending learned features
> digitally is strong once pictures get through, and the weak-signal gap is about
> failed deliveries as much as representation quality.

If asked whether this isolates the error-correction code: the feature encoders
differ, so it compares representations.

## Slide 13 — Peak power

Sixty seconds.

> The ordinary learned model sends symbols whose tallest spike is 20.06 dB above
> the average. No real amplifier would survive that, so we trained a second model
> under a hard 3 dB limit. It measured 3.000002 dB, inside the frozen tolerance,
> and scores 3.1 points better at −8 dB and 0.2 points worse at +18 dB.

The peak-to-average ratio here is measured on the numbers we put on the radio.
Both models are separately trained.

## Slide 14 — Four controls

Sixty seconds, and only if time allows. One line per panel. Keep the JPEG point
in:

> JPEG reaches 88.6% at +18 dB, with a full outage at +9 dB. That point is in the
> figure because it is what we measured.

## Slide 15 — The numbers

Sixty seconds. Don't read it out. Point at where the two tables differ and use
the reading guide underneath.

## Slide 16 — How to read these results

Ninety seconds, unhurried. Read the five rows. Spend the time on "Whole systems"
and let the rest go. The point of this slide is that the panel gets the
qualifications once, in one place, and can then ask real questions.

## Slide 17 — Course correction

Two minutes. This is the slide `Methodology` is scoring.

> Four times we changed the approach because of what we found.

- **AM-34.** We had pinned the digital system to the weakest radio setting, which
  made the lines crossing impossible by arithmetic. We let it pick a stronger
  setting as the signal improves. The rule we adopted is that every change either
  strengthens the digital system or is decided in advance.
- **AM-52.** Our signal strengths were spaced 2 dB apart exactly where the
  sharpest drop happens. We added three more before it mattered.
- **AM-58.** The script that checks our packet sizes reported no problems while
  breaking four of its own rules. We fixed the script, and every result came out
  the same.
- **AM-60.** The exam set was set to unlock in week 9, about three weeks before the
  document that locks it existed. We moved it to week 11. It has stayed locked
  ever since.

## Slide 18 — What we claim as new

Ninety seconds. Two claims, then the list on the right:

> We build on four things that are already established: systems that degrade as
> the link worsens, avoiding the sudden drop-off, sending task-aware features
> digitally, and training a neural encoder and decoder. Our two claims are both
> about experimental design — the control that sends learned features digitally,
> and charging the file and framing overhead to the digital system so it is tuned
> on the payload it can actually use.

## Slide 19 — Timeline

Sixty seconds. The week of 5 October is the one that matters: one guarded opening
of the exam set. The report is due on 20 November, inside Final Review week, so
the week before that is a freeze.

## Slide 20 — Close

Thirty seconds. One sentence for the result, one for what's next, then the sealed
exam set. Stop there.

## Anticipated questions

### Is the crossing an artifact of the fallback?

Partly, and the mechanism is real. The digital system has a floor because its
rules have one: when delivery fails, every picture takes the same answer. The
learned system has no floor because its rules return a label every time. Below
the threshold the comparison is partly about which system fails more gracefully.
Above it, both are delivering and the digital system is simply more accurate.
That's why we present both regimes.

### A different grader on each side — does that break the comparison?

It makes it a whole-system comparison, which is what we're claiming. Both grader
streams are in the same published file, so anyone can recompute the digital
system under the other grader and see the size of the effect.

### Why not just send the answer?

That's in the deck as the predicted-answer control. It reaches 82.0% once it
arrives, so receiver-side intelligence is worth something real here, and the
learned system's weak-signal advantage is more than just "it knows the task".

### Why let the digital system change radio settings?

Pinning the weakest setting fixes how much it can send however clean the link
gets, which makes a crossing impossible by construction rather than by
measurement. We removed the pin for that reason.

### Why a simulated channel?

The first tier is a simulated link and the project is designed to finish without
hardware. The deployment dossier maps it onto a real one. Real-radio replay is
scheduled for late October.

### The spike measurement is on the numbers we send. Does that transfer to a real
transmitter?

Filter and amplifier behaviour both change the peak ratio. That measurement is a
design constraint and a reasonable stand-in; the RF measurement is late-October
work.

### One training run. Is that enough for the crossing decision?

The rule was fixed in advance and applied to exact arithmetic over frozen
held-back curves, which is what setting the rule in advance asks for. Broader
inference is what the exam campaign and more training runs are for.

### Has the exam set been read?

It stays sealed until the week of 5 October.

### What if the exam results differ?

Our specification treats a crossing and the learned system winning everywhere as
two complete outcomes, and finishing means running the agreed protocol properly.
We report what comes out. The final review is where the exam numbers land.