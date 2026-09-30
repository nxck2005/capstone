# Second Review package

Capstone project · Second Review · 29 September – 3 October 2026 · W10

This directory is the Second Review deliverable. It is derived from the
annotated git tag `review-2-basis` on this branch, not reconstructed afterwards.

## What is in here

| File | What it is |
|---|---|
| `semantic-communication-second-review.pptx` | The deck. Native editable shapes, so anyone can revise wording without rebuilding. |
| `semantic-communication-second-review.pdf` | Rendered copy for machines without an Office renderer. |
| `semantic-communication-second-review-contact-sheet.png` | All 19 slides on one page, for a fast check of flow. |
| `previews/slide-NN.png` | Per-slide previews at 1600×900. |
| `review-2-presenter-guide.md` | What to say on each slide, and answers to the questions we expect. |
| `ITERATION-NOTES.md` | What changed between drafts of this deck. |

Builder: `tools/build_second_review_ppt.py`. Re-run it to regenerate everything
from the committed W10 figures.

## Look

The deck follows `deliverables/review-1/semantic-communication-first-review.pptx`
— the plain First Review deck, not the `academic-v2` one. White ground, Arial
throughout, black and grey type, no colour accents. Emphasis comes from weight,
hairline rules and upper-case label chips rather than from hue. Body type runs
10.5–13 pt so it reads from the back of a room on a 25-minute slot.

If you are extending the deck, keep to that palette. A new colour will make this
one deck look like it came from a different project than the First Review did.

## Tone

The audience for this review is not reading the spec. Some of them were not in
the room in August. So the deck earns the technical vocabulary instead of
opening with it:

- Slide 4 is a plain-language primer. Four concepts in ordinary words, the term
  we use for each, and where it appears on the charts.
- Everything before slide 4 is setup and contains no measurement vocabulary.
- After slide 4 the technical terms are safe, because they have been defined.
- Numbers are introduced as "728 of 1,000 pictures", not as percentages with a
  denominator nobody has been told about.
- No acronym appears without being spelled at least once. ER-9, PAPR, LDPC and
  QPSK are written out; the acronyms survive only in the citation footer where
  a panel member can trace them.

## Where the results come from

Every number on every slide traces to the published W10 v11 validation
aggregate. Nothing was re-derived for the deck and no curve was redrawn.

- Aggregate CSV: `presentation-results/data/w10_primary_252.csv` — 252 rows,
  one per unit, 12 arms × 21 SNRs
- Scorer streams: `presentation-results/data/w10_scorers_357.csv`
- Figures 01–08: `presentation-results/figures/{png,pdf,svg}/`
- Written findings: `presentation-results/report/findings.md`
- Source evidence: `results/learned/w10/`

`test_access` is 0 and `test` is `SEALED` across the whole project. Nothing in
this package was computed on test data.

## Delivery acceptance contract

1. Deck is committed under `deliverables/` and derived from the `review-2-basis`
   annotated tag (SPEC PR-8(d)).
2. Objectives slide states objectives as completion criteria, not outcomes
   (PR-8(a)).
3. The §17 course-correction slide is present (PR-8(c)). This one is mandatory
   for the Second and Third packages specifically.
4. Every results slide carries its held-back-set and single-seed qualification.
5. The Gantt at `docs/gantt-plan.md` is current and shows W11–W17 honestly.
6. Nothing in the package claims test-split results.

## 1. Rubric coverage

The Second Review is 30 marks, carrying six criteria from the rubric
spreadsheet. The deck maps onto them like this.

| Rubric criterion | Sub-marks | Slides | How the deck answers it |
|---|---|---|---|
| Results (graphs/tables/test cases) | 20 | 6, 7, 8, 9, 10, 11, 12, 14 | Eight measured figures plus two numeric tables at both rates. 252 published measurements. |
| Originality | 10 | 17 | Two claims, each tied to experimental design, each with the prior art it is distinguished from. |
| Analytical skills | 5 | 5, 7, 8, 13, 14, 15 | Why the curves break where they do, what the controls buy, and a slide on what the evidence cannot support. |
| Presentation | 5 | all | 19 slides for a 25-minute slot. Presenter guide carries the timing. |
| Methodology | 5 | 16 | The §17 distillation: four documented course corrections. Required by PR-8(c). |
| Timeline | 5 | 2, 18 | Progress since the First Review, and the remaining weeks with their gates. |

## 2. Narrative contract

Four things the deck has to do, in this order.

**Teach the vocabulary before using it.** Slide 4 does this in one pass. If the
panel has understood "the flat 10% line means nothing arrived", every later chart
is readable without further explanation.

**Establish the evidence base before any claim.** Slide 5 gives the size and
shape of the measurement set — 252 measurements, 21 signal strengths, 1,000
pictures each, zero exam pictures — before slide 6 shows a curve.

**Separate what was measured from what it means.** The learned systems are
marked by the classifier built into them; the digital systems by one trained on
compressed images. So the headline gap is a whole-system quantity. The deck says
so on slide 5 and repeats it wherever the comparison appears.

**State the limits before the panel finds them.** Slide 15 does this early
enough that slide 19 does not have to walk anything back.

## 3. Twenty-five-minute plan

| Slides | Time | Content |
|---|---|---|
| 1–3 | 4 min | Title, progress since August, objectives as completion criteria |
| 4 | 2 min | The plain-language primer. Do not rush this one. |
| 5 | 1 min | What we measured |
| 6–8 | 5 min | Headline, the delivery transition, bandwidth |
| 9–13 | 6 min | Randomised training, the control, peak power, secondary controls, delivery |
| 14–15 | 3 min | Numbers table, limits |
| 16 | 2 min | Course correction (SPEC §17) |
| 17–18 | 2 min | Novelty, timeline |
| 19 | 1 min | Close |

That is roughly 26 minutes of planned content against a 25-minute slot, so the
four minutes of slack the plan assumes have to come out of slides 12 and 13.
Both are mechanism detail and the numbers survive in the report.

## 4. Slide content contract

### Slides 1–3 — Frame it

Slide 1 states the thesis in one sentence. Slide 2 is the progress log: what
existed in August versus what exists now. Slide 3 is the objectives slide, and it
does the most quiet work. The rubric scores `Objectives Met` at the Third Review
by whether objectives set at the First were met, so they need to be the things
we did. The right-hand panel says what the objective deliberately is not.

### Slide 4 — The on-ramp

The most structurally important slide in the deck, and the one a presenter is
most tempted to skip. It defines four words, gives a worked example, and
explains the flat 10% line that otherwise looks like a suspicious floor.

### Slides 5–8 — The headline and its mechanism

Slide 6 is the one to spend time on. Slide 7 zooms into the delivery transition.
Slide 8 quarters the radio time and shows the two systems respond differently.
Slide 9 adds the randomised-training result with a two-row table.

### Slides 10–13 — The controls

Slide 10 is the task-aware digital control, and it is the most important slide
after the headline. Without it the weak-signal story looks like "beats JPEG",
which is not what we measured. Slide 11 is peak power. Slide 12 is the
secondary controls, including the JPEG point at +9 dB that does not flatter us.
Slide 13 explains the mechanism behind the cliff.

### Slides 14–15 — Numbers and limits

Slide 14 is the table version of the curves, at both rates, with the crossing
decision stated. Slide 15 is the limits slide.

### Slide 16 — Course correction

Four corrections, required by PR-8(c). AM-34 strengthened the digital system. AM-52
changed the signal grid. AM-58 caught a checker that was reporting success while
breaking four of its own rules. AM-60 moved the exam-set release three weeks
later. If the panel asks what we changed after seeing results, this is the slide.

### Slides 17–19 — Novelty, timeline, close

Slide 17 is the novelty statement with its negative check. Slide 18 is the
remaining schedule. Slide 19 is the close, with the locked exam set stated
plainly.

## 5. Required figures and provenance

| Figure | File | Qualification that must travel with it |
|---|---|---|
| 01 | `01_headline_full_snr.png` | Held-back pictures, one seed, different graders on the two sides |
| 02 | `02_low_snr_closeup.png` | Gap between −5 and −4 dB, no measured point inside it |
| 03 | `03_bandwidth_efficiency.png` | Compare within each panel; the crossing recrosses at the quarter rate |
| 04 | `04_training_robustness.png` | Two separately trained models — not run-to-run variance |
| 05 | `05_task_aware_digital.png` | Does not isolate the error-correction code |
| 06 | `06_papr_tradeoff.png` | Measured on the numbers we send, not a real amplifier |
| 07 | `07_secondary_controls.png` | The rebuilding panel uses a different grader |
| 08 | `08_delivery_coverage.png` | The flat 10% is the fallback rule, not a delivery |

Use the PDF or SVG in the report. Use the PNG in the deck.

## 6. Anticipated questions

Short answers are in the presenter guide. The four most likely:

**Is the crossing real or an artifact of the fallback?** Partly fallback. The
learned line has no floor because its rules promise a label at every signal
strength; the digital line has one because when nothing arrives, every picture
falls back to a single answer. That is the honest mechanism, and it is why slide
13 exists.

**A different grader on each side — does that break the comparison?** It makes
it a whole-system comparison rather than a controlled test of error-correction
coding. Both grader streams are published, so anyone can check.

**Why not just send the answer?** That is in the deck as a control. It reaches
82.0% once it arrives, which bounds what receiver-side intelligence is worth
here.

**One training run. Is that enough?** No, and we say so. It is enough to decide
the crossing question on rules we fixed in advance, and not enough for a
significance claim. More runs and the exam campaign are scheduled.

## 7. Second Review scope boundary

In scope: the held-back measurement results, the controls, the analysis, the
documented course corrections, the novelty statement, the updated schedule.

Out of scope and not claimed: any exam-set result, hardware or waveform
measurements, the demo, the poster, the report. Tier 2 and Tier 3 remain
stretch and nothing in this package depends on them.

There is no fallback in this review. The crossing was found, so the fallback
clause never fired, and the objectives were not restated at the Second Review as
PR-8(b) permits.

## 8. Readiness matrix

| Item | State | Where |
|---|---|---|
| Deck, editable | Ready | `semantic-communication-second-review.pptx` |
| PDF proof and previews | Ready | same directory |
| Plain-language primer | Ready | slide 4 |
| Results figures | Ready, held-back set only | `presentation-results/figures/` |
| Objectives slide, PR-8(a) | Ready | slide 3 |
| §17 distillation slide, PR-8(c) | Ready | slide 16 |
| Novelty negative check, PR-7 | Ready | slide 17 |
| Gantt current, PR-2 | Ready | `docs/gantt-plan.md`, slide 18 |
| Exam-set results | None exist, none claimed | — |
| `review-2-basis` tag | Cut on this branch | — |
| Guide hardware acknowledgement | Carried from the First Review, unverified | — |
| Four-member rehearsal | Carried from the First Review, unverified | — |