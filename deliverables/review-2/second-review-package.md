# Second Review package

Capstone project · Second Review · 29 September – 3 October 2026 · W10

This directory is the Second Review deliverable. It is derived from the
annotated git tag `review-2-basis` on this branch, not reconstructed afterwards.

## What is in here

| File | What it is |
|---|---|
| `semantic-communication-second-review.pptx` | The deck. Native editable shapes, so anyone can revise wording without rebuilding. |
| `semantic-communication-second-review.pdf` | Rendered copy for machines without an Office renderer. |
| `semantic-communication-second-review-contact-sheet.png` | All 18 slides on one page, for a fast check of flow. |
| `previews/slide-NN.png` | Per-slide previews at 1600×900. |
| `review-2-presenter-guide.md` | What to say on each slide, and answers to the questions we expect. |
| `ITERATION-NOTES.md` | What changed between drafts of this deck. |

Builder: `tools/build_second_review_ppt.py`. Re-run it to regenerate everything
from the committed W10 figures.

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
4. Every results slide carries its validation-only and single-seed-cell
   qualification.
5. The Gantt at `docs/gantt-plan.md` is current and shows W11–W17 honestly.
6. Nothing in the package claims test-split results.

## 1. Rubric coverage

The Second Review is 30 marks, carrying six criteria from the rubric
spreadsheet. The deck maps onto them like this.

| Rubric criterion | Sub-marks | Slides | How the deck answers it |
|---|---|---|---|
| Results (graphs/tables/test cases) | 20 | 5, 6, 7, 8, 9, 10, 11, 13 | Eight measured figures plus two numeric tables at both ratios. 252 published units. |
| Originality | 10 | 16 | Two claims, each tied to experimental design, each with the prior art it is distinguished from. |
| Analytical skills | 5 | 4, 6, 7, 12, 13, 14 | Why the curves break where they do, what the controls buy, and a slide on what the evidence cannot support. |
| Presentation | 5 | all | 18 slides for a 25-minute slot. Presenter guide carries the timing. |
| Methodology | 5 | 15 | The §17 distillation: four documented course corrections. Required by PR-8(c). |
| Timeline | 5 | 2, 17 | Progress since Review 1, and the remaining weeks with their gates. |

## 2. Narrative contract

Three things the deck has to do, in this order.

**Establish the evidence base before any claim.** Slide 4 gives the size and
shape of the measurement set — 252 units, 21 SNRs, 1,000 images per point, zero
test images — before slide 5 shows a curve. A number without its denominator is
not a result.

**Separate what was measured from what it means.** The learned arms are scored
by their own task head; the classical arms by the artifact-finetuned classifier.
So the headline gap is a system-level quantity. The deck says so on slide 4 and
repeats it wherever the comparison appears, rather than burying it in a caption.

**State the limits before the panel finds them.** Slide 14 is a limits slide,
and it is deliberately early enough that slide 18 does not have to walk anything
back. One seed cell, validation only, unmatched optimisation effort, no
waveform data, and steps rather than smooth degradation. Each of these is a
question the panel will ask, so answering them first is cheaper than defending
them later.

## 3. Twenty-five-minute plan

| Slides | Time | Content |
|---|---|---|
| 1–3 | 4 min | Title, progress since August, objectives as completion criteria |
| 4 | 2 min | What was measured |
| 5–7 | 6 min | Headline, low-SNR close-up, bandwidth |
| 8–12 | 8 min | Randomised training, ER-9 control, PAPR, secondary controls, delivery coverage |
| 13–14 | 3 min | Numbers table, limits |
| 15 | 2 min | Course correction (SPEC §17) |
| 16–17 | 3 min | Novelty, timeline |
| 18 | 1 min | Close |

Leaves roughly four minutes of slack, which is where questions will go. If you
are running long, slides 11 and 12 are the two to compress — both are mechanism
detail and the numbers survive in the report.

## 4. Slide content contract

### Slides 1–3 — Frame it

Slide 1 states the thesis. Slide 2 is the progress log: what existed in August
versus what exists now. Slide 3 is the objectives slide, and it is the one that
does the most quiet work. The rubric scores `Objectives Met` at the Third Review
by whether objectives set at the First were met, so they need to be the things
we did. The right-hand panel says what the objective deliberately is not.

### Slides 4–7 — The headline and its mechanism

Slide 5 is the one to spend time on. Slide 6 zooms into the delivery transition.
Slide 7 changes the budget fourfold and shows the two arms respond differently.
Slide 8 adds the SNR-randomised training result with the two-row table.

### Slides 9–12 — The controls

Slide 9 is ER-9, and it is the most important slide after the headline. Without
it the low-SNR story looks like "beats JPEG", which is not what we measured. Slide
10 is PAPR. Slide 11 is the secondary controls, including the JPEG outage at
+9 dB that does not flatter us. Slide 12 explains the mechanism behind the cliff.

### Slides 13–14 — Numbers and limits

Slide 13 is the table version of the curves, at both ratios, with the G-10
decision stated. Slide 14 is the limits slide.

### Slide 15 — Course correction

Four corrections, required by PR-8(c). AM-34 strengthened the baseline. AM-52
changed the SNR grid. AM-58 caught a checker that was reporting success while
breaking four of its own rules. AM-60 moved the test-split release three weeks
later. If the panel asks what we changed after seeing results, this is the slide.

### Slides 16–18 — Novelty, timeline, close

Slide 16 is the novelty statement with its negative check. Slide 17 is the
remaining schedule. Slide 18 is the close, with the sealed test split stated
plainly.

## 5. Required figures and provenance

| Figure | File | Qualification that must travel with it |
|---|---|---|
| 01 | `01_headline_full_snr.png` | Validation, one seed cell, different scorers on the two sides |
| 02 | `02_low_snr_closeup.png` | Bracket between −5 and −4 dB, no measured point inside it |
| 03 | `03_bandwidth_efficiency.png` | Compare within each panel; the crossover recrosses at 1/24 |
| 04 | `04_training_robustness.png` | Two checkpoints, one seed cell — not seed variance |
| 05 | `05_task_aware_digital.png` | Does not isolate the channel code |
| 06 | `06_papr_tradeoff.png` | Symbol-domain PAPR, not waveform or amplifier |
| 07 | `07_secondary_controls.png` | Reconstruction panel uses a different scorer |
| 08 | `08_delivery_coverage.png` | 10.0% floor is the outage policy, not a delivery |

Use the PDF or SVG in the report. Use the PNG in the deck.

## 6. Anticipated questions

Short answers are in the presenter guide. The four most likely:

**Is the crossover real or an artifact of the outage fallback?** Partly outage.
The learned curve has no floor because its contract returns a label at every
SNR; the digital curve has one because when it cannot deliver, every image
falls back to a single class. That is the honest mechanism, and it is why slide
12 exists.

**Different classifiers on each side — does that invalidate the comparison?** It
makes it a whole-system comparison rather than a controlled test of channel
coding. Both scorer streams are published, so a reader can check.

**Why not just send the class label?** That is in the deck as a control. It
reaches 82.0% once it delivers, which bounds what any receiver-side intelligence
is worth here.

**One seed cell. Is that enough?** No, and we say so. It is enough to close the
crossover gate on preregistered semantics, and not enough for a significance
claim. The test campaign at G-12 and the additional seed cells are scheduled for
that.

## 7. Second Review scope boundary

In scope: the validation rehearsal results, the controls, the analysis, the
documented course corrections, the novelty statement, the updated schedule.

Out of scope and not claimed: any test-split result, hardware or waveform
measurements, the demo, the poster, the report. Tier 2 and Tier 3 remain
stretch and nothing in this package depends on them.

There is no fallback in this review. G-10 found a crossover, so DEC-16's
fallback clause never fired, and the objectives were not restated at the Second
Review as PR-8(b) permits.

## 8. Readiness matrix

| Item | State | Where |
|---|---|---|
| Deck, editable | Ready | `semantic-communication-second-review.pptx` |
| PDF proof and previews | Ready | same directory |
| Results figures | Ready, validation-only | `presentation-results/figures/` |
| Objectives slide, PR-8(a) | Ready | slide 3 |
| §17 distillation slide, PR-8(c) | Ready | slide 15 |
| Novelty negative check, PR-7 | Ready | slide 16 |
| Gantt current, PR-2 | Ready | `docs/gantt-plan.md`, slide 17 |
| Test-split results | None exist, none claimed | — |
| `review-2-basis` tag | Cut on this branch | — |