# Second Review package

Capstone project · Second Review · 29 September – 3 October 2026 · W10

Derived from the annotated git tag `review-2-basis` on this branch, re-cut at
each substantive revision so it points at the deck held here rather than an
earlier draft.

## What is in here

| File | What it is |
|---|---|
| `semantic-communication-second-review.pptx` | The deck. Native editable shapes, so anyone can revise wording without rebuilding. |
| `semantic-communication-second-review.pdf` | Rendered copy for machines without an Office renderer. |
| `semantic-communication-second-review-contact-sheet.png` | All 20 slides on one page, for a fast check of flow. |
| `previews/slide-NN.png` | Per-slide previews at 1600×900. |
| `review-2-presenter-guide.md` | What to say on each slide, and answers to the questions we expect. |
| `ITERATION-NOTES.md` | What changed between drafts. |

Builder: `tools/build_second_review_ppt.py`. Re-run it to regenerate everything
from the committed W10 figures.

## Look

The deck follows `deliverables/review-1/semantic-communication-first-review.pptx`
— the plain First Review deck, not the `academic-v2` one. That file is the
reference, and note that `tools/build_first_review_ppt.py` no longer generates it:
the shipped `.pptx` is newer than the script and has no header banner, no section
marker and no footer chips. Copy the rendered deck, not the builder.

The idiom, in short:

- Title and one hairline rule. No banner, no page furniture above the line.
- A footer of plain text: `Rubric: ...`, a citation, and `N / 20`.
- Sharp-cornered white boxes with a thin grey border. No rounded corners.
- Real tables — every row a bordered band with vertical separators.
- Bold term on the left, plain text on the right, for anything explanatory.
  No boxes needed.
- One typeface, black and two greys. No colour, no circles, no accents.

There is no `label()` or chip helper in the builder, and adding one back would
make this deck look unlike the First Review one.

## How it talks

Slides state what the measurements support. The qualifications live on slide 15
and are referenced from there — they are not restated as a closing note on each
results slide, because a caveat repeated on every chart stops being read.

Concretely:

- The sealed-exam-set fact appears once in the evidence base (slide 5, as a table
  row) and once as the closing statement on slide 19.
- The single-training-run qualification appears on slide 15.
- The scorer difference appears on slide 5 and on slide 15, where it belongs.
- Mechanism explanations stay on the slide they explain. The 10% fallback floor is
  on slide 4; the delivery mechanism is on slide 8, directly behind the close-up
  that raises the question.

Slide titles are claims or questions, never admissions. "What this evidence
cannot tell you" became "How to read these results"; "Not claimed as new" became
"What this builds on"; "What we did not aim for" became "How these are scored".

Bandwidth ratios keep their spec notation — `1/6` and `1/24` — because the
embedded figures are titled that way. An earlier draft called them "half rate"
and "quarter rate", which contradicted both the chart on the slide and the
spec, where 1/6 is not half of anything.

## What does not appear on a slide

The audience is a review panel. Our own machinery stays out of the deck:

- No repository paths, no directory names
- No requirement IDs, gate names or amendment numbers
- No internal week numbering — the schedule slide runs on dates
- No spec section references

The four course corrections on slide 16 are labelled by what each one *was*
(Baseline too weak, Signal grid too coarse, A faulty check, Exam set unlocked too
early) rather than by amendment number. An automated scan for paths, gate IDs,
requirement IDs, amendment IDs, spec refs and week numbers returns zero across
all 20 slides.

This constraint is about the deck only. The package note, the iteration notes and
the presenter guide are internal working documents and do carry the IDs, because
the presenting team needs them to trace a claim back to its source.

## Where the results come from

Every number traces to the published W10 v11 validation aggregate. Nothing was
re-derived for the deck and no curve was redrawn.

- Aggregate CSV: `presentation-results/data/w10_primary_252.csv` — 252 rows,
  12 system variants × 21 signal strengths
- Scorer streams: `presentation-results/data/w10_scorers_357.csv`
- Figures 01–08: `presentation-results/figures/{png,pdf,svg}/`
- Written findings: `presentation-results/report/findings.md`
- Source evidence: `results/learned/w10/`

`test_access` is 0 and `test` is `SEALED` across the whole project.

## Delivery acceptance contract

1. Deck is committed under `deliverables/` and derived from the `review-2-basis`
   annotated tag (SPEC PR-8(d)).
2. Objectives slide states objectives as completion criteria (PR-8(a)).
3. The §17 course-correction slide is present (PR-8(c)) — mandatory for the
   Second and Third packages.
4. Every results slide carries its held-back-set qualification, in the citation
   footer.
5. `docs/gantt-plan.md` is current and shows W11–W17 honestly.
6. No claim in the package rests on test-split results.

## 1. Rubric coverage

The Second Review is 30 marks, carrying six criteria. The deck maps onto them:

| Criterion | Sub-marks | Slides | How the deck answers it |
|---|---|---|---|
| Results (graphs/tables/test cases) | 20 | 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 | Eight measured figures plus two numeric tables at both rates. 252 published measurements. |
| Originality | 10 | 18 | Two claims tied to experimental design, with the prior art each rests on. |
| Analytical skills | 5 | 5, 6, 8, 9, 15, 16 | Why the curves break where they do, what the controls buy, how to read the numbers. |
| Presentation | 5 | all | 20 slides for a 25-minute slot. Presenter guide carries the timing. |
| Methodology | 5 | 17 | The one-slide course-correction summary required by PR-8(c). |
| Timeline | 5 | 2, 19 | Progress since the First Review, and the remaining weeks with their gates. |

## 2. Narrative contract

**Teach the vocabulary before using it.** Slide 4 does this in one pass. Once a
panel understands "the flat 10% line means nothing arrived", every later chart is
readable.

**Establish the evidence base before any claim.** Slide 5 gives the size and shape
of the measurement set before slide 6 shows a curve.

**Separate what was measured from what it means.** The two sides are marked by
different classifiers, so the headline gap is a whole-system quantity. Slide 5
says so, slide 15 records it as a qualification, and slides 10–13 supply the
controls that let a panel judge it.

**Give the qualifications once, in one place.** Slide 15.

## 3. Twenty-five-minute plan

| Slides | Time | Content |
|---|---|---|
| 1–3 | 4 min | Title, progress since August, objectives as completion criteria |
| 4 | 2 min | The plain-language primer |
| 5 | 1.5 min | What we measured |
| 6 | 1.5 min | The two models |
| 7–9 | 5 min | Headline, the delivery transition, and what causes it |
| 10–14 | 6 min | Bandwidth, randomised training, the control, peak power, secondary controls |
| 15–16 | 3 min | Numbers table, how to read them |
| 17 | 2 min | Course correction |
| 18–19 | 2 min | Novelty, timeline |
| 20 | 1 min | Close |

Roughly 28 minutes of planned content against a 25-minute slot, so the slack
comes out of slides 13 and 14. Both are secondary controls and the numbers
survive in the report.

## 4. Slide content contract

### Slides 1–3 — Frame it

Slide 1 states the thesis. Slide 2 is the progress log. Slide 3 is the objectives
slide and does the most quiet work: the rubric scores `Objectives Met` at the
Third Review by whether the First Review's objectives were met, and the right
panel states that completion is independent of the result.

### Slide 4 — The on-ramp

The structurally important slide. It defines four words, works an example, and
explains the flat 10% line that otherwise looks like a suspicious floor.

### Slide 6 — The models

Architecture, training recipe, the classifier that reads the pictures, and what
training cost. Every figure is traceable to committed evidence.

### Slides 7–9 — The headline and its mechanism

Slide 7 is the one to spend time on. Slide 8 zooms into the delivery transition,
slide 9 explains what causes it.

### Slides 10–14 — The controls

Slide 12 is the task-aware digital control, the most important slide after the
headline. Slide 13 is peak power, and slide 14 the secondary controls including
the JPEG outage at +9 dB.

### Slides 15–16 — Numbers and qualifications

Slide 15 is the table version of the curves at both rates. Slide 16 is the single
place qualifications are stated.

### Slide 17 — Course correction

Four corrections, required by PR-8(c).

### Slides 18–20 — Novelty, timeline, close

## 5. Required figures and provenance

| Figure | File | Qualification that travels with it |
|---|---|---|
| 01 | `01_headline_full_snr.png` | Held-back pictures, one run, different graders on the two sides |
| 02 | `02_low_snr_closeup.png` | Transition lies between −5 and −4 dB |
| 03 | `03_bandwidth_efficiency.png` | Compare within each panel; the crossing happens twice at 1/24 rate |
| 04 | `04_training_robustness.png` | Two separately trained models |
| 05 | `05_task_aware_digital.png` | Feature encoders differ, so this compares representations |
| 06 | `06_papr_tradeoff.png` | Measured on the numbers we send |
| 07 | `07_secondary_controls.png` | The rebuilding panel uses a different grader |
| 08 | `08_delivery_coverage.png` | The flat 10% is the fallback rule |

PDF or SVG in the report, PNG in the deck.

## 6. Second Review scope boundary

In scope: the held-back measurement results, the controls, the analysis, the
documented course corrections, the novelty statement, the updated schedule.

Out of scope: any exam-set result, hardware or waveform measurements, the demo,
the poster, the report. Tier 2 and Tier 3 remain stretch.

There is no fallback in this review. The crossing was found, so the fallback
clause never fired and the objectives were not restated as PR-8(b) permits.

## 7. Readiness matrix

| Item | State | Where |
|---|---|---|
| Deck, editable | Ready | `semantic-communication-second-review.pptx` |
| PDF proof and previews | Ready | same directory |
| Plain-language primer | Ready | slide 4 |
| Results figures | Ready, held-back set only | `presentation-results/figures/` |
| Objectives slide, PR-8(a) | Ready | slide 3 |
| Model parameters | Ready | slide 6 |
| §17 distillation slide, PR-8(c) | Ready | slide 17 |
| Novelty negative check, PR-7 | Ready | slide 18 |
| Gantt current, PR-2 | Ready | `docs/gantt-plan.md`, slide 19 |
| Exam-set results | None exist, none claimed | — |
| Guide hardware acknowledgement | Carried from the First Review, unverified | — |
| Four-member rehearsal | Carried from the First Review, unverified | — |