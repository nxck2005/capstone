# Iteration notes

Second Review deck · `tools/build_second_review_ppt.py`

## v8 — 2026-09-30, "arm" removed and the mechanism slide moved up

**The undefined term.** "Arm" is this project's own name for a system variant —
it comes from the column header in the W10 aggregate — and it appeared on two
slides without ever being defined. A panel reading "This arm is harder" on slide
11 has no way to know what an arm is. Slide 5 already introduces the plain term
("Versions of the system"), so the deck now uses *system* throughout and the word
"arm" appears nowhere.

**The flow.** Slide 13 explained why the digital line breaks — delivery coverage
tracking accuracy — but it sat five slides after the close-up that shows the
effect and six after the headline that raises the question, with three unrelated
controls in between. A reader who wanted to know *why* had to hold the question
for half the deck. It moves to position 8, directly behind the close-up, so the
results run:

```
6  headline          the claim
7  close-up          the evidence
8  why it breaks     the explanation
9  quarter the radio first variation
10 random training    learned-system variation
11 task-aware control the attribution control
12 peak power         a design constraint
13 four controls      the remaining controls
14 numbers            both tables
15 how to read them   the qualifications
```

Claim, evidence, explanation, variations, controls, tables, qualifications. The
old order put the explanation after the variation slides, which is backwards for
a reader who has just been shown something they do not yet understand.

Slides 8–13 renumbered to 9–14. The presenter guide's sections were reordered to
match and the package note's slide references updated, so the 25-minute plan now
reads 6–8 for headline/close-up/mechanism and 9–13 for the variations and
controls.

## v7 — 2026-09-30, repository internals removed

The audience is a capstone review panel. v6 had left our own machinery on the
slides, which the panel has no way to read and no reason to care about:

- Slide 19 carried an "Evidence trail" block listing `results/learned/w10/`,
  `presentation-results/` and `spec/SPEC.md §17`
- Slide 17 cited `docs/literature-review.md`
- Slide 16 tagged each correction `AM-34` / `AM-52` / `AM-58` / `AM-60`, and its
  footer read "Required by PR-8(c): a one-slide distillation of SPEC §17"
- Slide 18 ran a `W10`–`W17` column and a note about "W15 internal freeze"
- Slides 5 and 19 referred to "gate G-12"
- Slide 15 referred to "W14"

All of it is gone. The four correction boxes on slide 16 are now labelled by what
each correction *was* — Baseline too weak, Signal grid too coarse, A faulty
check, Exam set unlocked too early — which is what a reader needs and the
amendment numbers only meant to us. Slide 18's week column is dropped in favour
of the dates it duplicated. "Gate G-12" becomes "week 11" everywhere.

Slide 19's closing block changed purpose rather than just losing its paths: an
"Evidence trail" listing directories is replaced by "What we are asking for",
which says the per-picture records are available on request and states the seal.
A panel wants to know what they can ask for, not where our files sit.

An automated scan for paths, gate IDs, requirement IDs, amendment IDs, spec
section refs and week numbers now returns zero across all 19 slides. The
presenter guide's spoken lines were cleaned in the same pass, because a presenter
reading "it stays sealed until gate G-12" aloud in a viva would have to stop and
translate it.

## v6 — 2026-09-30, second humanisation pass

v5 killed the negations. A keyword audit then came back nearly clean — 7
negations, 0 "not just X", 0 buzzwords, 13% of sentences opening with "The" —
which is the point at which the remaining tells are structural rather than
lexical and a script stops finding them. So this pass was done by reading all
263 sentences of deck copy instead.

What that found:

**A note restating the row above it.** Slide 2 ended with "Everything above ran
on held-back pictures. The exam set stays locked until gate G-12", directly under
a final row reading "Still locked. It opens in week 11 once the locking document
is signed." Deleted. Slide 19's "So far, zero exam pictures opened" sat under a
closing line saying the same thing. Deleted.

**Whole slides built from chained participles.** Slide 2's eight rows were
"Measured…, tested…, and froze…" / "Built…, trained…, and froze…" /
"Trained…, and compared…" — every row the same shape, which is how a model writes
a list. Four rewritten to break the pattern.

**Leads rewording their own title.** Slide 6's title said where the digital
system gives up and takes over; the lead then said the same thing a third way.
The lead now does the job a lead should: "One line runs the whole width of the
chart. The other has a step in it near the left." Slide 16's "The rubric asks
what we decided based on the results obtained. These are the four that count"
became "What we changed once we could see the measurements." Slide 17's lead
repeated its own title and its own footer, and was deleted outright.

**Rhetorical flourishes.** Slide 10 opened "If we had only compared against a
normal compressed image, the weak-signal story would be too easy to believe" — a
move that argues rather than informs. Now: "A JPEG image is an easy thing to
beat. This arm is harder, and it is the one worth reporting."

**Tricolon where two would do.** "the link, the representation, and the marker"
became "whole systems, link and marker included."

**Five rows opening the same way.** Slide 15 ran "Every figure is…" / "The learned
systems are…" / "The digital settings…" / "Peak-to-average ratio is…" / "The
fallback rule…". Reopened so the row rhythm varies.

**Em dashes doing a full stop's job.** Four down to two; the remaining pair are
table labels ("Half rate — 12,800 uses of the radio"), which is label syntax
rather than prose punctuation.

Also cut a "That is a 23 point loss" calculator line on slide 8 — the two numbers
either side already say it — and softened "Two things that govern how every later
chart should be read" to "Two things to keep in mind".

Negations went 7 → 8, and that is fine: the new one is a classification ("Five are
controls, not headline systems"). All eight remaining are statements about
measurements, about prior work, or about how a row is categorised.

## v5 — 2026-09-30, negative framing removed

An audit of the built file counted **60 strings containing a negation**
(`not`, `never`, `no`, `nothing`, `cannot`, contractions). Across 19 slides that
is three per slide, and they clustered into three habits:

1. **Repeated caveats.** "The exam set has never been opened" appeared on slides
   1, 4, 5 and 19. "One training run" appeared on 5, 9 and 15. Repeating a
   qualification on every chart is what makes a reader stop reading it.
2. **Disclaimers as content.** Slide 14 carried a whole block titled "What this
   table is not". Slide 3's right-hand panel was titled "What we did not aim
   for". Slide 17 had a panel titled "Not claimed as new". Each was a slide
   region whose job was to say what the work was not.
3. **Defensive trailing notes.** Slides 6, 7, 8, 9, 11 and 13 each ended with a
   hairline and a small italic note apologising for the figure directly above it.

The fix was mostly deletion rather than rewording, because rewording a caveat
into a positive sentence leaves a caveat wearing a hat. What remains:

- The sealed-exam-set fact appears **once** in the evidence base (slide 5, as a
  table row reading `0 · Sealed until gate G-12`) and **once** as the closing
  line of slide 19.
- The single-run qualification appears **once**, on slide 15.
- Slide 14's disclaimer block is gone, replaced by "Reading the two tables" —
  which says what the tables show.
- Slide 3's panel is now "How these are scored", which makes the rubric point
  positively instead of disclaiming a strawman objective.
- Slide 17's panel is now "What this builds on", framing the four established
  ideas as foundations.
- Slide 15 is retitled "How to read these results", and its five rows are
  positive headings: One training run, Whole systems, Different amounts of work,
  Simulated channel, Frozen operating points.
- Six trailing apology notes were deleted outright.

Where a qualifier is genuinely load-bearing it stayed: the 10% fallback floor on
slide 4 is the mechanism that makes the headline intelligible, and the delivery
explanation on slide 13 is the answer to the most technical question in the room.

**60 → 7.** The seven survivors are all factual statements about results or about
prior work — "the digital system delivers nothing", "1,000 pictures never
arrived", "reported no problems while breaking four of its own rules", "prior
work reports the gap without this control". None of them hedge the project's own
confidence.

The presenter guide was rewritten for the same reason and to match: it had been
quoting the deleted panels verbatim, so it was both out of date and teaching the
mannerism. Its "never say X" rules are now "say Y" instructions.

## v4 — 2026-09-30, decoration removed

v3 fixed the palette but kept a decorative vocabulary the plain First Review
deck does not use at all: pill-shaped footer chips, a header banner with a `§ NN`
marker, rounded corners, thick accent bars on the left of every box, numbered
circles on the objectives slide, filled circles on the timeline, and five
oversized numeral tiles on slide 5. For a first-year project review that reads as
a designed product rather than an academic artifact.

The reference was checked in the *rendered* file, not the builder script, because
the two disagree: `tools/build_first_review_ppt.py` still emits a banner, a `§`
marker and footer chips, but the shipped `.pptx` (23 Aug, newer than the script)
has none of them. The deck that was actually presented is the ground truth, so
its geometry was read shape by shape and copied.

What review-1 plain actually does, and what v4 now does:

* **Header** — title at x 0.62, y 0.48, one hairline at y 1.18. No banner, no
  section marker. v4 matches exactly.
* **Footer** — hairline at y 7.02, then plain text: `Rubric: ...` at x 0.62, a
  right-aligned citation at x 3.45, and `N / 19` at x 11.72. The chips are gone.
* **Boxes** — sharp corners, white fill, `777777` border. Every `radius=` in
  build_scenes is gone.
* **Tables** — every row is its own bordered band with vertical separators,
  `F3F3F3` header. v3's rules-only table was a different animal; v4 uses a real
  grid.
* **Explanatory text** — bold term on the left, plain text on the right, hairline
  between rows, and no box around it at all. This is the idiom the plain deck
  uses for every definition on every slide, and it replaces most of v3's cards.
* **Colour** — black, one grey, one hairline grey, and a border grey. Nothing
  else.

Deleted outright: `label()`, `gloss()`, `stat()`, and the `card()` accent strip.
Slide 5's five numeral tiles became a table. Slide 11's four tiles became a
table. Slide 12's four boxes and slide 15's five boxes became term/definition
rows. Slide 18's timeline lost its filled circles and became ruled rows.

Verified against the reference rather than by eye:

```
review-1 plain   TEXT_BOX 243  LINE 102  AUTO_SHAPE 62   ovals 0  rounded 0  Arial
review-2         TEXT_BOX 407  LINE 126  AUTO_SHAPE 41   ovals 0  rounded 0  Arial
```

Same three shape classes, no ovals, no rounded rectangles, one typeface. The
extra `PICTURE` class is the eight embedded W10 figures, which the First Review
deck had no equivalent of because it had no results yet.

Two layout faults fixed on the way: slide 14's right-hand table ran 0.6 in past
the right margin, and three closing notes crossed the footer rule.

## v3 — 2026-09-30, style corrected to the non-v2 First Review deck

v1 and v2 were built against
`deliverables/review-1/semantic-communication-first-review-academic-v2.pptx`. That
was the wrong reference. The deck to match is
`semantic-communication-first-review.pptx` — the plain one, which is also the
newer file on disk (23 Aug against 18 Aug). The two decks share an architecture
but not a look:

| | non-v2 First Review (target) | academic-v2 (what v1/v2 used) |
|---|---|---|
| ground | white | ivory `#F7F5EF` |
| fonts | Arial only | Georgia headings, Cascadia Mono numbers, Cambria Math |
| body colour | `#111111` ink, `#555555` muted | `#18212B` ink, `#5B6570` muted |
| accents | none — black, grey and rules | navy, burgundy, green, amber |
| body type | 10–13 pt, median 12 | 8–10 pt, median 9.1 |
| stat numerals | up to 34 pt | 27 pt |

So v3 collapses the palette. `IVORY`, `PAPER` and `WHITE` all become `FFFFFF`;
`INK` and every accent become `111111` or a grey; the four pale tints all become
`F3F3F3`; `LINE` becomes `D9D9D9`. All fonts become Arial, with Courier New for
figures.

**Colour had been carrying meaning.** With accents gone, emphasis moves to
weight, rules and label chips: heading weight instead of hue, a `D9D9D9` rule
instead of a tinted border, and an upper-case label chip for every block. Where
a slide flagged something unflattering, the accent bar steps down to `999999`
rather than to burgundy — slide 12's JPEG result and slide 15's amplifier row.

**Type scale raised and copy cut to fit.** Body copy went from 9–10.5 pt to
10.5–13 pt, stat numerals to 30 pt, table values to 11.5 pt. That does not fit
the old word counts, so every dense block was shortened rather than shrunk. The
figures shrank slightly on slides 7, 9 and 12 to buy the space.

**Layout fixes found by looking at the render.** The footer page number was
wrapping to two lines from slide 10 onward — its box was 0.52 in wide for
"10 / 19", now 1.10 in. Slide 9's comparison table ran under the footer rule.
Slide 7's caption and slide 15's closing note crossed into the footer band.
Geometry scan now reports zero collisions.

## v2 — 2026-09-30, plain-language pass

Nineteen slides, up from eighteen. One slide added and almost every line of body
copy rewritten.

**The problem this fixed.** v1 opened on "252 measurement units, each one row of
the published aggregate" and had `end-to-end accuracy`, `coverage`, `task head`,
`artifact-finetuned classifier`, `ER-9`, `outage` and `PAPR` scattered through the
results section with nothing defining them first. The First Review deck does not
do that — it opens with "Imagine a remote camera sending images to a server that
decides what it sees" and only then reaches for AWGN. v2 follows that pattern.

**New slide 4, "Before the numbers, one picture."** Four concepts in ordinary
words, the term used for each, and where it appears on the charts: signal
strength, pictures that got through, how many we labelled correctly, and how many
uses of the radio. The right-hand panel works the −8 dB case end to end and
closes on the flat 10% — it is not a bad prediction, it is 1,000 pictures that
never arrived. That one number is the most misread thing in the deck and slide 4
is where it gets explained.

**Vocabulary swaps.** `classical chain` → `digital system`; `coverage` →
`pictures that got through`; `top-1 accuracy` → `how many we labelled
correctly`; `channel-use budget` → `radio time`; `outage` → `arrived` or
`fallback answer`; `task head` → `the classifier built into it`; `SNR-randomised`
→ `random each time`; `test split` → `exam set`. ER-9, PAPR, LDPC and QPSK are
written out on first use and kept only in citation footers where a panel member
can trace them.

**Numbers as counts.** Percentages became "728 of the 1,000 pictures" wherever
the sentence allows it, because a percentage with an unstated denominator is the
thing that makes a results slide hard to follow.

**Titles.** Rewritten as plain statements or questions: "Where the digital system
gives up, and where it takes over" rather than "Learned DJSCC against the
adaptive classical chain".

**Layout.** New rows on slide 4 at 0.78 in spacing inside taller panels, after
the first render put the closing note on top of the last row.

## v1 — 2026-09-30, first build

Eighteen slides for the 25-minute Second Review slot, built from the same scene
description pattern as the First Review builder so the two decks share a visual
language: ivory ground, navy and burgundy accents, serif headings, sans body,
mono for numbers, and a footer that names the rubric criteria each slide answers.

The First Review deck had twelve slides and no images. This one embeds the eight
committed W10 figures as PNGs rather than redrawing curves from the CSV, so the
plotted data and the deck cannot drift apart. Adding image elements meant
extending the element model with a `src` field and a paste step in the PIL
renderer.

### Rubric mapping

The Second Review carries six criteria. Every slide declares which ones it
serves, and those names print in the footer. Slide 16 exists specifically to
answer `Methodology` and slide 3 to answer `Objectives Met`, both required by
SPEC PR-8.

### Layout corrections made during the build

Three overflows showed up in the rendered previews and were fixed against the
previews rather than guessed at:

1. Slide 11's four control notes used the shared `card()` helper at a height of
   0.70 in, which left no room for a heading and body, so the text ran past the
   footer. Replaced with a compact two-column note layout at 0.80 in and the
   figure was reduced to make space.
2. Slide 12's figure was centred by default and overlapped the reading panel on
   its right. Pinned to an explicit left origin.
3. Slide 10's `20.059284 dB` stat value wrapped onto its caption at the default
   size. Added a `value_size` argument to `stat()` and dropped that slide's values
   to 21 pt rather than truncating the number.
4. Slide 4's closing note sat on top of its last row in the first v2 render.
   Rows respaced to 0.78 in inside taller panels.

A geometry scan over every slide now reports zero elements outside the content
area and zero image-to-panel collisions.

### What was left out, deliberately

- The reconstructed-vs-direct comparison is on slide 11 as a panel only. It
  changes the scorer, so it is a pathway ablation rather than a measurement of
  reconstruction quality, and giving it its own slide would overstate it.
- The `clean` classical scorer stream is not plotted. It belongs in a labelled
  supplementary table in the report, where the caveat can travel with it.
- No test-split result appears anywhere, because none exists.