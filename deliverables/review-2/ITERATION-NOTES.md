# Iteration notes

Second Review deck · `tools/build_second_review_ppt.py`

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