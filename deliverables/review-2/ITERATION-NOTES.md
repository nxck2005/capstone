# Iteration notes

Second Review deck · `tools/build_second_review_ppt.py`

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
serves, and those names print in the footer. Slide 15 exists specifically to
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

A geometry scan over every slide now reports zero elements outside the content
area and zero image-to-panel collisions.

### What was left out, deliberately

- The reconstructed-vs-direct comparison is on slide 11 as a panel only. It
  changes the scorer, so it is a pathway ablation rather than a measurement of
  reconstruction quality, and giving it its own slide would overstate it.
- The `clean` classical scorer stream is not plotted. It belongs in a labelled
  supplementary table in the report, where the caveat can travel with it.
- No test-split result appears anywhere, because none exists.