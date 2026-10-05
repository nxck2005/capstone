# W10 validation results package

**Source of truth:** published closeout commit `c31dd2bf9edac3a699515852997ab919e73e5d00`, `results/learned/w10/w10_continuation_closeout_v11.json`. The source is the completed W10 v11 closeout spanning v8, v9 and v11 execution epochs: 252 authenticated validation units, 357 scorer streams, 12 arms × 21 measured SNRs. When this package was built the final test was **SEALED** (`test_access=0`). These are validation findings, not final-test results.

> **Update 2026-10-05:** this package predates the G-12 test campaign. G-12 has since run once on the 3,925 test images (2026-10-02) and all four hypotheses are supported. For test numbers use `data/g12_test_curves.csv` and `data/g12_test_differences.csv`, the paper's figures and [`docs/RESULTS.md`](../docs/RESULTS.md). The validation numbers here are unchanged and still correct.

## Use in slides or a paper

Insert the `figures/png/*.png` files into slides. They are 300 dpi and sized for widescreen presentation. Use the matching `figures/pdf/*.pdf` or `figures/svg/*.svg` files for paper/vector workflows. Keep the caption and scorer qualification from [figure-index.md](report/figure-index.md) with each figure. Recommended order and plain-language speaker notes are in [results-slides.md](presentation-notes/results-slides.md).

The three most useful capstone figures are **01** (full comparison), **02** (low-SNR behavior), and **08** (delivery coverage explaining the cliff). Figure **03** adds the bandwidth tradeoff; Figure **06** answers the PAPR fairness question.

## Contents

- `figures/{png,pdf,svg}/`: eight numbered figures in each format.
- `data/w10_primary_252.csv`: one row per published unit, including primary scorer accuracy, delivery counts, PAPR, identity and epoch.
- `data/w10_scorers_357.csv`: every published scorer stream, including the clean-scorer secondary classical streams.
- `data/g12_test_curves.csv`, `data/g12_test_differences.csv`: the later G-12 **test** tables (seed-averaged curves with 95% intervals, and paired differences), written by `tools/export_g12_tables.py`; the paper's figures use them.
- `report/findings.md`: scientific findings, metric definitions and limitations.
- `report/figure-index.md`: captions and exact curves for all figures.
- `presentation-notes/results-slides.md`: a six-slide results sequence and speaker notes.
- `plot_results.py`: editable, read-only extractor and plotter.

## Reproduction and provenance

From repository root at the published commit:

```bash
python presentation-results/plot_results.py
```

Requires Python 3 and Matplotlib (tested with 3.11.1); no model weights, image corpus, validation inference, network access or test access. The script reads only the four committed `results/learned/w10/w10_continuation_*_v11.json` files. It verifies the closeout's SHA-256 links to the unit and per-image manifests, cardinalities, ordinals, unit identities, denominators, delivery arithmetic, scorer counts and the exact 21-point grid before producing any output. Each plotted y value is a published `n_correct / 1000` or `coverage_rate`; each x value is its published SNR. No interpolated measurements or uncertainty bands are used. The CSVs are derived reporting tables, not replacement evidence.

The published per-image manifest authenticates 357 streams but their bulk row files are outside this checkout. This package therefore makes no paired confidence-interval or significance claim. The full-range lines merely connect *observed* points for readability. The scientific source, frozen evidence, CI and test boundary are untouched.
