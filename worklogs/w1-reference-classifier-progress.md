# W1 Reference Classifier Progress

- Corrective base HEAD: `b99bbdf33a4f0dbb762ef1215ed90624e85f1d4c`
- Last updated: 2026-07-29
- Current status: final three pre-G-1 trainer-integrity corrections staged; not committed.
- Full 100-epoch Imagenette training: not run.
- G-1: not executed or adjudicated.

## Final pre-G-1 trainer-integrity correction

A subsequent production audit of the prior staged checkpoint work found three remaining integrity gaps.
They are corrected without changing AM-78 or any normative parameter:

1. **Official full lineage only** — only `run_epochs(..., execution_mode="full",
   full_run_requested=True)` can establish full lineage. It establishes that state before production
   dataset construction or artifact work, rejects external training/validation datasets, bounds and
   non-configured full schedules, and constructs only `TrainingClassifierDataset` and
   `ValidationClassifierDataset`. Public `train_epoch()` and `validate_epoch()` are permanently
   smoke/test hooks even when unbounded; a smoke trainer cannot be promoted and its checkpoint cannot
   resume in full mode.
2. **Complete optimizer recipe validation** — resume constructs a detached fresh optimizer from the
   resolved configuration, requires identical group count, parameter cardinality, complete non-`params`
   key set, exact value types and values, and substitutes only the completed epoch's configured
   learning rate. This detects `maximize`, `foreach`, `differentiable`, `fused`, all configured SGD
   fields, missing/extra keys and group/cardinality changes before any live mutation.
3. **Integer-evidence validation history** — every validation record requires finite exact
   `n_correct / n_total` accuracy and is normalized from those counts. Full lineage requires precisely
   the configured validation epoch sequence through the completed epoch. Best state is recomputed
   from validated counts, with the earliest epoch retained on an exact tie; serialization and resume
   both reject inconsistent metric, epoch, schedule or completion state.

## Regression coverage and verification

- Extended `tests/test_classifier_training.py`, `tests/test_classifier_checkpoint.py` and
  `tests/test_classifier_cli.py` for direct
  unbounded hooks, smoke/full resume separation, internal full-view construction, rejection before
  state/artifact mutation, complete optimizer-group mutation coverage, count/accuracy integrity,
  full schedule omissions/duplicates/unexpected/out-of-order records, transactional rejection and
  earliest-tie selection.
- Baseline focused suite before correction: `35 passed`.
- Final focused training/checkpoint/CLI suite: `62 passed`.
- Final classifier-related suite: `114 passed`.
- Final complete suite: `250 passed`.
- Manual exploit outcomes: arbitrary direct data remained smoke-only and full resume rejected it;
  full-mode external train/validation data rejected before work or artifact creation; `maximize=True`
  and representative optional `foreach=True` payloads rejected transactionally; inconsistent
  `3/6 -> 0.999` payloads rejected transactionally; removing a scheduled full validation record
  rejected transactionally. These were run as disposable `PYTHONPATH=src:. .venv/bin/python - <<'PY'`
  harnesses against temporary checkpoints, with no repository artifacts created.
- Repository checks passed: spec views, documentation consistency, literal lint, packetisation,
  archive provenance, manifest materialization, real train/validation dataset verification and
  `git diff --check`.

## Current staged paths

`NEXT.md`, `src/training/reference_classifier.py`, `tests/test_classifier_training.py`,
`tests/test_classifier_checkpoint.py`, `tests/test_classifier_cli.py`,
`tools/train_reference_classifier.py`, `worklogs/w1-reference-classifier-progress.md`.

No archive download, full Imagenette campaign or G-1 execution occurred. The next action is user
review of the staged corrective diff.
