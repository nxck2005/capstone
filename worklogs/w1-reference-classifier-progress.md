# W1 Reference Classifier Progress

- Corrective base HEAD: `b99bbdf33a4f0dbb762ef1215ed90624e85f1d4c`
- Production code commit: `89a3af48c48a91d6d272ba62337f890c59bb40a5`
- Last updated: 2026-07-29
- Current status: integrity implementation committed; full production evidence generated.
- Full 100-epoch Imagenette training: completed fresh from epoch zero.
- G-1: **PASS** — W1 complete, W2 open.

## Production campaign and G-1 adjudication

The exact production command was:

```bash
.venv/bin/python tools/train_reference_classifier.py \
  --config configs/reference-classifier-clean.yaml \
  --dataset imagenette160 \
  --device cuda \
  --full-run
```

No production artifact or checkpoint existed before launch, so the run was fresh, not resumed. It
used the committed clean configuration without smoke controls, an output override, external
datasets or direct epoch helpers. The original process ran uninterrupted from epoch 0 through 99.

- Best validation result: **898/1000 = 0.898**, epoch 99.
- Final validation result: **898/1000 = 0.898**, epoch 99.
- Preregistered floor: 0.88; result: **G-1 PASS**.
- Final and best checkpoint ID:
  `9c37362347a0203597d6e8e9d9a58fde30ba286f3cec9b4d2f800bd8a3256002`.
- Config hash:
  `a9717575d71f2b3e9dd411b10b7735bdb3946c985fead48cb3c5af07423f12e1`.
- Imagenette archive/dataset version:
  `64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5`.
- Imagenette split-manifest hash:
  `224309422f15bf89460559381aea4b00c4779c52d3652f7f679a213369f3f889`.
- Architecture/variant/seed: Torchvision ResNet-18, `clean`, seed 0, weights null.
- Parameters: 11,181,642 total and trainable.
- Accelerator: NVIDIA GeForce RTX 4060 Laptop GPU, driver 592.82, Torch CUDA 13.0.

Independent verification loaded the finalized checkpoint on CPU, recomputed its exact-file SHA-256,
reloaded and rehashed the committed configuration, and compared both with every aggregate artifact.
It checked 100 contiguous training records and 100 scheduled validation records for epochs 0–99;
each validation accuracy was exactly its integral `n_correct / n_total`; every keyed epoch
permutation matched an independent regeneration; the optimizer recipe and epoch-99 scheduler state
matched config; no smoke bound was present; and all full-run completion and G-1 lineage fields were
consistent. Recomputed maximum selection retained the earliest exact tie and selected epoch 99.
The trainer's detached transactional full-mode resume validator also accepted the finalized
checkpoint.

The wider gate passed:

- Archive byte length/SHA-256 and all three manifest hashes reproduced; manifests regenerated
  byte-for-byte with the configured split counts.
- Registry coverage, split integrity, config round-trip/fingerprint, canonical source identity,
  preprocessing, CUDA assertions and the clean CPU-lock install all passed their existing checks.
- The production CLI established full lineage before constructing its internally owned
  `TrainingClassifierDataset` and `ValidationClassifierDataset` views or writing aggregate results.
  Artifact timestamps place all four aggregate outputs after the finalized epoch-99 checkpoint.
- Test remained sealed. The ordinary registry rejects `test`; the AST guard proves no production
  module imports `src/data/test_access.py`; the full trainer can construct only train and validation
  views; and the instrumented provenance scan reported zero decoder and zero canonicalization calls
  for every published test split. The provenance regression also forbids the model-facing loader.
  Consequently the G-1 path had zero test-image decoding, canonicalization, model-facing loading,
  inference and accuracy computation. No new data path or regression test was needed.

Preflight results were: 186 requirements (2 retired) and 10 generated files current; 11 current
hand-written docs consistent; literal lint scanned 17 files with 0 findings and 10 reasoned
annotations; packetisation checked 216 configurations with 215 feasible, all 144 proof-obligation
cases feasible and 0 failures; archive, manifest and real-data checks passed; the CPU lock passed a
clean hashed install; CUDA device matrix multiplication succeeded; and the complete suite was
**250 passed in 10.79s**. Final verification repeated the spec, documentation, literal,
packetisation, archive, manifest and real-data checks successfully; the real-data verifier again
reported zero test decoder/canonicalization calls for all three datasets; pytest reported
**250 passed**; and `git diff --check` passed. The final diff contains no recipe, source, test,
specification, generated-view, manifest or lockfile change.

## Historical pre-G-1 trainer-integrity correction

A subsequent production audit of the then-staged checkpoint work found three remaining integrity
gaps. They were corrected without changing AM-78 or any normative parameter and committed in
`89a3af4`:

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

## Historical staged paths

`NEXT.md`, `src/training/reference_classifier.py`, `tests/test_classifier_training.py`,
`tests/test_classifier_checkpoint.py`, `tests/test_classifier_cli.py`,
`tools/train_reference_classifier.py`, `worklogs/w1-reference-classifier-progress.md`.

At that point no archive download, full Imagenette campaign or G-1 execution had occurred, and the
next action was user review of the staged corrective diff. This paragraph is retained only as a
point-in-time record; the current status and production evidence are above.
