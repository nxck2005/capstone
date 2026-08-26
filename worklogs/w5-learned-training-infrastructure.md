# W5 learned-training infrastructure closeout

**Date:** 2026-08-27  
**Scope:** training-system engineering and non-scientific plumbing only  
**Verdict:** W5 GREEN; W6 requires separate owner authorization

## Frozen basis

- G8 remained scientifically frozen at `g8closeout-07526958…`.
- The terminal binding presentation defect was repaired additively as
  `g8bindingcorrection-1bff458e…`; historical closeout SHA-256 remains
  `4db4ae531fd20fdfb9c5b44d6b09beb2bc14f95b96e439c43ca6441fe9a4171b`.
- AM-91 and `instructions/W5.txt` froze all output-affecting W5 runtime semantics
  before the first optimizer smoke. Schema version 1 is
  `spec/schemas/w5_training_artifacts.schema.json`.

## Implementation

The production trainer is `src/training/djscc.py`, with the train-only stable-ID
view in `src/data/djscc_training.py`. Entry points are
`tools/run_djscc_training.py`, `tools/run_w5_training_smoke.py`, and
`tools/verify_w5_training_system.py`.

Training channel noise is a stateless keyed function of dataset/version,
manifest, stable sample, train/channel seeds, epoch, channel, ratio, `k`, and
training SNR. Ambient Python/NumPy/Torch/CUDA RNG does not determine the
trajectory. Immutable checkpoints authenticate exact bytes, config/data/source/
profile/model/optimizer/scheduler/scaler lineage, and publish checkpoint →
sidecar → latest atomically.

## Smoke custody

- Attempt 1 exposed incorrect rejection of an expected initial GradScaler
  overflow. It is preserved at `w5_smoke_attempt_1_failure.json` with zero
  optimizer steps and zero completion coverage.
- Attempt 2 proved exact CIFAR kill/resume but bounded selected-ratio coverage
  stopped one backoff too early. It is preserved at
  `w5_smoke_attempt_2_failure.json`; two physical W5-only optimizer steps and
  zero completion coverage.
- Attempt 3 is terminal: `w5smoke-9868ecda…`, SHA-256
  `2dc04add556614dba643bff9848232c1e9de3aee5da07e8d259e70bc72da463a`.
  The uninterrupted and fresh-process resumed branches match exactly in sample,
  augmentation/noise identity, losses, LR, model, optimizer, scheduler, scaler,
  epoch and global step. Ratios `r_1_6` (`k=12800`) and `r_1_24` (`k=3200`)
  each completed finite/nonzero encoder, reconstruction-head and task-head
  backward plumbing.

Across all attempts W5 executed six non-scientific optimizer steps. Scientific
learned runs, W7 pilots, W8 runs, validation selection, learned test inference
and test access are all zero.

## Verification

- Targeted W5/model/loss/channel/power/RNG/config/test-isolation/G8 provenance:
  167 passed.
- Evidence gate: PASS (11 focused tests).
- Full-local: PASS (2,536 tests).
- ci-cpu: PASS.
- Pre-smoke exact source CI: run `33011201226`, job `98317625691`.
- Pre-smoke manifest-carrier CI: run `33012842982`, job `98323311218`.

Terminal completion is `results/learned/w5/w5_completion.json`,
`w5completion-680b2688dc761a30a7a68aee91c021fe057bbb726b44b614bdffd19712c5fc70`.
No smoke accuracy was recorded or used. Next action is separate W6 authorization;
do not continue into W6/W7 from this closeout.
