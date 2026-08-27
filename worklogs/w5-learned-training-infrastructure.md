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

Across attempts 1–3 the historical artifacts record six non-scientific
optimizer steps (0 + 2 + 4). The attempt-3 count is retained as historically
recorded rather than strengthened after the accounting defect described below.
Scientific learned runs, W7 pilots, W8 runs, validation selection, learned test
inference and test access are all zero.

## Verification

- Targeted W5/model/loss/channel/power/RNG/config/test-isolation/G8 provenance:
  167 passed.
- Evidence gate: PASS (11 focused tests).
- Full-local: PASS (2,536 tests).
- ci-cpu: PASS.
- Pre-smoke exact source CI: run `33011201226`, job `98317625691`.
- Pre-smoke manifest-carrier CI: run `33012842982`, job `98323311218`.

Historical completion is `results/learned/w5/w5_completion.json`,
`w5completion-680b2688dc761a30a7a68aee91c021fe057bbb726b44b614bdffd19712c5fc70`.

## Additive GradScaler accounting repair

A hostile audit found an **IMPLEMENTATION DEFECT**: the trainer inferred whether
`GradScaler.step()` applied an update from only encoder, reconstruction-head and
task-head gradients, while GradScaler examines every optimizer-owned gradient.
Shared decoder ingress/residual parameters were therefore outside the accounting
classification. AM-91 already required standard skip/backoff semantics, so no
scientific recipe changed and no new amendment was needed.

Source epoch `w5source-af58f018…` binds repair commit `755b1e138c87…` plus the
regressions. The tests inject Inf and NaN into shared
`decoder.ingress.0.weight` while all three named regions remain finite, proving
the update is skipped, scale backs off, global step does not increment and no
parameter update occurs. The finite complement proves exactly one increment;
checkpoint mutation coverage rejects a trace that calls a skipped update an
applied global step.

Successor attempt 4's immutable raw execution output is `w5smoke-2fbae18d…`,
SHA-256 `5df4e4b5e579f9b329377389ce36f791d33da187e1dba8cc9211920b1f7717c3`.
Its accounting proof was initially emitted as a new top-level field, outside the
already-frozen schema-v1 top-level set. No execution or measured value was
repeated or changed: an additive canonical projection moved that object under
the schema-v1 unrestricted `training` object and bound the raw path/ID/SHA. The
terminal attempt-4 artifact is `w5smoke-3d77765c…`, SHA-256
`978b4fd8e040deb4595608a74900ae37e2017f1aec0c4294a1bc36636f968ef8`.
The launcher and verifier were corrected additively for that existing schema.
Attempt 4 ran only from CI-green `local_4060_cu130` execution commit
`6554731c420a…`, verified four actually applied W5-only steps and exact fresh-
process resume, and rechecked finite/nonzero gradients at `r_1_6` and
`r_1_24`. It recorded no accuracy or selection. Attempts 1–3 and their
completion remain byte-identical.
The additive current closeout is
`results/learned/w5/w5_gradscaler_accounting_repair_completion.json`
(`w5repaircompletion-8b2fa917…`).

Future hardening remains deliberately out of this repair: a sole-writer OS lock
is mandatory before the first separately authorized scientific W7 optimizer
step; this launcher remains W5-only/ineligible; and terminal verification-record
semantic rederivation can be strengthened separately. No smoke accuracy was
recorded or used. Next action is separate W6 authorization; do not continue into
W6/W7 from this closeout.
