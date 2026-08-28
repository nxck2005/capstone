# W7-A pre-science test-hardening repair — 2026-08-27

## Boundary

This is a narrow additive pre-science repair. It changes no G-4 protocol,
λ grid, seed, SNR, floor, ratio, profile selection, GPU selection, model
architecture, loss, optimizer recipe, validation metric, or adjudication rule.
No W7 scientific optimizer step, candidate result, G-4 adjudication, W8 run, or
model-facing test access occurred.

Starting carrier and local/remote parity were independently checked at
`88477925af2f6ec28c0a348d33d5057500e53f51`.

## Tests-first evidence and exposed defects

The new trainer, validation, and campaign tests were written against the
existing W7 production source before either production fix. After fixture-only
collection corrections, this exact focused command:

```text
.venv/bin/python -m pytest -q \
  tests/test_w7_trainer_hardening.py \
  tests/test_w7_validation_hardening.py \
  tests/test_w7_campaign_hardening.py
```

returned 25 passed and three failures. The three failures represented two
production defects:

1. `test_fresh_instance_resume_is_exact_and_preserves_latest_predecessor` and
   `test_incomplete_candidate_resumes_exact_latest_checkpoint` showed that
   `_restore_payload()` restored the loaded checkpoint's predecessor as the
   predecessor for the *next* publication. A resumed epoch therefore linked to
   the grandparent (or `null`) rather than the exact latest checkpoint. Model,
   optimizer and scheduler restoration were exact; the defect was campaign
   checkpoint-lineage/control only.
2. `test_w7_shared_decoder_nonfinite_is_optimizer_wide_skip_and_backoff`
   showed that `apply_optimizer_update()` correctly skipped/backed off and did
   not advance parameters or global step, but the W7 epoch record retained its
   pre-unscale optimizer-wide finiteness snapshot. A non-finite value introduced
   during GradScaler unscale in shared decoder ingress was therefore reported as
   finite in the W7 compact epoch record. The defect was accounting metadata
   only; optimizer behavior and training math were already correct.

## Minimal repair

Only `src/training/w7_g4.py` changed in the production path:

- resumed state now records the exact authenticated loaded checkpoint ID as the
  predecessor for the next checkpoint;
- compact optimizer-wide finiteness now incorporates the authoritative
  post-unscale classification returned by `apply_optimizer_update()`.

The historical W5 trainer/regressions and all prior W7-A artifacts remain
byte-identical. The repair changes campaign control lineage and compact
accounting only. It does not alter forward/backward math, loss scaling,
optimizer stepping, sample accumulation, validation results, runtime shape,
VRAM policy, or profile batch selection. Consequently the successful real-data
Pascal profile remains applicable and was not rerun.
