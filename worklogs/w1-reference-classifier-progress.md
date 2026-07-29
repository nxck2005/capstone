# W1 Reference Classifier Progress

- Starting HEAD: `43be6eee7910ecb7c50a941e8fabc7589cbb19f1`
- Corrective-audit HEAD: `f92b8230645a819c9cfbcd51b68390dc7f927340`
- Last updated: 2026-07-29
- Current status: corrected remainder staged; not committed.
- Full 100-epoch Imagenette training: not run.
- G-1: not executed or adjudicated.

## Final pre-G-1 audit correction

The prior implementation audit initially failed. Four findings were corrected in the first focused
continuation; a follow-up audit then found two remaining production checkpoint gaps. Both are now
closed, so all six finding categories below are corrected without reopening the AM-78 recipe:

1. **HTTP Content-Range validation** — `src/data/provenance.py` now rejects resumed `206` responses unless start, inclusive end, total, inclusive range length, optional Content-Length, streamed byte count and final partial size all exactly match the requested remainder. The audit case (`offset=4`, total `10`, `Content-Range: bytes 4-8/10`, six bytes) fails before opening the partial for append, so no final archive can be created and the original useful partial remains.
2. **Smoke lineage cannot become G-1 eligible** — `src/training/reference_classifier.py` checkpoint metadata now records `execution_mode`, `smoke_steps`, `smoke_val_batches`, `full_run_requested` and `lineage_g1_eligible` alongside completion/eligibility. Execution lineage is established on the trainer before work and serialized only from that state; `save_checkpoint()` accepts no lineage overrides. Any direct `max_steps` or `max_batches` use irreversibly downgrades the trainer to smoke-only, including when only one bound is present, and later full-run promotion is rejected. Smoke checkpoints are permanently smoke/incomplete/ineligible; full resumes accept only full lineage; smoke artifacts remain confined to the ignored smoke root.
3. **Transactional checkpoint resume and recipe validation** — resume validates the exact top-level schema, type/value/provenance/config/history/counter/lineage contracts, model keys/shapes/dtypes, and detached temporary model, optimizer and epoch-scheduler states before mutating live trainer state. Every SGD param group must match the config-derived momentum, weight decay, Nesterov and dampening values plus the completed epoch's configured learning rate. Rejected checkpoints leave model, optimizer, scheduler, counters, histories and lineage unchanged.
4. **Standalone deterministic backend** — `tools/train_reference_classifier.py` invokes the existing `env.set_deterministic_backend()` before configuration-derived construction, data views or artifact creation. The helper applies and read-back verifies the two configured SR-12 mappings (`cudnn.deterministic=true`, `cudnn.benchmark=false`). The current spec deliberately configures no separate `torch.use_deterministic_algorithms` policy.
5. **Recovery patch removal** — `worklogs/w1-reference-classifier-checkpoint.patch` was removed from the staged repository state with `git rm -f` after the requested non-forced form refused its already-staged changes.
6. **Progress record correction** — this file replaces the contradicted prior completion claim with the corrective audit record and evidence below.

## Corrected files and regression coverage

- Corrected: `src/data/provenance.py`, `src/training/reference_classifier.py`, `tools/train_reference_classifier.py`.
- Tests extended: `tests/test_provenance.py`, `tests/test_classifier_training.py`, `tests/test_classifier_checkpoint.py`, `tests/test_classifier_cli.py`.
- New coverage includes wrong range end, inconsistent inclusive range, reverse range, valid exact range, Content-Length mismatch, retained partial/no-final-archive behavior; direct lineage API contradictions; direct bounded training and validation; irreversible smoke-only state; checkpoint metadata derived only from trainer state; smoke/full resume compatibility and rejection; full-lineage resume; smoke never complete/eligible; configured momentum, weight decay, Nesterov, dampening and epoch learning-rate rejection; CLI argument rejection before construction; deterministic backend ordering; and complete live-state invariance for missing/unexpected/reshaped model state, invalid optimizer/scheduler or optimizer recipe, malformed history, differing resolved config, invalid counters and smoke/full mismatch.

## Verification evidence

- Final focused checkpoint/training/CLI suite: `35 passed`.
- Final complete suite: `223 passed`.
- Repository checks passed: spec views, documentation consistency, literal lint, packetisation, archive provenance, manifest materialization and real train/validation dataset verification. No full training, G-1, archive download or smoke execution was needed for this correction.

## Final staged paths expected after repository verification

`AGENTS.md`, `NEXT.md`, `README.md`, `src/artifacts/rng.py`, `src/config/run_config.py`, `src/data/classifier.py`, `src/data/provenance.py`, `src/models/__init__.py`, `src/models/reference_classifier.py`, `src/training/__init__.py`, `src/training/reference_classifier.py`, `tests/test_classifier_checkpoint.py`, `tests/test_classifier_cli.py`, `tests/test_classifier_config.py`, `tests/test_classifier_data.py`, `tests/test_classifier_training.py`, `tests/test_provenance.py`, `tests/test_reference_classifier.py`, `tests/test_run_config.py`, `tools/train_reference_classifier.py`, `worklogs/w1-reference-classifier-progress.md`; deleted: `worklogs/w1-reference-classifier-checkpoint.patch`.

Exact next action:

```bash
git diff --cached
```
