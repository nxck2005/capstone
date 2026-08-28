# W7-B1 detached-launch boundary defect — 2026-08-28

## Tests-first evidence

Starting local, `origin/main`, and remote `main` were all
`8670534d5b5fe8911fdf8cf1e4dfbe319ccccc43`, with a clean worktree. Before any
production edit, `tests/test_w7_b1_launch_boundary.py` exercised the real
`run_w7_campaign.run()` path. It did not replace or monkeypatch the source
manifest verifier.

The pre-fix launcher imported `verify` from `gen_w7_source_manifest`, whose
schema is the historical version 1 schema. The focused command passed because
it asserted the two required fail-closed outcomes:

```text
.venv/bin/python -m pytest -q tests/test_w7_b1_launch_boundary.py
2 passed
```

Direct invocation recorded the underlying failures:

```text
v2 ValueError W7 source manifest schema/role differs
v1 ValueError W7 current source byte drift: src/training/w7_g4.py
```

The first line proves the accepted additive
`results/learned/w7/w7_source_manifest_v2.json` could not be consumed by the
real detached launcher. The second proves that resigning the historical v1
manifest does not make it a current scientific authority after the W7-A
trainer hardening changed `src/training/w7_g4.py`. No candidate, optimizer
step, validation evaluation, G-4 adjudication, or test access occurred.

This record is additive evidence. Historical W7-A manifests and completion
artifacts are preserved byte-identically.
