# ER-9 Stage-1 checkpoint publication incident

This is an immutable custody and classification record for the first ER-9
Stage-1 attempt. It preserves the failed runtime and does not make its partial
payload an ER-9 result or authorize resume from it.

## Classification

- IMPLEMENTATION DEFECT
- PARTIAL SCIENTIFIC EXECUTION
- ZERO ACCEPTED ER-9 RESULT COVERAGE

The source-freeze commit was `11e4914a3cf4bbe997a238267ec7581d77b75202`,
bound to source manifest
`ersource-40b72c1965720969213d96f17770908f978c69d4e2edfb97c203a10c46d95358`
and Stage-1 authorization
`er9stage1auth-26d154351ba2e54709b092f2d0ff49708905fbe819f26b4debfedc37f27d54c2`.

The defect was in `src/training/er9.py`: `save_checkpoint()` passed the
sidecar mapping directly to `_publish_immutable()`, whose contract requires
canonical bytes. The failure was a `TypeError` while publishing the sidecar,
after the epoch record and checkpoint payload had been published and before
the sidecar and `latest.json` transaction completed.

## Failed-attempt custody

The preserved runtime is:

`checkpoints/er9/stage1/D64_b2/`

Its emitted files are:

| Relative path | Bytes | SHA-256 |
|---|---:|---|
| `checkpoints/epoch-0000.pt` | 9,745,835 | `e879471a7d9c012afd524f1b8da9fc7dba1e925896dfae079d25f40063370119` |
| `epochs/epoch-0000.json` | 1,265 | `d160c371c4601ea3397e0f25ba05f99cb6a950280b4b1bde3e06436c5b8d630b` |

The record reports epoch 0, 8,469 samples, 265 microbatches, 265
optimizer-step opportunities, 261 applied optimizer steps, four GradScaler
skips, global optimizer step 261, and the fixed 7 dB training-path validation
summary `140/1000`. No scientific sidecar, `latest.json`, selected
checkpoint, or run completion exists.

The attempt is classified as an `INCOMPLETE_SCIENTIFIC_EPOCH`. Its checkpoint
and epoch record are permanently ineligible for resume, selection, validation
evidence, G-11, G-10, and test. A sidecar must not be manufactured and the
runtime must not be reused or overwritten.

## Successor boundary

The repair is additive: the successor source uses a canonical JSON publisher,
adds a regression test, binds successor source-manifest and authorization
records, and writes fresh deterministic runs under
`checkpoints/er9_successor/`. No result-affecting training rule changed and no
failed payload is loaded. The original Stage-1 authorization remains immutable
history; the successor authorization is a new staged authority. G-10 remains
terminal and unchanged, randomized ER-2 remains zero, G-11 remains zero, and
the test split remains sealed.
