# ER-9 Stage-1 sidecar-return incident

This is an immutable custody and classification record for the second ER-9
Stage-1 attempt. It preserves the authenticated checkpoint transaction and
does not make the attempt a completed ER-9 candidate or authorize reuse.

## Classification

- IMPLEMENTATION DEFECT
- PARTIAL SCIENTIFIC EXECUTION
- ZERO ACCEPTED ER-9 RESULT COVERAGE

The attempt used source/authority successor v2 at `d89457bf06052bceeb18f51219e8c5c6342da0ed`.
The first repaired D64/b2 epoch completed its epoch record, checkpoint,
sidecar, and `latest.json` transaction, then failed while constructing the
in-memory return record because it read `sidecar["sidecar_path"]`, a field not
present in the sidecar schema. The failure was a `KeyError` after all files in
the checkpoint transaction had been durably published.

## Failed-attempt custody

The preserved runtime is:

`checkpoints/er9_successor/stage1/D64_b2/`

Its emitted files are:

| Relative path | Bytes | SHA-256 |
|---|---:|---|
| `checkpoints/epoch-0000.pt` | 9,745,835 | `a790c22170ed7a0ebe837f087b0fa9fcc220e3f3cce8929e19c34e46882f8840` |
| `checkpoints/epoch-0000.sidecar.json` | 753 | `4cc22cb188a7aebae9e3bac127a54ab8d483971252e53eb73eefa9bd4f3ef9ea` |
| `epochs/epoch-0000.json` | 1,265 | `2c7b877fdda66a694cf8592b473d1e9a4121633d368022327d2b04af5eb67093` |
| `latest.json` | 272 | `0937e3a30d8684199f75ff1b89d7d345540a9ca05c77dd374af7a35d7a6e35af` |

The record reports epoch 0, 8,469 samples, 265 microbatches, 265
optimizer-step opportunities, 261 applied optimizer steps, four GradScaler
skips, global optimizer step 261, and training-path validation `140/1000` at
7 dB. No selected checkpoint or run completion exists.

The attempt is an `INCOMPLETE_SCIENTIFIC_CANDIDATE`. Its files are preserved
for custody only and are not promoted, evaluated through the real digital
chain, resumed, or counted as an accepted Stage-1 run. The successor source
uses the direct canonical relative sidecar path in the return record and a
fresh runtime namespace `checkpoints/er9_successor_v2/`.

G-10 remains terminal and unchanged; randomized ER-2, G-11, and W10 remain
zero; and the test split remains sealed.
