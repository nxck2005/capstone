# G8_F / F0 authorization — 2026-08-24

## Verdict

**F0 GREEN — G8_F EXECUTION CONTRACT/AUTHORIZATION FROZEN; F1 NOT STARTED;
SEPARATE OWNER/OPERATOR LAUNCH REQUIRED.**

The owner independently accepted AM-88 and authorized F0 only. The canonical
handoff is `results/baseline/g8_f/f0_execution_authorization.json`:

- ID: `g8ff0auth-92189865202e4b6cb400a0a86cee101b8ad8a7bdf5ea9d5a78ae96ab49a365b4`
- file SHA-256: `17a88e36201d42b3b2ace190b0b5b5f3b34aeb3afb48f8a84e26db159b86de94`
- authorized F1 source commit: `c437ff80eebd464ee7b256f2e69240a7d2f514a8`
- profile/device: `local_4060_cu130` / `cuda:0`
- lock SHA-256: `ee68e2323a50b81967558e76da69894176a26e8d0d2dce444b5eb8c5cc7eb5cd`

F0 does not authorize F1. `tools/run_g8_f_f1.py --start` additionally requires
a later owner-issued F1 launch artifact bound to the exact F0 ID/file SHA and
source commit. The F0 preflight cannot accept that artifact or enter the
production loop.

## Re-authenticated frozen protocol

| Item | Result |
|---|---|
| G8_C | GREEN/CLOSED; verification only, no recomputation |
| G8_D | GREEN/CLOSED; D0/D7 verified, no recomputation |
| G8_E | GREEN through E7; pass one exactly once; no rerun |
| G1 clean classifier | frozen checkpoint re-authenticated; no retraining |
| AM-87 | ID `g8fcorpusplan-6320ea3e5299a2175a730a2cb8c2d835e756bd11e7424f4b1221948f6f148148`, SHA `733daa01614781c62f2af6a7e992bf29f9103a1b2afface5d77b610bb657858c`, 120 qualities, 8,469 train IDs |
| AM-88 | ID `g8fsamplerplan-d6d64ead5295b93c2a73aefd5f0719dd438bd6c0425286a33a31f1fba3ff64d6`, SHA `eca85a9891bcf2054e132e5fc277430d2c85962a78f8438c9da0604d98447e23`, six/image, 50,814 pairs |
| ordered/set pairs | `c7c29729c2e0a94646c9e1ef16f45fe240badfb4a1baf82f27883681bac65229` / `255eab85aca45e4ca910f040b0752ea773630c642c8697579df056104dcb594e` |
| train manifest | `224309422f15bf89460559381aea4b00c4779c52d3652f7f679a213369f3f889`; validation/test assignment count 0 |
| codec | `g8dcodec-39f14b7eaba4f727c70759eb1c5250e8e13f7d5e871c0831aa6b602aef706858`, config `2daf597fd914f56eb9e59df7bc20a88b02816522b3b0b4fd3f2db14d7451a0fa`, Glymur 0.14.3 / OpenJPEG 2.5.4 |

The live profile authenticated Python 3.14.6, Torch 2.13.0+cu130/CUDA 13.0,
RTX 4060 UUID `GPU-607a5795-c53b-eab2-8c04-71164b173a32`, driver 592.82, and
OpenJPEG 2.5.4. At freeze, the destination had 913,266,221,056 available bytes
against the frozen 7,323,313,680-byte planning reserve (planning values only).

## F1 readiness and synthetic scope

The source commit adds an exact AM-88 assignment loader, immutable request/result
records, content-authenticated codestream and reconstruction objects, exact-prefix
resume, train-only admission, and an F1 CLI with a separate-launch boundary.
Typed image-level codec infeasibility records the assigned omission without
replacement or resampling. Arbitrary exceptions raise HOLD. The runner has no
classifier, optimizer, pass-two, fallback, ratio, learned-training, or test
entry point.

Focused smoke used only a generated in-memory PNG and injected synthetic
backends. It proved deterministic reuse, exact-prefix resume, typed
infeasibility/no-resampling, unexpected-failure HOLD, and validation/test
refusal. It was explicitly non-scientific and invoked no real JPEG2000 codec.

## Historical verifier compatibility

Updating the current `instructions/G8_F.txt` cursor changed only instructions
after all G8_C–G8_E measurements and pass one were closed. The historical G8_A
manifest remains byte-identical. Its existing fail-closed source verifier was
extended only to accept the exact new instruction SHA-256
`f952fb37573a055596be54c11544c894ce5818c77b5b12ad4d514ca3e0d776be` and the
corresponding exact verifier-source projection. No measurement path, plan,
selection, result, or historical artifact byte changed.

## Adversarial closeout

A–D: AM-87 remains 120; AM-88 remains six assignments for each of 8,469 IDs,
50,814 total, with unchanged ordered/set digests. E: validation/test did not
affect assignment. F–L: no production artifact, classifier inference,
optimizer step, pass two, test access, pass-one rerun, or G8_C/D/E science rerun
occurred. M: F0 binds the exact source/runtime closure. N: the runner admits only
AM-88 and rejects the 1,016,280 Cartesian multiplicity. O: typed infeasibility
cannot resample. P: arbitrary failure is HOLD. Q–R: F0 does not start F1; a
separate deliberate owner/operator launch is required.
