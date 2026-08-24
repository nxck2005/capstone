# G8_F / F0-v2 pre-F1 resume-authentication repair — 2026-08-24

## Verdict

**F0-V2 GREEN — RESUME/OBJECT AUTHENTICATION REPAIRED AND EXECUTION
AUTHORIZATION REFROZEN; F1 REMAINS ZERO AND REQUIRES A SEPARATE OWNER LAUNCH.**

This is an implementation/provenance repair before the first F1 artifact, not a
scientific protocol change. No AM entry was required. AM-87 and AM-88 remain
byte-identical.

## Historical F0-v1 preservation and supersession

F0-v1 remains byte-identical at
`results/baseline/g8_f/f0_execution_authorization.json`:

- ID: `g8ff0auth-92189865202e4b6cb400a0a86cee101b8ad8a7bdf5ea9d5a78ae96ab49a365b4`
- file SHA-256: `17a88e36201d42b3b2ace190b0b5b5f3b34aeb3afb48f8a84e26db159b86de94`
- intended source: `c437ff80eebd464ee7b256f2e69240a7d2f514a8`
- prior production coverage: zero

Its scientific protocol was correct, but its resume prefix validator checked
only ordinal names and assignment IDs, while direct result reuse omitted
referenced-object authentication. It is therefore explicitly
`superseded_before_F1` for
`incomplete_resume_and_referenced_object_authentication`; its bytes and Git
history were not rewritten.

## Active F0-v2

The active handoff is
`results/baseline/g8_f/f0_v2_execution_authorization.json`:

- ID: `g8ff0v2auth-dbcac1f4dcf76238a4222629e590372004f5dad3e4fb1316e28b6fd0b93c6f31`
- file SHA-256: `b14691ca26b6086d9b8e08b563027047cdba114b438311208fe6d413f5c29ce9`
- repaired F1 source commit: `b1ee63d95de4fe86b9758ae90dbbb7b428a63635`
- profile/device: `local_4060_cu130` / `cuda:0`
- lock SHA-256: `ee68e2323a50b81967558e76da69894176a26e8d0d2dce444b5eb8c5cc7eb5cd`
- codec: `g8dcodec-39f14b7eaba4f727c70759eb1c5250e8e13f7d5e871c0831aa6b602aef706858`, configuration hash `2daf597fd914f56eb9e59df7bc20a88b02816522b3b0b4fd3f2db14d7451a0fa`, Glymur 0.14.3 / OpenJPEG 2.5.4

F0-v2 authenticates F0-v1 as historical evidence and records its exact ID,
file SHA, source commit, reason, zero coverage, and superseded-before-F1 state.
It does not issue an F1 launch authorization and cannot start F1.

## Repair

`src/baseline/g8_f_materializer.py` now uses one shared fail-closed path for
both restart-prefix admission and direct existing-result reuse. Before an
assignment can be skipped it authenticates:

- canonical request/result JSON and content-derived IDs;
- exact AM-88 assignment body, ordinal, stable ID, class, quality, budget,
  encode axis, Imagenette/train manifest/archive identity, codec ID/hash, and
  scientific flag;
- exact request/result linkage and no replacement/resampling semantics;
- the closed outcome taxonomy;
- typed codec infeasibility with both objects null and exact omission semantics;
- materialized codestream/reconstruction schemas, canonical content-addressed
  paths, regular non-symlink files beneath the runtime root, exact byte lengths,
  and actual-byte SHA-256;
- reconstruction shape/dtype and byte-count reconciliation.

The prefix scan remains O(N) once per restart. Result holes, foreign ordinals,
foreign requests/results, corrupt records, and missing/corrupt objects HOLD.
Exactly one orphan request at the next ordinal is admitted after interruption;
it must authenticate structurally during the scan and reproduce the exact
deterministic request bytes when that assignment is retried. No corrupt result
is demoted to incomplete or silently recomputed.

The request/result schema is version 2 for the unopened production runtime. No
schema-1 F1 production record exists.

## Frozen scientific identities

AM-87 remains
`g8fcorpusplan-6320ea3e5299a2175a730a2cb8c2d835e756bd11e7424f4b1221948f6f148148`
/ file SHA
`733daa01614781c62f2af6a7e992bf29f9103a1b2afface5d77b610bb657858c`:
120 qualities and 8,469 exact training stable IDs. AM-88 remains
`g8fsamplerplan-d6d64ead5295b93c2a73aefd5f0719dd438bd6c0425286a33a31f1fba3ff64d6`
/ file SHA
`eca85a9891bcf2054e132e5fc277430d2c85962a78f8438c9da0604d98447e23`:
six qualities per image and 50,814 attempts. Ordered/set pair digests remain
`c7c29729c2e0a94646c9e1ef16f45fe240badfb4a1baf82f27883681bac65229`
/
`255eab85aca45e4ca910f040b0752ea773630c642c8697579df056104dcb594e`.
The training manifest remains
`224309422f15bf89460559381aea4b00c4779c52d3652f7f679a213369f3f889`.

## Exact repaired source closure

F0-v2 binds these paths at `b1ee63d95de4fe86b9758ae90dbbb7b428a63635`:

| Path | SHA-256 |
|---|---|
| `src/baseline/g8_f_f0.py` | `59eccb3e95087f14862d5bbe61e4f52dd5ae82a2fe87947cf183fd12e94e86f8` |
| `src/baseline/g8_f_materializer.py` | `0f1d5283f5f0920658678d139918bfbe8ab92ede0db0839a8c4f8750e4109f72` |
| `src/baseline/g8_f_sampler_plan.py` | `45ade7d2ede4942da266533eda87fe52b6b718902ce0e57323d0fd0dcdcc83df` |
| `src/baseline/g8_f_corpus_plan.py` | `51ea6e0e947d21300c76fb222c22727731487ebc8ac366f789cc5c6aafdabd06` |
| `src/baseline/j2k.py` | `90a0dc1bfdbf37fcb1cd36a539f4623d0d247c20cfc04246da00cad4cc377b7e` |
| `src/config/params.py` | `4f9c464469cb00c99c5f1b60f48dd388c695b752cf63215bd34c670defb9d7dc` |
| `src/config/execution_profiles.py` | `4cde66962dcc228d0a5231d7c3ff1d84274d65d6734d021b5cccff3988175061` |
| `src/data/adapters.py` | `97f36eb4a226bedc8579a000b048119e15915495382c759b79fe0582209bcfca` |
| `src/data/identity.py` | `e03e0033d6d0174584c2818fb4b6968d9e33c133527a8fbab1644f6a5850f249` |
| `src/data/manifests.py` | `28cb6de265b7f2cf5eff65e0da9bba3a35a592371924bb52a991b6682353bfdd` |
| `src/data/preprocessing.py` | `1aa5eafc8d02be5e79ee10b342cb8fb901e2600ba7a77224ecaf29dbcbcbacdf` |
| `src/data/provenance.py` | `24a74828fda74e337aaf865ff06974cd25e9ce11a8e014bef7d9a2cf1a961b33` |
| `src/data/registry.py` | `da790dcee14835f756302a360defa9158f63a56eded87605fe383ade6297b8ae` |
| `src/env.py` | `18f32856644b243ecafdb4dee07f17aeeb71c0ca15f3f9899539b229c6a296dc` |
| `tools/run_g8_f_f1.py` | `52cdcf47eb53a22712696cf68b51374c30cfd6bc6d96edcb3c66f7b08dd86e90` |
| `tools/verify_g8_f_f0.py` | `64d41cb831839cfe9b5e1f83d9c6cab0d72b4dadbe840d00b5905282a2ff67e7` |

## Zero boundary and non-scientific testing

At refreeze, `results/baseline/g8_f/runtime` and
`results/baseline/g8_f/f1_launch_authorization.json` were absent. Production
artifacts, real F1 JPEG2000 invocations, artifact-classifier inference,
optimizer steps, pass two, fallback, ratio adjudication, learned-system
training, test access, and prior-science reruns were all zero.

Focused tests use only an in-memory synthetic PNG and injected fake codec
backends. They are non-scientific. They cover canonical request/result identity,
body/link mutations, false scientific metadata, no-resampling, deleted/corrupt
objects, wrong object length/hash/path, symlink refusal, typed infeasibility,
prefix holes, foreign assignments, direct reuse, legal orphan-request resume,
unexpected-failure HOLD, validation/test refusal, and exclusion of AM-87's old
Cartesian multiplicity. No real JPEG2000 codec or training image was invoked.

At refreeze the destination had 913,263,181,824 available bytes against the
7,323,313,680-byte frozen planning reserve. This is planning preflight, not a
scientific measurement.

## Next action

A later owner/operator must separately issue the F1 launch artifact and run the
exact command below. It was not run during this repair:

```bash
.venv/bin/python tools/run_g8_f_f1.py --start \
  --f0-authorization results/baseline/g8_f/f0_v2_execution_authorization.json \
  --f1-launch-authorization results/baseline/g8_f/f1_launch_authorization.json \
  --runtime-root results/baseline/g8_f/runtime
```
