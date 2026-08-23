# G8_E clean-checkout runtime incident — 2026-08-23

## Classification

`TEST_HARNESS_PRODUCTION_ESCAPE_SEPARATE_RUNTIME`

The local full-quality-gate run in the clean worktree
`/home/nick/projects/capstone-ci-clean` unintentionally started the historical
corrected-v3 production runner. This was not an owner-authorized scientific
campaign, is not a continuation of either preserved local E2 or the completed
worker-successor E2, contributes zero successor coverage, and is permanently
merge-ineligible. The runtime is preserved in place as incident evidence.

## Trigger and root cause

The process command was:

```text
/home/nick/projects/capstone/.venv/bin/python tools/run_g8_e_corrected_v3.py --start --campaign-id g8e-v3-c20d9c4f4638687ad9e4e3e69bf7b9dbdf509a62c2c3a4d95dbbe6771ced57b5
```

It came from
`tests/test_g8_e_corrective_v3.py::test_production_runner_refuses_old_v2_and_missing_authorization_before_payload`.
The test expected the writer machine's ignored preserved runtime to exist and
make `--start` refuse. A clean checkout did not contain that ignored runtime,
while the tracked authorization and matching campaign ID were present, so the
test command crossed into production execution instead of refusal.

The process was stopped after BTOP showed `01:13:41` elapsed, `86.0 MiB` read
and `4.68 GiB` written. A subsequent process census returned no
`run_g8_e_corrected_v3.py`, `pytest`, or `run_quality_gate.py` process.

## Runtime custody

Preserved local predecessor runtime:

- path: `/home/nick/projects/capstone/results/baseline/g8_e/e1_corrected_v3/runtime/`
- filesystem bytes: `4,655,012,710`
- filesystem inodes including directories: `207,068`
- durable state file SHA-256: `2c1e4f691efc0ae0f1184e0268cb8d0cd093dbb2edd01fbdde796b9c6cb10f89`
- authenticated prefix: `47409/288000`
- status: `RUNNING`, with the historical in-progress claim at ordinal `47409`
- state SHA-256: `0690b6433289339b46a5280e301a87b34e38d4e5d6068ce4c2026c3ffe05623f`
- rolling prefix digest: `ebdea4a44f15950908438fc2a789620578e6cad5fd7dd7d7df56b1cfd9f428ad`
- newest visible runtime file mtime: `2026-08-21 23:17:28.673961167 +0530`

Unauthorized clean-worktree runtime:

- path: `/home/nick/projects/capstone-ci-clean/results/baseline/g8_e/e1_corrected_v3/runtime/`
- filesystem bytes: `4,219,718,637`
- filesystem inodes including directories: `187,540`
- durable state file SHA-256: `a2b31e970c9d526e9432420926de070fbeb78edc0829c8c6573c2e622a4cd09b`
- authenticated prefix: `42704/288000`
- status: `READY`, with no durable in-progress claim
- state SHA-256: `b011ab49578b3d9392cdae6f0c4c46b53fd79536f3050e074f3249e432bb5923`
- rolling prefix digest: `1996c5b7a00457e28f3ae4d0466f961d275efff854a70035c3f399afcac03d95`
- visible runtime mtime interval: `2026-08-23 02:41:06.776668654 +0530` through
  `2026-08-23 03:54:39.360205588 +0530`
- interrupted temporary state:
  `.campaign_state.json.z49sma17.tmp`, file SHA-256
  `ab9027ee2737d90c80850a5e775b71b502a570777699b1f47944104839f1f045`,
  status `RUNNING`, in-progress ordinal `42704`

Both authenticated states retain `training = 0` and `test_access = 0`. The
scratch state records `37704` validation decodes/inferences plus `5000` codec
infeasibilities. These are unauthorized validation measurements and must not be
reported, selected from, merged, resumed, or treated as scientific evidence.

## Separation and integrity findings

- The trees are on the same filesystem but share zero `(device, inode)` pairs
  across their visible files. Their root directories, state files and lock
  files also have distinct inodes.
- No scratch-only path exists in the compared scientific subdirectories after
  excluding coordination state and lock files.
- Every shared record, codec metadata object, observation, reconstruction and
  checkpoint is byte-identical. All `37,704` shared raw `.j2kcache` files are
  byte-different; no cause is inferred and none may be substituted between
  runtimes.
- The preserved local predecessor tree's newest visible mtime predates the accidental process by
  more than one day. Its exact prefix independently re-authenticates at
  `47409/288000` with the historical state and rolling digests above.
- The scratch tree independently re-authenticates at `42704/288000`; this
  authentication records what exists but does not authorize or legitimize it.
- The tracked worker-successor E2/E3/E4 artifacts, pass-one record and E6 freeze
  remain unchanged. `tools/verify_g8_e_complete.py` still returns the terminal
  G8_E GREEN verdict with zero training, pass two and test access.

## JPEG 2000 cache payload addendum

A 2026-08-23 read-only audit compared all `37,704` cache identities shared by
the preserved local predecessor and the quarantined scratch runtime. All
`37,704` outer `.j2kcache` ZIP byte streams differ. Inside them, however, all
`37,704` raw and parsed `metadata.json` payloads are byte-identical, every
metadata identity hashes to its cache filename, and every declared decoded-image
identity agrees. The `34,704` feasible objects have byte-identical embedded
`codestream.j2k` payloads whose SHA-256 values match the authoritative hashes in
both metadata records; the other `3,000` objects are identically infeasible and
contain metadata only. There are zero inner-payload, identity, declared-hash or
ZIP-structure mismatches.

The outer difference is therefore non-scientific ZIP-container nondeterminism:
the cache writer stores identical members with per-write ZIP timestamps. No
cache object was copied, normalized, deleted or substituted during this audit.

## Containment and repair

The test no longer invokes the current production campaign against the default
runtime. Both old- and current-campaign refusal paths now use explicit
`tmp_path` runtime and missing-authorization paths. The current path runs
in-process with transaction-construction and payload-loading barriers that fail
the test immediately if reached. It also asserts that neither isolated runtime
is created and that the preserved local predecessor state remains absent or
byte-unchanged.

Post-repair verification completed before this record:

- corrected-v3 test module: `41 passed`
- E5 pass-one, E7 closeout and CI allowlist set: `15 passed`
- read-only preserved local predecessor prefix authentication: PASS at `47409/288000`
- read-only scratch prefix authentication: PASS at `42704/288000`
- terminal G8_E verifier: PASS, G8_E GREEN

## Permanent handling rule

Do not resume, reconcile, merge, copy into the canonical tree, delete, or
normalize the scratch runtime. Do not use the contaminated clean worktree for
scientific execution. Any later disposal or relocation requires an explicit
owner custody decision that preserves this record and authenticates the bytes
being moved or removed.
