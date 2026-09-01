# W8-B2 partial-checkpoint incident freeze

Incident companion: `results/learned/w8/w8_b2_partial_checkpoint_incident.json`

Incident ID: `w8b2incident-feb598220d1a32143944e1d7a343fff00de43387e255d92c033289e4afcde8c2`

This is an immutable custody and classification record for the failed first
W8 scientific campaign. It does not make the partial payload a result and does
not authorize W8 training, resume, validation, G10, ER2, PAPR, ER9, or test.

## Classification

- IMPLEMENTATION DEFECT
- SCIENTIFIC-LAUNCH BLOCKING
- PARTIAL SCIENTIFIC EXECUTION
- ZERO ACCEPTED W8 RESULT COVERAGE

The original source was `c5a8b70563b1a9e4056c42bca785414924c11fa2`. At that
commit, `src/training/w8_protocol.py:47` defines
`W8_CHECKPOINT_SIDECAR_ROLE = "W8_FINAL_TRAINING_CHECKPOINT_SIDECAR"`.
The scientific branch of `src/training/w8_final.py:947` and its scientific
sidecar validator at line 1193 reference that name, but the
`from training.w8_protocol import (...)` block at lines 47–74 does not import
it. The non-scientific branch uses the locally defined
`W8_SMOKE_SIDECAR_ROLE`, which is why W8-A smoke did not expose the defect.
The root cause is therefore an implementation defect that blocks scientific
launch; it is not a protocol defect or a model-training defect.

The exact attempt-2 traceback is preserved outside the campaign root at
`/home/nick/w8-b2-launch-attempt-2-incident.json` (7,119 bytes,
SHA-256 `e94862fa24644bfdac7d8ffdc15371cac476a93255ec20664ce290766309485e`).
The failure occurred in `trainer.save_checkpoint(record)` after the epoch
record and checkpoint payload were published and before the scientific sidecar
and `latest.json` were published.

## Failed campaign custody

Original W8-A authorization: `w8auth-e36e5882f06e4af46d2b0dbde5f198064f0499def869e60859377bfc092d8727`

Original source manifest: `w8source-66b6558938bef496e16f72c63b854d92fa10d5aae5a805ae3b4abd0eb37a997d`
(file SHA-256 `db6aab4ef8d6cec146da2d6ee989ab0ddb60ca37f0ad3729af61c5da9ade8755`)

Original launch authorization:
`w8blaunch-99ae8a71508aa35bfbb763142faaf1364d9ed976b212b7311f37b128d53d73c4`

Campaign ID: `w8-final-pascal-20260831`

Campaign manifest ID:
`w8campaignmanifest-eed9f6123527a9d9072159772d5fe8c79e8e05dc1888c31b0e339f7a017e844c`

Campaign manifest SHA-256:
`09a0b9140d79a128464e24b30094179f07f66b2b0f1fd12c401bd1f978c46818`

The read-only Confessor inventory has six paths (three directories and three
files). Its custody digest is
`da6f18be2545d59298c64cc890aa4842b436de1e7e6fc44dc6e22dd6f64cf4aa`, over
sorted LF-terminated lines of `TYPE`, relative path, byte length, and file
SHA-256. The files are:

| Relative path | Bytes | SHA-256 |
|---|---:|---|
| `campaign_manifest.json` | 3,714 | `09a0b9140d79a128464e24b30094179f07f66b2b0f1fd12c401bd1f978c46818` |
| `run-01-r_1_6-train0-channel0/checkpoints/epoch-0000.pt` | 18,942,699 | `ff89322a795e437994ff5eaaf5c7157fd7e751aed8bafb2f42cee950d371b55c` |
| `run-01-r_1_6-train0-channel0/epochs/epoch-0000.json` | 206,345 | `becb208afc383e6e55569d17d9f701ad949e308ac19a3c329c46ab23776977b6` |

The epoch record was inspected only for structural fields. It records epoch 0,
8,469 samples, 265 microbatches, 265 optimizer-step opportunities, 259
applied steps, 6 GradScaler skips, global optimizer step 259, run
`w8-r_1_6-train0-channel0`, ratio `r_1_6`, train/channel seeds 0/0, and the
original source identity.

Present: campaign manifest, epoch-0 record, checkpoint payload.

Absent: scientific sidecar, latest pointer, validation summary, run
completion, selected-checkpoint result, run-02-or-later state, and campaign
completion.

The attempt-2 stdout log is 1,626 bytes with SHA-256
`ca524a4b23d7f1a2a45ebf278a9d26b893c1c8d5f741cf542bdc94e929b09273`.
The separate attempt-1 stdout log is zero bytes with SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`, and its
operations incident is 2,445 bytes with SHA-256
`8c5fb9b9f45cb43a4fcd94983ab41b12b23e10280f9e5f24a32f13cd467ff34b`.
Attempt 1 was an operational launch failure with zero scientific execution.

## Accepted coverage boundary

The historical failed attempt executed 259 scientific optimizer steps out of
265 opportunities and recorded 6 GradScaler skips. Because the required
scientific sidecar/latest checkpoint transaction did not complete, it is an
`INCOMPLETE_SCIENTIFIC_EPOCH`, not an authenticated completed epoch.

| Namespace | Value |
|---|---:|
| Historical scientific optimizer steps executed | 259 |
| Accepted W8 optimizer-step coverage | 0 |
| Authenticated completed epoch cycles | 0 |
| Accepted W8 checkpoints | 0 |
| Validation measurements | 0 |
| Completed W8 runs | 0 |

The partial checkpoint is permanently ineligible for resume, W8 result, G10,
and test. Its sidecar must not be manufactured, its payload must not be
loaded for continuation or quality inspection, and the old campaign root must
not be reused. The root remains at
`/home/nick/w8-final-pascal-20260831` on Confessor, untouched and preserved.

G10 = 0; ER2 = 0; PAPR constrained = 0; ER9 = 0; test model-facing access = 0;
learned test inference = 0.

## Successor source repair

The repaired scientific source is the separately pushed branch
`repair/w8-b2-sidecar-source`, descended from c5a8, at commit
`d52d85dd60bac0c816a7ba249e4453045723277b`. Its source diff classification is:

- result-affecting training logic: NONE;
- checkpoint implementation: import the canonical
  `W8_CHECKPOINT_SIDECAR_ROLE` from `training.w8_protocol`;
- campaign lineage/operations: bind a distinct successor ID and root,
  heartbeat, and stdout path;
- tests: direct scientific epoch-record → payload → sidecar → latest coverage,
  fresh same-run authenticated resume, invalid/missing sidecar rejection,
  foreign W8/W7 rejection, and separate smoke-role coverage;
- other scientific-source drift: NONE.

The six scientific cells and all result-affecting training controls are
unchanged. The successor identity is frozen in source as:

- campaign ID: `w8-final-pascal-20260901-r1`;
- root: `/home/nick/w8-final-pascal-20260901-r1`;
- heartbeat: `/home/nick/w8-final-pascal-20260901-r1.heartbeat.json`;
- stdout: `/home/nick/w8-final-pascal-20260901-r1.stdout.log`;
- eventual tmux session: `w8-final-r1`.

The successor root is absent, no successor launch authorization exists, and no
scientific training was performed by the repair. A successor must use fresh
deterministic initialization; the incomplete payload is not a resume source.

## Verification and next gate

Focused W8 trainer, protocol, validation, and campaign tests passed. The
independent W7-G4 verifier, spec generation, documentation/literal checks, and
`git diff --check` passed. The clean-checkout `ci-cpu` quality gate passed after
the source commit. All smoke/test artifacts are synthetic or non-scientific
and are ineligible for W8 result, G10, and test claims.

The current-main repair adds this record and its machine-readable companion,
repairs only the stale synthetic CI cursor fixture in
`tests/test_doc_consistency.py`, and does not merge the scientific source
branch or alter original W8 authority bytes. The next safe action is owner
audit of the exact pushed source/CI and incident boundary, followed only by a
new source manifest and separately authorized successor execution.
