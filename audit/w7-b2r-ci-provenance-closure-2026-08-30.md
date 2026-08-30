# W7-B2R CI / provenance closure — 2026-08-30

This is an additive audit record for the terminal CI/provenance closure. It is
not a scientific repair.

## Published B2R carrier history

### Initial published carrier

- Carrier: `15b387bf072958d6abe974005496cbea50a8ad25`
- Parent: `a3665b854dd1e9065a8082a66680a69ce29a10c1`
- CI: GitHub Actions run `33322088488`
- Outcome: cancelled at the old 30-minute hosted timeout.

### Sibling replacement

The published main state at `15b387bf072958d6abe974005496cbea50a8ad25` was
improperly replaced non-fast-forward by its sibling
`41e0c82f505fbbdc82493f0b62554cac0953277d`, which also has parent
`a3665b854dd1e9065a8082a66680a69ce29a10c1`. This replacement is explicitly
classified as:

- **PROCESS-PROVENANCE DEFECT**
- **SCIENTIFIC IMPACT ZERO**

The published main state replacement is not concealed. No attempt is made here
to merge, reparent, or rewrite the dangling carrier back into main.

## Scientific-byte continuity

The W7-B2R scientific/reconciliation evidence bytes did **not** change between
the abandoned carrier and the accepted carrier. The identical Git blob
identities are:

| Evidence path | Git blob identity |
| --- | --- |
| `results/learned/w7/w7_b2_checkpoint_custody.json` | `225de028ff39ac22a0d7f53b85717bac1d39e996` |
| `results/learned/w7/w7_b2_common_noise_audit.json` | `7bd9399844bd656dedf00479217baf013555e870` |
| `results/learned/w7/w7_b2_completion.json` | `2ea930cd2206af38cf6a3541fdf8bf0f3884aaf5` |
| `results/learned/w7/w7_b2_reconciliation.json` | `81219aa29fcc6bceac17cb19ca93d2abeefc3e9f` |
| `results/learned/w7/w7_b2_reconciliation_index.json` | `81578bd39685945e2c3ea32eaba1699506398146` |

The reconciliation, verifier, and test source bytes also remained the same.
The sibling carrier change was quality-gate invocation hardening only:
`verify_w7_b2r.py verify --skip-upstream` avoids duplicating already-authenticated
upstream traversals while retaining the standalone B2R check.

## Workflow attempts and classification

The first `41e0c82f505fbbdc82493f0b62554cac0953277d` workflow classification failed
because the GitHub push event's previous branch SHA was the now-dangling
`15b387bf072958d6abe974005496cbea50a8ad25` carrier, while `41e0c82f505fbbdc82493f0b62554cac0953277d` was its sibling. Change
classification therefore could not resolve the non-fast-forward comparison in
the fresh checkout.

Subsequent attempts were:

- `e492968277707cb9e64f7a71eb352014c6ed462b` — docs-only portable verifier
  handoff; CI `33323823251`; cancelled under the 30-minute limit.
- `20336ec119062d194771d7520acce017a93deadf` — timeout raised from 30 to 45;
  CI `33325368975`; the complete gate reached pytest terminally and failed only
  because `tests/test_ci_workflows.py` retained the stale expectation `"30"`.

## Scientific boundary

For this closure:

- no scientific source changed;
- no candidate reran;
- no model-facing inference reran;
- no G-4 adjudication occurred;
- `lambda_core` remains provisional;
- W8 remains unopened;
- the test split remains sealed.
