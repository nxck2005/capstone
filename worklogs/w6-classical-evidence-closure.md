# W6-B terminal classical evidence closure

**Date:** 2026-08-27
**Scope:** terminal publication and read-only verification only
**Authorization:** the owner-authorized W6-B takeover instruction. No W7/G-4,
learned optimizer step, W8, validation selection, test access, or scientific
rerun was authorized or performed.

## Accepted W6-A authority

The repository started clean at carrier `ad99dc9597e4b23290825ed11afb06ef941d04b5`,
with `origin/main` and the remote `main` at the same SHA. The accepted source
epoch is `d0e04d0ccc92e2fa7dae0be798da4b6bd8960854`, whose parent is
`fc0117f511f8309040807f80a162006dbeb0e89c`. The source commit timestamp is
`2026-08-27T18:02:54+05:30` and the carrier timestamp is
`2026-08-27T18:04:05+05:30`; both are 2026-08-27 UTC. The accepted exact-SHA
CI evidence is run `33073771159`, job `98522521217`, carrier SHA
`ad99dc9597e4b23290825ed11afb06ef941d04b5`, conclusion `success`.

The W6 contract remains byte-identical: ID/SHA
`w6acontract-d2378ea58aaf2cd255e21be5b9f6597786748c386485b5b5d81b8cdf9e0f80ab`.
Its line `Frozen: 2026-08-28` is retained unchanged. That human date is a
one-day process-provenance nit relative to the actual 2026-08-27 UTC commit
chronology; scientific/protocol effect is zero and no amendment was made.

W6-A source manifest
`results/baseline/w6/w6_a_source_manifest.json` is ID
`w6asource-43327095c174e03caec0d8f21a8132cee15357dfd20eca828d6ab1d5624f3eea`
with file SHA-256
`eec8d2ba010ec821ac466a36595ab65c79008ac26f583b1979aa9bdc30749c9f` and ten
source entries. The accepted W6-A source-critical files, index, matrix and
manifest were rehashed against the source/carrier commits; no accepted W6-A
source-critical byte changed. The manifest's terminal-publication flag remains
false.

## Terminal publication

The additive completion is
`results/baseline/w6/w6_completion.json`:

- completion ID: `w6completion-d7df2e37b34c68754b5b1a638e74a0726ef4d868b743457fedbd3a17d3267142`;
- canonical content SHA-256: `4181b0d756177955f78c7eb2e1aca9b81424b4d5e8230d9b6921ae2a4686d34c`;
- file SHA-256: `7e8aaf4adcc867da8b2cf6ccb2414b5f674f202498050d7483ca741956663dfa`.

`tools/verify_w6_complete.py` is the terminal verifier (source SHA-256
`eea6ecfa2353887611d4bf51ef133878636425638cfa1ade9aaf8dd9e8f402f1`). It
reconstructs the completion from current bytes and independently invokes the
existing read-only readiness/closeout verifiers. Publication is exclusive and
same-directory fsync-backed; replacement is refused.

The accepted W6-A index and matrix remain exact:

| Artifact | ID | file SHA-256 |
|---|---|---|
| `results/baseline/w6/w6_classical_evidence_index.json` | `w6aindex-ac05dbada7d28ad9e209ed498baddccbb71fe62c5430c75536c726ef4d6dee9d` | `efa879d7f592e6c07e0a2c0ad17199af6d91e17e243521c1834c206afb3f035d` |
| `results/baseline/w6/w6_requirement_matrix.json` | `w6amatrix-d1a1add6bfa93f066ec27d3cc6afa11698e5629c5c395581ca5117250e1b3708` | `88c00d24c8e9d15d6aefde881ddde151fd53cc5b649fbd97f3c9d191e301f3a4` |

The deterministic index/matrix regeneration and W6-A verifier pass. Matrix
counts are 21 `W6_REQUIRED_AND_SATISFIED`, 0
`W6_REQUIRED_AND_MISSING`, 9 `FROZEN_UPSTREAM_INPUT`, 14
`FUTURE_G12_TEST_EXECUTION`, and 2 `NOT_APPLICABLE_TO_W6` (46 total).

## Machine-bound readiness and frozen science

Read-only readiness is machine-bound to the G-1 adjudication and verifier,
G-2 adjudication and verifier, and W4 integration adjudication and verifier.
The terminal verifier re-runs those checks rather than trusting completion
prose. G-1 is PASS at 898/1000 on Imagenette-160 with the test split sealed;
G-2 is PASS with 24 adjudicated rows and its recorded runtime readjudication;
W4 is bounded-integration complete with G8 still explicitly unresolved and no
operating-point selection or test access.

The re-authenticated frozen classical boundary is:

- G8_C successor table `g8pblertable-69ecc729…`, 153 curves, 3,213 measured
  points, 5,000 trials per point; predecessor contribution is zero;
- G8_D D7 and G8_E E2–E7 are verified, with pass one exactly once and 378/378
  cells selected;
- F1 is exactly 50,814 assignments/results, 44,039 materialized artifacts,
  6,775 typed codec infeasibilities and zero unexpected outcomes;
- F2 is frozen at 20 epochs and 6,900 optimizer steps, with zero-based epoch
  17 selected at 890/1000;
- F3 rescored exactly 288,000 historical validation rows without re-encoding;
- pass two is exactly one execution: 18 calls, 8,190 candidate evaluations,
  378 SNR cells and 95 tie breaks; 162 cells changed relative to pass one;
- pass three, fallback training, learned training and test access are zero;
- G8 terminal closeout and its additive binding correction are unchanged.

The F1 corpus manifest is bound at 50,814 rows and SHA-256
`792cce92bd8a72f99b7ddee58511d1b5b7e908a4d0cd4178bbb08b9e1ba2d144`.
Ordered request/result, set and codestream/reconstruction object digests and
the worker custody policy are carried in the completion. The 4.1-GiB worker
runtime is not copied into Git, and the raw artifact-training corpus is not
required by W7, W8 or W11 after the artifact-classifier freeze, F3 scoring
freeze and pass-two closeout.

The selected operating points remain efficiency `r_1_24`, crossover/headline
`r_1_6`, and low `r_1_24`. Both selected ratios retain four feasible LDPC rates
below half-budget format overhead. ER-1 remains full strength only at headline
`r_1_6` and sweep strength at efficiency `r_1_24`. BR-16 remains JPEG 2000,
axis 160, QAM16, rate 1/2, one packet, design SNR 7 dB. H2 remains 3–7 dB
with a 79 percentage-point classical fixed-MCS change. BLER lookup,
composition, candidate authority, tie-break policy, scorer and no-re-encode
boundaries are authenticated; no interpolation or pass-three mechanism is
introduced.

The frozen-selection consumer uses the default exact-terminal path for both
pass-two state and candidate-authority bytes, returns the already-selected
`cand-15e6711e9b406157262234a8` cell, and performs no selection, scoring,
interpolation, codec execution, channel simulation, validation inference or
test loading. Repository search found no production caller using its internal
non-terminal testing opt-out. Future W11 scientific consumers must use exact
terminal authentication unless a later explicit protocol amendment says
otherwise.

The current W5 prerequisite is the repaired authority
`w5repaircompletion-8b2fa917…`; the superseded pre-repair completion is not the
current authority. Its scientific and protected counters remain zero.

## Protected future boundary

The terminal record explicitly leaves W7/G-4, W8, learned validation results,
actual classical test rows, paired test outcomes, the JPEG secondary test
curve, fixed-modulation and BR-16 fixed-MCS test curves, packet-count
sensitivity, G-12 and final hypotheses incomplete. The test split remains
sealed. All W6 protected counters for G8 changes, F1/F2/F3/pass reruns,
BLER regeneration, reselection, learned training/validation/test activity,
W7/W8 work and model-facing test access are zero.

Targeted W6-A plus W6-B tests pass (56 tests in the combined invocation),
including resigned completion attacks for source/manifests, index/matrix,
readiness, corpus, scorer, pass counts, ratios, ER-1, nondegeneracy, BR-16,
H2, W5, protected counters and the future boundary. The current terminal
verifier and deterministic evidence checks pass. The `full-local` quality gate
also passed, including the complete local pytest invocation and all static
checks. Exact-final-SHA CI remains the final publication check after the
terminal commit is pushed.

**Next:** W6 is GREEN/CLOSED. W7/G-4 requires separate owner authorization;
W8 and test remain sealed.
