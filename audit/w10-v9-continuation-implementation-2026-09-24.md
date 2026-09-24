# W10 v9 continuation implementation handoff — 2026-09-24

**Verdict for implementation:** source repair and continuation infrastructure are ready for an independent source-freeze audit. No v9 manifest, continuation authority, launch grant, continuation plan, suffix unit, closeout, G12 artifact, or test access was created by this task.

## Historical authentication before source changes

The original `confessor` checkout was at execution commit `c906ab920cba652c8ea681b4cefda869d5e829b2`. Its reflog places that commit at 2026-09-24 00:28:19 IST, before the service start at 00:52:52 IST. The failed `w10-validation-rehearsal.service` invocation was `.venv-pascal/bin/python tools/run_w10_rehearsal.py execute`; it exited status 1 at 14:56:13 IST with `KeyError: 'authority_candidate_id'` at `src/evaluation/w10_classical.py:177` on ordinal 126. The service and traceback were read from systemd on `confessor`.

The read-only audit used the original v8 source and authority. It called the repository's canonical authority, unit, scorer-stream, per-image, noise and pair validators, then checked exact file sets. The frozen custody record is [`w10_v8_failed_prefix_custody.json`](../results/learned/w10/w10_v8_failed_prefix_custody.json). Its `custody_id` is `w10v8prefix-a4cbafd6366eb105c24c3806c9182689c81ff4323aee4d45138d75ad4b80171f`, file SHA-256 `4b20751860633e815bb63236ad5796bfba0092496ec59456428d465d6f7b53a1`, and complete-prefix digest `4f83fc7d5889ab1209f7e3c66a7d9e485e9c940509c981252c8c05d31f9769a5`.

It records exactly 126 completed units, ordinals 0–125, and 168 complete scorer streams. Unit 126 and unexpected or incomplete unit/stream files are absent. Every stream has the expected 1,000 stable IDs in validation order, canonical identity and digest, recomputed correctness, scheduled noise identity and pair identity. The original `plan.json` is validation only. There is no W10 closeout; G12 is unopened; test is `SEALED` and `test_access=0`. The original runtime, unit files, per-image files and plan remain worker-local and byte-identical.

| Historical object | Identity | Exact byte SHA-256 |
| --- | --- | --- |
| v8 source manifest | `w10downstreamsourcev8-e13ba8905c1910645794ffdf5d7a06c55c55b8a01beec84a392a4a59f9bc48fc` | `c37ce2c845acaa111b9ee0d49552867fc671e2ffe7e593589102c8d7e1a1f3c3` |
| v8 W10 authority | `w10rehearsalauth-18582571f86b71aadcd546808f1199410d1c55cd0237def20689d985aa21eb7d` | `0db7e8900c8cb4415e34e62167e8065dba5ecda8648f28d5e2885729c20abdb0` |
| Failed prefix custody | `w10v8prefix-a4cbafd6366eb105c24c3806c9182689c81ff4323aee4d45138d75ad4b80171f` | `4b20751860633e815bb63236ad5796bfba0092496ec59456428d465d6f7b53a1` |

## BR-16 repair and prefix equivalence

`classical_unit()` now resolves three explicit selection kinds. Adaptive pass two still reads the frozen candidate ID and produces the exact prior tuple `(modulation, LDPC rate, encode axis, canonical candidate hash)`. The candidate-authority file is checked against the authority's frozen SHA-256 before use. BR-16 reads its frozen QAM16/rate-1/2/axis-160 fixed configuration directly and has no invented candidate ID. JPEG still uses its exact frozen quality and PHY point and rejects a quality mismatch. Missing fields raise; the resolver adds no defaults.

The failed v8 expression `point["authority_candidate_id"]` was exercised on all 21 BR-16 points and raised the original `KeyError`; the corrected resolver returned the fixed configuration for all 21. A synthetic `classical_unit()` test also completed the actual BR-16 route without running a scientific evaluation.

The source diff from `c906ab9` leaves `w10_dispatch.py`, `w10_backends.py`, `w10_evidence.py`, `w10_scope.py`, `w10_bindings.py`, `w10_selections.py` and `djscc_validation.py` byte-identical. This preserves completed learned loading, checkpoints, classifiers, validation order, scheduled noise and SR-18 schemas. The new orchestration branches apply only to continuation authority; original v8 unit bodies are resumed and validated without rewriting. On `confessor`, all 84 completed learned units' checkpoint and relevant protocol bindings matched the frozen authority, and all 42 completed adaptive units' recorded PHY and configuration hashes matched both the v8 candidate lookup formula and the corrected resolver for every SNR.

| Completed arm | Ordinals | Existing binding digest over 21 SNRs |
| --- | --- | --- |
| Ordinary learned, r_1_6 | 0–20 | `58fb71b87bf5cae4d4e2107716f1e6e19894423ace0af2ddae94ec812c97a75f` |
| Ordinary learned, r_1_24 | 21–41 | `93daa27ec60f7358c754d46ad35b88eea9982de26a5cab03f1886add99fd9014` |
| Randomized-SNR learned | 42–62 | `d6456cbef8dc93f4df066415df7c6af800933a421c79af619c5bf5e51495e976` |
| PAPR-constrained learned | 63–83 | `51818fa90e3a2c7e9f25f144f5a0fd255ed1be503816cf13252c4297fa119257` |
| Adaptive classical, r_1_6 | 84–104 | `de3b14fab206ea841de72e781457a998598a734e5f736df3320f16cbfce61017` |
| Adaptive classical, r_1_24 | 105–125 | `f3b908d72212d2ec01066d8d835f18b5df2973bc5f71e94269e2e4e7e2a44587` |

The existing hardcoded J2K cache root remains `checkpoints/w10_rehearsal/j2k_cache`. Cache keys include canonical pixels, byte budget, encode axis, codec configuration and OpenJPEG version. Existing entries are validated on read; a missing key is written atomically. The cache is not scientific unit or scorer evidence and is excluded from the custody record. The sole-writer continuation does not rewrite original unit, stream or plan paths.

## v9 dependency and execution boundary

| Stage | Exact inputs | Output and gate |
| --- | --- | --- |
| Source-v9 freeze, later task | Audited implementation commit; exact v8 manifest bytes; v8 authority; failed-prefix custody; PAPR completion; JPEG selection under v5; ER-12 selection under v8 | `w10_downstream_source_manifest_v9.json`; explicitly records v8 scientific work, completed 0–125 and failed 126. No before-science supersession claim. |
| Continuation authority freeze, later task | Authenticated source-v9, original frozen 252-unit scope and bindings, custody digest, exact Confessor TITAN Xp UUID `GPU-46acd0f2-2ff5-1a43-cac9-2ae20e56dc9a` | Separate `w10_continuation_authorization_v9.json`; only ordinals 126–251 are new. Original v8 authority stays byte-identical. |
| Plan, later task | v9 authority and complete worker-prefix reauthentication | Separate `continuation_plan_v9.json`; original `plan.json` unchanged. |
| Owner launch grant, later task | Authenticated v9 authority, plan and worker-prefix custody | Separate content-bound `w10_continuation_launch_authorization_v9.json`; executor refuses to run without it. |
| Suffix and closeout, later task | All prior gates and exact suffix file set | New units carry v9 execution authority in their binding. Closeout maps each ordinal to its actual v8 or v9 source and authority. Hosted verifier checks committed bodies and digests; worker verifier checks per-image bytes. |

The unchanged scientific scope digest is `bd2ddf6bb86b79808eacdecae231872e27e38fb13d72466d1aca90fe2fda9c52`. PAPR completion remains `paprcompletion-2d23d342ad7fe2661c9a3926f5e025b3d7174c290f5e56983d05efbbc65d74ca` with byte SHA-256 `ced0e6a955071252863558dd9b1db41fc93d408435cdb2131d729e4b18afdfa4`. JPEG selection remains `w10jpegselection-ff847b73e8f117bf6061b74f21d33fa48db4fbedd7399a105e8a5733b039601d` under source-v5; ER-12 selection remains `w10er12selection-46c1dc6b10c2eb4557b2f2ce702928c971f93d306050f5029c43be43aa4e5543` under source-v8. Neither is reselected.

## Checks and later commands

Focused synthetic and historical tests: **65 passed** across the v9 continuation, classical route, v8 lineage, authority, selections, protocol, JPEG carrier and ER-12 binding tests. `py_compile`, `check_literals.py`, `gen_spec_views.py --check`, `check_doc_consistency.py -v`, and `git diff --check` passed. The full static gate and two existing live-source tests stop at the deliberate v8 live-source closure: protected v9 implementation bytes exist while the v9 manifest is intentionally absent. This is a deterministic transition HOLD, separate from the known hosted 45-minute timeout. The production source guard was not weakened; isolated historical-source and synthetic fixtures cover the transition.

After an independent audit of the committed implementation source, the source-freeze agent should run on a clean checkout:

```sh
git rev-parse HEAD
.venv/bin/python tools/gen_w10_downstream_source_manifest.py --epoch v9
.venv/bin/python tools/verify_w10_downstream_source.py
```

Before that freeze, the worker can independently reauthenticate the exact historical runtime after fetching the implementation commit:

```sh
.venv-pascal/bin/python tools/verify_w10_v8_prefix.py --historical-source
```

Its expected result is the custody ID above, `units=126`, `streams=168`, `failed=126`. The v9 manifest ID must have prefix `w10downstreamsourcev9-` and bind the audited committed HEAD; its complete ID and byte SHA-256 are only knowable after the separate source freeze. Independently verify and publish that manifest before the separate authority freeze. On `confessor`, first preflight the future authority:

```sh
.venv-pascal/bin/python tools/gen_w10_continuation_authorization.py --preflight
```

After the separately authorized authority freeze, create the separate plan and preflight the later launch grant:

```sh
.venv-pascal/bin/python tools/run_w10_continuation.py plan
.venv-pascal/bin/python tools/gen_w10_continuation_launch_authorization.py --preflight
```

The authority generator without `--preflight`, the launch generator without `--preflight`, and `run_w10_continuation.py execute` require their respective later owner decisions. None was run here. The worker-local per-image bytes remain necessary for the terminal verifier; hosted CI authenticates only published unit bodies, historical custody and content digests.
