# Resume ledger — PA / PB_1 / PB_2 / PB_3

**This file is the single source of truth for where the four-phase sequence stands.**
It is committed, so it survives a session dying mid-step. Prose in `NEXT.md` is a hand-off summary;
this file is the operational cursor. If they disagree, this file is right about progress and
`NEXT.md` needs updating.

Read this before anything else. Update it in the same commit as the work it describes.

## Rules

1. **Commit at every checkpoint.** Never let more than one checkpoint's worth of work sit
   uncommitted. Sessions end abruptly and without warning.
2. **Mark a step `in-progress` and commit that *before* starting it** if the step is long or
   expensive (a campaign, a sweep, a multi-file refactor). A crash then leaves evidence of where
   you were, not silence.
3. **Work-in-progress commits go on `main` with a `wip(<phase>):` prefix.** They are expected to be
   non-green. Push every one — an unpushed commit is not durable. Do not rewrite pushed history.
4. **A phase ends with one green commit** using its real conventional-commit message, after which
   every step below is `done` and the full verification block passes.
5. **Never `git reset`, `git checkout --`, `git clean`, or stash to "tidy up" on cold start.**
   Uncommitted changes are the previous session's unfinished work. Inspect and finish or commit
   them; do not discard them.
6. **Record observed facts here** (test counts, selected classes, SHAs) so the next session does not
   pay to re-derive them. Mark anything not yet re-verified.

## Cold-start protocol

    git status --short
    git log -12 --oneline
    git rev-parse HEAD && git fetch origin && git rev-parse origin/main

Then:

* If HEAD ≠ origin/main, or the worktree is dirty, reconcile that first. Uncommitted work belongs
  to the step marked `in-progress` below — read the diff before touching it.
* If the top commit is a `wip(...)` commit, the phase is mid-flight. Resume at the first step below
  that is not `done`.
* If a step is marked `in-progress`, do **not** assume its outputs are correct. Re-verify that one
  step's outputs from scratch, then continue.
* Only steps marked `done` may be trusted without re-checking.

---

## Status

**Current phase:** PB_1 — in progress (B1.1–B1.4 landed)
**Last green commit:** `dcf84a865b3249f9842e8755ebcaaee74b6aa805` (`docs(handoff): record the PA green commit SHA`)
**Next action:** run `instructions/PB_1.txt` step B1.5 — add `tests/test_classical_mutations.py`
covering the nine mutation classes (LLR sign reversal, disabled 16-QAM interleaver, wrong k,
dropped filler, unaccounted CRC, codec size above budget, silently skipped infeasibility, a
substituted channel implementation, sequential rather than keyed noise). Then B1.6 bounded
executions, then B1.7. Note: PB_1 commits are `--no-gpg-sign` (the user chose this; pinentry
timed out).

---

## PA — recover and harden post-G-2 state

| Step | State | Notes |
|---|---|---|
| A1 establish exact state | done | fresh run; no prior `wip(handoff)` commits, clean worktree, HEAD = origin/main |
| A2 fixture workflow + docs | done | fetch-tool audit (do NOT re-audit): **already present** — pinned-asset-only fetch, complete-asset SHA-256 vs `baseline.ldpc_golden_vector_asset_sha256`, produces the ignored `.npz`, records source rung 2, never writes third-party bytes anywhere tracked (`.gitignore:157` + `/data/*`). **Added** — (a) `.npz`-absent guard: was gated on the *asset tarball*, now a network-free no-op when the fixture exists, `--force` to re-materialize; (b) inner-archive verification widened from encoder-only to *every* pinned archive in `baseline.ldpc_golden_vector_sha256` (encoder + rate_matcher + segmenter). Docs: fetch+verify_g2 lines added to `AGENTS.md`, fetch line added to `README.md`, both with the fresh-clone rationale. Offline floor left ungated. |
| A3 complete preflight | done | all 14 commands pass with newly observed output (facts table below); 473 tests passed *as measured at A3*, later 501 after A4b and A5c added tests; CUDA present, so nothing is expected to fail on this install path |
| A4 repair hand-off + consistency check | done | A4b: `tools/check_doc_consistency.py` gained checks 6 and 7 — (6) NEXT.md's live sections must agree with the phase declared in the table under `## Single next task`: no live section may prohibit the declared frontier (unless it names a narrowing sub-scope such as the full BR-4 sweep), none may direct a completed subject as next, each *present* frontier section must name the frontier, and the frontier must be named outside the declaration; (7) any fenced preflight block must run `fetch_ldpc_golden_vectors.py` before `-m pytest`. History is exempt via struck headings/lines, DONE/Complete/PASS markers and `## Session log`. 10 mutation tests in `tests/test_doc_consistency.py`; verified firing on the two real defects. A4a: `NEXT.md` rewritten — canonical six-line current-path block added under `## Single next task`; the stale "Do not begin W4, G-8, or the reference-classifier fallback ladder" replaced; the Cold-start directive changed from "begin the transparency-bitrate probe only" to bounded W4, and its command block gained `verify_transparency_bitrate_probe` / `fetch_ldpc_golden_vectors` / `verify_g2_adjudication` before `pytest` (expect 473). Session-log "Next: W3 ..." entries left alone (dated history). |
| A5 G-2 source provenance manifest | done | **A5a** — `results/baseline/g2/execution_source_manifest.json` generated by new `tools/gen_g2_source_manifest.py` (`--check` mode too): 14 sources, 4 roles (runtime / measurement_runner / configuration / record). **A5b done** — `tools/verify_g2_adjudication.py` extended and passing against the real repo; `evidence_commit: null` removed from `g2_adjudication.json`, replaced by an explicit `evidence_commit_resolution` policy string, documented in the verifier docstring, and the verifier now *rejects* any recorded `evidence_commit`. HOLD condition checked: all 8 `src/baseline/ldpc/` files byte-identical to `968e907…`. **AM judgment: no amendment.** No requirement, gate, decision or parameter changed; G-2's §13 gate text names no evidence filenames and no source-binding rule, the recorded results are unchanged, and evidence-commit resolution through Git path history was already the verifier's behaviour before this change. This only strengthens verification. `schema_version` stays 1 for the same reason: a field that recorded nothing (`null`) was replaced by a statement of the policy already in force. |
| A5b G-2 mutation tests | done | **all ten classes pass** (A5c): unreachable-measurement, unreachable-evidence, wrong-ancestry, missing-source, unexpected-source, wrong-blob-sha, wrong-byte-sha (×2 roles), modified-current-runtime, wrong-measurement-config (config_sha256 + params_sha256), wrong-packetisation-record. Plus six beyond the required set: manifest-binds-a-different-commit, wrong-role, readjudication-permits-drift, readjudication-of-another-path-does-not, recorded-evidence_commit-rejected, manifest-is-regenerable. `tests/test_g2_adjudication.py` 23 → 41 tests. |
| A6 green commit + push | done | full 14-command preflight re-run at A6, all pass (see facts table); `git diff --check` and `git status --short` clean. PA's `wip(handoff)` commits are left in history deliberately — nothing was rebased or force-pushed. |

## PB_1 — classical transport path

| Step | State | Notes |
|---|---|---|
| B1.0 confirm PA green | done | clean worktree, HEAD = origin/main = `dcf84a8`, `verify_g2_adjudication.py` PASS (`measurement=968e907237bb, rows=24, test_split_access=0, sources=14`), LDPC fixture present |
| B1.1 `channel_transport.py` | done | `src/baseline/classical/{__init__,channel_transport}.py` + `tests/test_classical_transport.py` (22 tests, all pass). **Defect found and fixed in `src/baseline/ldpc/transport.py`:** `transmit_transport` built the Sionna encoder with `K` (systematic length *including* our explicit filler); Sionna re-derives `K_b`/`Z` from the information length it is given, so it selected a different lifting size and the pre-existing `lifting_size` guard raised on real plans. Passing `K'` reproduces the TS 38.212 §5.2.2 lifting size — verified over **all 232 (configuration, `E_r`) pairs of the committed packetisation record: 0 mismatches with `K'`, 46 mismatches/errors with `K`.** Sionna owns filler insertion; `segment()`'s contract is unchanged and the block is sliced to `K'` at the seam. Also added `receive_transport_verified` → `ReceivedTransport` (CRC verdicts reported, not raised) so decode failure stays classifiable; `receive_transport` delegates and keeps its old raising behaviour. These paths were never exercised before — G-2 measured through the adapter directly. |
| B1.2 `pipeline.py` | done | `src/baseline/classical/pipeline.py` + `tests/test_classical_pipeline.py` (13 tests). Full segment wired: `codec_input` → `codec_downsample` → `J2KCodec.encode_to_budget` → zero-filler padding to A → `transmit_transport` → `transport_round_trip` → CRC → EOC-truncated payload → `J2KCodec.decode_codestream` → `codec_upsample`. Receiver recovers the codestream with **no signalled length**: filler is zero bytes so the *last* `ff d9` EOC in the padded payload is the real one (`control_plane_policy` stays honest). Added public `J2KCodec.decode_codestream` so the receiver decodes what it received, not the encoder's cached image. |
| B1.3 accounting + failure taxonomy | done | Four verdicts, all observed and mutually exclusive: `structural_infeasibility` (before any encoding, `accounting`/`source_coding`/`transport`/`noise_id` all `None`), `codec_infeasibility` (`packet_feasible=True`, accounting present, per-axis reasons recorded), `decode_failure` (transmission happened; measurements survive), `delivered`. **Per-axis sub-reasons** `budget_exceeded` / `codec_configuration_error` are recorded inside `codec_infeasibility` — see the open issue below. |
| B1.4 required tests | done | **All 14 required areas covered** across `tests/test_classical_transport.py` (22 tests) + `tests/test_classical_pipeline.py` (13 tests): exact channel uses (215 feasible × ratio × modulation × rate), exact bit reconciliation (every committed packetisation row, both identities), shared channel object (registry factory spy), keyed noise for one identity (unchanged by intervening draws), three modulations, four LDPC rates, partial final code block (`E=(8532,8532,8536)`), structural-vs-codec distinction, decode-failure classification, J2K emitted-byte authority, J2K cache identity, no per-packet rescaling, realised symbol energy, PAPR, validation-only loading with test access sealed. |
| B1.5 mutation tests | not-started | |
| B1.6 bounded executions | not-started | |
| B1.7 green commit + push | not-started | |

## PB_2 — outage, records, smoke evidence

| Step | State | Notes |
|---|---|---|
| B2.0 confirm PB_1 green | not-started | |
| B2.1 `outage.py` + validation selection | not-started | |
| B2.2 `records.py` + identities | not-started | |
| B2.3 smoke runner + configs | not-started | |
| B2.4 `verify_w4_baseline_integration.py` v1 | not-started | |
| B2.5 required + mutation tests | not-started | |
| B2.6 bounded executions + evidence | not-started | |
| B2.7 green commit + push | not-started | |

## PB_3 — BR-4 selection infrastructure + W4 adjudication

| Step | State | Notes |
|---|---|---|
| B3.0 confirm PB_2 green | not-started | |
| B3.1 `composition.py` arithmetic | not-started | |
| B3.2 BLER lookup + support guard | not-started | |
| B3.3 candidate cache + tie-break | not-started | |
| B3.4 system modes + two-pass limit | not-started | |
| B3.5 sweep budget guard | not-started | |
| B3.6 required + mutation tests | not-started | |
| B3.7 adjudication evidence | not-started | |
| B3.8 spec bookkeeping judgment | not-started | |
| B3.9 green commit + push | not-started | |

---

## Observed facts (carry forward; re-verify anything marked stale)

| Fact | Value | Observed at | Verified |
|---|---|---|---|
| A1 HEAD | `174cf19bfa2b10cb89d85211ab330e5cd8251de0` | A1 | yes |
| A1 origin/main | `174cf19bfa2b10cb89d85211ab330e5cd8251de0` | A1 | yes |
| A1 worktree | clean; `git diff --check` clean | A1 | yes |
| adjudicated fixture SHA-256 (local, matches `results/baseline/g2/golden_vector_summary.json`) | `55754b508ab1b6eb6625eae301d2d0a3fefcdf7b03e98038264b76b71e26aae0` | A2 | yes |
| fixture regeneration reproducibility | re-materializing from the pinned asset reproduces all 6 arrays and the same `.npz` SHA-256 | A2 | yes |
| A1 divergence from PA.txt expected HEAD | none material — `174cf19` is docs-only (`chore: add instructions for post W3`, adds `instructions/` only, 5 files / +1443 lines); last code commit is still `82f6c56` | A1 | yes |
| A3 cmd 1 `gen_spec_views --check` | ok: 187 requirements (2 retired), 10 generated files up to date | A3 | yes |
| A3 cmd 2 `check_doc_consistency -v` | ok: 11 current docs consistent; 1 valid historical plan excluded; 23 stale rules; 187 reqs / 79 AMs | A3 | yes |
| A3 cmd 3 `check_literals -v` | ok: 37 Python files scanned, 0 findings, 43 reasoned literal-ok annotations | A3 | yes |
| A3 cmd 4 `check_packetisation` | 215 feasible (144 obligation), 215/215 byte-aligned, 215/215 `B' % C == 0`, 215/215 zero-slack E sum, 3 min-rate clamped, **0 failures** | A3 | yes |
| A3 cmd 5 `fetch_datasets --check` | all 3 archive pins verified (cifar10 / imagenette160 / stl10) | A3 | yes |
| A3 cmd 6 `materialize_manifests --check` | all 3 manifests byte-identical to committed; 45000/5000/10000, 8469/1000/3925, 4500/500/8000 | A3 | yes |
| A3 cmd 7 `verify_datasets` | all 3 real train/val smoke pass; `test_scan_decoder_calls=0`, `test_scan_canonicalization_calls=0` per dataset | A3 | yes |
| A3 cmd 8 `verify_g1_adjudication` | PASS: 100 epochs, best=898/1000, local checkpoint verified | A3 | yes |
| A3 cmd 9 `verify_g7_profile` | PASS: commit=26b631ede27a, params=1640957, epoch=48.684s, reserved=1.004 GB, projected=1.352 h | A3 | yes |
| A3 cmd 10 `verify_transparency_bitrate_probe` | PASS: A=90007f165f8f, B=7896c7a74414, C=2ebb2cefade2, 68000 cells, 5pp=1330 B, 2pp=3200 B | A3 | yes |
| A3 cmd 11 `verify_g2_adjudication` | PASS: measurement=968e907237bb, rows=24, test_split_access=0 | A3 | yes |
| A3 cmd 12 `pytest` (full suite) | **473 passed, 0 failed** in 60.76s at A3; **483 passed** after A4b added 10 tests | A3 / A4b | yes |
| A3 cmd 13 `verify_cpu_lock --clean-install` | ok: structurally CUDA-free and clean plain-pip install CUDA-free (torch 2.13.0+cpu, torchvision 0.28.0+cpu) | A3 | yes |
| A3 cmd 14 `git diff --check` | clean; `git status --short` empty | A3 | yes |
| A6 preflight re-run | all 14 commands pass again, plus `gen_g2_source_manifest.py --check`. 501 tests pass; `tests/test_test_access.py` 4 pass; CPU lock clean install CUDA-free; `git diff --check` and `git status --short` clean | A6 | yes |
| `check_literals` scope | 37 files is correct and unchanged: `config.literal_lint_scope` is `src/` only, so the new `tools/gen_g2_source_manifest.py` is deliberately out of scope | A6 | yes |
| total tests | 483 passed (0 failed, 0 skipped) — 473 at A3 plus 10 added by A4b | A4b | yes |
| `tests/test_test_access.py` count | 4 tests, all pass | A3 | yes |
| CUDA available | **yes** — torch 2.13.0+cu130, `torch.version.cuda=13.0`, `is_available()=True`, NVIDIA GeForce RTX 4060 Laptop GPU. So `tests/test_env.py::test_cuda_build` (AM-67) passes; this is not the CPU-only install path | A3 | yes |
| test-isolation counters | all zero: decoder=0, canonicalization=0 (per dataset, `verify_datasets`), inference=0, accuracy=0 (`results/baseline/g2/resolved_config.json` `test_split_access`, re-verified live by `verify_g2_adjudication`) | A3 | yes |
| G-2 runtime vs adjudicated implementation | all 8 `src/baseline/ldpc/` files byte-identical to `968e907…`; HOLD condition **not** triggered | A5 | yes |
| G-2 manifest cross-checks | 4/4 agree with hashes the campaign recorded independently (`config_sha256`, `params_sha256`, `solver_record_sha256`, `script_sha256`) | A5 | yes |
| suite timing note | 501 tests in ~129 s, up from 60 s at A3. The new G-2 tests account for 2.6 s; the rest is pre-existing `test_transparency_bitrate_probe.py` (~97 s alone), which reads many blobs through git. `git count-objects -v` shows 810 loose objects / 47.9 MiB against 84 in-pack — a repack would likely recover most of it. Not done here: repacking is repo-wide and outside PA's scope | A5 | observed, not acted on |
| **OPEN ISSUE — `j2k_resolutions` vs CIFAR-10 axes** | `baseline.j2k_resolutions = 6` requires every tile dimension ≥ `2^5 = 32`, but `baseline.downsample_axis_px.cifar10 = [32, 24, 16]`. OpenJPEG hard-errors at 24 px and 16 px (*"Number of resolutions is too high in comparison to the size of tiles"*) for **every** image, so two of CIFAR-10's three configured axes cannot encode at all. Never caught before because the transparency probe ran Imagenette only (160/128/96/64). B1 records it per axis as `codec_configuration_error` rather than silently skipping. **Needs a spec decision (probably an AM): either clamp `j2k_resolutions` per axis to `min(6, log2(axis)+1)`, or drop 24/16 from the CIFAR-10 axis list.** Not decided in PB_1 — it changes a frozen codec parameter and therefore the codec configuration hash and every J2K cache key. | B1.2 | yes |
| **OPEN ISSUE — cache-key field spelling** | `baseline.j2k_cache_key` names `j2k_impl_version`; `J2KCodec._cache_identity` spells the same value `openjpeg_version`. Values agree, names do not. **Deliberately not renamed** — the committed transparency-probe evidence records cache keys produced under the current spelling, so a rename would invalidate them. Tested by value. | B1.2 | yes |
| B1 J2K payload framing | no length is signalled; filler is zero bytes, so the receiver truncates at the **last** `ff d9` EOC in the padded payload. Verified byte-identical recovery on every delivered case. | B1.2 | yes |
| selected outage class | — | — | |
| outage class measured val accuracy | — | — | |
| W4 implementation commits | — | — | |
