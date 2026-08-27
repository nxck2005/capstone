# AGENTS.md

This file provides guidance to coding agents (Claude Code, and any other agent that reads `AGENTS.md`) when working with code in this repository.

## Repository status

**Implementation started 2026-07-28. W1, W2 and W3 are complete; G-9, G-1,
G-7 and G-2 passed; the validation-only transparency-bitrate probe is complete
and lineage-bound.** W4, G8_A and G8_B are complete. The owner-authorized
Pascal G8_C successor execution and C3-C7 closeout are green: its authenticated
evidence is 3,213/3,213 identities accepted at 5,000 trials each and its
successor `BlerTable` is frozen. No selection, inference, training, validation
decoding or test access occurred, and the test split remains sealed. G8_C
remains green and closed. G8_D is GREEN and closed; the first G8_E E0/E1 freeze
is preserved as superseded-before-data history; the first corrected E1 is also
preserved as superseded-before-data history. Corrected-v2 is also preserved as
superseded-before-data history with zero coverage. The partial corrected-v3
local_4060_cu130 E2 campaign was owner-aborted at exact prefix 47409/288000
(`PARTIAL_OWNER_ABORTED_PROFILE_RELOCATION`, evidence preserved, zero successor
coverage), and execution relocated to the additive worker-successor epoch
`results/baseline/g8_e/e2_confessor_successor/` (contract
`g8econtractcorrectedv3s-2831f47d…`, campaign
`g8e-v3s-85354d3db97c74adfd01bc1c5fe2148e05dfebfb0d832229a3bce5ca10ebf588`,
source commit `ed0b92a3c9e3b38c8868e4845f0992e893b1cac2`, owner E2–E4-only
authorization issued_sha256
`be4291881601e35cffc54555d9ec34107990916971ea7c6f9ab56d2258a8c49f`,
commit `493d656`). **Worker-successor E2 is COMPLETE at exactly 288000/288000
on `confessor` (`confessor_pascal_cu126`/`cuda:0`) and VERIFIED; E3 exact-set
closure and E4 count-derived measured accuracy objects are COMPLETE and
VERIFIED** (outcome mix delivered 264,000 / codec-infeasible 24,000 /
decode-failure 0 / structural 0; training and test access 0; custody record
`results/baseline/g8_e/e2_confessor_successor/closeout_provenance.json`). A
closeout-layer identity-loading defect was repaired additively with no bound
byte changed (repair provenance `g8ecloseoutrepair-5ac7129b…`, commit
`0f65fb4`; corrected entry points `tools/closeout_g8_e_v3s.py verify|merge|
aggregate`). **G8_E is now GREEN and CLOSED through E7** under the owner's
2026-08-22 takeover authorization: the narrow E5 pass-one authorization issued
`d6b0ac7e15299d3b08d9baff63e5361b2fac90aa9166ee0a93672a76c1b1bc33` (marker
`c27100431317cc2dc4fffc434705361215157d9805b3a5217574843ed0387fb4`, commit
`d84ee90`) executed selection pass one EXACTLY ONCE into the immutable record
`g8epassone-1b12616866e248c3320d0d12248e3c543fd553cc8f5eac88e1d76837878bc413`
(content SHA `7d5ad533af0fc8a2ebfd85bc4f2a8a1639f65d6b7c7a85eeeee447e69eff00fa`,
file SHA `91d6ac9d17386a8d5a5a584cf1605e1b382e5416c9ffbaa6e3a204aaee016446`):
18 frozen calls, 8,190 mode-admissible/ranked candidate evaluations all eligible
(not the larger raw candidate-call space before per-mode filtering), 378/378 SNR
cells selected, zero infeasible or uncharacterized, and training/pass-two/pass-
three/fallback/ratio/test counters all zero. E6 froze the additive corpus-spec
lineage completion
`g8ee6freeze-ac45f8cf13094b72727ec9d9a626d439791649a91d4d3e6427a5c7cb9d2cb303`
(E1 corpus-spec bytes untouched). E7 `tools/verify_g8_e_complete.py` returns
PASS with verdict **G8_E GREEN — VALIDATION CAMPAIGN AND PASS ONE FROZEN;
G8_F READY; NO TRAINING OR PASS TWO** and authenticates the terminal handoff
`g8ee7handoff-1af54fbf248cfa233ea74dc516697f0ca9153f4562798680de5b20d35da0a4d8`
(file SHA-256 `a726a6a433fd42e0b0dcb97f1b12615a44528fee25af55a157f594e393824c49`).
The ci-cpu lane was repaired
host-independently on 2026-08-22 (commit `824d49c`, Actions run `32579946365`
green). A 2026-08-23 clean-checkout test-harness defect unintentionally created
a separate unauthorized corrected-v3 runtime at prefix 42704/288000 under
`/home/nick/projects/capstone-ci-clean`; it independently authenticates but has
zero successor coverage and is permanently merge-ineligible. Its custody and
the canonical-runtime separation proof are recorded in
`audit/g8-e-clean-checkout-runtime-incident-2026-08-23.md`. Preserve that
runtime in place and do not use the contaminated worktree for scientific
execution. Do not reopen or rerun G8_C/G8_D, do not rerun or resume E2, do not
merge or ingest either preserved partial runtime, do not widen the closed
E2–E7 scope, and do not alter pass-one selections or the frozen corpus-spec
bytes. The separately authorized G8_F/F1 sole writer completed and is closed;
do not rerun it, train any classifier, run inference/pass two, invoke fallback,
adjudicate ratios or access test. **AM-87 resolves the
post-pass-one BR-12 support-breadth defect; AM-88 supersedes only its exhaustive
Cartesian execution multiplicity before F0.** AM-87 plan
`g8fcorpusplan-6320ea3…` still freezes 120 exact deduplicated artifact qualities
and all 8,469 eligible training stable IDs. AM-88 metadata-only plan
`g8fsamplerplan-d6d64ead5295b93c2a73aefd5f0719dd438bd6c0425286a33a31f1fba3ff64d6`
(file SHA-256 `eca85a9891bcf2054e132e5fc277430d2c85962a78f8438c9da0604d98447e23`)
assigns exactly six distinct supported qualities per image: 50,814 attempts,
global quality counts 423–424, every class range 0–1, no duplicate pair. The owner independently accepted AM-88 and first authorized **F0 only** on
2026-08-24. F0-v1 remains byte-identical historical evidence at
`results/baseline/g8_f/f0_execution_authorization.json`
(`g8ff0auth-92189865202e4b6cb400a0a86cee101b8ad8a7bdf5ea9d5a78ae96ab49a365b4`,
file SHA-256 `17a88e36201d42b3b2ace190b0b5b5f3b34aeb3afb48f8a84e26db159b86de94`)
and is superseded-before-F1 for incomplete resume/object authentication. F0-v2
also remains byte-identical at
`results/baseline/g8_f/f0_v2_execution_authorization.json`
(`g8ff0v2auth-dbcac1f4dcf76238a4222629e590372004f5dad3e4fb1316e28b6fd0b93c6f31`,
file SHA-256 `b14691ca26b6086d9b8e08b563027047cdba114b438311208fe6d413f5c29ce9`)
and is superseded-before-F1 with zero coverage solely because the owner selected
`confessor_pascal_cu126` before the first F1 measurement; no scientific protocol
changed. **Active F0-v3 is Pascal-bound and GREEN** at
`results/baseline/g8_f/f0_v3_execution_authorization.json`
(`g8ff0v3auth-e261cd53d3bb9fdee1cdde0778f36c2a686e17507b660ff8ec42891bde102497`,
file SHA-256 `391cd81553ed2de869ddf3ad1f0a401523781289342eefaddc7ad27cb005517e`),
binding source `6f06aa81ae2d624bae0d406904982f3a61278d93`, profile
`confessor_pascal_cu126`, Torch `cuda:0` TITAN Xp UUID
`GPU-46acd0f2-2ff5-1a43-cac9-2ae20e56dc9a`, Pascal lock SHA `d3561c8e…`, and
the exact AM-88 assignment. Separate owner F1 launch authorization
`g8ff1launch-a88fc23774b38763858e2fec717bf27f0f79893bb6b768708c0f3d38a570ee74`
is frozen at `results/baseline/g8_f/f1_launch_authorization.json`; it permitted
only F1 JPEG2000 corpus materialization/resume. **F1 is GREEN and CLOSED.** The
stopped `confessor` writer exited 0 after exact prefix 50,814/50,814; every
request, result and referenced object authenticates. The frozen outcome mix is
44,039 materialized artifacts / 6,775 typed image-codec infeasibilities / 0
other, with no orphan, hole, foreign or duplicate assignment and no resampling.
Compact evidence is `results/baseline/g8_f/f1_completion.json`
(`g8ff1completion-b5bb834a…`, file SHA-256 `d4f9d44a…`) plus the exact
50,814-row `f1_corpus_manifest.csv` (`g8fcorpus-adeae507…`, SHA-256
`792cce92…`); the 4.1-GiB worker corpus remains immutable at
`/home/nick/projects/capstone/results/baseline/g8_f/runtime`. The transient
Discord HOLD alerts were non-atomic read-only sampling races; final authenticated
closeout was delivered and F1 polling disabled without deleting webhook
configuration. The owner-authorized F2/BR-12 artifact-classifier fine-tune is **GREEN and
CLOSED**. The stopped `confessor` worker exited 0 after 20/20 epochs and exactly
6,900 optimizer steps over all 44,039 materialized F1 assignment rows per epoch;
6,775 typed omissions were not trained, and the 42,932 unique reconstructions
were not used to deduplicate assignment multiplicity. The preregistered
validation-top-1/max/earliest-tie rule selected zero-based epoch 17 at 890/1000.
Completion `g8ff2completion-659cf7d…` and classifier freeze
`g8fclassifierfreeze-fb8a410f…` bind selected checkpoint
`468710ba5e6426d2daeaba50af331b498d5d079726476538d69e2fd3b6355ca1`;
the checkpoint is durably hosted through the existing GitHub Release policy and
loads strictly. F2 checkpoint-selection validation inference is complete as
authorized (20,000 images). **G8 is GREEN and CLOSED.** F3 authenticated and
re-scored exactly 288,000 historical validation rows (264,000 delivered and
24,000 frozen outage) without re-encoding; BR-4 pass two executed exactly once
(18 calls, 8,190 candidate evaluations, 378/378 cells, 95 ties) and changed 162
selections. AM-90 freezes efficiency `r_1_24`, crossover/headline `r_1_6`, and
low ratio `r_1_24`; BR-16 freezes QAM16/rate-1/2/axis-160 at 7 dB and the H2
3–7 dB window (79 pp classical point drop). Terminal closeout is
`results/baseline/g8/g8_closeout.json` (`g8closeout-07526958…`). Pass three,
fallback training, scientific learned training and test access remain zero. Do
not rerun F2/F3/pass two or add optimizer steps. **W5 is GREEN and CLOSED as a
training-infrastructure gate only, including its additive GradScaler accounting
repair.** AM-91 remains unchanged. The repaired trainer classifies finiteness
across every optimizer-owned gradient; targeted Inf/NaN regressions cover shared
decoder ingress outside the three named gradient regions. Historical attempts
1–3 and completion `w5completion-680b2688…` remain byte-identical; attempt 3's
four steps are described only as historically recorded. Successor source epoch
`w5source-af58f018…` and schema-v1 exact non-scientific
`local_4060_cu130` attempt-4 projection `w5smoke-3d77765c…` (raw output
`w5smoke-2fbae18d…` preserved) prove four genuinely applied W5-only steps, exact fresh-
process kill/resume and both selected Imagenette ratio gradients. Additive
closeout is `w5repaircompletion-8b2fa917…`. No lambda, architecture, checkpoint
or validation result was scientifically selected. W6 is GREEN/CLOSED at terminal
completion `w6completion-d7df2e37…`; W7/G-4, W8 and test remain unopened. A
sole-writer OS lock is still required before any later scientific W7 optimizer
step. Do not begin W7/G-4 without separate owner authorization.

<!-- capstone-current-pascal-state: execution=complete; coverage=3213/3213; evidence=published; next=w7-g4-separate-owner-authorization; bler_table=frozen; g8_d=d7-complete; g8_e_e2e4=complete-verified; g8_e_e5e7=complete-green-pass-one-frozen; readiness_state=w5-green-repaired-infrastructure-only-test-sealed; runtime_state=w5-attempt4-closed-prior-attempts-preserved; rerun=forbidden; old_local=immutable-zero-successor-coverage -->

**Current compute model:** two independently authenticated production execution
profiles exist: `local_4060_cu130` and the qualified
`confessor_pascal_cu126`. Neither replaces the other. Every new scientific run
or campaign selects exactly one eligible profile before its first measurement,
freezes it in provenance, and keeps that profile for the complete run. A host
cannot be changed opportunistically; an interruption requires explicit
supersession or a new run. The selected host is the sole writer. The ordinary
publication handoff is verify → reconcile → commit → authenticated GitHub HTTPS
push → fetch/parity. **AM-86 adds one narrow cadence exception for the
owner-authorized Pascal G8_C successor:** authenticated per-unit evidence may
accumulate continuously in its separate mutable runtime on `confessor`, with
Git publication after the unattended campaign or at an owner-selected manual
checkpoint and prepublication loss accepted as an owner custody risk. Final
verification plus complete commit/push/parity remain mandatory before table
freeze or G8_D release. Commit signing is optional prospectively; hashes,
contracts, durable commits, push success and parity remain mandatory at every
actual publication. Historical signing requirements and incidents remain
historical facts.

**Current G8_C migration:** the old local RTX4060/cu130 campaign is preserved as
valid superseded history (748 accepted units plus its request-only trailing
attempt) and contributes zero successor `BlerTable` coverage. The clean
`confessor_pascal_cu126` successor campaign
`g8p-1da44d1fecf684375a0055624abc3c554ecdaf3875b41ee1a13f603f9abe2eca`
completed 3,213/3,213 identities at 5,000 trials each, using source commit
`426110b05161e73e4d819bdc01f4857c012d6d59` and production-contract SHA-256
`dcb2446d9b7974edb87b00c73691589f5cca49ae50806583097126269e07031b`. The
canonical runtime was imported at
`results/baseline/g8_pascal_successor/runtime/` and published evidence contains
3,215 requests, 3,215 results and 3,213 states. The two extra request/result
pairs are immutable attempt-1 retry history for ordinals 0 and 1; each failed
before measurement with zero trials because of the already-repaired
nested-request adapter defect, while attempt 2 supplied the accepted 5,000-trial
result. Failed final units, terminal-invalid units and unresolved units are all
zero; the intended total is 16,065,000 trials; protected counters
`inference=0`, `training=0`, `validation_decoding=0`, `test_access=0`; and
`old_result_ingest=false`. Do not remove or normalize the retry records.

The top-level
`results/baseline/g8_pascal_successor/campaign_state.json` is immutable
zero-coverage readiness history and must remain 0/3213. The separate
`results/baseline/g8_pascal_successor/runtime/campaign_state.json` is the
completed production aggregate and is authoritative for the finished execution.
No G8_C Pascal worker may be started; do not rerun that completed campaign,
resume the old RTX4060 suffix, ingest predecessor results, or alter completed
runtime evidence. This prohibition does not block the separately authorized
G8_F/F1 sole writer bound to active Pascal F0-v3.
G8_C remains green and closed; G8_D D0, D1, D2, D3, D4, D5, D6 and D7 are complete.
G8_E E0 and corrected-v3 E1 are complete with zero validation coverage at their freeze times. The partial local_4060_cu130 E2 campaign was owner-aborted at exact prefix 47409/288000 (`PARTIAL_OWNER_ABORTED_PROFILE_RELOCATION`, evidence preserved, locally excluded runtime). The additive worker-successor epoch `results/baseline/g8_e/e2_confessor_successor/` is frozen (contract `g8econtractcorrectedv3s-2831f47d…`, campaign `g8e-v3s-85354d3d…`, source commit `ed0b92a3…`) and was authorized E2–E4-only (commit `493d656`); production E2 completed at exactly 288000/288000 on `confessor` (`confessor_pascal_cu126`, `cuda:0` TITAN Xp) and E3/E4 are complete and verified (closeout provenance `g8ecloseout-1422804d…`; corrected entry points `tools/closeout_g8_e_v3s.py`). G8_E is GREEN and closed through E7 with pass one executed exactly once and frozen; do not rerun it or widen its scope. AM-87 freezes the complete G8_F quality support, AM-88 freezes the balanced six-variant-per-image sampler, Pascal F0-v3 is GREEN/frozen, and the separate F1 owner launch authorization is frozen. F1 is closed and must not rerun. The AM-89 F2 artifact-classifier run is GREEN, authenticated and frozen after exactly 20 epochs/6,900 optimizer steps; zero-based epoch 17 is the preregistered best checkpoint at 0.890 validation top-1. F2/F3 and pass two are closed; G8 validation-side adjudication is frozen. Learned-system training and test remain closed pending separate owner authorization. C3-C7 closeout is now bound by
`results/baseline/g8_pascal_successor/successor_bler_merge_report.json`,
`successor_bler_table.json` and `successor_closeout_provenance.json`: 153 curves,
3,213 measured points and 16,065,000 trials. The merge ID is
`g8pmerge-2e861c39d8981af0e2d57dc8ded5828b9ed56a1459491e04929b5e9c3418de89`,
the table ID is
`g8pblertable-69ecc729f3b7dc3d67c0a3a5d8cf071cab927ad0a1e0cd5b18a6bbe674b9126f`,
and the closeout source/provenance closure is independently verified. G8_D D0,
D1, D2, D3, D4, D5, D6 and D7 are complete. The first G8_E E0/E1 freeze is
preserved as `superseded-before-data` history; the additive corrected E1 epoch
is frozen as an executable pre-data contract with zero validation coverage and
E2 still awaits owner authorization. The full validation campaign remains
closed.

**2026-08-18 clean-checkout verification repair:** the G8_C measurements and
frozen table remain scientifically unchanged. The defect was an
implementation/clean-checkout provenance-verifier defect: the legacy
`runtime_tree_sha256` includes Git-lossy modes and local coordination paths.
`results/baseline/g8_pascal_successor/portable_scientific_runtime_manifest.json`
now authenticates only the scientific runtime namespace by canonical relative
path, byte length and exact-byte SHA-256; the strict loader independently
reconstructs all 3,213 units and curves from those bytes. The legacy tree hash
`dde5a45a2c58320b9b28e13afa459a8cbf2db1614939ad8ff790d42edc27f14b` remains
immutable historical production/import provenance and is not a portable Git
identity. G8_D was reverified and non-scientifically rebound under
`results/baseline/g8_d/portable_rebind_provenance.json`; no campaign rerun or
validation measurement occurred. The first G8_E E0/E1 freeze is now historical
`superseded-before-data`; the corrected E1 epoch is frozen pre-data and E2 is
still unopened.

**First Review delivery contract — user-fixed; do not ask the user to restate or
reinterpret it.** The maintained acceptance checklist is
[`deliverables/review-1/first-review-package.md`](deliverables/review-1/first-review-package.md).
Review 1 requires: a polished approximately 10–12-slide PPT covering all six
rubric categories; the ≥25-reference literature review; the corrected Gantt;
architecture/methodology readiness across all four team members; G-1 and
existing implementation evidence ready for viva; the deployment dossier plus a
dated guide acknowledgement of the simulation-first Tier-1/no-required-hardware
path; exact corrected H1 wording and the 18–22 August 2026 dates; the final
package under `deliverables/review-1/`; and an annotated `review-1-basis`
snapshot cut only from that final review basis. Backing documents do not by
themselves complete the package.

**First Review scope boundary.** The PPT is the main artifact and MUST expose
the six-criterion mapping; a generic project deck is insufficient. PR-1 must be
synthesis, not a bibliography dump. The Gantt must show unfinished later work
honestly. Four-member technical readiness and the guide response are human
facts: agents may prepare notes, viva questions and the exact
[`PENDING` acknowledgement record](deliverables/review-1/guide-hardware-alternative-acknowledgement.md),
but MUST NOT fabricate either. Current valid evidence is sufficient; G8_C
completion, the final BLER table, neural training, learned-versus-classical
results, demo, thesis chapters, paper, poster, plagiarism report, hardware
purchase and SDR implementation are **not** First Review prerequisites. NEVER
rerun, weaken, bypass or reinterpret science to produce more review graphs, and
NEVER rewrite provenance history to clean it up. The evidence-based readiness
matrix at the end of the package is the final gate.

**Historical final G8_B handoff:** the prior live cursor was `G8_B/tooling_smoke_complete` with campaign-state
SHA-256 `09f1655f570fe947f93bf2477b7bb3b3a7e871c32a98addde0b9d1e7b3400a77`. The seven registered
artifacts are unchanged except for the authenticated v3 runner binding
(`g8runner-49e4facbe266117c74ff802b4252bcba87a7331c34a7ffe228b4648469728583`,
`a238478eb9f231c984258e1f99c4c54d1a0fba353faa683908bca460c6c03763`, 18,612 bytes) and the
v3-bound schema-2 smoke (`1f574b8d8442b68c44211693d128f768bc17f0bfa5472ae144d6c9e5b8ef6635`,
36,572 bytes). Completed IDs are empty, the in-progress ID is null, all four counters are zero,
the production runtime root is absent, and no full-strength work, selection, authorization,
inference, training, validation decoding or test access occurred. **No specification amendment.**

B1 remains a historical design checkpoint. B1C is the pre-execution correction: tooling, request and result schemas are version 2 and supersede B1 version 1 for all future G8 work; no version-1 request or result exists. B2 and later must use the corrected contract, and no production G8_B or G8_C source may use a stale version-1 builder or validator.

B2 remains a historical implementation checkpoint; **B2C** is its pre-execution correction and is what B3 must build on. Unit-state schema version 2 and state-contract schema version 2 supersede B2's version 1, and the B2C state contract explicitly supersedes contract `g8state-77ff4556…` (9,390 bytes). Every unit state now binds the registered B2C state-contract ID and SHA-256 as well as the B1C tooling contract, so a state written under the superseded contract is rejected rather than silently accepted. First publication is crash-atomic — bytes are staged in a unique same-directory file and published with a descriptor-relative no-follow hard link that cannot replace — and replacement is linearizable inside one exclusive per-unit `flock` critical section, so two writers holding the same predecessor digest can never both succeed. A `result_linked` state is request-bound and terminal: only exact canonical-byte idempotence is permitted, and the one legal resharding path is a clean claim on exactly the next attempt. Because no unit-state file has ever existed, no per-unit migration was needed or written. **No production code may reintroduce a direct write to a final state pathname, an unlocked read-then-`os.replace()` described as compare-and-swap, an `exists() and is_symlink()` guard, or a swallowed directory-fsync failure.**

**W4 progress: PA, PB_1 (including PB_1C), PB_2 (including PB_2C) and PB_3 are all complete.**

**PB_3 built the BR-4 selection machinery and deliberately did not run it.** `src/baseline/classical/composition.py` implements AM-51's analytic composition — `P(TB success)` as the product over code blocks of `1 - BLER_r`, expected accuracy as `P × acc_clean + (1 - P) × acc_outage` — with **both** accuracy terms as types carrying counts and provenance rather than floats. That is not decoration: AM-58 forbids assuming `1 / n_classes` for the outage term, and on this repository's exactly stratified manifest the assumption and the measurement are *both* `0.1`, so a float comparison could never tell them apart. The BLER lookup is keyed on the complete physical-layer identity — the eight fields of `params.baseline.ldpc_bler_reference_must_match` **plus the code rate**, which the committed evidence fixes and the spec's list omits — and is bound to the SHA-256 `g2_adjudication.json` records for `bler_results.csv`. The committed G-2 evidence characterises exactly one configuration (`K=128, N=256, BG2, Z=22, rate 1/2, offset-min-sum, offset 0.5, 50 iterations`) at four SNR points per modulation; everything else returns an explicit `uncharacterized` verdict whose BLER is `None` and **never `0.0`**, and an uncharacterized candidate is ineligible rather than low-scoring. Interpolation is confined to the measured span and to the representation `bler_reference.json` declares. **The full-sweep guard is a safety boundary, not a convenience:** `select_operating_points()` refuses more than 64 candidates, 25 samples per cell or 512 combined cells unless an explicit typed `G8Authorization` is passed, there is no environment variable and no default-true flag, and no tracked non-test file constructs one — each absence asserted by a test. `results/baseline/w4/integration_adjudication.json`, generated by `tools/gen_w4_integration_adjudication.py` and verified by `tools/verify_w4_baseline_integration.py`, closes W4 and records that G-8 remains unresolved. PB_3 needed no amendment.

**PB_3C corrected two defects in that machinery and needed no amendment either.** The `classical_fixed_mod` curve *searched* for its modulation — enumerating every modulation in the grid and keeping whichever summed highest — where BR-9 makes `params.baseline.core_modulation` the definition of that curve; it is now read (`qpsk`), never chosen. A configured modulation with no candidate at a required SNR raises and names the SNR, while one whose candidates are all infeasible or uncharacterized is **preserved** as a curve point with `selected = None`, because that is the cell G-8's completeness preflight has to be able to refuse. Separately, resumed campaign state was trusted: `run_pass()` enforced seven invariants and `_admit_resumed()` four, so the crash-recovery path accepted pass two without pass one, reversed sequences, duplicated scorers and malformed stored selections. Resumed state must now be an **exact ordered prefix** of `selection_passes()`, validated in the order supplied and never sorted, with both paths sharing one set of helpers. The tie-break order is **unchanged** but is now frozen before G-8 and fingerprinted as `selection_policy_sha256`, which a future G-8 campaign manifest must bind along with the adjudication's own SHA-256; the verifier recomputes that digest independently. The adjudication `schema_version` moved 1 → 2 because the verifier now requires fields version 1 never emitted.

**PB_3C provenance, corrected without rewriting history.** Its terminal handoff is `39c43e327573f33011c561c6de22bd05ff93c068`, whose actual subject is `fix: fix push failure due to gpg for resume.md`; do not attribute the intended scientific subject to that SHA. The implementation/adjudication checkpoint is `08dd358c0f1bd55c70152af900f2932f50d95d19`; PB_3's implementation green is `32edbbb58983e54103b2f252c4d8d8f30aa2378e`; the latest scientific-measurement green remains PB_2C `3324393a3e1692478bba8cf1020708bf52947f6d`.

PB_2C corrected eight defects the first W4 verifier could not see, because it checked consistency *between committed artifacts* rather than whether each artifact described the cell it claimed to describe: one 18 dB `config_hash` reused for every cell (with modulation, LDPC rate and encode axis absent from the fingerprint entirely), a null `noise_id` on infeasible rows that made them unpairable with a transmitting arm, BR-11 byte columns averaged over delivered rows only, a `Psot` offset six bytes early, a row timer that excluded classifier inference, a summary wall clock that ignored pre-resume rows, and an OpenJPEG preflight that ran after the results directory existed. **Every scientific outcome survived unchanged.** Three amendments landed: **AM-80** cuts `params.baseline.downsample_axis_px.cifar10` to the single native `[32]` rung, because a flat `j2k_resolutions` of 6 needs every tile dimension ≥ 32 px and OpenJPEG rejects 24 px and 16 px before rate control; **AM-81** defines the BR-11 `header_bytes`/`payload_bytes` columns arithmetically and aggregates them over every emitted codestream (delivered *and* decode-failure), bumping `params.config.analysis_version` to 2; **AM-82** binds the completed transparency-bitrate probe's codec configuration as history, because AM-80 moves the content-addressed codec configuration hash and therefore **every J2K cache key**. Each bounded cell now has its own archived `RunConfig` under `results/baseline/w4/run_configs/<config_hash>.json`, and `results/baseline/w4/overhead_table.json` finally discharges BR-11's long-outstanding archived overhead table. `instructions/PB_2C.txt` is the durable instruction (its §18 addendum is binding) and `instructions/RESUME.md` is the operational cursor.

**PB_2 added the outage policy, the record layer and the bounded W4 evidence.**
`src/baseline/classical/outage.py` freezes the constant-class outage prediction by counting labels
across the *entire* committed Imagenette-160 validation manifest — `data/manifests.py` enforces an
exactly stratified validation split, so all ten classes tie at 100 of 1000 and the configured
lowest-index tie-break selects **class 0** at a measured `100/1000 = 0.1`. That equals `1/n_classes`,
so **never compare the float**: the artifact carries numerator, denominator and the full count
vector, and both `policy_from_record` and the verifier re-derive the selection from counts.
`src/baseline/classical/records.py` emits rows conforming exactly to `params.artifacts.csv_schema`
and `params.artifacts.per_image_schema`, read at runtime — note that `instructions/PB_2.txt` calls
these `analysis.*`, which is stale shorthand for a parameter root that does not exist. The system
value is `classical_fixed_mcs`, not `classical_adaptive`, until PB_3 builds and verifies adaptation.

**The frozen G-1 checkpoint is an Imagenette-160 classifier and must never score another dataset.**
CIFAR-10 has ten class indices too, but they are a different vocabulary, so a CIFAR "accuracy" from
this model would be meaningless. `records.score_result`, the runner and
`tools/verify_w4_baseline_integration.py` all fail closed on it; CIFAR-10 stays a transport,
verdict, accounting and cache plumbing smoke with no task score.

**W4 evidence binds its execution sources and has no re-adjudication escape hatch.**
`results/baseline/w4/execution_source_manifest.json` binds 40 sources at the clean runner-ready
commit. Unlike G-2, a changed runtime source is answered by **rerunning** the bounded run — it takes
about 45 seconds — never by recording an exception. `per_image.csv` and `aggregate.csv` are
deliberately not bound there, because they are outputs of the execution commit; they are bound by
SHA-256 inside `smoke_summary.json`, which the verifier recomputes from disk.

**`src/data/test_access.py` is the sole guarded boundary to the test split and nothing else may import it** (SR-22, DEC-12). That rule is enforced by an AST-walking test, not by convention, and it is the reason the module exists as its own file. Test access releases at `params.evaluation.test_access_gate` — **G-12, W11** — not G-10, which AM-60 caught pointing three weeks early.

`spec/SPEC.md` is the normative description of what gets built and how it will be judged. Read it before writing any code: it fixes the bandwidth budgets, SNR grids, baseline fairness rules, and the preregistered hypotheses. Settled decisions are `DEC-1`..`DEC-16` in §3; requirements carry stable IDs (`SR`/`BR`/`ER`/`DR`/`HR`/`PR`/`OPT`/`FW`) that code and commit messages should cite. Retired IDs live in §14 and are never reused. **Changes to the spec are recorded, not made silently:** every amendment gets an `AM` entry in §17 saying what changed and why, and the amended item carries an `(AM-n)` back-reference — see the amendment convention below.

**Start here each session: [`NEXT.md`](NEXT.md).** It is the inter-session hand-off file — what to do
next, open questions, and recently-settled things that must not be reopened. It is scrappy and
non-normative by design (`spec/SPEC.md` wins on any conflict), and it is **expected to be updated
before a session ends** if the state changed. Promote anything durable out of it: decisions become a
`DEC` in `SPEC.md` §3, risks and provisional values go to `SPEC.md` §16, explanations go to `docs/`.

**Where the spec stands.** It now carries 201 requirements (2 retired), of which 91 are `AM` amendment records. **AM-77 makes dataset provenance and pre-freeze manifest construction executable:** exact archive length/SHA-256 pins, dataset-specific source-payload and authoritative-class rules, canonical CSV bytes, and a provenance-only published-test scan that is forbidden from decoding or canonicalizing. **AM-78 fixes deterministic, resumable reference-classifier training details without changing its scientific recipe.** **AM-79 freezes G-2's complete-asset golden-vector checksum, independent BLER reference and progressive-packetisation design.** **Round 16 (AM-80..AM-82) is the W4 PB_2C corrective repair:** CIFAR-10's codec axes cut to `[32]`, the BR-11 byte semantics defined arithmetically with `analysis_version` bumped to 2, and the transparency-probe codec-configuration binding recorded as history behind one byte-pinned drift record. **AM-86 narrows only the Pascal successor's Git-publication cadence while preserving sole-writer evidence authentication and mandatory final publication/parity. AM-87 discloses and repairs BR-12's post-pass-one/pre-G8F support-breadth preregistration defect with a complete authority-projected metadata-only plan. AM-88 preserves that support and freezes a deterministic balanced six-variant-per-training-image sampler, reducing nominal attempts 20-fold; neither amendment itself authorized execution. AM-89 completes the exact F2 initialization/training/validation/resume recipe before optimizer step 1 under the owner's separate F2-only authorization.** AM-71 remains the stable-source-byte identity clarification, and AM-72..AM-76 remain the implemented-contract remediation. The adjudicated EXT-6 findings and their arithmetic remain recorded in §17; do not reopen them without new evidence. W0 is done; G-9, G-1, G-7 and G-2 passed; W1, W2 and W3 are complete. The validation-only transparency-bitrate probe is lineage-bound, remotely reproducible and scientifically unchanged. Bounded W4 baseline integration is complete through PB_2/PB_2C; successor-specific G8_C C3-C7 closeout is complete with the successor `BlerTable` frozen, and G8_D D0, D1, D2, D3, D4, D5, D6 and D7 are complete and GREEN; corrected-v2 G8_E E1 is superseded-before-data history and corrected-v3 E1 is the current executable pre-data contract with zero coverage; the partial local_4060_cu130 campaign was owner-aborted at 47409/288000 and the worker-successor epoch completed verified E2–E4 on `confessor_pascal_cu126`, E5 pass one completed exactly once, AM-88's G8_F balanced sampler plan is frozen, active Pascal F0-v3 is GREEN, and the separate F1-only owner launch is frozen.

### Commands

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # PyYAML only, for the spec tooling
python tools/gen_spec_views.py            # regenerate the derived spec views
python tools/gen_spec_views.py --check    # validate spec + fail on stale generated files
python tools/check_doc_consistency.py     # hand-written docs vs the spec; -v lists what passed
python tools/check_literals.py            # SR-1 numeric-literal lint; -v lists scanned files
python spec/evidence/check_packetisation.py            # TS 38.212 conformance, no GPU/network, <1s
python spec/evidence/check_packetisation.py --json spec/evidence/packetisation_record.json
.venv/bin/python tools/verify_cpu_lock.py --clean-install
.venv/bin/python tools/fetch_datasets.py                 # verify pinned byte length/SHA-256, then extract
.venv/bin/python tools/fetch_datasets.py --check         # network-free archive provenance verification
.venv/bin/python tools/materialize_manifests.py
.venv/bin/python tools/materialize_manifests.py --check  # regenerate in memory; compare exact committed bytes
.venv/bin/python tools/verify_datasets.py                 # real train/val smoke + zero-call test-provenance audit
.venv/bin/python tools/train_reference_classifier.py --config configs/reference-classifier-clean.yaml --dataset imagenette160 --device cuda --smoke-steps 3 --smoke-val-batches 2  # bounded, ignored smoke only
.venv/bin/python tools/train_reference_classifier.py --config configs/reference-classifier-clean.yaml --dataset imagenette160 --device cuda --full-run  # production G-1 campaign; completed 2026-07-29
.venv/bin/python tools/verify_g1_adjudication.py       # network-free frozen G-1 cross-check; hashes local checkpoint when present
.venv/bin/python tools/profile_djscc_g7.py             # requires a clean --git-repo worktree at the configured implementation commit; see W2 worklog
.venv/bin/python tools/verify_g7_profile.py            # network-free frozen G-7 config/commit/metric/gate cross-check
.venv/bin/python tools/verify_transparency_bitrate_probe.py  # validation-only J2K probe evidence
.venv/bin/python tools/fetch_ldpc_golden_vectors.py     # materialize the ignored rung-2 LDPC fixture; run before pytest
.venv/bin/python tools/gen_g2_source_manifest.py        # regenerate the G-2 execution-source manifest from the measurement commit
.venv/bin/python tools/gen_g2_source_manifest.py --check # regenerate in memory; compare exact committed bytes
.venv/bin/python tools/verify_g2_adjudication.py        # network-free frozen G-2 evidence, source-provenance and lineage cross-check
.venv/bin/python tools/gen_w4_outage_policy.py           # freeze the constant-class outage artifact
.venv/bin/python tools/gen_w4_outage_policy.py --check   # re-derive the selection and compare
.venv/bin/python tools/run_classical_baseline_w4_smoke.py --restart  # bounded W4 run, ~45 s, crash-resumable
.venv/bin/python tools/gen_w4_source_manifest.py         # bind the bounded evidence to its execution commit
.venv/bin/python tools/gen_w4_source_manifest.py --check
.venv/bin/python tools/gen_w4_integration_adjudication.py        # regenerate the W4 closing adjudication
.venv/bin/python tools/gen_w4_integration_adjudication.py --check
.venv/bin/python tools/verify_w4_baseline_integration.py # network-free bounded W4 evidence + BR-4 selection-machinery cross-check
.venv/bin/python tools/gen_g8_e_e7_handoff.py --check    # authenticate deterministic E7 handoff and incident-audit binding
.venv/bin/python tools/verify_g8_e_complete.py           # terminal G8_E verifier including E7 handoff
.venv/bin/python -m pytest              # project test suite; config is in pyproject.toml
```

**On a fresh clone, run the fetch line before `pytest`.** `tests/fixtures/ldpc_ts38212_golden.npz` is
git-ignored on purpose (`.gitignore:157`, AM-25 — third-party vector bytes are never committed, only
their checksums and a fetcher), and `tests/test_ldpc.py::test_srsran_encoder_and_rate_matched_fixture_exact`
hard-asserts the file exists rather than skipping, so an unmaterialized clone fails the suite for a
provenance reason rather than a scientific one. `fetch_ldpc_golden_vectors.py` authenticates the
complete release asset and every pinned inner vector archive against `params.baseline`, is a
network-free no-op once the fixture is present (`--force` re-materializes), and never writes
third-party bytes anywhere tracked. The project-owned offline-floor test immediately above the srsRAN
test is deliberately ungated and needs no network.

**AM-77 provenance pins:** Imagenette-160 `imagenette2-160.tgz`, 99,003,388 bytes,
SHA-256 `64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5`; STL-10
`stl10_binary.tar.gz`, 2,640,397,119 bytes,
`f31fd99273a1acb8609c8db427cebb1de3f71de77758cdc0e22956e1289b9866`; CIFAR-10
`cifar-10-python.tar.gz`, 170,498,071 bytes,
`6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`. Manifest pins:
`data/manifests/imagenette160.csv` → `224309422f15bf89460559381aea4b00c4779c52d3652f7f679a213369f3f889`;
`data/manifests/stl10.csv` → `67936da779dc0010160b37b3b40001490304a5873eb978d261e3a57947387b47`;
`data/manifests/cifar10.csv` → `09e9debf4743831ca61f17154a997e60becdd7046a585bdbd94b5db4bf12a537`.

**GPU check on this machine — it is WSL2, so look for `/dev/dxg`, not `/dev/nvidia*`:**

```bash
ls -l /dev/dxg                                      # WSL GPU device; expect crw-rw-rw- 10,125
/usr/lib/wsl/lib/nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
.venv/bin/python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
```

Measured 2026-07-28: `/dev/dxg` present as `crw-rw-rw- root 10,125`, `NVIDIA GeForce RTX 4060 Laptop GPU`, driver `592.82`, and torch `13.0 True` with a real device matmul succeeding. **`nvidia-smi` reports `CUDA Version: 13.1` while torch is built for `13.0` — that is normal minor-version compatibility, not a mismatch; do not "fix" it by moving the pin.** An agent that probes `/dev/nvidia*` will wrongly conclude there is no GPU, and the `nvidia-smi` on `PATH` is not always the same binary as the one under `/usr/lib/wsl/lib/`.

Device access and network access are **separate** permissions, and only the **third** command settles the one that matters: a visible `/dev/dxg` and a working `nvidia-smi` do not guarantee torch can initialise CUDA, since that additionally needs `libcuda` resolvable through the WSL shim. Check both before assuming which W1 steps you can complete.

Note: `uv run <script>` warns *"No `requires-python` value found in the workspace"*. That is expected and must not be "fixed" — `pyproject.toml` deliberately carries pytest configuration only, with no `[project]` table, so there is no packaging and no install step (SR-21 owns dependencies via the lockfiles). Run project code with `.venv/bin/python`, not `uv run`.

`pytest` needs the **runtime** environment (below), not just the spec tooling. Its config lives in `pyproject.toml`, which sets `pythonpath = ["src", "tools"]` — that is why there is no install step and no packaging. The suite is **not expected to pass on the CPU-only install path**: `tests/test_env.py::test_cuda_build` hard-asserts a CUDA build with no skip marker and no environment-variable escape hatch, deliberately (AM-67), because a variable exported once in a shell profile would silently disarm the only check that catches a CPU build on the machine that trains.

`check_doc_consistency.py` guards what `gen_spec_views.py --check` cannot: the **current hand-written documentation** defined by AM-76. Validly marked historical plans are excluded only with the exact banner and a link resolving to root `NEXT.md`; malformed banners fail explicitly. Its stale-rule table is the tool's memory and is meant to grow.

**G-2 binds its execution sources, not just its outputs.** `results/baseline/g2/execution_source_manifest.json` records the Git blob id, byte SHA-256 and byte length of all 14 sources that participated in the measurement, at commit `968e907237bbe571adf6ec48e4711ea021831719`, each with a role. Only the `runtime` role — `src/baseline/ldpc/` — is asserted byte-identical at HEAD; `measurement_runner`, `configuration` and `record` are bound as history, because `82f6c56` added CLI import bootstrapping to the runners after the campaign and `spec/params.generated.yaml` is regenerated on every amendment. **Do not "fix" a runtime mismatch by regenerating the manifest** — a change under `src/baseline/ldpc/` means the recorded BLER numbers describe a different implementation, and the honest responses are to revert it or to record a real re-adjudication in `readjudications`. The evidence commit is **resolved**, never recorded (`git log -1 --format=%H -- results/baseline/g2/g2_adjudication.json`); the verifier rejects a stored `evidence_commit` because a file cannot contain the hash of the commit that adds it.

**A `readjudications` entry is the only thing that permits a runtime file to differ from the adjudicated bytes, and it is fail-closed (manifest `schema_version` 2).** It must declare a `kind` — `recampaigned` (a new G-2 campaign really ran) or `off_measurement_path` (the changed definitions were provably unreachable from the measurement) — plus a `justification`, a `readjudicated_at`, a non-empty `evidence` list, the `measurement_sha256` it supersedes, and the `current_sha256` it covers. **It is pinned to those exact bytes**, so the next edit to a re-adjudicated file re-raises the HOLD instead of inheriting the old justification, and `verify_g2_adjudication.py` prints `runtime_readjudicated=[...]` so it is never silent. `gen_g2_source_manifest.py` carries committed entries forward verbatim — nothing in Git can regenerate a hand-written judgment. One entry exists today: `src/baseline/ldpc/transport.py`, `off_measurement_path`, recorded at W4/PB_1 because `run_ldpc_g2.py` imports only `build_packet_plan` from that module and that function is byte-identical; see `worklogs/w4-classical-baseline-progress.md`.

`check_packetisation.py` must be **re-run and its record regenerated** after any change to `params.bandwidth`, `params.baseline` or `params.digital_semantic_control` — it asserts byte alignment, `B' % C == 0`, `K = 22Z`/`10Z`, filler accounting, `Σ E_r == G` and the per-block rate floor across all 216 configurations, and the record carries the params and script hashes that produced it. It reported **zero failures while violating four of those rules** before AM-58, which is the reason it now asserts them rather than printing summaries.

Project runtime dependencies are **not** in `requirements.txt`: SR-21 requires a hashed `requirements.lock`, generated at W1 from `requirements.in` by **`uv`** (`params.environment.lock_tool`, decided in AM-61 — do not substitute `pip-tools`) and installed from `params.environment.torch_index_url`.

```bash
uv pip compile requirements.in --generate-hashes --emit-index-url \
    --index-strategy unsafe-best-match -o requirements.lock            # cu130 index
uv pip compile requirements-cpu.in --generate-hashes --emit-index-url \
    --index-strategy unsafe-best-match -o requirements-cpu.lock
uv pip sync requirements.lock --index-strategy unsafe-best-match
.venv/bin/python -c "import torch; assert torch.version.cuda is not None, 'CPU BUILD'"
```

**Both non-default flags are load-bearing and are recorded as `params.environment.lock_index_strategy` and `params.environment.lock_emit_index_url` (AM-66), not just here.** `--index-strategy unsafe-best-match` is needed because the default stops at the first index carrying a package *name*, and PyPI carries `torch` but not the `+cu130` local version. `--emit-index-url` is needed because a lockfile that does not record its indices cannot be installed by anything — including plain `pip` — and the failure names only the version, never the missing index, so it reads like a bad pin.

Three constraints on that block: a bare resolve silently yields the **CPU build**, and the only check that catches it is `torch.version.cuda is not None` rather than a successful import (AM-23); the emitted lockfile MUST stay installable by plain `pip install --require-hashes`, because `uv` is pinned so that *resolution* reproduces, not so the project gains a runtime dependency on it; and the lockfile **covers the spec tooling as well as the runtime stack** (AM-65) — `uv pip sync` makes the environment *exactly* the lockfile, so a runtime-only lock would uninstall PyYAML and break `gen_spec_views.py` and `check_doc_consistency.py`. `requirements.txt` is unchanged and stays the dependency-light bootstrap.

OpenJPEG is not provisioned by either Python lock (AM-75). The reference system command is
`pacman -S openjpeg2`; `openjpeg2 2.5.4-1` was observed on this machine on 2026-07-29 and is dated
evidence only. The normative condition is the loaded version `2.5.4`, checked before any J2K path
creates artifacts; learned-only metadata may record `openjpeg_version: null` when it is unavailable.

Note: system Python has no PyYAML and `pip install` into it is blocked (PEP 668), so the venv is required.

`SPEC.md` is hand-written and authoritative. `spec/DATASHEET.md`, `spec/concerns/*.md` and `spec/params.generated.yaml` are **generated from it** — edit `SPEC.md` and regenerate, never edit the generated files. `--check` also validates the spec itself: unique requirement IDs with live-plus-retired numbering contiguous per prefix, every `params.*` citation resolving, every parameter section cited by some requirement, a `*(verify: ...)*` clause on every `SR`/`BR`/`ER`/`DR`/`HR`/`PR` line, and the `k = ratio × n` symbol-budget arithmetic. Run it after any spec edit; a stale generated file fails the check.

To retire a requirement, strike it through in place (`- ~~**G-3**~~ — reason`) under §14 rather than deleting it: the number stays reserved so live IDs are never renumbered. Passing `--check` means the document is *structurally* consistent — it says nothing about whether the experiment is scientifically valid.

**The amendment convention (§17).** Changing any requirement, decision, parameter or gate means adding an `AM-n` entry to §17 stating what changed, why, and who raised it, and adding an `(AM-n)` reference to the item you changed. Amendments are append-only: an amendment that is later revisited gets a *new* `AM` entry citing the old one, never an edit to the old one. Two formatting traps — `AM` entries must be a **single line** however long (the requirement regex is anchored and single-line, so a wrapped entry silently fails to parse and surfaces only as an ID-contiguity error), and a `` `params.x.y` `` written inside an `AM` entry is parsed as a real citation and must resolve, so an amendment describing a *rename* must quote the old key in plain text rather than in backticked `params.` form. The rationale is in §17's preamble: this document is read by examiners and external reviewers who never saw the previous version.

Training code MUST read `spec/params.generated.yaml` rather than parsing markdown or hard-coding constants (SR-1).

## What the project is

The capstone is **Semantic Communication + AI** — idea #1 in `ideas/Proposals.pdf`, which the user has finalized. The other ideas in that PDF (DisasterMesh, DyslexiaLens, SafeScreen, CodeProof, scam detection, ReproCheck, RL malloc, thermal scheduling, cache eviction) are rejected/parked; don't propose work on them.

**The scope is the "AI overview" section of that idea, not the bullet summary above it.** The state/action/reward/hardware framing in the bullets is earlier exploratory phrasing; the AI overview paragraphs (the idea, why the learned method wins, method, tiered scope, success criterion & demo) are what's actually being built.

Core thesis: transmit the *meaning* needed to accomplish a downstream task rather than every original bit, cutting bandwidth/energy on edge/IoT links. Concretely, **deep joint source-channel coding (DJSCC)**: train a neural encoder (sender) → differentiable noisy-channel model → neural decoder (receiver) end-to-end, optimizing task error *after* channel noise at a fixed bandwidth budget. Proposed narrow task: image → classification over an AWGN channel. This is supervised end-to-end training — **not** reinforcement learning; don't reintroduce an RL framing unless the user asks for one.

### Non-negotiables from the proposal

These are the terms the project is being judged on; preserve them in any design work. `spec/SPEC.md` §1–§2 states them normatively (thesis, then completion criteria plus four preregistered hypotheses); this summary is for orientation and defers to the spec on conflict.

- **Fair baseline.** A properly-tuned source-codec + LDPC pipeline evaluated on the *identical* task, SNR, and bandwidth budget. A strawman baseline invalidates the whole contribution. Note the codec of record is **JPEG 2000, not JPEG** (DEC-9): JPEG's ~250–290 byte container floor is a large or total fraction of the channel budget at these ratios, which would make a low-SNR "cliff" a file-format artifact. JPEG is kept as a labelled secondary curve.
- **Structural (not cosmetic) advantage.** The argument is that the *task-agnostic reconstruction baseline* cannot express a task-success objective, and that Shannon separation is only optimal for infinitely long messages — real IoT links send *short* messages over *noisy* channels, a regime where separation pays finite-blocklength penalties and joint coding *may* gain. Signature behavior: graceful degradation (separated coding hits a cliff and yields nothing; semantic gets blurrier but stays task-correct). Do **not** restate this as "classical coding cannot express task success" in general — a digital system can send features or logits, which is exactly what the ER-9 control does.
- **Success criterion.** See `SPEC.md` §2, which is normative and has been revised. Completion is defined by running the preregistered protocol properly, *not* by the outcome. The primary hypothesis is a **paired** accuracy-difference interval above zero at three consecutive low-SNR points. **A curve crossing is reported if seen but is not required** — at low bandwidth ratios the learned system is expected to dominate everywhere, which supports the thesis rather than failing it. Never reintroduce "the curves must cross" as a pass condition.
- **The baseline adapts; don't "simplify" that away (DEC-16).** Modulation is an *adaptive* axis of BR-4's per-SNR tuning — the baseline climbs from QPSK to 16-QAM as the link cleans up, exactly as deployed radios do. Capping it at QPSK would be an artificial handicap that flattens the classical curve and destroys any possibility of a crossover, so `params.baseline.modulations` and `modulation_tuning` are load-bearing, not decoration. The governing rule for any crossover work: **every lever must strengthen the baseline or be preregistered; never handicap the learned system.** BR-15 requires the resulting adaptation asymmetry (baseline re-tuned per SNR, learned model trained once and frozen) to be disclosed in the methods section and every headline figure caption. `docs/crossover-explained.md` explains the whole thing.
- **Attribution.** A learned-vs-classical gap conflates task-aware representation with joint source-channel coding. ER-9 (quantised learned features over the same LDPC/QPSK chain, matched *k*) is the control that separates them, and is one of the two claimed novelty items (DEC-13).
- **Demo.** Live SNR slider driving both pipelines side-by-side on the same image, with the accuracy-vs-SNR plot updating in real time.
- **Tiered, simulation-first scope.** Tier 1 = the full defensible capstone on a *simulated* channel, built and proven first. Tier 2 (offline SDR replay) and Tier 3 (live Raspberry Pi demo) are **stretch goals with a pre-recorded demonstration as the expected outcome** (DEC-14), not planned deliverables. The project must succeed if Tiers 2–3 never land.
- **LDPC is settled (DEC-10).** Sionna `2.0.1` for base graphs, encoding, rate matching and decoding, behind an adapter seam (BR-14); TB CRC, code-block segmentation, per-block budget distribution and concatenation are written in this project because Sionna does not provide them. **Sionna no longer depends on TensorFlow** — PHY/SYS migrated to PyTorch in 2.0.0 — so the "second DL framework contradicts DEC-3" concern was checked against the release notes and is obsolete. Don't reopen it. `offset_min_sum` is a Sionna built-in, so no custom check-node callable is needed.
- **Graded deliverables are in scope.** Roughly 30 of 100 rubric marks sit on literature review, Gantt chart, standards register, A0 poster, plagiarism report and report format — tracked as `PR-1`..`PR-8`, not as an afterthought.

### On `vault/capstone/Project requirements.md`

**Stale — do not treat as constraints.** That note (RL preferred, MATLAB math component, paper potential, hardware clause) was a filter used while choosing between ideas. It has been superseded; the AI overview in `ideas/Proposals.pdf` governs. Its hardware clause is still loosely reflected in the Tier 2/3 SDR and Raspberry Pi tiers, but nothing in it overrides the tiered scope above.

## Layout

- `NEXT.md` — inter-session hand-off: next steps, open questions, don't-reopen list, session log. Non-normative and frequently rewritten. Read first, update last.
- `ideas/Proposals.pdf` — all candidate ideas with state/action/reward/hardware/baseline/impact breakdowns and blunt verdicts ("BASIC & SOLVED", "UNSOLVED & UNFIT"). The source of truth for project intent.
- `vault/capstone/` — Obsidian vault of course administrivia: proposal report template, thesis format, rubrics, the Fall 2026-27 circular (scanned images, no extractable text), and `Project requirements.md`. Deliverable formats live here; the `.obsidian/` directory is editor config, not content. `Capstone Project Rubrics.xlsx` is the grading scheme and drives the `PR` requirements — extract it by unzipping and reading `xl/sharedStrings.xml`. It scores First Review 10 / Second Review 30 / Third Review 40 / Project Report 20, with **Novelty worth 15** (the line DEC-13 exists to answer) and the review checkpoints at **W4 / W10 / W17** — resolved from the circular's own table, not the spreadsheet's 2023 template dates (AM-59). The circular is scanned images with *no extractable text*, which is not the same as unreadable: render its pages and read them.
- `spec/SPEC.md` — the project specification: thesis, preregistered hypotheses, decisions, parameters, requirements, schedule with go/no-go gates, non-goals, and the §16 open-items register. Normative and self-sufficient.
- `spec/DATASHEET.md`, `spec/concerns/`, `spec/params.generated.yaml` — generated views (see Commands above). `spec/concerns/programme.md` holds the `PR` course deliverables; `spec/concerns/amendments.md` is the §17 amendment record — what changed in the spec and why, the file to read before re-litigating a decision or acting on an external review; the others group `SR`/`BR`/`ER`/`DR`/`HR` by concern, with retired IDs shown under a "Retired" heading.
- `spec/evidence/` — supporting material for measured claims in the spec, currently the W0 LDPC spike behind AM-24 and AM-25: the machine-readable spike record, the scripts that produced it, and the golden-vector cross-check with its log. Not normative; it exists so the spec's numbers can be checked rather than trusted. Third-party vector data is **not** committed there — the directory carries checksums and a fetcher instead (AM-25), and `.gitignore` enforces it. Read its `README.md` before adding anything.
- `tools/gen_spec_views.py` — the generator and spec validator.
- `src/` — project code. `config/params.py` is the SR-1 loader every other module reads its constants through; `env.py` holds the CUDA assertion, the determinism settings and the run-metadata record (SR-21, SR-12). `data/adapters.py`, `identity.py`, `manifests.py`, `provenance.py` and `registry.py` implement AM-77; `data/classifier.py`, `models/reference_classifier.py` and `training/reference_classifier.py` implement AM-78's deterministic pre-G-1 classifier contract. `baseline/classical/` is the W4 classical arm: `composition.py` (the BR-4 analytic selection machinery — the composition, the fail-closed BLER lookup, the feasibility cache, the three system modes, the two-pass limit and the full-sweep guard; built at PB_3 and **not** executed), `channel_transport.py` (bits → interleaving → mapping → the shared AWGN under keyed noise → demapping → LDPC decode, with exact bit reconciliation, realised symbol energy and PAPR) and `pipeline.py` (the whole segment, returning one of four verdicts — `structural_infeasibility`, `codec_infeasibility`, `decode_failure`, `delivered` — and never skipping a case). It builds the channel through `channels.registry` and refuses anything that is not `channels.awgn.AWGN`, because a second implementation would silently break every paired comparison. The classifier normalizes inside the model, initializes from the keyed `init` identity, and uses a keyed seed/epoch permutation. Nothing here is a package: `pyproject.toml` puts `src` on `pythonpath` for pytest, so there is no install step.
- `data/manifests/` — the only tracked part of root `data/`: `imagenette160.csv`, `stl10.csv` and `cifar10.csv`. Downloaded archives, verified extractions and range/cache files remain ignored.
- `tests/` — the test modules plus `conftest.py`, run with `.venv/bin/python -m pytest`. Module and test counts are deliberately not recorded here: they changed with almost every checkpoint and a fixed number in this file was stale more often than it was right. For the current figures run `.venv/bin/python -m pytest --collect-only -q | tail -1`. Several modules exist because a comment would not have caught what they catch: `test_env.py` hard-asserts the CUDA build and OpenJPEG boundary; `test_cpu_lock.py` rejects CUDA distributions structurally; `test_doc_consistency.py` mutation-tests stale values, historical-plan banners and `NEXT.md`'s current-phase agreement; `test_artifact_rng.py` proves exact-key and control-flow invariance; `test_test_access.py` walks the import graph around the guarded test split; `test_g2_adjudication.py` mutation-tests every G-2 fail-closed class including execution-source binding; `test_classical_interleaver_conformance.py` checks transmit bit order against an independently derived TS 38.212 reference; `test_classical_outage.py` proves the outage accuracy is count-derived rather than a hardcoded `1/n`; `test_w4_verification.py` builds a complete valid evidence directory from scratch so its mutation coverage does not depend on the committed bounded run; `test_classical_composition.py` proves the BR-4 composition's measured inputs are passed through rather than reconstructed, using a deliberately non-stratified outage measurement so a reconstruction from `1/n_classes` disagrees; `test_w4_integration_adjudication.py` mutates the closing adjudication and the live selection-machinery checks one property at a time; and the dataset/manifest/provenance tests use synthetic local sources so the default suite stays network-free.
- `requirements.in` → `requirements.lock`, and `requirements-cpu.in` → `requirements-cpu.lock` — the SR-21 environment locks, hashed and committed. Source files are hand-written from `params.environment`; the locks are generated (see Commands) and must not be hand-edited.
- `docs/` — hand-written background notes, not generated and not normative. `crossover-explained.md` explains why §2's crossover criterion was replaced, in plain language and then technically; it is written to feed the thesis discussion chapter and viva prep, and it is the thing to hand a supervisor who asks why the success criterion changed.

### Reading the planning documents

Neither `pdftotext` nor `pypdf` is installed system-wide, and `pip install` into the system Python is blocked (PEP 668). To read the PDFs and `.docx` files, create a venv in the scratchpad and install `pypdf` there; `.docx` files are zip archives whose text can be pulled from `word/document.xml`.
