# G8_F / BR-12 artifact-corpus breadth protocol repair — 2026-08-23

## Verdict and classification

**A — SCIENTIFIC / PROTOCOL PREREGISTRATION DEFECT.** The defect was discovered
after immutable G8_E pass one and before G8_F F0 authorization, corpus
materialization, artifact-classifier training, pass two or test access. Prior
scientific evidence is not invalidated and no measurement requires
recomputation. The repair is additive AM-87 plus a deterministic metadata-only
corpus plan. It does not authorize execution.

Canonical main at the audit start was
`2099b66bb5b29c72417a89c0f87a1245ec01ff96`. Its current verified E7 handoff is
`g8ee7handoff-1af54fbf248cfa233ea74dc516697f0ca9153f4562798680de5b20d35da0a4d8`
with file SHA-256
`a726a6a433fd42e0b0dcb97f1b12615a44528fee25af55a157f594e393824c49`.
The earlier `g8ee7handoff-77605965…` / `a4f29224…` pair is the version at
commit `2338498` before PR #10's additive incident-audit cleanup; current main's
E7 generator and terminal verifier both reproduce the later pair. No E7 byte is
changed by this repair.

## Proof that the hold was real

The audit covered the normative specification and generated views,
`instructions/G8_F.txt`, the immutable E1 corpus specification and E6 lineage
freeze, pass-one state, candidate authority, logical-to-structural measurement
authority, E4 count-derived objects, G8_D and G8_E JPEG 2000/cache identities,
BR-4 composition/selection, packet budgets, source, tools and tests. AM-6,
AM-54 and AM-59 were traced semantically.

The only frozen corpus rules were:

- `union_of_br4_selected_qualities_at_or_below_train_snr`;
- `train_only`;
- `preregistered_feasible_quality_band_not_pass_one_winners_alone`;
- E1/E6's immutable pass-one-reference lineage and validation/test prohibition.

No source defined an artifact-quality tuple, band axis or width, feasibility
level, candidate projection, ordering, deduplication, PHY alias multiplicity or
per-image infeasibility action. The G8_E physical cache did define a complete
*image-specific* physical key, but no code projected candidates to a corpus
quality. The E1 corpus bytes deliberately defer the G8_F source to F0 and state
only that selected configs are referenced, not recomputed. The phrase was thus
genuinely non-executable.

There is also a concrete inconsistency: pass one selected some low-SNR outage
candidates whose requested JPEG 2000 quality emitted no codestream for any of
the 1,000 validation images. A literal union of selected qualities includes
those requests, while a literal feasible-only band excludes them. No frozen
text resolved that conflict.

## Additive completed-campaign verification compatibility

Regenerating `spec/params.generated.yaml` for AM-87 correctly changed its exact
bytes. The completed Pascal source verifier initially rejected that drift before
loading G8_C, which in turn made the G8_E terminal verifier stop. This was a
fail-closed provenance response, not changed evidence. The generated-parameter
diff was independently reduced to exactly 16
`reference_classifier.artifact_finetune_*` leaves; none is on the completed
G8_C measurement path or completed G8_E validation/pass-one path.

Additive compatibility record
`g8postsource-bdd9e60947a3e8d04bd9203d3c5ec3f861dbbfeae39945bd0f5d6d47fb5e33cf`
(file SHA-256
`e057dc1a3c06200a3b485fd017dce2ff75e93c89aba285e4cb0d7c1827b27782`)
pins the archived/current parameter bytes and the exact post-campaign and
historical-campaign verifier implementations. Both reconstruct the archived
YAML from measurement source commit `426110b…`, compute the leaf diff, permit
exactly those 16 G8_F paths and fail on every other drift. The record also pins
both verifiers' current bytes, so it is not a general allowlist. A second exact source-compatibility record,
`g8esourcecompat-1dca57402923146dc3ac03d5c3d11a497f6158c563b51ab93f2961eb14399beb`
(file SHA-256
`fd556f3f662a1c03d381da44739a5805568933d0fa1a8fb1823ed3387060e29f`),
preserves the already-frozen D7 contract identity when synthetic builders run
under AM-87 and admits exactly that historical builder plus the post-E7 source
verifier; both current and archived source bytes are pinned. No production
contract, source manifest, request, result, state, BLER table, D7 object,
E2/E3/E4 object, pass-one record or E7 handoff byte changed. After this additive verifier repair, `tools/verify_g8_e_complete.py`
again returns the original GREEN report and original E7 ID/SHA.

## Frozen structures that constrain AM-87

The correction is derived from structures fixed before pass one, except where
the original AM-6 rule genuinely requires immutable pass-one output to anchor
scope:

1. candidate authority `g8eauthority-dd09fa9b…`: 12,096 logical candidates,
   ordered by candidate ID;
2. measurement authority `g8emeasurementauthority-819f0a28…`: exact
   logical-to-structural map, packet accounting and 576 structural identities;
3. the pre-pass-one G8_D JPEG 2000 configuration identity
   `g8dcodec-39f14b7e…`, configuration hash `2daf597f…`, OpenJPEG `2.5.4`;
4. the G8_E physical-cache namespace hash `f677bc75…` and runtime `2.5.4`;
5. the E4 count-derived feasibility objects, used only to audit/estimate and
   never to select membership;
6. the frozen G1 checkpoint and Imagenette manifest;
7. AM-6/AM-59's existing headline dataset, at-or-below-training-SNR and
   training-only scope;
8. immutable pass-one authority references, used only to recover the original
   dataset/ratio scope across all three frozen modes. Expected-accuracy values,
   tie rank and margins are not read for membership.

At SNR <= 7 dB, pass one supplies 288 anchor references. Their scope is
Imagenette-160 at all six frozen ratios. The candidate authority then supplies
4,608 logical rows in that scope, mapping to 288 structural identities. This is
an expansion through pre-pass-one authority, not a neighbourhood chosen around
observed winners.

## Artifact-quality identity audit

| Field | Logical candidate field? | Changes pre-channel JPEG 2000 artifact? | In quality identity? | Reason/source |
|---|---:|---:|---:|---|
| `dataset` | yes | fixes canonical dimensions/data domain | yes | keeps the original headline scope explicit; training manifest is separately bound |
| `source_codec` | yes | yes | yes | only frozen `jpeg2000` is admitted |
| `payload_budget_bytes` | structural, derived from ratio/MCS | yes | yes | passed as `budget_bytes` to `J2KCodec.encode_to_budget`; part of both G8_D and backend cache keys |
| `encode_axis_px` | yes | yes | yes | fixes pre-encode downsampling and is in both cache keys |
| `codec_configuration_id` | authority binding | yes | yes | binds complete output-affecting J2K/preprocessing snapshot, hash and runtime version |
| `ratio` | yes | only through resulting payload budget | no | once budget is fixed, J2K never reads ratio; retained only in scope/source lineage |
| `modulation` | yes | only through resulting payload budget | no | PHY delivery field; no J2K call input |
| `ldpc_rate` | yes | only through resulting payload budget | no | PHY delivery field; no J2K call input |
| `snr_db` | yes | no | no | explicitly absent from codec search/physical keys |
| `packet_config_id` | yes | only identifies packet geometry producing budget | no | packet arithmetic is checked before projection; exact budget remains |
| `candidate_id` / composition identity | yes | no | no | logical lineage/ranking identities only |
| source bytes, canonical-pixel hash/shape, stable ID | per-image, not quality | yes, per object | no | these identify one image-quality object and are supplied at F1; plan binds the complete train-ID set |
| emitted codestream/hash or requested compression-ratio result | no; codec output | yes, image-specific output | no | cannot be known without materialization and must never define pre-execution membership |

The exact quality identity is canonical JSON of schema/type plus
`(dataset, source_codec, payload_budget_bytes, encode_axis_px,
codec_configuration_id)`, hashed as `g8fquality-<sha256>`. Exact equality of
that tuple deduplicates PHY aliases. Altering SNR, ratio, modulation, LDPC rate,
packet ID or candidate lineage cannot create another training example if the
projected tuple is unchanged.

## Corrected deterministic corpus rule

Let `P` be immutable pass-one selections in every frozen system mode with
`snr_db <= params.channel.train_snr_db_fixed`. Let
`S = {(dataset(p), ratio(p)) : p in P}`. No score, rank, margin or tie metadata
is used. Let `C` be every frozen candidate-authority row with `(dataset, ratio)
in S` and SNR in the same region. Each `c in C` must map through the frozen
measurement authority to structural identity `m(c)` with positive
`payload_budget_bytes`, equal packet-accounting payload, and true packet,
channel and channel-use reconciliation. Project:

`q(c) = (dataset, source_codec, payload_budget_bytes, encode_axis_px,
codec_configuration_id)`.

The quality universe is the complete finite set `Q = unique({q(c): c in C})`,
ordered by `g8fquality-` ID ascending. No N-nearest width, endpoints or quality
band exists. For each `q in Q` in that order and each complete training stable
ID in ascending manifest order, F1 will make exactly one attempt. Thus the plan
freezes a quality-major Cartesian attempt order without enumerating a million
pairs.

## Feasibility and multiplicity

Feasibility levels are not conflated:

- **structural admission:** frozen packet identity, positive payload budget and
  complete reconciliation; failure refuses the quality plan;
- **configuration-level codec feasibility:** deliberately not inferred from
  validation; the codec floor is image-dependent and validation outcomes do
  not choose membership;
- **image-level codec feasibility at F1:** a typed `feasible=false` from the
  exact frozen JPEG 2000 search omits that pair and records attempted,
  materialized and omitted counts per quality and class;
- **all other failures:** decoder failure, exception, corrupt/foreign identity
  or unverified reconstruction puts F1 on HOLD.

There is no outage-image, adjacent-quality or other substitution. This avoids
validation/test leakage and performance-conditioned distribution changes.
Omission can still change class balance, so F1 must expose exact coverage rather
than silently training on whatever survived. E4 currently classifies 104
qualities as emitted for all validation images and 16 as typed codec-infeasible
for all validation images; those counts estimate cost only and remove no
quality.

Each PHY alias has multiplicity zero after the first exact projected identity.
Each quality × training stable ID has multiplicity exactly one attempt. The
actual object count is deterministically variable:

`objects = sum_{q,id} 1[verified reconstruction returned]`,

bounded by the exact attempt count and frozen at F1 with omission coverage.

## Frozen plan and compute consequence

Tracked plan:
`results/baseline/g8_f/corpus_plan.json`, ID
`g8fcorpusplan-6320ea3e5299a2175a730a2cb8c2d835e756bd11e7424f4b1221948f6f148148`,
file SHA-256
`733daa01614781c62f2af6a7e992bf29f9103a1b2afface5d77b610bb657858c`.

- unique quality identities `B = 120`;
- training stable IDs `N_train = 8,469`;
- exact attempts / maximum objects `B*N = 1,016,280`;
- validation-incidence object estimate `880,776`;
- exact object count before F1: variable by the rule above, not guessed;
- no corpus object exists in this repair.

The storage estimate reuses the frozen G8_E storage basis only: 6,838 backend
cache bytes + 5,753 v3 codec-cache bytes + 102,705 reconstruction-cache bytes =
115,296 bytes per materialized pair. That gives about 101.55 GB expected from
validation incidence, 117.17 GB maximum, and 146.47 GB maximum with 25% safety
(94.58, 109.13 and 136.41 GiB respectively).

No timing probe was run. The only confessor scaling evidence used is the
conservative Git publication window from the owner-authorization commit
`493d656` at 01:34:05 to the first E2–E4 closeout publication `5d76142` at
18:10:44: 59,799 seconds for the existing 120,000 unique physical jobs,
including unknown start delay and closeout overhead. Linear scaling gives about
121.92 h (5.08 d) for the validation-incidence estimate and 140.68 h (5.86 d)
for all attempts. These are planning upper-window estimates, not measured F1
throughput; later authorization should reserve roughly six days plus operational
margin if one Pascal worker is retained.

## Scientific direction

The artifact classifier exists to prevent the clean classifier's codec-domain
shift from artificially weakening the classical comparator. Exposing it to all
authority-implied budgets and axes generally **favours the comparator** relative
to winners-only support by increasing distortion coverage and reducing the
chance that pass two selects an unseen artifact family. It is
comparator-favourable in intent and support. The net finite-model effect is
**empirically ambiguous**: a much broader mixture can cause interference or
reduce emphasis on common operating artifacts. AM-87 therefore does not call a
performance outcome conservative or guaranteed. Its defensible property is the
removal of post-hoc discretion while retaining a strong baseline.

## Terminal boundary

G8_C and G8_D remain GREEN and unchanged. G8_E, E2, E3, E4 and immutable pass
one remain unchanged and verify. G8_F's protocol hold is resolved at plan level,
but F0 execution is not authorized. Corpus materialization, classifier
inference on new G8_F data, optimizer steps, training, pass two and test access
remain zero. The exact next action is **OWNER AUDIT OF THE CORRECTED G8_F CORPUS
PLAN / SEPARATE F0 AUTHORIZATION**. Do not merge this protocol amendment and do
not start Pascal as part of this repair session.
