# W4 — classical baseline integration: progress log

Bounded W4 integration required before G-8. **Not** the BR-4 validation sweep, and nothing here
selects an operating point. Every number below is plumbing evidence from the validation split.

---

## PB_1 — classical transport path (complete)

Built the classical arm's end-to-end path from a canonical source image to a decoded image plus a
verdict, reusing the existing J2K, LDPC, channel and preprocessing modules rather than duplicating
them.

### Modules added

* `src/baseline/classical/channel_transport.py` — bits → per-code-block TS 38.212 §5.4.2.2 bit
  interleaving → TS 38.211 Gray mapping → the **shared** registry AWGN under keyed counter-based
  noise → max-log-APP demapping → LDPC decode. Returns `TransportAccounting` (every bit counted and
  reconciled), realised per-packet symbol energy, symbol-domain PAPR, the unit-noise digest, and the
  CRC verdicts.
* `src/baseline/classical/pipeline.py` — the full segment, plus `ChannelIdentity` (the run-level
  half of `params.artifacts.noise_id_key`) and the four-verdict taxonomy.

Outage policy, records, metrics and BR-4 selection are deliberately absent: they are PB_2 and PB_3.

### Two prohibitions implemented as prohibitions

* **No per-packet power rescaling.** The only normalisation is the fixed constellation
  normalisation. Realised energy is therefore a measurement, not a constant — 16-QAM measured
  `0.926562` and `0.989500` on real images, where a renormalised packet would read exactly `1.0`.
* **No second channel implementation.** `_shared_channel()` requires the model to be in
  `params.channel.models_supported` and the constructed object to be `channels.awgn.AWGN`. A
  plausible lookalike is rejected, which is what the paired learned-vs-classical comparison depends
  on.

### The verdict taxonomy

Every invocation returns exactly one of `structural_infeasibility`, `codec_infeasibility`,
`decode_failure`, `delivered`. Nothing is skipped. Inside `codec_infeasibility`, each attempted
downsample axis records its own reason (`budget_exceeded` or `codec_configuration_error`) so a
budget failure is never confused with a codec that could not run at all.

### Payload framing

No length is signalled — `params.baseline.control_plane_policy` excludes out-of-band control from
the budget for every system, and the raw JPEG 2000 codestream is self-terminating. Residual payload
slack is carried as zero filler and counted, so the *last* `ff d9` EOC in the padded payload is the
real one. Byte-identical codestream recovery was asserted on every delivered case.

---

## Defect found and fixed: `transmit_transport` used K where TS 38.212 uses K'

`src/baseline/ldpc/transport.py` constructed the Sionna encoder with `K` — the systematic length
that already carries this project's explicit filler. Sionna derives `K_b`, and hence the lifting
size `Z`, from the information length it is given, so it re-derived them from the padded length and
selected a `Z` that disagreed with the packetisation. The pre-existing lifting-size guard then
raised on real plans.

TS 38.212 §5.2.2 derives `Z` from `K'`, so `K'` is the correct argument; Sionna owns filler
insertion and shortening. Verified against the committed packetisation record over all **232
(configuration, `E_r`) pairs**: `K'` reproduces the packetisation lifting size on **all 232**, while
`K` mismatches or errors on **46**.

These functions had never been executed before — G-2 measured BLER through `SionnaLDPCAdapter`
directly and imported only `build_packet_plan` from this module — which is why the defect survived
a passed gate.

Also added `receive_transport_verified` / `ReceivedTransport`, which *reports* CRC outcomes instead
of raising, because a decode failure has to stay classifiable.

### G-2 re-adjudication

`src/baseline/ldpc/transport.py` is bound by the G-2 execution-source manifest with the `runtime`
role, which must stay byte-identical at HEAD, so the fix tripped the HOLD. It was resolved by
recording a real re-adjudication of kind `off_measurement_path`, not by regenerating the manifest:

* `tools/run_ldpc_g2.py` imports exactly one name from this module, `build_packet_plan`, which is
  byte-identical to the measurement commit;
* the diff is confined to `transmit_transport` and `receive_transport`; `PacketPlan`,
  `PacketPlan.metadata`, `_minimum_rate`, `_candidate` and `build_packet_plan` are unchanged;
* the other seven files under `src/baseline/ldpc/` are byte-identical;
* `check_packetisation.py` still reports 0 failures over all 216 configurations, and the runtime
  solver still reproduces every committed record row exactly.

G-2's recorded BLER numbers, waterfall displacements and verdict are unchanged. No new campaign was
run and no spec amendment was made — no requirement, gate, decision or parameter moved.

The mechanism itself was tightened at the same time (manifest `schema_version` 1 → 2). A
re-adjudication entry must now declare a `kind` from `{recampaigned, off_measurement_path}`, a
`justification`, a `readjudicated_at`, non-empty `evidence`, the measurement bytes it supersedes,
and the `current_sha256` it covers. It is pinned to those exact bytes, so the *next* edit to a
re-adjudicated file re-raises the HOLD rather than inheriting the old justification, and the
verifier now prints `runtime_readjudicated=[...]` so a re-adjudicated runtime is never silent.

---

## Open issue for a spec decision: `j2k_resolutions` versus the CIFAR-10 axes

`params.baseline.j2k_resolutions = 6` requires every tile dimension to be at least `2^5 = 32`, but
`params.baseline.downsample_axis_px.cifar10` is `[32, 24, 16]`. OpenJPEG hard-errors at 24 px and
16 px — *"Number of resolutions is too high in comparison to the size of tiles"* — for **every**
image, so two of CIFAR-10's three configured axes cannot encode at all. This was never caught
because the transparency-bitrate probe ran Imagenette only (160/128/96/64).

PB_1 records it per axis rather than hiding it. It needs a decision before the BR-4 sweep, probably
an `AM`: either clamp the resolution count per axis to `min(6, log2(axis) + 1)`, or drop 24 and 16
from the CIFAR-10 axis list. **Not decided in PB_1** — it changes a frozen codec parameter, and
therefore the codec configuration hash and every J2K cache key.

A second, smaller mismatch: `params.baseline.j2k_cache_key` names `j2k_impl_version` while
`J2KCodec._cache_identity` spells the same value `openjpeg_version`. Deliberately not renamed — the
committed transparency-probe evidence records cache keys produced under the current spelling.

---

## Bounded executions (PB_1 B1.6)

Nine plumbing checks against the **validation split**, ~7 s total. No sweep, no training, no test
access, and no accuracy number here is an experimental result.

| Execution | Verdict | Observed |
|---|---|---|
| CIFAR-10 smoke, 5 real val images, `r_1_2/qpsk/(1/2)` @ 12 dB | 5/5 `delivered` | `k=1536 Qm=2 G=3072 A=1520 C=1 ΣE=3072`; capacity 190 B, emitted 184–187 B, filler 3–6 B; codestream recovered exactly |
| CIFAR-10 `r_1_2/qpsk/(1/2)` @ 18 dB | `delivered` | Es = 1.000000, PAPR 0.0000 dB |
| CIFAR-10 `r_1_2/qam16/(1/2)` @ 18 dB | `delivered` | `Qm=4 G=6144 A=3056`; 382 B cap, 380 B emitted; **Es = 0.926562**, PAPR 2.8840 dB |
| CIFAR-10 `r_1_2/bpsk/(1/2)` @ 18 dB | `codec_infeasibility` | 94 B budget is genuinely too small; `32: budget_exceeded`, `24/16: codec_configuration_error` |
| STL-10 `r_1_2/bpsk/(1/2)` @ 18 dB | `delivered` | `k=13824 Qm=1 G=13824 A=6888`; axis 96, 861 B cap, 842 B emitted, 19 B filler |
| Imagenette-160 `r_1_24/qam16/(2/3)` @ 18 dB (multi-code-block) | `delivered` | `k=3200 Qm=4 G=12800 A=8504 C=2 ΣE=12800`; axis 160, 1063 B cap / 1062 B emitted / 1 B filler; Es = 0.989500, PAPR 2.5986 dB |
| CIFAR-10 `r_1_48/bpsk/(1/3)` | `structural_infeasibility` | the one infeasible cell in the committed record; nothing downstream ran |
| CIFAR-10 `r_1_48/qpsk/(1/2)` | `codec_infeasibility` | packetisation feasible (`A=48`, capacity **6 B**); no axis produced a codestream |
| CIFAR-10 `r_1_2/qpsk/(1/2)` @ **−10 dB** | `decode_failure` | `crc=False`; measurements still emitted |
| Cached JPEG 2000 repeat | `delivered` | `cache_hit=True`, same cache key and `codestream_sha256`, byte-identical decoded image, same `unit_noise_sha256` |

### Worked accounting examples

    CIFAR-10  r_1_2  qpsk  1/2   k=1536   Qm=2   k*Qm=3072
      A=1520  TB CRC(crc16)=16  CB CRC=0  C=1  K'=1536  Z=160  K=1600
      filler = 1*(1600-1536) = 64      1520+16+0+64 = 1600 = C*K
      E=(3072,)   sum E = 3072 = k*Qm
      payload capacity 190 B, emitted 187 B, payload filler 3 B

    Imagenette-160  r_1_24  qam16  2/3   k=3200   Qm=4   k*Qm=12800
      A=8504  TB CRC(crc24a)=24  C=2  CB CRC total=48  K'=4288  Z=208  K=4576
      filler = 2*(4576-4288) = 576     8504+24+48+576 = 9152 = 2*4576 = C*K
      E=(6400, 6400)   sum E = 12800 = k*Qm
      payload capacity 1063 B, emitted 1062 B, payload filler 1 B

    Imagenette-160  r_1_3   qpsk  5/6   k=25600  Qm=2   k*Qm=51200   (BR-10's worked example)
      A=42624  TB CRC(crc24a)=24  C=6  CB CRC total=144  K'=7132  Z=352  K=7744
      filler = 6*(7744-7132) = 3672    42624+24+144+3672 = 46464 = 6*7744 = C*K
      E=(8532, 8532, 8534, 8534, 8534, 8534)   sum E = 51200 = k*Qm

---

## Tests

`565 passed, 0 failed` (501 before PB_1). Added:

* `tests/test_classical_transport.py` — 22 tests. All 14 required areas: exact channel uses over
  every ratio × modulation × rate (215 feasible, 1 infeasible), exact bit reconciliation against
  every committed packetisation row, the shared channel object, keyed noise for one identity, three
  modulations, four LDPC rates, a partial final code block, J2K emitted-byte authority, J2K cache
  identity, no per-packet rescaling, realised energy, PAPR, validation-only loading.
* `tests/test_classical_pipeline.py` — 13 tests over the verdict taxonomy, budgets, cache and split
  isolation.
* `tests/test_classical_mutations.py` — 18 tests covering all nine required mutation classes.
* `tests/test_g2_adjudication.py` — 41 → 52 tests, covering the tightened re-adjudication contract.

Test-isolation counters remain zero: `verify_g2_adjudication.py` reports `test_split_access=0`, and
`tests/test_test_access.py` passes its 4 checks. `load_dataset(..., "test")` still refuses.

---

# PB_1C — corrective audit of the classical transport path

PB_1 was marked complete. An external audit then raised a likely standards-conformance defect: the
TS 38.212 §5.4.2.2 modulation bit interleaver may be applied **twice** on the transmit path and
undone twice on the receive path. This section records the independent verification. It does not
rewrite anything above it; observations that this correction supersedes are marked in place.

## C1.1 — Sionna interleaver ownership, verified against installed source

Installed package: **Sionna 2.0.1**, `.venv/lib/python3.14/site-packages/sionna/`.

| What | Where |
|---|---|
| encoder | `.venv/lib/python3.14/site-packages/sionna/phy/fec/ldpc/encoding.py` |
| decoder | `.venv/lib/python3.14/site-packages/sionna/phy/fec/ldpc/decoding.py` |

**Encoder.** `LDPC5GEncoder.__init__` stores `num_bits_per_symbol` at `encoding.py:172` and, when it
is not `None`, immediately builds both permutations at `encoding.py:173-179` via
`generate_out_int(n, num_bits_per_symbol)` (`encoding.py:303-339`), registering them as the
`_out_int` / `_out_int_inv` buffers exposed as the `out_int` / `out_int_inv` properties
(`encoding.py:268-274`). Validation is inside `generate_out_int`: integral, positive, and
`n % num_bits_per_symbol == 0`.

The permutation is built as

```python
perm_seq[i + j * num_bits_per_symbol] = i * int(n / num_bits_per_symbol) + j
```

`encoding.py:336-339`, i.e. exactly the "write by rows of Qm, read by columns" §5.4.2.2 map.

In `call()`, the order is unambiguous: filler padding and encoding (`encoding.py:749-765`), filler
removal, **then rate matching** — the `2Z` puncture skip and the length-`n` selection at
`encoding.py:770-780` — and only *after* that the interleaver, at `encoding.py:791-793`:

```python
# Output interleaver (Sec. 5.4.2.2) — works on last dim for any rank
if self._num_bits_per_symbol is not None:
    c_out = c_out[..., self._out_int]
```

So the encoder **does** apply the output interleaver, and it applies it **after** rate matching.

**Decoder.** `LDPC5GDecoder` is constructed *from the encoder instance* and reaches through it. At
`decoding.py:1646-1649` (and the multi-RV branch at `decoding.py:1636-1637`) it applies
`self._encoder.out_int_inv` to the channel LLRs, and it does so **before** rate recovery — the
de-interleaved LLRs are what get padded into `llr_buf` and reassembled into the full `n_ldpc` vector
at `decoding.py:1654-1667`. `decoding.py:1693-1694` re-applies `out_int` on the way out when
`return_infobits` is false. The pairing is automatic: nothing in the project selects it, and nothing
can disable it while the encoder carries a `num_bits_per_symbol`.

**Behaviour by modulation.** Probed directly against the installed package (`k=200, n=400, bg2`):

| Qm | `encoder.out_int` equals project `interleaver_indices(n, Qm)` | identity permutation |
|---|---|---|
| 1 (BPSK) | yes | **yes** |
| 2 (QPSK) | yes | no |
| 4 (16-QAM) | yes | no |

and `encoder.out_int_inv` equals `np.argsort(project interleaver_indices(n, Qm))` for all three. A
second probe confirmed the composition directly: for Qm ∈ {2, 4},
`LDPC5GEncoder(..., num_bits_per_symbol=Qm)(u)` is bit-identical to
`interleave(LDPC5GEncoder(...)(u), Qm)`.

That is the whole finding. The project's own `interleaver_indices`
(`src/baseline/ldpc/modulation.py:24-32`) is not merely *a* valid §5.4.2.2 interleaver — it is the
**same permutation Sionna already applied**.

## Conclusion — the audit is confirmed

`SionnaLDPCAdapter` (`src/baseline/ldpc/adapter.py:26-32`) always passes `num_bits_per_symbol=q_m`,
so every encode is already interleaved. `channel_transport.modulate()` then applies `interleave()`
again per code block, and `demodulate()` applies `deinterleave()` before handing LLRs back to the
paired Sionna decoder, which applies `out_int_inv` a second time. The realised chain is

```text
rate matching → Sionna interleaver → project interleaver → mapping
             → demapping → project inverse → Sionna inverse → rate recovery → decode
```

which contains **two** modulation bit interleavers where TS 38.212 specifies one. The transmitted
bit order is the permutation *squared*, which is not the identity for Qm = 2 or 4.

**Why the PB_1 round-trip tests missed it.** They are self-consistency tests: the extra transmit
permutation is exactly undone by the extra receive permutation, so CRC passes, the codestream is
recovered byte-exactly, and every accounting identity still holds — bit *counts* are permutation-
invariant. Nothing in PB_1 ever compared the sequence entering the mapper against an independently
derived reference, which is the only check that can see a paired error.

**Scope of the consequence.** No bit is lost and no count changes, so the PB_1 accounting evidence
stands. What is wrong is the *channel-facing bit order*: BICM exists to spread a code block's bits
across symbol bit-positions, and applying the map twice partially re-clusters them. Realised symbol
energy, PAPR and the noise realisation are therefore not the values a conforming transmitter would
produce, and BLER at a given SNR is not the standards-conformant one. Since BR-4 tunes the baseline
per SNR, a non-conformant baseline link is exactly the kind of unfair-baseline defect the project's
non-negotiables forbid. **BPSK is unaffected** — at Qm = 1 both permutations are the identity.

Repair belongs in `src/baseline/classical/`, not in `src/baseline/ldpc/`: Sionna owns the
interleaver, so the project-side application in `channel_transport.py` is what must go. The
standalone utilities in `src/baseline/ldpc/modulation.py` stay — they are G-2 known-answer material
and are now also the independent reference this correction tests against.

## C1.5 — JPEG-2000 resolution constraint: adjudicated as **unresolved, with an explicit downstream block**

`params.baseline.j2k_resolutions = 6` requires every tile dimension to be at least `2^5 = 32` px;
`params.baseline.downsample_axis_px.cifar10` is `[32, 24, 16]`. OpenJPEG hard-errors at the 24 px
and 16 px axes for every image and every budget.

**Is there an existing normative rule that resolves it?** No. BR-1 (via AM-51) freezes
`j2k_resolutions` as a flat scalar alongside the other codec flags, and AM-58 makes
`downsample_axis_px` dataset-specific with `downsample_axis_never_upscales` as its only stated
invariant. Neither states a clamping rule, a per-axis resolution rule, or a minimum-axis
precondition. The two parameters were frozen in different amendments and were never checked against
each other — the transparency probe ran Imagenette only (160/128/96/64), all of which clear 32 px.

**Decision: leave it unresolved during PB_1C.** PB_1 correctness does not require changing it.
CIFAR-10 is a plumbing smoke path only (DEC-1), its 32 px axis encodes correctly, and the pipeline
already reports the two failing axes honestly rather than skipping them. Changing a frozen codec
parameter to tidy up an unrelated interleaver repair would be exactly the silent-amendment failure
the §17 convention exists to prevent, and it would invalidate every committed J2K cache key.

**Candidate resolutions, both recorded, neither selected:**

1. remove axes 24 and 16 from `params.baseline.downsample_axis_px.cifar10`;
2. make `j2k_resolutions` axis-dependent or deterministically clamped to
   `min(6, floor(log2(axis)) + 1)`.

Either needs the next valid `AM` entry: both change the codec configuration and therefore the
`j2k_cache_key` for every dataset, not only CIFAR-10.

**Blocked on the decision:** PB_3, the full BR-4 sweep, G-8. **Not blocked:** PB_2, which emits
outage policy, records and smoke evidence without selecting or sweeping codec-rate candidates.

**Executable reproduction** —
`tests/test_classical_pipeline.py::test_j2k_resolutions_cannot_encode_cifar10s_small_axes` pins
`j2k_resolutions == 6` and the CIFAR-10 axis list, then asserts that under one packet plan the 32 px
axis reports `budget_exceeded` while 24 px and 16 px report `codec_configuration_error` — and that
the distinction is not a budget artefact, since 32 px succeeds outright at a generous budget. Keeping
the two reasons separate is the point: a configuration fault reported as "the codestream did not
fit" would read as a channel result.

## C1.2–C1.3 — independent evidence and the repair

### Why the PB_1 tests could not have caught it

Every PB_1 transport test was a *round trip*. The extra transmit permutation was undone by the extra
receive permutation, so CRC passed, the codestream came back byte-exact, and every accounting
identity held — bit **counts** are permutation-invariant. Nothing compared the sequence entering the
mapper against a reference derived independently of the code producing it. That is the only check
that can see a paired error, and PB_1 had none.

### Independent pre-fix failing evidence

`tests/test_classical_interleaver_conformance.py` builds its reference two ways, neither of which
touches the project's interleaver:

* `_ts_38212_out_int(E, Qm)` — the §5.4.2.2 permutation written as
  `np.arange(E).reshape(Qm, E // Qm).T.reshape(-1)`, deliberately a different spelling from
  `modulation.interleaver_indices`'s index generator, so a shared formula mistake cannot hide behind
  an identically shaped copy of itself;
* `_encode_uninterleaved(...)` — a raw `LDPC5GEncoder` built *without* `num_bits_per_symbol`, which
  therefore rate-matches and stops.

Their composition is the standards-conformant rate-matched, once-interleaved sequence.

Against the pre-correction tree the module reported **`12 failed, 14 passed in 5.21s`**. The two
tests that validate the reference itself (`out_int` / `out_int_inv` agreement for Qm ∈ {1, 2, 4};
identity only at Qm = 1) both **passed**, so the failures could not be blamed on a wrong reference.
The primary failing assertion was `assert np.array_equal(observed, expected)` with `observed` and
`expected` of **identical length** — 3072 for QPSK, 6144 for 16-QAM, 12800 for the two-block
Imagenette case — differing only in order. A permutation defect, not a count defect. The `[bpsk]`
parametrisation passed throughout, confirming the Qm = 1 identity conclusion.

### The repair

Entirely inside `src/baseline/classical/channel_transport.py`. No file under `src/baseline/ldpc/`
was touched, so the existing G-2 `off_measurement_path` re-adjudication for `transport.py` is
preserved exactly and no new one was added.

* `mapper_input_bits()` — new seam, now the body of `modulate()`. It concatenates the adapter's
  already-interleaved code blocks and does nothing else. The per-block `interleave()` is gone.
* `demodulate()` — max-log demaps the packet and returns `split_llr_blocks()` at exact `E_r`
  boundaries. The per-block `deinterleave()` is gone; the paired Sionna decoder's `out_int_inv` is
  now the only inverse.
* `split_llr_blocks()` — new, exact `E_r` cutting, raises on a total mismatch.
* Unused `interleave` / `deinterleave` imports dropped; the module docstring's interleaver clause
  rewritten to name Sionna as the single owner.

`_require_interleaver()` and `_require_fixed_normalisation()` are still called: they are what make
the adapter's `Qm` argument mandatory rather than optional. Preserved unchanged: exact `k × Qm`
accounting, partial final code blocks, fixed mapping and normalisation, no per-packet rescaling,
realised energy and PAPR, the shared AWGN, keyed noise identity, the four verdicts, exact codestream
recovery. The standalone utilities in `src/baseline/ldpc/modulation.py` were left in place — they
are G-2 known-answer material and are now also the conformance test's cross-check.

### Four PB_1 tests updated, not deleted

They asserted the defect, so they had to move rather than be weakened:

* `test_modulation_applies_only_the_fixed_constellation_normalisation` — its `expected` re-interleaved
  the blocks by hand;
* `test_qam16_bit_interleaver_is_required_and_actually_changes_the_symbols` and
  `test_qam16_interleaver_is_not_a_no_op` — both asserted `modulate() != map_bits(concat(blocks))`,
  which was true only *because* of the duplicate. Both now make the same claim at the seam that owns
  it (`adapter.encoder.num_bits_per_symbol == Qm`, `out_int != arange`) **and** additionally assert
  that the project adds nothing on top;
* `test_disabled_qam16_interleaver_is_rejected_and_corrupts_the_link` part (b) — patched the removed
  `channel_transport.interleave`. It now injects a *second* application (the PB_1C defect itself) and
  asserts the link dies at 20 dB.

### Conformance tests and mutation coverage

`tests/test_classical_interleaver_conformance.py`, 35 tests. All seven required mutation classes are
caught by an independent seam or known-answer property rather than by an eventual CRC failure:
adapter built without `num_bits_per_symbol`; wrong `Qm` to the adapter; an additional transmitter
interleaver; an additional receiver inverse interleaver; bypassing the sole required interleaver;
incorrect code-block concatenation; incorrect LLR block splitting.
`test_independent_fixture_rejects_every_transmitter_mutation` rebuilds the transmit path with exactly
one defect injected and asserts the known-answer equality breaks **while the bit count is
preserved** — the explicit demonstration that counting identities alone could never have caught this.

Targeted regression at C1.6: **97 passed** — conformance 35, transport 22, mutations 18, pipeline 18,
test-access 4.

### Explicit-axis correction (C1.4)

`_encode_source` in `src/baseline/classical/pipeline.py` previously accepted any explicit
`encode_axis_px` that did not upscale, so an unconfigured axis (28 px for CIFAR-10, say) could reach
OpenJPEG and mint cache keys and evidence for a configuration the spec never authorised. An explicit
axis is now a *selection* from `configured_axes(dataset, canonical_shorter_side)`; membership is
required, and both that check and the upscale check run **before** `codec_downsample` or
`encode_to_budget`. No second configuration source was introduced. Four tests added
(`tests/test_classical_pipeline.py` 13 → 18 including the C1.5 reproduction), one of which
monkeypatches both codec entry points to raise, proving rejection precedes codec execution.

### Corrected bounded observations (C1.7)

All ten bounded PB_1 executions were rerun against real validation data; the full per-execution
record is in the `instructions/RESUME.md` C1.7 fact rows. Summary of the comparison against B1.6:

**Invariant** — ratio, modulation, Qm, nominal rate, verdict, `k`, `G = k × Qm`, `A`, payload bytes,
TB/CB CRC names and widths, code-block count, LDPC filler, `E` and `ΣE`, selected axis, capacity,
emitted bytes, `codestream_sha256`, cache key and cache-hit behaviour, exact codestream recovery,
CRC outcome, delivery outcome. Bit counts are permutation-invariant and source coding is upstream of
the defect, so none of this could have moved.

**Changed** — symbol ordering, which is the repair. The two order-dependent measurements therefore
moved on the non-constant-modulus modulation:

| case | B1.6 (superseded) | C1.7 (corrected) |
|---|---|---|
| CIFAR-10 `r_1_2`/`qam16`/`1/2` @ 18 dB | Es 0.926562, PAPR 2.8840 dB | Es 0.985417, PAPR 2.6165 dB |
| Imagenette-160 `r_1_24`/`qam16`/`2/3` @ 18 dB | Es 0.989500, PAPR 2.5986 dB | Es 0.994000, PAPR 2.5789 dB |

Expected: realised symbol energy and PAPR are measurements over the realised symbol *sequence*, and
PB_1 grouped bits into 16-QAM symbols under the permutation squared. **BPSK and QPSK are unchanged**
— BPSK because Qm = 1 is the identity, QPSK because its constellation is constant-modulus, so Es ≡ 1
and PAPR ≡ 0 dB whatever the bit order.

**The B1.6 QPSK/16-QAM `Es` and `PAPR` figures recorded earlier in this worklog are superseded**:
they describe a non-conformant transmit order. Every other B1.6 observation stands. `noise_id` and
`unit_noise_sha256` also differ, for the unrelated reason that C1.7 used a different
`ChannelIdentity` fixture.

### G-2 impact

None. `git diff --name-only` across the whole correction shows no file under `src/baseline/ldpc/`.
`tools/verify_g2_adjudication.py` passes unchanged at every checkpoint:
`measurement=968e907237bb, rows=24, test_split_access=0, sources=14,
runtime_readjudicated=['src/baseline/ldpc/transport.py']`. No manifest was regenerated, no evidence
recreated, no BLER campaign re-run, and no second re-adjudication added.

### Amendment judgment — no amendment required

Removing an accidentally duplicated interleaver **restores** the behaviour the specification already
requires. `params.baseline.modulation_bit_interleaver` is still `ts_38212_5_4_2_2` and
`modulation_bit_interleaver_required` is still true; BR-1's chain is unchanged; no requirement, gate,
decision or frozen parameter is modified. It is an implementation bug fix, and §17's convention does
not treat those as amendments. The C1.4 explicit-axis guard is likewise a bug fix: it *enforces*
`params.baseline.downsample_axis_px` rather than changing it. C1.5 deliberately changed no normative
behaviour, so no `AM` entry arises from it either.

### Remaining open issues

* **`j2k_resolutions` vs CIFAR-10's 24/16 px axes** — ~~unresolved by decision; blocks PB_3, the full
  BR-4 sweep and G-8; does not block PB_2.~~ **Resolved at PB_2C/C2.1 by AM-80**; recorded here as it
  stood at PB_1C. See the C1.5 section above and the PB_2C section below.
* **Cache-key field spelling** — `baseline.j2k_cache_key` names `j2k_impl_version`,
  `J2KCodec._cache_identity` spells it `openjpeg_version`. Values agree; deliberately not renamed
  because the committed transparency-probe evidence records keys produced under the current
  spelling. Tested by value. Unchanged by PB_1C.

---

# PB_2 — outage policy, records, and bounded W4 evidence

Driven by `instructions/PB_2D.txt` (the durable instruction committed at B2.0, which supersedes
`instructions/PB_2.txt` for execution). This section is append-only and rewrites no PB_1 or PB_1C
history. Green implementation/evidence commit and per-step SHAs are in `instructions/RESUME.md`.

One naming correction worth stating up front, because it would waste a session: `instructions/PB_2.txt`
names the record schemas `analysis.csv_schema` and `analysis.per_image_schema`. There is **no
`analysis` root** in `spec/params.generated.yaml`. The real parameters are `artifacts.csv_schema`
(52 fields) and `artifacts.per_image_schema` (16 fields), together with `artifacts.system_values`,
`artifacts.run_id_key`, `artifacts.analysis_cell_id_key`, `artifacts.noise_id_key`,
`artifacts.pair_id_key`, `artifacts.pair_id_excludes` and `artifacts.checkpoint_id_form`.

## Outage-policy method, and why the measured value is not a hardcoded 1/n

`src/baseline/classical/outage.py` selects the frozen constant class by counting labels across the
**entire** committed Imagenette-160 validation manifest. It decodes no image, runs no classifier,
consults no loader order and reads no sample subset; rows outside the `val` split — every `train` and
every `test` row — are discarded before any counting happens. The artifact is frozen to
`results/baseline/w4/outage_policy.json` by `tools/gen_w4_outage_policy.py`, which has a `--check`
mode that regenerates the record in memory and compares it field by field, ignoring only
`generated_at` and `selection_source_commit`.

**The full validation count is `[100] * 10` over 1000 rows.** All ten classes tie at the maximum, so
the configured `lowest_class_index` tie-break is the only thing that picks a winner: **class 0**,
with numerator 100, denominator 1000, measured accuracy **0.1**.

That number coincides exactly with `1 / n_classes`, and the coincidence is not an accident —
`data/manifests.py::_validate_counts_and_stratification` **enforces** an exactly stratified
validation split, so no valid manifest in this project can produce an unbalanced validation
histogram. This is precisely why a float comparison is worthless here: a hardcoded `0.1` and a
measured `100/1000` are indistinguishable by value. The artifact therefore records the numerator,
denominator, the full class-count vector, the maximum, and the tied set; `policy_from_record`
recomputes the selection from those counts and rejects the record if any of them disagree; and the
verifier re-derives the counts from the manifest itself rather than trusting the artifact.
`tests/test_classical_outage.py::test_artifact_that_hardcodes_one_over_n_without_matching_counts_fails`
exhibits the discriminating mutation: halving one class count leaves the theoretical value untouched
while the measured one moves.

Runtime prediction never reselects. `OutagePolicy.predict()` is argument-free and side-effect-free —
no sample identity, no label, no manifest and no system control flow can reach it — and a test
monkeypatches `count_validation_labels` and `validate_manifest_bytes` to raise, then asserts
prediction still works. Per-row correctness stays strictly binary; no row is ever fractionally right.

## Sensitivity variant

`keyed_uniform_random_label` draws one integer in `[0, n_classes)` from the central counter-based
Philox stream. `params.baseline.outage_rng_key` spells the purpose as the pseudo-field
`purpose=outage_label`, but `artifacts.rng.keyed_generator` takes the purpose as its first argument
and **rejects** it inside the identity, so `configured_rng_identity_fields()` strips the marker and
asserts the remaining three fields equal `params.artifacts.rng_identity_fields.outage_label`
— `{split_manifest_hash, stable_sample_id, channel_seed}`. The draw is proved invariant to row order,
to four batch sizes, to being computed for a subset, and to intervening draws from an unrelated
generator, and proved to move when any declared component moves. It is recorded as a secondary
comparison and is deliberately **not** a per-image schema field.

## Classifier dataset boundary

The adjudicated G-1 checkpoint is an **Imagenette-160** classifier
(`9c37362347a0…`, config `a9717575d71f…`). CIFAR-10 has ten class indices too, but they are a
different vocabulary, so applying the frozen model to a CIFAR reconstruction would produce a number
that looks like an accuracy and means nothing. `records.score_result` refuses it by name, the runner
refuses to task-score any dataset that is not the frozen classifier's, and the verifier fails closed
if the CIFAR section of the summary carries `top1_acc`, `n_correct`, `task_accuracy` or
`classifier_inference_performed: true`, or if any per-image row is not Imagenette-160. **No CIFAR
task accuracy was computed at any point.** CIFAR remains a transport, verdict, accounting, cache and
schema plumbing smoke, and its summary says so in those exact words.

## Record architecture and identities

`src/baseline/classical/records.py` reads both schemas from `params.artifacts.*` at runtime;
`validate_row` reports missing, unexpected and reordered fields as distinct failures, and duplicate
configured field names are rejected outright. The only hand-written table is `FIELD_SEMANTICS`,
which *annotates* the configured fields and is asserted to cover them exactly — production code
holds no second copy of a schema list.

Identities reuse `make_run_id`, `make_analysis_cell_id`, `make_noise_id` and `make_pair_id`; no
parallel hashing or canonicalisation was written. `dataset_version` is the configured archive
SHA-256, `config_hash` is the versioned resolved `RunConfig` fingerprint (not an ad hoc YAML hash),
`checkpoint_id` is the SHA-256 of the exact frozen checkpoint bytes, and `noise_id` is proved equal
to PB_1's `ChannelIdentity` result. Changing `system` changes `run_id` but not `pair_id`; changing
split, config, checkpoint or classifier variant changes `run_id`; row order and batching change no
identity.

**System value: `classical_fixed_mcs`.** PB_2 runs one explicitly fixed (ratio, modulation, LDPC
rate) configuration and implements no per-SNR selection. `classical_adaptive` would assert an
adaptation that PB_3 has not yet constructed or verified, so it is not used.

`RunIdentity` whitelists the splits a record may describe rather than blacklisting the sealed one.
That is deliberate: PB_1C left a standing invariant that no file under `src/baseline/classical/`
may contain the literal `"test"`, and a blacklist would have silently eroded it.

## Field semantics, and one flagged interpretation

`resolved_config.json` carries a machine-readable entry for every field of both schemas, giving
source, type, unit, nullability, not-applicable representation and aggregation denominator. The
documented not-applicable representation is JSON `null` / an empty CSV cell.

Two field meanings were resolved from the spec rather than guessed:

* **`source_bytes` is exactly `A/8`**, per BR-10 ("`source_bytes` is exactly `A/8` rather than a
  floor with a bit remainder"). Confirmed against `spec/evidence/packetisation_record.json`, where
  `A = 12776` gives `source_bytes = 1597`. It is not the original archive bytes and not the canonical
  RGB byte length.
* **`effective_code_rate` is `K' / max(E_r)`**, the worst-block realised rate — confirmed against the
  same record (`K' = 6424`, `max E_r = 19200`, recorded `0.334583`), and consistent with
  `params.baseline.min_coderate_predicate`, which is also evaluated on the worst block.

**One interpretation is flagged rather than asserted.** BR-11 requires `header_bytes` and
`payload_bytes` to be reported separately "so the fraction of the budget spent on format overhead is
visible", and `params.baseline.container_policy` puts every emitted container byte *inside* the
payload budget — but the spec never spells the two columns out arithmetically. The resolution used
here is:

    bytes_sent    = A/8, the complete transport-block payload placed on the channel
    header_bytes  = JPEG 2000 raw-codestream container bytes (SOC, main-header marker
                    segments, tile-part headers through SOD, EOC)
    payload_bytes = emitted_bytes - header_bytes, the entropy-coded image data
    residual      = bytes_sent - header_bytes - payload_bytes = zero payload filler

The residual has no schema column, so it is reported in `accounting_examples.json` rather than folded
into either column — that keeps both denominators clean. A reader could instead take `payload_bytes`
to mean `payload_bits / 8`, which would make it identical to `bytes_sent`. The choice is recorded in
`records.BYTE_ACCOUNTING_NOTE`, in the committed field-semantics artifact and in `RESUME.md`, and the
bounded evidence is ~45 s to regenerate if it is decided differently. **This is the one PB_2
interpretation that should be confirmed rather than inherited.**

The container split is computed by a marker-walking parser over the raw codestream, and the two
counts are asserted to sum to the emitted byte count on every row — an approximate header figure
would make BR-11's overhead fraction unfalsifiable. The parser is exercised on real encodes.

## Aggregate formulas and denominators

`n == rows`; `n_test == n` despite the legacy field name (it is **not** a test-split count);
`n_correct == sum(correct)` counting outage rows; `top1_acc == n_correct / n`;
`coverage_rate == delivered / n`; `decode_failure_rate == decode_failures / n`;
`infeasible_rate == (structural + codec) / n`; and
`delivered + decode_failure + structural + codec == n` is enforced. `acc_given_delivery` is
`delivered_correct / delivered_count`, or `null` when nothing was delivered. **PSNR and SSIM are
aggregated over delivered rows only**, and that denominator is recorded explicitly in the summary as
`psnr_ssim_denominator` because the CSV schema does not carry it. PAPR is a mean over transmitted
rows (delivered plus decode failures); `bytes_sent` is fixed per configuration; `header_bytes` and
`payload_bytes` are means over delivered rows. `reconcile_aggregate` recomputes all of it
independently of the builder, and the verifier recomputes it again from the CSV.

## Crash-resume behaviour

Every completed row is appended to `smoke_rows.partial.jsonl` and `fsync`ed before the next row
starts; progress metadata and all finalised JSON/CSV are written through atomic temporary-file
replacement. Resuming re-validates six bindings — source commit, config hash, checkpoint hash,
manifest hashes, worklist hash and plan hash — and refuses to mix rows if any of them moved. The run
timestamp is captured once and reused on resume.

The drill: `--max-rows 3 --restart` left 3 durable rows; a second invocation reported `resuming with
3 durable rows of 55`, recomputed none of them and appended 4 more, leaving 7 rows with 0 duplicates
and `complete: false`. Tampering with the recorded `source_commit` and then the `config_hash` each
produced a refusal naming the differing field. Finalisation from 55 already-durable rows recomputed
nothing. Partial-row identity, duplication and schema mismatches are covered by unit tests.

One real defect surfaced here: partial rows are written with `sort_keys=True` for byte determinism,
which alphabetises the per-image record, and per-image field *order* is part of the contract. Rather
than weakening `validate_row` to ignore order, the loader restores schema order and rejects any row
whose field *set* differs.

## Bounded executions

**55 rows in 44.3 s** from a clean tree at `b8462316c3c9`. No sweep, no candidate comparison, no
operating-point selection, no training, no test access.

* **CIFAR-10 transport-only** — 5 real validation samples, `r_1_2/qpsk/(1/2)` @ 11 dB, explicit 32 px
  axis, **5/5 delivered**, `codestream_exact` on all five. `k=1536 Qm=2 G=3072 A=1520`,
  `bytes_sent=190 B` = 157 B container + 27–30 B entropy data + 3 B filler. **83% of that transport
  block is JPEG-2000 container** — the clearest illustration in this project of why DEC-9 rejected
  JPEG and why BR-11 exists. No classifier inference and no task score.
* **Imagenette-160 task-scored** — 24 images by `lowest_stable_sample_id_first`, at 18 dB and −8 dB,
  `r_1_24/qam16/(2/3)`, `k=3200 Qm=4 G=12800 A=8504 C=2`, `bytes_sent=1063 B`.
  * **18 dB, n=24**: 24/24 delivered, `top1_acc = 18/24 = 0.75`, `acc_given_delivery = 0.75`,
    PSNR 27.034 dB and SSIM 0.7736 over a denominator of 24 delivered rows, PAPR 2.693 dB,
    1063 = 157 + 905 + 1 bytes.
  * **−8 dB, n=24**: 24/24 real `decode_failure`, `coverage_rate = 0`, every row taking the frozen
    class 0, `top1_acc = 3/24 = 0.125` — the three rows whose true label is class 0.
    `acc_given_delivery`, `psnr_db` and `ssim` are all null under the zero-delivery denominator;
    PAPR is still recorded, because the packet really was transmitted.
* **Fixtures** — `structural_infeasibility_cifar10` (`r_1_48/bpsk/(1/3)`, no legal byte-aligned A)
  and `codec_infeasibility_imagenette160` (64-byte budget, all four configured axes reporting
  `budget_exceeded`). Both are labelled fixtures and are kept out of the task-scored aggregate.
* **Cached JPEG-2000 repeat** — 24 distinct cache keys, all 24 repeated across the two SNR points,
  every repeat a cache hit reproducing a byte-identical codestream.

**None of these accuracies is an experimental result, a finding, or an estimate of test performance.**
Every one is stated with its sample size, and the two SNR cells are reported separately and never
pooled.

A fact worth recording because a reader would otherwise expect it: **no configured
`(bw_ratio, modulation, ldpc_rate)` triple is structurally infeasible on Imagenette-160** — all 72
packetise. A structural-infeasibility *record* is therefore unreachable on the frozen classifier's
dataset, so that fixture lives on CIFAR-10 and the Imagenette structural→outage record path is
covered by unit tests instead.

## Verifier and source binding

`tools/verify_w4_baseline_integration.py` recomputes rather than trusts: it re-derives the outage
class from the committed manifest, recomputes every aggregate rate from the per-image rows, re-hashes
the emitted CSVs against the hashes the summary claims, and re-hashes every bound source at the
declared execution commit. `tools/gen_w4_source_manifest.py` binds **37 sources** in four roles.

Unlike G-2 there is deliberately **no re-adjudication mechanism**: the bounded run takes ~45 s, so a
changed runtime source is answered by rerunning the evidence, never by recording an exception. That
rule earned its keep twice during B2.6 — the first run recorded a stale `execution_source_commit`
because the runner was still uncommitted, and the second was invalidated when the runner changed to
record the container split on transport-only rows. Both times the drift check refused the evidence.

`per_image.csv` and `aggregate.csv` are deliberately **not** bound in the manifest: they are outputs
of the execution commit and cannot exist at it. They are bound by SHA-256 inside `smoke_summary.json`,
which the verifier recomputes from disk.

Mutation coverage is in `tests/test_w4_verification.py` (39 tests), which builds a complete valid
evidence directory from scratch so it proves discrimination independently of whether the real bounded
run exists. All 13 required classes are covered plus 12 more.

## Amendment judgment — no amendment required

PB_2 implements outage handling, record emission, identities and bounded evidence that
`spec/SPEC.md` already specifies. No requirement, gate, decision or frozen parameter changed. The
outage policy, its tie-break, its freeze point and its sensitivity variant are all read from
`params.baseline.*` as written; the schemas and identity key sets are read from `params.artifacts.*`
as written; the frozen G-1 checkpoint is unchanged. The two field meanings resolved above are
*readings* of BR-10 and BR-11, not modifications of them, and the one that could reasonably be read
otherwise is flagged rather than buried. §17's convention does not treat implementation of an
existing requirement as an amendment.

## ~~Remaining block before PB_3~~ — **SUPERSEDED at PB_2C/C2.1 by AM-80. Completed snapshot as written at PB_2.**

> `baseline.downsample_axis_px.cifar10` is now the single native `[32]` rung, so nothing below still
> blocks PB_3. Retained unedited as the PB_2-era record; see the PB_2C section for the resolution.

**`j2k_resolutions` vs CIFAR-10's 24/16 px axes remains unresolved by decision.** PB_2 did not touch
it: the CIFAR smoke pins the working 32 px axis explicitly so the conflict cannot contaminate
ordinary evidence, and the conflict itself stays reproduced by
`tests/test_classical_pipeline.py::test_j2k_resolutions_cannot_encode_cifar10s_small_axes`. The W4
verifier fails closed if any evidence marks the issue resolved. It still blocks PB_3, the full BR-4
sweep and G-8.

The cache-key field-spelling issue recorded at PB_1 is likewise unchanged.

---

## PB_2C — corrective provenance, pairing and JPEG-2000 accounting repair

Driven by `instructions/PB_2C.txt`. This section is append-only and does not rewrite anything above:
PB_2's implementation happened and stands. What PB_2C corrects is its **completion judgment and its
bounded evidence**, both of which were accepted on the strength of a verifier that could not see the
defects.

### What the audit found

Eight defects, all confirmed directly against the tree at `e0155c3` before any change was made.

1. **One `RunConfig` for every cell.** The runner resolved a single configuration from `snr_db[0]`
   (18 dB) and threaded its hash into every row and every aggregate. So `config_hash` `ba59d1e7…`
   was attached to CIFAR-10 `r_1_2`/qpsk/(1/2) rows while its `resolved` block described
   Imagenette-160 `r_1_24`/qam16/(2/3). Worse, **modulation, LDPC rate and encode axis were not in
   the fingerprint at all** — they lived only in the execution plan, so two genuinely different
   configurations could have shared a run fingerprint.
2. **Infeasible rows could not pair.** `per_image_row` took `noise_id` straight off the pipeline
   result, which is `None` when nothing was transmitted, and fed that same `None` into `pair_id`. An
   infeasible classical row therefore could never share a `pair_id` with a transmitting comparison
   arm — silently removing from the paired comparison exactly the images where two systems differ
   most.
3. **Byte columns were delivered-only.** `score_result` returned `(None, None)` for every
   non-delivered verdict, so a cell whose rows all failed to decode reported **no overhead at all** —
   the regime where format overhead dominates the budget, and the one BR-11 exists to expose.
4. **`Psot` was read at the wrong offset.** `_PSOT_OFFSET` was 4; the SOT segment is
   `SOT(2) | Lsot(2) | Isot(2) | Psot(4) | TPsot(1) | TNsot(1)`, so Psot begins at **6**.
5. **The row timer excluded scoring.** It bracketed `run_classical_pipeline()` only, so classifier
   inference — the most expensive part of a delivered row — was never counted.
6. **The summary wall clock ignored pre-resume rows.** It measured `perf_counter()` from *after* the
   resume load, so a resumed run reported a total smaller than the sum of its own aggregate rows.
7. **OpenJPEG preflight ran too late.** The first `assert_j2k_runtime()` fired inside
   `encode_to_budget` on the first encode — after the results directory, the outage policy and the
   frozen classifier had been touched — contradicting the docstring on `src/env.py:119` and SR-21.
8. **Two normative questions were still open**: the CIFAR-10 axis conflict, and the arithmetic
   meaning of `header_bytes`/`payload_bytes`.

### Why the old verifier passed

`tools/verify_w4_baseline_integration.py` verified **consistency between committed artifacts**, not
**whether each artifact describes the cell it claims to describe**. It re-hashed the CSVs,
re-derived the outage class from the manifest and recomputed every aggregate *rate* from the
per-image rows — all genuinely useful, and all blind here. It never loaded a `RunConfig`, never
recomputed `noise_id`, `pair_id`, `analysis_cell_id` or `run_id`, never parsed a codestream, never
read a row timing, and never opened a raw-row file at all. Its one configuration check compared
`resolved_config.json`'s `config_hash` against `smoke_summary.json`'s — and both carried the same
wrong hash, so the check passed by construction.

The `Psot` defect deserves a note of its own, because it is the clearest example of why counting
identities are not enough. At offset 4 the parser reads `Isot || high16(Psot)`. For these small
single-tile-part codestreams `Isot = 0` and `Psot < 65536`, so the read yields **zero**, which routes
into the legitimate `Psot = 0` last-tile fallback and lands on the correct boundary *by luck*. The
committed PB_2 overhead numbers were therefore not wrong — but `header + payload == len(codestream)`
held for a reason unrelated to the parser being right, and a multi-tile-part or large codestream
would have been mis-split with no error. This is why C2.4 added known-answer fixtures asserting the
two counts **individually** rather than only that they sum.

### The three amendments

**AM-80 — CIFAR-10 codec axes.** `params.baseline.downsample_axis_px.cifar10` becomes `[32]`. A flat
`j2k_resolutions = 6` requires every tile dimension to be at least `2**5 = 32` px, so OpenJPEG
hard-errored at the 24 px and 16 px rungs for every image and every budget. They were invalid codec
configurations, not low-rate candidates. The rejected alternative — an axis-dependent or clamped
resolution rule — would have added a new codec rule and more cache-identity complexity without
helping either headline dataset, and CIFAR-10 is a DEC-1 plumbing smoke path whose 32 px rung works.

**AM-81 — BR-11 byte semantics, and `analysis_version` 1 → 2.** `bytes_sent = source_bytes = A/8`.
`header_bytes` is **all** structural codestream bytes — SOC, every main-header marker segment, every
SOT marker segment, every tile-part header through and including SOD, EOC, and each tile-part's
equivalent structural bytes. `payload_bytes` is **all** tile-part data bytes after SOD and before the
next tile-part boundary; it is deliberately *not* described as pure entropy-coded sample data,
because that region may also carry packet-header information and the narrower wording would be
false. `emitted_codestream_bytes = header_bytes + payload_bytes` exactly, and
`payload_filler_bytes = bytes_sent − emitted_codestream_bytes` is reported separately, never folded
into either column. Both columns are means over every row that **emitted a codestream** — delivered
*and* decode-failure — excluding the two infeasibility verdicts, and are null only when nothing was
emitted. Redefining an aggregate column's meaning and denominator is an analysis-implementation
change under `params.config.analysis_version_bump_rule`, so the version bump is required rather than
optional; it intentionally re-namespaces every `run_id` and `config_hash`.

**AM-82 — the transparency-probe codec-configuration binding.** This one was not anticipated by the
instruction and was adjudicated with the user before any spec edit. `downsample_axis_px` sits inside
the content-addressed JPEG 2000 codec-configuration snapshot (`src/baseline/j2k.py:107`), so AM-80
moves that hash from `1a0b0d74bef1caed…` to `2daf597fd914f56e…` — **for every dataset**, and with it
every J2K cache key. `tools/verify_transparency_bitrate_probe.py:754-755` compared the probe's
recorded hash against a **live** `J2KCodec` built from HEAD, so AM-80 alone would have failed a
verifier that PB_2C §11.5 requires to pass, while §15 forbids re-running that 68,000-cell campaign.
G-1 and G-7 were unaffected because they hash the *archived* configuration; G-2 was unaffected
because it binds `spec/params.generated.yaml` as history.

The resolution follows the G-2 precedent exactly: the probe's codec configuration is now bound as
**history**, verified by reproducing the archived snapshot's own recorded hash, and the difference
from HEAD is permitted only by a single byte-pinned off-measurement-path record
(`results/probes/transparency_bitrate/codec_configuration_readjudication.json`). That record names
the amendment, the archived evidence commit, both hashes, the exact changed parameter path with its
old and new values, the probe's exclusive `imagenette160` dataset, and a reachability argument. The
verifier does not take the argument on trust: it computes the archived-vs-current difference set and
requires it to **equal** the declared paths, recomputes the probe dataset's configured axis ladder
under both snapshots and requires them identical, refuses any drift touching the probe's own dataset,
refuses a stale record when nothing has drifted, and pins both hashes. Thirty mutation tests show
that changing `j2k_resolutions`, the wavelet, the progression order, the code-block size, the tile
size, the rate control, the search settings, the cache key, the implementation version, the
preprocessing interpolation or the OpenJPEG version **still invalidates the probe**. It is a single
record, not an allowlist. The probe was not re-run.

### Old versus corrected provenance

| | PB_2 | PB_2C |
|---|---|---|
| configurations resolved | 1, at 18 dB | 5, one per cell |
| in the fingerprint | dataset, ratio, SNR, seeds, system, variant | + modulation, LDPC rate, encode axis |
| 18 dB vs −8 dB | same hash `ba59d1e7…` | `676f0311…` vs `cec413d8…` |
| archived configurations | none | `run_configs/<config_hash>.json`, each reproducing its own hash |
| `resolved_config.json` | one configuration | schema-2 execution index over all five |
| infeasible-row `noise_id` | empty | the scheduled identity |
| decode-failure overhead | blank | 157.0 / 892.54 bytes |
| `analysis_version` | 1 | 2 |

### Corrected bounded evidence

55 rows in 50.0 s from the clean commit `76e789c9f3d0`, fresh cache namespace. Crash-resume drill
first: three rows, stop, resume, no recomputation, no duplication, per-cell identities intact, and
cumulative timing rising from 1.387 s to 4.218 s across the resume — the pre-resume rows really are
counted now.

**The scientific outcomes did not move.** `n`, `top1_acc`, `n_correct`, `coverage_rate`,
`decode_failure_rate`, `infeasible_rate`, `acc_given_delivery`, `psnr_db`, `ssim`, `papr_db`,
`bytes_sent`, and every per-image `true_label`/`pred_label`/`correct` are identical to B2.6.
CIFAR-10 is 5/5 delivered with no task score; Imagenette-160 is 24/24 delivered at 18 dB with
`top1 = 18/24 = 0.75` and PSNR 27.034 dB, and 24/24 real decode failures at −8 dB with
`top1 = 3/24 = 0.125` through the frozen class-0 outage prediction. **None of these accuracies is an
experimental result**; they are plumbing observations at n = 24.

Exactly four families moved, each one a defect being corrected: the −8 dB cell's byte columns
(blank → 157.0 / 892.54), `config_hash` (one → five), `run_id` (which follows, because `config_hash`
and `analysis_version` are both keyed), and `wall_clock_s` (18 dB 19.41 → 25.21 s), which rose
because the timer now covers scoring. The codec-infeasibility per-image row's `noise_id` went from
empty to `7f9a9850e506…`, and its `pair_id` with it — that single row is the whole point of the
pairing repair.

`results/baseline/w4/overhead_table.json` now exists. BR-11's verify clause has always required an
archived overhead-fraction table and none had ever been produced. It is declared bounded —
`evidence_scope: bounded_integration`, `complete_for_full_validation_grid: false` — and lists exactly
the three executed cells with no synthesised combination. CIFAR-10 at `r_1_2` spends **82.6%** of its
transport block on container bytes; Imagenette-160 at `r_1_24` spends **14.8%**. That contrast is
precisely the first-order effect BR-11 exists to make visible, and it was invisible at −8 dB before
this repair.

### What the verifier can now see

`smoke_rows.jsonl` is finalised atomically in worklist order and the partial file is removed, so a
truncated run cannot be read as evidence. The verifier reconstructs every archived `RunConfig` and
requires its hash to *come out of* it; recomputes `noise_id`, `pair_id`, `analysis_cell_id` and
`run_id` per row; requires `per_image.csv` and `smoke_rows.jsonl` to describe the same rows;
recomputes both byte identities per row and every aggregate mean over the emitted-codestream
denominator; and checks the timing, the OpenJPEG version and the preflight ordering. All 26 required
mutation classes fail for their own independent property rather than because some downstream hash
moved. One gap surfaced while writing them and was closed: a *deleted* raw row was initially
self-consistent, because the row count and worklist digest are recomputed from what remains — the
per-image CSV is now the independent witness.

### Remaining frontier

PB_3 (BR-4 selection infrastructure) is not started and is now **unblocked**: the CIFAR-10 axis
question is settled by AM-80 and the BR-11 semantics by AM-81. The full BR-4 validation sweep has not
run, G-8 is unresolved, no ratio or operating point has been selected, no model has been trained or
fine-tuned, λ is uncalibrated, ER-9 is unimplemented and the test split is sealed until G-12 at W11.
The cache-key field-spelling issue recorded at PB_1 is unchanged and remains deliberately untouched.

---

# PB_3 — BR-4 selection infrastructure, built and not executed

**This closes W4.** PB_3 implements the analytic validation-selection machinery that BR-4 requires
and G-8 will later run, adds the W4 integration adjudication, and extends the verifier to cover both.
It ran no sweep, opened no gate, selected no operating point, trained nothing and never touched the
test split. Starting point: the PRE_B3 green `81372a5f1139bbfa9e086d229bf807c7cf6a8bce`, with
`3324393a3e1692478bba8cf1020708bf52947f6d` (PB_2C) still the latest scientific-evidence green.

## The composition, and why both of its inputs are types rather than floats

AM-51 writes the arithmetic into the specification because every selection in the project flows
through it:

    P(TB success)     = product over code blocks of (1 - BLER_r)
    expected accuracy = P(TB success) * acc_clean + (1 - P(TB success)) * acc_outage

Under a transport-block CRC one failed code block kills the transport block, so the blocks compose
multiplicatively. That much is uncontroversial. The interesting part is AM-58's requirement that
`acc_outage` be the **measured** validation accuracy of the frozen constant class rather than
`1 / classes` — and the reason the requirement needed writing down at all is that on this project's
committed manifest the two are numerically identical. `data/manifests.py` enforces an exactly
stratified validation split, all ten Imagenette-160 classes tie at 100 of 1000, and the configured
lowest-index tie-break selects class 0 at `100/1000 = 0.1 = 1/10`.

So a substitution of the assumption for the measurement would produce **the same number today** and
a wrong one the moment the split stopped being exactly stratified. A test comparing floats could
never see it. `MeasuredOutageAccuracy` therefore takes a numerator, a denominator, the selected class
and a provenance string — never a ratio — and `expected_accuracy()` is keyword-only and typed, so a
bare float cannot be passed for either accuracy term at all. `MeasuredCodecAccuracy` is the same
shape for `acc_clean`, which BR-4 calls "a required measured input ... which is a real artifact and
MUST be produced rather than assumed"; it additionally refuses any split but `val`, because BR-4
selection is a validation-split activity and the test split is sealed until G-12 (SR-22).
`measured_outage_accuracy_from_record()` reads PB_2's committed artifact by its **counts**, refuses a
record whose recorded float disagrees with them, and refuses a record written under a different
`baseline.outage_policy`.

The test that carries this is deliberately not run at the committed value:
`test_measured_inputs_are_passed_through_not_reconstructed` composes against a **137/1000** outage
measurement, so any implementation that quietly rebuilt the term from the class count disagrees.

One smaller decision worth recording: an empty code-block sequence **raises** rather than returning
the vacuous product `1.0`. A transport block always has at least one code block, and "no blocks were
characterised" reporting as "certain success" is exactly the silent-optimism failure this phase
exists to prevent.

## The BLER evidence characterises one configuration, and the lookup says so

This is the part of PB_3 with the largest capacity to manufacture a result quietly. The committed
G-2 evidence — `results/baseline/g2/bler_results.csv`, 24 rows — characterises **one** physical-layer
configuration: `K=128, N=256, BG2, Z=22, rate 1/2, flooding offset-min-sum, offset 0.5, 50
iterations, 5000 blocks per point`, at four Eb/N0 points for each of BPSK, QPSK and 16-QAM. That is
the entire measured support. BR-4's sweep, by contrast, ranges over six bandwidth ratios, three
modulations, four LDPC rates and four codec axes per dataset — thousands of cells, almost none of
which that evidence describes.

The lookup is therefore keyed on the **complete** identity: all eight fields of
`params.baseline.ldpc_bler_reference_must_match` (`k_and_n`, `base_graph`, `lifting_size`,
`modulation`, `decoder_algorithm`, `decoder_offset`, `iterations`, `snr_convention`) **plus `rate`**,
which the committed evidence fixes and the spec's must-match list does not name. Requiring more than
the spec does is the safe direction: a curve measured at rate 1/2 says nothing about rate 5/6 at the
same (K, N). A key missing any required field raises `BlerLookupError`; a key carrying an
unrecognised field raises too, rather than being trimmed to fit.

An identity that is not an exact match returns `uncharacterized`. An SNR outside the curve's measured
span returns `uncharacterized`. In both cases `bler` is `None` — **never `0.0`** — and `.require()`
raises `UncharacterizedBlerError`. The distinction between that exception and `BlerLookupError`
matters: a partial key is a caller bug, whereas an uncharacterized cell is a legitimate answer the
selection must act on by treating the candidate as **ineligible**.

That last point is the one worth being explicit about. An uncharacterized candidate is *not* a
low-scoring candidate. Scoring it at all — at any value, including a pessimistic one — means
inventing the evidence that would justify the score. `select_best()` never ranks it, and
`evaluate_candidate()` marks the whole transport block uncharacterized if **one** of its code blocks
is, rather than composing over the blocks that happened to resolve.

Interpolation is permitted only strictly inside the span and only in the representation
`bler_reference.json` itself declares (`waterfall_interpolation: linear_in_snr_vs_log10_bler`); the
module raises `NotImplementedError` if that declaration ever changes, and refuses to interpolate
through a non-positive measured BLER. The test checks the midpoint against `sqrt(a*b)` — the
geometric mean — rather than against the implementation's own weighted log-average, so it is a
genuine cross-check and not the code restated.

Both declared SNR conventions are carried: the reference platform's own `eb_n0_per_information_bit`
and the `es_n0_per_symbol` column derived from it by the per-modulation conversion
`g2_adjudication.json` records. They are **distinct identities**, so reading an Es/N0 number under
the Eb/N0 convention does not resolve — which is a real confusion the guard catches, since 16-QAM's
Es/N0 waterfall at ~7.9 dB sits far above its Eb/N0 span of 4.0–5.25 dB.

Finally, the table is **hash-bound**: it is built from `bler_results.csv` only if the file's SHA-256
still matches the value `g2_adjudication.json` records for it. Composing a selection against curve
bytes the G-2 gate never adjudicated would be a provenance failure that no downstream number would
reveal.

## Caching, and the field that is excluded on purpose

BR-4 requires the sweep to be a cached feasibility table rather than a per-image simulation.
Structural feasibility — does a legal TS 38.212 packetisation exist, and can the codec emit inside
the resulting payload budget — is expensive and deterministic, so it is computed once per
configuration.

The entire risk of a cache is a shared key. `Candidate.__post_init__` therefore asserts that every
declared field is classified either into `FEASIBILITY_KEY_FIELDS` or into
`FEASIBILITY_KEY_EXCLUSIONS` **with a written reason**; a field added to the candidate later cannot
silently fall out of the key, because construction fails until someone classifies it. There is one
exclusion, `snr_db`, and its reason is that structural feasibility reads the transport-block geometry
and the codec payload budget, neither of which is a function of the channel SNR — the SNR enters the
composition through the BLER lookup instead. A test asserts the exclusion behaves that way rather
than trusting the comment, and five more assert that changing any keyed field forces a fresh
computation.

## Tie-breaking, stated rather than emergent

`TIE_BREAK_ORDER` is applied left to right and only among candidates whose expected accuracy is
**exactly** equal: expected accuracy descending, then `P(TB success)` descending, then `Qm`
ascending, then LDPC rate ascending, then encode axis descending, then the candidate's canonical
identity string ascending. After accuracy the order prefers the more reliable link, then the more
robust modulation, then the stronger channel code, then more source information — and the final key
makes the order **total**, which is the property that makes the selection independent of the order
candidates were enumerated in. A test checks all 24 permutations of a four-candidate set.

Equality is exact float equality, not equality within a tolerance. A tolerance would be an
unpreregistered free parameter sitting directly on the selection, which is the kind of knob DEC-16's
governing rule exists to keep out. The spec fixes the objective function and says nothing about
ties, so any deterministic total order satisfies it; what is not acceptable is an order that emerges
from dictionary iteration and changes between runs.

## Three modes, and two passes

`params.artifacts.system_values` carries `classical_adaptive`, `classical_fixed_mod` and
`classical_fixed_mcs`, and they are three different experiments rather than three labels. They are
represented as policies over *what may move between SNR points* — `(adapts_modulation,
adapts_ldpc_rate, adapts_encode_axis)` = `(T,T,T)`, `(F,T,T)`, `(F,F,F)` — and `resolve_curve()`
makes them behave differently on the same candidates: adaptive re-selects per SNR, fixed-modulation
picks the one modulation whose per-SNR bests sum highest and adapts underneath it, and fixed-MCS
selects once at `baseline.fixed_mcs_design_snr_db` (7 dB) and holds the whole configuration. If the
design SNR is not a point on the supplied grid, fixed-MCS **refuses to snap** to a neighbour: letting
it would make the fixed arm's design depend on grid spacing. PB_2's bounded run was
`classical_fixed_mcs` precisely because it fixed one configuration and built no adaptation; PB_3
builds the adaptation but does not run it.

AM-54 caps selection at two passes: pass one under BR-8's clean-trained classifier, then BR-12 trains
on the corpus those selections define and re-scores the cached sweep once, and iteration terminates
there. The rationale is preregistration — a loop that may be re-run can be run until it flatters a
result — so the cap is enforced by `SelectionCampaign`'s state machine rather than by documentation.
A third pass raises; so does a repeated pass, an out-of-order pass, a non-integer or unknown
identifier, a pass reusing an earlier pass's scorer, and a **resumed** state that repeats a pass,
carries an unknown pass or ran under another mode. `PassContext.result_of()` refuses any pass
identifier at or above the current one, so pass one cannot read pass two even through a context
retained past its own pass, and `PassResult` is frozen, so pass two cannot edit pass one.

PB_3 does not train the artifact-finetuned classifier. `reference_classifier.artifact_finetune_gate`
is G-8; the second-pass scorer is an argument the caller supplies, and this repository supplies none.

## The sweep guard

This is the single mechanism standing between an ordinary call and an accidental G-8 campaign, so it
is worth stating what it does and, more importantly, what is deliberately missing from it.

`select_operating_points()` runs `check_sweep_budget()` **before any work**. Unauthorized limits are
64 candidates, 25 samples per cell and a combined workload of 512 cells — three separate checks,
because a sweep can be too large in three ways, including as the product of two individually-modest
numbers. Above any of them the call raises `SweepBudgetError` unless a typed `G8Authorization` naming
gate `G-8`, an authoriser and a reason is passed explicitly, and an authorization permits only the
limits it declares.

Deliberately absent, each asserted by a test rather than left as a claim: **no environment variable**
is read anywhere in the module (a variable exported once in a shell profile is how a guard gets
disarmed and stays disarmed); **no default-true flag** — `authorization` defaults to `None` on every
entry point, checked through `inspect.signature`; and **no tracked non-test file in this repository
constructs an authorization**, checked by scanning `git ls-files`. The validation runs both at
construction and at use, so an instance built through `object.__new__` — which skips
`__post_init__` — is still refused, as is any duck-typed look-alike. The boundary probes sit at
exactly the limit (accepted) and one above it (refused), which is what makes a mutation from `>` to
`>=` fail rather than pass.

## The adjudication, and what the verifier now recomputes

`results/baseline/w4/integration_adjudication.json` is generated by
`tools/gen_w4_integration_adjudication.py`, not hand-written: every hash is computed from the files
on disk, the selection-machinery description is read out of the module itself, the worked composition
example is computed by the real `compose()`, and the characterised BLER identities are enumerated
from the committed curves. It states, in machine-readable fields and in prose, that this is bounded
validation/plumbing integration, **not** the BR-4 full validation sweep, **not** a G-8
operating-point selection and **not** test evidence, that G-8 remains unresolved and that the test
split stayed sealed.

Two binding decisions are worth recording:

* it carries **no** `evidence_commit`, null or otherwise. A file cannot contain the hash of the
  commit that adds it, so the commit is resolved from Git path history the way G-2's is
  (`git log -1 --format=%H -- results/baseline/w4/integration_adjudication.json`), and the verifier
  *rejects* any stored value;
* `src/baseline/classical/composition.py` is **not** added to
  `results/baseline/w4/execution_source_manifest.json`. That manifest binds the 40 sources that
  participated in the bounded measurement at `76e789c9f3d0`, where this module did not exist —
  adding it would claim it participated in a measurement it postdates. It is bound instead under its
  own `selection_sources` role, at HEAD, alongside its test module.

`tools/verify_w4_baseline_integration.py` gained two checks. `check_integration_adjudication()`
recomputes rather than reads: bound evidence hashes from disk, selection-source hashes and byte
lengths at HEAD, the worked composition example through the real composition function, the BLER
characterisation from the committed curves, and the selection-machinery and sweep-guard descriptions
from the module. It cross-checks the outage counts against `outage_policy.json`, requires the outage
accuracy to be its own numerator over its own denominator, requires all four test-access counters to
be zero and the release gate to be the configured one, and requires all three provisional bandwidth
values — `efficiency_ratio`, `crossover_ratio`, `low_ratio_operating_point` — to still hold their
committed values and their `provisional_until_G-8` status.

`check_selection_machinery_behaviour()` then exercises the module live, so the adjudication describes
behaviour that still holds at HEAD rather than behaviour that once did: an incomplete BLER key must
raise, a wrong identity and an out-of-support SNR must both come back uncharacterized, no two
distinct configurations may share a feasibility key, tie-breaking must be order-independent, a third
pass and a pass reading its own or a later pass must both raise, and the sweep guard must refuse an
over-budget unauthorized workload. Every probe is unit-scale — no dataset, no codec, no channel — and
the guard probe is deliberately the *refusing* path, so running the verifier can never start a sweep.

`tests/test_w4_integration_adjudication.py` (59 tests) mutates one property at a time and asserts the
failure comes from that property: altered arithmetic, an incomplete BLER key, an uncharacterized
candidate reported as characterized, silent extrapolation, an altered outage measurement, a cache
collision, a nondeterministic tie, a third pass, leaked pass state, a bypassed sweep guard, an
environment-variable bypass, an adjudication falsely claiming a full sweep or G-8 resolution, each of
the three provisional ratios changed or declared settled, each test-access counter drifting, and a
stale evidence hash, selection-source hash or byte length.

## Amendment judgment — no amendment required

PB_3 implemented and verified existing specification semantics without changing them. The composition
is AM-51's, transcribed; the measured-outage requirement is AM-58's, enforced structurally; the
complete-identity BLER key is `params.baseline.ldpc_bler_reference_must_match`, read at runtime; the
two-pass cap is AM-54 and `reference_classifier.br4_selection_terminates_after_pass`, both read at
runtime; the three modes are `params.artifacts.system_values` entries and
`baseline.modulation_tuning`; the fixed-MCS design point is `baseline.fixed_mcs_design_snr_db`. No
requirement, gate, decision or frozen parameter moved, and the three provisional bandwidth values are
byte-identical and still `provisional_until_G-8`.

Three choices are stricter than the specification and none of them is a change to it: requiring
`rate` in the BLER identity (the spec's must-match list omits it; the committed evidence fixes it),
the documented tie-break order (the spec fixes the objective and is silent on exact ties), and the
sweep budget (an implementation safety boundary with no scientific content). Recording any of these
as an amendment would use the amendment record to dignify an implementation detail, which §17's
preamble is explicitly against.

## PB_3C — the fixed-modulation reference, resumed state, and the frozen selection policy

PB_3 was substantially right and is not reopened. Four things were corrected.

**The fixed-modulation curve searched for its modulation.** BR-9 says
`params.baseline.core_modulation` *defines* the fixed-modulation reference curve. The
`classical_fixed_mod` branch of `resolve_curve()` instead enumerated every modulation present in the
supplied grid, summed each one's per-SNR best expected accuracies and kept whichever total was
highest. That is a second optimizer wearing the reference arm's label, and on a grid where BPSK
dominates it would have reported a BPSK curve as the QPSK reference. The correction reads
`baseline.core_modulation` through the config interface — the value is never written in source — and
checks it against `baseline.modulations`. The old test could not have caught this: it asserted only
that one modulation was held across the grid, which a searching implementation also satisfies. The
replacement asserts the held modulation *is* the configured one, on grids where BPSK and where
16-QAM each win the whole-grid total outright, with the adaptive curve on the same grid confirming
the dominant modulation really would have been chosen.

**Three cases are now distinguished, and the third is the interesting one.** An undeclared
`core_modulation` is a contradiction between two parameters and raises. No candidate using it at a
required SNR is an incomplete candidate grid and raises, naming the SNR. But candidates that exist
and are *all* infeasible or uncharacterized do **not** raise and are **not** replaced: the cell is
preserved with `selected = None` and `reason = no_eligible_candidate`, carrying every evaluation.
Raising there would have been the tidier-looking choice and the wrong one — structural
infeasibility, codec infeasibility and missing BLER characterization are exactly what G-8's
completeness preflight exists to refuse, and it cannot refuse a cell that was deleted or that
terminated the resolve.

**Resumed campaign state was trusted rather than validated.** `run_pass()` enforced seven
invariants; `_admit_resumed()` enforced four. So the crash-recovery path — the whole reason the
campaign is serializable — honoured the weaker contract, and would accept pass two with no pass one,
a reversed `(2, 1)`, a gap, a blank or duplicated scorer, or a `PassResult` whose `selections` held
strings. Resumed state is now required to be an **exact ordered prefix** of `selection_passes()`:
`result.pass_id == allowed[i]` at each index, validated in the order supplied and never sorted,
because sorting malformed state into validity hides precisely the corruption worth catching. Four
shared helpers carry the invariants and both paths call them, so they cannot drift again;
`run_pass()`'s `len(completed) + 1` arithmetic became `allowed[len(completed)]`, and a test
monkeypatches the sequence to `(1, 2, 3)` to prove nothing assumes passes run 1 then 2.

**The tie-break order did not change; its status did.** The order was already good and already
documented. The problem was that it could still be revised after the BR-4 table existed, and a
ranking rule chosen after seeing the data is not preregistered. It is now recorded in the
adjudication as frozen before G-8, checked field-by-field against the live module, and reduced to a
`selection_policy_sha256` over canonical serialization of six policy fields — tie-break order, tie
equality, the fixed-modulation source and configured value, the pass sequence and the termination
pass. Recording the order alone would not have been enough, because implementation and generator
could be edited together; the verifier therefore recomputes the digest independently, and a future
G-8 campaign manifest must bind it alongside the adjudication's own SHA-256 and refuse to resume or
adjudicate if either differs. Exact float equality remains the definition of a tie — no tolerance
parameter was added, and the verifier asserts that by name.

The adjudication `schema_version` moved **1 → 2**. The verifier now *requires* fields version 1
never emitted, so a version-1 artifact can no longer satisfy it; additive-but-mandatory is still a
break for any reader that trusts the version, and this repository has no rule exempting it. That is
an adjudication-schema change only — no PB_2C measurement artifact, no bounded execution, no
parameter and no specification text moved.

**No amendment.** The spec defines no BR-4 selection tie-break parameter at all
(`baseline.outage_class_tie_break` is BR-13's outage class, `reference_classifier.checkpoint_tie_break`
is which epoch to keep), so freezing an implementation-level total order contradicts nothing. And
the fixed-modulation repair *restores* BR-9's existing semantics rather than altering them — the
same situation as PB_1C, which also needed none. Every existing requirement is satisfiable without
changing any of them, which is the bar §17's preamble sets.

## Remaining frontier

W4 is complete, including PB_3C. **G8_B is complete through B6 at
`G8_B/tooling_smoke_complete`; G8_C is ready but not started.** The authenticated runner,
exact resume/merge validation, atomic state machinery and bounded smoke are green. No
full-strength characterization, BLER table, selection, authorization, inference, training,
validation decoding or test access has occurred; the production runtime root is absent and the
test split remains sealed. The exact G8_C restart command is stored in the campaign state and is
`rg -n "G8_C|characterization_open|full_strength|run_g8_bler|resume_plan|merge_report|tooling_smoke_complete" src/baseline tools tests instructions`
and is not executed here.

The committed G-2 BLER evidence characterises exactly one physical-layer identity (`K=128, N=256,
BG2, Z=22, rate 1/2, offset-min-sum 0.5, 50 iterations`) at four SNR points per modulation. It is a
conformance artifact. It remains valid for G-2 and **must not be extrapolated or generalised** into
the BR-4 characterization table; the lookup already fails closed outside it, and that behaviour is
to be preserved rather than worked around.

G-8 must: (1) enumerate the complete **structural candidate/configuration grid** and the code-block
identity grid; (2) identify every required `(rate, SNR, block identity, modulation)`
characterization; (3) run and archive **full-strength BR-4 physical-layer BLER characterization** at
the configured trial count; (4) build a separate hash-bound G-8 `BlerTable` artifact and loader;
(5) verify complete coverage before selection; (6) generate cached codec reconstructions and
measured clean-classifier accuracies on validation; (7) construct measured codec-accuracy objects
from verified artifacts rather than manual counts; (8) execute pass one; (9) build the training-only
artifact corpus; (10) fine-tune the artifact classifier; (11) execute pass two once; and
(12) adjudicate the operating ratios and the other G-8 outputs.

Two terms are load-bearing. It is a *structural* candidate grid, not a "feasible" one — codec
feasibility is unknown until the codec-search artifacts exist, while structural transport identities
can be enumerated first. And it is *physical-layer* BLER characterization, not "validation" BLER —
BLER is a channel-simulation artifact; "validation" is reserved for codec and classifier records
derived from the validation split.

G-8's outputs are `efficiency_ratio`, `crossover_ratio`, `low_ratio_operating_point`, classical
non-degeneracy, the one-ratio-versus-two-ratio full-strength ER-1 decision, the artifact-finetuned
classifier release, the final pass-two classical selections, and the frozen H2 validation window.
Note that G-8 selects the *parameter named* `crossover_ratio` by ER-3's learned-blind classical
rule; it does **not** decide whether a learned-versus-classical curve crossover exists, which stays
at G-10 after learned models exist. Changing the tie-break order once the sweep has started
invalidates the campaign.

PB_3C's historical note remains unchanged, but B6 has now completed the G8_B tooling gate. As of
this handoff: the full BR-4 sweep has not started, no G-8 characterization has run, G-8 remains
unresolved, no real `G8Authorization` exists in any tracked non-test file, no bandwidth ratio or
operating point has been selected, no model has been trained or fine-tuned, the artifact-finetuned
classifier does not exist, λ is uncalibrated, ER-9 is unimplemented, and the test split is sealed
until G-12 at W11. PR-1, PR-2 and PR-9 remain outstanding programme deliverables.

## G8_A opening — durable partition and corrected terminal provenance

G8_A began from clean local/origin/remote parity at `39c43e327573f33011c561c6de22bd05ff93c068`. The campaign is partitioned before scientific data inspection under `instructions/G8.txt` and phase instructions G8_A through G8_G: contract/preflight; characterization tooling and bounded smoke; full BLER characterization and table freeze; validation-measurement tooling and bounded smoke; full validation measurement plus pass one; training-only artifact corpus/fine-tune plus the single pass two; and final adjudication. Later phases may not silently reinterpret earlier artifacts.

The PB_3C terminal handoff provenance is explicit rather than inferred from an intended subject: terminal SHA `39c43e327573f33011c561c6de22bd05ff93c068`, actual subject `fix: fix push failure due to gpg for resume.md`; implementation/adjudication checkpoint `08dd358c0f1bd55c70152af900f2932f50d95d19`; PB_3 implementation green `32edbbb58983e54103b2f252c4d8d8f30aa2378e`; latest scientific-measurement green `3324393a3e1692478bba8cf1020708bf52947f6d`. History is preserved.

G8_A is complete. It froze the pre-data contract, enumerated structural candidates and required physical-layer identities, and added state/preflight verification without running characterization, loading validation pixels, measuring accuracy, selecting anything, training, issuing authorization, invoking fallback, or accessing test. G8_B active. B0, B1, B1C, B2 and B2C complete. B3 next — implement exact resume and merge validation. No runner, simulation, smoke or characterization has started. G8_C–G8_G remain prohibited. B1 remains a historical design checkpoint; B1C corrected and hardened the executable contract before data. The corrected tooling, request and result schemas are version 2, no version-1 request or result exists, and B2 and later must use the corrected contract only. The characterization runner does not exist yet. B2 froze deterministic sharding, safe paths, closed unit snapshots and atomic publication without interpreting state history.

## G8_A green — contract frozen, G8_B released

G8_A enumerated 12,096 structural candidates over the headline and already-specified fallback roles, 144 packet configurations, and 3,213 unique physical-layer BLER work units. Exact G-2 coverage is 0/3,213: all required cells differ in physical identity, 24 measured G-2 convention/point records lie outside the required set, and neither interpolation nor extrapolation was used. This expected insufficiency releases characterization *tooling* work, not scientific execution.

`results/baseline/g8/campaign_manifest.json` binds the W4 adjudication, selection-policy fingerprint, selection sources, normative spec/generated parameters, split-manifest bytes, phase order and G8_A contract sources. `required_bler_identities.json` is generator-owned; `campaign_state.json` is crash-safe and manifest-bound. B0 opened it at `G8_B/tooling_open` with no work units and all decoding/inference/training/test counters zero, B1 registered `bler_tooling_contract.json`, and B2 registered `bler_state_contract.json` without touching a counter or work unit; B3 is next. Later phases may not silently reinterpret these artifacts.

## G8_B B0 — verify G8_A and open the phase (complete)

G8_B is active; B0, B1, B1C, B2 and B2C are complete and B3 is next. The first B0 command after the marker was
`.venv/bin/python tools/gen_g8_campaign_manifest.py --check`; the exact B1 restart command is
`rg -n "trials_per_point|bler_trials|seed|BlerIdentity|run_ldpc_g2" spec/SPEC.md spec/params.generated.yaml tools/run_ldpc_g2.py src/baseline`.
No characterization, sweep, validation decoding, inference, training, test access, ratio
selection, authorization, codec image load, or smoke execution has started. G8_C–G8_G remain
prohibited. The G8_A manifest, required-identity artifact, and all immutable G8_A sources remain
byte-bound and may not be modified.

The marker commit was `3f0fc4d945756f22a5a655ce614a3f0b001b4735`; the phase-opening utility and
current-phase verifier were added with 24 focused tests. The pre-transition full suite was 1204
passed, and the exact transition was once from `G8_A/preflight_complete` to
`G8_B/tooling_open`, producing state SHA-256
`f7b21df77f812d68ca55bf92dc78a1ec0b003be89189170983f04f093205c7ed`. The campaign ID, manifest
hash, required-identity hash, produced-artifact bindings, and zero counters are unchanged.

## G8_B B1 — freeze runner schemas and seed derivation (complete)

B1 froze the scientific and machine-readable contract the later G8_B checkpoints must implement.
It created contracts, pure validators and seed utilities only: **the characterization runner does
not exist yet**, no sharding, no per-unit checkpoints, no merge logic, no simulation and no smoke.

The trial count is **5000**, read only through `params.baseline.bler_characterisation_trials`.
The G-2 reference key `params.baseline.ldpc_bler_reference.blocks_per_snr` currently holds the
same value but belongs to the narrow G-2 experiment; the contract records it as excluded, and a
mutation test moves it to 7 while the G-8 count stays at 5000. A second mutation test moves
`bler_characterisation_trials` and the contract follows it.

Seeds bind `campaign_id`, `work_unit_id`, `purpose` and a domain separator under the identity
already recorded in the live state, `sha256(campaign_id,work_unit_id,purpose)-v1`, with separator
`capstone:g8:bler-seed:v1`. The pre-image is the compact UTF-8 JSON *array*
`["<domain>","<campaign_id>","<work_unit_id>","<purpose>"]`, so no dictionary ordering and no
whitespace is identity and JSON escaping makes delimiter injection non-colliding; the seed is the
first eight SHA-256 bytes read big-endian, with no modulo and zero valid. The three scientific
purposes — `information_bits`, `awgn_real`, `awgn_imag` — are a closed set with separate streams,
so a change to information-bit generation cannot silently move the noise draws. G-2's
`_seed(root, modulation, snr)` was deliberately **not** reused: it cannot address a work-unit
identity or separate purposes. That is an implementation gap, not a normative contradiction, so
the judgment is **no amendment**.

The RNG contract was measured rather than assumed. `Generator.integers(0, 2, dtype=uint8)` is
**not** chunk-boundary invariant under NumPy 2.5.1 — its bit buffering makes the flat sequence
depend on how the caller splits the request — so it was rejected as the bit API. `Philox.advance(n)`
was measured to skip `4n` uint64 words. The frozen information-bit stream is therefore an indexed
raw-word stream with `bit_i = (word[i // 64] >> (i % 64)) & 1`, LSB-first, `uint8`, C order, with
trailing bits discarded and never carried between work units; it is invariant at boundaries
0/1/62/63/64/65/127/128, at irregular chunkings and at lengths not divisible by 64, and is also
randomly addressable. Gaussians use `Generator(Philox(key=seed)).standard_normal`, `float64`,
which is chunk-boundary invariant but consumed sequentially from index zero.

That is sufficient because B1 also freezes **work-unit-atomic resume**: only a complete,
atomically committed work-unit result is resumable evidence, an interrupted unit is discarded and
restarted from trial zero on the same seeds, there is no mid-work-unit trial cursor, and shard
layout or execution order can never change an output.

Counts are authoritative. BER, BLER and both confidence bounds are recomputed from
`trials_completed`, `information_bits`, `bit_errors` and `block_errors`; a stored float that
disagrees is rejected. Zero observed errors at the full trial count is valid characterized
evidence, as is every block failing; zero *completed* trials reports `null` rather than zero.
Bounded smoke is capped at 3 work units and 16 trials per unit, is visibly labelled
`NON-SCIENTIFIC BOUNDED SMOKE`, carries `scientific_evidence=false` and
`required_coverage_contribution=0`, and can never satisfy the full-strength validator even when
relabelled. The diagnostic interval is 95 percent Wilson score, project-owned: no general
confidence parameter governs G-8, and `baseline.ldpc_bler_reference.confidence_percent` is
G-2-specific and is not read. It is recorded as diagnostic only — not used in BR-4 ranking or
eligibility and not a stopping rule — and no adaptive stopping exists.

`results/baseline/g8/bler_tooling_contract.json` is generator-owned, with contract ID
`g8bler-878b218e60743dd5c85859348dfdbacdac847b344389d5688e182739e312dbbd` derived from canonical
content excluding the ID and independent of timestamps, absolute paths, hostname and commit SHA.
It binds only the three B1 sources and never its own hash or a future runner/shard/merge file.
The current-phase verifier now requires every original G8_A produced-artifact binding to remain
byte-identical while allowing validated additions, gained a repeatable `--require-artifact`, and
closed the B0 gap by naming the manifest's scientific base, interpretation rules and
selection-policy invalidation clause. A narrow registration utility added the one new binding
without touching a counter or work unit; state SHA-256 is
`4b47e62aca0cd61930d9e389908b3dbdfb31e4f6d0be452fc9cd2cf0dfc2c3ab`.

## G8_B B2 — deterministic sharding and atomic unit state (complete)

**G8_B active. B0, B1, B1C, B2 and B2C complete. B3 next — implement exact resume and merge
validation. No runner, simulation, smoke or characterization has started. G8_C–G8_G remain
prohibited.** B2 is infrastructure only: no required work unit was claimed, no request or result
was created, no BLER path was invoked, and no state directory was scanned for execution history.

The authenticated `AuthenticatedExecutionContext` binds campaign ID, manifest, required-identity,
selection-policy and corrected B1C contract ID/SHA, schemas 2, all 3,213 complete records and the
exact artifact order. It stores canonical bytes, tuples and read-only mappings; public lookups are
fresh decoded copies. Construction authenticates the required artifact once per context, and the
instrumented 92-test suite observed one loader call while performing record lookups and 11 shard
plans. `canonical_ordinal_modulo_v1` uses `ordinal % shard_count == shard_index`, preserves order,
and never enters seed derivation. Plans bind the full authority and a digest over their complete
identity. State paths use the exact UTF-8 work-unit-ID SHA-256 under a lowercase two-hex bucket.

Unit state schema v1 separates identity from closed runtime metadata and binds the corrected B1C
request/result schemas v2, canonical ordinal, record hash, shard ownership and plan digest. The
identity digest covers identity only. Exclusive creation uses `O_EXCL`/`O_NOFOLLOW` where supported,
file fsync and directory fsync; replacement is a canonical same-directory compare-and-swap with
expected previous SHA, fsynced temporary, atomic replace, directory fsync and reread. The
independent contract verifier exercises races, stale writers, malformed state, interruption
before/after replace and cleanup only in temporary directories.

The generated B2 contract is `g8state-77ff45564fbe282179a860d70f2cc509264d06e1855d7360a50994a4fabaaa7c`,
SHA-256 `2422c4c2a019c2a901cfd8732747555262dfca5601b28b0e700ff33743d4d939`, 9,390 bytes. The
registration-owned campaign state has four produced artifacts, SHA-256
`c75254513f2edc31a37957a5f2cfa13d532062e959f887091fabac04c4d91c92`, completed IDs `[]`,
in-progress `null`, and all four scientific counters zero. No amendment was needed: B2 freezes
implementation mechanics left open by the normative spec and contradicts no requirement.

## G8_B B2C — correct and harden unit-state publication before B3 (complete)

**G8_B active. B0, B1, B1C, B2 and B2C complete. B3 next — implement exact resume and merge
validation. No runner, simulation, smoke or characterization has started. G8_C–G8_G remain
prohibited.** B2C is infrastructure only: no required work unit was claimed, no request or result
was created, no BLER path was invoked, and no state directory was scanned for execution history.
It began and ended with `completed_work_unit_ids == []`, `in_progress_work_unit_id == null` and all
four scientific counters zero.

**Why B2C exists.** B2's sharding and schema direction was right and is kept unchanged. Its
publication primitives were not, and a line-by-line audit found nine defects that B3's exact resume
and merge validation would have inherited:

1. **First publication was not crash-atomic.** `create_unit_state_exclusive` opened the *final*
   pathname with `O_CREAT | O_EXCL` and wrote into it. The exclusivity was real, but a hard kill
   between `open` and `fsync` leaves a truncated file at the authoritative name, and a later reader
   cannot tell "never written" from "half written".
2. **Replacement was not linearizable.** It read the current state, compared its SHA-256 with
   `expected_previous_sha256`, then wrote a temporary and called `os.replace()` — with nothing held
   across that window. Two writers starting from the same predecessor digest could both pass the
   comparison and both report success, the second silently destroying the first. That is a
   read-then-write; the B2 contract described it as compare-and-swap.
3. **A descriptor could be closed twice**, in `except` and again in `finally`, so a real publication
   failure could reach the caller as a secondary `EBADF`.
4. **Symlink guards were fail-open.** `path.exists() and path.is_symlink()` follows the link, so a
   *dangling* symlink at the final name, the root, the bucket or a staging name reported
   `exists() == False` and passed. Validation also compared two `resolve()` calls derived from the
   same candidate, which proves nothing an attacker-controlled parent has not already decided.
5. **Directory durability was silently downgraded** — `EACCES` was treated as "unsupported" and
   every caller discarded the returned flag.
6. **Results were not request-bound and not terminal**: a `result_linked` state could carry a null
   `request_sha256`, and once linked its result path, result SHA, trials or attempt could still be
   rewritten by any writer holding the current digest.
7. **Trials could decrease**, a bound request SHA could change, and the scientific flag could go
   true → false.
8. **Unit states did not bind their own contract**, so a state written under the superseded B2
   contract was indistinguishable from one written under B2C.
9. **The independent verifier was not independent** — it imported every expected constant, field
   list and status rule from the module under test, so a mutation moved the "expected" values in
   lockstep and the verifier still passed. A permanent assertion that the runtime work-unit tree
   never exists was also scheduled to fail the moment B3 or B4 legitimately executes.

**What B2C installs.** Two explicit authority layers. `AuthenticatedExecutionContext` keeps the B1C
campaign/work-unit/sharding authority and remains sufficient for contract generation and shard
planning. `AuthenticatedUnitStateContext` wraps it and additionally authenticates the *registered*
B2C state-contract artifact against campaign state — path, byte count, SHA-256, contract ID, schema
version, checkpoint, supersession and source bindings — and is required by every unit-state build,
validate, read, create and replace. A plain execution context is rejected where the state layer is
required. The circular dependency is avoided rather than papered over: the contract artifact never
binds its own SHA-256, so the generator can build it before it is installed, and the external
SHA-256 that every unit state binds comes from the authenticated campaign-state binding.

Unit-state schema version 2 adds `bler_state_contract_id` and `bler_state_contract_sha256`. A state
binding the superseded B2 contract ID or SHA is rejected, as is one binding any contract other than
the registered B2C artifact.

First publication renders complete canonical bytes, opens the root and bucket descriptor-relative
with `O_DIRECTORY | O_NOFOLLOW`, rejects any object already at the final name by no-follow `lstat`,
creates a unique same-directory staging file with `O_CREAT | O_EXCL | O_NOFOLLOW` mode `0600`,
writes, flushes and `fsync`s it, then publishes with
`os.link(..., follow_symlinks=False)` against the bucket descriptor — a primitive that **cannot**
replace, so a regular file, symlink, dangling symlink, directory or any other occupant produces a
domain conflict. It then `fsync`s the directory, removes the staging name and rereads and validates
the installed bytes. The final pathname is never opened for writing and there is no fallback that
would do so; if the filesystem cannot supply the no-replace primitive the operation fails closed.

Replacement holds one exclusive per-unit critical section — `fcntl.flock(LOCK_EX)` on a canonical
lock file under `.locks/`, which can never collide with a two-hex-digit state bucket, plus a
process-local keyed lock so threads in one process cannot race either — and performs the reread,
the digest comparison, the transition validation and the publication inside it. Locks release on
normal exit, on exception and on process death.

A valid `result_linked` state requires a non-null lowercase-hex `request_sha256`, a result path and
SHA, `scientific_execution_performed == true`, `trials_completed > 0` and `test_split_access == 0`;
no result may exist without a request binding in any status. It is **terminal**: the only permitted
operation is exact canonical-byte idempotence, and different bytes raise `StateConflictError` even
when the writer supplies the current digest. Within one attempt, trials never decrease, the
scientific flag never goes true → false, a bound request SHA never changes or clears, shard
assignment is immutable, and `failed → claimed` and `failed → result_linked` are both forbidden. A
non-result state may begin a new attempt only as `old_attempt + 1` with a completely clean claim,
and **that transition is the only legal resharding path** — which is what lets an abandoned unit be
reassigned without deleting its prior state while guaranteeing a completed result can never move.

**Verification.** The independent verifier now defines or derives every expected value locally —
immutable campaign and B1C values, superseded B2 values, field sets, schema versions, status and
transition rules, the shard formula, path derivation and the publication guarantees — reads the
campaign manifest, required identities, tooling contract, campaign state and state contract itself,
and uses its own canonical-JSON implementation. It imports the production module only as the system
under test; an AST test enforces that it never uses `from baseline.g8_bler_work_units import …` for
expected constants. Its drills fork real child processes: eight simultaneous creators with exactly
one winner, a hard exit before publication leaving the final path absent, and a hard exit after
publication leaving complete canonical bytes.

The permanent no-live-state assertion became the explicit `--require-no-live-state` option, used
when closing B2C. Ordinary contract verification no longer requires the runtime tree to be absent,
so B3, B4 and G8_C can legitimately create one; tracked unit-state or lock files are rejected
always, through `git ls-files`.

**Migration.** `tools/migrate_g8_bler_state_contract.py` replaces exactly one already-registered
produced-artifact binding. It accepts only the exact B2 or the exact B2C pair, refuses any
scientific or phase drift, stages the corrected contract, verifies it independently against a
staged campaign state that registers exactly the staged bytes — which is how the artifact/state
circularity is broken without weakening either check — atomically replaces the artifact, `fsync`s
the directory, then atomically updates only the state-contract entry. It preserves the campaign ID,
manifest, required identities, B1C contract, every unrelated artifact, phase, stage, completed IDs,
in-progress ID, all counters, seed identity and the exact B3 restart command, and is idempotent and
recoverable from all four artifact/state pairs and from interruption before, between and after the
two replacements. **Because no unit-state file has ever existed, no per-unit migration was needed
or written.**

**Amendment judgment: no specification amendment.** B2C corrects implementation mechanics before
any data exists and changes no scientific parameter, hypothesis, gate, operating rule or selection
rule.

**Also recorded during B2C, and deliberately not acted on:** DJSCC training infrastructure remains a
later W5 task; PR-1, PR-2, PR-3 and PR-9 remain outstanding and are governed by the actual calendar
review deadline rather than by the completion of the engineering work package called W4; and
`er1_projected_total_hours_status` remains an open profiling/governance item.

## G8_B B6 — final green and handoff (complete)

The existing B5 row in `instructions/RESUME.md` closed after the signed commits
`faf089edcf121f8827e7c2f231702177d781463c` (literal correction, v3 contract and migration support),
`14f584ba2bf747ecfe649bb6da525d6822f0940e` (live v2→v3 runner migration and v3-bound schema-2
smoke), `bd77ac5287473c690c29de9c5a0b70059afa5dd1` (migration/publication adversarial matrix), and
`28ba80a5dbb18c0a68d6db04851bfb5e2a338b43` (B5 ledger closure). All are signed and pushed with
local/origin/remote parity.

The final v3 runner is `g8runner-49e4facbe266117c74ff802b4252bcba87a7331c34a7ffe228b4648469728583`,
SHA-256 `a238478eb9f231c984258e1f99c4c54d1a0fba353faa683908bca460c6c03763`, 18,612 bytes, with
the complete v2→v1 supersession history. The v3-bound smoke is SHA-256
`1f574b8d8442b68c44211693d128f768bc17f0bfa5472ae144d6c9e5b8ef6635`, 36,572 bytes. The exact
selected smoke IDs are `bler-0020cd25150d4f59a8fbb7c0`, `bler-002ad933f0d8e5617ad23e11` and
`bler-00a9c6a1b521eb82a351382c`; each ran 16 trials at attempt 1 and every contribution is zero.

B6 passed every requested generator/verifier, W4 and G-2 integration, spec/docs/literal checks,
resume `218`, runner `49`, combined `772`, and full-suite `1953 passed, 0 failed, 0 skipped` in
`283.15 s`; `verify_cpu_lock.py --clean-install` passed. The only state change was the validated,
atomic stage advance to `G8_B/tooling_smoke_complete`, producing campaign-state SHA-256
`09f1655f570fe947f93bf2477b7bb3b3a7e871c32a98addde0b9d1e7b3400a77`. It preserves seven artifact
bindings, empty completed IDs, null in-progress ID and four zero counters. No full-strength work,
selection, authorization, test access, `BlerTable`, inference or training occurred. Frozen G8_A,
B1C, B2C, B3, LDPC, G-2, W4, dataset, parameter and specification files remain byte-identical to
the B5 starting boundary. **No specification amendment.**

## G8_C C0/C1 — open and freeze characterization before data

C0 resolved the prior HOLD without reopening any frozen authority. The manifest-bound
`instructions/G8_C.txt` was restored from the authenticated G8_B green commit byte-for-byte:
2,820 bytes, SHA-256
`c0846909c30895780c03269486c154e58c2f9987a46b6699bfc521be46a38815`. The withdrawn requirement
to expand that file in place was recorded as an operational correction; the campaign manifest,
campaign ID, G8_A/B contracts, scientific parameters and evidence were unchanged. Signed commits
`490138ef3ecc1a96c528d0f75ca90fadeb8db14f` and
`1363fa73338e9d44151f751bf1bba47f73812ebb` restored the instruction and atomically opened
`G8_C/characterization_open`. No runtime root or full-strength artifact existed and all protected
counters remained zero.

C1 froze the C-owned orchestration and registered the pre-data source manifest before any
full-strength trial. The manifest is
`g8charsrc-6926319673ca1f55b95f8746062518c12cfa499aa827448e67850b5a1f74702a`, SHA-256
`a917f839f945232e85852d6d27f02de4b5dc272adc72b1966a95e9b5e62a014e`, 6,672 bytes. It binds six
G8_C sources and the frozen B1C/B2C/B3/runner/classical-composition dependencies, but not the
historical expanded instruction or an operational note. The coordinator validates the exact
production root, resolves one worker per supported GPU, authenticates one runner context per
worker process, rebuilds a fresh B3 plan before each unit, and leaves campaign-state publication
to the coordinator. The measured-only table loader reconstructs curves from the accepted merge
rows and the independent verifier has explicit C4 merge-registration and C5 table-registration
modes. C1 state SHA-256 is
`0ac72d8c6e0c8459b78f0c97a45db517e59da4d30a521522fb4f445d80482153`, with eight artifacts,
empty completed IDs, null in-progress ID, zero counters and an absent production root. The final
pre-data gates are being rerun before C2. **No specification amendment.**

The first C2 durable-batch marker is prepared from C1 completion
`50a62bba4c806f88066993a0cfcf99fd979e5d22`. The exact restart command is the coordinator command
in `instructions/RESUME.md`; its read-only plan probe authenticated one CUDA worker on the WSL2
`NVIDIA GeForce RTX 4060 Laptop GPU`, resolved batch size 64, shard plan `1/0`, and maximum durable
batch 128. The fresh B3 plan digest is
`9ff51133ea1b2ade32838668c9a6a0b7c40a2aee9ce41b881e592a81ce17e674`, with 3,213 remaining units;
the first intended durable prefix contains 128 authority-ordered IDs from
`bler-0020cd25150d4f59a8fbb7c0` through `bler-09c7fddde6d6ef0ad7f436fe`. The plan probe did not
create the production root. The first worker transaction completed
`bler-0020cd25150d4f59a8fbb7c0` at attempt 1 with exactly 5,000 trials and request/result/state
digests `7670fabefe4f94a1b897513a78ad4f93d479dfb9c9e41391d6d1b7377a452201`,
`7dd4cbbdda922112608b86a88783719f2b9061f504d481e37864ba23d236fec4` and
`8b24cef4dc7c078cf540b540226ed8e3ccaf9c3caa5ed317a34e07df2b7d5de9`. The worker summary then
hit a `KeyError: measurement` after the frozen runner had published and linked the complete
transaction; the coordinator reconciled one completed ID and 3,212 remaining, with no recoverable,
failed, duplicate or unknown item. The source remains byte-frozen; subsequent suffix execution
uses the exact same CLI and clean B3 plan/reconciliation path. All four protected counters remain
zero. **No specification amendment.**

The next four frozen-CLI invocations completed authority ordinals 1–4
(`bler-002ad933f0d8e5617ad23e11`, `bler-0032dc2b9c53168091b0dd75`,
`bler-004e65f1c85e812b81fb02aa` and `bler-0058cf3520d20daab0af1163`) at attempt 1,
each with 5,000 trials and no retryable failure. Their request/result/state chains were independently
reconciled; the campaign now contains five completed IDs and 3,208 remaining. Because the frozen
worker summary projection has the known `KeyError: measurement` after each successful publication,
these are being checkpointed as a bounded four-unit operational batch while the source-manifest
bytes remain unchanged. **No specification amendment.**
