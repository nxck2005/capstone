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

* **`j2k_resolutions` vs CIFAR-10's 24/16 px axes** — unresolved by decision; blocks PB_3, the full
  BR-4 sweep and G-8; does not block PB_2. See the C1.5 section above.
* **Cache-key field spelling** — `baseline.j2k_cache_key` names `j2k_impl_version`,
  `J2KCodec._cache_identity` spells it `openjpeg_version`. Values agree; deliberately not renamed
  because the committed transparency-probe evidence records keys produced under the current
  spelling. Tested by value. Unchanged by PB_1C.
