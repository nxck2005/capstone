# G-9 evidence — the W0 LDPC spike

Supporting material for **AM-24** and **AM-25** in [`../SPEC.md`](../SPEC.md) §17. This directory
exists so that the measured claims in the specification can be checked by someone who was not there,
rather than taken on the author's word.

**Not normative.** `SPEC.md` governs. This is the record of how its numbers were obtained.

Measured 2026-07-27 on the primary device: Python 3.14.6 · `torch 2.13.0+cu130` ·
`sionna-no-rt 2.0.1` · NVIDIA RTX 4060 Laptop, 8 GB.

## What is here

| File | What it supports |
|---|---|
| `g9_spike_record.json` | AM-24. Machine-readable result of all seven spike checks. **Labelled `"status": "historical"`** — see below. |
| `spike_ldpc.py` | The script that produced it. Reads every constant from `../params.generated.yaml`, never from itself, so a spec edit cannot silently invalidate the record. |
| `run_spike.sh` | The environment recipe — the exact install that produces a CUDA-enabled stack. |
| `golden_vectors_check.py` | AM-25. Checks Sionna's encoder against srsRAN's MATLAB-generated vectors. |
| `golden_vectors_check.log` | Its output: **17 exact matches, 0 mismatches**, both base graphs, lifting sizes 2–288. |
| `fetch_srsran_vectors.sh` | Fetches and verifies those vectors. |
| `srsran_vectors.sha256` | Their checksums — what makes the fetch verifiable. |
| `check_packetisation.py` | AM-49, AM-55, **rewritten by AM-58**. TS 38.212 conformance across the whole configuration grid, and ER-9's feasibility at every proof-obligation configuration. |
| `packetisation_record.json` | Its output: the per-configuration record BR-10's verify clause requires, carrying the commit, params hash and script hash that produced it. |

## Reproducing

```bash
./run_spike.sh                 # builds the venv, installs, runs the spike
./fetch_srsran_vectors.sh      # fetches + checksum-verifies the vectors
python golden_vectors_check.py srsran_vectors
```

`run_spike.sh` is idempotent. Note that a bare `pip install torch` resolves to the **CPU build** on
Python 3.14; the `--index-url .../cu130` is not optional, and the check that matters is
`torch.version.cuda is not None`, not a successful import (AM-23).

## What is deliberately absent, and why

The vector `.dat` files and srsRAN's `ldpc_encoder_test_data.h` are **not committed**, and
`.gitignore` keeps them out. They are third-party AGPLv3 files published as a release asset of
`srsRAN_Project`. Committing them here and submitting this repository academically would be
distribution. `params.baseline.ldpc_golden_vector_vendored` is therefore false: this directory
carries the *checksums* and the *fetcher*, which preserve byte-exact reproducibility without
redistributing anything. Checksums are facts about a file, not copies of one.

**The upstream source is archived, and the successor does not replace it.** srsRAN became
[OCUDU](https://gitlab.com/ocudu/ocudu) in December 2025; the GitHub repository is archived and its
default branch now carries only a notice. Release tags and assets remain, and the fetcher pins an
immutable one — verified reachable on 2026-07-27 — but this can disappear without warning.

An external review proposed repointing at OCUDU, on the grounds that the successor ships the same
MATLAB-generated vectors under BSD-3 rather than AGPLv3, which would close the risk and permit
vendoring. **Checked, and it does not work (AM-30).** The licence half is right: OCUDU is a Linux
Foundation project under the BSD-3-Clause Open MPI variant. But vector tests moved out of the main
repository into a separate `ocudu-matlab` companion plugin, and OCUDU's own MATLAB tutorial states
that running the suite requires a working and licensed copy of MATLAB and its 5G Toolbox. There are
**no pre-generated vectors to download**, so BSD-3 covers generators rather than data, and adopting
the suggestion would trade a working rung for the licence dependency rung 2 exists to avoid.

The consequence runs opposite to the recommendation: the risk is **larger** than previously
recorded, because the successor publishes no replacement rung at all. Hence two things. Fetch and
archive the pinned asset locally, outside git, before the W3 fixture needs it rather than when it
does. And `params.baseline.ldpc_golden_vector_offline_floor` — a committed, hand-derived case that
always runs and needs no network — now carries more weight than AM-25 assumed, because it is the
only rung nobody can revoke.

## Throughput varies between runs; the spec records the committed run

`SPEC.md` cites **625.2 code blocks/s**, which is what `g9_spike_record.json` measures at batch 32,
and re-running on the same machine has produced up to **663**. That spread is thermal and clock
variance on a laptop GPU. The specification carries the figure from the *committed* run rather than
the best or the remembered one, because ER-7 requires every reported number to resolve to an
artifact in this repository — it previously read 634, which was **faster** than the evidence beside
it and traceable to nothing (AM-29). `params.compute.ldpc_decode_cb_per_s_observed_range` records
the spread so the variance is visible rather than implied.

The projection is insensitive to the difference at this magnitude — across that throughput range ER-1
at two operating ratios moves by well under an hour, and G-8's decision would be the same anywhere in
it. If a re-run ever produces a figure that changes that decision, that is a finding and belongs in
a new `AM`, not a quiet edit.

**The projection itself has moved twice, and the second time this README was wrong too (AM-59).** It
read "about 3.9 h to 4.1 h", which was computed against an **18-point** grid. AM-52 added three SNR
points and nothing that counts them was updated — not §2's family-wise arithmetic, not §16's cost
prose, not this file. The current figures are **2.42 h at one ratio and 4.83 h at two**, and they are
now an *upper* bound rather than a point estimate: they assume the worst-case eleven code blocks per
image that only 16-QAM at rate 5/6 needs, while the noisy end of the grid — where BR-4 selects BPSK
and the channel-bit budget is a quarter as large — needs two. `params.compute` records both ends and
the basis each is computed from, because a bare scalar is what let the stale one survive three rounds.

Note also what the projection does **not** cover: it is LDPC decode alone, excluding JPEG 2000
encode and decode, the classifier forward passes and the 16-QAM demapper. Its parameter name now
says so, and `params.compute.er1_projected_total_hours_status` records that the end-to-end figure is
owed at W3/W4. One thing that does not threaten it: the decoder documents no early stopping, so the
fixed 50 iterations make this a worst case with respect to SNR.

## The packetisation check is separate from the spike, on purpose

`check_packetisation.py` needs no GPU, no Sionna and no network, and runs in under a second. It
exists because the packetisation *contract* changed after the spike ran (AM-49): the transport-block
CRC is conditional on payload size, the maximum code-block size depends on the base graph, and the
base graph is selected once per transport block from (A, R) **before** segmentation rather than
afterwards from a per-code-block rate. `spike_ldpc.py` stays as the archived W0 measurement and is
not retrofitted — what changed is the contract, not what was measured.

For that reason `g9_spike_record.json` now carries `"status": "historical"` and a note naming the
grid it measured (**18 SNR points, 180 configurations**, before AM-52 and AM-58). Its measurements —
throughput, exact `E_r`, smallest workable payload, the LLR sign, the BG1 rate floor, the decoder
spelling — stand and are still the evidence for AM-24. Its *derived* ER-1 projection does not, and is
marked superseded in the file itself. Evidence that outlives the specification it was measured
against needs a label, not a quiet edit; `packetisation_record.json` carries the current numbers and
records the commit, params hash and script hash that produced them.

**The first version of this script reported zero failures while breaking four of its own rules
(AM-58).** It is worth stating plainly, because "the check passes" is exactly the sentence that stops
anyone looking:

| Defect | Rows affected |
|---|---|
| `A` not byte-aligned, under a solver whose parameter promises byte alignment | 92 / 215 |
| `B'/C` non-integral, silently rescued with `ceil` | 21 / 215 |
| BG2 filler computed from the selection `K_b` instead of the encoded `K = 10Z` | 47 / 103 BG2 rows |
| Rate floor tested strictly, and against the *smallest* `E_r` — the block least likely to fail | all rows |

TS 38.212 §5.2.2 defines `K' = B'/C` as an exact division; real NR gets that for free from the
TS 38.214 transport-block-size quantisation, which this project does not use because it supplies
arbitrary source sizes. And the library's own guard is `if bg == "bg1" and r < 1/3: raise` — BG1's
mother code is 22Z systematic over 66Z transmitted, which *is* 1/3 — so a strict comparison
manufactured three clamps that never needed to exist.

**Correcting all four is a repair, not a collapse.** 215 configurations remain feasible and **every**
headline-dataset configuration remains feasible; source capacity moves in 18 rows by −8 to +1 byte;
and the clamps fall from six to three. The canonical case changes to **A = 42,624 bits (5,328 source
bytes), B = 42,648, C = 6, B' = 42,792, K' = 7,132, Z = 352, K = 7,744, filler 612 per block and
3,672 in total**, with `E` still summing to 51,200. AM-24's BG1 clamp survives in kind — every
clamped configuration still has a payload above the CRC threshold and a rate above 0.25 — but there
are three of them, not six, and the three that vanished are STL-10 configurations sitting at exactly
1/3.

**The proof obligation is also wider than it was.** It used to cover the 72 configurations that
today's three *provisional* ratios name. ER-3 may select any rung of the ladder at G-8, so that
proved nothing about the configuration actually used; it now covers all 144 on both headline
datasets. Broadening it immediately paid for itself: ER-9's admissible (dimension, width) pairs fall
from 23 at Imagenette's `r_1_3` to 7 at STL-10's `r_1_12`, 3 at `r_1_24` and **1** at `r_1_48` — so
the control stays feasible everywhere, but at the bottom rung its two-stage validation search would
have exactly one candidate to choose from.

## Two things a reader should not misread

**The 85 skipped cases are not failures.** `golden_vectors_check.py` lets the library infer the
lifting size from `(k, n)`; where it infers a different one, the comparison is not apples-to-apples
and the case is skipped. Every structurally valid comparison matched **exactly**. The W3 fixture must
pin the lifting size rather than infer it, which is what unlocks the remaining cases.

**The alignment is the load-bearing detail.** srsRAN stores the codeword with the first 2Z systematic
bits *already* punctured (BG1's codeword is 68Z, the stored buffer 66Z) and marks filler positions
with the byte 254; Sionna deletes filler positions *before* puncturing. Aligning means dropping the
254s and nothing else. Three earlier attempts agreed with the reference at **0.50** — chance, and
indistinguishable from a library defect. A fixture built on the naive alignment would have failed
G-2 while looking like someone else's bug.
