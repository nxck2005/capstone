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
| `g9_spike_record.json` | AM-24. Machine-readable result of all seven spike checks. |
| `spike_ldpc.py` | The script that produced it. Reads every constant from `../params.generated.yaml`, never from itself, so a spec edit cannot silently invalidate the record. |
| `run_spike.sh` | The environment recipe — the exact install that produces a CUDA-enabled stack. |
| `golden_vectors_check.py` | AM-25. Checks Sionna's encoder against srsRAN's MATLAB-generated vectors. |
| `golden_vectors_check.log` | Its output: **17 exact matches, 0 mismatches**, both base graphs, lifting sizes 2–288. |
| `fetch_srsran_vectors.sh` | Fetches and verifies those vectors. |
| `srsran_vectors.sha256` | Their checksums — what makes the fetch verifiable. |

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

**The upstream source is archived.** srsRAN became [OCUDU](https://gitlab.com/ocudu/ocudu) in
December 2025; the GitHub repository is archived and its default branch now carries only a notice.
Release tags and assets remain, and the fetcher pins an immutable one — but this can disappear
without warning. That is why BR-2 also requires
`params.baseline.ldpc_golden_vector_offline_floor`: a committed, hand-derived case that always runs
and needs no network, so G-2 degrades to narrower coverage rather than failing outright.

## Throughput varies between runs; the spec records the slow end

`SPEC.md` cites **634 code blocks/s**, and re-running the spike on the same machine has produced up
to **663**. That is thermal and clock variance on a laptop GPU, not a discrepancy: the specification
deliberately carries the **slower** observed figure, because it feeds a compute *budget* and the
conservative direction is the safe one. `g9_spike_record.json` is whichever run last executed, so it
may not read exactly 634.

The projection is insensitive to the difference at this magnitude — ER-1 at two operating ratios
lands at 3.9 h versus 4.1 h across that range, and G-8's decision would be the same anywhere in it.
If a re-run ever produces a figure that changes that decision, that is a finding and belongs in a new
`AM`, not a quiet edit.

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
