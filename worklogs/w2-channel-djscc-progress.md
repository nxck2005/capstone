# W2 channel/DJSCC foundation and G-7 adjudication

**Date:** 2026-07-29
**Starting commit:** `846208257b4b1206a620582d6ceb91e99f077d86`
**Implementation commit:** `26b631ede27a6f88f1d004a66b845c52a658e07c`
**Gate:** **G-7 PASS**

## Scope completed

W2 now has exact complex-symbol budgets for all 18 dataset/ratio combinations, explicit per-image
unit-power normalization, registered AWGN, keyed Philox complex noise, generic symbol-domain PAPR,
a capped-power constraint, `djscc_residual_v1`, a task-head registry, config-derived
`CE + lambda * MSE`, deterministic keyed initialization, and both parameter caps. The offline G-7
verifier has mutation coverage for every required failure class.

The narrow G-1 verifier gap was also closed: preflight and final check mappings must have their
exact historical key sets, including the unchanged `pytest_250` names. Missing and unexpected
entries both fail.

## Channel, power, and PAPR

AWGN uses configured Es/N0 per complex channel use after per-image unit-average-power
normalization. For linear SNR `gamma`, `E|n|² = 1/gamma` and each real component has variance
`1/(2 gamma)`. Scalar and per-image SNR are supported. Training may draw stochastic noise;
evaluation fails closed without externally supplied unit-standard complex noise.

Keyed evaluation noise is a pure function of exactly one `noise_id`. Reordering, rebatching, prior
draws, control flow, and system identity cannot change a row.

PAPR is measured per image as `10 log10(max |x|² / mean |x|²)` in the symbol domain. It is not
oversampled waveform PAPR. `PeakPowerConstraint` uses a bisection-solved capped-power projection
that preserves phase and the requested bound at unit mean power. No project-wide threshold was
selected and no constrained variant was trained.

## `djscc_residual_v1`

The encoder receives unnormalized RGB `[0,1]`. It applies 5×5 stride-2 convolutions
`3 → 64 → 128`, each with eight-group GroupNorm and channelwise PReLU; exactly two 128-wide
residual blocks (`3×3 → GroupNorm → PReLU → 3×3 → GroupNorm → add → PReLU`); and a 3×3
projection to `2C`. Packing reshapes `[B,2C,h,w]` to `[B,C,2,h,w]`, index 0 real and index 1
imaginary, then flattens complex channel, row, column. Under AMP, channel symbols are complex64.

The decoder exactly reverses packing, applies a 3×3 ingress to width 128 and two matching residual
blocks, then branches. Reconstruction uses 4×4 stride-2 transposed convolutions
`128 → 64 → 3` and sigmoid. The default registered task head uses adaptive average pooling and a
linear class projection. Dataset and ratio only determine dimensions, `C`, and `k`.

At the profiled Imagenette-160 `r_1_2`, `C = 24`, the latent grid is 40×40, `k = 38,400`, and the
complete model has **1,640,957 parameters**, below 10,000,000 and 11,181,642.

## CUDA smoke and G-7

A bounded smoke used the real manifest-backed Imagenette training view, batch 2, `r_1_2`, AWGN,
AMP, loss, backward, and one Adam step. It wrote no checkpoint or scientific result.

The corrected measured profile imported every critical project module from a clean detached
worktree at the implementation commit. A pre-measurement audit and the measured worker both resolved
the same module paths; the report binds each executed file to its SHA-256 and immutable git blob SHA.
It used all
8,469 Imagenette training examples, `r_1_2`, batch 32, zero workers, Adam, AMP, and three excluded
warm-up steps. `results/profiling/g7_djscc_profile.json` records config hash
`a31e426f11418c38ff094fde639f7d3bb9a9fa31261079d5efc84c363bd63974`.
The profiling tool accepts `--git-repo <clean-worktree-at-26b631e>` and
`--data-repo <root-with-verified-training-data>` so the immutable implementation state is distinct
from the later evidence files.

| Measurement | Result |
|---|---:|
| Device | NVIDIA GeForce RTX 4060 Laptop GPU |
| Driver / Torch / CUDA | 592.82 / 2.13.0+cu130 / 13.0 |
| Compute capability / total memory | 8.9 / 8,585,216,000 bytes |
| Achieved configured batch | 32 |
| Full epoch | 265 batches, 8,469 examples |
| Measured epoch time | 48.68431210900235 s |
| Throughput | 173.9574748645565 images/s |
| Peak allocated VRAM | 966,199,296 bytes = 0.8998432159423828 GiB |
| Peak reserved VRAM | 1,077,936,128 bytes = 1.00390625 GiB |
| Projected 100-epoch time | 1.352342003027843 h |
| Parameters | 1,640,957 |

Every G-7 condition passed: complete primary-architecture epoch, configured batch 32, reserved VRAM
under 7.0 GB, projection under four hours, both parameter caps, clean implementation commit, real
CUDA, training-only data, and offline report verification. The primary passed, so
`width_halved_djscc_residual_v1` was not implemented.

## Boundaries

No reference-classifier code, weights, metric, identity, or G-1 evidence changed. No full learned
campaign, λ calibration, constrained training, LDPC, modulation, JPEG 2000, classical baseline,
ER-3, W3 work, or test evaluation occurred. Only training data and synthetic fixtures were used.
The AST guard still proves no production module imports `src/data/test_access.py`; provenance
instrumentation remains at zero test decoder and canonicalization calls.

SR-16 infrastructure exists, but all-system PAPR reporting remains incomplete because the
classical and ER-9 systems do not yet exist, and constrained training remains downstream.

## Next

Run the transparency-bitrate probe with the frozen reference classifier before W3/W4 baseline work.
