<!-- GENERATED FROM spec/SPEC.md BY tools/gen_spec_views.py - DO NOT EDIT -->

# System

Requirements extracted from [`SPEC.md`](../SPEC.md). This view is for focused reading and review; the spec text is normative.

## SR

- **SR-1** — Every run MUST be fully determined by a configuration file derived from `params.generated.yaml`; no experiment-affecting constant may be hard-coded in source. *(verify: unit test asserting config round-trip, plus a lint rule flagging numeric SNR/k literals outside `src/config/` and tests)*
- **SR-2** — All datasets in `params.datasets` MUST be selectable by name through one code path, with no dataset-specific branching in the encoder, decoder or training loop (DEC-1). *(verify: unit test instantiating each dataset)*
- **SR-3** — For a configured dataset and ratio, the encoder MUST emit exactly the number of complex symbols given by `params.bandwidth.k_symbols`. *(verify: unit test on output shape for every dataset × ratio pair)*
- **SR-4** — Transmitted symbols MUST satisfy `params.bandwidth.power_constraint` per image via an explicit normalisation layer, so SNR is unambiguous. *(verify: unit test that empirical mean power is 1.0 ± 1e-3)*
- **SR-5** — Channel models MUST live behind a registry exposing `forward(x, snr_db)` and be selectable by name. `params.channel.models_supported` is required now; `params.channel.models_planned` MUST be addable without modifying encoder, decoder or training code (FW-1). *(verify: unit test registering a stub channel and training one step through it)*
- **SR-6** — The channel MUST be differentiable end-to-end; gradients from the loss MUST reach encoder parameters. *(verify: unit test asserting non-zero encoder gradients after one backward pass)*
- **SR-7** — Noise power MUST be derived from `params.channel.snr_definition` alone, and the same definition MUST be used by the learned system, the baseline and the hardware tiers. *(verify: unit test measuring empirical SNR against the requested value)*
- **SR-8** — The decoder MUST carry both heads in `params.learned_system.decoder_heads`, trained as `params.learned_system.loss` with λ read from config (DEC-2). Setting λ = 0 MUST yield a pure-task model without code changes (FW-2). *(verify: unit test on both head outputs and on λ=0 disabling the MSE term)*
- **SR-9** — λ MUST be calibrated at G-4, **on the validation split** (DEC-12), to the smallest value in `params.learned_system.lambda_grid` whose top-1 accuracy at `params.learned_system.lambda_calibration_snr_db` is within `params.learned_system.lambda_acc_tolerance_pp` of the λ=0 model while reconstruction PSNR at `params.learned_system.lambda_psnr_eval_snr_db` is at least `params.learned_system.lambda_psnr_floor_db`, falling back to `params.learned_system.lambda_psnr_floor_relaxed_db` before any DEC-2 model split. Calibration happens at `params.learned_system.lambda_calibration_ratio` and the chosen λ transfers to the other ratios DEC-11 trains. Those three parameters replace the phrases "the core SNR" and "15 dB SNR", which named no parameter at all and were bare numeric literals of exactly the kind SR-1's own lint rule exists to flag — the requirement and its linter disagreed (AM-35). DEC-2's relaxed PSNR floor exists partly to absorb λ's transfer to smaller ratios, where the floor is harder to clear. The chosen value MUST replace `params.learned_system.lambda_core` in this file and clear `lambda_status`. The search is one pilot training run per grid entry at a single seed, so G-4's cost is bounded in advance rather than open-ended; and because λ sits in the training objective, the final multi-seed training of §13 MUST happen **after** this gate, never before it (AM-7). *(verify: calibration run archived under `params.artifacts.results_dir`, one row per `params.learned_system.lambda_grid` entry)*
- **SR-10** — Training MUST checkpoint every `params.compute.checkpoint_every_epochs` epochs and resume from checkpoint with no metric discontinuity, so runs survive Colab/Kaggle session limits (DEC-4). *(verify: kill-and-resume test comparing loss curves across the seam)*
- **SR-11** — A profiling run at `params.compute.profiling_gate` MUST establish the achievable batch size, epoch time and peak memory on `params.compute.primary_device`; thereafter no scheduled run may exceed `params.compute.max_wall_clock_hours_per_run` or `params.compute.vram_budget_gb`, and sweeps MUST be decomposable into independent runs each meeting those bounds. `params.learned_system.batch_size` is a target, not a constraint: gradient accumulation per `params.learned_system.grad_accumulation_allowed` MAY be used to hold the effective batch size. *(verify: profiling report archived, and measured wall clock and peak VRAM logged per run)*
- **SR-12** — Runs MUST be deterministic given a seed: on the same pinned software environment and hardware class, the same seed and config reproduce reported metrics within 0.5 percentage points; across environments, reproduction is required only within the reported interval. The lockfile, CUDA and driver versions MUST be recorded with the results. *(verify: repeat-run test on the pinned environment plus an archived environment manifest)*
- **SR-13** — Every result row MUST record its `run_id`, `git_commit`, `git_dirty` and `config_hash`, and artifacts MUST be written under the directories in `params.artifacts`. *(verify: schema test on emitted CSV)*
- **SR-14** — The learned model MUST NOT exceed `params.learned_system.max_params_millions` million parameters. The binding justification is `params.learned_system.max_params_policy`: encoder, decoder and head together MUST NOT exceed the parameter count of `params.reference_classifier`, so the learned arm is never larger than the network scoring the classical arm. This replaces the original rationale — "keeping the RPi tier plausible" — which rested a Tier 1 constraint on a stretch tier that DEC-14 makes explicit upside and HR-5 forbids any Tier 1 requirement from depending on. The numbers happen to coincide: a ResNet-18 at ten classes is about 11.2M parameters, so the 10M cap is already below its own comparator and the re-justification costs nothing (AM-36). *(verify: unit test on parameter count, asserting both the absolute cap and that it does not exceed the measured reference-classifier count)*
- **SR-15** — The downstream task head MUST sit behind a registry so a different task can be added without touching the encoder, channel or training loop (FW-4). *(verify: unit test registering a stub task head)*
- **SR-16** — Peak-to-average power ratio MUST be measured and reported per `params.learned_system.papr_report_required` for **all three systems**, not the learned one alone — adaptive 16-QAM is not constant modulus either, so reporting the learned arm's PAPR in isolation invites the reader to assume the digital arms are at 0 dB. A peak-power/clipping constraint MUST be available per `params.learned_system.peak_power_constraint_available`, and per `params.learned_system.papr_constrained_variant_required` a PAPR-constrained learned variant MUST be trained and reported as a secondary curve at `params.bandwidth.headline_ratio` over `params.learned_system.papr_constrained_variant_seeds` seed. Rationale: an average-power constraint lets a learned encoder buy SNR with peaky symbols that a constant-modulus scheme cannot use and a real amplifier cannot deliver; unreported, this is an unearned advantage in simulation and a Tier 2 discrepancy on hardware. Disclosure alone answers "did you know?" but not "is the comparison fair?", and this is the first objection a reader with a communications background reaches for — one extra training run converts it into a figure this project produced itself (AM-44). Reported PAPR is the **symbol-domain** peak-to-average ratio; oversampled waveform PAPR after pulse shaping is a different and larger quantity, and the two MUST NOT be conflated in the report. *(verify: PAPR logged in every result row for every system, a unit test that the clipping constraint bounds measured PAPR, and the constrained variant present as a distinct `system` value)*
- **SR-17** — A validation split MUST be carved per `params.evaluation.split_rule` from each dataset's `val_images` count and used for **every** selection decision — baseline configuration, λ, training checkpoint, operating ratio, architecture (DEC-12). The test split MUST NOT be read by any selection code path. *(verify: unit test asserting split disjointness, and an audit that no selection routine can reach the test loader)*
- **SR-18** — Every evaluation run MUST emit a per-image outcome file matching `params.artifacts.per_image_schema` under `params.artifacts.per_image_dir`, keyed by `run_id`. A `run_id` MUST be unique per `params.artifacts.run_id_key` — the tuple that makes it one. Aggregate rows alone cannot support the paired inference in ER-10, and ER-10 pairs per-image outcomes across systems at matched (image, SNR, seed) *entirely through `run_id`*, because `params.artifacts.per_image_schema` carries none of those columns itself. Left undefined, the join is unspecified and the pairing silently wrong (AM-37). *(verify: schema test; a test that recomputing `top1_acc` from the per-image file matches the aggregate row; and a test that no two result rows share a `run_id` with differing `params.artifacts.run_id_key` values)*
- **SR-19** — One canonical image MUST be defined per `params.preprocessing` and produced **before** either pipeline sees the data: `canonical_image`, `resize_interpolation`, `antialias`, `train_crop`, `eval_crop`, `colour_space`, `bit_depth` and `tensor_range`, with `eval_augmentation_permitted` false and `channel_normalisation` applied inside the model rather than in the pipeline. The classical codec MUST compress `params.preprocessing.codec_input` — the same canonical pixels the learned encoder receives — so BR-3's equal-channel-uses claim is not quietly comparing two different images. Reconstructions MUST be clipped per `params.preprocessing.reconstruction_clipped_before_metrics` before scoring, and PSNR and SSIM MUST use `params.preprocessing.psnr_data_range`, `params.preprocessing.ssim_impl` and `params.preprocessing.ssim_gaussian_weights`. Rationale: none of this was specified anywhere, and every item on the list changes a reported number — an unclipped reconstruction inflates PSNR, a mismatched `data_range` shifts it by decibels, and a codec fed differently-resized pixels from the encoder makes the whole comparison invalid rather than merely noisy (AM-28). *(verify: unit test asserting the canonical tensor is bit-identical on both paths for the same image index, plus a metrics test against a known-PSNR image pair)*

## Parameters referenced here

| Parameter | Value |
| --- | --- |
| `artifacts` | *(see datasheet)* |
| `artifacts.per_image_dir` | results/per_image/ |
| `artifacts.per_image_schema` | image_index, true_label, pred_label, correct, outage, outage_reason, source_bytes |
| `artifacts.results_dir` | results/ |
| `artifacts.run_id_key` | system, dataset, bw_ratio, test_snr_db, train_seed, channel_seed |
| `bandwidth.headline_ratio` | crossover_ratio |
| `bandwidth.k_symbols` | *(see datasheet)* |
| `bandwidth.power_constraint` | unit_average_power |
| `channel.models_planned` | rayleigh_block, rayleigh_fast |
| `channel.models_supported` | awgn |
| `channel.snr_definition` | Es/N0 in dB per complex channel use, measured after unit-average-power normalisation |
| `compute.checkpoint_every_epochs` | 1 |
| `compute.max_wall_clock_hours_per_run` | 4 |
| `compute.primary_device` | rtx_4060_mobile_8gb |
| `compute.profiling_gate` | G-7 |
| `compute.vram_budget_gb` | 7.0 |
| `datasets` | *(see datasheet)* |
| `evaluation.split_rule` | val carved deterministically from the published train split by evaluation.split_seed; the test split is never used for any selection |
| `learned_system.batch_size` | *(see datasheet)* |
| `learned_system.decoder_heads` | reconstruction, classification |
| `learned_system.grad_accumulation_allowed` | true |
| `learned_system.lambda_acc_tolerance_pp` | 1.0 |
| `learned_system.lambda_calibration_ratio` | headline_ratio |
| `learned_system.lambda_calibration_snr_db` | 7 |
| `learned_system.lambda_core` | 1.0 |
| `learned_system.lambda_grid` | 0.0, 0.1, 0.3, 1.0, 3.0 |
| `learned_system.lambda_psnr_eval_snr_db` | 15 |
| `learned_system.lambda_psnr_floor_db` | 20 |
| `learned_system.lambda_psnr_floor_relaxed_db` | 16 |
| `learned_system.loss` | CE + lambda * MSE |
| `learned_system.max_params_millions` | 10 |
| `learned_system.max_params_policy` | not_to_exceed_reference_classifier |
| `learned_system.papr_constrained_variant_required` | true |
| `learned_system.papr_constrained_variant_seeds` | 1 |
| `learned_system.papr_report_required` | true |
| `learned_system.peak_power_constraint_available` | true |
| `preprocessing` | *(see datasheet)* |
| `preprocessing.codec_input` | canonical_8bit_pixels |
| `preprocessing.psnr_data_range` | 1.0 |
| `preprocessing.reconstruction_clipped_before_metrics` | true |
| `preprocessing.ssim_gaussian_weights` | true |
| `preprocessing.ssim_impl` | skimage_structural_similarity |
| `reference_classifier` | *(see datasheet)* |
