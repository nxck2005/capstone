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
- **SR-9** — λ MUST be calibrated at G-4, **on the validation split** (DEC-12), to the smallest value in `params.learned_system.lambda_grid` whose top-1 accuracy at the core SNR is within `params.learned_system.lambda_acc_tolerance_pp` of the λ=0 model while reconstruction PSNR at 15 dB SNR is at least `params.learned_system.lambda_psnr_floor_db`, falling back to `params.learned_system.lambda_psnr_floor_relaxed_db` before any DEC-2 model split. The chosen value MUST replace `params.learned_system.lambda_core` in this file and clear `lambda_status`. The search is one pilot training run per grid entry at a single seed, so G-4's cost is bounded in advance rather than open-ended; and because λ sits in the training objective, the final multi-seed training of §13 MUST happen **after** this gate, never before it (AM-7). *(verify: calibration run archived under `params.artifacts.results_dir`, one row per `params.learned_system.lambda_grid` entry)*
- **SR-10** — Training MUST checkpoint every `params.compute.checkpoint_every_epochs` epochs and resume from checkpoint with no metric discontinuity, so runs survive Colab/Kaggle session limits (DEC-4). *(verify: kill-and-resume test comparing loss curves across the seam)*
- **SR-11** — A profiling run at `params.compute.profiling_gate` MUST establish the achievable batch size, epoch time and peak memory on `params.compute.primary_device`; thereafter no scheduled run may exceed `params.compute.max_wall_clock_hours_per_run` or `params.compute.vram_budget_gb`, and sweeps MUST be decomposable into independent runs each meeting those bounds. `params.learned_system.batch_size` is a target, not a constraint: gradient accumulation per `params.learned_system.grad_accumulation_allowed` MAY be used to hold the effective batch size. *(verify: profiling report archived, and measured wall clock and peak VRAM logged per run)*
- **SR-12** — Runs MUST be deterministic given a seed: on the same pinned software environment and hardware class, the same seed and config reproduce reported metrics within 0.5 percentage points; across environments, reproduction is required only within the reported interval. The lockfile, CUDA and driver versions MUST be recorded with the results. *(verify: repeat-run test on the pinned environment plus an archived environment manifest)*
- **SR-13** — Every result row MUST record its `run_id`, `git_commit`, `git_dirty` and `config_hash`, and artifacts MUST be written under the directories in `params.artifacts`. *(verify: schema test on emitted CSV)*
- **SR-14** — The learned model MUST NOT exceed `params.learned_system.max_params_millions` million parameters, keeping the RPi tier plausible. *(verify: unit test on parameter count)*
- **SR-15** — The downstream task head MUST sit behind a registry so a different task can be added without touching the encoder, channel or training loop (FW-4). *(verify: unit test registering a stub task head)*
- **SR-16** — Peak-to-average power ratio MUST be measured and reported per `params.learned_system.papr_report_required`, and an optional peak-power/clipping constraint MUST be available per `params.learned_system.peak_power_constraint_available`. Rationale: an average-power constraint lets a learned encoder buy SNR with peaky symbols that QPSK's constant modulus cannot use and a real amplifier cannot deliver; unreported, this is an unearned advantage in simulation and a Tier 2 discrepancy on hardware. *(verify: PAPR logged in every learned-system result row, and a unit test that the clipping constraint bounds measured PAPR)*
- **SR-17** — A validation split MUST be carved per `params.evaluation.split_rule` from each dataset's `val_images` count and used for **every** selection decision — baseline configuration, λ, training checkpoint, operating ratio, architecture (DEC-12). The test split MUST NOT be read by any selection code path. *(verify: unit test asserting split disjointness, and an audit that no selection routine can reach the test loader)*
- **SR-18** — Every evaluation run MUST emit a per-image outcome file matching `params.artifacts.per_image_schema` under `params.artifacts.per_image_dir`, keyed by `run_id`. Aggregate rows alone cannot support the paired inference in ER-10. *(verify: schema test, and a test that recomputing `top1_acc` from the per-image file matches the aggregate row)*

## Parameters referenced here

| Parameter | Value |
| --- | --- |
| `artifacts` | *(see datasheet)* |
| `artifacts.per_image_dir` | results/per_image/ |
| `artifacts.per_image_schema` | image_index, true_label, pred_label, correct, outage |
| `artifacts.results_dir` | results/ |
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
| `learned_system.lambda_core` | 1.0 |
| `learned_system.lambda_grid` | 0.0, 0.1, 0.3, 1.0, 3.0 |
| `learned_system.lambda_psnr_floor_db` | 20 |
| `learned_system.lambda_psnr_floor_relaxed_db` | 16 |
| `learned_system.loss` | CE + lambda * MSE |
| `learned_system.max_params_millions` | 10 |
| `learned_system.papr_report_required` | true |
| `learned_system.peak_power_constraint_available` | true |
