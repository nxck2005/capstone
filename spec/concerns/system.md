<!-- GENERATED FROM spec/SPEC.md BY tools/gen_spec_views.py - DO NOT EDIT -->

# System

Requirements extracted from [`SPEC.md`](../SPEC.md). This view is for focused reading and review; the spec text is normative.

## SR

- **SR-1** — Every run MUST be fully determined by a configuration file derived from `params.generated.yaml`; no experiment-affecting constant may be hard-coded in source. *(verify: unit test asserting config round-trip and no literal SNR/k values in `src/`)*
- **SR-2** — All datasets in `params.datasets` MUST be selectable by name through one code path, with no dataset-specific branching in the encoder, decoder or training loop (DEC-1). *(verify: unit test instantiating each dataset)*
- **SR-3** — For a configured dataset and ratio, the encoder MUST emit exactly the number of complex symbols given by `params.bandwidth.k_symbols`. *(verify: unit test on output shape for every dataset × ratio pair)*
- **SR-4** — Transmitted symbols MUST satisfy `params.bandwidth.power_constraint` per image via an explicit normalisation layer, so SNR is unambiguous. *(verify: unit test that empirical mean power is 1.0 ± 1e-3)*
- **SR-5** — Channel models MUST live behind a registry exposing `forward(x, snr_db)` and be selectable by name. `params.channel.models_supported` is required now; `params.channel.models_planned` MUST be addable without modifying encoder, decoder or training code (FW-1). *(verify: unit test registering a stub channel and training one step through it)*
- **SR-6** — The channel MUST be differentiable end-to-end; gradients from the loss MUST reach encoder parameters. *(verify: unit test asserting non-zero encoder gradients after one backward pass)*
- **SR-7** — Noise power MUST be derived from `params.channel.snr_definition` alone, and the same definition MUST be used by the learned system, the baseline and the hardware tiers. *(verify: unit test measuring empirical SNR against the requested value)*
- **SR-8** — The decoder MUST carry both heads in `params.learned_system.decoder_heads`, trained as `params.learned_system.loss` with λ read from config (DEC-2). Setting λ = 0 MUST yield a pure-task model without code changes (FW-2). *(verify: unit test on both head outputs and on λ=0 disabling the MSE term)*
- **SR-9** — λ MUST be calibrated at G-4 to the smallest value in the calibration search whose top-1 accuracy at the core SNR is within 1 percentage point of the λ=0 model while reconstruction PSNR at 19 dB SNR is at least 20 dB. The chosen value MUST replace `params.learned_system.lambda_core` in this file. *(verify: calibration run archived under `params.artifacts.results_dir`)*
- **SR-10** — Training MUST checkpoint every `params.compute.checkpoint_every_epochs` epochs and resume from checkpoint with no metric discontinuity, so runs survive Colab/Kaggle session limits (DEC-4). *(verify: kill-and-resume test comparing loss curves across the seam)*
- **SR-11** — No single run may exceed `params.compute.max_wall_clock_hours_per_run` on `params.compute.primary_device` or `params.compute.vram_budget_gb` of VRAM. Sweeps MUST be decomposable into independent runs each meeting this bound. *(verify: measured wall clock and peak VRAM logged per run)*
- **SR-12** — Runs MUST be deterministic given a seed from `params.evaluation.seeds`: same seed and config reproduce reported metrics within 0.1 percentage points. *(verify: repeat-run test)*
- **SR-13** — Every result row MUST record its `run_id` and `git_commit`, and artifacts MUST be written under the directories in `params.artifacts`. *(verify: schema test on emitted CSV)*
- **SR-14** — The learned model MUST NOT exceed `params.learned_system.max_params_millions` million parameters, keeping the RPi tier plausible. *(verify: unit test on parameter count)*
- **SR-15** — The downstream task head MUST sit behind a registry so a different task can be added without touching the encoder, channel or training loop (FW-4). *(verify: unit test registering a stub task head)*

## Parameters referenced here

| Parameter | Value |
| --- | --- |
| `artifacts` | *(see datasheet)* |
| `artifacts.results_dir` | results/ |
| `bandwidth.k_symbols` | *(see datasheet)* |
| `bandwidth.power_constraint` | unit_average_power |
| `channel.models_planned` | rayleigh_block, rayleigh_fast |
| `channel.models_supported` | awgn |
| `channel.snr_definition` | Es/N0 in dB per complex channel use, measured after unit-average-power normalisation |
| `compute.checkpoint_every_epochs` | 1 |
| `compute.max_wall_clock_hours_per_run` | 2 |
| `compute.primary_device` | rtx_4060_mobile_8gb |
| `compute.vram_budget_gb` | 7.0 |
| `datasets` | *(see datasheet)* |
| `evaluation.seeds` | 0, 1, 2 |
| `learned_system.decoder_heads` | reconstruction, classification |
| `learned_system.lambda_core` | 1.0 |
| `learned_system.loss` | CE + lambda * MSE |
| `learned_system.max_params_millions` | 10 |
