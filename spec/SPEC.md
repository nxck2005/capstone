# Semantic Communication over Noisy Channels — Project Specification

**Status:** normative, hand-maintained. This file is the single source of truth for the project.
`DATASHEET.md`, `concerns/*.md` and `params.generated.yaml` are **generated from this file** — never edit those by hand.

**Keywords.** MUST, MUST NOT, SHOULD, SHOULD NOT and MAY are used in the RFC-2119 sense. A MUST that cannot be met is a spec change, not a quiet exception.

**Requirement IDs.** `SR` system · `BR` baseline · `ER` experiment · `DR` demo · `HR` hardware · `OPT` optional/non-blocking · `FW` future work · `DEC` settled decision · `G` schedule gate. IDs are permanent: retire, never renumber.

---

## 1. Purpose & thesis

Standard wireless practice compresses a source (JPEG) and separately protects the bits against noise (LDPC), then rebuilds the file bit-for-bit at the receiver — an architecture optimised for pixel-perfect reconstruction. This project instead trains a neural **encoder** (sender) and **decoder** (receiver) end-to-end through a differentiable channel model, so that only what the downstream task needs survives the link.

The claim is structural, not a tuning result:

1. Classical coding has no representation of what the bits are *for*. A task-success objective is not expressible in the JPEG+LDPC pipeline at all; the learned system optimises it directly.
2. Shannon's separation theorem makes compress-then-protect optimal only in the limit of infinitely long messages. Real edge/IoT links send **short** messages over **noisy** channels — precisely the regime where joint source-channel coding wins.
3. The observable signature is **graceful degradation**: classical coding has a cliff below which the receiver gets nothing, while the learned system degrades continuously — blurrier, still task-correct.

The Tier 1 deliverable is this effect demonstrated and quantified on a simulated channel, against a baseline tuned in the baseline's own favour.

## 2. Success criterion

Tier 1 **passes** if, on the headline dataset at the core bandwidth ratio, with both systems given identical test images, identical channel realisations and an identical number of complex channel uses:

> the top-1 accuracy-vs-SNR curves of the learned and classical systems **cross**, with the learned system above the classical one at low SNR, and the 95% confidence intervals of the two systems **do not overlap at three or more consecutive SNR grid points** below the crossover.

Tier 1 **fails** if the curves do not cross, or if the separation is within confidence intervals. A failure is reported as a negative result (ER-8), not hidden by re-tuning the baseline downward.

Everything else in this spec — hardware tiers, demo, extra experiments — is subordinate to that sentence.

## 3. Settled decisions

- **DEC-1** — **Dataset ladder.** The headline result targets Imagenette at 160px. Demotion to STL-10 at 96px is permitted only at G-6, and only if the Tier 3 live demo cannot otherwise run live. Demotion to CIFAR-10 is permitted at G-3 if compute limits or a non-reproducing crossover force it. Dataset MUST be a configuration axis, never a code fork, and CIFAR-10 MUST stay wired throughout as the fast smoke path. Rationale: highest-resolution headline that the hardware can actually carry, with two pre-agreed step-downs so the schedule never stalls on a judgement call.
- **DEC-2** — **Dual-head decoder.** One decoder carrying a reconstruction head and a classification head, trained with `loss = CE + λ·MSE`. Rationale: the accuracy curve and the "blurry but still task-correct" demo visual come from one model rather than two, at a measured accuracy cost (SR-9). Reversal: if the λ calibration cannot meet SR-9, split into two models and record the change here.
- **DEC-3** — **Python primary.** Learned system and classical baseline are both Python; MATLAB appears only as non-blocking cross-checks (OPT-1..OPT-3). Rationale: one language, one CI path, no license dependency on the critical path.
- **DEC-4** — **Compute.** An RTX 4060 Mobile (8 GB) is the assumed trainer, with Colab/Kaggle free tier as overflow. University cluster access MUST NOT appear on any critical path. Consequence: checkpoint/resume (SR-10) and a per-run wall-clock cap (SR-11) are hard requirements, not conveniences.
- **DEC-5** — **Radio hardware deferred.** Tiers 2 and 3 are specified as capability requirements plus a budget range (HR-1), not a named device, and no purchase happens before G-5.
- **DEC-6** — **Required experiments.** Core crossover, SNR-mismatch robustness and bandwidth-ratio sweep are required (ER-1..ER-3). Rayleigh fading and the λ sweep are future work (FW-1, FW-2), but the extension points that admit them (SR-5, SR-8) are required now.
- **DEC-7** — **Demo styling.** The Streamlit demo and the thesis figures share one publication-grade plotting module (DR-4), so a demo screenshot is directly usable in the report.
- **DEC-8** — **Document structure.** This file is authoritative and self-sufficient: a reader who never runs the generator loses nothing. All other files under `spec/` are derived views.

## 4. Parameters

Every number the project commits to lives in this block. Code reads the generated YAML, never this markdown.

```yaml params
project:
  id: semcom-djscc
  task: image-classification-over-noisy-channel
  primary_metric: top1_accuracy_vs_snr
  tier1_channel: simulated

datasets:
  imagenette160:
    role: headline
    image_size: [160, 160, 3]
    n: 76800
    classes: 10
    train_images: 9469
    test_images: 3925
    clean_acc_floor: 0.90
  stl10:
    role: demotion_tier3
    image_size: [96, 96, 3]
    n: 27648
    classes: 10
    train_images: 5000
    test_images: 8000
    clean_acc_floor: 0.85
  cifar10:
    role: fallback_and_smoke
    image_size: [32, 32, 3]
    n: 3072
    classes: 10
    train_images: 50000
    test_images: 10000
    clean_acc_floor: 0.93

bandwidth:
  symbol_type: complex_baseband
  power_constraint: unit_average_power
  ratios:
    r_1_6: "1/6"
    r_1_12: "1/12"
    r_1_24: "1/24"
  core_ratio: r_1_12
  k_symbols:
    imagenette160: {r_1_6: 12800, r_1_12: 6400, r_1_24: 3200}
    stl10: {r_1_6: 4608, r_1_12: 2304, r_1_24: 1152}
    cifar10: {r_1_6: 512, r_1_12: 256, r_1_24: 128}

channel:
  snr_definition: "Es/N0 in dB per complex channel use, measured after unit-average-power normalisation"
  models_supported: [awgn]
  models_planned: [rayleigh_block, rayleigh_fast]
  train_snr_db_fixed: 7
  train_snr_db_set: [1, 4, 7, 13, 19]
  test_snr_grid_db: [-5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]

learned_system:
  framework: pytorch
  encoder: conv_downsample_to_k_symbols
  decoder_heads: [reconstruction, classification]
  loss: "CE + lambda * MSE"
  lambda_core: 1.0
  lambda_calibration_gate: G-4
  optimizer: adam
  lr: 0.001
  lr_schedule: cosine
  amp: true
  max_params_millions: 10
  batch_size: {imagenette160: 32, stl10: 64, cifar10: 128}
  epochs: {imagenette160: 100, stl10: 200, cifar10: 150}

baseline:
  source_codec: jpeg
  jpeg_quality_grid: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
  channel_code: 5g_nr_ldpc
  ldpc_rates: ["1/3", "1/2", "2/3", "5/6"]
  ldpc_decoder: normalized_min_sum
  ldpc_max_iters: 50
  modulations: [bpsk, qpsk]
  core_modulation: qpsk
  budget_rule: "payload_bits = floor(k * bits_per_symbol * rate); the JPEG file MUST fit within payload_bits"
  outage_policy: chance_level
  tuning: best_feasible_config_per_test_snr

reference_classifier:
  arch: resnet18
  trained_on: clean_images
  frozen: true
  shared_by: [classical_baseline, semantic_recon_ablation]

evaluation:
  seeds: [0, 1, 2]
  repeats: 3
  ci: student_t_95
  test_subset_size: 2000
  full_test_split_required_for: [ER-1]
  metrics: [top1_acc, psnr_db, ssim, bytes_sent, decode_failure_rate]

compute:
  primary_device: rtx_4060_mobile_8gb
  overflow: [colab_free, kaggle_free]
  vram_budget_gb: 7.0
  max_wall_clock_hours_per_run: 2
  checkpoint_every_epochs: 1

artifacts:
  results_dir: results/
  checkpoint_dir: checkpoints/
  figures_dir: figures/
  csv_schema: [run_id, timestamp, git_commit, system, dataset, n, k, bw_ratio, channel,
               train_snr_db, test_snr_db, seed, jpeg_quality, ldpc_rate, modulation,
               top1_acc, psnr_db, ssim, bytes_sent, decode_failure_rate, n_test, test_subset]

demo:
  framework: streamlit
  figure_style_module: src/viz/style.py
  fonts: serif_computer_modern
  palette: colorblind_safe
  offline: true
  cpu_only_capable: true

hardware_tier23:
  needs: [iq_playback, iq_capture, tx_capable_pair, wired_loopback_with_attenuator]
  min_sample_rate_msps: 1
  candidates: [adalm_pluto_x2, hackrf_one_plus_rtlsdr]
  budget_inr_range: [25000, 40000]
  edge_node: raspberry_pi_4_or_5
  live_demo_latency_budget_ms: 500
  purchase_gate: G-5
```

## 5. System requirements (SR)

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

## 6. Baseline requirements (BR)

The baseline exists to be beaten honestly. Every requirement here is a defence against an unfair comparison.

- **BR-1** — The classical chain MUST be `params.baseline.source_codec` → `params.baseline.channel_code` → modulation → the same channel implementation the learned system uses. *(verify: integration test asserting the shared channel object)*
- **BR-2** — The LDPC implementation MUST be validated against published 5G NR BER curves before any learned-vs-classical comparison is reported. *(verify: archived BER-vs-SNR plot with reference curve overlaid)*
- **BR-3** — The baseline MUST receive exactly the same number of complex channel uses `k` as the learned system, per `params.baseline.budget_rule`. Bandwidth matching is counted in channel uses, not in bytes. *(verify: unit test counting emitted symbols for both systems)*
- **BR-4** — At each test SNR the baseline MUST be tuned in its own favour: sweep `params.baseline.jpeg_quality_grid` × `params.baseline.ldpc_rates` and report the **best feasible** configuration, per `params.baseline.tuning`. Reporting a single fixed configuration across all SNRs is prohibited. *(verify: sweep artifact showing the selected config per SNR)*
- **BR-5** — If no JPEG quality produces a file fitting the payload budget, the transmission MUST be recorded as infeasible and scored per `params.baseline.outage_policy`, never silently skipped. *(verify: unit test at the smallest ratio where infeasibility is expected)*
- **BR-6** — A file that cannot be decoded after LDPC decoding MUST be scored per `params.baseline.outage_policy` and counted in `decode_failure_rate`. *(verify: unit test injecting an undecodable block)*
- **BR-7** — Both systems MUST see identical test images and identical noise realisations at a given seed and SNR. *(verify: test asserting bitwise-identical noise draws across the two pipelines)*
- **BR-8** — A frozen `params.reference_classifier` trained on clean images MUST meet each dataset's `clean_acc_floor`, and the same instance MUST score both the classical reconstructions and the semantic reconstruction ablation (ER-4). *(verify: measured clean accuracy per dataset, archived)*
- **BR-9** — `params.baseline.core_modulation` is used for headline results; other entries in `params.baseline.modulations` MAY be reported as supporting evidence. *(verify: config test)*

## 7. Experiment requirements (ER)

- **ER-1** — **Core crossover.** Both systems, full test split, every SNR in `params.channel.test_snr_grid_db`, at `params.bandwidth.core_ratio`, repeated over `params.evaluation.seeds`, reported with `params.evaluation.ci` intervals, measuring `params.project.primary_metric`. Pass/fail is §2. *(verify: results CSV + accuracy-vs-SNR figure)*
- **ER-2** — **SNR mismatch.** Train once at `params.channel.train_snr_db_fixed`, evaluate across the whole test grid, and report the degradation profile of both systems to evidence the cliff-versus-graceful contrast. *(verify: results CSV + figure)*
- **ER-3** — **Bandwidth sweep.** Repeat the comparison across every entry in `params.bandwidth.ratios` to locate where the learned advantage is largest. *(verify: results CSV + figure)*
- **ER-4** — **Task-training ablation.** The semantic reconstruction head MUST also be scored through the frozen `params.reference_classifier` (BR-8), separating gain attributable to joint coding from gain attributable to a task-trained classifier. Reported as a distinct `system` value. *(verify: results CSV containing the ablation rows)*
- **ER-5** — Every experiment MUST emit rows matching `params.artifacts.csv_schema` exactly — same columns, same order. *(verify: schema validation script over all result CSVs)*
- **ER-6** — Evaluation on `params.evaluation.test_subset_size` images is permitted for sweeps only; `params.evaluation.full_test_split_required_for` MUST use the full split, and the `test_subset` column MUST record which was used. *(verify: schema test asserting the flag)*
- **ER-7** — Every number appearing in the thesis MUST be traceable to a `run_id` and `git_commit` in a committed CSV. *(verify: audit script resolving each reported figure to its rows)*
- **ER-8** — If ER-1 fails, the negative result MUST be reported with the same rigour as a positive one, together with the diagnostic evidence. Weakening the baseline to manufacture a crossover is prohibited. *(verify: review against BR-4 sweep artifacts)*

## 8. Demo requirements (DR)

- **DR-1** — An SNR slider spanning `params.channel.test_snr_grid_db` MUST drive both pipelines live on the same input image. *(verify: manual demo script walkthrough)*
- **DR-2** — The interface MUST show, side by side, the classical output, the semantic reconstruction, and each system's predicted label with confidence. *(verify: manual walkthrough)*
- **DR-3** — The accuracy-vs-SNR crossover plot MUST update live with a marker at the current slider position. *(verify: manual walkthrough)*
- **DR-4** — The demo MUST render figures through `params.demo.figure_style_module`, the same module that renders thesis figures, using `params.demo.fonts` and `params.demo.palette`, with default framework chrome suppressed (DEC-7). *(verify: pixel-level comparison of a demo figure and its thesis counterpart)*
- **DR-5** — The demo MUST run with `params.demo.offline` true and remain usable on CPU only. *(verify: run with networking disabled on a CPU-only machine)*
- **DR-6** — The demo MUST consume frozen checkpoints and committed result CSVs; it MUST NOT train, fine-tune or recompute reported metrics. *(verify: code review)*

## 9. Hardware requirements (HR)

- **HR-1** — Tier 2/3 hardware MUST satisfy `params.hardware_tier23.needs` at no less than `params.hardware_tier23.min_sample_rate_msps`, within `params.hardware_tier23.budget_inr_range`. Devices in `params.hardware_tier23.candidates` are indicative, not selected. *(verify: procurement checklist against the capability list)*
- **HR-2** — No hardware may be purchased before `params.hardware_tier23.purchase_gate` passes. *(verify: gate record)*
- **HR-3** — **Tier 2.** Encoder output MUST be replayed as IQ through a real link (wired loopback with attenuator) and captured, then decoded offline and compared against the simulated result at matched measured SNR. *(verify: measured-vs-simulated accuracy table)*
- **HR-4** — **Tier 3.** A live encoder/decoder demo on `params.hardware_tier23.edge_node` MUST meet `params.hardware_tier23.live_demo_latency_budget_ms` end-to-end. If it cannot at the headline dataset, resolve at G-6 by DEC-1 demotion or by pre-recording. *(verify: measured latency distribution)*
- **HR-5** — No Tier 1 requirement may depend on hardware availability. Tier 1 MUST be completable, reportable and defensible with simulation alone. *(verify: review of SR/BR/ER for hardware dependencies)*

## 10. Optional cross-checks (OPT)

Non-blocking. None of these may appear on the critical path (DEC-3).

- **OPT-1** — MATLAB Communications Toolbox reproduction of the LDPC BER curve as an independent check on BR-2.
- **OPT-2** — MATLAB or symbolic treatment of channel capacity and the separation theorem for the thesis mathematics chapter, tying §1 claim 2 to a derivation.
- **OPT-3** — An independent MATLAB reimplementation of the JPEG+LDPC chain to cross-validate baseline accuracy at two or three SNR points.

## 11. Future work & mandated extension points (FW)

Not built now; the spec is shaped so each is additive rather than a redesign.

- **FW-1** — Rayleigh block and fast fading, via the channel registry (SR-5). Expected to strengthen the graceful-degradation claim, since classical schemes suffer disproportionately under fading.
- **FW-2** — λ sweep quantifying the accuracy cost of a viewable reconstruction, with λ=0 as the pure-task upper bound, via SR-8.
- **FW-3** — SNR-adaptive or variable-rate coding, where the transmitter adjusts rate to measured channel state.
- **FW-4** — Alternative downstream tasks (segmentation, detection) via the task-head registry (SR-15).
- **FW-5** — Digital/entropy-coded semantic variants for comparison against the analog-symbol design.
- **FW-6** — Additional datasets beyond `params.datasets`.

## 12. Schedule & gates

Weeks are relative (W1 = first working week) and rescale to the actual semester length. Gates are go/no-go: a failed gate triggers its stated fallback, not an extension.

| Week | Work | Gate |
|---|---|---|
| W1 | Repo scaffold, config plumbing (SR-1), data loaders (SR-2), reference classifier trained (BR-8) | **G-1** |
| W2 | Channel model + power normalisation (SR-4..SR-7), LDPC integration and BER validation (BR-2) | **G-2** |
| W3 | Classical baseline end-to-end with budget matching and per-SNR tuning (BR-3, BR-4) | |
| W4 | DJSCC training loop, dual head, checkpoint/resume (SR-8, SR-10) | |
| W5 | First accuracy-vs-SNR curves on CIFAR-10 smoke path | |
| W6 | Crossover reproduction attempt; debug week if needed | **G-3** |
| W7 | λ calibration (SR-9) | **G-4** |
| W8 | Headline dataset training at core ratio | |
| W9 | ER-2 SNR-mismatch experiment | |
| W10 | ER-3 bandwidth sweep, ER-4 ablation | |
| W11 | Full ER-1 with seeds and confidence intervals; results frozen | **G-5** |
| W12 | Streamlit demo and shared figure-style module (DR-1..DR-6) | |
| W13 | Tier 2 SDR offline replay (HR-3), if hardware approved at G-5 | |
| W14 | Tier 3 live RPi demo (HR-4) | **G-6** |
| W15 | Thesis figures, paper draft, results audit (ER-7) | |
| W16 | Buffer, viva preparation | |

- **G-1** — Reference classifier meets `clean_acc_floor` for the smoke dataset. Fallback: switch backbone or extend training before any DJSCC work begins.
- **G-2** — LDPC BER matches published curves within tolerance. Fallback: change LDPC library. No comparison may be reported before this passes.
- **G-3** — Crossover reproduced on the CIFAR-10 smoke path. Fallback: one debug week, then invoke DEC-1 demotion to CIFAR-10 as the headline dataset and re-plan.
- **G-4** — λ calibrated per SR-9. Fallback: DEC-2 reversal to two separate models.
- **G-5** — Tier 1 frozen: ER-1..ER-4 complete with confidence intervals and the success criterion decided either way. Passing unlocks the hardware purchase (HR-2). Failing means Tier 2/3 are abandoned and effort moves to reporting the negative result (ER-8).
- **G-6** — Tier 3 latency budget met at the headline dataset. Fallback: DEC-1 demotion to STL-10, or pre-recorded demonstration.

## 13. Non-goals

Explicitly out of scope. Listed so that scope creep is a visible spec change.

- Bit-exact or perceptually-optimal reconstruction as an objective in its own right — reconstruction exists to serve the task and the demo.
- Reinforcement learning of any kind. The system is trained end-to-end by supervised gradient descent.
- Multi-user, MIMO, relay or interference channels; a single point-to-point link is assumed.
- Video, audio or real-time streaming sources.
- On-device training or adaptation; edge nodes run frozen models.
- Adversarial robustness, security or privacy of the learned representation.
- Beating state-of-the-art DJSCC results. The comparison of record is against the classical baseline, not against the literature.
