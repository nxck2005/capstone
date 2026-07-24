# Semantic Communication over Noisy Channels — Project Specification

**Status:** normative, hand-maintained. This file is the single source of truth for the project.
`DATASHEET.md`, `concerns/*.md` and `params.generated.yaml` are **generated from this file** — never edit those by hand.

**Keywords.** MUST, MUST NOT, SHOULD, SHOULD NOT and MAY are used in the RFC-2119 sense. A MUST that cannot be met is a spec change, not a quiet exception.

**Requirement IDs.** `SR` system · `BR` baseline · `ER` experiment · `DR` demo · `HR` hardware · `PR` programme deliverable · `OPT` optional/non-blocking · `FW` future work · `DEC` settled decision · `G` schedule gate. IDs are permanent: retire, never renumber. A retired ID is struck through in place and keeps its number reserved (§14).

---

## 1. Purpose & thesis

Standard wireless practice compresses a source and separately protects the bits against noise, then rebuilds the file bit-for-bit at the receiver — an architecture optimised for pixel-perfect reconstruction. This project instead trains a neural **encoder** (sender) and **decoder** (receiver) end-to-end through a differentiable channel model, so that only what the downstream task needs survives the link.

The claim is structural, not a tuning result:

1. The **task-agnostic image-reconstruction baseline of record** — a standard source codec plus a standard channel code — has no representation of what the bits are *for*; its objective is fidelity, and task success is not expressible in it. This is a statement about *that* pipeline, not about digital systems in general: a digital transmitter could send quantised task features, logits or labels instead of an image. ER-9 builds exactly such a system as a control, so the thesis is tested rather than assumed.
2. Shannon's separation theorem makes compress-then-protect optimal only in the limit of infinitely long messages. Real edge/IoT links send **short** messages over **noisy** channels — a regime where separation is known to incur finite-blocklength and delay penalties, and where joint source-channel coding *may* therefore gain (Kostina & Verdú, arXiv:1209.1317). That is the hypothesis under test, not a theorem being applied.
3. The observable signature is **graceful degradation**: separated coding has a cliff below which the receiver gets nothing, while the learned system degrades continuously — blurrier, still task-correct.

Because a single learned-vs-classical gap conflates two distinct advantages — **task-aware representation** versus **joint coding** — the comparison of record is a three-way one: classical reconstruction pipeline, task-aware *digital* pipeline (ER-9), and learned JSCC. ER-4 and ER-9 together decompose the gap; without them the result cannot be attributed.

Two further claims are **not** made. The systems are compared at equal channel uses and equal average symbol power, which is equal transmitted energy under the stated model: the result is better accuracy at equal resources, **not** a measured energy saving. And no claim is made about beating published DJSCC results (§13).

The Tier 1 deliverable is this effect demonstrated, quantified and *attributed* on a simulated channel, against a baseline tuned in the baseline's own favour.

## 2. Success criterion and preregistered hypotheses

Completion and outcome are deliberately separated. A capstone commits to a defensible process, not to a favourable measurement.

**Tier 1 is complete** when both systems (and the ER-9 control) are implemented, validated (G-1, G-2, G-7), bandwidth-matched (BR-3), fully bit-accounted (BR-10, BR-11), evaluated at the operating point selected by the learned-blind rule in ER-3 (G-8), and reported with paired inference (ER-10) under this preregistered protocol. **Completion does not depend on which way the result falls.**

**The hypotheses**, each decided by the paired procedure in ER-10 and each reported either way:

- **H1 — low-SNR separation (primary).** Supported if the paired 95% interval for (learned − classical) top-1 accuracy lies strictly above zero at **three or more consecutive** SNR grid points at or below `params.channel.train_snr_db_fixed`. Consecutiveness is a preregistered run rule, not three independent tests; no per-point significance is claimed.
- **H2 — graceful versus cliff.** Supported if, over the steepest 4 dB window of the grid, the classical system loses at least `params.evaluation.cliff_drop_pp` percentage points of top-1 accuracy while the learned system loses no more than `params.evaluation.graceful_drop_pp`.
- **H3 — convergence.** Supported if the paired gap contracts monotonically (within interval width) as SNR rises. A **crossover is reported if observed but is not required**: at low bandwidth ratios the learned system is expected to dominate at every SNR, which supports H1–H3 and is not a failure. DEC-16 states how the operating point and the baseline's modulation adaptivity are chosen to make a crossover observable if one exists, and what is reported if it does not.
- **H4 — attribution.** Joint coding is credited only to the extent the learned system also exceeds the ER-9 task-aware digital control under the H1 rule. If it does not, the gain is attributed to task-aware representation and reported as such.

**Outcome reporting.** Any combination of supported and unsupported hypotheses is a valid, complete Tier 1. ER-8 governs: a negative or partial result is reported with the same rigour as a positive one, and weakening the baseline to manufacture a result is prohibited.

## 3. Settled decisions

- **DEC-1** — **Dataset ladder.** The headline result targets Imagenette at 160px, with STL-10 at 96px as the sole fallback headline, invoked at G-8 if compute or a degenerate baseline forces it. **CIFAR-10 is a plumbing smoke path only** — shape checks, gradient flow, kill-and-resume, schema — and MUST NOT become a headline dataset: at every ratio in `params.bandwidth.ratios` its channel budget is below the floor of any real image container (BR-11), so its classical baseline is pinned at chance for a file-format reason and no comparison made on it means anything. Dataset MUST be a configuration axis, never a code fork. Rationale: highest-resolution headline the hardware can carry, with one pre-agreed step-down, and no fallback that leads off a cliff.
- **DEC-2** — **Dual-head decoder.** One decoder carrying a reconstruction head and a classification head, trained with `loss = CE + λ·MSE`. Rationale: the accuracy curve and the "blurry but still task-correct" demo visual come from one model rather than two, at a measured accuracy cost (SR-9). Reversal ladder, cheapest first: (a) relax the reconstruction floor to `params.learned_system.lambda_psnr_floor_relaxed_db`, since §13 already makes reconstruction quality a non-goal; only if that also fails, (b) split into two models and record the change here. (b) roughly doubles the training budget and MUST NOT be taken after G-4 without re-planning the schedule.
- **DEC-3** — **Python primary.** Learned system and classical baseline are both Python; MATLAB appears only as non-blocking cross-checks (OPT-1..OPT-3). Rationale: one language, one CI path, no license dependency on the critical path.
- **DEC-4** — **Compute.** An RTX 4060 Mobile (8 GB) is the assumed trainer, with Colab/Kaggle free tier as overflow. University cluster access MUST NOT appear on any critical path. Consequence: checkpoint/resume (SR-10) and a measured compute budget (SR-11, G-7) are hard requirements, not conveniences.
- **DEC-5** — **Radio hardware deferred.** Tiers 2 and 3 are specified as capability requirements plus a budget range (HR-1), not a named device, and no purchase happens before G-5.
- **DEC-6** — **Required experiments.** Operating-point selection, core comparison, SNR-mismatch robustness, the task-training ablation, the task-aware digital control and the paired inference procedure are required (ER-1..ER-4, ER-9, ER-10). Rayleigh fading and the λ sweep are future work (FW-1, FW-2), but the extension points that admit them (SR-5, SR-8) are required now.
- **DEC-7** — **Demo styling.** The Streamlit demo and the thesis figures share one publication-grade plotting module (DR-4), so a demo screenshot is directly usable in the report.
- **DEC-8** — **Document structure.** This file is authoritative and self-sufficient: a reader who never runs the generator loses nothing. All other files under `spec/` are derived views.
- **DEC-9** — **Source codec.** The classical baseline of record uses **JPEG 2000**, not JPEG. Rationale: JPEG carries an irreducible container floor of roughly 250–290 bytes (quantisation tables, frame and scan headers) even with optimised Huffman tables, which at these channel budgets is a large or total fraction of the payload — at the core ratio it consumes most of Imagenette's budget and all of CIFAR-10's, and on STL-10 it makes the low-rate LDPC configurations infeasible outright, so the low-SNR "cliff" would be a file-format artifact rather than a channel effect. JPEG 2000 has no comparable floor and supports exact target-byte rate control, which also collapses the quality sweep to a direct budget solve. JPEG is retained as a **secondary reported curve** (`params.baseline.source_codec_secondary`) because it is what practitioners actually deploy, and BR-11 makes the overhead difference visible rather than hidden.
- **DEC-10** — **LDPC provenance.** **Sionna `2.0.1`** provides base-graph and lifting-size selection, encoding, rate matching and decoding; everything in `params.baseline.ldpc_impl_local` is implemented in this project, behind the adapter seam required by BR-14. Reasoning, recorded so it is not re-litigated: (a) Sionna 2.0.0 migrated PHY and SYS to PyTorch and no longer depends on TensorFlow, so the "second deep-learning framework" objection this decision originally carried was checked against the release notes and is **obsolete** — the environment stays single-framework and DEC-3 is satisfied; (b) the classical baseline is never trained, so the codec need not be differentiable, which makes correctness and throughput the only real criteria; (c) Sionna implements TS 38.212 rate matching, which is the genuinely error-prone part, but **not code-block segmentation**, which is why the local layer exists and is where the standards work for PR-3 actually lives; (d) `params.baseline.ldpc_decoder` is a Sionna built-in check-node update, so no custom callable and no scaling factor to justify. Rejected: implementing the whole chain locally — roughly a week landing at W3 with no slack, gating G-2 which gates everything, and a silent base-graph transcription error is the worst failure mode available; AFF3CT via `py_aff3ct` — a C++ build on WSL2, CPU-only marshalling per codeword, and a second toolchain on the critical path, which is now the only remaining DEC-3 violation among the candidates; generic Python LDPC packages (`pyldpc`, ProtographLDPC) — Gallager/protograph codes rather than 5G NR, so they cannot satisfy G-2 or PR-3 at all; pyAerial — needs the Aerial SDK and a datacenter GPU. Fallback ladder: pin an earlier Sionna if 2.0.x proves unsound at G-2, then drop to `params.baseline.ldpc_impl_fallback` behind the BR-14 seam with the segmentation layer already written and tested. The version pin, golden-vector provenance and measured throughput are recorded at G-9.
- **DEC-11** — **Training-SNR protocol.** One model is trained **per bandwidth ratio at `params.channel.train_snr_db_fixed`** and evaluated across the whole test grid. Not one model per test SNR, and not SNR-randomised training — the latter becomes OPT-4. Rationale: this is what makes ER-2's mismatch profile meaningful, and it bounds the training count at one run per ratio per seed.
- **DEC-12** — **Selection uses validation data only.** Every choice that could flatter a result — baseline codec quality and LDPC rate (BR-4), λ (SR-9), training checkpoint, operating ratio (ER-3), architecture — MUST be made on `params.datasets.*.val_images`, disjoint from the test split. The test split is touched once, for the frozen configuration. Rationale: `best config per test SNR` selected on test labels is test-set leakage and would invalidate every interval in §2.
- **DEC-13** — **Novelty position.** The course rubric scores novelty, while §13 disclaims beating published DJSCC results. These are reconciled by naming the contribution precisely: the novel elements of record are (a) the ER-9 attribution decomposition separating task-aware representation from joint coding, and (b) the BR-11 format-overhead-controlled classical baseline, which the DJSCC literature does not report. Reproduction quality carries the results marks; these two carry the novelty marks. PR-7 makes this an explicit written deliverable rather than a viva improvisation.
- **DEC-14** — **Tiers 2 and 3 are stretch goals.** The expected and planned demonstration is **pre-recorded**; a live SDR replay or RPi demo is upside, not the baseline plan. Rationale: procurement lead time plus IQ framing, synchronisation, frequency offset and calibration (HR-6) is a second project, and HR-5 already makes Tier 1 standalone. Framing them as expected deliverables converts likely upside into apparent shortfall.
- **DEC-15** — **Pretrained weights prohibited for the reference classifier.** `params.reference_classifier` MUST be trained from scratch. Rationale: Imagenette is a subset of ImageNet, so ImageNet-pretrained weights are label leakage an examiner will spot immediately. `clean_acc_floor` values are set to from-scratch reality accordingly.
- **DEC-16** — **Crossover strategy, and its last-resort fallback.** A crossover requires the classical system's high-SNR ceiling to exceed the learned system's, which the original parameters made arithmetically impossible — the baseline was capped at QPSK, so its payload could not grow no matter how clean the link became, and both curves flattened into parallel lines (worked through in `docs/crossover-explained.md`). The remedy is to **remove an artificial cap on the baseline, never to weaken the learned system**: `params.baseline.modulations` gains 16-QAM and `params.baseline.modulation_tuning` makes modulation an adaptive axis of BR-4's per-SNR tuning, so the baseline transmits more per channel use as conditions improve, exactly as deployed systems do. Because BR-4 reports the best feasible configuration at each SNR, this strictly raises the classical curve at high SNR and cannot lower it at low SNR, so the cliff that H2 depends on is untouched. Governing rule: every lever used to obtain a crossover MUST either strengthen the baseline or be preregistered; handicapping the learned system to manufacture one is prohibited and would violate ER-8. **Fallback, last resort only:** if the adaptive baseline still does not cross by G-8, the project reports learned dominance across the whole grid, promotes the reconstruction-quality crossover to the secondary figure, and relies on §2, which already makes that a complete Tier 1. This fallback exists to keep the demo and the thesis intact, not as a preferred outcome; taking it MUST be recorded here together with the G-8 evidence that the adaptive baseline was genuinely attempted first.

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
    train_images: 8469
    val_images: 1000
    test_images: 3925
    clean_acc_floor: 0.88
  stl10:
    role: fallback_headline
    image_size: [96, 96, 3]
    n: 27648
    classes: 10
    train_images: 4500
    val_images: 500
    test_images: 8000
    clean_acc_floor: 0.75
  cifar10:
    role: smoke_only
    image_size: [32, 32, 3]
    n: 3072
    classes: 10
    train_images: 45000
    val_images: 5000
    test_images: 10000
    clean_acc_floor: 0.93

bandwidth:
  symbol_type: complex_baseband
  power_constraint: unit_average_power
  ratios:
    r_1_2: "1/2"
    r_1_3: "1/3"
    r_1_6: "1/6"
    r_1_12: "1/12"
    r_1_24: "1/24"
  core_ratio: r_1_3
  core_ratio_status: provisional_until_G-8
  low_ratio_operating_point: r_1_12
  k_symbols:
    imagenette160: {r_1_2: 38400, r_1_3: 25600, r_1_6: 12800, r_1_12: 6400, r_1_24: 3200}
    stl10: {r_1_2: 13824, r_1_3: 9216, r_1_6: 4608, r_1_12: 2304, r_1_24: 1152}
    cifar10: {r_1_2: 1536, r_1_3: 1024, r_1_6: 512, r_1_12: 256, r_1_24: 128}

channel:
  snr_definition: "Es/N0 in dB per complex channel use, measured after unit-average-power normalisation"
  snr_conversion: "Es/N0_dB = Eb/N0_dB + 10*log10(bits_per_symbol * code_rate); every published reference curve MUST be converted with this identity before comparison"
  models_supported: [awgn]
  models_planned: [rayleigh_block, rayleigh_fast]
  train_snr_db_fixed: 7
  train_snr_db_set: [1, 4, 7, 13, 19]
  test_snr_grid_db: [-8, -6, -4, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 18]
  grid_rationale: "the LDPC waterfall for QPSK spans roughly -1 dB (rate 1/3) to 5 dB (rate 5/6), so density is spent there; the grid then extends to 18 dB because 16-QAM at rate 5/6 does not decode until roughly 11-12 dB, and truncating earlier would engineer a crossover under DEC-16 and then fail to measure it"

learned_system:
  framework: pytorch
  encoder: conv_downsample_to_k_symbols
  decoder_heads: [reconstruction, classification]
  loss: "CE + lambda * MSE"
  lambda_core: 1.0
  lambda_status: provisional_until_G-4
  lambda_calibration_gate: G-4
  lambda_acc_tolerance_pp: 1.0
  lambda_psnr_floor_db: 20
  lambda_psnr_floor_relaxed_db: 16
  train_snr_protocol: one_model_per_ratio_at_fixed_snr
  optimizer: adam
  lr: 0.001
  lr_schedule: cosine
  amp: true
  grad_accumulation_allowed: true
  max_params_millions: 10
  batch_size: {imagenette160: 32, stl10: 64, cifar10: 128}
  batch_size_policy: target_not_binding
  epochs: {imagenette160: 100, stl10: 200, cifar10: 150}
  papr_report_required: true
  peak_power_constraint_available: true

baseline:
  source_codec: jpeg2000
  source_codec_secondary: jpeg
  j2k_rate_control: exact_target_bytes
  jpeg_quality_grid: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
  container_policy: "all emitted container bytes count against payload_bits; shared-table or stripped-header variants MAY be reported as a labelled sensitivity, never as the headline"
  channel_code: 5g_nr_ldpc
  ldpc_standard: 3gpp_ts_38_212
  ldpc_impl: sionna
  ldpc_impl_version: "2.0.1"
  ldpc_impl_provides: [base_graph_selection, lifting_size_selection, encoding, rate_matching, decoding]
  ldpc_impl_local: [tb_crc, code_block_segmentation, per_block_budget_distribution,
                    concatenation, crc_failure_detection]
  ldpc_impl_fallback: self_implemented_offset_min_sum
  ldpc_golden_vector_file: tests/fixtures/ldpc_ts38212_golden.npz
  ldpc_golden_vector_source_gate: G-9
  ldpc_base_graph: auto_per_ts_38212
  ldpc_rates: ["1/3", "1/2", "2/3", "5/6"]
  ldpc_decoder: offset_min_sum
  ldpc_max_iters: 50
  tb_crc_bits: 24
  cb_crc_bits: 24
  code_block_max_bits: 8448
  rate_matching: ts_38212_with_filler
  modulations: [bpsk, qpsk, qam16]
  modulation_tuning: adaptive_per_snr
  core_modulation: qpsk
  budget_rule: "usable_source_bytes = floor((floor(k * bits_per_symbol * rate) - tb_crc_bits - segmentation_and_filler_overhead) / 8); the complete compressed file, container bytes included, MUST fit within usable_source_bytes"
  outage_policy: uniform_random_label
  tuning: best_feasible_config_per_snr_on_validation_split

reference_classifier:
  arch: resnet18
  trained_on: clean_images
  pretrained_weights_permitted: false
  artifact_finetuned_variant_required: true
  frozen: true
  shared_by: [classical_baseline, digital_semantic_control, semantic_recon_ablation]

evaluation:
  train_seeds: [0, 1, 2]
  channel_seeds: [0, 1, 2]
  split_seed: 1337
  split_rule: "val carved deterministically from the published train split by evaluation.split_seed; the test split is never used for any selection"
  ci: paired_bootstrap_95
  bootstrap_resamples: 10000
  paired_test: mcnemar_exact
  hypothesis_rule: three_consecutive_snr_points
  cliff_drop_pp: 30
  graceful_drop_pp: 15
  test_subset_size: 2000
  full_test_split_required_for: [ER-1]
  metrics: [top1_acc, psnr_db, ssim, bytes_sent, header_bytes, payload_bytes,
            decode_failure_rate, infeasible_rate, papr_db]

compute:
  primary_device: rtx_4060_mobile_8gb
  overflow: [colab_free, kaggle_free]
  vram_budget_gb: 7.0
  profiling_gate: G-7
  max_wall_clock_hours_per_run: 4
  checkpoint_every_epochs: 1

artifacts:
  results_dir: results/
  per_image_dir: results/per_image/
  checkpoint_dir: checkpoints/
  figures_dir: figures/
  csv_schema: [run_id, timestamp, git_commit, git_dirty, config_hash, checkpoint_id,
               system, dataset, split, n, k, bw_ratio, channel,
               train_snr_db, test_snr_db, train_seed, channel_seed, lambda,
               source_codec, jpeg_quality, j2k_target_bytes, ldpc_rate, modulation,
               top1_acc, n_correct, n_test, psnr_db, ssim,
               bytes_sent, header_bytes, payload_bytes, papr_db,
               decode_failure_rate, infeasible_rate, test_subset,
               wall_clock_s, peak_vram_gb]
  per_image_schema: [image_index, true_label, pred_label, correct, outage]

demo:
  framework: streamlit
  figure_style_module: src/viz/style.py
  fonts: serif_computer_modern
  palette: colorblind_safe
  offline: true
  cpu_only_capable: true

hardware_tier23:
  status: stretch_goal
  expected_demonstration: pre_recorded
  needs: [iq_playback, iq_capture, tx_capable_pair, wired_loopback_with_attenuator]
  min_sample_rate_msps: 1
  candidates: [adalm_pluto_x2, hackrf_one_plus_rtlsdr]
  budget_inr_range: [25000, 40000]
  budget_note: "two Plutos sit at or above the top of this range once import duty lands; the HackRF + RTL-SDR pairing is what the range actually buys"
  edge_node: raspberry_pi_4_or_5
  live_demo_latency_budget_ms: 500
  purchase_gate: G-5
  framing: [preamble_correlation, rrc_pulse_shaping, timing_sync, cfo_estimation, pilot_aided_snr_measurement]

deliverables:
  review_weeks: {first: 4, second: 10, third: 16}
  literature_review_min_refs: 25
  time_plan_artifact: gantt_chart
  poster_format: a0
  plagiarism_report_required: true
  report_format_source: vault/capstone/CAPSTONE_THESIS_Format.docx
  standards: [3gpp_ts_38_212, itu_t_t_800_jpeg2000, itu_t_t_81_jpeg, ieee_754, ietf_rfc_2119]
  novelty_claims: [er9_attribution_decomposition, br11_format_overhead_controlled_baseline]
```

## 5. System requirements (SR)

- **SR-1** — Every run MUST be fully determined by a configuration file derived from `params.generated.yaml`; no experiment-affecting constant may be hard-coded in source. *(verify: unit test asserting config round-trip, plus a lint rule flagging numeric SNR/k literals outside `src/config/` and tests)*
- **SR-2** — All datasets in `params.datasets` MUST be selectable by name through one code path, with no dataset-specific branching in the encoder, decoder or training loop (DEC-1). *(verify: unit test instantiating each dataset)*
- **SR-3** — For a configured dataset and ratio, the encoder MUST emit exactly the number of complex symbols given by `params.bandwidth.k_symbols`. *(verify: unit test on output shape for every dataset × ratio pair)*
- **SR-4** — Transmitted symbols MUST satisfy `params.bandwidth.power_constraint` per image via an explicit normalisation layer, so SNR is unambiguous. *(verify: unit test that empirical mean power is 1.0 ± 1e-3)*
- **SR-5** — Channel models MUST live behind a registry exposing `forward(x, snr_db)` and be selectable by name. `params.channel.models_supported` is required now; `params.channel.models_planned` MUST be addable without modifying encoder, decoder or training code (FW-1). *(verify: unit test registering a stub channel and training one step through it)*
- **SR-6** — The channel MUST be differentiable end-to-end; gradients from the loss MUST reach encoder parameters. *(verify: unit test asserting non-zero encoder gradients after one backward pass)*
- **SR-7** — Noise power MUST be derived from `params.channel.snr_definition` alone, and the same definition MUST be used by the learned system, the baseline and the hardware tiers. *(verify: unit test measuring empirical SNR against the requested value)*
- **SR-8** — The decoder MUST carry both heads in `params.learned_system.decoder_heads`, trained as `params.learned_system.loss` with λ read from config (DEC-2). Setting λ = 0 MUST yield a pure-task model without code changes (FW-2). *(verify: unit test on both head outputs and on λ=0 disabling the MSE term)*
- **SR-9** — λ MUST be calibrated at G-4, **on the validation split** (DEC-12), to the smallest value in the calibration search whose top-1 accuracy at the core SNR is within `params.learned_system.lambda_acc_tolerance_pp` of the λ=0 model while reconstruction PSNR at 15 dB SNR is at least `params.learned_system.lambda_psnr_floor_db`, falling back to `params.learned_system.lambda_psnr_floor_relaxed_db` before any DEC-2 model split. The chosen value MUST replace `params.learned_system.lambda_core` in this file and clear `lambda_status`. *(verify: calibration run archived under `params.artifacts.results_dir`)*
- **SR-10** — Training MUST checkpoint every `params.compute.checkpoint_every_epochs` epochs and resume from checkpoint with no metric discontinuity, so runs survive Colab/Kaggle session limits (DEC-4). *(verify: kill-and-resume test comparing loss curves across the seam)*
- **SR-11** — A profiling run at `params.compute.profiling_gate` MUST establish the achievable batch size, epoch time and peak memory on `params.compute.primary_device`; thereafter no scheduled run may exceed `params.compute.max_wall_clock_hours_per_run` or `params.compute.vram_budget_gb`, and sweeps MUST be decomposable into independent runs each meeting those bounds. `params.learned_system.batch_size` is a target, not a constraint: gradient accumulation per `params.learned_system.grad_accumulation_allowed` MAY be used to hold the effective batch size. *(verify: profiling report archived, and measured wall clock and peak VRAM logged per run)*
- **SR-12** — Runs MUST be deterministic given a seed: on the same pinned software environment and hardware class, the same seed and config reproduce reported metrics within 0.5 percentage points; across environments, reproduction is required only within the reported interval. The lockfile, CUDA and driver versions MUST be recorded with the results. *(verify: repeat-run test on the pinned environment plus an archived environment manifest)*
- **SR-13** — Every result row MUST record its `run_id`, `git_commit`, `git_dirty` and `config_hash`, and artifacts MUST be written under the directories in `params.artifacts`. *(verify: schema test on emitted CSV)*
- **SR-14** — The learned model MUST NOT exceed `params.learned_system.max_params_millions` million parameters, keeping the RPi tier plausible. *(verify: unit test on parameter count)*
- **SR-15** — The downstream task head MUST sit behind a registry so a different task can be added without touching the encoder, channel or training loop (FW-4). *(verify: unit test registering a stub task head)*
- **SR-16** — Peak-to-average power ratio MUST be measured and reported per `params.learned_system.papr_report_required`, and an optional peak-power/clipping constraint MUST be available per `params.learned_system.peak_power_constraint_available`. Rationale: an average-power constraint lets a learned encoder buy SNR with peaky symbols that QPSK's constant modulus cannot use and a real amplifier cannot deliver; unreported, this is an unearned advantage in simulation and a Tier 2 discrepancy on hardware. *(verify: PAPR logged in every learned-system result row, and a unit test that the clipping constraint bounds measured PAPR)*
- **SR-17** — A validation split MUST be carved per `params.evaluation.split_rule` from each dataset's `val_images` count and used for **every** selection decision — baseline configuration, λ, training checkpoint, operating ratio, architecture (DEC-12). The test split MUST NOT be read by any selection code path. *(verify: unit test asserting split disjointness, and an audit that no selection routine can reach the test loader)*
- **SR-18** — Every evaluation run MUST emit a per-image outcome file matching `params.artifacts.per_image_schema` under `params.artifacts.per_image_dir`, keyed by `run_id`. Aggregate rows alone cannot support the paired inference in ER-10. *(verify: schema test, and a test that recomputing `top1_acc` from the per-image file matches the aggregate row)*

## 6. Baseline requirements (BR)

The baseline exists to be beaten honestly. Every requirement here is a defence against an unfair comparison.

- **BR-1** — The classical chain MUST be `params.baseline.source_codec` → `params.baseline.channel_code` → modulation → the same channel implementation the learned system uses. `params.baseline.source_codec_secondary` MUST be reported as a labelled second curve, not substituted for the first (DEC-9). *(verify: integration test asserting the shared channel object, and both codec curves present in the results CSV)*
- **BR-2** — The LDPC chain MUST be validated two ways before any learned-vs-classical comparison is reported. **Bit-exactly**: encoder output and rate-matched output MUST reproduce `params.baseline.ldpc_golden_vector_file`, a fixture generated from an implementation independent of `params.baseline.ldpc_impl` whose provenance is fixed at `params.baseline.ldpc_golden_vector_source_gate`. **Statistically**: measured BER/BLER MUST match published `params.baseline.ldpc_standard` curves **for every entry in `params.baseline.modulations` that BR-4 may select**, with every reference curve converted to this project's SNR convention using `params.channel.snr_conversion` and the conversion shown in the archived artifact. Higher-order modulations need soft-demapping to log-likelihood ratios rather than a sign decision, so an unvalidated 16-QAM demapper would silently degrade the baseline — the opposite of BR-4's intent and a direct threat to DEC-16. Rationale: a curve overlay is a judgement call and a fixture is a test, so the golden vectors are what turn G-2 into a red/green signal; and separately, a missed Eb/N0-to-Es/N0 conversion shifts rate 1/3 and rate 5/6 by about 4 dB *relative to each other*, which inverts BR-4's rate selection rather than merely offsetting the curve. *(verify: passing golden-vector test, plus an archived BER-vs-SNR plot with reference curve overlaid and the conversion arithmetic stated)*
- **BR-3** — The baseline MUST receive exactly the same number of complex channel uses `k` as the learned system, per `params.baseline.budget_rule`. Bandwidth matching is counted in channel uses, not in bytes. *(verify: unit test counting emitted symbols for both systems)*
- **BR-4** — At each SNR the baseline MUST be tuned in its own favour — sweeping codec target rate (or `params.baseline.jpeg_quality_grid` for the secondary curve) × `params.baseline.ldpc_rates` × `params.baseline.modulations` per `params.baseline.modulation_tuning`, and taking the **best feasible** configuration per `params.baseline.tuning` — with the selection made **on the validation split** and then frozen for the test evaluation (DEC-12). Modulation is an adaptive axis rather than a fixed choice (DEC-16): capping it would leave the baseline unable to exploit a clean link, which is an artificial handicap rather than a fair comparison. Reporting a single fixed configuration across all SNRs is prohibited. The sweep MUST be computed as a cached feasibility table plus per-(rate, SNR, blocklength) block-error characterisation composed analytically, not as a full per-image channel simulation of every cell. *(verify: sweep artifact showing the selected config per SNR, its provenance on the validation split, and the composition method)*
- **BR-5** — If no codec configuration produces a file fitting the payload budget, the transmission MUST be recorded as infeasible, counted in `infeasible_rate`, and scored per BR-13, never silently skipped. *(verify: unit test at the smallest ratio where infeasibility is expected)*
- **BR-6** — A file that cannot be decoded after LDPC decoding MUST be scored per BR-13 and counted in `decode_failure_rate`. *(verify: unit test injecting an undecodable block)*
- **BR-7** — Both systems MUST see identical test images and identical noise realisations at a given seed and SNR. *(verify: test asserting bitwise-identical noise draws across the two pipelines)*
- **BR-8** — A frozen `params.reference_classifier` trained from scratch on clean images (DEC-15) MUST meet each dataset's `clean_acc_floor`, and the same instance MUST score the classical reconstructions, the ER-9 control and the semantic reconstruction ablation (ER-4). *(verify: measured clean accuracy per dataset, archived, with a training log showing no pretrained initialisation)*
- **BR-9** — The headline classical curve is the adaptive one produced by BR-4 over `params.baseline.modulations`. `params.baseline.core_modulation` defines the **fixed-modulation reference curve**, which MUST also be reported so the contribution of adaptive modulation is separable, and it is the modulation the ER-9 control uses so that control stays matched to a single chain. *(verify: config test, plus both the adaptive and fixed-modulation classical curves present in the results CSV)*
- **BR-10** — Bit accounting MUST be complete and normative: `params.baseline.tb_crc_bits`, `params.baseline.cb_crc_bits`, code-block segmentation at `params.baseline.code_block_max_bits`, per-code-block rate-matching budget distribution (TS 38.212 §5.4.2.1 `E_r`), base-graph selection per `params.baseline.ldpc_base_graph`, and filler and rate matching per `params.baseline.rate_matching` all consume budget before any source byte does, exactly as in `params.baseline.budget_rule`. Partial final blocks and padding MUST be accounted, not discarded. The steps in `params.baseline.ldpc_impl_local` are implemented in this project rather than taken from `params.baseline.ldpc_impl`, which does not provide them (DEC-10). Rationale: without this the "equal channel uses" claim is disputable and the payload figure is wrong by a variable margin. Segmentation is unavoidable rather than optional at these budgets — Imagenette at the core ratio gives k=25600 symbols → 51200 channel bits → about 42,666 information bits at rate 5/6, which against `code_block_max_bits` 8448 and a 24-bit code-block CRC is six code blocks; even the low-ratio operating point needs two. *(verify: unit test reconciling emitted symbol count against k for several payload sizes including a single-block case, a partial final block, and a multi-block case)*
- **BR-11** — Container and header bytes MUST count against the budget per `params.baseline.container_policy`, and every baseline result row MUST report `header_bytes` and `payload_bytes` separately so the fraction of the budget spent on format overhead is visible at every operating point. Rationale: at these budgets container overhead is a first-order term, and an uncontrolled overhead difference would let a low-SNR "cliff" be a file-format artifact rather than a channel effect (DEC-9). *(verify: schema test on the two columns, and an archived overhead-fraction table per dataset × ratio × rate)*
- **BR-12** — A second `params.reference_classifier` instance, fine-tuned on codec-artifacted images at the operating quality, MUST also score the classical reconstructions, and both scores MUST be reported. Rationale: scoring heavily-artifacted reconstructions with a classifier trained only on clean images measures domain shift as well as delivered information, and the learned system's head has no equivalent handicap because it trains on its own post-channel outputs. *(verify: both scoring paths present as distinct `system` values in the results CSV)*
- **BR-13** — An outage — infeasible (BR-5) or undecodable (BR-6) — MUST be scored per `params.baseline.outage_policy` by drawing a label uniformly at random from the dataset's classes using the run's `channel_seed`, yielding a **binary** per-image outcome with expectation at chance level. Assigning a fractional accuracy is prohibited: it is not a per-image outcome and would break the paired inference in ER-10. *(verify: unit test that outage draws are reproducible under a fixed seed and that the empirical outage accuracy converges to 1/classes)*
- **BR-14** — `params.baseline.ldpc_impl` MUST sit behind an adapter exposing only the operations in `params.baseline.ldpc_impl_provides`, with every step in `params.baseline.ldpc_impl_local` implemented above that seam and unit-testable against a stub encoder. Substituting `params.baseline.ldpc_impl_fallback` MUST NOT require any change above the seam. Rationale: this is what makes DEC-10's fallback ladder real rather than aspirational, and it mirrors the registry pattern already required for channels (SR-5) and task heads (SR-15). *(verify: unit test exercising the segmentation layer against a stub encoder, plus an adapter conformance test run against both implementations)*
- **BR-15** — The **adaptation asymmetry MUST be disclosed** wherever the headline comparison appears: the classical system is re-tuned at every SNR across codec rate, `params.baseline.ldpc_rates` and `params.baseline.modulations`, while the learned system is trained once per DEC-11 and evaluated frozen. Rationale: the asymmetry is deliberate and favours the baseline, which is what BR-4 exists to do — but it is also a large part of why any crossover appears, and an advantage handed to the baseline is still a confound if it goes unstated. Disclosed in advance it reads as rigour; discovered by an examiner it reads as a thumb on the scale. OPT-4's SNR-randomised learned variant is the natural counterpart to report alongside. *(verify: the asymmetry stated in the thesis methods section and in the caption of every headline figure, and the fixed-modulation reference curve of BR-9 present for comparison)*

## 7. Experiment requirements (ER)

- **ER-1** — **Core comparison.** Learned, classical and ER-9 control systems, full test split, every SNR in `params.channel.test_snr_grid_db`, at `params.bandwidth.core_ratio`, repeated over `params.evaluation.train_seeds` and `params.evaluation.channel_seeds`, reported with `params.evaluation.ci` intervals, measuring `params.project.primary_metric`. Models are trained per DEC-11. Hypotheses are decided by §2 via ER-10. *(verify: results CSV + accuracy-vs-SNR figure)*
- **ER-2** — **SNR mismatch.** Train once at `params.channel.train_snr_db_fixed`, evaluate across the whole test grid, and report the degradation profile of both systems to evidence the cliff-versus-graceful contrast (H2). *(verify: results CSV + figure)*
- **ER-3** — **Operating-point selection and bandwidth sweep.** Before the headline run, sweep every entry in `params.bandwidth.ratios` on `params.evaluation.test_subset_size` validation images at one seed, and select `params.bandwidth.core_ratio` at G-8 by this preregistered, **learned-blind** rule: *the smallest ratio at which the classical system's high-SNR ceiling comes within 5 percentage points of its clean-image accuracy*. The rule inspects only the classical system, so it cannot be accused of selecting for the hypothesis; it places the comparison where the baseline is healthy rather than where it is starved. Because BR-4 now adapts modulation as well (DEC-16), the classical ceiling rises faster with SNR than it would under a fixed modulation, so the rule is expected to select a *smaller* ratio than it otherwise would — the crossover is bought with better use of the budget rather than with more budget. The sweep also reports `params.bandwidth.low_ratio_operating_point` as the bandwidth-starved regime where dominance rather than crossover is expected. *(verify: results CSV + figure, and an archived record of the rule's evaluation at every ratio)*
- **ER-4** — **Task-training ablation.** The semantic reconstruction head MUST also be scored through the frozen `params.reference_classifier` (BR-8), separating gain attributable to joint coding from gain attributable to a task-trained classifier. Reported as a distinct `system` value. *(verify: results CSV containing the ablation rows)*
- **ER-5** — Every experiment MUST emit rows matching `params.artifacts.csv_schema` exactly — same columns, same order — alongside the per-image file required by SR-18. *(verify: schema validation script over all result CSVs)*
- **ER-6** — Evaluation on `params.evaluation.test_subset_size` images is permitted for sweeps only; `params.evaluation.full_test_split_required_for` MUST use the full split, and the `test_subset` column MUST record which was used. *(verify: schema test asserting the flag)*
- **ER-7** — Every number appearing in the thesis MUST be traceable to a `run_id` and `git_commit` in a committed CSV. *(verify: audit script resolving each reported figure to its rows)*
- **ER-8** — If a hypothesis in §2 is unsupported, the negative result MUST be reported with the same rigour as a positive one, together with the diagnostic evidence. Weakening the baseline to manufacture support is prohibited. *(verify: review against BR-4 sweep artifacts and BR-11 overhead tables)*
- **ER-9** — **Task-aware digital control.** A third system MUST be built and evaluated at matched `k`: learned features, quantised and entropy-coded, carried over the same `params.baseline.channel_code` and `params.baseline.core_modulation` chain as the classical baseline, and scored by the same frozen classifier. Rationale: without it, the learned-versus-classical gap conflates task-aware representation with joint coding, and §1 claim 2 cannot be attributed (H4). This is the primary novelty claim of DEC-13. *(verify: results CSV rows under a distinct `system` value at the same k as ER-1)*
- **ER-10** — **Paired inference.** The §2 hypotheses MUST be decided on **paired per-image outcomes** (SR-18), not on the overlap of two independent intervals: `params.evaluation.paired_test` for the point decision and `params.evaluation.ci` with `params.evaluation.bootstrap_resamples` for the interval on the accuracy *difference*, with the image-level bootstrap nested inside the seed hierarchy so `params.evaluation.train_seeds` and `params.evaluation.channel_seeds` contribute as separate variance components. Rationale: both systems see identical images and noise draws, so a paired test is both the natural and the far more powerful analysis — a three-seed Student-t interval has two degrees of freedom and would demand roughly a five-point gap to clear non-overlap. *(verify: inference script reproducing every reported interval from the per-image files, with a unit test on synthetic data of known effect size)*

## 8. Demo requirements (DR)

- **DR-1** — An SNR slider spanning `params.channel.test_snr_grid_db` MUST drive both pipelines live on the same input image. *(verify: manual demo script walkthrough)*
- **DR-2** — The interface MUST show, side by side, the classical output, the semantic reconstruction, and each system's predicted label with confidence. *(verify: manual walkthrough)*
- **DR-3** — The accuracy-vs-SNR plot MUST update live with a marker at the current slider position. *(verify: manual walkthrough)*
- **DR-4** — The demo MUST render figures through `params.demo.figure_style_module`, the same module that renders thesis figures, using `params.demo.fonts` and `params.demo.palette`, with default framework chrome suppressed (DEC-7). *(verify: visual review that a demo figure and its thesis counterpart come from the same style module — pixel comparison is not required and is not achievable across DPI, backend and font hinting)*
- **DR-5** — The demo MUST run with `params.demo.offline` true and remain usable on CPU only. *(verify: run with networking disabled on a CPU-only machine)*
- **DR-6** — The demo MUST consume frozen checkpoints and committed result CSVs; it MUST NOT train, fine-tune or recompute reported metrics. *(verify: code review)*

## 9. Hardware requirements (HR)

Tier 2 and Tier 3 are stretch goals with a pre-recorded demonstration as the expected outcome (DEC-14).

- **HR-1** — Tier 2/3 hardware MUST satisfy `params.hardware_tier23.needs` at no less than `params.hardware_tier23.min_sample_rate_msps`, within `params.hardware_tier23.budget_inr_range` and subject to `params.hardware_tier23.budget_note`. Devices in `params.hardware_tier23.candidates` are indicative, not selected. *(verify: procurement checklist against the capability list)*
- **HR-2** — No hardware may be purchased before `params.hardware_tier23.purchase_gate` passes. *(verify: gate record)*
- **HR-3** — **Tier 2 (stretch).** Encoder output SHOULD be replayed as IQ through a real link (wired loopback with attenuator) and captured, then decoded offline and compared against the simulated result at matched measured SNR. *(verify: measured-vs-simulated accuracy table, or a recorded decision not to attempt)*
- **HR-4** — **Tier 3 (stretch).** A live encoder/decoder demo on `params.hardware_tier23.edge_node` SHOULD meet `params.hardware_tier23.live_demo_latency_budget_ms` end-to-end. If it cannot, resolve at G-6 by pre-recording per `params.hardware_tier23.expected_demonstration`. *(verify: measured latency distribution, or the pre-recorded artifact)*
- **HR-5** — No Tier 1 requirement may depend on hardware availability. Tier 1 MUST be completable, reportable and defensible with simulation alone. *(verify: review of SR/BR/ER for hardware dependencies)*
- **HR-6** — Any Tier 2 replay MUST specify and implement the full physical-layer wrapper in `params.hardware_tier23.framing`, and MUST state how channel SNR is *measured* rather than assumed, so the measured-versus-simulated comparison in HR-3 is meaningful. Rationale: an IQ replay without framing, timing recovery and frequency-offset correction does not produce a comparable link, and pilot overhead changes the channel-use accounting that BR-3 depends on. *(verify: framing design note plus a loopback capture showing locked timing and residual CFO within tolerance)*

## 10. Programme deliverables (PR)

Graded course deliverables. They are not engineering requirements, but they carry roughly thirty of the hundred available marks and are invisible in an engineering-only schedule, so they are tracked here with the same discipline.

- **PR-1** — A literature review of at least `params.deliverables.literature_review_min_refs` references MUST be drafted before the first review and maintained thereafter, covering DJSCC, separation and finite-blocklength theory, and learned image compression. *(verify: reference list committed and cited in the report draft)*
- **PR-2** — A time plan in the form of `params.deliverables.time_plan_artifact` MUST exist by the first review week in `params.deliverables.review_weeks` and be updated at each subsequent review. *(verify: committed chart artifact with revision history)*
- **PR-3** — A standards and tools register MUST list every entry in `params.deliverables.standards` with where each is used in the implementation. *(verify: register committed and each entry resolvable to code or a spec requirement)*
- **PR-4** — A poster in `params.deliverables.poster_format` MUST be prepared and submitted. *(verify: submitted artifact)*
- **PR-5** — A plagiarism report MUST be produced per `params.deliverables.plagiarism_report_required` and submitted with the final report. *(verify: submitted artifact)*
- **PR-6** — The final report MUST conform to `params.deliverables.report_format_source`, proof-read and ratified by the guide. *(verify: format checklist signed off)*
- **PR-7** — A written novelty statement MUST name `params.deliverables.novelty_claims` explicitly, state what in each is not present in the prior work of §1, and be defensible independently of whether §2's hypotheses are supported (DEC-13). *(verify: statement committed and reviewed against the literature list from PR-1)*
- **PR-8** — Each review in `params.deliverables.review_weeks` MUST have a prepared package matching that review's rubric weighting, and the engineering gates MUST be scheduled so the required evidence exists beforehand. *(verify: review package committed before each review week)*

## 11. Optional cross-checks (OPT)

Non-blocking. None of these may appear on the critical path (DEC-3). OPT-1 and OPT-3 further depend on a MATLAB licence being confirmed at G-9; if it is not, they are dropped without prejudice and BR-2's golden vectors are sourced from another independent implementation instead.

- **OPT-1** — MATLAB 5G Toolbox reproduction of the LDPC BER curve as an independent check on BR-2, and as a generator for its golden vectors if the licence is available.
- **OPT-2** — MATLAB or symbolic treatment of channel capacity, the separation theorem and the finite-blocklength bounds behind §1 claim 2, for the thesis mathematics chapter.
- **OPT-3** — An independent MATLAB reimplementation of the JPEG 2000 + LDPC chain to cross-validate baseline accuracy at two or three SNR points.
- **OPT-4** — SNR-randomised training over `params.channel.train_snr_db_set` as a comparison against the fixed-SNR protocol of DEC-11.

## 12. Future work & mandated extension points (FW)

Not built now; the spec is shaped so each is additive rather than a redesign.

- **FW-1** — Rayleigh block and fast fading, via the channel registry (SR-5). Expected to strengthen the graceful-degradation claim, since classical schemes suffer disproportionately under fading.
- **FW-2** — λ sweep quantifying the accuracy cost of a viewable reconstruction, with λ=0 as the pure-task upper bound, via SR-8. The `lambda` column already exists in `params.artifacts.csv_schema` so this is additive.
- **FW-3** — SNR-adaptive or variable-rate coding, where the transmitter adjusts rate to measured channel state.
- **FW-4** — Alternative downstream tasks (segmentation, detection) via the task-head registry (SR-15).
- **FW-5** — Further digital semantic variants beyond the ER-9 control — learned entropy models, joint quantisation training.
- **FW-6** — Additional datasets beyond `params.datasets`.

## 13. Schedule & gates

Weeks are relative (W1 = first working week) and rescale to the actual semester length. Gates are go/no-go: a failed gate triggers its stated fallback, not an extension. Review weeks are fixed by `params.deliverables.review_weeks` and the engineering schedule is arranged so each review has evidence to show (PR-8).

| Week | Work | Gate |
|---|---|---|
| W0 | Pre-flight: §2 protocol ratified with the supervisor; DEC-9..DEC-15 closed; LDPC spike (install, exact-budget check, throughput, smallest workable payload) and golden-vector provenance fixed; PR-1 literature review begun | **G-9** |
| W1 | Repo scaffold, config plumbing (SR-1), data loaders and validation splits (SR-2, SR-17), reference classifier trained from scratch, clean and artifact-finetuned (BR-8, BR-12) | **G-1** |
| W2 | Channel model, power normalisation and PAPR (SR-4..SR-7, SR-16); DJSCC skeleton; compute profiling | **G-7** |
| W3 | LDPC integration, BER validation with SNR conversion (BR-2), full bit accounting and packetisation (BR-10, BR-11) | **G-2** |
| W4 | Classical baseline end-to-end with budget matching and validation-split tuning (BR-3, BR-4); **First Review** package (PR-1, PR-2, PR-8) | |
| W5 | DJSCC training loop, dual head, checkpoint/resume (SR-8, SR-10); CIFAR-10 plumbing smoke path; results schema and per-image outcomes (ER-5, SR-18) | |
| W6 | ER-3 operating-point sweep on validation subset across all ratios; `core_ratio` selected | **G-8** |
| W7 | Headline dataset training at the selected ratio | |
| W8 | λ calibration on validation split (SR-9) | **G-4** |
| W9 | ER-2 SNR-mismatch experiment; ER-9 task-aware digital control | |
| W10 | ER-4 ablation; paired inference implemented (ER-10); **Second Review** package (PR-8) | |
| W11 | Full ER-1: full test split, train × channel seeds, paired intervals | |
| W12 | Results frozen; Tier 1 reported either way (ER-8) | **G-5** |
| W13 | Streamlit demo and shared figure-style module (DR-1..DR-6) | |
| W14 | Stretch: Tier 2 SDR replay (HR-3, HR-6) if G-5 passed and hardware arrived; otherwise pre-recorded demonstration. Poster draft (PR-4) | |
| W15 | Thesis figures, report in prescribed format (PR-6), results audit (ER-7), novelty statement (PR-7), plagiarism report (PR-5). Tier 3 attempt if Tier 2 landed | **G-6** |
| W16 | **Third Review**, buffer, viva preparation | |

- **G-1** — Reference classifier meets `clean_acc_floor` on the smoke dataset and on the headline dataset, from scratch (DEC-15), in both the clean and artifact-finetuned variants. Fallback: switch backbone or extend training before any DJSCC work begins.
- **G-2** — Golden vectors pass bit-exactly, LDPC BER matches published curves within tolerance with the SNR conversion shown, and the BR-10 packetisation tests pass. Fallback: descend DEC-10's ladder — pin an earlier library version, then substitute `params.baseline.ldpc_impl_fallback` behind the BR-14 seam. No comparison may be reported before this passes.
- *(G-3 is retired — see §14. It required a crossover reproduced on the CIFAR-10 smoke path, which cannot happen for the reasons in DEC-1. Superseded by G-8.)*
- **G-4** — λ calibrated per SR-9. Fallback: relaxed PSNR floor, then DEC-2 reversal to two separate models with a re-planned schedule.
- **G-5** — Tier 1 frozen: ER-1..ER-4, ER-9 and ER-10 complete with paired intervals and every §2 hypothesis decided. Passing unlocks the hardware purchase (HR-2). Failing means Tier 2/3 are abandoned and effort moves to reporting per ER-8. Note that Tier 1 completion does not require the hypotheses to be supported (§2).
- **G-6** — Tier 3 latency budget met, if attempted. Fallback: pre-recorded demonstration per DEC-14. This gate MUST NOT change the dataset: the headline dataset is frozen at G-5 and any demotion here is demo-only.
- **G-7** — Compute profile established per SR-11: achievable batch size, epoch time and peak VRAM measured on the primary device. Fallback: reduce model width or resolution before the training schedule is committed.
- **G-8** — `core_ratio` selected per ER-3's learned-blind rule, and the classical baseline confirmed non-degenerate there — feasible at more than one LDPC rate, with format overhead below half the budget (BR-11). This gate also decides DEC-16: whether the adaptive-modulation baseline produces an observable crossover, or whether the last-resort fallback is taken. Fallback: if no ratio in `params.bandwidth.ratios` satisfies the rule on Imagenette, invoke the DEC-1 demotion to STL-10; if none satisfies it there either, report the format-floor finding as a first-class result and run the comparison at the largest available ratio.
- **G-9** — Pre-flight decisions closed: §2 ratified with the supervisor, DEC-9..DEC-15 recorded. The LDPC spike is complete — `params.baseline.ldpc_impl_version` installs and runs on the primary device, `params.baseline.ldpc_decoder` is confirmed available, the encoder is confirmed to hit an exact channel-bit budget (which BR-3 depends on), batched decode throughput and the smallest workable payload are measured and recorded, and golden-vector provenance is named per `params.baseline.ldpc_golden_vector_source_gate`. MATLAB licence availability is resolved either way, settling whether OPT-1 and OPT-3 are live. Fallback: none — W1 does not begin until this passes, because every one of these decisions is expensive to reverse after code exists.

## 14. Retired requirements

Retired IDs keep their numbers reserved. Nothing may reuse them.

- ~~**G-3**~~ — retired W0; see §13.

## 15. Non-goals

Explicitly out of scope. Listed so that scope creep is a visible spec change.

- Bit-exact or perceptually-optimal reconstruction as an objective in its own right — reconstruction exists to serve the task and the demo.
- Reinforcement learning of any kind. The system is trained end-to-end by supervised gradient descent.
- Multi-user, MIMO, relay or interference channels; a single point-to-point link is assumed.
- Video, audio or real-time streaming sources.
- On-device training or adaptation; edge nodes run frozen models.
- Adversarial robustness, security or privacy of the learned representation.
- Beating state-of-the-art DJSCC results. The comparison of record is against the classical baseline, not against the literature; the novelty claimed is the attribution decomposition and the overhead-controlled baseline (DEC-13, PR-7), not headline numbers.
- Energy measurement. The systems are matched on channel uses and average power, which is equal transmitted energy by construction; no power-meter claim is made (§1).

## 16. Open items and carried risks

Working state, not normative: what is still provisional, what is pending, and what risk is knowingly being carried. This section shrinks as gates pass. It exists so that a reader — or the author after a gap — can tell at a glance which numbers in §4 are measurements and which are estimates.

**Pending before W1 (all inside G-9).**

- §2's rewrite needs **supervisor ratification**. The "curves must cross" wording came from the accepted proposal, so it cannot be silently redefined; present it as a preregistration refinement, using `docs/crossover-explained.md` as the supporting document — it carries the ceiling arithmetic in both plain and technical form, and its Part 4 is written as the argument to make. This is the only open item with an external dependency, so it should go first.
- The **LDPC spike**. Its load-bearing item is confirming the encoder hits an *exact* channel-bit budget — BR-3's equal-channel-uses claim rests entirely on that, and it is a half-hour check at W0 versus a structural problem at W3. Also: install on the primary device, confirm `params.baseline.ldpc_decoder` is accepted, measure batched decode throughput against the roughly 1.1M code-block decodes ER-1 needs at the core ratio, and record the smallest workable payload size.
- **Golden-vector provenance** for BR-2, which is contingent on the MATLAB licence question resolving either way.

**Provisional values that a gate will replace.** Each is an estimate standing in for a measurement; none should be cited as a result.

- `bandwidth.core_ratio` = `r_1_3`, provisional until G-8 selects it by ER-3's learned-blind rule. The estimate comes from the classical ceiling reaching clean accuracy at roughly 1.5–2.0 bpp for Imagenette. Under the original QPSK-only cap that put the crossover near r ≈ 2/5; with DEC-16's adaptive modulation, 16-QAM doubles bits per channel use and halves the requirement to **r ≈ 1/5**, so `r_1_6` (1.67 bpp at 16-QAM rate 5/6) may well suffice and `r_1_3` (3.33 bpp) is the conservative provisional pick. Expect G-8 to select at or below `r_1_3` — that reduction is the point of DEC-16, since it buys the crossover with better use of the budget rather than more of it.
- `learned_system.lambda_core` = 1.0, provisional until G-4 calibrates it per SR-9.
- `datasets.*.clean_acc_floor`, set to from-scratch estimates under DEC-15. If G-1 shows them wrong, move them **at G-1** as a recorded spec change rather than quietly missing them later.

**Risks carried knowingly.**

- **Sionna 2.0.x is a recent framework rewrite** — PHY/SYS moved from TensorFlow to PyTorch in 2.0.0, with 2.0.1 released March 2026 — so early-version defects are plausible. This is the main risk accepted in DEC-10. Mitigations are already in place: BR-2's golden vectors turn a defect into a failing test, BR-14's seam makes substitution cheap, and DEC-10 states the fallback ladder. Pin the exact version in the lockfile required by SR-12.
- **Python 3.14 is ahead of the declared floors** for both Sionna (3.11) and PyTorch (2.9). Both resolve on the primary device, but it is not a combination either project tests. The W0 spike settles it; pinning a 3.12 or 3.13 interpreter is the fallback and costs nothing else.
- **A crossover may not appear even with the adaptive baseline.** DEC-16 removes the artificial modulation cap, which is the strongest legitimate lever available, but both systems are converging on the same wall — the classifier's accuracy on a perfect image. If the learned system also reaches that wall, the curves meet rather than cross. G-8 decides this; §2 already makes the outcome a complete Tier 1 either way, and the reconstruction-quality crossover is the documented fallback figure. Do not respond to a missing crossover by weakening the learned system.
- **16-QAM adds baseline engineering that must not be skimped.** Soft-demapping to log-likelihood ratios is materially harder than the QPSK sign decision, and a subtly wrong demapper degrades the baseline invisibly — which would defeat the entire purpose of DEC-16 and hand an examiner a real objection. BR-2 now requires per-modulation validation for this reason; budget two to three days at W3, not an afternoon.
- **ER-9 is both a novelty claim and an attribution requirement.** It is the one required experiment with no prior art to copy, so it carries more design risk than ER-1..ER-4 — and because DEC-13 leans on it for rubric novelty, it should not be the first thing cut under schedule pressure.
- **The W14 stretch row assumes hardware ordered after G-5 arrives in time.** DEC-14 already makes this upside rather than plan, so the risk is to the stretch goal only, never to Tier 1 (HR-5).
