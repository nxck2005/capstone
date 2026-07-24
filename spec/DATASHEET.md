<!-- GENERATED FROM spec/SPEC.md BY tools/gen_spec_views.py - DO NOT EDIT -->

# Datasheet

Every committed parameter, flattened. Normative source: [`SPEC.md`](SPEC.md) §4.

## project

| Parameter | Value | Cited by |
| --- | --- | --- |
| `project.id` | semcom-djscc | - |
| `project.task` | image-classification-over-noisy-channel | - |
| `project.primary_metric` | top1_accuracy_vs_snr | ER-1 |
| `project.tier1_channel` | simulated | - |

## datasets

| Parameter | Value | Cited by |
| --- | --- | --- |
| `datasets.imagenette160.role` | headline | FW-6, SR-2 |
| `datasets.imagenette160.image_size` | 160, 160, 3 | FW-6, SR-2 |
| `datasets.imagenette160.n` | 76800 | FW-6, SR-2 |
| `datasets.imagenette160.classes` | 10 | FW-6, SR-2 |
| `datasets.imagenette160.train_images` | 9469 | FW-6, SR-2 |
| `datasets.imagenette160.test_images` | 3925 | FW-6, SR-2 |
| `datasets.imagenette160.clean_acc_floor` | 0.9 | FW-6, SR-2 |
| `datasets.stl10.role` | demotion_tier3 | FW-6, SR-2 |
| `datasets.stl10.image_size` | 96, 96, 3 | FW-6, SR-2 |
| `datasets.stl10.n` | 27648 | FW-6, SR-2 |
| `datasets.stl10.classes` | 10 | FW-6, SR-2 |
| `datasets.stl10.train_images` | 5000 | FW-6, SR-2 |
| `datasets.stl10.test_images` | 8000 | FW-6, SR-2 |
| `datasets.stl10.clean_acc_floor` | 0.85 | FW-6, SR-2 |
| `datasets.cifar10.role` | fallback_and_smoke | FW-6, SR-2 |
| `datasets.cifar10.image_size` | 32, 32, 3 | FW-6, SR-2 |
| `datasets.cifar10.n` | 3072 | FW-6, SR-2 |
| `datasets.cifar10.classes` | 10 | FW-6, SR-2 |
| `datasets.cifar10.train_images` | 50000 | FW-6, SR-2 |
| `datasets.cifar10.test_images` | 10000 | FW-6, SR-2 |
| `datasets.cifar10.clean_acc_floor` | 0.93 | FW-6, SR-2 |

## bandwidth

| Parameter | Value | Cited by |
| --- | --- | --- |
| `bandwidth.symbol_type` | complex_baseband | - |
| `bandwidth.power_constraint` | unit_average_power | SR-4 |
| `bandwidth.ratios.r_1_6` | 1/6 | ER-3 |
| `bandwidth.ratios.r_1_12` | 1/12 | ER-3 |
| `bandwidth.ratios.r_1_24` | 1/24 | ER-3 |
| `bandwidth.core_ratio` | r_1_12 | ER-1 |
| `bandwidth.k_symbols.imagenette160.r_1_6` | 12800 | SR-3 |
| `bandwidth.k_symbols.imagenette160.r_1_12` | 6400 | SR-3 |
| `bandwidth.k_symbols.imagenette160.r_1_24` | 3200 | SR-3 |
| `bandwidth.k_symbols.stl10.r_1_6` | 4608 | SR-3 |
| `bandwidth.k_symbols.stl10.r_1_12` | 2304 | SR-3 |
| `bandwidth.k_symbols.stl10.r_1_24` | 1152 | SR-3 |
| `bandwidth.k_symbols.cifar10.r_1_6` | 512 | SR-3 |
| `bandwidth.k_symbols.cifar10.r_1_12` | 256 | SR-3 |
| `bandwidth.k_symbols.cifar10.r_1_24` | 128 | SR-3 |

## channel

| Parameter | Value | Cited by |
| --- | --- | --- |
| `channel.snr_definition` | Es/N0 in dB per complex channel use, measured after unit-average-power normalisation | SR-7 |
| `channel.models_supported` | awgn | SR-5 |
| `channel.models_planned` | rayleigh_block, rayleigh_fast | SR-5 |
| `channel.train_snr_db_fixed` | 7 | ER-2 |
| `channel.train_snr_db_set` | 1, 4, 7, 13, 19 | - |
| `channel.test_snr_grid_db` | -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25 | DR-1, ER-1 |

## learned_system

| Parameter | Value | Cited by |
| --- | --- | --- |
| `learned_system.framework` | pytorch | - |
| `learned_system.encoder` | conv_downsample_to_k_symbols | - |
| `learned_system.decoder_heads` | reconstruction, classification | SR-8 |
| `learned_system.loss` | CE + lambda * MSE | SR-8 |
| `learned_system.lambda_core` | 1.0 | SR-9 |
| `learned_system.lambda_calibration_gate` | G-4 | - |
| `learned_system.optimizer` | adam | - |
| `learned_system.lr` | 0.001 | - |
| `learned_system.lr_schedule` | cosine | - |
| `learned_system.amp` | true | - |
| `learned_system.max_params_millions` | 10 | SR-14 |
| `learned_system.batch_size.imagenette160` | 32 | - |
| `learned_system.batch_size.stl10` | 64 | - |
| `learned_system.batch_size.cifar10` | 128 | - |
| `learned_system.epochs.imagenette160` | 100 | - |
| `learned_system.epochs.stl10` | 200 | - |
| `learned_system.epochs.cifar10` | 150 | - |

## baseline

| Parameter | Value | Cited by |
| --- | --- | --- |
| `baseline.source_codec` | jpeg | BR-1 |
| `baseline.jpeg_quality_grid` | 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95 | BR-4 |
| `baseline.channel_code` | 5g_nr_ldpc | BR-1 |
| `baseline.ldpc_rates` | 1/3, 1/2, 2/3, 5/6 | BR-4 |
| `baseline.ldpc_decoder` | normalized_min_sum | - |
| `baseline.ldpc_max_iters` | 50 | - |
| `baseline.modulations` | bpsk, qpsk | BR-9 |
| `baseline.core_modulation` | qpsk | BR-9 |
| `baseline.budget_rule` | payload_bits = floor(k * bits_per_symbol * rate); the JPEG file MUST fit within payload_bits | BR-3 |
| `baseline.outage_policy` | chance_level | BR-5, BR-6 |
| `baseline.tuning` | best_feasible_config_per_test_snr | BR-4 |

## reference_classifier

| Parameter | Value | Cited by |
| --- | --- | --- |
| `reference_classifier.arch` | resnet18 | BR-8, ER-4 |
| `reference_classifier.trained_on` | clean_images | BR-8, ER-4 |
| `reference_classifier.frozen` | true | BR-8, ER-4 |
| `reference_classifier.shared_by` | classical_baseline, semantic_recon_ablation | BR-8, ER-4 |

## evaluation

| Parameter | Value | Cited by |
| --- | --- | --- |
| `evaluation.seeds` | 0, 1, 2 | ER-1, SR-12 |
| `evaluation.repeats` | 3 | - |
| `evaluation.ci` | student_t_95 | ER-1 |
| `evaluation.test_subset_size` | 2000 | ER-6 |
| `evaluation.full_test_split_required_for` | ER-1 | ER-6 |
| `evaluation.metrics` | top1_acc, psnr_db, ssim, bytes_sent, decode_failure_rate | - |

## compute

| Parameter | Value | Cited by |
| --- | --- | --- |
| `compute.primary_device` | rtx_4060_mobile_8gb | SR-11 |
| `compute.overflow` | colab_free, kaggle_free | - |
| `compute.vram_budget_gb` | 7.0 | SR-11 |
| `compute.max_wall_clock_hours_per_run` | 2 | SR-11 |
| `compute.checkpoint_every_epochs` | 1 | SR-10 |

## artifacts

| Parameter | Value | Cited by |
| --- | --- | --- |
| `artifacts.results_dir` | results/ | SR-9, SR-13 |
| `artifacts.checkpoint_dir` | checkpoints/ | SR-13 |
| `artifacts.figures_dir` | figures/ | SR-13 |
| `artifacts.csv_schema` | run_id, timestamp, git_commit, system, dataset, n, k, bw_ratio, channel, train_snr_db, test_snr_db, seed, jpeg_quality, ldpc_rate, modulation, top1_acc, psnr_db, ssim, bytes_sent, decode_failure_rate, n_test, test_subset | ER-5, SR-13 |

## demo

| Parameter | Value | Cited by |
| --- | --- | --- |
| `demo.framework` | streamlit | - |
| `demo.figure_style_module` | src/viz/style.py | DR-4 |
| `demo.fonts` | serif_computer_modern | DR-4 |
| `demo.palette` | colorblind_safe | DR-4 |
| `demo.offline` | true | DR-5 |
| `demo.cpu_only_capable` | true | - |

## hardware_tier23

| Parameter | Value | Cited by |
| --- | --- | --- |
| `hardware_tier23.needs` | iq_playback, iq_capture, tx_capable_pair, wired_loopback_with_attenuator | HR-1 |
| `hardware_tier23.min_sample_rate_msps` | 1 | HR-1 |
| `hardware_tier23.candidates` | adalm_pluto_x2, hackrf_one_plus_rtlsdr | HR-1 |
| `hardware_tier23.budget_inr_range` | 25000, 40000 | HR-1 |
| `hardware_tier23.edge_node` | raspberry_pi_4_or_5 | HR-4 |
| `hardware_tier23.live_demo_latency_budget_ms` | 500 | HR-4 |
| `hardware_tier23.purchase_gate` | G-5 | HR-2 |
