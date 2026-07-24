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
| `datasets.imagenette160.train_images` | 8469 | FW-6, SR-2 |
| `datasets.imagenette160.val_images` | 1000 | FW-6, SR-2 |
| `datasets.imagenette160.test_images` | 3925 | FW-6, SR-2 |
| `datasets.imagenette160.clean_acc_floor` | 0.88 | FW-6, SR-2 |
| `datasets.stl10.role` | fallback_headline | FW-6, SR-2 |
| `datasets.stl10.image_size` | 96, 96, 3 | FW-6, SR-2 |
| `datasets.stl10.n` | 27648 | FW-6, SR-2 |
| `datasets.stl10.classes` | 10 | FW-6, SR-2 |
| `datasets.stl10.train_images` | 4500 | FW-6, SR-2 |
| `datasets.stl10.val_images` | 500 | FW-6, SR-2 |
| `datasets.stl10.test_images` | 8000 | FW-6, SR-2 |
| `datasets.stl10.clean_acc_floor` | 0.75 | FW-6, SR-2 |
| `datasets.cifar10.role` | smoke_only | FW-6, SR-2 |
| `datasets.cifar10.image_size` | 32, 32, 3 | FW-6, SR-2 |
| `datasets.cifar10.n` | 3072 | FW-6, SR-2 |
| `datasets.cifar10.classes` | 10 | FW-6, SR-2 |
| `datasets.cifar10.train_images` | 45000 | FW-6, SR-2 |
| `datasets.cifar10.val_images` | 5000 | FW-6, SR-2 |
| `datasets.cifar10.test_images` | 10000 | FW-6, SR-2 |
| `datasets.cifar10.clean_acc_floor` | 0.93 | FW-6, SR-2 |

## bandwidth

| Parameter | Value | Cited by |
| --- | --- | --- |
| `bandwidth.symbol_type` | complex_baseband | - |
| `bandwidth.power_constraint` | unit_average_power | SR-4 |
| `bandwidth.ratios.r_1_2` | 1/2 | DEC-1, ER-3, G-8 |
| `bandwidth.ratios.r_1_3` | 1/3 | DEC-1, ER-3, G-8 |
| `bandwidth.ratios.r_1_6` | 1/6 | DEC-1, ER-3, G-8 |
| `bandwidth.ratios.r_1_12` | 1/12 | DEC-1, ER-3, G-8 |
| `bandwidth.ratios.r_1_24` | 1/24 | DEC-1, ER-3, G-8 |
| `bandwidth.core_ratio` | r_1_3 | ER-1, ER-3 |
| `bandwidth.core_ratio_status` | provisional_until_G-8 | - |
| `bandwidth.low_ratio_operating_point` | r_1_12 | ER-3 |
| `bandwidth.k_symbols.imagenette160.r_1_2` | 38400 | SR-3 |
| `bandwidth.k_symbols.imagenette160.r_1_3` | 25600 | SR-3 |
| `bandwidth.k_symbols.imagenette160.r_1_6` | 12800 | SR-3 |
| `bandwidth.k_symbols.imagenette160.r_1_12` | 6400 | SR-3 |
| `bandwidth.k_symbols.imagenette160.r_1_24` | 3200 | SR-3 |
| `bandwidth.k_symbols.stl10.r_1_2` | 13824 | SR-3 |
| `bandwidth.k_symbols.stl10.r_1_3` | 9216 | SR-3 |
| `bandwidth.k_symbols.stl10.r_1_6` | 4608 | SR-3 |
| `bandwidth.k_symbols.stl10.r_1_12` | 2304 | SR-3 |
| `bandwidth.k_symbols.stl10.r_1_24` | 1152 | SR-3 |
| `bandwidth.k_symbols.cifar10.r_1_2` | 1536 | SR-3 |
| `bandwidth.k_symbols.cifar10.r_1_3` | 1024 | SR-3 |
| `bandwidth.k_symbols.cifar10.r_1_6` | 512 | SR-3 |
| `bandwidth.k_symbols.cifar10.r_1_12` | 256 | SR-3 |
| `bandwidth.k_symbols.cifar10.r_1_24` | 128 | SR-3 |

## channel

| Parameter | Value | Cited by |
| --- | --- | --- |
| `channel.snr_definition` | Es/N0 in dB per complex channel use, measured after unit-average-power normalisation | SR-7 |
| `channel.snr_conversion` | Es/N0_dB = Eb/N0_dB + 10*log10(bits_per_symbol * code_rate); every published reference curve MUST be converted with this identity before comparison | BR-2 |
| `channel.models_supported` | awgn | SR-5 |
| `channel.models_planned` | rayleigh_block, rayleigh_fast | SR-5 |
| `channel.train_snr_db_fixed` | 7 | DEC-11, ER-2 |
| `channel.train_snr_db_set` | 1, 4, 7, 13, 19 | OPT-4 |
| `channel.test_snr_grid_db` | -8, -6, -4, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 18 | DR-1, ER-1 |
| `channel.grid_rationale` | the LDPC waterfall for QPSK spans roughly -1 dB (rate 1/3) to 5 dB (rate 5/6), so density is spent there; the grid then extends to 18 dB because 16-QAM at rate 5/6 does not decode until roughly 11-12 dB, and truncating earlier would engineer a crossover under DEC-16 and then fail to measure it | - |

## learned_system

| Parameter | Value | Cited by |
| --- | --- | --- |
| `learned_system.framework` | pytorch | - |
| `learned_system.encoder` | conv_downsample_to_k_symbols | - |
| `learned_system.decoder_heads` | reconstruction, classification | SR-8 |
| `learned_system.loss` | CE + lambda * MSE | SR-8 |
| `learned_system.lambda_core` | 1.0 | SR-9 |
| `learned_system.lambda_status` | provisional_until_G-4 | - |
| `learned_system.lambda_calibration_gate` | G-4 | - |
| `learned_system.lambda_acc_tolerance_pp` | 1.0 | SR-9 |
| `learned_system.lambda_psnr_floor_db` | 20 | SR-9 |
| `learned_system.lambda_psnr_floor_relaxed_db` | 16 | DEC-2, SR-9 |
| `learned_system.train_snr_protocol` | one_model_per_ratio_at_fixed_snr | - |
| `learned_system.optimizer` | adam | - |
| `learned_system.lr` | 0.001 | - |
| `learned_system.lr_schedule` | cosine | - |
| `learned_system.amp` | true | - |
| `learned_system.grad_accumulation_allowed` | true | SR-11 |
| `learned_system.max_params_millions` | 10 | SR-14 |
| `learned_system.batch_size.imagenette160` | 32 | SR-11 |
| `learned_system.batch_size.stl10` | 64 | SR-11 |
| `learned_system.batch_size.cifar10` | 128 | SR-11 |
| `learned_system.batch_size_policy` | target_not_binding | - |
| `learned_system.epochs.imagenette160` | 100 | - |
| `learned_system.epochs.stl10` | 200 | - |
| `learned_system.epochs.cifar10` | 150 | - |
| `learned_system.papr_report_required` | true | SR-16 |
| `learned_system.peak_power_constraint_available` | true | SR-16 |

## baseline

| Parameter | Value | Cited by |
| --- | --- | --- |
| `baseline.source_codec` | jpeg2000 | BR-1 |
| `baseline.source_codec_secondary` | jpeg | BR-1, DEC-9 |
| `baseline.j2k_rate_control` | exact_target_bytes | - |
| `baseline.jpeg_quality_grid` | 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95 | BR-4 |
| `baseline.container_policy` | all emitted container bytes count against payload_bits; shared-table or stripped-header variants MAY be reported as a labelled sensitivity, never as the headline | BR-11 |
| `baseline.channel_code` | 5g_nr_ldpc | BR-1, ER-9 |
| `baseline.ldpc_standard` | 3gpp_ts_38_212 | BR-2 |
| `baseline.ldpc_impl` | sionna | BR-2, BR-10, BR-14 |
| `baseline.ldpc_impl_version` | 2.0.1 | G-9 |
| `baseline.ldpc_impl_provides` | base_graph_selection, lifting_size_selection, encoding, rate_matching, decoding | BR-14 |
| `baseline.ldpc_impl_local` | tb_crc, code_block_segmentation, per_block_budget_distribution, concatenation, crc_failure_detection | BR-10, BR-14, DEC-10 |
| `baseline.ldpc_impl_fallback` | self_implemented_offset_min_sum | BR-14, DEC-10, G-2 |
| `baseline.ldpc_golden_vector_file` | tests/fixtures/ldpc_ts38212_golden.npz | BR-2 |
| `baseline.ldpc_golden_vector_source_gate` | G-9 | BR-2, G-9 |
| `baseline.ldpc_base_graph` | auto_per_ts_38212 | BR-10 |
| `baseline.ldpc_rates` | 1/3, 1/2, 2/3, 5/6 | BR-4, BR-15 |
| `baseline.ldpc_decoder` | offset_min_sum | DEC-10, G-9 |
| `baseline.ldpc_max_iters` | 50 | - |
| `baseline.tb_crc_bits` | 24 | BR-10 |
| `baseline.cb_crc_bits` | 24 | BR-10 |
| `baseline.code_block_max_bits` | 8448 | BR-10 |
| `baseline.rate_matching` | ts_38212_with_filler | BR-10 |
| `baseline.modulations` | bpsk, qpsk, qam16 | BR-2, BR-4, BR-9, BR-15, DEC-16 |
| `baseline.modulation_tuning` | adaptive_per_snr | BR-4, DEC-16 |
| `baseline.core_modulation` | qpsk | BR-9, ER-9 |
| `baseline.budget_rule` | usable_source_bytes = floor((floor(k * bits_per_symbol * rate) - tb_crc_bits - segmentation_and_filler_overhead) / 8); the complete compressed file, container bytes included, MUST fit within usable_source_bytes | BR-3, BR-10 |
| `baseline.outage_policy` | uniform_random_label | BR-13 |
| `baseline.tuning` | best_feasible_config_per_snr_on_validation_split | BR-4 |

## reference_classifier

| Parameter | Value | Cited by |
| --- | --- | --- |
| `reference_classifier.arch` | resnet18 | BR-8, BR-12, DEC-15, ER-4 |
| `reference_classifier.trained_on` | clean_images | BR-8, BR-12, DEC-15, ER-4 |
| `reference_classifier.pretrained_weights_permitted` | false | BR-8, BR-12, DEC-15, ER-4 |
| `reference_classifier.artifact_finetuned_variant_required` | true | BR-8, BR-12, DEC-15, ER-4 |
| `reference_classifier.frozen` | true | BR-8, BR-12, DEC-15, ER-4 |
| `reference_classifier.shared_by` | classical_baseline, digital_semantic_control, semantic_recon_ablation | BR-8, BR-12, DEC-15, ER-4 |

## evaluation

| Parameter | Value | Cited by |
| --- | --- | --- |
| `evaluation.train_seeds` | 0, 1, 2 | ER-1, ER-10 |
| `evaluation.channel_seeds` | 0, 1, 2 | ER-1, ER-10 |
| `evaluation.split_seed` | 1337 | - |
| `evaluation.split_rule` | val carved deterministically from the published train split by evaluation.split_seed; the test split is never used for any selection | SR-17 |
| `evaluation.ci` | paired_bootstrap_95 | ER-1, ER-10 |
| `evaluation.bootstrap_resamples` | 10000 | ER-10 |
| `evaluation.paired_test` | mcnemar_exact | ER-10 |
| `evaluation.hypothesis_rule` | three_consecutive_snr_points | - |
| `evaluation.cliff_drop_pp` | 30 | - |
| `evaluation.graceful_drop_pp` | 15 | - |
| `evaluation.test_subset_size` | 2000 | ER-3, ER-6 |
| `evaluation.full_test_split_required_for` | ER-1 | ER-6 |
| `evaluation.metrics` | top1_acc, psnr_db, ssim, bytes_sent, header_bytes, payload_bytes, decode_failure_rate, infeasible_rate, papr_db | - |

## compute

| Parameter | Value | Cited by |
| --- | --- | --- |
| `compute.primary_device` | rtx_4060_mobile_8gb | SR-11 |
| `compute.overflow` | colab_free, kaggle_free | - |
| `compute.vram_budget_gb` | 7.0 | SR-11 |
| `compute.profiling_gate` | G-7 | SR-11 |
| `compute.max_wall_clock_hours_per_run` | 4 | SR-11 |
| `compute.checkpoint_every_epochs` | 1 | SR-10 |

## artifacts

| Parameter | Value | Cited by |
| --- | --- | --- |
| `artifacts.results_dir` | results/ | SR-9, SR-13 |
| `artifacts.per_image_dir` | results/per_image/ | SR-13, SR-18 |
| `artifacts.checkpoint_dir` | checkpoints/ | SR-13 |
| `artifacts.figures_dir` | figures/ | SR-13 |
| `artifacts.csv_schema` | run_id, timestamp, git_commit, git_dirty, config_hash, checkpoint_id, system, dataset, split, n, k, bw_ratio, channel, train_snr_db, test_snr_db, train_seed, channel_seed, lambda, source_codec, jpeg_quality, j2k_target_bytes, ldpc_rate, modulation, top1_acc, n_correct, n_test, psnr_db, ssim, bytes_sent, header_bytes, payload_bytes, papr_db, decode_failure_rate, infeasible_rate, test_subset, wall_clock_s, peak_vram_gb | ER-5, FW-2, SR-13 |
| `artifacts.per_image_schema` | image_index, true_label, pred_label, correct, outage | SR-13, SR-18 |

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
| `hardware_tier23.status` | stretch_goal | - |
| `hardware_tier23.expected_demonstration` | pre_recorded | HR-4 |
| `hardware_tier23.needs` | iq_playback, iq_capture, tx_capable_pair, wired_loopback_with_attenuator | HR-1 |
| `hardware_tier23.min_sample_rate_msps` | 1 | HR-1 |
| `hardware_tier23.candidates` | adalm_pluto_x2, hackrf_one_plus_rtlsdr | HR-1 |
| `hardware_tier23.budget_inr_range` | 25000, 40000 | HR-1 |
| `hardware_tier23.budget_note` | two Plutos sit at or above the top of this range once import duty lands; the HackRF + RTL-SDR pairing is what the range actually buys | HR-1 |
| `hardware_tier23.edge_node` | raspberry_pi_4_or_5 | HR-4 |
| `hardware_tier23.live_demo_latency_budget_ms` | 500 | HR-4 |
| `hardware_tier23.purchase_gate` | G-5 | HR-2 |
| `hardware_tier23.framing` | preamble_correlation, rrc_pulse_shaping, timing_sync, cfo_estimation, pilot_aided_snr_measurement | HR-6 |

## deliverables

| Parameter | Value | Cited by |
| --- | --- | --- |
| `deliverables.review_weeks.first` | 4 | PR-2, PR-8 |
| `deliverables.review_weeks.second` | 10 | PR-2, PR-8 |
| `deliverables.review_weeks.third` | 16 | PR-2, PR-8 |
| `deliverables.literature_review_min_refs` | 25 | PR-1 |
| `deliverables.time_plan_artifact` | gantt_chart | PR-2 |
| `deliverables.poster_format` | a0 | PR-4 |
| `deliverables.plagiarism_report_required` | true | PR-5 |
| `deliverables.report_format_source` | vault/capstone/CAPSTONE_THESIS_Format.docx | PR-6 |
| `deliverables.standards` | 3gpp_ts_38_212, itu_t_t_800_jpeg2000, itu_t_t_81_jpeg, ieee_754, ietf_rfc_2119 | PR-3 |
| `deliverables.novelty_claims` | er9_attribution_decomposition, br11_format_overhead_controlled_baseline | PR-7 |
