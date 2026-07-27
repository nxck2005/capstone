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
| `datasets.imagenette160.role` | headline | ER-3, FW-6, SR-2 |
| `datasets.imagenette160.image_size` | 160, 160, 3 | ER-3, FW-6, SR-2 |
| `datasets.imagenette160.n` | 76800 | ER-3, FW-6, SR-2 |
| `datasets.imagenette160.classes` | 10 | ER-3, FW-6, SR-2 |
| `datasets.imagenette160.train_images` | 8469 | ER-3, FW-6, SR-2 |
| `datasets.imagenette160.val_images` | 1000 | ER-3, FW-6, SR-2 |
| `datasets.imagenette160.test_images` | 3925 | ER-3, FW-6, SR-2 |
| `datasets.imagenette160.clean_acc_floor` | 0.88 | ER-3, FW-6, SR-2 |
| `datasets.stl10.role` | fallback_headline | ER-3, FW-6, SR-2 |
| `datasets.stl10.image_size` | 96, 96, 3 | ER-3, FW-6, SR-2 |
| `datasets.stl10.n` | 27648 | ER-3, FW-6, SR-2 |
| `datasets.stl10.classes` | 10 | ER-3, FW-6, SR-2 |
| `datasets.stl10.train_images` | 4500 | ER-3, FW-6, SR-2 |
| `datasets.stl10.val_images` | 500 | ER-3, FW-6, SR-2 |
| `datasets.stl10.test_images` | 8000 | ER-3, FW-6, SR-2 |
| `datasets.stl10.clean_acc_floor` | 0.75 | ER-3, FW-6, SR-2 |
| `datasets.cifar10.role` | smoke_only | ER-3, FW-6, SR-2 |
| `datasets.cifar10.image_size` | 32, 32, 3 | ER-3, FW-6, SR-2 |
| `datasets.cifar10.n` | 3072 | ER-3, FW-6, SR-2 |
| `datasets.cifar10.classes` | 10 | ER-3, FW-6, SR-2 |
| `datasets.cifar10.train_images` | 45000 | ER-3, FW-6, SR-2 |
| `datasets.cifar10.val_images` | 5000 | ER-3, FW-6, SR-2 |
| `datasets.cifar10.test_images` | 10000 | ER-3, FW-6, SR-2 |
| `datasets.cifar10.clean_acc_floor` | 0.93 | ER-3, FW-6, SR-2 |

## preprocessing

| Parameter | Value | Cited by |
| --- | --- | --- |
| `preprocessing.canonical_image` | resize_shorter_side_then_crop_to_dataset_image_size | AM-28, SR-19 |
| `preprocessing.resize_interpolation` | bilinear | AM-28, SR-19 |
| `preprocessing.antialias` | true | AM-28, SR-19 |
| `preprocessing.train_crop` | random_resized_crop | AM-28, SR-19 |
| `preprocessing.eval_crop` | center_crop | AM-28, SR-19 |
| `preprocessing.eval_augmentation_permitted` | false | AM-28, SR-19 |
| `preprocessing.colour_space` | rgb | AM-28, SR-19 |
| `preprocessing.bit_depth` | 8 | AM-28, SR-19 |
| `preprocessing.tensor_range` | unit_interval | AM-28, SR-19 |
| `preprocessing.channel_normalisation` | inside_model_never_in_the_pipeline | AM-28, SR-19 |
| `preprocessing.codec_input` | canonical_8bit_pixels | AM-28, SR-19 |
| `preprocessing.reconstruction_clipped_before_metrics` | true | AM-28, SR-19 |
| `preprocessing.psnr_data_range` | 1.0 | AM-28, SR-19 |
| `preprocessing.ssim_impl` | skimage_structural_similarity | AM-28, SR-19 |
| `preprocessing.ssim_gaussian_weights` | true | AM-28, SR-19 |

## bandwidth

| Parameter | Value | Cited by |
| --- | --- | --- |
| `bandwidth.symbol_type` | complex_baseband | - |
| `bandwidth.power_constraint` | unit_average_power | SR-4 |
| `bandwidth.ratios.r_1_2` | 1/2 | AM-41, DEC-11, ER-3, G-8 |
| `bandwidth.ratios.r_1_3` | 1/3 | AM-41, DEC-11, ER-3, G-8 |
| `bandwidth.ratios.r_1_6` | 1/6 | AM-41, DEC-11, ER-3, G-8 |
| `bandwidth.ratios.r_1_12` | 1/12 | AM-41, DEC-11, ER-3, G-8 |
| `bandwidth.ratios.r_1_24` | 1/24 | AM-41, DEC-11, ER-3, G-8 |
| `bandwidth.ratios.r_1_48` | 1/48 | AM-41, DEC-11, ER-3, G-8 |
| `bandwidth.crossover_ratio` | r_1_3 | AM-26, DEC-11, ER-1, ER-3 |
| `bandwidth.crossover_ratio_status` | provisional_until_G-8 | - |
| `bandwidth.efficiency_ratio` | r_1_6 | AM-26, DEC-11, ER-3, ER-11 |
| `bandwidth.efficiency_ratio_status` | provisional_until_G-8 | - |
| `bandwidth.efficiency_ratio_threshold_pp` | 5 | ER-3 |
| `bandwidth.crossover_ratio_threshold_pp` | 2 | ER-3 |
| `bandwidth.crossover_ratio_unsatisfiable_fallback` | efficiency_ratio | AM-41, ER-3 |
| `bandwidth.ladder_bottom_saturation_rule` | extend_downward_and_resweep | ER-3, G-8 |
| `bandwidth.headline_ratio` | crossover_ratio | AM-26, AM-41, BR-1, ER-1, ER-2, ER-3, ER-11, ER-12, G-8, PR-9, SR-16 |
| `bandwidth.low_ratio_operating_point` | r_1_12 | AM-12, AM-24, BR-10, DEC-11, ER-3, ER-11, G-8 |
| `bandwidth.low_ratio_operating_point_status` | provisional_until_G-8 | - |
| `bandwidth.low_ratio_rule` | smallest_ratio_at_least_two_rungs_below_headline | ER-3, G-8 |
| `bandwidth.k_symbols.imagenette160.r_1_2` | 38400 | SR-3 |
| `bandwidth.k_symbols.imagenette160.r_1_3` | 25600 | SR-3 |
| `bandwidth.k_symbols.imagenette160.r_1_6` | 12800 | SR-3 |
| `bandwidth.k_symbols.imagenette160.r_1_12` | 6400 | SR-3 |
| `bandwidth.k_symbols.imagenette160.r_1_24` | 3200 | SR-3 |
| `bandwidth.k_symbols.imagenette160.r_1_48` | 1600 | SR-3 |
| `bandwidth.k_symbols.stl10.r_1_2` | 13824 | SR-3 |
| `bandwidth.k_symbols.stl10.r_1_3` | 9216 | SR-3 |
| `bandwidth.k_symbols.stl10.r_1_6` | 4608 | SR-3 |
| `bandwidth.k_symbols.stl10.r_1_12` | 2304 | SR-3 |
| `bandwidth.k_symbols.stl10.r_1_24` | 1152 | SR-3 |
| `bandwidth.k_symbols.stl10.r_1_48` | 576 | SR-3 |
| `bandwidth.k_symbols.cifar10.r_1_2` | 1536 | SR-3 |
| `bandwidth.k_symbols.cifar10.r_1_3` | 1024 | SR-3 |
| `bandwidth.k_symbols.cifar10.r_1_6` | 512 | SR-3 |
| `bandwidth.k_symbols.cifar10.r_1_12` | 256 | SR-3 |
| `bandwidth.k_symbols.cifar10.r_1_24` | 128 | SR-3 |
| `bandwidth.k_symbols.cifar10.r_1_48` | 64 | SR-3 |

## channel

| Parameter | Value | Cited by |
| --- | --- | --- |
| `channel.snr_definition` | Es/N0 in dB per complex channel use, measured after unit-average-power normalisation | SR-7 |
| `channel.snr_conversion` | Es/N0_dB = Eb/N0_dB + 10*log10(bits_per_symbol * code_rate); every published reference curve MUST be converted with this identity before comparison | BR-2 |
| `channel.models_supported` | awgn | SR-5 |
| `channel.models_planned` | rayleigh_block, rayleigh_fast | SR-5 |
| `channel.train_snr_db_fixed` | 7 | AM-3, DEC-11, ER-2 |
| `channel.train_snr_db_set` | 1, 4, 7, 13, 19 | AM-45, ER-2 |
| `channel.test_snr_grid_db` | -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 18 | AM-52, BR-16, DR-1, ER-1, ER-11, G-9 |
| `channel.grid_rationale` | three waterfalls have to be resolved, not one. BPSK at rate 1/3 -- which BR-4 selects at the noisy end and which AM-15 relies on for roughly 3 dB of extra reach -- decodes around Es/N0 = -4 to -5 dB, measured at W0 as BER 0.0 at -4 dB and 0.31 at -8 dB, so the region from -8 to -2 carries 1 dB spacing: it is where the classical cliff H2 measures actually falls, and 2 dB spacing there could smear or miss it entirely. The QPSK waterfall spans roughly -1 dB (rate 1/3) to 5 dB (rate 5/6), so density is spent there too. The grid then extends to 18 dB because 16-QAM at rate 5/6 does not decode until roughly 11-12 dB, and truncating earlier would engineer a crossover under DEC-16 and then fail to measure it | - |

## learned_system

| Parameter | Value | Cited by |
| --- | --- | --- |
| `learned_system.framework` | pytorch | AM-27, AM-48 |
| `learned_system.encoder` | conv_downsample_to_k_symbols | AM-27, AM-48 |
| `learned_system.encoder_arch` | djscc_residual_v1 | AM-27, AM-48 |
| `learned_system.encoder_downsample_factor` | 4 | AM-27, AM-48 |
| `learned_system.encoder_stem_channels` | 64 | AM-27, AM-48 |
| `learned_system.encoder_body_channels` | 128 | AM-27, AM-48 |
| `learned_system.encoder_residual_blocks` | 2 | AM-27, AM-48 |
| `learned_system.encoder_activation` | prelu | AM-27, AM-48 |
| `learned_system.encoder_norm` | groupnorm | AM-27, AM-48 |
| `learned_system.encoder_complex_packing` | channel_pairs_to_real_imag | AM-27, AM-48 |
| `learned_system.encoder_output_complex_channels.r_1_2` | 24 | AM-27, AM-48 |
| `learned_system.encoder_output_complex_channels.r_1_3` | 16 | AM-27, AM-48 |
| `learned_system.encoder_output_complex_channels.r_1_6` | 8 | AM-27, AM-48 |
| `learned_system.encoder_output_complex_channels.r_1_12` | 4 | AM-27, AM-48 |
| `learned_system.encoder_output_complex_channels.r_1_24` | 2 | AM-27, AM-48 |
| `learned_system.encoder_output_complex_channels.r_1_48` | 1 | AM-27, AM-48 |
| `learned_system.decoder_arch` | mirror_of_encoder_with_two_heads | AM-27, AM-48 |
| `learned_system.decoder_upsample` | transposed_conv | AM-27, AM-48 |
| `learned_system.arch_fallback` | width_halved_djscc_residual_v1 | AM-27, AM-48 |
| `learned_system.arch_freeze_gate` | G-7 | AM-27, AM-48 |
| `learned_system.decoder_heads` | reconstruction, classification | AM-27, AM-48, SR-8 |
| `learned_system.loss` | CE + lambda * MSE | AM-27, AM-48, SR-8 |
| `learned_system.lambda_core` | 1.0 | AM-27, AM-48, SR-9 |
| `learned_system.lambda_status` | provisional_until_G-4 | AM-27, AM-48 |
| `learned_system.lambda_calibration_gate` | G-4 | AM-27, AM-48 |
| `learned_system.lambda_grid` | 0.0, 0.1, 0.3, 1.0, 3.0 | AM-7, AM-27, AM-48, G-4, SR-9 |
| `learned_system.lambda_acc_tolerance_pp` | 1.0 | AM-27, AM-48, SR-9 |
| `learned_system.lambda_psnr_floor_db` | 20 | AM-27, AM-48, SR-9 |
| `learned_system.lambda_psnr_floor_relaxed_db` | 16 | AM-27, AM-48, DEC-2, SR-9 |
| `learned_system.lambda_calibration_snr_db` | 7 | AM-27, AM-35, AM-48, SR-9 |
| `learned_system.lambda_psnr_eval_snr_db` | 15 | AM-27, AM-48, SR-9 |
| `learned_system.lambda_calibration_ratio` | headline_ratio | AM-27, AM-48, SR-9 |
| `learned_system.train_snr_protocol` | one_model_per_ratio_at_fixed_snr | AM-27, AM-48 |
| `learned_system.optimizer` | adam | AM-27, AM-48 |
| `learned_system.lr` | 0.001 | AM-27, AM-48 |
| `learned_system.lr_schedule` | cosine | AM-27, AM-48 |
| `learned_system.augmentation` | random_resized_crop, horizontal_flip | AM-27, AM-48 |
| `learned_system.amp` | true | AM-27, AM-48 |
| `learned_system.grad_accumulation_allowed` | true | AM-27, AM-48, SR-11 |
| `learned_system.max_params_millions` | 10 | AM-27, AM-48, SR-14 |
| `learned_system.max_params_policy` | not_to_exceed_reference_classifier | AM-27, AM-36, AM-48, SR-14 |
| `learned_system.batch_size.imagenette160` | 32 | AM-27, AM-48, SR-11 |
| `learned_system.batch_size.stl10` | 64 | AM-27, AM-48, SR-11 |
| `learned_system.batch_size.cifar10` | 128 | AM-27, AM-48, SR-11 |
| `learned_system.batch_size_policy` | target_not_binding | AM-27, AM-48 |
| `learned_system.epochs.imagenette160` | 100 | AM-27, AM-48 |
| `learned_system.epochs.stl10` | 200 | AM-27, AM-48 |
| `learned_system.epochs.cifar10` | 150 | AM-27, AM-48 |
| `learned_system.papr_report_required` | true | AM-27, AM-48, SR-16 |
| `learned_system.peak_power_constraint_available` | true | AM-27, AM-48, SR-16 |
| `learned_system.papr_constrained_variant_required` | true | AM-27, AM-44, AM-48, SR-16 |
| `learned_system.papr_constrained_variant_seeds` | 1 | AM-27, AM-48, SR-16 |

## baseline

| Parameter | Value | Cited by |
| --- | --- | --- |
| `baseline.source_codec` | jpeg2000 | AM-51, BR-1 |
| `baseline.source_codec_secondary` | jpeg | AM-51, BR-1, DEC-9 |
| `baseline.j2k_rate_control` | largest_codestream_within_budget | AM-51, BR-1 |
| `baseline.j2k_rate_control_method` | cached_search_over_compression_ratio | AM-51, BR-1 |
| `baseline.j2k_emitted_size_authoritative` | true | AM-51, BR-1 |
| `baseline.j2k_container` | raw_codestream | AM-51, BR-1 |
| `baseline.j2k_impl` | openjpeg | AM-51, BR-1 |
| `baseline.j2k_impl_version` | 2.5.4 | AM-51, BR-1 |
| `baseline.j2k_binding` | glymur | AM-51, BR-1 |
| `baseline.j2k_wavelet` | irreversible_9_7 | AM-51, BR-1 |
| `baseline.j2k_progression_order` | RPCL | AM-51, BR-1 |
| `baseline.j2k_resolutions` | 6 | AM-51, BR-1 |
| `baseline.j2k_code_block_size` | 64, 64 | AM-51, BR-1 |
| `baseline.j2k_tile_size` | whole_image | AM-51, BR-1 |
| `baseline.j2k_search_method` | bisection_on_compression_ratio | AM-51, BR-1 |
| `baseline.j2k_search_bounds` | 1.0, 4000.0 | AM-51, BR-1 |
| `baseline.j2k_search_tolerance_bytes` | 16 | AM-51, BR-1 |
| `baseline.j2k_search_max_iters` | 24 | AM-51, BR-1 |
| `baseline.j2k_cache_key` | image_hash, budget_bytes | AM-51, AM-54, BR-1, BR-4 |
| `baseline.j2k_nonmonotone_policy` | keep_largest_codestream_at_or_below_budget | AM-51, BR-1 |
| `baseline.downsample_axis_px` | 160, 128, 96, 64 | AM-51, BR-1 |
| `baseline.downsample_selection` | best_feasible_per_snr_on_validation_split | AM-51, BR-1 |
| `baseline.jpeg_impl` | pillow | AM-51, BR-1 |
| `baseline.jpeg_chroma_subsampling` | 4:2:0 | AM-51, BR-1 |
| `baseline.jpeg_optimise_huffman` | true | AM-51, BR-1 |
| `baseline.jpeg_quality_grid` | 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95 | AM-51, BR-4 |
| `baseline.bler_characterisation_trials` | 5000 | AM-51, BR-4 |
| `baseline.fixed_mcs_design_snr_db` | 7 | AM-51, AM-53, BR-16 |
| `baseline.container_policy` | all emitted container bytes count against payload_bits; shared-table or stripped-header variants MAY be reported as a labelled sensitivity, never as the headline | AM-51, BR-11 |
| `baseline.channel_code` | 5g_nr_ldpc | AM-51, BR-1, ER-9, ER-12 |
| `baseline.ldpc_standard` | 3gpp_ts_38_212 | AM-51, BR-2, G-9 |
| `baseline.ldpc_standard_release` | rel_17 | AM-51, G-9 |
| `baseline.ldpc_standard_version` | 17.13.0 | AM-51 |
| `baseline.ldpc_standard_version_date` | 2026-02 | AM-51 |
| `baseline.ldpc_standard_version_pin_gate` | G-9 | AM-51, G-9 |
| `baseline.ldpc_impl` | sionna | AM-49, AM-51, BR-2, BR-10, BR-14 |
| `baseline.ldpc_impl_version` | 2.0.1 | AM-51, G-9 |
| `baseline.ldpc_impl_provides` | base_graph_selection, lifting_size_selection, encoding, rate_matching, decoding | AM-51, BR-14 |
| `baseline.ldpc_impl_local` | tb_crc, code_block_segmentation, per_block_budget_distribution, concatenation, crc_failure_detection | AM-51, BR-10, BR-14, DEC-10 |
| `baseline.ldpc_impl_fallback` | self_implemented_offset_min_sum | AM-51, BR-14, DEC-10, G-2 |
| `baseline.ldpc_golden_vector_file` | tests/fixtures/ldpc_ts38212_golden.npz | AM-51, BR-2, DEC-10 |
| `baseline.ldpc_golden_vector_source_gate` | G-9 | AM-51, BR-2, G-9 |
| `baseline.ldpc_golden_vector_source_ladder` | matlab_5g_toolbox, srsran_project_release_testvectors, aff3ct_or_oai_one_shot_build, hand_verified_small_case | AM-51, BR-2, DEC-10 |
| `baseline.ldpc_golden_vector_licence_check_required` | true | AM-51, BR-2 |
| `baseline.ldpc_golden_vector_source_rung` | 2 | AM-51, BR-2, G-9 |
| `baseline.ldpc_golden_vector_vendored` | false | AM-51, BR-2, DEC-10 |
| `baseline.ldpc_golden_vector_upstream_release` | release_25_10 | AM-51, BR-2 |
| `baseline.ldpc_golden_vector_upstream_asset` | phy_testvectors.tar | AM-51, BR-2 |
| `baseline.ldpc_golden_vector_upstream_url` | https://github.com/srsran/srsRAN_Project/releases/download/release_25_10/phy_testvectors.tar | AM-30, AM-51, BR-2 |
| `baseline.ldpc_golden_vector_upstream_successor` | https://gitlab.com/ocudu/ocudu | AM-51 |
| `baseline.ldpc_golden_vector_sha256.ldpc_encoder_test_data.tar.gz` | cb92fe900682632a50959cbc5b164e873f733a4c911108f374e37dda3606143d | AM-51, BR-2 |
| `baseline.ldpc_golden_vector_sha256.ldpc_rate_matcher_test_data.tar.gz` | fc5e333bd94a836c4304dcdda82eac583554685aed37f0bed3758f9e573719a4 | AM-51, BR-2 |
| `baseline.ldpc_golden_vector_sha256.ldpc_segmenter_test_data.tar.gz` | f53c6ab5baac521def745e8ca591197a8dbf8b06d2d60cc3b4335a36752d6fe1 | AM-51, BR-2 |
| `baseline.ldpc_golden_vector_offline_floor` | hand_verified_small_case | AM-30, AM-51, BR-2 |
| `baseline.ldpc_base_graph` | auto_per_ts_38212 | AM-51, BR-10 |
| `baseline.ldpc_bg1_min_coderate` | 0.3333333333333333 | AM-51, BR-10 |
| `baseline.ldpc_rates` | 1/3, 1/2, 2/3, 5/6 | AM-24, AM-51, BR-4, BR-9, BR-10, BR-15 |
| `baseline.ldpc_decoder` | offset_min_sum | AM-51, BR-14, DEC-10, G-9 |
| `baseline.ldpc_decoder_impl_spelling` | offset-minsum | AM-51, BR-14 |
| `baseline.ldpc_llr_convention` | log_p1_over_p0 | AM-51, BR-14 |
| `baseline.ldpc_max_iters` | 50 | AM-24, AM-51 |
| `baseline.tb_crc.threshold_payload_bits` | 3824 | AM-49, AM-51, BR-5, BR-10 |
| `baseline.tb_crc.small_bits` | 16 | AM-49, AM-51, BR-5, BR-10 |
| `baseline.tb_crc.small_polynomial` | crc16 | AM-49, AM-51, BR-5, BR-10 |
| `baseline.tb_crc.large_bits` | 24 | AM-49, AM-51, BR-5, BR-10 |
| `baseline.tb_crc.large_polynomial` | crc24a | AM-49, AM-51, BR-5, BR-10 |
| `baseline.cb_crc_bits` | 24 | AM-51, BR-10 |
| `baseline.code_block_max_bits.bg1` | 8448 | AM-49, AM-51, BR-10 |
| `baseline.code_block_max_bits.bg2` | 3840 | AM-49, AM-51, BR-10 |
| `baseline.base_graph_selection_rule` | ts_38212_7_2_2_from_transport_block_A_and_R | AM-49, AM-51, BR-10 |
| `baseline.base_graph_pinned_at_seam` | true | AM-51, BR-10, BR-14 |
| `baseline.bg2_min_coderate` | 0.2 | AM-51, BR-10 |
| `baseline.payload_solver` | largest_byte_aligned_A_whose_full_packetisation_fits_G | AM-51 |
| `baseline.rate_matching` | ts_38212_with_filler | AM-51, BR-10 |
| `baseline.modulations` | bpsk, qpsk, qam16 | AM-24, AM-51, AM-52, BR-2, BR-4, BR-9, BR-10, BR-15, DEC-16 |
| `baseline.modulation_tuning` | adaptive_per_snr | AM-51, BR-4, DEC-16 |
| `baseline.core_modulation` | qpsk | AM-15, AM-51, BR-9 |
| `baseline.budget_rule` | solved, not evaluated in closed form: choose the largest byte-aligned transport-block payload A whose complete TS 38.212 packetisation — conditional TB CRC per baseline.tb_crc, base-graph selection from (A, R), segmentation at the selected graph's code_block_max_bits, per-block CRCs, filler and rate matching — fits exactly within G = k * bits_per_symbol coded bits and satisfies every library constraint. The complete compressed file, container bytes included, MUST fit within A/8 bytes | AM-49, AM-51, BR-3, BR-10, ER-9 |
| `baseline.control_plane_policy` | out_of_band_excluded_for_all_systems | AM-51, BR-10 |
| `baseline.outage_policy` | uniform_random_label | AM-51, BR-13 |
| `baseline.tuning` | best_feasible_config_per_snr_on_validation_split | AM-51, BR-4 |

## reference_classifier

| Parameter | Value | Cited by |
| --- | --- | --- |
| `reference_classifier.arch` | resnet18 | AM-27, AM-36, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.trained_on` | clean_images | AM-27, AM-36, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.pretrained_weights_permitted` | false | AM-27, AM-36, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.optimizer` | sgd_momentum | AM-27, AM-36, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.lr` | 0.1 | AM-27, AM-36, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.momentum` | 0.9 | AM-27, AM-36, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.weight_decay` | 0.0005 | AM-27, AM-36, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.lr_schedule` | cosine | AM-27, AM-36, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.lr_warmup_epochs` | 5 | AM-27, AM-36, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.epochs` | 100 | AM-27, AM-36, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.batch_size` | 128 | AM-27, AM-36, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.label_smoothing` | 0.1 | AM-27, AM-36, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.augmentation` | random_resized_crop, horizontal_flip | AM-27, AM-36, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetuned_variant_required` | true | AM-27, AM-36, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_gate` | G-8 | AM-6, AM-27, AM-36, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_corpus` | union_of_br4_selected_qualities_at_or_below_train_snr | AM-27, AM-36, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.br4_selection_passes` | 2 | AM-27, AM-36, AM-54, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.br4_selection_terminates_after_pass` | 2 | AM-27, AM-36, BR-4, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.headline_scorer` | artifact_finetuned | AM-27, AM-36, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.frozen` | true | AM-27, AM-36, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.shared_by` | classical_baseline, semantic_recon_ablation | AM-5, AM-27, AM-36, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |

## digital_semantic_control

| Parameter | Value | Cited by |
| --- | --- | --- |
| `digital_semantic_control.role` | attribution_control_for_h4 | - |
| `digital_semantic_control.shared_with_learned` | encoder_arch, encoder_trunk, task_head_arch, train_split, augmentation, optimizer, epochs | AM-5, AM-27, AM-55, ER-9 |
| `digital_semantic_control.shared_with_learned_conceptual` | encoder_trunk, task_head_arch, train_split | - |
| `digital_semantic_control.differs_only_in` | channel_interface | ER-9 |
| `digital_semantic_control.channel_interface_comprises` | transmit_dim, quantiser, entropy_coder | ER-9 |
| `digital_semantic_control.transmit_layer` | encoder_output | ER-9 |
| `digital_semantic_control.transmit_dim_grid` | 64, 128, 256, 512, 1024, 2048, 4096, 8192 | ER-9 |
| `digital_semantic_control.transmit_dim_units` | real_values | ER-9 |
| `digital_semantic_control.transmit_dim_realised_by` | output_channel_count_and_adaptive_pooling | AM-56, ER-9 |
| `digital_semantic_control.width_selection` | joint_with_quantiser_bits_on_validation_split | ER-9 |
| `digital_semantic_control.selection_search` | two_stage_coarse_width_then_bits | ER-9 |
| `digital_semantic_control.selection_search_is_cross_product` | false | ER-9 |
| `digital_semantic_control.quantiser` | uniform_scalar | ER-9 |
| `digital_semantic_control.quantiser_bits_grid` | 2, 4, 6, 8 | ER-9 |
| `digital_semantic_control.quantiser_training` | straight_through_estimator | - |
| `digital_semantic_control.entropy_coder` | static_range_coder | ER-9 |
| `digital_semantic_control.entropy_model` | fitted_offline_on_train_split | - |
| `digital_semantic_control.entropy_model_learned_permitted` | false | - |
| `digital_semantic_control.entropy_table_bytes_counted` | false | ER-9 |
| `digital_semantic_control.transport_tuning` | best_feasible_config_per_snr_on_validation_split | BR-9, ER-9 |
| `digital_semantic_control.scored_by` | own_task_head | ER-9 |

## evaluation

| Parameter | Value | Cited by |
| --- | --- | --- |
| `evaluation.train_seeds` | 0, 1, 2 | AM-31, BR-1, ER-1, ER-2, ER-10, ER-11, ER-12 |
| `evaluation.channel_seeds` | 0, 1, 2 | AM-31, ER-1, ER-10 |
| `evaluation.seed_pairing` | zipped_not_cross_product | AM-17, AM-31, ER-1, ER-10 |
| `evaluation.seed_cell_interpretation` | compound_replicate | AM-31, ER-10 |
| `evaluation.split_seed` | 1337 | - |
| `evaluation.split_rule` | val carved deterministically from the published train split by evaluation.split_seed; the test split is never used for any selection | SR-17 |
| `evaluation.ci` | paired_bootstrap_95 | AM-3, ER-1, ER-10 |
| `evaluation.bootstrap_resamples` | 10000 | ER-10 |
| `evaluation.paired_test` | mcnemar_exact | ER-10 |
| `evaluation.hypothesis_rule` | three_consecutive_snr_points | - |
| `evaluation.h1_effect_size` | mean_paired_difference_at_or_below_train_snr | ER-10 |
| `evaluation.h1_run_calibration` | sign_flip_permutation_over_image_trajectories | AM-32, ER-10 |
| `evaluation.h1_run_permutation_resamples` | 10000 | ER-10 |
| `evaluation.gap_trend_test` | wls_slope_of_paired_gap_vs_snr | ER-10 |
| `evaluation.gap_trend_bootstrap_unit` | per_image_seed_trajectory | ER-10 |
| `evaluation.h3_requires_positive_low_snr_gap` | true | AM-39 |
| `evaluation.cliff_window_db` | 4 | AM-1 |
| `evaluation.cliff_window_selection` | largest_fixed_mcs_drop_on_validation_split | AM-56 |
| `evaluation.cliff_drop_pp` | 30 | AM-53, BR-16 |
| `evaluation.graceful_drop_pp` | 15 | - |
| `evaluation.cliff_reference_system` | classical_fixed_mcs | AM-53, AM-56, BR-16 |
| `evaluation.h2_test` | paired_difference_in_differences | - |
| `evaluation.h2_thresholds_apply_to` | point_estimates | - |
| `evaluation.h2_threshold_check_gate` | G-8 | - |
| `evaluation.ber_match_statistic` | waterfall_displacement_at_bler_1e-2 | AM-50, G-2 |
| `evaluation.ber_match_tolerance_db` | 0.5 | G-2 |
| `evaluation.test_subset_size` | 2000 | AM-14, AM-42, ER-3, ER-6, ER-11 |
| `evaluation.full_test_split_required_for` | ER-1 | ER-6 |
| `evaluation.metrics` | top1_acc, psnr_db, ssim, bytes_sent, header_bytes, payload_bytes, decode_failure_rate, infeasible_rate, coverage_rate, acc_given_delivery, papr_db | AM-38 |

## compute

| Parameter | Value | Cited by |
| --- | --- | --- |
| `compute.primary_device` | rtx_4060_mobile_8gb | AM-24, SR-11 |
| `compute.overflow` | colab_free, kaggle_free | AM-24 |
| `compute.vram_budget_gb` | 7.0 | AM-24, SR-11 |
| `compute.profiling_gate` | G-7 | AM-24, SR-11 |
| `compute.max_wall_clock_hours_per_run` | 4 | AM-24, SR-11 |
| `compute.checkpoint_every_epochs` | 1 | AM-24, SR-10 |
| `compute.ldpc_decode_cb_per_s_measured` | 625.2 | AM-24, AM-29 |
| `compute.ldpc_decode_cb_per_s_observed_range` | 625.2, 663 | AM-24 |
| `compute.ldpc_decode_batch_measured` | 32 | AM-24 |
| `compute.ldpc_decode_measured_at_iters` | 50 | AM-24 |
| `compute.ldpc_decode_early_termination` | false | AM-24 |
| `compute.ldpc_decode_measured_gate` | G-9 | AM-24, G-9 |
| `compute.er1_projected_ldpc_decode_hours_one_ratio` | 2.07 | AM-24, AM-29 |
| `compute.er1_projected_ldpc_decode_hours_two_ratios` | 4.14 | AM-24 |
| `compute.er1_projected_ldpc_decode_scope` | LDPC decode only; excludes JPEG 2000 encode and decode, classifier forward passes, and soft demapping | AM-24 |
| `compute.er1_projected_total_hours_status` | pending_measurement_at_W3_W4 | AM-24 |

## artifacts

| Parameter | Value | Cited by |
| --- | --- | --- |
| `artifacts.results_dir` | results/ | SR-9, SR-13 |
| `artifacts.per_image_dir` | results/per_image/ | SR-13, SR-18 |
| `artifacts.checkpoint_dir` | checkpoints/ | SR-13 |
| `artifacts.figures_dir` | figures/ | SR-13 |
| `artifacts.run_id_key` | system, dataset, bw_ratio, test_snr_db, train_seed, channel_seed | AM-37, SR-13, SR-18 |
| `artifacts.system_values` | learned, learned_papr_constrained, learned_snr_randomised, classical_adaptive, classical_fixed_mcs, classical_fixed_mod, classical_jpeg_secondary, classical_finetune_scored, er9_digital, label_transmission_bound, semantic_recon_ablation | AM-38, SR-13 |
| `artifacts.csv_schema` | run_id, timestamp, git_commit, git_dirty, config_hash, checkpoint_id, system, dataset, split, n, k, bw_ratio, channel, train_snr_db, test_snr_db, train_seed, channel_seed, lambda, source_codec, jpeg_quality, j2k_target_bytes, ldpc_rate, modulation, top1_acc, n_correct, n_test, psnr_db, ssim, bytes_sent, header_bytes, payload_bytes, papr_db, decode_failure_rate, infeasible_rate, coverage_rate, acc_given_delivery, test_subset, wall_clock_s, peak_vram_gb, classifier_variant, quantiser_bits, transmit_dim, entropy_stream_bytes, entropy_table_bytes, side_information_bytes, tb_crc_type, base_graph, lifting_size, num_codeblocks, filler_bits, effective_code_rate, model_param_count | AM-38, ER-5, FW-2, SR-13 |
| `artifacts.per_image_schema` | image_index, true_label, pred_label, correct, outage, outage_reason, source_bytes | AM-37, SR-13, SR-18 |

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
| `deliverables.review_dates_status` | pending_confirmation_from_2026_circular | - |
| `deliverables.objectives_stated_as` | completion_terms_not_outcomes | AM-46, PR-8 |
| `deliverables.objectives_modification_point` | second_review | PR-8 |
| `deliverables.literature_review_min_refs` | 25 | PR-1 |
| `deliverables.time_plan_artifact` | gantt_chart | PR-2 |
| `deliverables.poster_format` | a0 | PR-4 |
| `deliverables.plagiarism_report_required` | true | PR-5 |
| `deliverables.report_format_source` | vault/capstone/CAPSTONE_THESIS_Format.docx | PR-6 |
| `deliverables.standards` | 3gpp_ts_38_212, 3gpp_ts_38_211, itu_t_t_800_jpeg2000, itu_t_t_81_jpeg, ieee_754, ietf_rfc_2119 | AM-50, G-2, PR-3 |
| `deliverables.novelty_claims` | er9_attribution_decomposition, br11_format_overhead_controlled_baseline | PR-7 |
