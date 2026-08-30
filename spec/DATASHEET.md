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
| `datasets.archive_dir` | data/archives/ | AM-77, ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.extracted_dir` | data/datasets/ | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.manifest_dir` | data/manifests/ | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.manifest_format` | csv_stable_id_label_split | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.manifest_encoding` | utf-8 | AM-77, ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.manifest_newline` | lf | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.manifest_columns` | stable_sample_id, label, split | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.manifest_splits` | train, val, test | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.manifest_sort` | globally_by_stable_sample_id | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.manifest_hash` | sha256_of_exact_committed_csv_bytes | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.manifest_duplicate_id_policy` | hard_failure_within_dataset | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.stable_sample_id_rule` | sha256_of_original_per_sample_source_bytes_truncated_16_hex | AM-71, ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.split_carve_stratified` | true | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.split_rng` | numpy_default_rng_pcg64 | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.split_draw` | choice_without_replacement_per_class_in_ascending_label_order | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.split_pre_shuffle_order` | sorted_by_stable_sample_id | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.class_index_source` | dataset_specific_authoritative_metadata | AM-77, ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.imagenette160.role` | headline | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.imagenette160.image_size` | 160, 160, 3 | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.imagenette160.n` | 76800 | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.imagenette160.classes` | 10 | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.imagenette160.train_images` | 8469 | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.imagenette160.val_images` | 1000 | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.imagenette160.test_images` | 3925 | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.imagenette160.clean_acc_floor` | 0.88 | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.imagenette160.loader` | torchvision_datasets_imagenette | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.imagenette160.loader_size_arg` | 160px | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.imagenette160.source_url` | https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.imagenette160.archive_filename` | imagenette2-160.tgz | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.imagenette160.archive_bytes` | 99003388 | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.imagenette160.archive_sha256` | 64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5 | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.imagenette160.manifest_filename` | imagenette160.csv | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.imagenette160.manifest_sha256` | 224309422f15bf89460559381aea4b00c4779c52d3652f7f679a213369f3f889 | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.imagenette160.source_payload` | exact_encoded_jpeg_file_bytes | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.imagenette160.class_index_source` | sorted_authoritative_class_directory_identifiers | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.imagenette160.licence` | apache_2_0 | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.imagenette160.published_val_becomes` | test | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.stl10.role` | fallback_headline | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.stl10.image_size` | 96, 96, 3 | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.stl10.n` | 27648 | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.stl10.classes` | 10 | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.stl10.train_images` | 4500 | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.stl10.val_images` | 500 | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.stl10.test_images` | 8000 | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.stl10.clean_acc_floor` | 0.75 | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.stl10.loader` | torchvision_datasets_stl10 | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.stl10.loader_size_arg` | None | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.stl10.source_url` | https://cs.stanford.edu/~acoates/stl10/stl10_binary.tar.gz | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.stl10.archive_filename` | stl10_binary.tar.gz | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.stl10.archive_bytes` | 2640397119 | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.stl10.archive_sha256` | f31fd99273a1acb8609c8db427cebb1de3f71de77758cdc0e22956e1289b9866 | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.stl10.manifest_filename` | stl10.csv | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.stl10.manifest_sha256` | 67936da779dc0010160b37b3b40001490304a5873eb978d261e3a57947387b47 | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.stl10.source_payload` | exact_per_image_record_from_train_X_or_test_X_before_axis_transpose_or_pil | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.stl10.class_index_source` | class_names_txt_line_order_corresponding_to_one_based_labels | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.stl10.licence` | research_use_see_source | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.stl10.published_val_becomes` | test | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.cifar10.role` | smoke_only | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.cifar10.image_size` | 32, 32, 3 | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.cifar10.n` | 3072 | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.cifar10.classes` | 10 | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.cifar10.train_images` | 45000 | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.cifar10.val_images` | 5000 | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.cifar10.test_images` | 10000 | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.cifar10.clean_acc_floor` | 0.93 | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.cifar10.loader` | torchvision_datasets_cifar10 | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.cifar10.loader_size_arg` | None | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.cifar10.source_url` | https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.cifar10.archive_filename` | cifar-10-python.tar.gz | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.cifar10.archive_bytes` | 170498071 | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.cifar10.archive_sha256` | 6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.cifar10.manifest_filename` | cifar10.csv | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.cifar10.manifest_sha256` | 09e9debf4743831ca61f17154a997e60becdd7046a585bdbd94b5db4bf12a537 | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.cifar10.source_payload` | exact_3072_byte_per_image_chw_batch_record_before_hwc_or_pil | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.cifar10.class_index_source` | batches_meta_label_names_order_corresponding_to_payload_labels | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.cifar10.licence` | mit | ER-3, FW-6, SR-2, SR-17, SR-20 |
| `datasets.cifar10.published_val_becomes` | test | ER-3, FW-6, SR-2, SR-17, SR-20 |

## preprocessing

| Parameter | Value | Cited by |
| --- | --- | --- |
| `preprocessing.canonical_image` | resize_shorter_side_then_crop_to_dataset_image_size | AM-28, SR-19 |
| `preprocessing.canonical_and_augmentation_are_separate_stages` | true | AM-28, SR-19 |
| `preprocessing.resize_interpolation` | bilinear | AM-28, SR-19 |
| `preprocessing.antialias` | true | AM-28, SR-19 |
| `preprocessing.train_crop` | random_resized_crop | AM-28, SR-19 |
| `preprocessing.train_crop_scale` | 0.35, 1.0 | AM-28, SR-19 |
| `preprocessing.train_crop_ratio` | 0.75, 1.3333333333333333 | AM-28, SR-19 |
| `preprocessing.train_hflip_p` | 0.5 | AM-28, SR-19 |
| `preprocessing.eval_crop` | center_crop | AM-28, SR-19 |
| `preprocessing.eval_augmentation_permitted` | false | AM-28, SR-19 |
| `preprocessing.colour_space` | rgb | AM-28, SR-19 |
| `preprocessing.bit_depth` | 8 | AM-28, SR-19 |
| `preprocessing.tensor_range` | unit_interval | AM-28, SR-19 |
| `preprocessing.channel_normalisation` | inside_model_never_in_the_pipeline | AM-28, SR-19 |
| `preprocessing.channel_mean` | 0.485, 0.456, 0.406 | AM-28, SR-19 |
| `preprocessing.channel_std` | 0.229, 0.224, 0.225 | AM-28, SR-19 |
| `preprocessing.codec_input` | canonical_8bit_pixels | AM-28, SR-19 |
| `preprocessing.codec_downsample_interpolation` | bilinear | AM-28, SR-19 |
| `preprocessing.codec_upsample_interpolation` | bicubic | AM-28, SR-19 |
| `preprocessing.codec_resize_preserves_aspect` | true | AM-28, SR-19 |
| `preprocessing.reconstruction_clipped_before_metrics` | true | AM-28, SR-19 |
| `preprocessing.psnr_data_range` | 1.0 | AM-28, SR-19 |
| `preprocessing.ssim_impl` | skimage_structural_similarity | AM-28, SR-19 |
| `preprocessing.ssim_channel_axis` | -1 | AM-28, SR-19 |
| `preprocessing.ssim_gaussian_weights` | true | AM-28, SR-19 |
| `preprocessing.ssim_sigma` | 1.5 | AM-28, SR-19 |
| `preprocessing.ssim_k1` | 0.01 | AM-28, SR-19 |
| `preprocessing.ssim_k2` | 0.03 | AM-28, SR-19 |
| `preprocessing.ssim_use_sample_covariance` | false | AM-28, SR-19 |
| `preprocessing.ssim_effective_gaussian_support_samples` | 11 | AM-28, SR-19 |

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
| `bandwidth.crossover_ratio` | r_1_6 | AM-26, AM-90, DEC-11, ER-1, ER-3 |
| `bandwidth.crossover_ratio_status` | selected_at_G-8 | - |
| `bandwidth.efficiency_ratio` | r_1_24 | AM-26, AM-90, DEC-11, ER-3, ER-11 |
| `bandwidth.efficiency_ratio_status` | selected_at_G-8 | - |
| `bandwidth.efficiency_ratio_threshold_pp` | 5 | ER-3 |
| `bandwidth.crossover_ratio_threshold_pp` | 2 | ER-3 |
| `bandwidth.crossover_ratio_unsatisfiable_fallback` | efficiency_ratio | AM-41, ER-3 |
| `bandwidth.ladder_bottom_saturation_rule` | extend_downward_and_resweep | ER-3, ER-9, G-8 |
| `bandwidth.headline_ratio` | crossover_ratio | AM-26, AM-41, AM-90, BR-1, BR-16, ER-1, ER-2, ER-3, ER-11, ER-12, G-8, SR-16 |
| `bandwidth.low_ratio_operating_point` | r_1_24 | AM-12, AM-24, AM-90, BR-10, DEC-11, ER-3, ER-11, G-8 |
| `bandwidth.low_ratio_operating_point_status` | selected_at_G-8 | - |
| `bandwidth.low_ratio_rule` | exactly_two_ordered_ladder_rungs_below_headline | AM-59, ER-3, G-8 |
| `bandwidth.low_ratio_boundary_rule` | nearest_available_rung_below_headline_if_two_steps_unavailable | ER-3 |
| `bandwidth.ceiling_definition` | error_free_codec_validation_accuracy_per_br4 | ER-3 |
| `bandwidth.ceiling_rule` | lcb95_of_codec_ceiling_minus_clean_accuracy_at_least_negative_t | ER-3 |
| `bandwidth.upper_grid_saturation_check` | extend_upper_grid_if_adaptive_validation_baseline_still_rising_at_18db | ER-3 |
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
| `channel.train_snr_db_fixed` | 7 | AM-3, BR-12, DEC-11, ER-2 |
| `channel.train_snr_db_set` | 1, 4, 7, 13, 19 | AM-45, ER-2 |
| `channel.test_snr_grid_db` | -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 18 | AM-52, BR-16, DR-1, ER-1, ER-11, G-9 |
| `channel.grid_rationale` | three waterfalls have to be resolved, not one. BPSK at rate 1/3 -- which BR-4 selects at the noisy end and which AM-15 relies on for roughly 3 dB of extra reach -- decodes around Es/N0 = -4 to -5 dB, measured at W0 as BER 0.0 at -4 dB and 0.31 at -8 dB, so the region from -8 to -2 carries 1 dB spacing: it is where the classical cliff H2 measures actually falls, and 2 dB spacing there could smear or miss it entirely. The QPSK waterfall spans roughly -1 dB (rate 1/3) to 5 dB (rate 5/6), so density is spent there too. The grid then extends to 18 dB because 16-QAM at rate 5/6 does not decode until roughly 11-12 dB, and truncating earlier would engineer a crossover under DEC-16 and then fail to measure it | - |

## learned_system

| Parameter | Value | Cited by |
| --- | --- | --- |
| `learned_system.framework` | pytorch | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.encoder` | conv_downsample_to_k_symbols | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.encoder_arch` | djscc_residual_v1 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.encoder_downsample_factor` | 4 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.encoder_stem_channels` | 64 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.encoder_body_channels` | 128 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.encoder_residual_blocks` | 2 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.encoder_activation` | prelu | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.encoder_norm` | groupnorm | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.encoder_complex_packing` | channel_pairs_to_real_imag | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.encoder_output_complex_channels.r_1_2` | 24 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.encoder_output_complex_channels.r_1_3` | 16 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.encoder_output_complex_channels.r_1_6` | 8 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.encoder_output_complex_channels.r_1_12` | 4 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.encoder_output_complex_channels.r_1_24` | 2 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.encoder_output_complex_channels.r_1_48` | 1 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.decoder_arch` | mirror_of_encoder_with_two_heads | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.decoder_upsample` | transposed_conv | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.arch_fallback` | width_halved_djscc_residual_v1 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.arch_freeze_gate` | G-7 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.decoder_heads` | reconstruction, classification | AM-27, AM-48, AM-91, SR-8, SR-10 |
| `learned_system.loss` | CE + lambda * MSE | AM-27, AM-48, AM-91, SR-8, SR-10 |
| `learned_system.lambda_core` | 3.0 | AM-27, AM-48, AM-91, AM-92, SR-9, SR-10 |
| `learned_system.lambda_status` | selected_at_G-4 | AM-27, AM-48, AM-91, AM-92, SR-10 |
| `learned_system.lambda_calibration_gate` | G-4 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.lambda_grid` | 0.0, 0.1, 0.3, 1.0, 3.0 | AM-7, AM-27, AM-48, AM-91, G-4, SR-9, SR-10 |
| `learned_system.lambda_acc_tolerance_pp` | 1.0 | AM-27, AM-48, AM-91, SR-9, SR-10 |
| `learned_system.lambda_psnr_floor_db` | 20 | AM-27, AM-48, AM-91, SR-9, SR-10 |
| `learned_system.lambda_psnr_floor_relaxed_db` | 16 | AM-27, AM-48, AM-91, DEC-2, SR-9, SR-10 |
| `learned_system.lambda_calibration_snr_db` | 7 | AM-27, AM-35, AM-48, AM-91, SR-9, SR-10 |
| `learned_system.lambda_psnr_eval_snr_db` | 15 | AM-27, AM-48, AM-91, SR-9, SR-10 |
| `learned_system.lambda_calibration_ratio` | headline_ratio | AM-27, AM-48, AM-91, SR-9, SR-10 |
| `learned_system.train_snr_protocol` | one_model_per_ratio_at_fixed_snr | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.optimizer` | adam | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.optimizer_implementation` | torch.optim.Adam | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.adam_beta1` | 0.9 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.adam_beta2` | 0.999 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.adam_epsilon` | 1e-08 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.adam_weight_decay` | 0.0 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.adam_amsgrad` | false | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.adam_maximize` | false | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.adam_foreach` | false | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.adam_capturable` | false | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.adam_differentiable` | false | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.adam_fused` | false | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.lr` | 0.001 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.lr_schedule` | cosine | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.lr_schedule_equation` | lr(e) = lr_min + (lr - lr_min) * 0.5 * (1 + cos(pi * e / max(E - 1, 1))) for zero-based epoch e | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.lr_min` | 0.0 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.lr_warmup_epochs` | 0 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.scheduler_step_unit` | epoch_start | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.scheduler_epoch_indexing` | zero_based | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.scheduler_resume_state` | completed_epoch | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.augmentation` | random_resized_crop, horizontal_flip | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.batch_order` | keyed_philox_permutation_per_epoch | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.drop_last` | false | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.dataloader_workers` | 4 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.pin_memory` | true | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.amp` | true | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.amp_device_type` | cuda | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.amp_dtype` | float16 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.grad_scaler_enabled` | true | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.grad_scaler_init_scale` | 65536.0 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.grad_scaler_growth_factor` | 2.0 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.grad_scaler_backoff_factor` | 0.5 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.grad_scaler_growth_interval` | 2000 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.grad_accumulation_allowed` | true | AM-27, AM-48, AM-91, SR-10, SR-11 |
| `learned_system.max_params_millions` | 10 | AM-27, AM-48, AM-91, SR-10, SR-14 |
| `learned_system.max_params_policy` | not_to_exceed_reference_classifier | AM-27, AM-36, AM-48, AM-91, SR-10, SR-14 |
| `learned_system.batch_size.imagenette160` | 32 | AM-27, AM-48, AM-91, SR-10, SR-11 |
| `learned_system.batch_size.stl10` | 64 | AM-27, AM-48, AM-91, SR-10, SR-11 |
| `learned_system.batch_size.cifar10` | 128 | AM-27, AM-48, AM-91, SR-10, SR-11 |
| `learned_system.batch_size_policy` | effective_target_with_profile_bound_physical_microbatch | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.production_physical_batch_size.local_4060_cu130.imagenette160` | 32 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.production_physical_batch_size.local_4060_cu130.stl10` | 64 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.production_physical_batch_size.local_4060_cu130.cifar10` | 128 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.production_accumulation_factor.local_4060_cu130.imagenette160` | 1 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.production_accumulation_factor.local_4060_cu130.stl10` | 1 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.production_accumulation_factor.local_4060_cu130.cifar10` | 1 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.accumulation_gradient_rule` | sample_weighted_mean_over_effective_batch | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.final_partial_accumulation` | optimizer_step_over_all_remaining_samples_no_drop | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.scheduler_steps_under_accumulation` | once_per_epoch_at_epoch_start_not_per_optimizer_step | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.epochs.imagenette160` | 100 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.epochs.stl10` | 200 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.epochs.cifar10` | 150 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.checkpoint_every_epochs` | 1 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.checkpoint_timing` | after_completed_epoch_and_before_next_epoch | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.checkpoint_resume_unit` | authenticated_completed_epoch | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.corrupt_latest_checkpoint_policy` | hold_no_older_fallback | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.incomplete_epoch_policy` | replay_from_latest_authenticated_completed_epoch | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.checkpoint_schema_version` | 1 | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.checkpoint_selection_split` | validation | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.checkpoint_selection_metric` | top1_accuracy | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.checkpoint_selection_mode` | max | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.checkpoint_selection_tie_break` | earliest_epoch | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.w5_checkpoint_selection` | prohibited_non_scientific_smoke_only | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.execution_profile_required` | true | AM-27, AM-48, AM-91, SR-10 |
| `learned_system.papr_report_required` | true | AM-27, AM-48, AM-91, SR-10, SR-16 |
| `learned_system.peak_power_constraint_available` | true | AM-27, AM-48, AM-91, SR-10, SR-16 |
| `learned_system.papr_constrained_variant_required` | true | AM-27, AM-44, AM-48, AM-91, SR-10, SR-16 |
| `learned_system.papr_constrained_variant_seeds` | 1 | AM-27, AM-48, AM-91, SR-10, SR-16 |

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
| `baseline.j2k_resolutions` | 6 | AM-51, AM-80, BR-1 |
| `baseline.j2k_code_block_size` | 64, 64 | AM-51, BR-1 |
| `baseline.j2k_tile_size` | whole_image | AM-51, BR-1 |
| `baseline.j2k_search_method` | bisection_on_compression_ratio | AM-51, BR-1 |
| `baseline.j2k_search_bounds` | 1.0, 4000.0 | AM-51, BR-1 |
| `baseline.j2k_search_tolerance_bytes` | 16 | AM-51, BR-1 |
| `baseline.j2k_search_max_iters` | 24 | AM-51, BR-1 |
| `baseline.j2k_cache_key` | canonical_pixels_sha256, budget_bytes, encode_axis_px, codec_config_hash, j2k_impl_version | AM-51, AM-54, AM-58, BR-1, BR-4 |
| `baseline.j2k_nonmonotone_policy` | keep_largest_codestream_at_or_below_budget | AM-51, BR-1 |
| `baseline.downsample_axis_px.imagenette160` | 160, 128, 96, 64 | AM-51, AM-58, AM-80, AM-82, BR-1 |
| `baseline.downsample_axis_px.stl10` | 96, 80, 64, 48 | AM-51, AM-58, AM-80, AM-82, BR-1 |
| `baseline.downsample_axis_px.cifar10` | 32 | AM-51, AM-58, AM-80, AM-82, BR-1 |
| `baseline.downsample_axis_never_upscales` | true | AM-51, BR-1 |
| `baseline.downsample_selection` | best_feasible_per_snr_on_validation_split | AM-51, BR-1 |
| `baseline.packet_count_grid` | 1, 2, 4 | AM-51, BR-16 |
| `baseline.packet_count_role` | progressive_layered_sensitivity_at_headline_ratio | AM-51 |
| `baseline.packet_count_gate` | G-2 | AM-51, BR-16, G-2 |
| `baseline.fixed_mcs_packet_count` | 1 | AM-51, BR-16 |
| `baseline.jpeg_impl` | pillow | AM-51, BR-1 |
| `baseline.jpeg_chroma_subsampling` | 4:2:0 | AM-51, BR-1 |
| `baseline.jpeg_optimise_huffman` | true | AM-51, BR-1 |
| `baseline.jpeg_quality_grid` | 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95 | AM-51, BR-4 |
| `baseline.bler_characterisation_trials` | 5000 | AM-51, BR-4 |
| `baseline.fixed_mcs_design_snr_db` | 7 | AM-51, AM-53, BR-16 |
| `baseline.container_policy` | all emitted container bytes count against payload_bits; shared-table or stripped-header variants MAY be reported as a labelled sensitivity, never as the headline | AM-51, AM-81, BR-11 |
| `baseline.channel_code` | 5g_nr_ldpc | AM-51, BR-1, ER-9, ER-12 |
| `baseline.ldpc_standard` | 3gpp_ts_38_212 | AM-51, BR-2, G-9 |
| `baseline.ldpc_standard_release` | rel_17 | AM-51, G-9 |
| `baseline.ldpc_standard_version` | 17.13.0 | AM-51 |
| `baseline.ldpc_standard_version_date` | 2026-02 | AM-51 |
| `baseline.ldpc_standard_version_pin_gate` | G-9 | AM-51, G-9 |
| `baseline.ldpc_impl` | sionna | AM-49, AM-51, AM-58, BR-2, BR-10, BR-14 |
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
| `baseline.ldpc_golden_vector_asset_sha256` | 816d75db7c0d175ea906cae1c515a6ea5295d91e1db14b5285a950c452fa70b5 | AM-51 |
| `baseline.ldpc_golden_vector_upstream_successor` | https://gitlab.com/ocudu/ocudu | AM-51 |
| `baseline.ldpc_golden_vector_sha256.ldpc_encoder_test_data.tar.gz` | cb92fe900682632a50959cbc5b164e873f733a4c911108f374e37dda3606143d | AM-51, BR-2 |
| `baseline.ldpc_golden_vector_sha256.ldpc_rate_matcher_test_data.tar.gz` | fc5e333bd94a836c4304dcdda82eac583554685aed37f0bed3758f9e573719a4 | AM-51, BR-2 |
| `baseline.ldpc_golden_vector_sha256.ldpc_segmenter_test_data.tar.gz` | f53c6ab5baac521def745e8ca591197a8dbf8b06d2d60cc3b4335a36752d6fe1 | AM-51, BR-2 |
| `baseline.ldpc_golden_vector_offline_floor` | hand_verified_small_case | AM-30, AM-51, BR-2 |
| `baseline.ldpc_golden_vector_cases` | {'index': 23, 'base_graph': 1, 'lifting_size': 36, 'modulation': 'qam16'}, {'index': 81, 'base_graph': 2, 'lifting_size': 64, 'modulation': 'qam16'} | AM-51 |
| `baseline.ldpc_base_graph` | auto_per_ts_38212 | AM-51, BR-10 |
| `baseline.ldpc_lifting_sizes` | 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384 | AM-51 |
| `baseline.ldpc_mother_code_columns.bg1` | 68 | AM-51 |
| `baseline.ldpc_mother_code_columns.bg2` | 52 | AM-51 |
| `baseline.ldpc_punctured_systematic_columns` | 2 | AM-51 |
| `baseline.ldpc_base_graph_thresholds.small_payload_bits` | 292 | AM-51 |
| `baseline.ldpc_base_graph_thresholds.medium_payload_bits` | 3824 | AM-51 |
| `baseline.ldpc_base_graph_thresholds.medium_max_rate` | 0.67 | AM-51 |
| `baseline.ldpc_base_graph_thresholds.robust_max_rate` | 0.25 | AM-51 |
| `baseline.ldpc_bg2_kb_thresholds.kb10_above_bits` | 640 | AM-51 |
| `baseline.ldpc_bg2_kb_thresholds.kb9_above_bits` | 560 | AM-51 |
| `baseline.ldpc_bg2_kb_thresholds.kb8_above_bits` | 192 | AM-51 |
| `baseline.ldpc_bg1_min_coderate` | 0.3333333333333333 | AM-51, BR-10 |
| `baseline.ldpc_rates` | 1/3, 1/2, 2/3, 5/6 | AM-24, AM-51, BR-4, BR-9, BR-15 |
| `baseline.ldpc_decoder` | offset_min_sum | AM-51, BR-14, DEC-10, G-9 |
| `baseline.ldpc_decoder_impl_spelling` | offset-minsum | AM-51, BR-14 |
| `baseline.ldpc_decoder_offset` | 0.5 | AM-51, BR-14, G-2 |
| `baseline.ldpc_decoder_llr_clip` | 20.0 | AM-51 |
| `baseline.ldpc_decoder_schedule` | flooding | AM-51 |
| `baseline.ldpc_decoder_vn_update` | sum | AM-51 |
| `baseline.ldpc_llr_convention` | log_p1_over_p0 | AM-51, BR-14 |
| `baseline.ldpc_max_iters` | 50 | AM-24, AM-51 |
| `baseline.ldpc_bler_reference_source` | lcrypto_simple_5g_ldpc_platform | AM-51, G-2 |
| `baseline.ldpc_bler_reference_must_match` | k_and_n, base_graph, lifting_size, modulation, decoder_algorithm, decoder_offset, iterations, snr_convention | AM-51, G-2 |
| `baseline.ldpc_bler_reference.repository` | https://github.com/Lcrypto/Simple-platform-to-Study-5G-LDPC-codes-and-decoders | AM-51, G-2 |
| `baseline.ldpc_bler_reference.commit` | 2fde4c43bae04c0d8397b3e7e46eaa6070e16b3c | AM-51, G-2 |
| `baseline.ldpc_bler_reference.archive_url` | https://codeload.github.com/Lcrypto/Simple-platform-to-Study-5G-LDPC-codes-and-decoders/tar.gz/2fde4c43bae04c0d8397b3e7e46eaa6070e16b3c | AM-51, G-2 |
| `baseline.ldpc_bler_reference.archive_sha256` | 457a90726a40dd40f7c115b2477f4b5ce11cecea968f8a48e51f52d0aff43ffc | AM-51, G-2 |
| `baseline.ldpc_bler_reference.licence` | MIT | AM-51, G-2 |
| `baseline.ldpc_bler_reference.decoder_source` | mexFunction/decode.c | AM-51, G-2 |
| `baseline.ldpc_bler_reference.graph_source` | BG2_iLS6 | AM-51, G-2 |
| `baseline.ldpc_bler_reference.reconstruction` | batched_literal_port_with_early_termination_disabled | AM-51, G-2 |
| `baseline.ldpc_bler_reference.k` | 128 | AM-51, G-2 |
| `baseline.ldpc_bler_reference.n` | 256 | AM-51, G-2 |
| `baseline.ldpc_bler_reference.base_graph` | 2 | AM-51, G-2 |
| `baseline.ldpc_bler_reference.lifting_size` | 22 | AM-51, G-2 |
| `baseline.ldpc_bler_reference.rate` | 0.5 | AM-51, G-2 |
| `baseline.ldpc_bler_reference.source_snr_convention` | eb_n0_per_information_bit | AM-51, G-2 |
| `baseline.ldpc_bler_reference.snr_grid_ebn0_db.bpsk` | 1.5, 2.0, 2.5, 2.75 | AM-51, G-2 |
| `baseline.ldpc_bler_reference.snr_grid_ebn0_db.qpsk` | 1.5, 2.0, 2.5, 2.75 | AM-51, G-2 |
| `baseline.ldpc_bler_reference.snr_grid_ebn0_db.qam16` | 4.0, 4.5, 5.0, 5.25 | AM-51, G-2 |
| `baseline.ldpc_bler_reference.blocks_per_snr` | 5000 | AM-51, G-2 |
| `baseline.ldpc_bler_reference.simulation_seed` | 20260730 | AM-51, G-2 |
| `baseline.ldpc_bler_reference.confidence_percent` | 95 | AM-51, G-2 |
| `baseline.ldpc_bler_reference.waterfall_target_bler` | 0.01 | AM-51, G-2 |
| `baseline.ldpc_bler_reference.waterfall_interpolation` | linear_in_snr_vs_log10_bler | AM-51, G-2 |
| `baseline.ldpc_bler_reference.required_minimum_block_errors_below_waterfall` | 5 | AM-51, G-2 |
| `baseline.progressive_packetisation_sensitivity.status` | frozen_not_run_at_G2 | AM-51 |
| `baseline.progressive_packetisation_sensitivity.ratio` | future_G8_headline_ratio | AM-51 |
| `baseline.progressive_packetisation_sensitivity.packet_counts` | 1, 2, 4 | AM-51 |
| `baseline.progressive_packetisation_sensitivity.allocation` | contiguous_equal_payload_layers_with_final_remainder | AM-51 |
| `baseline.progressive_packetisation_sensitivity.comparison` | same_total_channel_symbols_and_frozen_mcs | AM-51 |
| `baseline.progressive_packetisation_sensitivity.execution_gate` | after_G8 | AM-51 |
| `baseline.tb_crc.threshold_payload_bits` | 3824 | AM-49, AM-51, BR-5, BR-10 |
| `baseline.tb_crc.small_bits` | 16 | AM-49, AM-51, BR-5, BR-10 |
| `baseline.tb_crc.small_polynomial` | crc16 | AM-49, AM-51, BR-5, BR-10 |
| `baseline.tb_crc.large_bits` | 24 | AM-49, AM-51, BR-5, BR-10 |
| `baseline.tb_crc.large_polynomial` | crc24a | AM-49, AM-51, BR-5, BR-10 |
| `baseline.cb_crc_bits` | 24 | AM-51, BR-10, BR-14 |
| `baseline.cb_crc_polynomial` | crc24b | AM-51, BR-14 |
| `baseline.crc_spec.crc16.poly_hex` | 0x1021 | AM-51, BR-14, G-2 |
| `baseline.crc_spec.crc16.width` | 16 | AM-51, BR-14, G-2 |
| `baseline.crc_spec.crc16.init` | 0 | AM-51, BR-14, G-2 |
| `baseline.crc_spec.crc16.final_xor` | 0 | AM-51, BR-14, G-2 |
| `baseline.crc_spec.crc16.bit_order` | msb_first | AM-51, BR-14, G-2 |
| `baseline.crc_spec.crc24a.poly_hex` | 0x1864CFB | AM-51, BR-14, G-2 |
| `baseline.crc_spec.crc24a.width` | 24 | AM-51, BR-14, G-2 |
| `baseline.crc_spec.crc24a.init` | 0 | AM-51, BR-14, G-2 |
| `baseline.crc_spec.crc24a.final_xor` | 0 | AM-51, BR-14, G-2 |
| `baseline.crc_spec.crc24a.bit_order` | msb_first | AM-51, BR-14, G-2 |
| `baseline.crc_spec.crc24b.poly_hex` | 0x1800063 | AM-51, BR-14, G-2 |
| `baseline.crc_spec.crc24b.width` | 24 | AM-51, BR-14, G-2 |
| `baseline.crc_spec.crc24b.init` | 0 | AM-51, BR-14, G-2 |
| `baseline.crc_spec.crc24b.final_xor` | 0 | AM-51, BR-14, G-2 |
| `baseline.crc_spec.crc24b.bit_order` | msb_first | AM-51, BR-14, G-2 |
| `baseline.code_block_max_bits.bg1` | 8448 | AM-49, AM-51, BR-10 |
| `baseline.code_block_max_bits.bg2` | 3840 | AM-49, AM-51, BR-10 |
| `baseline.systematic_length_rule.bg1` | 22Z | AM-51, BR-10 |
| `baseline.systematic_length_rule.bg2` | 10Z | AM-51, BR-10 |
| `baseline.base_graph_selection_rule` | ts_38212_7_2_2_from_transport_block_A_and_R | AM-49, AM-51, BR-10 |
| `baseline.base_graph_pinned_at_seam` | true | AM-51, BR-10, BR-14 |
| `baseline.bg2_min_coderate` | 0.2 | AM-51, BR-10 |
| `baseline.min_coderate_predicate` | worst_block_K_prime_over_max_E_r_equality_accepted | AM-51, BR-10 |
| `baseline.segmentation_requires_exact_division` | true | AM-51, BR-10 |
| `baseline.payload_solver` | largest_byte_aligned_A_whose_full_packetisation_fits_G | AM-51, BR-10 |
| `baseline.payload_solver_clamp` | smallest_larger_byte_aligned_A_meeting_every_per_block_floor | AM-51, AM-58, BR-10 |
| `baseline.rate_matching` | ts_38212_with_filler | AM-51, BR-10 |
| `baseline.modulation_bit_interleaver` | ts_38212_5_4_2_2 | AM-51 |
| `baseline.modulation_bit_interleaver_arg` | num_bits_per_symbol | AM-51, BR-14 |
| `baseline.modulation_bit_interleaver_required` | true | AM-51, BR-14 |
| `baseline.modulations` | bpsk, qpsk, qam16 | AM-24, AM-51, AM-52, BR-2, BR-4, BR-9, BR-15, DEC-16 |
| `baseline.modulation_tuning` | adaptive_per_snr | AM-51, BR-4, DEC-16 |
| `baseline.core_modulation` | qpsk | AM-15, AM-51, BR-9 |
| `baseline.constellation_mapping` | ts_38211_gray | AM-51, BR-14 |
| `baseline.constellation_normalisation` | fixed_unit_average_energy_under_uniform_labels | AM-51 |
| `baseline.per_packet_power_rescaling_permitted` | false | AM-51 |
| `baseline.realised_symbol_energy_logged` | true | AM-51 |
| `baseline.demapper` | max_log_app | AM-51 |
| `baseline.demapper_noise_variance_convention` | two_sided_n0_over_2_per_real_dimension | AM-51 |
| `baseline.budget_rule` | solved, not evaluated in closed form: choose the largest byte-aligned transport-block payload A whose complete TS 38.212 packetisation — conditional TB CRC per baseline.tb_crc, base-graph selection from (A, R), segmentation at the selected graph's code_block_max_bits, per-block CRCs, filler and rate matching — fits exactly within G = k * bits_per_symbol coded bits and satisfies every library constraint. The complete compressed file, container bytes included, MUST fit within A/8 bytes | AM-49, AM-51, BR-3, BR-10, ER-9 |
| `baseline.control_plane_policy` | out_of_band_excluded_for_all_systems | AM-51, BR-10 |
| `baseline.outage_policy` | frozen_constant_class_selected_on_validation | AM-51, AM-58, BR-13 |
| `baseline.outage_class_selection` | highest_validation_accuracy_constant_prediction | AM-51, BR-13 |
| `baseline.outage_class_tie_break` | lowest_class_index | AM-51, BR-13 |
| `baseline.outage_class_frozen_before` | test | AM-51, BR-13 |
| `baseline.outage_policy_sensitivity` | keyed_uniform_random_label | AM-51, BR-13 |
| `baseline.outage_rng_key` | split_manifest_hash, stable_sample_id, channel_seed, purpose=outage_label | AM-51, BR-13 |
| `baseline.standards_claim` | a TS 38.212-derived 5G NR LDPC coding and rate-matching core, with custom byte-framed payloads and generic BPSK/QPSK/16-QAM over abstract AWGN | AM-51, PR-3 |
| `baseline.standards_claim_excludes` | ts_38214_tbs_and_mcs, resource_allocation, ofdm_resource_mapping, harq, pi_over_2_bpsk, full_nr_link_conformance | AM-51, PR-3 |
| `baseline.tuning` | best_feasible_config_per_snr_on_validation_split | AM-51, BR-4 |

## reference_classifier

| Parameter | Value | Cited by |
| --- | --- | --- |
| `reference_classifier.arch` | resnet18 | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.implementation` | torchvision_models_resnet | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.weights` | None | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.clean_variant_name` | clean | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.clean_train_seed` | 0 | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.init_component_path_template` | reference_classifier.{arch} | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.trained_on` | clean_images | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.pretrained_weights_permitted` | false | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.optimizer` | sgd_momentum | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.loss` | cross_entropy | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.lr` | 0.1 | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.momentum` | 0.9 | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.nesterov` | false | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.weight_decay` | 0.0005 | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.lr_schedule` | cosine | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.lr_warmup_epochs` | 5 | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.lr_warmup_schedule` | linear | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.lr_warmup_start_factor` | 0.1 | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.lr_min` | 0.0 | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.scheduler_step_unit` | epoch_start | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.scheduler_epoch_indexing` | zero_based | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.epochs` | 100 | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.batch_size` | 128 | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.label_smoothing` | 0.1 | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.augmentation` | random_resized_crop, horizontal_flip | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.batch_order` | keyed_philox_permutation_per_epoch | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.drop_last` | false | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.mixed_precision` | false | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.validation_every_epochs` | 1 | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.checkpoint_metric` | validation_top1_accuracy | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.checkpoint_mode` | max | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.checkpoint_tie_break` | earliest_epoch | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.checkpoint_schema_version` | 1 | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetuned_variant_required` | true | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_gate` | G-8 | AM-6, AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_corpus` | post_pass_one_pre_g8f_complete_authority_projected_artifact_request_universe_at_or_below_train_snr | AM-27, AM-36, AM-78, AM-87, BR-4, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_corpus_split` | train_only | AM-27, AM-36, AM-59, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_corpus_breadth` | complete_deduplicated_authority_projection_not_winners_or_a_new_neighbourhood | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_scope_anchor` | pass_one_selected_authority_references_used_only_for_dataset_ratio_scope | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_scope_modes` | all_frozen_pass_one_system_modes | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_scope_snr_rule` | at_or_below_channel_train_snr_db_fixed | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_candidate_source` | frozen_g8e_candidate_and_measurement_authorities | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_quality_identity_fields` | dataset, source_codec, payload_budget_bytes, encode_axis_px, codec_configuration_id | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_quality_projected_away` | candidate_id, composition_candidate_identity, ratio, snr_db, modulation, ldpc_rate, packet_config_id | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_candidate_admission` | authority_mapped_structural_identity_with_positive_reconciled_payload_budget | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_quality_set_rule` | complete_exact_projection_then_deduplicate_no_width_parameter | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_deduplication` | exact_canonical_artifact_quality_identity | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_support_plan` | results/baseline/g8_f/corpus_plan.json | AM-27, AM-36, AM-78, AM-88, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_sampling_plan` | results/baseline/g8_f/am88_sampler_plan.json | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_sampler_version` | g8_f_balanced_sampler_v1 | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_sampler_seed` | am88-g8f-balanced-sampler-20260824-v1 | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_assignment_algorithm` | sha256_keyed_stable_id_order_global_quality_permutation_class_chunks_cyclic_v1 | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_variants_per_training_image` | 6 | AM-27, AM-36, AM-78, BR-4, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_assignment_information` | training_stable_id, class_label, am87_quality_id_order, sampler_seed, sampler_version | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_assignment_forbidden_information` | pass_one_expected_accuracy, pass_one_score, pass_one_rank, pass_one_margin, selected_phy_tuple, validation_e4_feasibility, validation_artifact_performance, f1_codec_outcome, pass_two_result, learned_result, test_result, runtime_order | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_balance_rule` | concatenate_class_label_ordered_attempt_chunks_over_one_seed_permuted_quality_cycle | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_multiplicity` | exactly_six_distinct_attempted_supported_qualities_per_training_stable_id | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_traversal_order` | class_label_ascending_then_seed_keyed_stable_id_then_assignment_slot | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_image_codec_infeasibility` | omit_assigned_pair_record_coverage_no_resampling_or_substitution | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_non_codec_failure` | hold_no_artifact_or_substitution | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_validation_feasibility_role` | estimate_and_audit_only_never_membership | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.classifier_variant` | artifact_finetuned | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.initialization` | exact_frozen_g1_best_clean_checkpoint | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.parent_variant` | clean | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.train_seed` | 0 | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.optimizer` | sgd_momentum | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.optimizer_implementation` | torch.optim.SGD | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.loss` | cross_entropy | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.lr` | 0.01 | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.momentum` | 0.9 | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.nesterov` | false | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.weight_decay` | 0.0005 | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.lr_schedule` | cosine | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.lr_warmup_epochs` | 5 | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.lr_warmup_schedule` | linear | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.lr_warmup_start_factor` | 0.1 | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.lr_min` | 0.0 | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.scheduler_step_unit` | epoch_start | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.scheduler_epoch_indexing` | zero_based | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.epochs` | 20 | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.batch_size` | 128 | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.label_smoothing` | 0.1 | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.augmentation` | random_resized_crop, horizontal_flip | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.augmentation_input` | exact_authenticated_f1_reconstruction_pixels | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.batch_order` | keyed_philox_permutation_per_epoch_over_materialized_assignment_rows | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.drop_last` | false | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.mixed_precision` | false | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.dataloader_workers` | 4 | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.pin_memory` | true | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.validation_split` | imagenette160_validation_clean_canonical_view | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.validation_every_epochs` | 1 | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.checkpoint_every_epochs` | 1 | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.checkpoint_metric` | validation_top1_accuracy | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.checkpoint_mode` | max | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.checkpoint_tie_break` | earliest_epoch | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.resume_unit` | authenticated_completed_epoch | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.corrupt_latest_checkpoint_policy` | hold_no_older_fallback | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.incomplete_epoch_policy` | replay_from_latest_authenticated_completed_epoch | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.artifact_finetune_recipe.test_access` | prohibited | AM-27, AM-36, AM-78, AM-89, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.br4_selection_passes` | 2 | AM-27, AM-36, AM-54, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.br4_selection_terminates_after_pass` | 2 | AM-27, AM-36, AM-78, BR-4, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.headline_scorer` | artifact_finetuned | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.fallback_ladder` | extend_training_to_150_epochs, resnet34, resnet50 | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, G-1, SR-14 |
| `reference_classifier.fallback_ladder_selection_split` | validation | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, G-1, SR-14 |
| `reference_classifier.fallback_ladder_max_epochs` | 150 | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, G-1, SR-14 |
| `reference_classifier.fallback_ladder_stop_rule` | first_rung_meeting_clean_acc_floor | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, G-1, SR-14 |
| `reference_classifier.fallback_ladder_param_cap_applies` | true | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, G-1, SR-14 |
| `reference_classifier.frozen` | true | AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |
| `reference_classifier.shared_by` | classical_baseline, semantic_recon_ablation | AM-5, AM-27, AM-36, AM-78, BR-8, BR-12, DEC-15, ER-4, ER-9, SR-14 |

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
| `digital_semantic_control.transmit_dim_realised_by` | output_channel_count_and_adaptive_pooling | AM-56, ER-9, G-11 |
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
| `digital_semantic_control.framing_selector_bits` | 1 | AM-58, ER-9 |
| `digital_semantic_control.raw_escape_required` | true | ER-9 |
| `digital_semantic_control.raw_escape_rule` | transmit_min_of_range_coded_and_fixed_width_raw_selected_by_the_counted_selector_bit | ER-9 |
| `digital_semantic_control.budget_sized_against` | raw_escape_length_not_expected_compressed_length | ER-9, G-11 |
| `digital_semantic_control.transport_tuning` | best_feasible_config_per_snr_on_validation_split | BR-9, ER-9 |
| `digital_semantic_control.scored_by` | own_task_head | ER-9 |

## evaluation

| Parameter | Value | Cited by |
| --- | --- | --- |
| `evaluation.train_seeds` | 0, 1, 2 | AM-31, BR-1, BR-16, ER-1, ER-2, ER-10, ER-11, ER-12 |
| `evaluation.channel_seeds` | 0, 1, 2 | AM-31, ER-1, ER-10 |
| `evaluation.seed_pairing` | zipped_not_cross_product | AM-17, AM-31, ER-1, ER-10 |
| `evaluation.seed_cell_interpretation` | compound_replicate | AM-31, ER-10 |
| `evaluation.split_seed` | 1337 | SR-17 |
| `evaluation.split_rule` | val carved deterministically from the published train split by evaluation.split_seed; the test split is never used for any selection | SR-17 |
| `evaluation.ci` | paired_bootstrap_95 | AM-3, ER-1, ER-10 |
| `evaluation.bootstrap_resamples` | 10000 | ER-10 |
| `evaluation.paired_test` | mcnemar_exact | ER-10 |
| `evaluation.paired_test_role` | descriptive_per_cell_per_snr_only | ER-10 |
| `evaluation.estimand` | seed_conditional_mean_paired_difference | AM-57, ER-10 |
| `evaluation.estimand_conditioning` | conditional on the three preregistered trained models and the finite preregistered channel-realization schedule; not a population interval over arbitrary training or channel seeds | ER-10 |
| `evaluation.seed_aggregation` | mean_over_cells_within_image_before_bootstrap | ER-10 |
| `evaluation.bootstrap_unit` | stable_sample_id_full_trajectory | ER-10 |
| `evaluation.bootstrap_resample_scope` | once_across_all_systems_snrs_and_cells | ER-10 |
| `evaluation.per_cell_resampling_prohibited` | true | ER-10 |
| `evaluation.hypothesis_rule` | three_consecutive_snr_points | - |
| `evaluation.h1_effect_size` | mean_paired_difference_at_or_below_train_snr | ER-10 |
| `evaluation.h1_point_qualifier` | studentized_paired_mean_exceeds_1_96 | - |
| `evaluation.h1_run_calibration` | centered_multiplier_bootstrap_over_image_trajectories | AM-32, AM-57, ER-10 |
| `evaluation.h1_run_calibration_multiplier` | rademacher | ER-10 |
| `evaluation.h1_run_permutation_resamples` | 10000 | ER-10 |
| `evaluation.h1_run_calibration_binds_decision` | true | ER-10 |
| `evaluation.h1_run_calibration_alpha` | 0.05 | ER-10 |
| `evaluation.h4_uses_h1_rule` | true | ER-10 |
| `evaluation.h4_comparator` | er9_digital | ER-10 |
| `evaluation.h4_mde_gate` | G-11 | AM-70, ER-10, G-11 |
| `evaluation.h4_mde_simulation` | prospective_paired_precision_on_validation_discordance | ER-10, G-11 |
| `evaluation.gap_trend_test` | wls_slope_of_paired_gap_vs_snr | ER-10 |
| `evaluation.gap_trend_bootstrap_unit` | per_image_seed_trajectory | ER-10 |
| `evaluation.h3_requires_positive_low_snr_gap` | true | AM-39 |
| `evaluation.h3_low_snr_point_db` | -8 | ER-10 |
| `evaluation.h3_high_snr_point_db` | 18 | ER-10 |
| `evaluation.h3_requires_magnitude_contraction` | true | - |
| `evaluation.cliff_window_db` | 4 | AM-1 |
| `evaluation.cliff_window_selection` | largest_fixed_mcs_drop_on_validation_split | AM-56 |
| `evaluation.cliff_drop_pp` | 30 | AM-53, BR-16 |
| `evaluation.graceful_drop_pp` | 15 | - |
| `evaluation.cliff_reference_system` | classical_fixed_mcs | AM-53, AM-56, BR-16, DEC-16 |
| `evaluation.h2_test` | intersection_union_of_two_one_sided_margins | AM-57, ER-10 |
| `evaluation.h2_thresholds_apply_to` | one_sided_95_interval_bounds | AM-57, ER-10 |
| `evaluation.h2_did_reported_as` | additional_effect_size_not_a_support_criterion | ER-10 |
| `evaluation.h2_threshold_check_gate` | G-8 | - |
| `evaluation.h2_thresholds_frozen` | true | - |
| `evaluation.confirmatory_primary` | H1 | - |
| `evaluation.secondary_inferential` | H2, H3, H4 | - |
| `evaluation.ber_match_statistic` | waterfall_displacement_at_bler_1e-2 | AM-50, G-2 |
| `evaluation.ber_match_tolerance_db` | 0.5 | G-2 |
| `evaluation.ber_match_reference_status` | named_and_checksummed_before_G-2 | - |
| `evaluation.test_subset_size` | 2000 | AM-14, AM-42, ER-3, ER-6, ER-11 |
| `evaluation.test_access_gate` | G-12 | AM-60, DEC-12, G-12, SR-22 |
| `evaluation.test_access_requires_freeze_manifest` | true | - |
| `evaluation.test_campaign_is_single` | true | ER-6, G-12 |
| `evaluation.test_campaign_covers` | ER-1, ER-2, ER-4, ER-11, ER-12 | DEC-12, ER-6, G-12 |
| `evaluation.freeze_manifest_covers` | code_commit, resolved_config, split_manifest, checkpoints, classifier_variant, operating_points, h2_window, analysis_version | DEC-12, G-12, SR-22 |
| `evaluation.full_test_split_required_for` | ER-1 | ER-6 |
| `evaluation.metrics` | top1_acc, psnr_db, ssim, bytes_sent, header_bytes, payload_bytes, decode_failure_rate, infeasible_rate, coverage_rate, acc_given_delivery, papr_db | AM-38 |

## compute

| Parameter | Value | Cited by |
| --- | --- | --- |
| `compute.primary_device` | rtx_4060_mobile_8gb | AM-24 |
| `compute.primary_device_scope` | historical_local_profile_and_G7_measurement_not_global_default | AM-24 |
| `compute.overflow` | colab_free, kaggle_free | AM-24 |
| `compute.execution_profile_policy.eligible_profiles` | local_4060_cu130, confessor_pascal_cu126 | AM-24, SR-23 |
| `compute.execution_profile_policy.selection_scope` | per_scientific_run_or_campaign | AM-24, SR-23 |
| `compute.execution_profile_policy.selection_time` | before_first_scientific_measurement | AM-24, SR-23 |
| `compute.execution_profile_policy.implicit_mid_run_switching` | forbidden | AM-24, SR-23 |
| `compute.execution_profile_policy.interrupted_run_switching` | explicit_supersession_or_new_run_only | AM-24, SR-23 |
| `compute.execution_profile_policy.competing_sweep_profile_rule` | same_profile_for_entire_sweep | AM-24, SR-23 |
| `compute.execution_profile_policy.architecture_comparison_rule` | same_profile_or_preregistered_balanced_mapping | AM-24, SR-23 |
| `compute.execution_profile_policy.one_training_run_profile_rule` | fixed_for_complete_run | AM-24, SR-23 |
| `compute.execution_profile_policy.multi_seed_rule` | one_profile_preferred_or_preregistered_balanced_seed_mapping_applied_to_every_compared_system | AM-24, SR-23 |
| `compute.execution_profile_policy.final_test_rule` | one_frozen_profile_for_complete_paired_campaign_unless_preregistered_balanced_design | AM-24, SR-23 |
| `compute.execution_profile_policy.writer_rule` | selected_profile_host_is_sole_scientific_writer_for_that_task | AM-24, SR-23 |
| `compute.execution_profile_policy.shared_live_scientific_filesystem` | forbidden | AM-24, SR-23 |
| `compute.execution_profile_policy.durable_handoff` | verify_reconcile_commit_push_fetch | AM-24, SR-23 |
| `compute.vram_budget_gb` | 7.0 | AM-24, SR-11 |
| `compute.profiling_gate` | G-7 | AM-24, SR-11 |
| `compute.max_wall_clock_hours_per_run` | 4 | AM-24, G-8, SR-11 |
| `compute.checkpoint_every_epochs` | 1 | AM-24, SR-10 |
| `compute.ldpc_decode_cb_per_s_measured` | 625.2 | AM-24, AM-29 |
| `compute.ldpc_decode_cb_per_s_observed_range` | 625.2, 663 | AM-24 |
| `compute.ldpc_decode_batch_measured` | 32 | AM-24 |
| `compute.ldpc_decode_measured_at_iters` | 50 | AM-24 |
| `compute.ldpc_decode_early_termination` | false | AM-24 |
| `compute.ldpc_decode_measured_gate` | G-9 | AM-24, G-9 |
| `compute.er1_projected_ldpc_decode_hours_one_ratio` | 2.42 | AM-24, AM-29, AM-59 |
| `compute.er1_projected_ldpc_decode_hours_two_ratios` | 4.83 | AM-24 |
| `compute.er1_projected_ldpc_decode_basis.snr_points` | 21 | AM-24 |
| `compute.er1_projected_ldpc_decode_basis.seed_cells` | 3 | AM-24 |
| `compute.er1_projected_ldpc_decode_basis.test_images` | 3925 | AM-24 |
| `compute.er1_projected_ldpc_decode_basis.ldpc_arms` | 2 | AM-24 |
| `compute.er1_projected_ldpc_decode_basis.codeblocks_per_image` | 11 | AM-24 |
| `compute.er1_projected_ldpc_decode_basis.codeblocks_basis` | worst_case_qam16_rate_5_6_at_headline_ratio | AM-24 |
| `compute.er1_projected_ldpc_decode_hours_floor_one_ratio` | 0.44 | AM-24 |
| `compute.er1_projected_ldpc_decode_hours_floor_basis` | best_case_bpsk_rate_1_3_two_codeblocks | AM-24 |
| `compute.er1_image_transmissions_per_ldpc_arm` | 247275 | AM-24 |
| `compute.er1_j2k_test_decodes` | 247275 | AM-24 |
| `compute.er1_classifier_forwards_two_variants` | 494550 | AM-24 |
| `compute.er1_projected_ldpc_decode_scope` | LDPC decode only; excludes JPEG 2000 encode and decode, classifier forward passes, and soft demapping | AM-24 |
| `compute.er1_projected_total_hours_status` | pending_measurement_at_W3_W4 | AM-24 |
| `compute.schedule_cost_gate` | G-8 | AM-24 |
| `compute.schedule_cost_compared_as` | per_run_against_max_wall_clock_hours_per_run, aggregate_calendar_time | AM-24, G-8 |

## config

| Parameter | Value | Cited by |
| --- | --- | --- |
| `config.dir` | configs/ | AM-68, SR-1, SR-18 |
| `config.file_format` | yaml | AM-68, SR-1, SR-18 |
| `config.experiment_file_policy` | committed_choices_and_sweep_axes_only | AM-68, SR-1, SR-18 |
| `config.resolved_config_archive` | beside_run_results | AM-68, SR-1, SR-18 |
| `config.fingerprint_schema_version` | 2 | AM-68, AM-72, SR-1, SR-18 |
| `config.fingerprint_parameter_roots` | project, datasets, preprocessing, bandwidth, channel, learned_system, baseline, reference_classifier, digital_semantic_control, evaluation, compute, artifacts, environment | AM-68, SR-1, SR-18 |
| `config.fingerprint_excluded_roots` | config, demo, hardware_tier23, deliverables | AM-68, SR-1, SR-18 |
| `config.run_config_hash_form` | sha256_over_versioned_resolved_and_parameter_snapshot_canonical_json | AM-68, SR-1, SR-18 |
| `config.analysis_version` | 2 | AM-68, AM-69, AM-81, SR-1, SR-18 |
| `config.analysis_version_bump_rule` | bump_on_inference_estimand_or_analysis_implementation_change | AM-68, AM-81, SR-1, SR-18 |
| `config.dataset_version_rule` | archive_sha256 | AM-68, AM-69, SR-1, SR-18 |
| `config.literal_lint_scope` | src/ | AM-68, SR-1, SR-18 |
| `config.literal_lint_excluded_paths` | src/config/ | AM-68, SR-1, SR-18 |
| `config.literal_lint_exempt_values` | -1, 0, 1, 2, 3 | AM-68, SR-1, SR-18 |
| `config.literal_lint_annotation` | # literal-ok: | AM-68, SR-1, SR-18 |

## artifacts

| Parameter | Value | Cited by |
| --- | --- | --- |
| `artifacts.results_dir` | results/ | AM-78, SR-9, SR-13 |
| `artifacts.per_image_dir` | results/per_image/ | AM-78, ER-10, SR-13, SR-18 |
| `artifacts.checkpoint_dir` | checkpoints/ | AM-78, SR-13 |
| `artifacts.figures_dir` | figures/ | AM-78, SR-13 |
| `artifacts.run_id_key` | system, dataset, dataset_version, split, split_manifest_hash, bw_ratio, test_snr_db, train_seed, channel_seed, config_hash, checkpoint_id, classifier_variant, ldpc_rate, modulation, quantiser_bits, transmit_dim, lambda, analysis_version | AM-37, AM-58, AM-69, AM-78, SR-13, SR-18 |
| `artifacts.run_id_form` | content_addressed_sha256_over_sorted_key_value_pairs | AM-78, SR-13, SR-18 |
| `artifacts.analysis_cell_id_key` | train_seed, channel_seed | AM-78, SR-13, SR-18 |
| `artifacts.noise_id_key` | dataset_version, split_manifest_hash, stable_sample_id, test_snr_db, channel_seed, channel, k, block_index, rng_purpose | AM-78, SR-13, SR-18 |
| `artifacts.pair_id_key` | analysis_cell_id, stable_sample_id, bw_ratio, test_snr_db, noise_id | AM-78, SR-13, SR-18 |
| `artifacts.pair_id_excludes` | system, comparison | AM-78, SR-13, SR-18 |
| `artifacts.rng_stream` | counter_based_keyed_not_sequential | AM-78, SR-13, SR-18 |
| `artifacts.rng_purposes` | channel_noise, training_channel_noise, outage_label, augmentation, init, batch_order | AM-78, AM-91, SR-13, SR-18 |
| `artifacts.rng_identity_fields.channel_noise` | noise_id | AM-78, SR-13 |
| `artifacts.rng_identity_fields.training_channel_noise` | dataset_version, split_manifest_hash, stable_sample_id, train_seed, channel_seed, epoch, channel, bw_ratio, k, train_snr_db | AM-78, SR-13 |
| `artifacts.rng_identity_fields.outage_label` | split_manifest_hash, stable_sample_id, channel_seed | AM-78, SR-13 |
| `artifacts.rng_identity_fields.augmentation` | stable_sample_id, train_seed, epoch | AM-78, SR-13 |
| `artifacts.rng_identity_fields.init` | train_seed, component_path | AM-78, SR-13 |
| `artifacts.rng_identity_fields.batch_order` | train_seed, epoch | AM-78, SR-13 |
| `artifacts.init_component_path_rule` | stable_model_qualified_name | AM-78, SR-13, SR-18 |
| `artifacts.checkpoint_id_form` | sha256_of_exact_checkpoint_file_bytes | AM-78, SR-13 |
| `artifacts.classifier_artifact_dir` | results/reference_classifier/ | AM-78, SR-13 |
| `artifacts.classifier_resolved_config_file` | results/reference_classifier/resolved_config.json | AM-78, SR-13 |
| `artifacts.classifier_epoch_log_file` | results/reference_classifier/epochs.jsonl | AM-78, SR-13 |
| `artifacts.classifier_validation_summary_file` | results/reference_classifier/validation_summary.json | AM-78, SR-13 |
| `artifacts.classifier_best_checkpoint_metadata_file` | results/reference_classifier/best_checkpoint.json | AM-78, SR-13 |
| `artifacts.classifier_checkpoint_dir` | checkpoints/reference_classifier/ | AM-78, SR-13 |
| `artifacts.system_values` | learned, learned_papr_constrained, learned_snr_randomised, classical_adaptive, classical_fixed_mcs, classical_fixed_mod, classical_jpeg_secondary, classical_finetune_scored, er9_digital, label_transmission_bound, semantic_recon_ablation | AM-38, AM-78, SR-13 |
| `artifacts.csv_schema` | run_id, timestamp, git_commit, git_dirty, config_hash, checkpoint_id, system, dataset, split, n, k, bw_ratio, channel, train_snr_db, test_snr_db, train_seed, channel_seed, lambda, source_codec, jpeg_quality, j2k_target_bytes, ldpc_rate, modulation, top1_acc, n_correct, n_test, psnr_db, ssim, bytes_sent, header_bytes, payload_bytes, papr_db, decode_failure_rate, infeasible_rate, coverage_rate, acc_given_delivery, test_subset, wall_clock_s, peak_vram_gb, classifier_variant, quantiser_bits, transmit_dim, entropy_stream_bytes, entropy_table_bytes, side_information_bytes, tb_crc_type, base_graph, lifting_size, num_codeblocks, filler_bits, effective_code_rate, model_param_count | AM-38, AM-78, ER-5, FW-2, SR-13 |
| `artifacts.per_image_schema` | run_id, pair_id, noise_id, analysis_cell_id, dataset, dataset_version, split, stable_sample_id, bw_ratio, test_snr_db, true_label, pred_label, correct, outage, outage_reason, source_bytes | AM-37, AM-58, AM-78, SR-13, SR-18 |
| `artifacts.per_image_storage` | content_addressed_release_artifact_with_committed_manifest | AM-78, ER-10, SR-13 |
| `artifacts.per_image_manifest` | results/per_image_manifest.csv | AM-78, ER-10, SR-13 |
| `artifacts.inference_summary_file` | results/inference_summary.csv | AM-78, ER-10, SR-13 |
| `artifacts.inference_summary_schema` | estimand, hypothesis, systems, bw_ratio, snr_region, window, seed_aggregation, estimate, ci_low, ci_high, p_value, calibration_statistic, support_decision, source_run_ids, per_image_artifact_sha256, analysis_commit, analysis_config_hash | AM-78, ER-10, SR-13 |
| `artifacts.freeze_manifest_file` | results/freeze_manifest.json | AM-78, G-12, SR-13, SR-22 |

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
| `deliverables.w1_start_date` | 2026-07-27 | AM-59 |
| `deliverables.review_weeks.first` | 4 | AM-59, PR-2, PR-8 |
| `deliverables.review_weeks.second` | 10 | AM-59, PR-2, PR-8 |
| `deliverables.review_weeks.third` | 17 | AM-59, PR-2, PR-8 |
| `deliverables.review_dates.first` | 2026-08-18/2026-08-22 | AM-64 |
| `deliverables.review_dates.second` | 2026-09-29/2026-10-03 | AM-64 |
| `deliverables.review_dates.third` | 2026-11-17/2026-11-21 | AM-64 |
| `deliverables.review_dates.report_due` | 2026-11-20 | AM-64 |
| `deliverables.review_marks.first` | 10 | - |
| `deliverables.review_marks.second` | 30 | - |
| `deliverables.review_marks.third` | 40 | - |
| `deliverables.review_marks.report` | 20 | - |
| `deliverables.review_dates_status` | confirmed_from_2026_27_circular_marked_tentative | - |
| `deliverables.review_dates_source` | vault/capstone/Circular Capstone FALL 2026-27 Semester.pdf | - |
| `deliverables.internal_report_freeze_week` | 15 | - |
| `deliverables.contingency_week` | 16 | - |
| `deliverables.contingency_week_allocation` | report_and_results_audit | - |
| `deliverables.objectives_stated_as` | completion_terms_not_outcomes | AM-46, PR-8 |
| `deliverables.objectives_modification_point` | second_review | PR-8 |
| `deliverables.literature_review_min_refs` | 25 | PR-1 |
| `deliverables.time_plan_artifact` | gantt_chart | PR-2 |
| `deliverables.poster_format` | a0 | PR-4 |
| `deliverables.plagiarism_report_required` | true | PR-5 |
| `deliverables.report_format_source` | vault/capstone/CAPSTONE_THESIS_Format.docx | PR-6 |
| `deliverables.standards` | 3gpp_ts_38_212, 3gpp_ts_38_211, itu_t_t_800_jpeg2000, itu_t_t_81_jpeg, ieee_754, ietf_rfc_2119 | AM-50, G-2, PR-3 |
| `deliverables.novelty_claims` | er9_attribution_decomposition, br11_format_overhead_controlled_baseline | PR-7 |
| `deliverables.hardware_alternative_clause` | circular_clause_5_significant_design_aspects_real_world_application | PR-9 |
| `deliverables.hardware_alternative_artifact` | deployment_system_design_dossier | PR-9 |
| `deliverables.hardware_alternative_dossier_contents` | deployment_architecture, link_budget, packet_and_frame_design, latency_compute_energy_estimates, simulation_to_radio_mapping | PR-9 |
| `deliverables.hardware_alternative_decision_due_week` | 4 | AM-63, PR-9 |
| `deliverables.hardware_alternative_acceptance` | guide_acknowledgement_recorded | AM-63, PR-9 |
| `deliverables.proposal_registration_status` | complete_confirmed_by_author_2026_07_28 | AM-63, PR-10 |
| `deliverables.review_package_dir` | deliverables/ | AM-64, PR-8 |
| `deliverables.review_snapshot_mechanism` | annotated_git_tag | PR-8 |
| `deliverables.review_snapshot_tag_format` | review-{n}-basis | PR-8 |
| `deliverables.review_1_ready_when` | PR-1, PR-2, G-1 | PR-8 |
| `deliverables.review_1_rubric_criteria` | motivation, objectives, hypothesis, problem_survey, subject_knowledge, time_plan | PR-8 |
| `deliverables.review_1_submarks_each` | 5 | PR-8 |
| `deliverables.review_1_slot_minutes` | 15 | PR-8 |
| `deliverables.proposal_registration_owner` | author | PR-10 |

## environment

| Parameter | Value | Cited by |
| --- | --- | --- |
| `environment.python_version` | 3.14.6 | AM-58, SR-21 |
| `environment.lock_format` | pip_requirements_with_hashes | AM-58, AM-61, SR-21 |
| `environment.lock_file` | requirements.lock | AM-58, AM-61, SR-21 |
| `environment.lock_source` | requirements.in | AM-58, SR-21 |
| `environment.lock_tool` | uv | AM-58, AM-61, AM-65, SR-21 |
| `environment.lock_tool_version_min` | 0.11 | AM-58, SR-21 |
| `environment.lock_index_strategy` | unsafe-best-match | AM-58, AM-66, SR-21 |
| `environment.lock_emit_index_url` | true | AM-58, AM-66, SR-21 |
| `environment.lock_covers_spec_tooling` | true | AM-58, AM-65, SR-21 |
| `environment.pyyaml` | 6.0.3 | AM-58, AM-65, SR-21 |
| `environment.cpu_lock_file` | requirements-cpu.lock | AM-58, AM-67, SR-21 |
| `environment.cpu_lock_source` | requirements-cpu.in | AM-58, AM-67, SR-21 |
| `environment.torch_cpu` | 2.13.0+cpu | AM-58, AM-73, SR-21 |
| `environment.torchvision_cpu` | 0.28.0+cpu | AM-58, SR-21 |
| `environment.torch_cpu_index_url` | https://download.pytorch.org/whl/cpu | AM-58, SR-21 |
| `environment.torch` | 2.13.0+cu130 | AM-58, SR-21 |
| `environment.torchvision` | 0.28.0+cu130 | AM-58, SR-21 |
| `environment.torch_index_url` | https://download.pytorch.org/whl/cu130 | AM-58, AM-61, SR-21 |
| `environment.sionna` | sionna-no-rt==2.0.1 | AM-58, SR-21 |
| `environment.numpy` | 2.5.1 | AM-58, SR-21 |
| `environment.pillow` | 12.3.0 | AM-58, SR-21 |
| `environment.scikit_image` | 0.26.0 | AM-58, SR-21 |
| `environment.glymur` | 0.14.3 | AM-58, SR-21 |
| `environment.openjpeg` | 2.5.4 | AM-58, SR-21 |
| `environment.openjpeg_provisioning` | externally_provisioned_and_verified_not_repository_locked | AM-58, SR-21 |
| `environment.pytest` | 9.1.1 | AM-58, SR-21 |
| `environment.cuda_assertion` | torch.version.cuda is not None | AM-58, SR-21 |
| `environment.record_in_run_metadata` | python_version, torch_version, cuda_version, driver_version, device_name, lock_file_sha256, openjpeg_version | AM-58, SR-21 |
| `environment.cpu_install_path_required_for` | analysis, demo | AM-58, AM-67, SR-21 |
| `environment.deterministic_backend.cudnn_deterministic` | true | AM-58, SR-12, SR-21 |
| `environment.deterministic_backend.cudnn_benchmark` | false | AM-58, SR-12, SR-21 |
| `environment.reproduction_tolerance.cpu_fixture` | exact_hash | AM-58, SR-21 |
| `environment.reproduction_tolerance.gpu_same_seed_retrain_top1_pp` | 0.1 | AM-58, SR-21 |
| `environment.execution_profile_registry_schema_version` | 1 | AM-58, SR-21 |
| `environment.execution_profile_id_required_for_new_science` | true | AM-58, SR-21 |
| `environment.execution_profile_record_fields` | execution_profile_id, lock_file, lock_file_sha256, python_version, torch_version, torch_cuda_build, torchvision_version, numpy_version, sionna_version, openjpeg_version, deterministic_backend, amp, gpu_name, gpu_uuid, gpu_compute_capability, gpu_index, git_commit, git_dirty, config_hash | AM-58, SR-21 |
| `environment.execution_profiles.local_4060_cu130.role` | eligible_production_execution_profile | AM-58, SR-21 |
| `environment.execution_profiles.local_4060_cu130.host_class` | local_wsl2 | AM-58, SR-21 |
| `environment.execution_profiles.local_4060_cu130.accelerator_class` | rtx_4060_mobile | AM-58, SR-21 |
| `environment.execution_profiles.local_4060_cu130.allowed_gpu_names` | NVIDIA GeForce RTX 4060 Laptop GPU | AM-58, SR-21 |
| `environment.execution_profiles.local_4060_cu130.allowed_gpu_uuids` | GPU-607a5795-c53b-eab2-8c04-71164b173a32 | AM-58, SR-21 |
| `environment.execution_profiles.local_4060_cu130.compute_capability` | 8.9 | AM-58, SR-21 |
| `environment.execution_profiles.local_4060_cu130.python_version` | 3.14.6 | AM-58, SR-21 |
| `environment.execution_profiles.local_4060_cu130.torch_version` | 2.13.0+cu130 | AM-58, SR-21 |
| `environment.execution_profiles.local_4060_cu130.torch_cuda_build` | 13.0 | AM-58, SR-21 |
| `environment.execution_profiles.local_4060_cu130.torchvision_version` | 0.28.0+cu130 | AM-58, SR-21 |
| `environment.execution_profiles.local_4060_cu130.numpy_version` | 2.5.1 | AM-58, SR-21 |
| `environment.execution_profiles.local_4060_cu130.sionna_version` | 2.0.1 | AM-58, SR-21 |
| `environment.execution_profiles.local_4060_cu130.openjpeg_version` | 2.5.4 | AM-58, SR-21 |
| `environment.execution_profiles.local_4060_cu130.lock_source` | requirements.in | AM-58, SR-21 |
| `environment.execution_profiles.local_4060_cu130.lock_file` | requirements.lock | AM-58, SR-21 |
| `environment.execution_profiles.local_4060_cu130.lock_file_sha256` | ee68e2323a50b81967558e76da69894176a26e8d0d2dce444b5eb8c5cc7eb5cd | AM-58, SR-21 |
| `environment.execution_profiles.local_4060_cu130.deterministic_backend.cudnn_deterministic` | true | AM-58, SR-21 |
| `environment.execution_profiles.local_4060_cu130.deterministic_backend.cudnn_benchmark` | false | AM-58, SR-21 |
| `environment.execution_profiles.local_4060_cu130.amp` | true | AM-58, SR-21 |
| `environment.execution_profiles.local_4060_cu130.scientific_writer_host` | local | AM-58, SR-21 |
| `environment.execution_profiles.confessor_pascal_cu126.role` | eligible_production_execution_profile | AM-58, SR-21 |
| `environment.execution_profiles.confessor_pascal_cu126.host_class` | remote_arch_linux | AM-58, SR-21 |
| `environment.execution_profiles.confessor_pascal_cu126.accelerator_class` | pascal_gp102 | AM-58, SR-21 |
| `environment.execution_profiles.confessor_pascal_cu126.allowed_gpu_names` | NVIDIA GeForce GTX 1080 Ti, NVIDIA TITAN Xp | AM-58, SR-21 |
| `environment.execution_profiles.confessor_pascal_cu126.allowed_gpu_uuids` | GPU-00214b86-48e7-fcf0-bf46-575fa7f85b6b, GPU-46acd0f2-2ff5-1a43-cac9-2ae20e56dc9a | AM-58, SR-21 |
| `environment.execution_profiles.confessor_pascal_cu126.compute_capability` | 6.1 | AM-58, SR-21 |
| `environment.execution_profiles.confessor_pascal_cu126.python_version` | 3.14.6 | AM-58, SR-21 |
| `environment.execution_profiles.confessor_pascal_cu126.torch_version` | 2.13.0+cu126 | AM-58, SR-21 |
| `environment.execution_profiles.confessor_pascal_cu126.torch_cuda_build` | 12.6 | AM-58, SR-21 |
| `environment.execution_profiles.confessor_pascal_cu126.torchvision_version` | 0.28.0+cu126 | AM-58, SR-21 |
| `environment.execution_profiles.confessor_pascal_cu126.numpy_version` | 2.5.1 | AM-58, SR-21 |
| `environment.execution_profiles.confessor_pascal_cu126.sionna_version` | 2.0.1 | AM-58, SR-21 |
| `environment.execution_profiles.confessor_pascal_cu126.openjpeg_version` | 2.5.4 | AM-58, SR-21 |
| `environment.execution_profiles.confessor_pascal_cu126.lock_source` | requirements-pascal.in | AM-58, SR-21 |
| `environment.execution_profiles.confessor_pascal_cu126.lock_file` | requirements-pascal.lock | AM-58, SR-21 |
| `environment.execution_profiles.confessor_pascal_cu126.lock_file_sha256` | d3561c8e930797d328cf45df1bdc5085842833665f9b5c9e5617d4c891e31a82 | AM-58, SR-21 |
| `environment.execution_profiles.confessor_pascal_cu126.deterministic_backend.cudnn_deterministic` | true | AM-58, SR-21 |
| `environment.execution_profiles.confessor_pascal_cu126.deterministic_backend.cudnn_benchmark` | false | AM-58, SR-21 |
| `environment.execution_profiles.confessor_pascal_cu126.amp` | true | AM-58, SR-21 |
| `environment.execution_profiles.confessor_pascal_cu126.scientific_writer_host` | confessor | AM-58, SR-21 |
| `environment.qualification.scientific_status` | non_scientific_zero_coverage | AM-58, SR-21, SR-23 |
| `environment.qualification.pascal_eligibility_status` | qualified_both_registered_devices_and_cross_profile_parity | AM-58, SR-21, SR-23 |
| `environment.qualification.pascal_qualification_artifacts` | results/execution_profiles/qualification | AM-58, SR-21, SR-23 |
| `environment.qualification.forbidden_activities` | G8_coverage, test_access, validation, selection, training_campaign | AM-58, SR-21, SR-23 |
| `environment.qualification.both_devices_separate` | true | AM-58, SR-21, SR-23 |
| `environment.qualification.parity_criterion` | no_systematic_decoder_regression_no_material_waterfall_displacement_no_catastrophic_paired_disagreement | AM-58, SR-21, SR-23 |
| `environment.qualification.bit_identical_gpu_decoding_required` | false | AM-58, SR-21, SR-23 |
| `environment.qualification.rng_byte_hashes` | information_bits, awgn_real_float64, awgn_imag_float64 | AM-58, SR-21, SR-23 |
| `environment.qualification.openjpeg_fixture_parity` | required_for_codec_eligible_profiles | AM-58, SR-21, SR-23 |
| `environment.historical_profile_compatibility.amendment` | AM-83 | AM-58, SR-1, SR-21 |
| `environment.historical_profile_compatibility.accepted_fingerprint_schema_versions` | 1 | AM-58, SR-1, SR-21 |
| `environment.historical_profile_compatibility.archived_profile` | local_4060_cu130 | AM-58, SR-1, SR-21 |
| `environment.historical_profile_compatibility.rule` | exact_archived_parameter_bytes_plus_additive_profile_registry_only | AM-58, SR-1, SR-21 |
| `environment.historical_profile_compatibility.unrelated_parameter_drift` | forbidden | AM-58, SR-1, SR-21 |
| `environment.historical_profile_compatibility.reinterpret_as_other_profile` | forbidden | AM-58, SR-1, SR-21 |
| `environment.scientific_writer_authentication.github_transport` | authenticated_https | AM-58, SR-21, SR-23 |
| `environment.scientific_writer_authentication.github_cli_device_flow_permitted` | true | AM-58, SR-21, SR-23 |
| `environment.scientific_writer_authentication.commit_signing` | optional | AM-58, SR-21, SR-23 |
| `environment.scientific_writer_authentication.mandatory` | authenticated_github_write, internal_artifact_hashes_and_contracts, durable_checkpoint_commits, push_success, local_remote_commit_parity | AM-58, SR-21, SR-23 |
