<!-- GENERATED FROM spec/SPEC.md BY tools/gen_spec_views.py - DO NOT EDIT -->

# Experiments

Requirements extracted from [`SPEC.md`](../SPEC.md). This view is for focused reading and review; the spec text is normative.

## ER

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

## Parameters referenced here

| Parameter | Value |
| --- | --- |
| `artifacts.csv_schema` | run_id, timestamp, git_commit, git_dirty, config_hash, checkpoint_id, system, dataset, split, n, k, bw_ratio, channel, train_snr_db, test_snr_db, train_seed, channel_seed, lambda, source_codec, jpeg_quality, j2k_target_bytes, ldpc_rate, modulation, top1_acc, n_correct, n_test, psnr_db, ssim, bytes_sent, header_bytes, payload_bytes, papr_db, decode_failure_rate, infeasible_rate, test_subset, wall_clock_s, peak_vram_gb |
| `bandwidth.core_ratio` | r_1_3 |
| `bandwidth.low_ratio_operating_point` | r_1_12 |
| `bandwidth.ratios` | *(see datasheet)* |
| `baseline.channel_code` | 5g_nr_ldpc |
| `baseline.core_modulation` | qpsk |
| `channel.test_snr_grid_db` | -8, -6, -4, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 18 |
| `channel.train_snr_db_fixed` | 7 |
| `evaluation.bootstrap_resamples` | 10000 |
| `evaluation.channel_seeds` | 0, 1, 2 |
| `evaluation.ci` | paired_bootstrap_95 |
| `evaluation.full_test_split_required_for` | ER-1 |
| `evaluation.paired_test` | mcnemar_exact |
| `evaluation.test_subset_size` | 2000 |
| `evaluation.train_seeds` | 0, 1, 2 |
| `project.primary_metric` | top1_accuracy_vs_snr |
| `reference_classifier` | *(see datasheet)* |
