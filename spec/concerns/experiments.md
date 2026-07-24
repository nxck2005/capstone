<!-- GENERATED FROM spec/SPEC.md BY tools/gen_spec_views.py - DO NOT EDIT -->

# Experiments

Requirements extracted from [`SPEC.md`](../SPEC.md). This view is for focused reading and review; the spec text is normative.

## ER

- **ER-1** — **Core crossover.** Both systems, full test split, every SNR in `params.channel.test_snr_grid_db`, at `params.bandwidth.core_ratio`, repeated over `params.evaluation.seeds`, reported with `params.evaluation.ci` intervals, measuring `params.project.primary_metric`. Pass/fail is §2. *(verify: results CSV + accuracy-vs-SNR figure)*
- **ER-2** — **SNR mismatch.** Train once at `params.channel.train_snr_db_fixed`, evaluate across the whole test grid, and report the degradation profile of both systems to evidence the cliff-versus-graceful contrast. *(verify: results CSV + figure)*
- **ER-3** — **Bandwidth sweep.** Repeat the comparison across every entry in `params.bandwidth.ratios` to locate where the learned advantage is largest. *(verify: results CSV + figure)*
- **ER-4** — **Task-training ablation.** The semantic reconstruction head MUST also be scored through the frozen `params.reference_classifier` (BR-8), separating gain attributable to joint coding from gain attributable to a task-trained classifier. Reported as a distinct `system` value. *(verify: results CSV containing the ablation rows)*
- **ER-5** — Every experiment MUST emit rows matching `params.artifacts.csv_schema` exactly — same columns, same order. *(verify: schema validation script over all result CSVs)*
- **ER-6** — Evaluation on `params.evaluation.test_subset_size` images is permitted for sweeps only; `params.evaluation.full_test_split_required_for` MUST use the full split, and the `test_subset` column MUST record which was used. *(verify: schema test asserting the flag)*
- **ER-7** — Every number appearing in the thesis MUST be traceable to a `run_id` and `git_commit` in a committed CSV. *(verify: audit script resolving each reported figure to its rows)*
- **ER-8** — If ER-1 fails, the negative result MUST be reported with the same rigour as a positive one, together with the diagnostic evidence. Weakening the baseline to manufacture a crossover is prohibited. *(verify: review against BR-4 sweep artifacts)*

## Parameters referenced here

| Parameter | Value |
| --- | --- |
| `artifacts.csv_schema` | run_id, timestamp, git_commit, system, dataset, n, k, bw_ratio, channel, train_snr_db, test_snr_db, seed, jpeg_quality, ldpc_rate, modulation, top1_acc, psnr_db, ssim, bytes_sent, decode_failure_rate, n_test, test_subset |
| `bandwidth.core_ratio` | r_1_12 |
| `bandwidth.ratios` | *(see datasheet)* |
| `channel.test_snr_grid_db` | -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25 |
| `channel.train_snr_db_fixed` | 7 |
| `evaluation.ci` | student_t_95 |
| `evaluation.full_test_split_required_for` | ER-1 |
| `evaluation.seeds` | 0, 1, 2 |
| `evaluation.test_subset_size` | 2000 |
| `project.primary_metric` | top1_accuracy_vs_snr |
| `reference_classifier` | *(see datasheet)* |
