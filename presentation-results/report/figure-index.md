# Figure index and slide captions

> **Update 2026-10-05:** this package predates the G-12 test campaign. G-12 has since run once on the 3,925 test images (2026-10-02) and all four hypotheses are supported. For test numbers use `data/g12_test_curves.csv` and `data/g12_test_differences.csv`, the paper's figures and [`docs/RESULTS.md`](../../docs/RESULTS.md). The validation numbers here are unchanged and still correct.

All x positions are measured channel SNR in dB. Accuracy is end-to-end `n_correct/1000` on the frozen validation split. Solid connecting segments guide the eye; they are not interpolated evidence. The PDF/SVG/PNG files have the same numbered basename.

| Figure | What is plotted | Suggested caption and qualification |
|---|---|---|
| **01** `headline_full_snr` | Standard learned DJSCC vs adaptive classical, 1/6, all 21 points | **System-level comparison.** Learned transmission remains useful below −4 dB; adaptive classical is more accurate once delivery succeeds. Learned uses its task head; classical uses the artifact-finetuned classifier. |
| **02** `low_snr_closeup` | Standard learned, randomized learned and adaptive classical, −8…+2 dB | **The sampled digital delivery transition lies between −5 and −4 dB.** Randomized training improves the observed low-SNR learned curve. No points were added between measured SNRs. Scorers differ across learned/classical. |
| **03** `bandwidth_efficiency` | Learned and adaptive classical at 1/24 and 1/6, all 21 points | **Four times fewer channel uses moves the tradeoff.** Compare within each ratio; 1/24 uses 3,200 symbols and 1/6 uses 12,800. Learned/classical scorer difference applies in both panels. |
| **04** `training_robustness` | Fixed-SNR and SNR-randomized learned checkpoints, 1/6 | **Randomized training lifts weak-channel performance in this frozen seed cell.** Both use own task-head architecture, with separately trained weights. |
| **05** `task_aware_digital` | Standard learned and ER-9 task-aware digital, matched 1/6 uses | **Task-aware digital features are strong after delivery; learned DJSCC avoids the low-SNR outage cliff.** Feature encoders and trained task heads differ; this does not isolate the channel code. |
| **06** `papr_tradeoff` | Learned accuracy across 21 points and symbol-domain PAPR mean→maximum for each checkpoint | **The constrained learned checkpoint meets the 3 dB cap with tolerance in the published validation measurement.** PAPR is symbol-domain; the checkpoints differ. The max is per observed image, not waveform PAPR. |
| **07** `secondary_controls` | Four grouped panels: adaptive/fixed MCS/fixed modulation; adaptive/JPEG; ER-9/predicted label; learned/reconstruction ablation | **Controls probe selection, codec, transmitted task decision and reconstruction path.** The first two panels share the artifact-finetuned scorer. The reconstruction panel deliberately has different scorers. The JPEG +9 dB outage is retained. |
| **08** `delivery_coverage` | Accuracy of learned, adaptive and ER-9 at 1/6 above; delivery of adaptive and ER-9 at 1/6 plus adaptive at 1/24 below | **Delivery coverage explains the digital cliff.** Learned predictions have 100% output availability in this contract; digital outage points use the measured fallback rule. Scorers in the upper panel differ. |

All figures use the **primary scorer stream** for each arm. The `clean` secondary classical streams remain available in `data/w10_scorers_357.csv` for a paper table or a carefully labeled supplementary figure.
