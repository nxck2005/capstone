# Six results slides to insert into the existing deck

**Opening bridge (20 seconds):** “We ask whether a system trained end to end for image classification can keep its decision useful when a short wireless link degrades. We compare at the same number of channel uses, then test what changes when training, bandwidth, digital controls and peak power change.” The result is a system-level validation comparison, not a claim that one channel code alone wins.

## 1. A usable decision below the digital delivery transition

**Figure:** 01 full-width. **Takeaway:** At 1/6 bandwidth, learned DJSCC remains useful at very low SNR; the adaptive classical chain is more accurate from −4 dB upward.

**Speaker notes:** “Every point represents the same 1,000 validation images. At −8 dB, learned achieves 72.8% while the digital chain cannot deliver and its fallback is 10%. At −4 dB, digital delivery succeeds and classical accuracy jumps to 83.4%, above learned's 79.4%. We show the entire −8 to +18 dB grid. The learned task head and classical artifact-tuned classifier differ, so we treat this as a complete-system comparison.”

## 2. Why the cliff happens

**Figures:** 02 large; 08 as a compact companion or next build. **Takeaway:** Delivery failure drives the abrupt digital change; randomized learned training strengthens low-SNR predictions.

**Speaker notes:** “The close-up has only actual measurements. Classical coverage is zero through −5 dB and 100% at −4 dB. Learned predictions degrade gradually. Randomized training gives 77.0% at −8 dB versus 72.8% for fixed-SNR training. Coverage also dips at isolated higher points, so the digital curve is not perfectly monotone.”

## 3. Bandwidth changes the answer

**Figure:** 03. **Takeaway:** Reducing uses from 12,800 to 3,200 hurts both systems, with a larger weak-SNR shift for the classical chain.

**Speaker notes:** “At 1/24 bandwidth, learned still scores 49.4% at −8 dB. Adaptive classical does not begin substantial delivery until higher SNR, and at 0 dB it scores 69.4% against learned's 78.7%. At stronger SNR it often regains the lead. One 9 dB coverage dip means we should say ‘often’, not declare one universal crossover.”

## 4. Task-aware digital is the fairer mechanism check

**Figure:** 05. **Takeaway:** Digital transmission of learned task features can be competitive after delivery, yet retains an outage cliff.

**Speaker notes:** “ER-9 sends task-aware features at the matched 1/6 channel-use budget. It reaches 82% when it delivers. This control narrows the story: the low-SNR learned advantage is not simply ‘task-aware versus generic JPEG’. The feature representation and task heads still differ, so this figure does not isolate the LDPC code or one neural component.”

## 5. Peak power and practical fairness

**Figure:** 06. **Takeaway:** A separately trained PAPR-constrained model meets the frozen 3 dB symbol-domain cap and preserves similar validation accuracy.

**Speaker notes:** “The unconstrained learned symbols have an observed maximum PAPR of 20.06 dB. The constrained run has 3.000002 dB, within the 0.0001 dB frozen tolerance. Its accuracy is 75.9% at −8 dB and 83.2% at +18 dB. This is a simulation-domain symbol measurement, not a measured RF amplifier result.”

## 6. What the controls establish—and what they do not

**Figure:** 07, ideally split its four panels over two builds; retain all panels in the paper. **Takeaway:** Adaptive selection, codec choice, transmitted labels and the reconstruction pathway each affect the result.

**Speaker notes:** “Fixed MCS exposes a later delivery threshold; JPEG is competitive high up but has an unfavorable outage at 9 dB. A transmitted predicted label reaches 82% after delivery; it is not the true label. Reconstructing first and then using the clean classifier underperforms direct task-head scoring, but the classifier also changes. The test set has never been opened.”

**Closing (15 seconds):** “The completed validation experiment establishes a real low-SNR robustness advantage for the learned system, while a strong adaptive classical system is better when it delivers. Next work would test more seed cells, assess uncertainty from the stored per-image streams, run the separately gated final test, and examine waveform PAPR or hardware only under a new authorized study.”

## Integration instructions

Use `figures/png/01_headline_full_snr.png` and the other numbered PNGs for slides; set the image to preserve aspect ratio and crop only outer whitespace. Use the matching PDF or SVG for a paper. Put the slide takeaway above the figure, and use the caption in `report/figure-index.md` as a small note below it. Keep “validation, n=1,000 images per point, single seed cell” on the opening results slide and “test sealed” on the closing slide. Figure 07 is dense: give it a full slide or split its panels via crop without removing the 9 dB JPEG point.
