# Results

The measured results of the project, in one place, with what each number means, what it does not mean, and where it comes from. The final test results (G-12) come first; the validation results that preceded them, written 2026-10-01, follow. Not normative: `spec/SPEC.md` governs, and the published evidence under `results/` is the source of truth.

## Final test results (G-12)

### Read this first

- **These numbers are from the test split**: 3,925 held-out Imagenette-160 images, read once by the G-12 campaign after every model, operating point and analysis choice was frozen on validation data (freeze `g12freeze-f96082101953546fdda096746c073bea342358a6f0e5f045c6435020d5fe511c`, 441/441 units, closed 2026-10-02).
- **DJSCC, adaptive JPEG 2000 + LDPC, the task-aware digital control and the fixed 16-QAM curve are averaged over three seed pairs**; every other variant is one seed pair. Intervals are 95% image-bootstrap intervals from the same shared bootstrap as the hypothesis tests.
- **The two sides are still graded by different classifiers** (DJSCC by its own head, the image systems by the artifact-fine-tuned ResNet-18), so a learned-versus-digital gap remains a whole-system difference.
- **9.9 means every packet failed** and the outage rule (always predict class 0) applied; class 0 is 387 of the 3,925 test images.

**Sources.** `results/g12/analysis.json` (hypothesis decisions), `results/g12/results.csv` (one row per unit), `presentation-results/data/g12_test_curves.csv` and `g12_test_differences.csv` (seed-averaged curves and paired differences, written by `tools/export_g12_tables.py`), and the per-image streams in release `g12-test-per-image-2026-10-02`. Post-freeze corrections 1–2 and the aborted first attempt are recorded under `results/g12/`.

### Preregistered hypotheses: all four supported

| Hypothesis | Comparator | Test result |
|---|---|---|
| H1 low-SNR separation (primary) | DJSCC vs adaptive JPEG 2000 + LDPC | Supported: calibrated p = 0.030; four consecutive qualifying points, −8 to −5 dB |
| H2 graceful versus cliff | DJSCC vs fixed 16-QAM over the frozen 3→7 dB window | Supported: classical drop 76.7 points (one-sided lower bound 75.5), DJSCC drop 0.2 points (upper bound 0.4); difference-in-differences 76.5 [75.1, 77.9] |
| H3 convergence | DJSCC vs adaptive JPEG 2000 + LDPC | Supported: gap slope −1.71 points/dB [−1.76, −1.67]; low-SNR gap 66.3 points, high-SNR gap −4.2 points |
| H4 attribution | DJSCC vs task-aware digital (ER-9) | Supported: calibrated p = 0.0096; longest qualifying run seven points, 1 to 7 dB |

### Headline numbers

Top-1 accuracy (%) on the 3,925 test images.

| r | System | Seeds | −8 | −7 | −6 | −4 | −2 | 0 | +4 | +7 | +18 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1/6 | DJSCC | 3 | 76.1 | 77.5 | 79.1 | 80.8 | 81.4 | 81.6 | 82.1 | 82.3 | 82.4 |
| 1/6 | DJSCC, SNR-randomized training | 1 | 77.6 | 78.8 | 80.0 | 81.1 | 82.1 | 82.6 | 82.9 | 82.6 | 83.0 |
| 1/6 | DJSCC, 3 dB PAPR cap | 1 | 76.7 | 78.3 | 79.3 | 81.0 | 81.4 | 81.8 | 82.4 | 82.3 | 82.3 |
| 1/6 | DJSCC reconstruction → clean ResNet-18 | 1 | 44.1 | 48.5 | 52.5 | 60.3 | 65.3 | 70.9 | 75.8 | 77.3 | 78.0 |
| 1/6 | JPEG 2000 + LDPC, adaptive | 3 | 9.9 | 9.9 | 9.9 | 81.6 | 83.2 | 85.9 | 86.2 | 86.5 | 86.6 |
| 1/6 | … same outputs, clean ResNet-18 | 3 | 9.9 | 9.9 | 9.9 | 59.2 | 78.2 | 84.4 | 86.4 | 87.5 | 87.6 |
| 1/6 | JPEG 2000 + LDPC, QPSK only | 1 | 9.9 | 9.9 | 9.9 | 9.9 | 9.9 | 85.9 | 86.2 | 86.5 | 86.5 |
| 1/6 | JPEG 2000 + LDPC, fixed 16-QAM r=1/2 | 3 | 9.9 | 9.9 | 9.9 | 9.9 | 9.9 | 9.9 | 9.9 | 86.5 | 86.5 |
| 1/6 | Baseline JPEG + LDPC | 1 | 9.9 | 9.9 | 9.9 | 66.4 | 74.5 | 81.2 | 85.1 | 85.5 | 86.1 |
| 1/6 | Task-aware digital | 3 | 9.9 | 9.9 | 9.9 | 81.1 | 81.1 | 81.1 | 81.1 | 81.1 | 81.1 |
| 1/6 | Task-aware digital, rate 1/5 (AM-100) | 1 | 9.9 | 56.9 | 80.9 | 80.9 | 80.9 | 80.9 | 80.9 | 80.9 | 80.9 |
| 1/6 | Label transmission | 1 | 9.9 | 9.9 | 9.9 | 81.0 | 81.0 | 81.0 | 81.0 | 81.0 | 81.0 |
| 1/24 | DJSCC | 1 | 47.3 | 53.9 | 59.1 | 68.3 | 73.9 | 76.3 | 78.8 | 79.8 | 80.0 |
| 1/24 | JPEG 2000 + LDPC, adaptive | 1 | 9.9 | 9.9 | 9.9 | 9.9 | 28.9 | 68.0 | 81.6 | 84.9 | 86.3 |
| 1/24 | … same outputs, clean ResNet-18 | 1 | 9.9 | 9.9 | 9.9 | 9.9 | 13.5 | 25.2 | 59.2 | 79.8 | 85.9 |

Paired per-image difference in percentage points (DJSCC minus the other system) at r = 1/6, three seed pairs, with 95% intervals:

| SNR (dB) | DJSCC − JPEG 2000 + LDPC | DJSCC − task-aware digital |
|---:|---:|---:|
| −8 | +66.3 [+64.8, +67.8] | +66.3 [+64.8, +67.8] |
| −5 | +70.2 [+68.8, +71.6] | +70.2 [+68.8, +71.6] |
| −4 | −0.9 [−2.0, +0.3] | −0.3 [−1.0, +0.4] |
| −3 | −0.7 [−1.9, +0.4] | −0.2 [−0.9, +0.5] |
| −2 | −1.8 [−2.9, −0.7] | +0.3 [−0.4, +0.9] |
| 0 | −4.3 [−5.4, −3.3] | +0.5 [−0.2, +1.1] |
| +1 | −4.1 [−5.1, −3.1] | +0.7 [+0.1, +1.4] |
| +4 | −4.0 [−5.0, −3.0] | +1.0 [+0.4, +1.7] |
| +7 | −4.2 [−5.2, −3.2] | +1.2 [+0.6, +1.9] |
| +18 | −4.2 [−5.2, −3.2] | +1.3 [+0.7, +2.0] |

### Findings

1. **The cliff holds on test.** At 1/6 neither digital system delivers anything from −8 to −5 dB, and DJSCC keeps 76–80% there. DJSCC leads by 66–70 points across that range.
2. **The crossover sits at −4 to −3 dB, where the two are statistically tied.** From −2 dB up, adaptive JPEG 2000 + LDPC is more accurate, by about four points at high SNR (86.6% against 82.4% at 18 dB).
3. **The task-aware digital control nearly matches DJSCC wherever it delivers.** It reaches 81.1%; DJSCC exceeds it by 0.7–1.3 points from 1 dB up, and the two are tied from −4 to 0 dB. That small, positive high-SNR margin is what H4 credits to joint coding.
4. **A lower code rate takes most of DJSCC's remaining low-SNR advantage.** The rate-1/5 task-aware variant (AM-100) delivers from −7 dB and scores 80.9% from −6 dB, leaving DJSCC a clear lead only at −7 dB and below.
5. **The receiver's classifier matters as much as the link.** Scoring the same JPEG 2000 outputs with the clean-trained ResNet-18 costs up to 43 points near the delivery threshold at 1/24, and 22 points at −4 dB at 1/6.
6. **A tighter budget moves the digital threshold up by several dB.** At 1/24 the adaptive digital system delivers nothing up to −3 dB, is still behind DJSCC at 1 dB (72.6% against 77.5%), and passes it from 2 dB (79.1% against 78.0%). These are one seed pair.
7. **The validation findings carry over, each on one seed pair.** SNR-randomized training scores 77.6% at −8 dB against 71.5% for fixed-SNR training in the same seed pair, and stays about 1.5 points ahead from −4 dB up. The 3 dB PAPR cap costs almost nothing. Reconstructing the image first and classifying it with a clean ResNet-18 loses 20–32 points from −8 to −4 dB.

### Wording for test results

| Say | Don't say |
|---|---|
| "On the 3,925-image test split, averaged over three seed pairs…" | "Our results show…" (no split or seeds named) |
| "All four preregistered hypotheses are supported" | "We proved DJSCC is better" |
| "Statistically tied at −4 and −3 dB" | "DJSCC wins at −4 dB" |
| "From −2 dB up, the adaptive digital system is more accurate" | "DJSCC is competitive at all SNRs" |
| "DJSCC exceeds the task-aware digital control by 0.7–1.3 points from 1 dB up" | "Joint coding adds over a point everywhere" |
| "With rate-1/5 coding the digital control delivers from −7 dB" | "Digital systems cannot work below −5 dB" |

## Validation results (W10 v11), written 2026-10-01 before the test campaign

> Everything from here down describes the **validation** split as it stood before G-12. Statements that the test split is sealed, or that H1–H4 are undecided, were true when written and are superseded by the section above. The validation numbers themselves remain correct.

### Read this first (validation)

- **Every number below is from the validation split.** 1,000 Imagenette-160 images, 100 per class. The 3,925-image test split is sealed and has never been read by a model (`test_access = 0`). Test results come from the single campaign at G-12.
- **The system curves come from one seed pair** (training seed 0, channel seed 0). The spec's hypotheses are decided on three seed pairs, on test, with paired bootstrap intervals. None of that has happened yet, so nothing here decides H1–H4.
- **No confidence intervals exist for the curves.** A difference of one or two points between two curves is not a finding. The one three-seed measurement we do have (the H4 diagnostic, below) shows why.
- **The two sides are graded by different classifiers.** DJSCC uses its own classification head. The JPEG 2000 and JPEG systems are graded by a ResNet-18 fine-tuned on JPEG 2000-compressed images. A learned-versus-digital gap is a whole-system difference, not a channel-coding difference.
- **Both sides were tuned on these same 1,000 images.** The digital operating points, the DJSCC checkpoint, λ, and the task-aware feature size were all chosen on validation. Every curve is somewhat optimistic; the digital curves more so, because they are re-tuned at every SNR.

**Sources.** Curves: `presentation-results/data/w10_primary_252.csv` (252 rows: 12 system variants × 21 SNRs) and `w10_scorers_357.csv` (all classifier streams), derived from the published closeout `results/learned/w10/w10_continuation_closeout_v11.json` (commit `c31dd2b`). Operating points: `results/learned/w10/w10_continuation_units_v11.json`. Paper figures: `deliverables/research-paper/figures/`. Full tables: the supplement, `deliverables/research-paper/supplement/`.

## The result in one paragraph

At a bandwidth ratio of 1/6 (12,800 channel uses per image), DJSCC is the only system that works at −5 dB and below (with the configured code rates, the lowest of which is 1/3; [`novelty-review.md`](novelty-review.md) led to AM-100's rate-1/5 variant, measured above), where it keeps 73–78% accuracy while both digital systems deliver nothing. Between −5 and −4 dB the digital systems start delivering, and from −4 dB upward the adaptive JPEG 2000 system is more accurate than DJSCC by 2.7 to 6.1 points. At 1/24 (3,200 channel uses) the digital system needs much more SNR: DJSCC stays ahead up to +3 dB. The task-aware digital control, which sends learned features over the same digital link, scores 82.0% whenever its packets arrive and nothing below −4 dB. So at this ratio most of DJSCC's low-SNR advantage comes from avoiding outage, not from a better representation.

## Headline numbers

Top-1 accuracy (%), 1,000 validation images per entry. 10.0 means every packet failed and the fallback rule (always predict class 0) was applied, which is right on exactly 100 of 1,000 images.

| r | System | −8 | −6 | −4 | −2 | 0 | 4 | 7 | 9 | 18 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1/6 | DJSCC | 72.8 | 75.7 | 79.4 | 80.7 | 81.2 | 82.8 | 83.6 | 83.8 | 83.4 |
| 1/6 | DJSCC, SNR-randomized training | 77.0 | 80.2 | 82.4 | 82.5 | 83.0 | 82.9 | 83.8 | 84.1 | 83.9 |
| 1/6 | DJSCC, 3 dB PAPR cap | 75.9 | 78.8 | 80.5 | 80.7 | 82.3 | 82.9 | 82.8 | 83.0 | 83.2 |
| 1/6 | DJSCC reconstruction → clean ResNet-18 | 44.7 | 54.5 | 61.1 | 63.9 | 69.0 | 74.1 | 76.1 | 77.5 | 76.8 |
| 1/6 | JPEG 2000 + LDPC, adaptive | 10.0 | 10.0 | 83.4 | 83.9 | 87.3 | 87.8 | 89.0 | 86.5 | 89.3 |
| 1/6 | … same outputs, clean ResNet-18 | 10.0 | 10.0 | 60.9 | 78.8 | 84.8 | 87.0 | 88.6 | 86.3 | 89.2 |
| 1/6 | JPEG 2000 + LDPC, QPSK only | 10.0 | 10.0 | 10.0 | 10.0 | 87.3 | 87.8 | 88.7 | 88.7 | 88.7 |
| 1/6 | JPEG 2000 + LDPC, fixed 16-QAM r=1/2 | 10.0 | 10.0 | 10.0 | 10.0 | 10.0 | 10.0 | 89.0 | 89.0 | 89.0 |
| 1/6 | Baseline JPEG + LDPC | 10.0 | 10.0 | 67.8 | 76.7 | 84.0 | 87.2 | 87.9 | 10.0 | 88.6 |
| 1/6 | Task-aware digital | 10.0 | 10.0 | 82.0 | 82.0 | 82.0 | 82.0 | 82.0 | 82.0 | 82.0 |
| 1/6 | Label transmission | 10.0 | 10.0 | 82.0 | 82.0 | 82.0 | 82.0 | 82.0 | 82.0 | 82.0 |
| 1/24 | DJSCC | 49.4 | 61.7 | 69.3 | 76.2 | 78.7 | 82.0 | 82.9 | 82.7 | 82.8 |
| 1/24 | JPEG 2000 + LDPC, adaptive | 10.0 | 10.0 | 10.0 | 31.0 | 69.4 | 83.4 | 85.8 | 82.4 | 87.7 |
| 1/24 | … same outputs, clean ResNet-18 | 10.0 | 10.0 | 10.0 | 14.2 | 26.2 | 60.9 | 80.6 | 80.7 | 87.0 |

The full 21-SNR grid for every variant, with delivery rates, is in Tables S7–S9 of the supplement.

## Findings

Each finding gives the numbers, what they support, and what not to claim.

### 1. DJSCC degrades gradually; the digital systems fall off a cliff

At 1/6, DJSCC loses 11 points across 17 dB, from 83.8% at 9 dB to 72.8% at −8 dB, and always produces a prediction. The adaptive JPEG 2000 system delivers nothing from −8 to −5 dB, then every packet at −4 dB, where it jumps from 10.0% to 83.4%.

- **Supports:** the graceful-degradation-versus-cliff picture the project set out to test.
- **Don't claim:** the exact cliff SNR. The measured transition lies between −5 and −4 dB, and no SNR in between was evaluated.

### 2. Above the cliff, the adaptive digital baseline is more accurate

From −4 dB upward the JPEG 2000 system leads at every measured SNR, by 2.7 to 6.1 points, reaching 89.3% against 83.4% at 18 dB. DJSCC saturates near 84%. The digital system reaches 89%, close to the clean classifier's 89.8% on uncompressed images.

- **Likely reason** (not tested): the DJSCC task head is a single linear layer on pooled decoder features, and DJSCC was trained at one SNR (7 dB). A stronger head might narrow the gap.
- **Don't claim:** that DJSCC "outperforms" the digital baseline. It does only where the baseline delivers nothing.

### 3. A tighter bandwidth budget moves the crossover by 8 dB

At 1/24, DJSCC drops most at low SNR: 49.4% at −8 dB against 72.8% at 1/6. At 18 dB it barely changes (82.8% against 83.4%). The digital system delivers nothing up to −3 dB. At −2 dB it gets 82% of packets through, but only 34.8% of those images are classified correctly, because the payload is tiny. DJSCC leads up to +3 dB, the digital system leads from +4 to +7 dB, DJSCC leads narrowly again at 9 dB (82.7% against 82.4%, where 62 digital packets fail), and the digital system leads by 3.8–4.9 points from 11 dB up.

- **Supports:** less bandwidth widens the region where DJSCC wins, from ≤ −5 dB to ≤ +3 dB.
- **Don't claim:** a single crossover SNR at 1/24. The curves cross more than once.

### 4. The task-aware digital control closes most of the gap, where it delivers

The task-aware digital system sends 2,048 learned features, quantized to 2 bits and range-coded, over the same LDPC link (BPSK, rate 1/3) with the same 12,800 channel uses. It scores 82.0% at every SNR from −4 dB upward: error-free delivery reproduces the transmitter's features exactly, so its accuracy cannot vary. Below −4 dB it fails completely, like the JPEG 2000 system.

- **Single seed (seed pair 0):** 2.6 points ahead of DJSCC at −4 dB, crossing between 0 and 2 dB, 0.8–1.8 points behind from 4 dB upward.
- **Across three seeds (G11/H4 diagnostic, finding 11):** at −4 dB the two are essentially tied (learned − digital = −0.4 points). From −2 to 7 dB DJSCC is ahead by 0.6–1.8 points.
- **Supports:** at 1/6, most of DJSCC's low-SNR advantage comes from avoiding outage. When both deliver, a task-aware representation sent digitally is within about 2 points of DJSCC.
- **Don't claim:** this isolates the channel code. The feature encoders, quantization and heads differ between the two systems.
- **Don't claim:** learned features are no better than images. The features (82.0%) are read by one linear layer and the images (83.4–89.3%) by a fine-tuned ResNet-18, so that gap says nothing about the representation.

### 5. The label-transmission bound inherits the task-aware system's threshold

Sending only the transmitter's predicted label (4 bits) scores 82.0% wherever it delivers, identical to the task-aware system, because it sends that system's own prediction. It also fails below −4 dB, because the one-byte label rides in the task-aware system's 4,096-bit packet and so has the same decoding threshold.

- **Supports:** an upper bound on the accuracy of transmitting a decision.
- **Don't claim:** how far a transmitted decision can reach. A link sized for a 4-bit payload would use a much lower code rate and should survive at lower SNR. That was not measured.

### 6. The receiver's classifier changes the crossover

The same JPEG 2000 outputs, graded by the clean ResNet-18 instead of the fine-tuned one:
- at 1/6, −4 dB: 60.9% instead of 83.4%, so DJSCC would lead up to −2 dB instead of −5 dB;
- at 1/24, 0 dB: 26.2% instead of 69.4%, a 43-point difference, so DJSCC would lead up to 9 dB.

Above about 4 dB (at 1/6) the two classifiers agree within a point.

- **Supports:** near the cliff the baseline sends small, heavily compressed images, and a classifier that has seen such images matters a great deal. A crossover SNR depends on the receiver's classifier, not just the transmission scheme.
- **Use it as:** evidence that the baseline was made strong. Fine-tuning the classifier helped the digital side.
- **Use it as:** the project's lead contribution (paper, Section I). That retraining on compressed images helps is known in computer vision (Janeiro et al. 2023); what is new, to our knowledge, is measuring how much it moves a learned-versus-digital comparison. Prior comparisons such as Huang et al. 2024 grade the digital images with a classifier trained on clean images.

### 7. Reconstructing the image first loses a lot

Classifying the DJSCC reconstruction with the clean ResNet-18 scores 44.7% at −8 dB and 76.8% at 18 dB, below the jointly trained head at every SNR (72.8% and 83.4%).

- **Supports:** for this task, reading the decoder features directly beats rebuilding the picture and classifying it.
- **Don't claim:** it measures reconstruction quality alone. The classifier changes as well.

### 8. SNR-randomized training helps at low SNR

Training with the SNR drawn uniformly from {1, 4, 7, 13, 19} dB gives 77.0% against 72.8% at −8 dB and 82.4% against 79.4% at −4 dB. It is higher at 19 of 21 SNRs, equal at one, and within a point of the fixed-SNR model at high SNR.

- **Don't claim:** an effect size. These are two separate training runs with one seed, so seed variation and the training distribution can't be separated.

### 9. A 3 dB peak-power cap costs almost nothing here

The unconstrained DJSCC encoder has a symbol-domain PAPR of 13.2 dB on average (20.1 dB maximum). The capped model stays at 3.000002 dB maximum, within the 3 dB cap and its 10⁻⁴ dB tolerance. Its accuracy is close: 75.9% against 72.8% at −8 dB and 83.2% against 83.4% at 18 dB, higher at 13 SNRs and lower at 6. For comparison, the adaptive JPEG 2000 system at 18 dB (16-QAM) has a mean PAPR of 2.6 dB and a maximum of 3.3 dB.

- **Don't claim:** RF or amplifier compliance. PAPR is measured on the complex symbols, not on an oversampled, pulse-shaped waveform. The capped model is also a separate training run.

### 10. Adaptivity is worth a lot to the baseline

Restricting the baseline shows what its per-SNR adaptation buys:
- **QPSK only:** first delivery moves from −4 to −1 dB, and the plateau drops to 88.7%.
- **One fixed operating point (16-QAM, rate 1/2):** nothing below 7 dB, 89.0% above, a 79-point drop between 7 and 6 dB.
- **Baseline JPEG instead of JPEG 2000:** up to 15.6 points lower just above the cliff (67.8% against 83.4% at −4 dB). This is consistent with JPEG's larger header overhead, and the classifier was fine-tuned on JPEG 2000 artifacts, not JPEG.

Two dips should be reported as observed:
- **JPEG at 9 dB fails on every image**, although the same setting delivers everything at 11 dB.
- **The adaptive JPEG 2000 system loses 38 packets at 9 dB.**

In both cases the selected setting sits on the steep part of its BLER curve. Selection uses the measured curves analytically, and the image-by-image run then disagrees by a few percent of packets.

- **Use it as:** evidence that a single-operating-point baseline would have made DJSCC look far better. The project deliberately did not use one.

### 11. H4 can resolve about 2 points, and only above the cliff

The G11/H4 precision diagnostic ran on validation with all three seed cells, DJSCC against the task-aware digital system at 1/6, with 10,000 paired bootstrap resamples. Minimum detectable difference at 80% power (median):

| SNR (dB) | −8 to −5 | −4, −3 | −2 to 7 |
|---|---:|---:|---:|
| Detectable difference (points) | 3.9–4.0 | 1.9 | 1.8–1.9 |
| Observed mean difference, learned − digital (points) | +68 to +71 | −0.4 to +0.1 | +0.6 to +1.8 |

The 2-point reference was met at 9 of the 16 points examined, all between −2 and 7 dB except 2 dB. The diagnostic is pointwise only. It does not certify the power of the full H4 procedure. A negative H4 would therefore be conservative, and would not rule out a real advantage of a point or two.

- **For the paper:** the Discussion now uses this measured figure (1.8–1.9 points above the cliff, about 4 points below) instead of the spec's analytic estimate of 1.4–3.2 points, and reports the single-seed 2.6-point lead beside the three-seed result. Source: `results/learned/g11/h4_pointwise_precision_v4.json`.

## Selection records

These were chosen on validation before any system comparison was run.

**Reconstruction weight λ** (`results/learned/w7/w7_g4_result.json`). Candidates were {0, 0.1, 0.3, 1, 3}, scored by accuracy at 7 dB and reconstruction PSNR at 15 dB. The rule admits every λ within one point of the λ=0 accuracy (floor 81.7%), keeps those with PSNR ≥ 20 dB, and takes the smallest.

| λ | Accuracy | PSNR |
|---|---:|---:|
| 0 | 82.7% | 9.7 dB |
| 0.1 | 83.5% | 17.6 dB |
| 0.3 | 82.2% | 18.7 dB |
| 1 | 82.8% | 19.9 dB |
| **3** | **83.2%** | **21.2 dB** |

Only λ=3 reached 20 dB.

**Task-aware feature size D** (2 bits per feature, real digital chain at 7 dB):

| D | Accuracy |
|---|---:|
| 64 | 28.4% |
| 128 | 55.2% |
| 256 | 75.9% |
| 512 | 79.3% |
| 1024 | 81.8% |
| **2048** | **82.0%** (selected) |

**Bandwidth ratios.** 1/6 was selected at G-8 as the main comparison ratio and 1/24 as the low-bandwidth ratio.

**Operating points of the adaptive baseline at 1/6** (paper Table V):

| SNR (dB) | Image size | Coding |
|---|---|---|
| −8 to −5 | — | nothing decodes |
| −4, −3 | 64 × 64 | BPSK, rate 1/3 |
| −2 | 96 px | BPSK, rate 1/2 |
| −1, 0 | 128 px | QPSK, rate 1/3 |
| 1 | 128 px | BPSK, rate 2/3 |
| 2 to 5 | 128 px | QPSK, rate 1/2 |
| 6 | 160 px | QPSK, rate 5/6 |
| 7 | 160 px | 16-QAM, rate 1/2 |
| 9 and up | 160 px | 16-QAM, rate 2/3 |

## Implementation checks

These show the machinery works. They are not comparisons.

| Check | Result |
|---|---|
| Clean ResNet-18 (from scratch) | 898/1000 validation, target 0.88, 11,181,642 parameters |
| Fine-tuned ResNet-18 | 20 epochs from the clean model on 44,039 JPEG 2000-decoded training images |
| DJSCC size | 1,567,197 parameters at 1/6; 1,539,537 at 1/24; 1,640,957 at 1/2 |
| DJSCC cost | 48.7 s per epoch and 1.0 GiB at batch 32 on an RTX 4060 Laptop GPU |
| LDPC conformance | waterfall within 0.0037 dB of an independent reference at BLER 10⁻² (BPSK, QPSK, 16-QAM) |
| Packetization | 216 configurations checked; 215 feasible, 1 structurally infeasible |
| BLER campaign | 153 curves, 3,213 points, 5,000 trials each (16,065,000 block trials) |
| Classical validation sweep | 288,000 image evaluations: 264,000 delivered, 24,000 codec-infeasible, 0 decode or structural failures |
| G-10 crossover check | 63 validation evaluations; crossover observed exactly once, between −5 and −4 dB at 1/6 |
| PAPR-capped training | 100/100 epochs; selected epoch 80 (human count) at 833/1000; max PAPR 3.000002 dB |

## Paper figures

The paper's figures now plot the **G-12 test** results (`g12_test_curves.csv`, `g12_test_differences.csv`):

| Paper figure | File | Shows |
|---|---|---|
| Fig. 1 | `fig_headline` | Accuracy and delivery at 1/6: DJSCC, adaptive JPEG 2000, task-aware digital and its rate-1/5 variant |
| Fig. 2 | `fig_differences` | Paired difference, DJSCC minus each digital system, at 1/6 |
| Fig. 3 | `fig_bandwidth` | DJSCC against JPEG 2000 at 1/6 and 1/24 |
| Fig. 4 | `fig_scorer` | (a) same JPEG 2000 outputs, two classifiers; (b) same DJSCC transmissions, two receivers |
| Fig. 5 | `fig_training` | Fixed-SNR, SNR-randomized and PAPR-capped DJSCC |
| Fig. 6 | `fig_controls` | Adaptive baseline against QPSK-only, fixed operating point and baseline JPEG |

The files are in `deliverables/research-paper/figures/`. Regenerate them with `python3 deliverables/research-paper/figures/make_figures.py`.

## Wording that is safe, and wording that is not (validation)

| Say | Don't say |
|---|---|
| "On the validation split…" | "Our results show…" (with no split named) |
| "DJSCC is the only system that works below −5 dB at r=1/6, with code rates no lower than 1/3" | "DJSCC outperforms the digital baseline" |
| "The measured transition lies between −5 and −4 dB" | "The cliff is at −4.5 dB" |
| "Higher at 19 of 21 SNRs in one training run" | "SNR randomization improves accuracy by 3 points" |
| "Whole-system comparison; the two sides use different classifiers" | "Joint coding beats separate coding by X points" |
| "Symbol-domain PAPR within the 3 dB cap" | "Meets amplifier/RF peak-power limits" |
| "Consistent with an outage-avoidance explanation" | "Proves the gain is from outage avoidance" |
| "To our knowledge, the first to measure how much of DJSCC's advantage comes from the comparison rather than the learning" | "The first to compare DJSCC with digital transmission" |
| "The features and images are read by different classifiers, so their gap says nothing about the representation" | "Learned features don't beat images" |
| "Within about 2 points of DJSCC across three seeds" | "2.6 points ahead of DJSCC" (one seed only) |
| "Accuracy at equal channel uses and equal average symbol energy" | "Saves energy" |
