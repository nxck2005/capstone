# Project Knowledge Transfer

Written 2026-10-01, replacing the version from before final training. This is the document to read first if you are new to the project, coming back to it, or preparing for a review or viva. It explains the idea, the experiment, where things stand, what was found, and how the repository is organised, in that order.

It is not normative. `spec/SPEC.md` governs the science, `NEXT.md` holds the session-to-session working state, and [`RESULTS.md`](RESULTS.md) holds every measured number with its caveats. If this document disagrees with any of those, they win.

## 1. The project in one minute

A small camera at the edge of a network sees an image. A server on the other side of a noisy wireless link has to say what is in it: one of ten classes. The camera and server can only use a fixed amount of radio time per image.

The conventional way is to compress the image, protect the bits with an error-correcting code, send them, rebuild the image at the server, and classify it. The alternative this project studies is **deep joint source–channel coding (DJSCC)**: a neural network at the camera maps the image straight to radio symbols, a second network at the server reads the noisy symbols and outputs the class, and the two are trained together through a simulated channel.

The project compares three systems at exactly the same radio budget and channel:

1. **Conventional:** JPEG 2000 compression, then 5G NR LDPC error correction and BPSK/QPSK/16-QAM modulation, re-tuned at every signal strength.
2. **Task-aware digital:** learned task features, quantised and sent over that same digital link.
3. **DJSCC:** learned end to end, with no bits in between.

The second system exists so the result can be explained. If DJSCC wins, is it because it learned what matters for the task, or because it codes source and channel jointly? Comparing systems 1 and 2 answers the first question; comparing 2 and 3 answers the second.

The project succeeds if this comparison is built, run, and reported properly under the preregistered protocol. DJSCC is not required to win.

## 2. Where the project stands

As of 2026-10-01, during Second Review week (29 September – 3 October).

| Stage | State |
|---|---|
| Environment, datasets, manifests, keyed randomness | Done (W1) |
| Clean reference classifier, ResNet-18 from scratch (G-1) | Done: 898/1000 on validation |
| DJSCC architecture and compute profile (G-7) | Done |
| LDPC conformance against an independent reference (G-2) | Done: within 0.0037 dB |
| Conventional baseline built, BLER curves measured, operating points selected (W4, G-8) | Done: 153 curves, 3,213 points; 288,000 validation evaluations |
| Classifier fine-tuned on JPEG 2000 artifacts | Done |
| λ pilot and selection (G-4) | Done: λ = 3 |
| Final DJSCC training, three seeds × two ratios (W8) | Done: six models |
| Crossover check on validation (G-10) | Done: crossover between −5 and −4 dB at 1/6 |
| Task-aware digital control, three seeds (ER-9) | Done: 2,048 features, 2 bits |
| SNR-randomized DJSCC (ER-2), PAPR-capped DJSCC | Done: one run each |
| H4 precision diagnostic (G-11) | Done: see [`RESULTS.md`](RESULTS.md), finding 11 |
| Full validation rehearsal: 12 variants × 21 SNRs (W10) | Done: 252 measurements, published at `c31dd2b` |
| Second Review deck, demo app, paper draft, supplement | Ready |
| **Test split** | **Sealed. Never read by a model.** |

**What comes next**, in order (the dates are the course's, from `params.deliverables`):

1. **W11: G-12 test release.** Commit the freeze manifest (`results/freeze_manifest.json`, which does not exist yet), covering the code commit, configs, split manifest, checkpoints, classifier variants and operating points. Only then open the test split for **one** campaign. That campaign runs every system on the full 3,925-image test split with three seed pairs, then runs the paired bootstrap that decides H1–H4. The freeze-manifest generator, the test-campaign runner and the H1–H4 analysis still have to be built.
2. **W12:** results frozen and reported whichever way they fall (G-5).
3. **W13–W14:** figure polish; optional hardware stretch (Tier 2 SDR replay), otherwise a pre-recorded demo; poster draft.
4. **W15:** report in the university format, results audit, novelty statement, plagiarism report. **Internal report freeze.**
5. **W16:** contingency, allocated to finishing the report.
6. **W17: Final Review, 17–21 November. Report due 20 November.**

## 3. The idea in more detail

### Why not just compress and send?

Shannon's separation theorem says compressing and error-protecting separately loses nothing, but only for infinitely long messages. Real links send short packets with a fixed latency budget, and at finite length separate coding pays a price: the code needs a safety margin, and some packets still fail. When a packet fails, the receiver gets nothing at all. This is the **cliff**: above some signal strength the digital link works almost perfectly; below it, nothing arrives.

A learned joint system has no packets to lose. As the signal gets worse its output gets noisier, but some information still arrives. This is **graceful degradation**. The hope is that noisy-but-useful beats nothing at low SNR.

### Why not just send the class label?

For ten classes the whole answer is four bits, and four heavily protected bits would survive almost any channel. The project answers this with its **premise**: the split between camera and server is fixed by the deployment. The camera runs an encoder, the server owns the task, and the camera can't run the whole classifier, perhaps because it is too small or because the task belongs to the server. Within that split, the question is the best way to transmit. The label-transmission variant in the results is there so the reader can see what the premise costs.

### What "semantic" means here

Nothing to do with language models. "Semantic" or "task-oriented" communication just means the system is judged on whether the receiver completes its task (here, top-1 classification accuracy), not on how faithfully it rebuilds the image.

## 4. The systems

### 4.1 Conventional: JPEG 2000 + LDPC (adaptive)

At each signal strength, the baseline picks the combination of image size (64–160 px), LDPC code rate (1/3, 1/2, 2/3, 5/6) and modulation (BPSK, QPSK, 16-QAM) that maximises expected accuracy on validation. Expected accuracy combines the measured chance that every block decodes with the measured accuracy on correctly decoded images. The JPEG 2000 file is made as large as the packet can carry. The LDPC chain follows 3GPP TS 38.212 (TB CRC, segmentation, CB CRC, rate matching). Sionna 2.0.1 does the LDPC encoding and decoding; the project writes the rest.

Delivered images are classified by a **ResNet-18 fine-tuned on JPEG 2000-compressed training images**. Failed packets get the **outage rule**: predict class 0, which is right 10% of the time on the balanced validation set. Failures stay in the accuracy; they are never dropped.

The baseline is deliberately strong: adaptive modulation, a codec chosen to avoid JPEG's header overhead, and a classifier that has seen compression artifacts. A weak baseline would make any DJSCC win meaningless.

### 4.2 Task-aware digital control (ER-9)

This system uses the same CNN trunk as DJSCC, then outputs 2,048 features (a 32 × 8 × 8 map), squashes them to [−1, 1], and quantises each to 2 bits. The indices are range-coded (or sent raw if that is shorter) and carried by the same LDPC link at BPSK, rate 1/3, in the same channel uses. It is trained without a channel, because the link either delivers the indices exactly or fails.

### 4.3 DJSCC

Two strided convolutions downsample the image by 4 to a 40 × 40 grid, followed by two residual blocks (GroupNorm, PReLU). A 3 × 3 convolution then produces 2c channels, and each pair becomes one complex symbol:
- c = 8 at 1/6, giving 12,800 symbols;
- c = 2 at 1/24, giving 3,200 symbols.

The symbols are normalised to unit average power per image and sent through complex AWGN. The receiver mirrors the encoder and has two heads: a reconstruction and a class prediction (global average pooling plus one linear layer). The loss is cross-entropy plus 3 × MSE. Training uses Adam at 10⁻³ with cosine decay, 100 epochs, batch 32, mixed precision, and a fixed training SNR of 7 dB. The model has about 1.57 M parameters at 1/6.

### 4.4 Variants measured alongside

| Variant | What it asks |
|---|---|
| DJSCC trained at random SNRs {1, 4, 7, 13, 19} dB (ER-2) | Does training on varied channels help at low SNR? |
| DJSCC with a 3 dB peak-to-average power cap | What does a hardware-friendly signal cost? |
| DJSCC reconstruction → clean ResNet-18 (ER-4) | Is rebuilding the image worse than reading the features directly? |
| JPEG 2000, QPSK only (BR-9) | What is adaptive modulation worth? |
| JPEG 2000, one fixed operating point (BR-16) | What does the sharpest possible cliff look like? |
| Baseline JPEG instead of JPEG 2000 (DEC-9) | Why JPEG 2000 was chosen as the codec |
| Label transmission (ER-12) | Upper bound on sending a decision instead of data |
| Every digital output also scored by the clean ResNet-18 (BR-12) | How much does the receiver's classifier matter? |

## 5. What makes the comparison fair

- **Same radio budget.** Every system gets exactly k complex channel uses per image. The digital packet is solved to fill that budget exactly, including every CRC and filler bit.
- **Same power and noise.** Unit average symbol power, and SNR defined as Es/N0 per complex channel use.
- **Same noise realisation.** The noise for each (image, SNR, ratio) comes from a keyed random generator that ignores which system asks. Every system at a given ratio sees the identical noise vector, which makes per-image paired comparisons valid.
- **Failures count.** Decode failures and impossible configurations stay in the denominator via the outage rule.
- **Validation for every choice.** Operating points, λ, checkpoints, feature size and ratios were all chosen on validation. The test split is guarded in code (`src/data/test_access.py`) and cannot be loaded without the freeze manifest.
- **The baseline adapts, the learned model does not.** The digital system is re-tuned at every SNR; each learned model is trained once. This favours the baseline and is disclosed.

The fairness claim is about accuracy at equal channel uses and equal average symbol energy. It is not a measured energy saving.

## 6. Data and channel

- **Imagenette-160**: a 10-class subset of ImageNet at 160 px (fast.ai). The published training set is split into 8,469 training and 1,000 validation images (100 per class). The published validation set (3,925 images) is our **test** split. Every image has a stable ID from a hash of its bytes, and membership is fixed by versioned manifests in `data/manifests/`.
- **Bandwidth ratio** r = k / (160·160·3). The main ratio is **1/6** (k = 12,800) and the low-bandwidth ratio **1/24** (k = 3,200).
- **Channel:** simulated complex AWGN, 21 SNRs: −8 to 7 dB in 1 dB steps, then 9, 11, 13, 15 and 18 dB. No fading, synchronisation errors or hardware effects (Tier 1 is simulation only).

## 7. What was found

The short version is in [`RESULTS.md`](RESULTS.md), which you should read in full before writing or presenting anything. Five points to keep in your head:

1. **The cliff is real.** At 1/6, both digital systems deliver nothing at −5 dB and below. DJSCC still gets 73–78% there. That cliff sits where it does partly because the lowest code rate allowed is 1/3; whether to disclose or test this is an open decision (see [`novelty-review.md`](novelty-review.md)).
2. **Above the cliff the baseline wins.** From −4 dB up, adaptive JPEG 2000 beats DJSCC by 2.7–6.1 points (89.3% against 83.4% at 18 dB).
3. **Less bandwidth, bigger DJSCC region.** At 1/24, DJSCC stays ahead up to +3 dB.
4. **Most of the low-SNR gain is outage avoidance.** The task-aware digital control reaches 82.0% whenever it delivers, within about 2 points of DJSCC across three seeds, but falls off the same cliff.
5. **The receiver's classifier matters.** Grading the digital images with a classifier that never saw compression artifacts costs up to 43 points near the cliff, and moves the range where DJSCC leads by 3–6 dB.

All of this is validation data from one seed pair (except the three-seed H4 diagnostic). It decides none of the hypotheses.

**What is new.** No single system here is new. To our knowledge, what is new is measuring how much of DJSCC's advantage comes from the way it is compared rather than from the learning: points 4 and 5 are the two controls that show it. The one-sentence answer, the closest prior work (Huang et al. 2024, Lokumarambage et al. 2026, SwinJSCC, Ren et al. 2025) and the prepared viva answers are in [`novelty-review.md`](novelty-review.md).

## 8. The hypotheses

Preregistered in `spec/SPEC.md` §2, decided only on the test split, averaged over three seed pairs, with image-level paired bootstrap intervals (10,000 resamples):

- **H1 (primary): low-SNR separation.** DJSCC beats adaptive JPEG 2000 at 1/6 at three or more consecutive SNRs at or below 7 dB, each with a paired interval above zero. The run rule is calibrated so that three-in-a-row isn't a lucky streak.
- **H2: graceful versus cliff.** Across a fixed 3 dB window chosen on validation, the fixed-operating-point baseline loses at least 30 points while DJSCC loses at most 15.
- **H3: convergence.** The gap shrinks as SNR rises.
- **H4: attribution.** DJSCC also beats the task-aware digital control under the H1 rule. If it doesn't, the gain is credited to task-aware representation. The G-11 diagnostic shows H4 can resolve about 2 points above the cliff, so a null result will be conservative.

A curve crossing is reported if seen, but it is not a pass condition. Completion doesn't depend on which way the results fall.

## 9. Deliverables

| Deliverable | Where | State |
|---|---|---|
| First Review deck | `deliverables/review-1/` | Delivered (18–22 Aug) |
| Second Review deck and presenter guide | `deliverables/review-2/` | Ready for 29 Sep–3 Oct |
| Research paper (IEEE) | `deliverables/research-paper/capstone_rp.tex` | Draft, retitled 2026-10-01 ("How Much of Deep Joint Source–Channel Coding's Advantage Survives a Fair Comparison?"); Sections IV–X being revised by the authors |
| Supplementary material (IEEE) | `deliverables/research-paper/supplement/` | Draft |
| Results package (figures, CSVs, notes) | `presentation-results/` | Done (validation) |
| Offline exhibition demo | `demo/` | Built; weights provisioned separately |
| Literature review (30 sources) | `docs/literature-review.md` | Done |
| Gantt chart | `docs/gantt-plan.md` | Keep current |
| Standards register | `docs/standards-and-tools-register.md` | Done |
| Poster, final report, plagiarism report | not started | W14–W15 |

The paper and supplement acknowledge AI assistance section by section, as IEEE requires. Keep that acknowledgment accurate as sections are rewritten.

## 10. Repository map

| Path | What it holds |
|---|---|
| `spec/SPEC.md` | The specification: thesis, hypotheses, decisions, requirements, schedule, gates, amendment record (§17). Normative. |
| `spec/params.generated.yaml` | Every experiment constant. Code reads constants only from here, via `src/config/params.py`. |
| `NEXT.md` | Session hand-off and working state. Read first, update last. Scrappy by design. |
| `AGENTS.md` | Guidance for coding agents, including commands. |
| `src/models/` | DJSCC (`djscc.py`), task-aware digital model (`er9_digital.py`), classifiers, task heads |
| `src/channels/` | AWGN with keyed noise, power normalisation, PAPR cap |
| `src/baseline/` | JPEG 2000 (`j2k.py`), JPEG, LDPC chain (`ldpc/`), classical pipeline and selection (`classical/`), G-8 campaign code |
| `src/training/` | Training loops, losses, SNR randomisation, PAPR training |
| `src/evaluation/` | ER-9 protocol, G-10, H4 diagnostic, W10 evaluation backends |
| `src/data/` | Dataset adapters, manifests, preprocessing, the guarded test boundary |
| `results/` | Published evidence: JSON records with hashes, one directory per phase |
| `presentation-results/` | W10 figures (PNG/PDF/SVG), CSVs, findings, slide notes |
| `deliverables/` | Review decks, paper, supplement |
| `demo/` | Offline React + FastAPI demo |
| `docs/` | Hand-written explanations (this file, results, literature review, crossover explainer, Gantt, standards) |
| `tests/` | Test suite (`.venv/bin/python -m pytest`) |
| `tools/` | Generators, verifiers, runners, deck and figure builders |

## 11. How the repository keeps itself honest

The repository is stricter than a typical student project. Every choice that could flatter a result is made before the result is seen, and that has to be provable afterwards. Four mechanisms do the work:

- **One source for constants.** No experiment number is hard-coded in `src/`. `tools/check_literals.py` enforces this, and literals that must stay are marked `# literal-ok`.
- **Keyed randomness.** Random draws come from generators keyed by what they are for: noise by image, SNR and ratio; shuffles by seed and epoch. Results don't depend on batch order, and the same noise is reproducible anywhere.
- **Evidence with hashes.** Each phase writes JSON evidence that names the code commit, config hash, inputs and outputs by SHA-256. Verifier scripts in `tools/` re-check it. Closed evidence is never edited; a correction is a new, additive record.
- **The spec changes by amendment.** Any change to the science is written into `SPEC.md` §17 as an AM entry with its reason. Nothing is changed silently.

This is why there are many files named like `..._v4.json`, `..._closeout.json` and `..._authorization.json`. They record who approved what, when, and on which exact inputs.

## 12. Rules before changing anything

- **Don't open the test split.** Only the G-12 campaign may, after the freeze manifest is committed. Any selection or tuning on test invalidates the project.
- **Don't re-run closed campaigns.** W8, ER-9, ER-2, the PAPR run, G-10, G-11 and W10 are closed. Re-running one because a number looks wrong is exactly the selective reporting the protocol forbids. If there's a real defect, it gets an amendment and a complete re-run.
- **Don't weaken the baseline.** Every change must strengthen the baseline or be preregistered. Never handicap the learned system either.
- **Don't hard-code constants.** Add them to the spec's parameters.
- **Don't edit published evidence.** Add a new record instead.
- **Report failures.** Outages stay in the denominator, and unfavourable points (like the JPEG 9 dB outage) stay in the figures.

## 13. Things that sound reasonable but are wrong here

- **"DJSCC beats the digital system."** Only where the digital system delivers nothing. Above the cliff it loses by up to 6 points.
- **"The curves must cross for the project to pass."** No. Crossing is reported if seen and is not a criterion.
- **"The project shows an energy saving."** No. It compares accuracy at equal channel uses and equal average symbol energy.
- **"Failed packets can be dropped, since there is no prediction."** No. They are scored by the outage rule and stay in the denominator.
- **"This is reinforcement learning."** No. It's supervised end-to-end training through a differentiable channel.
- **"Validation and test are both held out, so either can be used for tuning."** No. Every choice is made on validation, and test opens once.
- **"The PAPR-capped model meets RF power limits."** Only in the symbol domain, not on a transmitted waveform.
- **"One seed shows SNR randomisation adds 3 points."** It shows a 3-point difference between two runs. The effect size needs more seeds.
- **"Hardware is required."** No. Tiers 2 and 3 (SDR, Raspberry Pi) are stretch goals. The capstone stands on the simulation.

## 14. Useful commands

```bash
# checks (no GPU, no network)
python tools/gen_spec_views.py --check          # spec and generated views agree
python tools/check_doc_consistency.py           # docs agree with the spec
python tools/check_literals.py                  # no hard-coded constants
.venv/bin/python tools/fetch_ldpc_golden_vectors.py   # once per fresh clone, before pytest
.venv/bin/python -m pytest                      # full test suite

# paper and supplement
python3 deliverables/research-paper/figures/make_figures.py
python3 deliverables/research-paper/supplement/make_tables.py
cd deliverables/research-paper && latexmk -pdf capstone_rp.tex
cd supplement && latexmk -pdf capstone_rp_supplement.tex

# W10 presentation figures
python presentation-results/plot_results.py

# demo (see demo/README.md for provisioning weights first)
./demo/scripts/start.sh
```

`AGENTS.md` lists every other command, including dataset fetching and each phase's verifier.

## 15. Questions you should be able to answer

1. Why does the digital baseline score exactly 10.0% at low SNR?
2. Why is the task-aware digital system's accuracy flat at 82.0%?
3. Why was the classifier fine-tuned on JPEG 2000 images, and what happens to the crossover without it?
4. Why JPEG 2000 and not JPEG?
5. What is the difference between the 1/6 and 1/24 results, and why?
6. Why does the label-transmission bound fail at the same SNR as the task-aware system?
7. Why are the validation results not a test of H1?
8. What would H4 returning "unsupported" mean, given the G-11 diagnostic?
9. What does the PAPR result say, and what doesn't it say?
10. Why would re-running a closed campaign be a problem, even to fix a number?

The answers are in this file and in [`RESULTS.md`](RESULTS.md).

## 16. Glossary

| Term | Meaning |
|---|---|
| DJSCC | Deep joint source–channel coding: neural encoder to channel symbols, neural decoder, trained end to end |
| AWGN | Additive white Gaussian noise channel |
| SNR, Es/N0 | Signal-to-noise ratio, energy per complex symbol over noise density, in dB |
| r, k | Bandwidth ratio and number of complex channel uses per image |
| LDPC | Low-density parity-check code, the 5G NR data channel code |
| BLER | Block error rate: fraction of code blocks that fail to decode |
| MCS | Modulation and coding scheme: a modulation plus a code rate |
| Outage | A packet that fails, or an image that can't be encoded in the budget |
| PAPR | Peak-to-average power ratio of the transmitted symbols |
| Crossover | The SNR where one system's accuracy curve passes the other's |
| Cliff | The sharp drop to outage when a digital link stops decoding |
| Seed pair / seed cell | One (training seed, channel seed) combination; the design uses three |
| W*n* | Project week *n* of the 17-week schedule |
| G-*n* | A go/no-go gate in the schedule (G-12 is the test release) |
| AM-*n* | A numbered amendment to the spec (§17) |
| ER-, BR-, SR-, DR-, HR-, PR- | Requirement IDs: experiment, baseline, system, demo, hardware, programme |
| ER-9 | The task-aware digital control |
| ER-12 | The label-transmission bound |
| ER-2 | SNR-randomized training |
| ER-4 | The reconstruction ablation |
| BR-4 | The baseline's per-SNR operating-point selection |
| BR-9, BR-16 | Fixed-modulation and fixed-MCS baselines |
| BR-12 | Scoring with the artifact fine-tuned classifier |
| DEC-9 | The decision to use JPEG 2000 as the codec of record (JPEG kept as a secondary curve) |

## 17. Where to read more

- [`RESULTS.md`](RESULTS.md): every measured number, with what it means and what not to claim.
- [`novelty-review.md`](novelty-review.md): the novelty claim tested against published work, the one-sentence answer, and the open code-rate decision.
- `presentation-results/report/findings.md`: the W10 findings and figure captions.
- [`crossover-explained.md`](crossover-explained.md): why "the curves must cross" was dropped as a success criterion.
- [`literature-review.md`](literature-review.md): the 30-source review and the gap this project fills.
- `spec/SPEC.md`: the full specification. §1–2 for the thesis and hypotheses, §13 for the schedule, §17 for why things changed.
- `deliverables/research-paper/`: the paper and the supplement with pseudocode and full tables.
- `NEXT.md`: the current working state, and the session log.
