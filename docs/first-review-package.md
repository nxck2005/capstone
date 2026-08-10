# First Review Package

**Review window:** 2026-08-22–26  
**Presentation slot:** 15 minutes  
**Package baseline:** 2026-08-11  
**Normative project source:** [`spec/SPEC.md`](../spec/SPEC.md)

This is the maintained PR-8 First Review package. It is an evidence map and presentation script, not a replacement for the prescribed review template. The companion artifacts are:

- [PR-1 literature review](literature-review.md)
- [PR-2 Gantt plan](gantt-plan.md)
- [PR-3 standards and tools register](standards-and-tools-register.md)
- [PR-9 deployment dossier](deployment-dossier.md)

## 1. Review objective

Demonstrate that the project has a precise research question, a fair and executable comparison protocol, a reproducible implementation foundation, and a gated plan to complete the experiment without adapting the claim to preliminary outcomes.

The requested supervisor/panel decision is:

> Confirm the task boundary and fair-comparison protocol: image-classification accuracy over normalized AWGN at matched complex-symbol budget, comparing adaptive JPEG 2000 + 5G NR LDPC, continuous task-aware DJSCC, and a task-aware digital feature control.

No learned-vs-classical headline result exists at the First Review baseline. Bounded smoke evidence must not be shown as if it were the final experiment.

## 2. Rubric coverage

| First Review criterion | Evidence in package | Presentation location |
|---|---|---|
| Motivation | Short-packet edge/IoT links; task success rather than faithful bit recovery; finite-blocklength cost | Slides 2–3; literature §§2 and 6 |
| Objectives | Build, verify, and compare three systems at identical `k` and SNR; retain paired outcomes and explicit failures | Slides 3 and 6–7 |
| Hypotheses | Preregistered H1–H4; primary positive paired interval at three consecutive low-SNR points; crossover not required | Slide 5; `spec/SPEC.md` §2 |
| Problem survey | Information theory, learned compression, neural JSCC, task-oriented communication, fair baseline gap | Slide 4; literature review |
| Subject knowledge | AWGN normalization, finite blocklength, JPEG 2000, TS 38.212 LDPC, modulation, task-aware attribution | Slides 4, 6–7; standards register |
| Time plan | Gate-ordered critical path through G-8, training, test freeze, reporting, and hardware fallback | Slide 9; Gantt plan |

## 3. Fifteen-minute presentation plan

| Time | Slide | Content | Evidence/visual |
|---:|---:|---|---|
| 0:00–0:30 | 1 | Title and one-sentence thesis | Project title; three-arm comparison diagram |
| 0:30–1:45 | 2 | Motivation | Short packet, fixed bandwidth, downstream classification; separation caveat stated narrowly |
| 1:45–3:00 | 3 | Research question and success contract | Same image/SNR/`k`; completion independent of outcome |
| 3:00–5:00 | 4 | Literature synthesis | Four-column map: finite blocklength, learned compression, DeepJSCC, task-oriented inference |
| 5:00–6:30 | 5 | Objectives and preregistered hypotheses | H1–H4 summary; paired inference; no required crossing |
| 6:30–8:30 | 6 | System architecture | Learned arm, classical arm, shared AWGN, frozen classifier, sealed test boundary |
| 8:30–10:30 | 7 | Fairness and attribution controls | Adaptive modulation/rate/quality; exact overhead; outage denominator; ER-9 digital control |
| 10:30–12:15 | 8 | Evidence achieved | G-1, G-7, G-2, bounded W4 pipeline; explicit “not headline results” banner |
| 12:15–13:45 | 9 | Current state and Gantt | G8_C characterization active; remaining gates; fixed review/report dates |
| 13:45–15:00 | 10 | Standards, deployment, risks, decision request | Exact standards boundary; Tier 1 first; conducted SDR stretch/fallback; requested confirmation |

Keep backup slides outside the 15-minute sequence.

## 4. Slide content contract

### Slide 1 — Thesis

**Title:** Task-Oriented Deep Joint Source–Channel Coding for Bandwidth-Constrained Image Classification

**One sentence:** Transmit the information needed for the classifier, not necessarily every reconstructable source bit, and test the benefit against a tuned separated chain at exactly the same complex-symbol budget.

Do not describe the project as reinforcement learning. It is supervised end-to-end training over a differentiable channel.

### Slide 2 — Motivation

- Edge sensors face bandwidth, latency, and energy constraints.
- Conventional source reconstruction and channel recovery optimize interfaces that are not the final task metric.
- At finite blocklength, practical source/container/coding overhead and block errors matter.
- A classical system can still be task-aware; therefore the experiment needs a digital learned-feature control.

### Slide 3 — Research question

Show the three-arm comparison:

1. JPEG 2000 → LDPC → adaptive modulation → reconstruction → frozen classifier.
2. Quantized learned features → the same digital physical layer → classifier.
3. DJSCC encoder → AWGN → dual-head decoder/classifier.

Invariant comparison axes: same image identity, same split, same `k`, same $E_s/N_0$ definition, keyed noise identity, validation-only tuning, and one sealed test campaign.

### Slide 4 — Literature map

Use one representative citation and one unresolved issue per family:

| Family | Representative result | Issue carried into this project |
|---|---|---|
| Finite blocklength | Rate backs off from capacity; source/channel dispersions interact | Measure practical short-packet loss, do not invoke separation as a finite-length guarantee |
| Learned compression | End-to-end transforms and entropy priors improve rate–distortion | Reconstruction optimization is not task optimization |
| DeepJSCC | Continuous learned mappings show graceful degradation | Most image work scores reconstruction, not classification |
| Task-oriented communication | Learned features can improve inference under link constraints | Separate task awareness from joint coding with a matched digital control |

### Slide 5 — Objectives and hypotheses

Objectives:

- implement a reproducible normalized AWGN and DJSCC pipeline;
- implement a standards-derived, non-strawman JPEG 2000 + 5G NR LDPC baseline;
- tune the classical arm per SNR on validation;
- implement the task-aware digital control;
- evaluate paired image-level outcomes once on the sealed test split; and
- report positive, null, or negative outcomes under the same protocol.

State H1–H4 exactly from the spec on the final slide version. The primary criterion is a paired interval above zero at three consecutive low-SNR points. A curve crossing is descriptive, not required.

### Slide 6 — Architecture and data isolation

Use a simplified architecture diagram from the deployment dossier, but keep Tier 1 central. Show:

- Imagenette-160 headline data;
- canonical preprocessing;
- shared AWGN registry path;
- frozen clean/artifact-tuned classifier variants;
- result row and per-image paired outcome;
- `src/data/test_access.py` as the only guarded test boundary; and
- freeze manifest before G-12.

### Slide 7 — Fairness and attribution

Include these non-negotiable controls:

- JPEG 2000 rather than baseline JPEG as the codec of record;
- codec quality, LDPC rate, and BPSK/QPSK/16-QAM all available to validation tuning;
- exact channel-use and byte accounting, including failure rows;
- BLER table measured for every required physical identity rather than extrapolated from G-2;
- constant-class outage policy measured on validation and retained in the denominator;
- clean and artifact-finetuned classifier passes; and
- ER-9 quantized learned features over the same digital chain.

### Slide 8 — Completed evidence

Observed evidence safe to show:

| Gate/checkpoint | Observed result | Interpretation |
|---|---|---|
| G-1 | Imagenette-160 clean validation top-1 `898/1000 = 89.8%`, floor 88% | Reference task is viable; test remains sealed |
| G-7 | 1.64 M DJSCC parameters; batch 32; 48.68 s/full 8,469-image epoch; 1.004 GiB peak reserved VRAM; 1.35 h projected/100 epochs | Training plan fits the profiled RTX 4060 Laptop GPU |
| G-2 | Golden vectors pass; BPSK/QPSK/16-QAM BLER waterfall displacement within the 0.5 dB rule | Digital physical layer passes its conformance gate |
| W4 bounded integration | JPEG 2000, packetisation, LDPC, AWGN, decode, outage, classifier, records, and verifier execute end to end | Plumbing is integrated; bounded rows are not scientific headline evidence |
| G8_C | Full-strength BLER work-unit campaign active under authenticated resume contracts | Full BR-4 table is not yet frozen; no selection has run |

The slide title must say **Engineering and gate evidence — not learned-vs-classical results**.

### Slide 9 — Plan and status

Show the critical path from the Gantt:

`G8_C → G-8 → training loop → λ calibration/G-4 → final training → ER-9/G-11 → validation rehearsal → freeze/G-12 → one test campaign → G-5 → demo/report`.

State the hard dates: First Review 22–26 Aug, Second Review 29 Sep–3 Oct, Final Review 17–21 Nov, report due 20 Nov. State that W16 is allocated report contingency, not room for new experiment scope.

### Slide 10 — Standards, deployment, and decision

- Claim only TS 38.212-derived LDPC/rate matching over abstract AWGN; no full 5G NR link claim.
- Use OpenJPEG 2.5.4 for JPEG 2000.
- Tier 1 is simulation-first and sufficient.
- Tier 2/3 use conducted SDR replay only after G-5; expected outcome is prerecorded.
- Candidate cost-bounded topology: HackRF One TX + attenuator chain + RTL-SDR RX; no purchase yet.
- Ask panel to confirm the fair-comparison and attribution scope.

## 5. Backup slides

1. Complete SNR grid and why it is dense around three LDPC waterfalls.
2. Complex-symbol budgets for every dataset/ratio.
3. TS 38.212 packetisation and exact bit accounting.
4. G-2 independent BLER reference and source lineage.
5. Dataset provenance, stratified splits, and test-access guard.
6. BR-4 two-pass selection and full-sweep authorization boundary.
7. Statistical plan: paired image outcomes and simultaneous low-SNR criterion.
8. Deployment frame, wrapper overhead, link-budget assumptions, latency/energy estimates.
9. Risk register and gate fallbacks.
10. Detailed Gantt and review dates.

## 6. Required figures and provenance

| Figure | Source | Rule |
|---|---|---|
| Three-arm architecture | Redraw from spec/deployment dossier | Conceptual; no measured numbers |
| Literature synthesis | Literature review references | Cite primary sources on slide |
| G-1 accuracy | `results/reference_classifier/g1_adjudication.json` | Show validation and denominator |
| G-7 profile | `results/profiling/g7_djscc_profile.json` | Show device and measured/projected distinction |
| G-2 waterfall check | `results/baseline/g2/g2_adjudication.json` | Label conformance evidence, not BR-4 table |
| Gantt | `docs/gantt-plan.md` | Render with baseline date |
| Hardware topology | `docs/deployment-dossier.md` | Label provisional and gated |

Do not manually transcribe a metric without its denominator, split, and evidence file. Do not regenerate result figures in presentation software from memory.

## 7. Presenter checklist

Before submission:

- [ ] Use the university-prescribed First Review template if one is supplied.
- [ ] Keep the main sequence within 15 minutes in a timed rehearsal.
- [ ] Put citations on the literature slide and a compact references slide/appendix.
- [ ] Label every metric as measured, projected, bounded smoke, or planned.
- [ ] State that the test split is sealed and no learned-vs-classical result exists yet.
- [ ] State that the baseline adapts per SNR and that ER-9 is the attribution control.
- [ ] State the exact standards boundary and exclusions.
- [ ] Confirm all artifact links and current G8_C status against `instructions/RESUME.md` on the submission day.
- [ ] Export a PDF and retain the editable source.
- [ ] Record supervisor/panel feedback and any accepted objective change through the spec amendment process.

## 8. Anticipated questions

**Why not compare only against JPEG?**  
JPEG container overhead can consume a large fraction of the smallest budget. JPEG 2000 raw codestreams are the preregistered headline codec; JPEG remains a secondary curve.

**Is the baseline deliberately weak?**  
No. It tunes codec quality, LDPC rate, and modulation per SNR on validation, may climb to 16-QAM, and retains every failure in the denominator.

**Does a learned win prove JSCC is responsible?**  
No. The gap conflates task-aware representation and joint coding. ER-9 transmits quantized learned features through the same digital chain to decompose the attribution.

**Why AWGN rather than real radio first?**  
AWGN isolates the scientific mechanism and supports exact paired randomness. SDR effects add synchronization, CFO, clipping, and regulatory variables; they are a gated stretch demonstration.

**What if the hypothesis fails?**  
The project is complete if the preregistered protocol is executed correctly. A null or negative result is reported; it does not trigger retuning on the test split.

**Why no curve-crossing pass criterion?**  
At low bandwidth the learned system could dominate across the grid. Requiring a crossing would perversely treat stronger learned performance as failure. The protocol reports a crossing if observed and tests the paired low-SNR difference directly.
