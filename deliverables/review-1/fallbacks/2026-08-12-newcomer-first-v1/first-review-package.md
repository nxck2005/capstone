# First Review Package

**Review window:** 2026-08-18–22  
**Presentation slot:** 15 minutes  
**Package baseline:** 2026-08-12  
**Normative project source:** [`spec/SPEC.md`](../../spec/SPEC.md)  
**Final package directory:** `deliverables/review-1/`

**Presentation draft (2026-08-12 newcomer-first revision; awaiting author review):**
[editable PPTX](semantic-communication-first-review.pptx) ·
[PDF review copy](semantic-communication-first-review.pdf) ·
[contact sheet](semantic-communication-first-review-contact-sheet.png)

**Revision support:**
[presenter guide](review-1-presenter-guide.md) ·
[iteration notes](ITERATION-NOTES.md) ·
[pre-revision fallback](fallbacks/2026-08-12-pre-knowledge-transfer/README.md)

This is the maintained PR-8 First Review package. It is the delivery contract, evidence map and presentation script, not a replacement for a university-prescribed review template. The companion backing artifacts are:

- [PR-1 literature review](../../docs/literature-review.md)
- [PR-2 Gantt plan](../../docs/gantt-plan.md)
- [PR-3 standards and tools register](../../docs/standards-and-tools-register.md)
- [PR-9 deployment dossier](../../docs/deployment-dossier.md)
- [Guide hardware-alternative acknowledgement status](guide-hardware-alternative-acknowledgement.md)
- [Plain-language project knowledge transfer](../../docs/PROJECT-KNOWLEDGE-TRANSFER.md)

## Delivery acceptance contract

The First Review package is complete only when all of the following are true. This list is user-fixed; agents MUST use it without asking the user to restate or reinterpret it.

1. **PPT:** the main review artifact is a polished approximately 10–12-slide deck that explicitly maps its content to all six First Review rubric categories: Motivation, Objectives, Hypothesis, Problem Survey, Subject Knowledge and Time Plan. A generic project presentation does not pass.
2. **Literature:** the backing problem survey contains at least 25 credible references, synthesizes what the literature establishes and leaves unresolved, and is available for citations and viva questions. A bibliography dump does not pass.
3. **Plan:** the Gantt chart uses the current circular's dates, shows realistic remaining work and gates, and does not imply that unfinished later experimental stages are complete.
4. **Technical readiness:** all four team members can independently explain the three-arm architecture, channel and bandwidth-matching methodology, fairness controls, data isolation, hypothesis protocol and current implementation boundary. Agents may prepare notes and viva questions, but repository artifacts cannot substitute for the four members' understanding.
5. **Evidence:** G-1/reference-classifier evidence and the existing valid G-2, G-7, W4 and current G8_C implementation evidence are available and correctly labelled for viva. G8_C completion is not a First Review prerequisite; no experiment is rerun, weakened or reframed merely to produce more graphs, and bounded smoke is never presented as a headline comparison.
6. **Deployment:** the deployment dossier is preparation. The guide's real acknowledgement of the simulation-first Tier-1 path with no required hardware implementation must be obtained and recorded with its date. Agents may prepare the exact request and status record but MUST NOT fabricate, infer or mark the guide's response as recorded. Tier 2/3 remain gated stretch goals.
7. **Protocol housekeeping:** the deck uses the exact corrected H1 decision rule from `spec/SPEC.md`, uses the 18–22 August 2026 First Review window, and states completion objectives rather than promising a positive outcome. Current corrections preserve the authenticated provenance record; history is never rewritten to make it look cleaner.
8. **Frozen delivery:** the editable deck, exported PDF and supporting package are committed under `deliverables/review-1/`, then the annotated `review-1-basis` snapshot tag is cut from that final review basis.

## 1. Review objective

Demonstrate that the project has a precise research question, a fair and executable comparison protocol, a reproducible implementation foundation, and a gated plan to complete the experiment without adapting the claim to preliminary outcomes.

The requested supervisor/panel decision is:

> Confirm the task boundary and fair-comparison protocol: image-classification accuracy over normalized AWGN at matched complex-symbol budget, comparing adaptive JPEG 2000 + 5G NR LDPC, continuous task-aware DJSCC, and a task-aware digital feature control.

No learned-vs-classical headline result exists at the First Review baseline. Bounded smoke evidence must not be shown as if it were the final experiment.

Review 1 presents the valid state reached by the review date. It does not manufacture later-stage results to make the project appear further advanced.

## 2. Rubric coverage

| First Review criterion | Evidence in package | Presentation location |
|---|---|---|
| Motivation | Limited noisy links, remote classification, ordinary image transmission and its task-level limitation | Slides 2–4; literature §§2 and 6 |
| Objectives | Build, verify and compare three systems at identical `k` and SNR; retain paired outcomes and explicit failures | Slides 1–2, 5, 8 and 11 |
| Hypotheses | Preregistered H1–H4; for H1 a point qualifies when the studentized paired mean exceeds 1.96, and support requires both a run of at least three consecutive qualifying points at or below the training SNR and the calibrated run p-value ≤ 0.05; crossover not required | Slides 8 and 12; `spec/SPEC.md` §2 |
| Problem survey | Information theory, learned compression, neural JSCC, task-oriented communication and the attribution gap | Slide 6; literature review |
| Subject knowledge | Compression, LDPC, modulation, AWGN, DJSCC, resource matching, data isolation and evidence boundaries | Slides 3–7, 9 and 11–12; standards register |
| Time plan | Gate-ordered critical path through G-8, training, test freeze, reporting and hardware fallback | Slide 10; Gantt plan |

## 3. Narrative contract for this and future iterations

The presentation must work for a panel member who knows neither communications nor machine learning. Use this order unless the author explicitly changes it:

1. state the real-world task;
2. explain the conventional solution in plain language;
3. explain its limitation relative to the task metric;
4. introduce the learned change;
5. explain why three systems are required for a fair scientific claim;
6. establish the literature gap, methodology and hypotheses;
7. show evidence, unfinished work and the requested decision.

Do not begin with acronyms, gate names, equations or repository machinery. Introduce a term only when the audience needs it. Prefer short, literal prose. Do not use aphorisms, slogans or decorative themes. The visual default is black text on white, light rules, simple tables and only the diagrams needed to explain a flow.

This is a knowledge-transfer rule, not a reduction in technical substance. The deck must still expose all six rubric criteria, the three-system attribution design, the fairness controls, the exact corrected H1 rule, evidence boundaries and the honest remaining plan.

## 4. Fifteen-minute presentation plan

| Time | Slide | Audience question answered | Content |
|---:|---:|---|---|
| 0:00–0:35 | 1 | What is the project? | Formal title and one plain-language scope statement |
| 0:35–1:45 | 2 | What happens from camera to decision? | Observe → send → decide → compare; definition of semantic communication |
| 1:45–3:05 | 3 | How is this normally done? | Compression, error correction, modulation, noisy channel and classifier |
| 3:05–4:20 | 4 | What changes in the learned system? | Joint encoder/decoder training; fixed deployment split; not RL or an LLM |
| 4:20–5:35 | 5 | Why are there three systems? | Classical image link, digital feature control and learned joint link |
| 5:35–7:00 | 6 | What does prior work leave unresolved? | Four literature families and the attribution gap |
| 7:00–8:30 | 7 | How is the comparison fair? | Shared image, budget, SNR/noise, tuning boundary, failures and accounting |
| 8:30–9:50 | 8 | What will be completed and tested? | Objectives, H1–H4 in plain language and outcome-independent completion |
| 9:50–11:25 | 9 | What is already demonstrated? | G-1, G-7, G-2, W4 and paused G8_C evidence with explicit boundaries |
| 11:25–12:45 | 10 | What remains? | Gate-ordered plan and hard review/report dates |
| 12:45–14:10 | 11 | What should the panel confirm? | Research, standards and deployment boundaries; guide item |
| 14:10–15:00 | 12 | Where is the formal rule and rubric coverage? | Exact corrected H1 wording and explicit six-criterion map |

## 5. Slide content contract

### Slides 1–2 — Establish the task

Use the formal title, then explain the project as a four-step process: an edge device observes an image, sends through a limited noisy link, the receiver predicts a class, and the systems are compared at the same channel budget. Define semantic communication as preserving the information needed for the task rather than requiring every original pixel.

### Slide 3 — Explain the conventional approach first

Show the ordinary path before criticizing it:

`image → JPEG 2000 compression → LDPC and modulation → noisy channel → reconstruction → classifier`.

Define compression, error correction, modulation and classifier in one sentence each. Then state the narrow limitation: these stages mainly optimize image and bit recovery, while this project measures classification accuracy after transmission.

### Slide 4 — Introduce DJSCC in context

Show:

`image → neural encoder → noisy channel → neural decoder/classifier → class`.

Explain that the sender and receiver train together through the channel. State that the sender does not simply transmit a class label because the deployment split fixes an encoder at the sender and the decoder/classifier at the receiver. State that the method is supervised end-to-end learning, not reinforcement learning or an LLM.

### Slide 5 — Establish the three-system attribution design

Explain the purpose of each arm, not only its components:

1. the classical image link measures a strong conventional separated system;
2. the digital feature link measures task-aware representation over the same digital physical layer; and
3. the learned joint link measures the combined task-aware and joint-coding method.

The research question compares systems at identical bandwidth and channel conditions and asks both whether a performance difference exists and whether it is specifically consistent with joint coding.

### Slide 6 — Synthesize the literature

For finite-blocklength communication, learned compression, DeepJSCC and task-oriented communication, state one established result and one unresolved issue. The carried gap is a fair three-way image-classification comparison that retains failures and separates task awareness from joint coding.

### Slide 7 — Explain fairness as cause and effect

Each control must state what is held equal and why:

- same image and split prevent sample imbalance;
- same complex-symbol budget matches bandwidth;
- same SNR definition and keyed noise pair the channel condition;
- validation-only tuning protects the sealed test split;
- failed transmissions remain in the denominator; and
- complete byte and physical-layer identity prevent hidden overhead or missing BLER evidence from helping a system.

### Slide 8 — State completion objectives before statistical machinery

The project will build all three systems, match their resources, freeze choices, run one paired test campaign and report the result either way. State H1–H4 as questions in plain language. Point to slide 12 for the exact H1 rule.

### Slide 9 — Translate every evidence item

Show the evidence and its meaning, then state what it does not prove. The current G8_C wording must be refreshed from `instructions/RESUME.md` on submission day without copying volatile counts into this maintained contract. The slide must state that no worker is running and no selection, training, validation decoding or test access has occurred when that remains true.

### Slide 10 — Show unfinished work honestly

Use the critical path:

`G8_C → G-8 → DJSCC training and calibration → ER-9/G-11 → validation rehearsal → freeze/G-12 → one test campaign → demo/report`.

State the hard dates: First Review 18–22 Aug, Second Review 29 Sep–3 Oct, Final Review 17–21 Nov and report due 20 Nov. W16 is report contingency, not new experiment scope.

### Slide 11 — End with scope and the requested decision

Claim only TS 38.212-derived LDPC/rate matching over abstract AWGN, not a complete 5G NR link. Tier 1 is simulation-first and sufficient. SDR replay and edge deployment remain optional stretch work. Ask the panel to confirm the task boundary, three-system comparison, fairness controls and simulation-first completion path. Keep the real guide acknowledgement visibly pending until recorded.

### Slide 12 — Preserve the formal contract

Include the exact corrected H1 decision rule: a point qualifies when the studentized paired mean exceeds 1.96; `R_obs` is the longest consecutive qualifying run at or below the training SNR; H1 is supported only if `R_obs ≥ 3` and the calibrated run p-value is at most 0.05. State that the whole-region mean paired difference is the effect size of record and that a crossover is descriptive, not required. Map all six rubric criteria to slide numbers explicitly.

## 6. Backup slides

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

## 7. Required figures and provenance

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

## 8. Presenter checklist

Before submission:

- [ ] Use the university-prescribed First Review template if one is supplied.
- [ ] Keep the polished main sequence to approximately 10–12 slides and within 15 minutes in a timed rehearsal.
- [ ] Include an explicit slide-to-criterion map for all six First Review rubric categories; do not rely on an implicit generic-project narrative.
- [ ] Put citations on the literature slide and a compact references slide/appendix backed by the ≥25-reference synthesized review.
- [ ] Have all four members rehearse the architecture, methodology, fairness controls, H1 rule, evidence boundaries and current implementation status; record team confirmation only after that human check occurs.
- [ ] Label every metric as measured, projected, bounded smoke, or planned.
- [ ] State that the test split is sealed and no learned-vs-classical result exists yet.
- [ ] State that the baseline adapts per SNR and that ER-9 is the attribution control.
- [ ] State the exact standards boundary and exclusions.
- [ ] Obtain and record the guide's dated acknowledgement of the simulation-first Tier-1/no-required-hardware path.
- [ ] Confirm all artifact links and current G8_C status against `instructions/RESUME.md` on the submission day.
- [ ] Put the editable deck, exported PDF and supporting package under `deliverables/review-1/`.
- [ ] Cut annotated tag `review-1-basis` only from the final package basis.
- [ ] Record supervisor/panel feedback and any accepted objective change through the spec amendment process.

## 9. Anticipated questions

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

## 10. First Review scope boundary

The First Review requires a defensible proposal, technical foundation, current valid evidence and realistic plan. It does **not** require any of the following, and agents MUST NOT add them as readiness gates:

- completion of G8_C full-strength characterization;
- a final `BlerTable`;
- neural-model training;
- learned-versus-classical results;
- a complete demo;
- thesis chapters;
- a paper draft;
- a poster;
- a plagiarism report;
- hardware purchase; or
- SDR implementation.

No scientific run is repeated, weakened, bypassed or reinterpreted merely to create Review 1 graphs. No provenance history is rewritten to make the timeline look cleaner. The deck reports unfinished stages as unfinished and uses only valid evidence already available at the frozen review basis.

## 11. Review-1 readiness matrix

This matrix is the final gate and remains evidence-based. A row becomes `PASS` only after the actual deck or human rehearsal demonstrates it; backing material alone is not enough.

| Rubric criterion | Slide(s) | Supporting artifact | Status |
|---|---:|---|---|
| Motivation | 2–4 | — | PENDING — newcomer-first draft present; awaiting author review |
| Objectives | 1–2, 5, 8, 11 | `spec/SPEC.md`; this review package | PENDING — newcomer-first draft present; awaiting author review |
| Hypothesis | 8, 12 | `spec/SPEC.md` §2 preregistration | PENDING — exact rule is present; awaiting author review |
| Problem Survey | 6 | `docs/literature-review.md` (30 synthesized references) | PENDING — backing artifact passes; draft PPT awaits author review |
| Subject Knowledge | 3–7, 9, 11–12 | Architecture notes; G-1/G-2/G-7/W4/current G8_C evidence | PENDING — draft PPT present; requires four-member human confirmation |
| Time Plan | 10 | `docs/gantt-plan.md` | PENDING — corrected backing artifact passes; draft PPT awaits author review |

Separate final checks:

| Gate | Status |
|---|---|
| Guide hardware-alternative acknowledgement | **PENDING** — only a real, dated guide response can change this to `RECORDED` |
| Review snapshot | **PENDING** — `review-1-basis` must exist, be annotated and point to the final package basis |
| Package-of-record | **PENDING** — editable PPTX and PDF draft are present; final author approval and frozen package remain pending |
