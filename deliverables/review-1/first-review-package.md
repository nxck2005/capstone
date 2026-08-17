# First Review Package

**Review window:** 2026-08-18–22  
**Presentation slot:** 15 minutes  
**Package baseline:** 2026-08-12  
**Normative project source:** [`spec/SPEC.md`](../../spec/SPEC.md)  
**Final package directory:** `deliverables/review-1/`

**Presentation draft (2026-08-12 proposal-flow revision; awaiting author review):**
[editable PPTX](semantic-communication-first-review.pptx) ·
[PDF review copy](semantic-communication-first-review.pdf) ·
[contact sheet](semantic-communication-first-review-contact-sheet.png)

**Academic companion:**
[editable PPTX](semantic-communication-first-review-academic-v2.pptx) ·
[PDF review copy](semantic-communication-first-review-academic-v2.pdf) ·
[contact sheet](semantic-communication-first-review-academic-v2-contact-sheet.png)

**Revision support:**
[presenter guide](review-1-presenter-guide.md) ·
[iteration notes](ITERATION-NOTES.md) ·
[first newcomer version](fallbacks/2026-08-12-newcomer-first-v1/README.md) ·
[pre-knowledge-transfer fallback](fallbacks/2026-08-12-pre-knowledge-transfer/README.md)

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
5. **Evidence:** G-1/reference-classifier evidence and the existing valid G-2, G-7, W4, G8_C and G8_D implementation/handoff evidence are available and correctly labelled for viva. G8_C/G8_D completion is not a First Review prerequisite; no experiment is rerun, weakened or reframed merely to produce more graphs, and bounded smoke is never presented as a headline comparison.
6. **Deployment:** the deployment dossier is preparation. The guide's real acknowledgement of the simulation-first Tier-1 path with no required hardware implementation must be obtained and recorded with its date. Agents may prepare the exact request and status record but MUST NOT fabricate, infer or mark the guide's response as recorded. Tier 2/3 remain gated stretch goals.
7. **Protocol housekeeping:** the deck uses the exact corrected H1 decision rule from `spec/SPEC.md`, uses the 18–22 August 2026 First Review window, and states completion objectives rather than promising a positive outcome. Current corrections preserve the authenticated provenance record; history is never rewritten to make it look cleaner.
8. **Frozen delivery:** the editable deck, exported PDF and supporting package are committed under `deliverables/review-1/`, then the annotated `review-1-basis` snapshot tag is cut from that final review basis.

## 1. Review objective

Demonstrate that the project has a precise research question, a fair and executable comparison protocol, a reproducible implementation foundation, and a gated plan to complete the experiment without adapting the claim to preliminary outcomes.

The presentation should establish:

> A supervised AI/ML method and fair-comparison protocol: learn a channel-robust neural representation for edge inference over normalized AWGN, then measure image-classification accuracy at matched complex-symbol budget against adaptive JPEG 2000 + 5G NR LDPC and a task-aware digital feature control.

No learned-vs-classical headline result exists at the First Review baseline. Bounded smoke evidence must not be shown as if it were the final experiment.

Review 1 presents the valid state reached by the review date. It does not manufacture later-stage results to make the project appear further advanced.

## 2. Rubric coverage

| First Review criterion | Evidence in package | Presentation location |
|---|---|---|
| Motivation | Limited noisy links, edge AI inference, ordinary image transmission and its task-level limitation | Slides 1–4, 6 and 11 |
| Objectives | Build, verify and compare three systems at identical `k` and SNR; retain paired outcomes and explicit failures | Slides 6–7 and 11–12 |
| Hypotheses | Preregistered H1–H4; for H1 a point qualifies when the studentized paired mean exceeds 1.96, and support requires both a run of at least three consecutive qualifying points at or below the training SNR and the calibrated run p-value ≤ 0.05; crossover not required | Slide 8; `spec/SPEC.md` §2 |
| Problem survey | Information theory, learned compression, neural JSCC, task-oriented communication and the attribution gap | Slide 5; literature review |
| Subject knowledge | Neural representation learning, residual CNN, dual-head decoder, CE + λMSE, differentiable AWGN, ResNet-18, digital feature attribution, LDPC, resource matching and evidence boundaries | Slides 1–5, 7–9 and 11; architecture notes and standards register |
| Time plan | Gate-ordered critical path through G-8, training, test freeze, reporting and hardware fallback | Slide 10; Gantt plan |

## 3. Narrative contract for this and future iterations

The presentation must work for a panel member who knows neither communications nor machine learning. Use this order unless the author explicitly changes it:

1. state the real-world task;
2. explain the conventional solution in plain language;
3. explain its limitation relative to the task metric;
4. introduce the learned change;
5. establish what prior work already shows and the remaining gap;
6. state the problem, research question and completion objectives;
7. present the three-system methodology and fairness controls;
8. state the hypotheses and evaluation plan;
9. show feasibility evidence, unfinished work, application and risks; and
10. close with the project summary, next steps and rubric map.

Do not begin with acronyms, gate names, equations or repository machinery. Introduce a term only when the audience needs it. Prefer short, literal prose. Do not use aphorisms, slogans or decorative themes. The visual default is black text on white, light rules, simple tables and only the diagrams needed to explain a flow.

Do not front-load AI/ML buzzwords merely to signal the specialization. Slides
1–2 establish the ordinary camera–link–receiver problem. Slide 3 explains the
conventional solution. Slide 4 introduces the learned sender and receiver,
then names DJSCC and supervised end-to-end learning. The methodology section
provides the residual CNN, dual-head decoder, differentiable AWGN and `CE +
λMSE` details. This order keeps the AIML contribution clear without sacrificing
knowledge transfer.

This is a knowledge-transfer rule, not a reduction in technical substance. The deck must still expose all six rubric criteria, the three-system attribution design, the fairness controls, the exact corrected H1 rule, evidence boundaries and the honest remaining plan.

## 4. Fifteen-minute presentation plan

| Time | Slide | Audience question answered | Content |
|---:|---:|---|---|
| 0:00–0:30 | 1 | What is the project? | A camera, a limited noisy link and a remote classification task |
| 0:30–1:35 | 2 | What problem are we solving? | Observe → communicate → classify; why sending every image detail may be unnecessary |
| 1:35–2:50 | 3 | How is this normally done? | Compression, error correction, modulation, noisy channel and classifier |
| 2:50–4:05 | 4 | What changes in the learned system? | Joint encoder/decoder training; fixed deployment split; not RL or an LLM |
| 4:05–5:25 | 5 | What does prior work leave unresolved? | Four literature families and the attribution gap |
| 5:25–6:45 | 6 | What problem will the project solve? | Problem statement, research question and completion objectives |
| 6:45–8:35 | 7 | How will the project answer it? | Three-system methodology, shared inputs and evaluation controls |
| 8:35–10:10 | 8 | What exactly will be tested? | Exact H1 rule, H2–H4 and outcome-independent reporting |
| 10:10–11:35 | 9 | Is the proposed work feasible? | G-1, G-7, G-2, W4, complete G8_C and complete G8_D tooling evidence with explicit boundaries |
| 11:35–12:45 | 10 | What remains? | Gate-ordered plan and hard review/report dates |
| 12:45–13:50 | 11 | Is the scope practical and applicable? | Real-world application, required scope, risks, standards and guide item |
| 13:50–15:00 | 12 | What should the audience retain? | Project summary, immediate next steps and explicit six-criterion map |

## 5. Slide content contract

### Slides 1–2 — Establish the task

Use the formal title, then explain the setting without model terminology: an edge camera sees an image, a limited noisy link separates it from the receiver, and the receiver must classify the image. Ask whether every image detail must be sent. Do not introduce residual CNNs, differentiable channels or the loss function yet.

### Slide 3 — Explain the conventional approach first

Show the ordinary path before criticizing it:

`image → JPEG 2000 compression → LDPC and modulation → noisy channel → reconstruction → classifier`.

Define compression, error correction, modulation and classifier in one sentence each. Then state the narrow limitation: these stages mainly optimize image and bit recovery, while this project measures classification accuracy after transmission.

### Slide 4 — Introduce DJSCC in context

First show the plain flow:

`image → learned sender → noisy link → learned receiver → class`.

Then give its technical form:

`image → residual CNN encoder → differentiable AWGN → dual-head decoder → class logits + reconstruction`.

Explain that the sender and receiver train together through the channel. State that the sender does not simply transmit a class label because the deployment split fixes an encoder at the sender and the decoder/classifier at the receiver. State that the method is supervised end-to-end learning, not reinforcement learning or an LLM.

Name the multi-task objective: cross-entropy for the class head plus `λ × MSE`
for reconstruction. The architecture and loss are the clearest immediate
evidence that this is an AIML project rather than a classification-only topic.

### Slide 5 — Synthesize the literature

For finite-blocklength communication, learned compression, DeepJSCC and task-oriented communication, state one established result and one unresolved issue. The carried gap is a fair three-way image-classification comparison that retains failures and separates task awareness from joint coding.

### Slide 6 — State the problem, research question and objectives

State the gap as a problem the project can solve. Ask whether neural DJSCC learns channel-robust representations that preserve AI inference accuracy under matched bandwidth and channel conditions, and whether any remaining difference is attributable to joint coding. Objectives must be build-and-evaluate commitments, not promised positive outcomes.

### Slide 7 — Present the proposed methodology

Show all three arms and their scientific roles:

1. the classical image link is the strong conventional reference;
2. the digital feature link separates task awareness from joint coding; and
3. the learned joint link tests end-to-end joint coding.

Place the shared inputs above the table and the evaluation controls below it:

- same image and split prevent sample imbalance;
- same complex-symbol budget matches bandwidth;
- same SNR definition and keyed noise pair the channel condition;
- validation-only tuning protects the sealed test split;
- failed transmissions remain in the denominator; and
- complete byte and physical-layer identity prevent hidden overhead or missing BLER evidence from helping a system.

### Slide 8 — State the hypotheses and evaluation plan

Include the exact corrected H1 decision rule here: a point qualifies when the studentized paired mean exceeds 1.96; `R_obs` is the longest consecutive qualifying run at or below the training SNR; H1 is supported only if `R_obs ≥ 3` and the calibrated run p-value is at most 0.05. State that the whole-region mean paired difference is the effect size of record and that a crossover is descriptive, not required. Summarize H2–H4 in plain language and state that support, no support and adverse outcomes are all valid findings.

### Slide 9 — Translate every evidence item

Show the evidence and its meaning, then state what it does not prove. The current G8_C/G8_D wording must be checked against `instructions/RESUME.md` and the D7 handoff before submission without copying volatile work-unit details into this maintained contract. The slide must state that G8_E/full validation, pass one, training, validation decoding for the full campaign and test access have not occurred when that remains true.

### Slide 10 — Show unfinished work honestly

Use a graphical Gantt with calendar-scaled bars from August through November.
It must show G8_C BLER characterization and G8_D validation-measurement tooling
as complete, followed by planned G8_E full validation measurement/pass-one
selection, ratio selection, DJSCC training and calibration, digital control and
validation, freeze plus the one test campaign, demo/optional replay, and
report/final-review work. Mark the fixed review and report dates and show W16
as report contingency. Do not show Review 1 document preparation as a second
active experiment bar.

The bar order must preserve the critical path:

`G8_C → G8_D tooling → G8_E validation/pass one → G-8 ratio decision → DJSCC training and calibration → ER-9/G-11 → validation rehearsal → freeze/G-12 → one test campaign → demo/report`.

State the hard dates: First Review 18–22 Aug, Second Review 29 Sep–3 Oct, Final Review 17–21 Nov and report due 20 Nov. W16 is report contingency, not new experiment scope.

### Slide 11 — Show application, scope and risks

Explain the remote-camera and split edge/cloud inference application. Name the AI/ML contribution: channel-robust representation learning, residual encoder, dual-head decoder, supervised multi-task loss, and the digital feature attribution control. State that simulation-first Tier 1 is the required scope and that SDR/edge deployment is optional stretch work. Show the main risks and controls: authenticated resumption for incomplete BLER work, outcome-independent reporting for an unsupported hypothesis, and a gated hardware path to avoid radio confounders. Claim only TS 38.212-derived LDPC/rate matching over abstract AWGN, not a complete 5G NR link. Keep the real guide acknowledgement visibly pending until recorded.

### Slide 12 — Summarize and state next steps

Summarize the problem, proposal, current readiness and next work. State that the research question, three-system methodology, feasibility evidence and completion plan are defined, while the final learned-versus-classical result remains future work. Map all six rubric criteria to slide numbers explicitly.

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
- [ ] Confirm all artifact links and current G8_C/G8_D status against `instructions/RESUME.md` and the D7 handoff on the submission day.
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

- G8_C full-strength characterization or the final `BlerTable` (both are now
  available, but neither is a First Review gate);
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

This matrix separates artifact readiness from human confirmation. `CONTENT READY`
means the required material exists in the active deck and is backed by a named
source. It does not mean the author has approved the wording or that all four
members have demonstrated viva readiness.

**Current result:** rubric content is present for all six criteria. Final Review
1 acceptance remains **HOLD** on the human and package-freeze gates below.

### 11.1 Rubric-content matrix

| Rubric criterion | What the examiner is checking | Slide(s) | Backing evidence | Artifact state | Remaining confirmation |
|---|---|---:|---|---|---|
| Motivation | Importance of the problem and a genuine domain-based reason for choosing it | 2–4, 6, 11 | Project premise; deployment scenario; literature motivation | **CONTENT READY** | Author confirms the framing reflects the team's real motivation |
| Objectives | The problems to be solved are identified as achievable project work | 6–7, 11–12 | `spec/SPEC.md` §2; completion criterion; scope boundary | **CONTENT READY** | Author approves the objective wording before it becomes the Review 1 baseline |
| Hypothesis | A proposal exists for testing whether the objectives are achieved | 8 | Exact preregistered H1–H4 protocol in `spec/SPEC.md` §2 | **CONTENT READY** | All four members can explain H1 in plain language and state why a crossover is not required |
| Problem Survey | Prior work has been reviewed and the proposal follows from a defensible gap | 5 | `docs/literature-review.md`: 30 synthesized references | **CONTENT READY** | Team can explain the four literature families and the specific three-system gap without reading the slide |
| Subject Knowledge | The team understands the AI/ML, communications and experimental-design choices | 1–5, 7–9, 11 | Residual CNN encoder; dual-head decoder; CE + λMSE; differentiable AWGN; ResNet-18; digital feature attribution; standards register; G-1/G-2/G-7/W4, G8_C and G8_D evidence | **MATERIAL READY** | **HUMAN PENDING:** all four members independently explain the AI model, training objective, three-arm attribution and evidence boundary |
| Time Plan | The remaining work is practical, ordered and shown in a Gantt chart | 10 | `docs/gantt-plan.md`; fixed review dates and gate dependencies | **CONTENT READY** | Author confirms the workload and ownership plan; team can explain the critical path |

### 11.2 Package and human gates

| Gate | Owner | Evidence required to close | Current state |
|---|---|---|---|
| Author deck review | Project author | Explicit approval of the active PPTX/PDF content or a completed correction list | **AWAITING REVIEW** |
| Four-member technical readiness | Full team | Each member can independently explain the task, three systems, methodology, H1, current evidence and remaining plan | **HUMAN PENDING** |
| Guide hardware-alternative acknowledgement | Author and guide | Real response recorded with date, channel, guide identity and evidence location | **HUMAN PENDING** — the prepared record remains `PENDING` |
| Package-of-record freeze | Author, assisted by agent | Approved PPTX, PDF, literature review, Gantt, standards register, deployment dossier and acknowledgement are all under the final package basis | **FREEZE PENDING** |
| Annotated review snapshot | Author, assisted by agent | Annotated `review-1-basis` tag points to the approved package basis | **FREEZE PENDING** — do this only after every preceding gate closes |

### 11.3 Overall interpretation

- Nothing in the six rubric categories is currently missing as a repository
  artifact.
- Subject-knowledge readiness cannot be inferred from documents; it requires
  the four-member rehearsal.
- The guide acknowledgement cannot be inferred from silence or from the
  deployment dossier.
- The active deck remains a draft until the author finishes the review now in
  progress.
- G8_C/G8_D completion, the full validation campaign, learned training and
  comparative results are not First Review prerequisites and do not appear as
  blockers in this matrix.
