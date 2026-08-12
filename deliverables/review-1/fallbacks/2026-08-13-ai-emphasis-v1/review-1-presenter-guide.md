# Review 1 presenter guide

This guide is for a team member who understands the project but needs a clear
15-minute explanation. It follows the active
[`Review 1 deck`](semantic-communication-first-review.pptx). It is not a script
that must be memorized word for word.

The audience should leave with six facts:

1. This is an AI/ML project about neural representation learning for edge AI
   over a limited noisy link.
2. Image classification is the downstream evaluation task, not the complete
   project identity.
3. Semantic communication preserves information needed for the task rather
   than requiring perfect reconstruction of every pixel.
4. The experiment compares three systems because a two-system comparison could
   not separate task-aware representation from joint coding.
5. The comparison is controlled, preregistered and not yet using the test set.
6. Review 1 presents a sound question, method, foundation and plan. It does not
   claim the final learned-versus-classical result.

## General delivery rules

- Explain one new idea at a time.
- Do not read tables cell by cell. State the conclusion first, then point to
  the supporting row.
- Expand an acronym on first use. After that, use the acronym consistently.
- When a panel member looks uncertain, return to “camera → link → receiver →
  class decision.”
- Say whether a number is measured, projected, bounded integration evidence or
  planned.
- Do not imply that a positive hypothesis result is required for completion.

## Slide 1 — Project title

Purpose: name the project and define its boundary without beginning with an
equation.

Suggested explanation:

> This is an AI/ML project for edge inference over a limited noisy link. We
> train a neural encoder and dual-head decoder end to end through a
> differentiable channel. Image classification is the task we use to measure
> whether the learned representation preserves useful meaning.

Transition: “I will first explain the complete task without using the model
details.”

## Slide 2 — Edge AI inference over a noisy link

Purpose: give the audience the whole mental model.

Suggested explanation:

> An edge camera observes an image. A neural encoder learns a compact
> representation, the representation crosses a fixed-bandwidth noisy link,
> and the receiver AI predicts a class. We compare systems by the accuracy of
> that inference at the same channel budget.

Then state the identity rule directly: classification is the measurable AI
task; channel-robust representation learning is the method. Image quality can
still be measured, but task accuracy is the headline result.

Transition: “Before explaining the learned method, this is how the same task is
normally solved.”

## Slide 3 — Conventional image link

Purpose: teach the baseline before comparing against it.

Walk through the flow once from left to right:

- JPEG 2000 reduces the image data.
- LDPC adds redundancy that can repair damaged bits.
- Modulation turns bits into channel symbols.
- The receiver reconstructs the image and runs the frozen ResNet-18 AI model.

Do not call this system weak or obsolete. It is deliberately tuned as a strong
baseline. Its narrow limitation is that its stages mainly optimize image and
bit recovery while the experiment evaluates the final class decision.

Transition: “The learned system changes where that optimization happens.”

## Slide 4 — Learned joint system

Purpose: explain DJSCC only after the ordinary path is clear.

Suggested explanation:

> In deep joint source–channel coding, the encoder at the sender and the
> dual-head decoder at the receiver are trained together through a
> differentiable noisy channel. Supervised backpropagation therefore learns a
> transmitted representation that can preserve inference information after
> noise.

Name the implementation plainly: residual CNN encoder, GroupNorm/PReLU,
dual-head decoder, and the multi-task objective `cross-entropy + λ × MSE`.

If asked why the sender does not transmit a class label, explain the fixed
deployment split: the sender owns an encoder; the receiver owns the decoder and
classifier. Moving the whole classifier to the sender would answer a different
engineering question.

State plainly that this is supervised end-to-end learning, not reinforcement
learning and not an LLM system.

Transition: “A classical-versus-learned comparison alone would still leave one
important ambiguity.”

## Slide 5 — Literature synthesis

Purpose: show synthesis rather than reciting thirty citations.

Use one sentence per family:

- Finite-blocklength work explains why short practical links pay overhead and
  can fail.
- Learned compression improves reconstruction trade-offs, not necessarily task
  accuracy.
- DeepJSCC shows continuous learned transmission and graceful degradation, but
  most image work emphasizes reconstruction.
- Task-oriented work studies useful features, but often does not isolate task
  awareness from joint coding.

Conclude with the gap in the last row. The backing literature review contains
the full 30-source synthesis.

Transition: “That gap gives us a specific problem statement and a set of
completion objectives.”

## Slide 6 — Problem statement and objectives

Purpose: state exactly what the project will solve before presenting the system
design.

Explain the three levels in order:

1. The problem is that the communication stack is optimized separately from
   the receiver-side AI objective.
2. The research question asks whether neural DJSCC learns a channel-robust
   representation that preserves inference at the same resources.
3. The objectives commit the team to train the neural model, build the shared
   AI feature control, match resources, freeze choices and evaluate once.

Emphasize that the objectives describe work to complete. They do not promise a
positive result.

Transition: “This is the proposed experiment for answering that question.”

## Slide 7 — Proposed methodology

Purpose: present the system-level design expected in a preliminary review.

Start with the shared-input strip. Every arm receives the same image, split,
symbol budget, SNR definition and keyed channel realization.

Then explain the three rows by their scientific roles:

- The classical image link ends in the frozen ResNet-18 and is the conventional
  reference.
- The digital AI feature control shares the learned encoder and task head but
  retains a conventional digital channel chain.
- The neural DJSCC arm trains the channel-facing representation end to end.

The third arm matters because an improvement over classical image transmission
could otherwise come from task-aware representation rather than joint coding.

Finish with the evaluation controls. The test split remains sealed until all
choices are frozen at gate G-12.

Transition: “The hypotheses and their decision rules were fixed before that
test campaign.”

## Slide 8 — Hypotheses and evaluation plan

Purpose: satisfy the hypothesis criterion and show that the result will not be
chosen after seeing the data.

For H1, preserve these facts exactly:

- the comparator is learned versus `classical_adaptive` at the headline ratio;
- a point qualifies when the studentized paired mean exceeds 1.96;
- `R_obs` is the longest consecutive qualifying run at or below training SNR;
- H1 is supported only when `R_obs ≥ 3` and the calibrated run p-value is at
  most 0.05;
- the whole low-SNR-region mean paired difference is the effect size of record;
  and
- a curve crossing is reported if observed but is not required.

Then summarize H2–H4 from the right-hand column. End with the outcome rule:
support, no support and adverse findings are all reported under the same
protocol.

Transition: “The final comparison has not run, but the main feasibility risks
have already been tested.”

## Slide 9 — Current evidence

Purpose: distinguish readiness evidence from final scientific results.

Translate each row:

- G-1 shows that a ResNet-18 trained from scratch reaches 89.8% validation
  accuracy, so the clean AI task is viable.
- G-7 shows that the 1.64-million-parameter residual DJSCC model fits the
  available RTX 4060 profile.
- G-2 shows digital physical-layer conformance for its bounded gate.
- W4 shows that the conventional pipeline and evidence path work end to end.
- G8_C is the current full-strength BLER characterization phase and is paused
  at a durable checkpoint.

End with the boundary printed below the table. Do not show or quote volatile
G8_C work-unit counts. Confirm the exact live wording from
`instructions/RESUME.md` on submission day.

Transition: “These foundations reduce implementation risk, but the main
experiment remains ahead.”

## Slide 10 — Remaining plan

Purpose: show an honest, calendar-scaled Gantt rather than a list of dates.

Read the chart from top to bottom. Dark bars are current Review 1 work; light or
colored future bars are planned. Vertical markers show the fixed review and
report dates. The outlined section inside the reporting bar is W16 contingency.

Explain the dependencies rather than every week number:

1. finish BLER characterization and select the classical operating points;
2. train and freeze the learned system;
3. build the digital feature control;
4. rehearse on validation and freeze every choice;
5. run the test campaign once; and
6. prepare the final report and demonstration.

State the four hard dates on the slide. Do not describe planned work as active
or completed.

Transition: “The final two slides connect the AI/ML method to its application,
scope and remaining work.”

## Slide 11 — Application, scope and risks

Purpose: show that the proposal is practical and connected to the circular's
real-world application requirement.

Explain the columns in order:

- The application is receiver-side edge AI inference from remote cameras or
  sensors under link limits.
- The AI/ML contribution is channel-robust neural representation learning with
  a residual encoder, dual-head decoder, multi-task loss and attribution arm.
- The main risks have controls: authenticated campaign resumption, fixed
  outcome-independent reporting and hardware kept as gated stretch scope.

State the standards boundary accurately. The project uses OpenJPEG 2.5.4 and a
TS 38.212-derived LDPC/rate-matching chain over abstract AWGN. It does not claim
a complete 5G NR radio link.

The guide's hardware-alternative acknowledgement stays pending until a real,
dated response is recorded.

Transition: “The final slide summarizes the AI/ML proposal and the work that
remains after this review.”

## Slide 12 — Summary and next steps

Purpose: close the presentation rather than ending on an equation or an
internal status item.

Summarize four points:

- the problem is edge AI inference over a limited noisy link;
- the AI/ML method is a residual neural encoder and dual-head decoder trained
  through a differentiable channel;
- classification is the controlled evaluation task against two matched
  baselines;
- the current evidence establishes technical feasibility, not the final
  comparative result; and
- the remaining path is characterization, selection, training, freeze and one
  test campaign.

State that the research question, three-system methodology, feasibility
evidence and completion plan are defined, while the final comparative result
remains future work. Point to the rubric map as evidence that all six First
Review criteria are covered.

Do not claim that showing the map alone proves team readiness; all four team
members still need to explain the project.

## Short answers for common questions

### Why not send the class label?

That moves the full task model to the sender and changes the deployment split.
This project tests a small sender-side encoder communicating to a receiver that
owns the decoder and classifier.

### Why not compare only with JPEG?

The headline conventional system uses JPEG 2000 and a tuned digital physical
layer. It is designed to be a strong baseline rather than an easy comparator.

### Why is the digital feature system necessary?

Without it, a learned improvement could come from task-aware representation
alone. The control helps isolate whether joint coding adds value.

### Why AWGN instead of radio hardware?

Normalized additive white Gaussian noise isolates the communication mechanism
and permits exact paired comparisons. Hardware adds synchronization, frequency
offset, clipping and regulatory variables. It remains optional stretch work.

### What if H1 is not supported?

The result is reported as observed. Completion depends on executing the fixed
protocol correctly, not on obtaining a favorable outcome.

### Has the test set been used?

No. It remains sealed until gate G-12, after all choices are frozen.
