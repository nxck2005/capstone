# Review 1 presenter guide

This guide is for a team member who understands the project but needs a clear
15-minute explanation. It follows the active
[`Review 1 deck`](semantic-communication-first-review.pptx). It is not a script
that must be memorized word for word.

The audience should leave with five facts:

1. The project is about remote image classification over a limited noisy link.
2. Semantic communication preserves information needed for the task rather
   than requiring perfect reconstruction of every pixel.
3. The experiment compares three systems because a two-system comparison could
   not separate task-aware representation from joint coding.
4. The comparison is controlled, preregistered and not yet using the test set.
5. Review 1 presents a sound question, method, foundation and plan. It does not
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

> This project studies remote image classification over a limited noisy link.
> We ask whether the sender can transmit the information needed for the class
> decision without having to preserve every image detail.

Transition: “I will first explain the complete task without using the model
details.”

## Slide 2 — The project in one minute

Purpose: give the audience the whole mental model.

Suggested explanation:

> An edge camera observes an image. It must communicate through a link with a
> fixed bandwidth and added noise. The receiver makes the class decision. We
> compare systems by the accuracy of that decision at the same channel budget.

Then define semantic communication using the wording on the slide. Emphasize
that image quality can still be measured, but task accuracy is the headline
result.

Transition: “Before explaining the learned method, this is how the same task is
normally solved.”

## Slide 3 — Conventional image link

Purpose: teach the baseline before comparing against it.

Walk through the flow once from left to right:

- JPEG 2000 reduces the image data.
- LDPC adds redundancy that can repair damaged bits.
- Modulation turns bits into channel symbols.
- The receiver reconstructs the image and runs a frozen classifier.

Do not call this system weak or obsolete. It is deliberately tuned as a strong
baseline. Its narrow limitation is that its stages mainly optimize image and
bit recovery while the experiment evaluates the final class decision.

Transition: “The learned system changes where that optimization happens.”

## Slide 4 — Learned joint system

Purpose: explain DJSCC only after the ordinary path is clear.

Suggested explanation:

> In deep joint source–channel coding, the encoder at the sender and the
> decoder at the receiver are trained together through a noisy channel model.
> This allows the transmitted representation to preserve information useful
> for classification after noise.

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

1. The problem is that a conventional image link is not optimized for the
   receiver's final class decision.
2. The research question asks whether learned joint coding performs better at
   the same communication resources and whether any remaining difference is
   consistent with joint coding.
3. The objectives commit the team to build, match, freeze and evaluate all
   three systems.

Emphasize that the objectives describe work to complete. They do not promise a
positive result.

Transition: “This is the proposed experiment for answering that question.”

## Slide 7 — Proposed methodology

Purpose: present the system-level design expected in a preliminary review.

Start with the shared-input strip. Every arm receives the same image, split,
symbol budget, SNR definition and keyed channel realization.

Then explain the three rows by their scientific roles:

- The classical image link is the strong conventional reference.
- The digital feature control is task-aware but retains a conventional digital
  channel chain.
- The learned joint link trains the channel-facing representation end to end.

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

- G-1 shows that the reference classification task is viable.
- G-7 shows that the planned learned model fits the available GPU.
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

Purpose: show an honest critical path.

Explain the dependencies rather than every week number:

1. finish BLER characterization and select the classical operating points;
2. train and freeze the learned system;
3. build the digital feature control;
4. rehearse on validation and freeze every choice;
5. run the test campaign once; and
6. prepare the final report and demonstration.

State the four hard dates on the slide. Do not describe planned work as active
or completed.

Transition: “The remaining decision for this review is therefore about scope
and method, not about a result that does not yet exist.”

## Slide 11 — Application, scope and risks

Purpose: show that the proposal is practical and connected to the circular's
real-world application requirement.

Explain the columns in order:

- The application is remote camera or edge-sensor classification under link
  limits, with the task model at the receiver.
- The required project is simulation-first Tier 1, covering all three systems
  and one sealed test campaign.
- The main risks have controls: authenticated campaign resumption, fixed
  outcome-independent reporting and hardware kept as gated stretch scope.

State the standards boundary accurately. The project uses OpenJPEG 2.5.4 and a
TS 38.212-derived LDPC/rate-matching chain over abstract AWGN. It does not claim
a complete 5G NR radio link.

The guide's hardware-alternative acknowledgement stays pending until a real,
dated response is recorded.

Transition: “The final slide summarizes the proposal and the decisions needed
from this review.”

## Slide 12 — Summary and next steps

Purpose: close the presentation rather than ending on an equation or an
internal status item.

Summarize four points:

- the problem is remote classification over a limited noisy link;
- the proposal is task-aware joint coding compared with two matched digital
  systems;
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
