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

## Slide 5 — Why three systems

Purpose: make the novelty and attribution problem understandable.

Explain the rows by their scientific roles:

- The classical image link is the strong conventional reference.
- The digital feature control is task-aware but still uses the same digital
  channel chain.
- The learned joint link is task-aware and jointly trained with the channel.

The third arm matters because any improvement over classical image
transmission could otherwise come from sending task-aware features rather than
from joint coding itself.

Transition: “This distinction is also the gap that appears in the literature.”

## Slide 6 — Literature synthesis

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

Transition: “The experiment therefore needs a comparison that removes these
alternative explanations.”

## Slide 7 — Fair comparison

Purpose: explain methodology as simple cause and effect.

For each control, state why it exists. The most important controls are:

- same image and same channel realization;
- same complex-symbol budget and SNR definition;
- validation-only tuning;
- failed transmissions retained in accuracy; and
- complete overhead and BLER identity accounting.

If asked about the test set, say that it remains sealed until every selection
and model choice is frozen at gate G-12.

Transition: “With that comparison fixed, these are the questions registered in
advance.”

## Slide 8 — Objectives and hypotheses

Purpose: separate project completion from a desired outcome.

State the objectives first. Then summarize the hypotheses:

- H1 asks about low-SNR separation.
- H2 asks about graceful degradation versus a fixed-system cliff.
- H3 asks whether the gap contracts as SNR improves.
- H4 asks whether joint coding exceeds the task-aware digital control.

Do not explain the H1 equation here unless asked. Point to slide 12 and state
that the exact rule was fixed before test access.

Transition: “The final comparison has not run, but several foundations have
already passed their own checks.”

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

## Slide 11 — Review decision

Purpose: make the requested panel decision explicit.

Ask the panel to confirm:

- the image-classification task boundary;
- the three-system comparison;
- the fairness controls; and
- simulation-first Tier 1 as a sufficient completion path.

State the standards boundary accurately: the project uses a TS 38.212-derived
LDPC/rate-matching chain over abstract AWGN. It does not claim a complete 5G NR
radio link. Keep SDR and edge-device work labelled as optional stretch scope.

The guide's hardware-alternative acknowledgement stays pending until a real,
dated response is recorded.

Transition: “The final slide shows the exact primary rule and where every First
Review criterion appears.”

## Slide 12 — Formal H1 and rubric map

Purpose: expose the formal contract without forcing the opening of the talk to
carry it.

For H1, preserve these facts exactly:

- the comparator is learned versus `classical_adaptive` at the headline ratio;
- a point qualifies when the studentized paired mean exceeds 1.96;
- `R_obs` is the longest consecutive qualifying run at or below training SNR;
- H1 is supported only when `R_obs ≥ 3` and the calibrated run p-value is at
  most 0.05;
- the whole low-SNR-region mean paired difference is the effect size of record;
  and
- a curve crossing is reported if observed but is not required.

Point to the six-criterion map. Do not claim that showing the map alone proves
team readiness; all four team members still need to explain the project.

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

