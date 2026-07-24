<!-- GENERATED FROM spec/SPEC.md BY tools/gen_spec_views.py - DO NOT EDIT -->

# Roadmap

Requirements extracted from [`SPEC.md`](../SPEC.md). This view is for focused reading and review; the spec text is normative.

## DEC

- **DEC-1** — **Dataset ladder.** The headline result targets Imagenette at 160px. Demotion to STL-10 at 96px is permitted only at G-6, and only if the Tier 3 live demo cannot otherwise run live. Demotion to CIFAR-10 is permitted at G-3 if compute limits or a non-reproducing crossover force it. Dataset MUST be a configuration axis, never a code fork, and CIFAR-10 MUST stay wired throughout as the fast smoke path. Rationale: highest-resolution headline that the hardware can actually carry, with two pre-agreed step-downs so the schedule never stalls on a judgement call.
- **DEC-2** — **Dual-head decoder.** One decoder carrying a reconstruction head and a classification head, trained with `loss = CE + λ·MSE`. Rationale: the accuracy curve and the "blurry but still task-correct" demo visual come from one model rather than two, at a measured accuracy cost (SR-9). Reversal: if the λ calibration cannot meet SR-9, split into two models and record the change here.
- **DEC-3** — **Python primary.** Learned system and classical baseline are both Python; MATLAB appears only as non-blocking cross-checks (OPT-1..OPT-3). Rationale: one language, one CI path, no license dependency on the critical path.
- **DEC-4** — **Compute.** An RTX 4060 Mobile (8 GB) is the assumed trainer, with Colab/Kaggle free tier as overflow. University cluster access MUST NOT appear on any critical path. Consequence: checkpoint/resume (SR-10) and a per-run wall-clock cap (SR-11) are hard requirements, not conveniences.
- **DEC-5** — **Radio hardware deferred.** Tiers 2 and 3 are specified as capability requirements plus a budget range (HR-1), not a named device, and no purchase happens before G-5.
- **DEC-6** — **Required experiments.** Core crossover, SNR-mismatch robustness and bandwidth-ratio sweep are required (ER-1..ER-3). Rayleigh fading and the λ sweep are future work (FW-1, FW-2), but the extension points that admit them (SR-5, SR-8) are required now.
- **DEC-7** — **Demo styling.** The Streamlit demo and the thesis figures share one publication-grade plotting module (DR-4), so a demo screenshot is directly usable in the report.
- **DEC-8** — **Document structure.** This file is authoritative and self-sufficient: a reader who never runs the generator loses nothing. All other files under `spec/` are derived views.

## FW

- **FW-1** — Rayleigh block and fast fading, via the channel registry (SR-5). Expected to strengthen the graceful-degradation claim, since classical schemes suffer disproportionately under fading.
- **FW-2** — λ sweep quantifying the accuracy cost of a viewable reconstruction, with λ=0 as the pure-task upper bound, via SR-8.
- **FW-3** — SNR-adaptive or variable-rate coding, where the transmitter adjusts rate to measured channel state.
- **FW-4** — Alternative downstream tasks (segmentation, detection) via the task-head registry (SR-15).
- **FW-5** — Digital/entropy-coded semantic variants for comparison against the analog-symbol design.
- **FW-6** — Additional datasets beyond `params.datasets`.

## G

- **G-1** — Reference classifier meets `clean_acc_floor` for the smoke dataset. Fallback: switch backbone or extend training before any DJSCC work begins.
- **G-2** — LDPC BER matches published curves within tolerance. Fallback: change LDPC library. No comparison may be reported before this passes.
- **G-3** — Crossover reproduced on the CIFAR-10 smoke path. Fallback: one debug week, then invoke DEC-1 demotion to CIFAR-10 as the headline dataset and re-plan.
- **G-4** — λ calibrated per SR-9. Fallback: DEC-2 reversal to two separate models.
- **G-5** — Tier 1 frozen: ER-1..ER-4 complete with confidence intervals and the success criterion decided either way. Passing unlocks the hardware purchase (HR-2). Failing means Tier 2/3 are abandoned and effort moves to reporting the negative result (ER-8).
- **G-6** — Tier 3 latency budget met at the headline dataset. Fallback: DEC-1 demotion to STL-10, or pre-recorded demonstration.

## OPT

- **OPT-1** — MATLAB Communications Toolbox reproduction of the LDPC BER curve as an independent check on BR-2.
- **OPT-2** — MATLAB or symbolic treatment of channel capacity and the separation theorem for the thesis mathematics chapter, tying §1 claim 2 to a derivation.
- **OPT-3** — An independent MATLAB reimplementation of the JPEG+LDPC chain to cross-validate baseline accuracy at two or three SNR points.

## Parameters referenced here

| Parameter | Value |
| --- | --- |
| `datasets` | *(see datasheet)* |
