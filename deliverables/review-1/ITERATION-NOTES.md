# Review 1 iteration notes

This file records author feedback that must carry into later Review 1 deck
iterations. It is a design memory, not a scientific authority. `spec/SPEC.md`
remains authoritative for the experiment and
[`first-review-package.md`](first-review-package.md) remains authoritative for
the delivery contract.

## Fixed communication direction

- Do not front-load AI/ML terminology to prove that the project belongs to the
  AIML specialization. That makes the deck harder to understand.
- Slides 1–2 must first establish the camera, limited noisy link, receiver and
  classification task in ordinary language.
- Slide 3 explains the usual communication system. Slide 4 then introduces the
  learned sender and receiver, names DJSCC only after the basic idea is clear,
  and identifies supervised end-to-end training as the AIML method.
- Put architecture and loss details in the methodology section, where they
  support understanding instead of replacing it.
- Image classification is the downstream evaluation task. The learned
  communication model is the AIML method.
- Use the digital feature arm to explain the AI attribution question: it
  isolates representation learning from end-to-end joint channel coding.
- Assume the audience starts with no project context.
- Transfer the basic mental model before showing technical machinery.
- Begin with the task and the ordinary solution.
- Explain the ordinary solution's limitation before introducing DJSCC.
- Introduce acronyms only when they become necessary.
- Translate every gate or metric into a plain-language meaning.
- Separate “what this proves” from “what this does not prove.”
- Show unfinished work as unfinished.

## First Review structure

The First Review is a proposal and preliminary-design review. After transferring
the basic mental model, use the standard proposal sequence:

1. motivation and problem context;
2. conventional background and the proposed idea;
3. literature survey and research gap;
4. problem statement, research question and completion objectives;
5. proposed methodology and system-level comparison design;
6. hypotheses and evaluation plan;
7. preliminary feasibility and implementation evidence;
8. remaining time plan;
9. application, scope and risks; and
10. summary, next steps and rubric map.

Do not let the second half become a repository audit. Gate identifiers and
evidence controls support the proposal; they are not the proposal's narrative.

Use literal section titles so the standard review pages are unmistakable:
“Literature survey and research gap,” “Problem statement, research question
and objectives,” “Proposed methodology,” “Research hypotheses,” “Preliminary
work and feasibility,” “Project Gantt chart,” “Applications, project scope and
risks,” and “Summary and next steps.”

## Writing style

- Use simple, literal prose.
- Prefer short declarative sentences.
- Do not use aphorisms, slogans, flourishes or dramatic claims.
- Do not present repository governance as the project idea.
- Keep formal statistical wording where the rubric or protocol requires it,
  but place it after the intuitive explanation.

## Visual style

- Default to black text on white.
- Use light rules, simple tables and basic flow diagrams.
- Do not add decorative themes or over-produced visuals.
- Use one visual only when it materially explains a process or comparison.
- Keep editable native text and shapes in the PPTX.
- The Time Plan criterion must use a real calendar-scaled Gantt chart in the
  presentation, not only a schedule table or a row of phase boxes.
- Completed G8_C BLER characterization and completed G8_D validation-measurement
  tooling are marked complete. G8_E full validation measurement/pass-one
  selection and all later scientific work remain planned/not started.
  Review-package authoring is not shown as a second active experiment
  workstream.

## Iteration procedure

1. Preserve the current editable deck, review exports, previews and build source
   in a dated fallback directory before changing the active files.
2. Keep the active filenames stable so package links do not break.
3. Render the complete deck and inspect both the contact sheet and individual
   slides after every content revision.
4. Recheck the six-criterion rubric map, H1 wording, review dates, guide status
   and current scientific boundary.
5. Read the live G8_C/G8_D state from `instructions/RESUME.md` and the D7
   handoff on submission day. Do not copy volatile work-unit counts into this
   maintained note.

## Fallbacks

The versions that existed before the newcomer-first revision are preserved at
[`fallbacks/2026-08-12-pre-knowledge-transfer/`](fallbacks/2026-08-12-pre-knowledge-transfer/README.md).

The first newcomer-first version, before the proposal-flow correction, is
preserved at
[`fallbacks/2026-08-12-newcomer-first-v1/`](fallbacks/2026-08-12-newcomer-first-v1/README.md).

The first proposal-flow version, before “requested decisions” was removed from
the closing slide, is preserved at
[`fallbacks/2026-08-12-proposal-flow-v1/`](fallbacks/2026-08-12-proposal-flow-v1/README.md).

The versions immediately before the graphical Gantt replaced the schedule
summary are preserved at
[`fallbacks/2026-08-13-pre-gantt-slide/`](fallbacks/2026-08-13-pre-gantt-slide/README.md).

The graphical-Gantt versions immediately before the AI/ML emphasis correction
are preserved at
[`fallbacks/2026-08-13-pre-aiml-emphasis/`](fallbacks/2026-08-13-pre-aiml-emphasis/README.md).

The AI-emphasis version that front-loaded technical AIML wording before the
plain-language explanation is preserved at
[`fallbacks/2026-08-13-ai-emphasis-v1/`](fallbacks/2026-08-13-ai-emphasis-v1/README.md).
