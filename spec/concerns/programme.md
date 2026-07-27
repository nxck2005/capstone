<!-- GENERATED FROM spec/SPEC.md BY tools/gen_spec_views.py - DO NOT EDIT -->

# Programme deliverables

Requirements extracted from [`SPEC.md`](../SPEC.md). This view is for focused reading and review; the spec text is normative.

## PR

- **PR-1** — A literature review of at least `params.deliverables.literature_review_min_refs` references MUST be drafted before the first review and maintained thereafter, covering DJSCC, separation and finite-blocklength theory, and learned image compression. *(verify: reference list committed and cited in the report draft)*
- **PR-2** — A time plan in the form of `params.deliverables.time_plan_artifact` MUST exist by the first review week in `params.deliverables.review_weeks` and be updated at each subsequent review. *(verify: committed chart artifact with revision history)*
- **PR-3** — A standards and tools register MUST list every entry in `params.deliverables.standards` with where each is used in the implementation. *(verify: register committed and each entry resolvable to code or a spec requirement)*
- **PR-4** — A poster in `params.deliverables.poster_format` MUST be prepared and submitted. *(verify: submitted artifact)*
- **PR-5** — A plagiarism report MUST be produced per `params.deliverables.plagiarism_report_required` and submitted with the final report. *(verify: submitted artifact)*
- **PR-6** — The final report MUST conform to `params.deliverables.report_format_source`, proof-read and ratified by the guide. *(verify: format checklist signed off)*
- **PR-7** — A written novelty statement MUST name `params.deliverables.novelty_claims` explicitly, state what in each is not present in the prior work **as established by PR-1's completed review**, and be defensible independently of whether §2's hypotheses are supported (DEC-13). It MUST NOT claim novelty for graceful degradation, cliff-avoidance, or the existence of a task-aware digital semantic system — all three have substantial prior art — and any "not reported in the literature" assertion MUST be traceable to the reviewed reference list rather than asserted from this specification (AM-10). *(verify: statement committed, every novelty assertion resolvable to a PR-1 reference, and a negative-check listing the prior art each claim is distinguished from)*
- **PR-8** — Each review in `params.deliverables.review_weeks` MUST have a prepared package matching that review's rubric weighting, and the engineering gates MUST be scheduled so the required evidence exists beforehand. Verified sub-mark weights: First 30 → 10, Second 50 → 30, Third **60** → 40, Report 20 → 20. Three consequences are normative. **(a)** Objectives at every review MUST be stated per `params.deliverables.objectives_stated_as` — in §2's completion terms (build, validate, bandwidth-match, bit-account, evaluate at the learned-blind operating point, report with paired inference) — and MUST NOT be stated as outcomes such as "show that the learned system beats the classical one". The Third Review scores `Objectives Met` at 10 sub-marks, defined as whether objectives set at the first review or modified at the second were met; §2 makes completion independent of which way the result falls, but a rubric does not grade completion, so an objective phrased as an outcome converts DEC-16's fallback — which §2 declares a complete Tier 1 — into an objective visibly not met. Everything §2 gets right can be undone by one slide at W4. **(b)** If G-10 takes DEC-16's fallback, the objectives MUST be restated at `params.deliverables.objectives_modification_point`, which the rubric explicitly sanctions, with the G-8 and G-10 evidence attached. **(c)** The Second and Third packages MUST each carry a one-slide distillation of §17, which is what the `Methodology` rubric asks for — the approach decided *based on the results obtained*, i.e. documented course correction — and which this project has already done the work for (AM-46). *(verify: review package committed before each review week, with the objectives slide checked against §2's wording and the §17 slide present)*
- **PR-9** — A written deployment-scenario note MUST be produced: the edge or IoT link this targets, a rough link budget, why finite-blocklength penalties bite in that regime, and what `params.bandwidth.headline_ratio` corresponds to in real terms. One page, no hardware. Rationale: the `Application` rubric asks whether the project has real-world applications, and DEC-14 makes Tiers 2 and 3 upside while HR-5 makes Tier 1 standalone — correct engineering that nonetheless leaves those marks resting on a Streamlit demo of a simulation. This note answers the rubric directly and doubles as thesis introduction material (AM-46). *(verify: note committed and cited from the report introduction)*

## Parameters referenced here

| Parameter | Value |
| --- | --- |
| `bandwidth.headline_ratio` | crossover_ratio |
| `deliverables.literature_review_min_refs` | 25 |
| `deliverables.novelty_claims` | er9_attribution_decomposition, br11_format_overhead_controlled_baseline |
| `deliverables.objectives_modification_point` | second_review |
| `deliverables.objectives_stated_as` | completion_terms_not_outcomes |
| `deliverables.plagiarism_report_required` | true |
| `deliverables.poster_format` | a0 |
| `deliverables.report_format_source` | vault/capstone/CAPSTONE_THESIS_Format.docx |
| `deliverables.review_weeks` | *(see datasheet)* |
| `deliverables.standards` | 3gpp_ts_38_212, 3gpp_ts_38_211, itu_t_t_800_jpeg2000, itu_t_t_81_jpeg, ieee_754, ietf_rfc_2119 |
| `deliverables.time_plan_artifact` | gantt_chart |
