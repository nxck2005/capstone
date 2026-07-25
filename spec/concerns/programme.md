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
- **PR-8** — Each review in `params.deliverables.review_weeks` MUST have a prepared package matching that review's rubric weighting, and the engineering gates MUST be scheduled so the required evidence exists beforehand. *(verify: review package committed before each review week)*

## Parameters referenced here

| Parameter | Value |
| --- | --- |
| `deliverables.literature_review_min_refs` | 25 |
| `deliverables.novelty_claims` | er9_attribution_decomposition, br11_format_overhead_controlled_baseline |
| `deliverables.plagiarism_report_required` | true |
| `deliverables.poster_format` | a0 |
| `deliverables.report_format_source` | vault/capstone/CAPSTONE_THESIS_Format.docx |
| `deliverables.review_weeks` | *(see datasheet)* |
| `deliverables.standards` | 3gpp_ts_38_212, itu_t_t_800_jpeg2000, itu_t_t_81_jpeg, ieee_754, ietf_rfc_2119 |
| `deliverables.time_plan_artifact` | gantt_chart |
