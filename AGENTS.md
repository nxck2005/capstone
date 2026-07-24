# AGENTS.md

This file provides guidance to coding agents (Claude Code, and any other agent that reads `AGENTS.md`) when working with code in this repository.

## Repository status

There is **no implementation yet** — planning material, the specification under `spec/`, and the spec tooling in `tools/`. No training code, no test suite. When you add project tooling (test runner, lint config), record the commands here.

`spec/SPEC.md` is the normative description of what gets built and how it will be judged. Read it before writing any code: it fixes the bandwidth budgets, SNR grids, baseline fairness rules, and the preregistered hypotheses. Settled decisions are `DEC-1`..`DEC-16` in §3; requirements carry stable IDs (`SR`/`BR`/`ER`/`DR`/`HR`/`PR`/`OPT`/`FW`) that code and commit messages should cite. Retired IDs live in §14 and are never reused.

**Start here each session: [`NEXT.md`](NEXT.md).** It is the inter-session hand-off file — what to do
next, open questions, and recently-settled things that must not be reopened. It is scrappy and
non-normative by design (`spec/SPEC.md` wins on any conflict), and it is **expected to be updated
before a session ends** if the state changed. Promote anything durable out of it: decisions become a
`DEC` in `SPEC.md` §3, risks and provisional values go to `SPEC.md` §16, explanations go to `docs/`.

**Where the spec stands.** It was substantially revised (97 requirements, 1 retired) after four independent reviews converged on five blockers, all now addressed: §2's crossover requirement was unachievable at the specified operating point; JPEG's container floor made the classical baseline degenerate; baseline tuning leaked the test split; the statistical design could not support its own claims; and ~30 rubric marks of graded deliverables were unscheduled. **`SPEC.md` §16 is the open-items and risk register** — read it first when resuming. It lists what still needs supervisor sign-off, which `params` values are provisional estimates awaiting a gate (`core_ratio`, `lambda_core`, the `clean_acc_floor`s), and the risks knowingly carried. Nothing in the project has been executed yet: **W0 has not started**, and its gate G-9 is what unblocks W1.

### Commands

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # PyYAML only, for the spec tooling
python tools/gen_spec_views.py            # regenerate the derived spec views
python tools/gen_spec_views.py --check    # validate spec + fail on stale generated files
```

Note: system Python has no PyYAML and `pip install` into it is blocked (PEP 668), so the venv is required.

`SPEC.md` is hand-written and authoritative. `spec/DATASHEET.md`, `spec/concerns/*.md` and `spec/params.generated.yaml` are **generated from it** — edit `SPEC.md` and regenerate, never edit the generated files. `--check` also validates the spec itself: unique requirement IDs with live-plus-retired numbering contiguous per prefix, every `params.*` citation resolving, every parameter section cited by some requirement, a `*(verify: ...)*` clause on every `SR`/`BR`/`ER`/`DR`/`HR`/`PR` line, and the `k = ratio × n` symbol-budget arithmetic. Run it after any spec edit; a stale generated file fails the check.

To retire a requirement, strike it through in place (`- ~~**G-3**~~ — reason`) under §14 rather than deleting it: the number stays reserved so live IDs are never renumbered. Passing `--check` means the document is *structurally* consistent — it says nothing about whether the experiment is scientifically valid.

Training code MUST read `spec/params.generated.yaml` rather than parsing markdown or hard-coding constants (SR-1).

## What the project is

The capstone is **Semantic Communication + AI** — idea #1 in `ideas/Proposals.pdf`, which the user has finalized. The other ideas in that PDF (DisasterMesh, DyslexiaLens, SafeScreen, CodeProof, scam detection, ReproCheck, RL malloc, thermal scheduling, cache eviction) are rejected/parked; don't propose work on them.

**The scope is the "AI overview" section of that idea, not the bullet summary above it.** The state/action/reward/hardware framing in the bullets is earlier exploratory phrasing; the AI overview paragraphs (the idea, why the learned method wins, method, tiered scope, success criterion & demo) are what's actually being built.

Core thesis: transmit the *meaning* needed to accomplish a downstream task rather than every original bit, cutting bandwidth/energy on edge/IoT links. Concretely, **deep joint source-channel coding (DJSCC)**: train a neural encoder (sender) → differentiable noisy-channel model → neural decoder (receiver) end-to-end, optimizing task error *after* channel noise at a fixed bandwidth budget. Proposed narrow task: image → classification over an AWGN channel. This is supervised end-to-end training — **not** reinforcement learning; don't reintroduce an RL framing unless the user asks for one.

### Non-negotiables from the proposal

These are the terms the project is being judged on; preserve them in any design work. `spec/SPEC.md` §1–§2 states them normatively (thesis, then completion criteria plus four preregistered hypotheses); this summary is for orientation and defers to the spec on conflict.

- **Fair baseline.** A properly-tuned source-codec + LDPC pipeline evaluated on the *identical* task, SNR, and bandwidth budget. A strawman baseline invalidates the whole contribution. Note the codec of record is **JPEG 2000, not JPEG** (DEC-9): JPEG's ~250–290 byte container floor is a large or total fraction of the channel budget at these ratios, which would make a low-SNR "cliff" a file-format artifact. JPEG is kept as a labelled secondary curve.
- **Structural (not cosmetic) advantage.** The argument is that the *task-agnostic reconstruction baseline* cannot express a task-success objective, and that Shannon separation is only optimal for infinitely long messages — real IoT links send *short* messages over *noisy* channels, a regime where separation pays finite-blocklength penalties and joint coding *may* gain. Signature behavior: graceful degradation (separated coding hits a cliff and yields nothing; semantic gets blurrier but stays task-correct). Do **not** restate this as "classical coding cannot express task success" in general — a digital system can send features or logits, which is exactly what the ER-9 control does.
- **Success criterion.** See `SPEC.md` §2, which is normative and has been revised. Completion is defined by running the preregistered protocol properly, *not* by the outcome. The primary hypothesis is a **paired** accuracy-difference interval above zero at three consecutive low-SNR points. **A curve crossing is reported if seen but is not required** — at low bandwidth ratios the learned system is expected to dominate everywhere, which supports the thesis rather than failing it. Never reintroduce "the curves must cross" as a pass condition.
- **The baseline adapts; don't "simplify" that away (DEC-16).** Modulation is an *adaptive* axis of BR-4's per-SNR tuning — the baseline climbs from QPSK to 16-QAM as the link cleans up, exactly as deployed radios do. Capping it at QPSK would be an artificial handicap that flattens the classical curve and destroys any possibility of a crossover, so `params.baseline.modulations` and `modulation_tuning` are load-bearing, not decoration. The governing rule for any crossover work: **every lever must strengthen the baseline or be preregistered; never handicap the learned system.** BR-15 requires the resulting adaptation asymmetry (baseline re-tuned per SNR, learned model trained once and frozen) to be disclosed in the methods section and every headline figure caption. `docs/crossover-explained.md` explains the whole thing.
- **Attribution.** A learned-vs-classical gap conflates task-aware representation with joint source-channel coding. ER-9 (quantised learned features over the same LDPC/QPSK chain, matched *k*) is the control that separates them, and is one of the two claimed novelty items (DEC-13).
- **Demo.** Live SNR slider driving both pipelines side-by-side on the same image, with the accuracy-vs-SNR plot updating in real time.
- **Tiered, simulation-first scope.** Tier 1 = the full defensible capstone on a *simulated* channel, built and proven first. Tier 2 (offline SDR replay) and Tier 3 (live Raspberry Pi demo) are **stretch goals with a pre-recorded demonstration as the expected outcome** (DEC-14), not planned deliverables. The project must succeed if Tiers 2–3 never land.
- **LDPC is settled (DEC-10).** Sionna `2.0.1` for base graphs, encoding, rate matching and decoding, behind an adapter seam (BR-14); TB CRC, code-block segmentation, per-block budget distribution and concatenation are written in this project because Sionna does not provide them. **Sionna no longer depends on TensorFlow** — PHY/SYS migrated to PyTorch in 2.0.0 — so the "second DL framework contradicts DEC-3" concern was checked against the release notes and is obsolete. Don't reopen it. `offset_min_sum` is a Sionna built-in, so no custom check-node callable is needed.
- **Graded deliverables are in scope.** Roughly 30 of 100 rubric marks sit on literature review, Gantt chart, standards register, A0 poster, plagiarism report and report format — tracked as `PR-1`..`PR-8`, not as an afterthought.

### On `vault/capstone/Project requirements.md`

**Stale — do not treat as constraints.** That note (RL preferred, MATLAB math component, paper potential, hardware clause) was a filter used while choosing between ideas. It has been superseded; the AI overview in `ideas/Proposals.pdf` governs. Its hardware clause is still loosely reflected in the Tier 2/3 SDR and Raspberry Pi tiers, but nothing in it overrides the tiered scope above.

## Layout

- `NEXT.md` — inter-session hand-off: next steps, open questions, don't-reopen list, session log. Non-normative and frequently rewritten. Read first, update last.
- `ideas/Proposals.pdf` — all candidate ideas with state/action/reward/hardware/baseline/impact breakdowns and blunt verdicts ("BASIC & SOLVED", "UNSOLVED & UNFIT"). The source of truth for project intent.
- `vault/capstone/` — Obsidian vault of course administrivia: proposal report template, thesis format, rubrics, the Fall 2026-27 circular (scanned images, no extractable text), and `Project requirements.md`. Deliverable formats live here; the `.obsidian/` directory is editor config, not content. `Capstone Project Rubrics.xlsx` is the grading scheme and drives the `PR` requirements — extract it by unzipping and reading `xl/sharedStrings.xml`. It scores First Review 10 / Second Review 30 / Third Review 40 / Project Report 20, with **Novelty worth 15** (the line DEC-13 exists to answer) and the review checkpoints at roughly W4 / W10 / W16.
- `spec/SPEC.md` — the project specification: thesis, preregistered hypotheses, decisions, parameters, requirements, schedule with go/no-go gates, non-goals, and the §16 open-items register. Normative and self-sufficient.
- `spec/DATASHEET.md`, `spec/concerns/`, `spec/params.generated.yaml` — generated views (see Commands above). `spec/concerns/programme.md` holds the `PR` course deliverables; the others group `SR`/`BR`/`ER`/`DR`/`HR` by concern, with retired IDs shown under a "Retired" heading.
- `tools/gen_spec_views.py` — the generator and spec validator.
- `docs/` — hand-written background notes, not generated and not normative. `crossover-explained.md` explains why §2's crossover criterion was replaced, in plain language and then technically; it is written to feed the thesis discussion chapter and viva prep, and it is the thing to hand a supervisor who asks why the success criterion changed.

### Reading the planning documents

Neither `pdftotext` nor `pypdf` is installed system-wide, and `pip install` into the system Python is blocked (PEP 668). To read the PDFs and `.docx` files, create a venv in the scratchpad and install `pypdf` there; `.docx` files are zip archives whose text can be pulled from `word/document.xml`.
