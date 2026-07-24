# AGENTS.md

This file provides guidance to coding agents (Claude Code, and any other agent that reads `AGENTS.md`) when working with code in this repository.

## Repository status

There is **no implementation yet** — planning material, the specification under `spec/`, and the spec tooling in `tools/`. No training code, no test suite. When you add project tooling (test runner, lint config), record the commands here.

`spec/SPEC.md` is the normative description of what gets built and how it will be judged. Read it before writing any code: it fixes the bandwidth budgets, SNR grids, baseline fairness rules, and the falsifiable pass/fail criterion. Settled decisions are `DEC-1`..`DEC-8` in §3; requirements carry stable IDs (`SR`/`BR`/`ER`/`DR`/`HR`/`OPT`/`FW`) that code and commit messages should cite.

### Commands

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # PyYAML only, for the spec tooling
python tools/gen_spec_views.py            # regenerate the derived spec views
python tools/gen_spec_views.py --check    # validate spec + fail on stale generated files
```

Note: system Python has no PyYAML and `pip install` into it is blocked (PEP 668), so the venv is required.

`SPEC.md` is hand-written and authoritative. `spec/DATASHEET.md`, `spec/concerns/*.md` and `spec/params.generated.yaml` are **generated from it** — edit `SPEC.md` and regenerate, never edit the generated files. `--check` also validates the spec itself: unique and contiguous requirement IDs, every `params.*` citation resolving, every parameter section cited by some requirement, a `*(verify: ...)*` clause on every `SR`/`BR`/`ER`/`DR`/`HR` line, and the `k = ratio × n` symbol-budget arithmetic. Run it after any spec edit; a stale generated file fails the check.

Training code MUST read `spec/params.generated.yaml` rather than parsing markdown or hard-coding constants (SR-1).

## What the project is

The capstone is **Semantic Communication + AI** — idea #1 in `ideas/Proposals.pdf`, which the user has finalized. The other ideas in that PDF (DisasterMesh, DyslexiaLens, SafeScreen, CodeProof, scam detection, ReproCheck, RL malloc, thermal scheduling, cache eviction) are rejected/parked; don't propose work on them.

**The scope is the "AI overview" section of that idea, not the bullet summary above it.** The state/action/reward/hardware framing in the bullets is earlier exploratory phrasing; the AI overview paragraphs (the idea, why the learned method wins, method, tiered scope, success criterion & demo) are what's actually being built.

Core thesis: transmit the *meaning* needed to accomplish a downstream task rather than every original bit, cutting bandwidth/energy on edge/IoT links. Concretely, **deep joint source-channel coding (DJSCC)**: train a neural encoder (sender) → differentiable noisy-channel model → neural decoder (receiver) end-to-end, optimizing task error *after* channel noise at a fixed bandwidth budget. Proposed narrow task: image → classification over an AWGN channel. This is supervised end-to-end training — **not** reinforcement learning; don't reintroduce an RL framing unless the user asks for one.

### Non-negotiables from the proposal

These are the terms the project is being judged on; preserve them in any design work. `spec/SPEC.md` §1–§2 states them normatively (with a falsifiable pass/fail); this summary is for orientation and defers to the spec on conflict.

- **Fair baseline.** A properly-tuned JPEG + LDPC pipeline evaluated on the *identical* task, SNR, and bandwidth budget. A strawman baseline invalidates the whole contribution.
- **Structural (not cosmetic) advantage.** The argument is that classical coding cannot express a task-success objective, and that Shannon separation is only optimal for infinitely long messages — real IoT links send *short* messages over *noisy* channels, which is where joint learned coding wins. Signature behavior: graceful degradation (classical hits a cliff and yields nothing; semantic gets blurrier but stays task-correct).
- **Success criterion.** Higher task accuracy than JPEG+LDPC at matched bandwidth, especially at low SNR, shown as an accuracy-vs-SNR plot where the two curves cross.
- **Demo.** Live SNR slider driving both pipelines side-by-side on the same image, with the crossover plot updating in real time.
- **Tiered, simulation-first scope.** Tier 1 = the full defensible capstone on a *simulated* channel, built and proven first. Tier 2 = replay encoder output through a real SDR pair offline. Tier 3 = live Raspberry Pi encoder/decoder demo. The project must succeed if Tiers 2–3 never land. The week-one check is confirming the learned system beats the classical baseline in simulation.

### On `vault/capstone/Project requirements.md`

**Stale — do not treat as constraints.** That note (RL preferred, MATLAB math component, paper potential, hardware clause) was a filter used while choosing between ideas. It has been superseded; the AI overview in `ideas/Proposals.pdf` governs. Its hardware clause is still loosely reflected in the Tier 2/3 SDR and Raspberry Pi tiers, but nothing in it overrides the tiered scope above.

## Layout

- `ideas/Proposals.pdf` — all candidate ideas with state/action/reward/hardware/baseline/impact breakdowns and blunt verdicts ("BASIC & SOLVED", "UNSOLVED & UNFIT"). The source of truth for project intent.
- `vault/capstone/` — Obsidian vault of course administrivia: proposal report template, thesis format, rubrics, the Fall 2026-27 circular (scanned images, no extractable text), and `Project requirements.md`. Deliverable formats live here; the `.obsidian/` directory is editor config, not content.
- `spec/SPEC.md` — the project specification: thesis, success criterion, decisions, parameters, requirements, schedule with go/no-go gates, non-goals. Normative and self-sufficient.
- `spec/DATASHEET.md`, `spec/concerns/`, `spec/params.generated.yaml` — generated views (see Commands above).
- `tools/gen_spec_views.py` — the generator and spec validator.

### Reading the planning documents

Neither `pdftotext` nor `pypdf` is installed system-wide, and `pip install` into the system Python is blocked (PEP 668). To read the PDFs and `.docx` files, create a venv in the scratchpad and install `pypdf` there; `.docx` files are zip archives whose text can be pulled from `word/document.xml`.
