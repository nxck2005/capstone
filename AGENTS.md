# AGENTS.md

This file provides guidance to coding agents (Claude Code, and any other agent that reads `AGENTS.md`) when working with code in this repository.

## Repository status

**Implementation started 2026-07-28. W1 and W2 are complete; G-7 passed.** Five bounded W1 batches are committed: batch 1 (`e90a1e0`), batch 2 (`2b23c1e`), batch 3 (`72be2af`), batch 4 (`eba5bd2`) and the AM-77 dataset registry/provenance/manifest batch (`2c6f780`); the AM-72–76 remediation is committed as `8e59535`, and the final pre-G-1 reference-classifier integrity implementation is committed as `89a3af4`. The clean 100-epoch Imagenette-160 campaign ran from scratch on 2026-07-29 and validation-only G-1 **passed** at 898/1000 = 0.898, best and final at epoch 99. The final/best checkpoint ID is `9c37362347a0203597d6e8e9d9a58fde30ba286f3cec9b4d2f800bd8a3256002`; the config hash is `a9717575d71f2b3e9dd411b10b7735bdb3946c985fead48cb3c5af07423f12e1`. Its portable repository path is `checkpoints/reference_classifier/epoch-99.pt`; `results/reference_classifier/g1_adjudication.json` records the machine-readable gate decision and `tools/verify_g1_adjudication.py` verifies it offline. The single frozen checkpoint is preserved as GitHub Release `g1-reference-classifier-2026-07-29`, asset `reference-classifier-imagenette160-epoch99-9c37362347a0203597d6e8e9d9a58fde.pt`; the other 99 ignored checkpoints were not uploaded. W2's immutable implementation commit is `26b631ede27a6f88f1d004a66b845c52a658e07c`. The corrected G-7 profiler executes every critical project module from a clean detached worktree at that commit and records resolved paths, executed-byte SHA-256 values and git blob SHAs. On the RTX 4060 Laptop GPU, the corrected `r_1_2` profile completed all 8,469 training images at batch 32 in 48.684 s, reserved 1.004 GiB, projected 100 epochs to 1.352 h, and measured 1,640,957 parameters. G-7 passed and the primary architecture was retained. The single next task is the transparency-bitrate probe with the frozen classifier, before W3/W4 baseline work. Do not rerun G-1 or begin a fallback rung without new evidence requiring it.

**`src/data/test_access.py` is the sole guarded boundary to the test split and nothing else may import it** (SR-22, DEC-12). That rule is enforced by an AST-walking test, not by convention, and it is the reason the module exists as its own file. Test access releases at `params.evaluation.test_access_gate` — **G-12, W11** — not G-10, which AM-60 caught pointing three weeks early.

`spec/SPEC.md` is the normative description of what gets built and how it will be judged. Read it before writing any code: it fixes the bandwidth budgets, SNR grids, baseline fairness rules, and the preregistered hypotheses. Settled decisions are `DEC-1`..`DEC-16` in §3; requirements carry stable IDs (`SR`/`BR`/`ER`/`DR`/`HR`/`PR`/`OPT`/`FW`) that code and commit messages should cite. Retired IDs live in §14 and are never reused. **Changes to the spec are recorded, not made silently:** every amendment gets an `AM` entry in §17 saying what changed and why, and the amended item carries an `(AM-n)` back-reference — see the amendment convention below.

**Start here each session: [`NEXT.md`](NEXT.md).** It is the inter-session hand-off file — what to do
next, open questions, and recently-settled things that must not be reopened. It is scrappy and
non-normative by design (`spec/SPEC.md` wins on any conflict), and it is **expected to be updated
before a session ends** if the state changed. Promote anything durable out of it: decisions become a
`DEC` in `SPEC.md` §3, risks and provisional values go to `SPEC.md` §16, explanations go to `docs/`.

**Where the spec stands.** It now carries 186 requirements (2 retired), of which 78 are `AM` amendment records. **AM-77 makes dataset provenance and pre-freeze manifest construction executable:** exact archive length/SHA-256 pins, dataset-specific source-payload and authoritative-class rules, canonical CSV bytes, and a provenance-only published-test scan that is forbidden from decoding or canonicalizing. **AM-78 fixes deterministic, resumable reference-classifier training details without changing its scientific recipe.** AM-71 remains the stable-source-byte identity clarification, and AM-72..AM-76 remain the implemented-contract remediation. The adjudicated EXT-6 findings and their arithmetic remain recorded in §17; do not reopen them without new evidence. W0 is done; G-9, G-1 and G-7 passed; W1 and W2 are complete. The transparency-bitrate probe is the current engineering frontier and must precede W3/W4 baseline work.

### Commands

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # PyYAML only, for the spec tooling
python tools/gen_spec_views.py            # regenerate the derived spec views
python tools/gen_spec_views.py --check    # validate spec + fail on stale generated files
python tools/check_doc_consistency.py     # hand-written docs vs the spec; -v lists what passed
python tools/check_literals.py            # SR-1 numeric-literal lint; -v lists scanned files
python spec/evidence/check_packetisation.py            # TS 38.212 conformance, no GPU/network, <1s
python spec/evidence/check_packetisation.py --json spec/evidence/packetisation_record.json
.venv/bin/python tools/verify_cpu_lock.py --clean-install
.venv/bin/python tools/fetch_datasets.py                 # verify pinned byte length/SHA-256, then extract
.venv/bin/python tools/fetch_datasets.py --check         # network-free archive provenance verification
.venv/bin/python tools/materialize_manifests.py
.venv/bin/python tools/materialize_manifests.py --check  # regenerate in memory; compare exact committed bytes
.venv/bin/python tools/verify_datasets.py                 # real train/val smoke + zero-call test-provenance audit
.venv/bin/python tools/train_reference_classifier.py --config configs/reference-classifier-clean.yaml --dataset imagenette160 --device cuda --smoke-steps 3 --smoke-val-batches 2  # bounded, ignored smoke only
.venv/bin/python tools/train_reference_classifier.py --config configs/reference-classifier-clean.yaml --dataset imagenette160 --device cuda --full-run  # production G-1 campaign; completed 2026-07-29
.venv/bin/python tools/verify_g1_adjudication.py       # network-free frozen G-1 cross-check; hashes local checkpoint when present
.venv/bin/python tools/profile_djscc_g7.py             # requires a clean --git-repo worktree at the configured implementation commit; see W2 worklog
.venv/bin/python tools/verify_g7_profile.py            # network-free frozen G-7 config/commit/metric/gate cross-check
.venv/bin/python -m pytest              # project test suite; config is in pyproject.toml
```

**AM-77 provenance pins:** Imagenette-160 `imagenette2-160.tgz`, 99,003,388 bytes,
SHA-256 `64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5`; STL-10
`stl10_binary.tar.gz`, 2,640,397,119 bytes,
`f31fd99273a1acb8609c8db427cebb1de3f71de77758cdc0e22956e1289b9866`; CIFAR-10
`cifar-10-python.tar.gz`, 170,498,071 bytes,
`6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`. Manifest pins:
`data/manifests/imagenette160.csv` → `224309422f15bf89460559381aea4b00c4779c52d3652f7f679a213369f3f889`;
`data/manifests/stl10.csv` → `67936da779dc0010160b37b3b40001490304a5873eb978d261e3a57947387b47`;
`data/manifests/cifar10.csv` → `09e9debf4743831ca61f17154a997e60becdd7046a585bdbd94b5db4bf12a537`.

**GPU check on this machine — it is WSL2, so look for `/dev/dxg`, not `/dev/nvidia*`:**

```bash
ls -l /dev/dxg                                      # WSL GPU device; expect crw-rw-rw- 10,125
/usr/lib/wsl/lib/nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
.venv/bin/python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
```

Measured 2026-07-28: `/dev/dxg` present as `crw-rw-rw- root 10,125`, `NVIDIA GeForce RTX 4060 Laptop GPU`, driver `592.82`, and torch `13.0 True` with a real device matmul succeeding. **`nvidia-smi` reports `CUDA Version: 13.1` while torch is built for `13.0` — that is normal minor-version compatibility, not a mismatch; do not "fix" it by moving the pin.** An agent that probes `/dev/nvidia*` will wrongly conclude there is no GPU, and the `nvidia-smi` on `PATH` is not always the same binary as the one under `/usr/lib/wsl/lib/`.

Device access and network access are **separate** permissions, and only the **third** command settles the one that matters: a visible `/dev/dxg` and a working `nvidia-smi` do not guarantee torch can initialise CUDA, since that additionally needs `libcuda` resolvable through the WSL shim. Check both before assuming which W1 steps you can complete.

Note: `uv run <script>` warns *"No `requires-python` value found in the workspace"*. That is expected and must not be "fixed" — `pyproject.toml` deliberately carries pytest configuration only, with no `[project]` table, so there is no packaging and no install step (SR-21 owns dependencies via the lockfiles). Run project code with `.venv/bin/python`, not `uv run`.

`pytest` needs the **runtime** environment (below), not just the spec tooling. Its config lives in `pyproject.toml`, which sets `pythonpath = ["src", "tools"]` — that is why there is no install step and no packaging. The suite is **not expected to pass on the CPU-only install path**: `tests/test_env.py::test_cuda_build` hard-asserts a CUDA build with no skip marker and no environment-variable escape hatch, deliberately (AM-67), because a variable exported once in a shell profile would silently disarm the only check that catches a CPU build on the machine that trains.

`check_doc_consistency.py` guards what `gen_spec_views.py --check` cannot: the **current hand-written documentation** defined by AM-76. Validly marked historical plans are excluded only with the exact banner and a link resolving to root `NEXT.md`; malformed banners fail explicitly. Its stale-rule table is the tool's memory and is meant to grow.

`check_packetisation.py` must be **re-run and its record regenerated** after any change to `params.bandwidth`, `params.baseline` or `params.digital_semantic_control` — it asserts byte alignment, `B' % C == 0`, `K = 22Z`/`10Z`, filler accounting, `Σ E_r == G` and the per-block rate floor across all 216 configurations, and the record carries the params and script hashes that produced it. It reported **zero failures while violating four of those rules** before AM-58, which is the reason it now asserts them rather than printing summaries.

Project runtime dependencies are **not** in `requirements.txt`: SR-21 requires a hashed `requirements.lock`, generated at W1 from `requirements.in` by **`uv`** (`params.environment.lock_tool`, decided in AM-61 — do not substitute `pip-tools`) and installed from `params.environment.torch_index_url`.

```bash
uv pip compile requirements.in --generate-hashes --emit-index-url \
    --index-strategy unsafe-best-match -o requirements.lock            # cu130 index
uv pip compile requirements-cpu.in --generate-hashes --emit-index-url \
    --index-strategy unsafe-best-match -o requirements-cpu.lock
uv pip sync requirements.lock --index-strategy unsafe-best-match
.venv/bin/python -c "import torch; assert torch.version.cuda is not None, 'CPU BUILD'"
```

**Both non-default flags are load-bearing and are recorded as `params.environment.lock_index_strategy` and `params.environment.lock_emit_index_url` (AM-66), not just here.** `--index-strategy unsafe-best-match` is needed because the default stops at the first index carrying a package *name*, and PyPI carries `torch` but not the `+cu130` local version. `--emit-index-url` is needed because a lockfile that does not record its indices cannot be installed by anything — including plain `pip` — and the failure names only the version, never the missing index, so it reads like a bad pin.

Three constraints on that block: a bare resolve silently yields the **CPU build**, and the only check that catches it is `torch.version.cuda is not None` rather than a successful import (AM-23); the emitted lockfile MUST stay installable by plain `pip install --require-hashes`, because `uv` is pinned so that *resolution* reproduces, not so the project gains a runtime dependency on it; and the lockfile **covers the spec tooling as well as the runtime stack** (AM-65) — `uv pip sync` makes the environment *exactly* the lockfile, so a runtime-only lock would uninstall PyYAML and break `gen_spec_views.py` and `check_doc_consistency.py`. `requirements.txt` is unchanged and stays the dependency-light bootstrap.

OpenJPEG is not provisioned by either Python lock (AM-75). The reference system command is
`pacman -S openjpeg2`; `openjpeg2 2.5.4-1` was observed on this machine on 2026-07-29 and is dated
evidence only. The normative condition is the loaded version `2.5.4`, checked before any J2K path
creates artifacts; learned-only metadata may record `openjpeg_version: null` when it is unavailable.

Note: system Python has no PyYAML and `pip install` into it is blocked (PEP 668), so the venv is required.

`SPEC.md` is hand-written and authoritative. `spec/DATASHEET.md`, `spec/concerns/*.md` and `spec/params.generated.yaml` are **generated from it** — edit `SPEC.md` and regenerate, never edit the generated files. `--check` also validates the spec itself: unique requirement IDs with live-plus-retired numbering contiguous per prefix, every `params.*` citation resolving, every parameter section cited by some requirement, a `*(verify: ...)*` clause on every `SR`/`BR`/`ER`/`DR`/`HR`/`PR` line, and the `k = ratio × n` symbol-budget arithmetic. Run it after any spec edit; a stale generated file fails the check.

To retire a requirement, strike it through in place (`- ~~**G-3**~~ — reason`) under §14 rather than deleting it: the number stays reserved so live IDs are never renumbered. Passing `--check` means the document is *structurally* consistent — it says nothing about whether the experiment is scientifically valid.

**The amendment convention (§17).** Changing any requirement, decision, parameter or gate means adding an `AM-n` entry to §17 stating what changed, why, and who raised it, and adding an `(AM-n)` reference to the item you changed. Amendments are append-only: an amendment that is later revisited gets a *new* `AM` entry citing the old one, never an edit to the old one. Two formatting traps — `AM` entries must be a **single line** however long (the requirement regex is anchored and single-line, so a wrapped entry silently fails to parse and surfaces only as an ID-contiguity error), and a `` `params.x.y` `` written inside an `AM` entry is parsed as a real citation and must resolve, so an amendment describing a *rename* must quote the old key in plain text rather than in backticked `params.` form. The rationale is in §17's preamble: this document is read by examiners and external reviewers who never saw the previous version.

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
- `vault/capstone/` — Obsidian vault of course administrivia: proposal report template, thesis format, rubrics, the Fall 2026-27 circular (scanned images, no extractable text), and `Project requirements.md`. Deliverable formats live here; the `.obsidian/` directory is editor config, not content. `Capstone Project Rubrics.xlsx` is the grading scheme and drives the `PR` requirements — extract it by unzipping and reading `xl/sharedStrings.xml`. It scores First Review 10 / Second Review 30 / Third Review 40 / Project Report 20, with **Novelty worth 15** (the line DEC-13 exists to answer) and the review checkpoints at **W4 / W10 / W17** — resolved from the circular's own table, not the spreadsheet's 2023 template dates (AM-59). The circular is scanned images with *no extractable text*, which is not the same as unreadable: render its pages and read them.
- `spec/SPEC.md` — the project specification: thesis, preregistered hypotheses, decisions, parameters, requirements, schedule with go/no-go gates, non-goals, and the §16 open-items register. Normative and self-sufficient.
- `spec/DATASHEET.md`, `spec/concerns/`, `spec/params.generated.yaml` — generated views (see Commands above). `spec/concerns/programme.md` holds the `PR` course deliverables; `spec/concerns/amendments.md` is the §17 amendment record — what changed in the spec and why, the file to read before re-litigating a decision or acting on an external review; the others group `SR`/`BR`/`ER`/`DR`/`HR` by concern, with retired IDs shown under a "Retired" heading.
- `spec/evidence/` — supporting material for measured claims in the spec, currently the W0 LDPC spike behind AM-24 and AM-25: the machine-readable spike record, the scripts that produced it, and the golden-vector cross-check with its log. Not normative; it exists so the spec's numbers can be checked rather than trusted. Third-party vector data is **not** committed there — the directory carries checksums and a fetcher instead (AM-25), and `.gitignore` enforces it. Read its `README.md` before adding anything.
- `tools/gen_spec_views.py` — the generator and spec validator.
- `src/` — project code. `config/params.py` is the SR-1 loader every other module reads its constants through; `env.py` holds the CUDA assertion, the determinism settings and the run-metadata record (SR-21, SR-12). `data/adapters.py`, `identity.py`, `manifests.py`, `provenance.py` and `registry.py` implement AM-77; `data/classifier.py`, `models/reference_classifier.py` and `training/reference_classifier.py` implement AM-78's deterministic pre-G-1 classifier contract. The classifier normalizes inside the model, initializes from the keyed `init` identity, and uses a keyed seed/epoch permutation. Nothing here is a package: `pyproject.toml` puts `src` on `pythonpath` for pytest, so there is no install step.
- `data/manifests/` — the only tracked part of root `data/`: `imagenette160.csv`, `stl10.csv` and `cifar10.csv`. Downloaded archives, verified extractions and range/cache files remain ignored.
- `tests/` — twelve test modules plus `conftest.py`, run with `.venv/bin/python -m pytest`. Several exist because a comment would not have caught what they catch: `test_env.py` hard-asserts the CUDA build and OpenJPEG boundary; `test_cpu_lock.py` rejects CUDA distributions structurally; `test_doc_consistency.py` mutation-tests stale values and historical-plan banners; `test_artifact_rng.py` proves exact-key and control-flow invariance; `test_test_access.py` walks the import graph around the guarded test split; and the dataset/manifest/provenance tests use synthetic local sources so the default suite stays network-free.
- `requirements.in` → `requirements.lock`, and `requirements-cpu.in` → `requirements-cpu.lock` — the SR-21 environment locks, hashed and committed. Source files are hand-written from `params.environment`; the locks are generated (see Commands) and must not be hand-edited.
- `docs/` — hand-written background notes, not generated and not normative. `crossover-explained.md` explains why §2's crossover criterion was replaced, in plain language and then technically; it is written to feed the thesis discussion chapter and viva prep, and it is the thing to hand a supervisor who asks why the success criterion changed.

### Reading the planning documents

Neither `pdftotext` nor `pypdf` is installed system-wide, and `pip install` into the system Python is blocked (PEP 668). To read the PDFs and `.docx` files, create a venv in the scratchpad and install `pypdf` there; `.docx` files are zip archives whose text can be pulled from `word/document.xml`.
