# Semantic Communication over Noisy Channels

Capstone project. Instead of the standard wireless pipeline — compress a source (JPEG 2000),
separately protect the bits against noise (LDPC), and rebuild it bit-for-bit — this project
trains a neural **encoder** and **decoder** end-to-end through a differentiable channel model,
so that only what a downstream task needs survives the link. The technique is **deep joint
source-channel coding (DJSCC)**.

**The claim** is structural, not a tuning tweak: the task-agnostic reconstruction baseline has no
representation of what the bits are *for*, and Shannon's separation theorem is optimal only for
infinitely long messages — so short messages over noisy channels (edge/IoT links) are a regime
where separation pays finite-blocklength penalties and joint learned coding *may* gain. The
signature is graceful degradation: separated coding hits a noise cliff and yields nothing, while
the semantic system gets blurrier but stays task-correct. Because a raw learned-vs-classical gap
conflates *task-aware representation* with *joint coding*, a task-aware digital control system is
built alongside, so the gain can be attributed rather than assumed.

**Tier 1 deliverable** (simulation only, no radio hardware): an image-classification task over a
simulated channel, both systems bandwidth-matched and fully bit-accounted, evaluated under a
preregistered protocol with paired per-image inference. Tier 1 is complete when the experiment is
run properly — **not** conditional on the result going the hypothesis's way; a negative result is
reported with equal rigour. Tiers 2 (offline SDR replay) and 3 (live Raspberry Pi demo) are
stretch goals with a pre-recorded demonstration as the expected outcome; the project stands on
Tier 1 alone.

## Specification

[`spec/SPEC.md`](spec/SPEC.md) is the normative source of truth — thesis, completion criteria and
preregistered hypotheses, settled decisions, parameters, requirements, schedule with go/no-go gates,
non-goals, and an open-items register. The other files under `spec/` are **generated** from it:

- [`spec/DATASHEET.md`](spec/DATASHEET.md) — every committed parameter, flattened.
- [`spec/concerns/`](spec/concerns/) — requirements grouped by concern (system, baseline,
  experiments, demo, hardware, programme deliverables, roadmap).
- `spec/params.generated.yaml` — machine-readable parameters, consumed by the implementation.

[`docs/`](docs/) holds hand-written background notes that are *not* generated and *not* normative —
currently [`crossover-explained.md`](docs/crossover-explained.md), which explains in plain language
and then technically why the success criterion changed, and how the comparison is set up so that a
crossover is observable if one exists.

## Working with the spec

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python tools/gen_spec_views.py            # regenerate the derived views
python tools/gen_spec_views.py --check    # validate the spec + fail on stale generated files
```

Edit `spec/SPEC.md` and regenerate; never hand-edit the generated files. `--check` also validates
the spec (requirement-ID integrity, parameter citations, symbol-budget arithmetic) and is the
drift guard to run after any spec change.

## Status

Specification and tooling only — no implementation code yet, but **W0 is complete and W1 is open**.
Gate G-9 passed on 2026-07-27: the LDPC spike ran clean on the target hardware, and the golden
vectors match an independent MATLAB-derived reference bit-exactly. The spec has been through
repeated independent adversarial review and revised accordingly — [`spec/SPEC.md`](spec/SPEC.md) §17
records **nine amendment rounds** across 63 `AM` entries, and is the file to read before
re-litigating any decision. §16 records what is still provisional and which risks are being carried.
The most recent rounds (2026-07-28, `AM-57`..`AM-63`) answered the pre-implementation gate audit in
[`audit/`](audit/): it tightened all four preregistered hypotheses into uniquely executable form,
rewrote the packetisation evidence after finding it reported zero failures while breaking four rules
it claimed to enforce, and resolved the academic calendar.

Measured claims are backed by [`spec/evidence/`](spec/evidence/) rather than asserted: the W0 spike
record, the golden-vector cross-check, and a TS 38.212 packetisation conformance check that runs in
under a second with no GPU and no network. Both of the repository's checks are meant to be run, not
trusted:

```bash
.venv/bin/python tools/gen_spec_views.py --check       # 171 requirements, 10 generated files
.venv/bin/python spec/evidence/check_packetisation.py  # 215 feasible, 144 obligation, 0 failures
```

[`NEXT.md`](NEXT.md) is the short-lived working file for what happens next — read it first.
See [`AGENTS.md`](AGENTS.md) for how the repo is organized.

## License

MIT — see [`LICENSE`](LICENSE).
