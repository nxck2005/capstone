# Semantic Communication over Noisy Channels

Capstone project. Instead of the standard wireless pipeline — compress a source (JPEG),
separately protect the bits against noise (LDPC), and rebuild it bit-for-bit — this project
trains a neural **encoder** and **decoder** end-to-end through a differentiable channel model,
so that only what a downstream task needs survives the link. The technique is **deep joint
source-channel coding (DJSCC)**.

**The claim** is structural, not a tuning tweak: classical coding cannot express a task-success
objective, and Shannon's separation theorem is optimal only for infinitely long messages — so
short messages over noisy channels (edge/IoT links) are exactly where joint learned coding wins.
The signature is graceful degradation: classical coding hits a noise cliff and yields nothing,
while the semantic system gets blurrier but stays task-correct.

**Tier 1 deliverable** (simulation only, no radio hardware): on an image-classification task over
a simulated channel, the learned system's accuracy-vs-SNR curve crosses above a fairly-tuned
JPEG+LDPC baseline at low SNR, at matched bandwidth. Tiers 2 (offline SDR replay) and 3 (live
Raspberry Pi demo) are optional extensions; the project stands on Tier 1 alone.

## Specification

[`spec/SPEC.md`](spec/SPEC.md) is the normative source of truth — thesis, falsifiable success
criterion, parameters, requirements, schedule, and non-goals. The other files under `spec/` are
**generated** from it:

- [`spec/DATASHEET.md`](spec/DATASHEET.md) — every committed parameter, flattened.
- [`spec/concerns/`](spec/concerns/) — requirements grouped by concern (system, baseline,
  experiments, demo, hardware, roadmap).
- `spec/params.generated.yaml` — machine-readable parameters, consumed by the implementation.

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

Specification and tooling only — no implementation yet. See
[`CLAUDE.md`](CLAUDE.md) for how the repo is organized.

## License

MIT — see [`LICENSE`](LICENSE).
