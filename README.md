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

Implementation is underway: W1 batch 1 established the locked environment and repository scaffold,
batch 2 added the resolved run-configuration layer and SR-1 literal checker, batch 3 added the
content-addressed identity keys, the counter-based keyed RNG and the guarded test-split boundary,
and batch 4 (checkpoint `eba5bd2`) implements the canonical preprocessing contract. The
AM-72–76 W1 sweep remediation is committed as `8e59535`: complete versioned run
fingerprints, a genuinely CPU-only lock, source-bound preprocessing plus exact RNG/SSIM contracts,
honest OpenJPEG provisioning, and current-document consistency coverage. The AM-77 batch is
committed as `2c6f780`: one registry over Imagenette-160/STL-10/CIFAR-10,
real source-byte decoders, exact archive provenance, and deterministic committed split manifests.
The reference-classifier integrity implementation is committed as `89a3af4`: extraction-marker
binding, resumable archive fetches, AM-78's deterministic from-scratch classifier, model-owned
normalization, keyed initialization and epoch ordering, validation-only SGD, and atomic portable
checkpoints. **W1 is complete and validation-only G-1 passed on 2026-07-29.** The clean
Imagenette-160 campaign ran all 100 epochs from scratch and reached **898/1000 = 0.898 validation
top-1 at epoch 99**, above the preregistered 0.88 floor. The final and best checkpoint are both
`9c37362347a0203597d6e8e9d9a58fde30ba286f3cec9b4d2f800bd8a3256002`; the config hash is
`a9717575d71f2b3e9dd411b10b7735bdb3946c985fead48cb3c5af07423f12e1`. The four aggregate
classifier outputs plus the machine-readable `g1_adjudication.json` live under
`results/reference_classifier/`. The portable frozen-checkpoint path is
`checkpoints/reference_classifier/epoch-99.pt`. Only that final checkpoint was preserved externally:
GitHub Release `g1-reference-classifier-2026-07-29`, asset
`reference-classifier-imagenette160-epoch99-9c37362347a0203597d6e8e9d9a58fde.pt`, with the exact
SHA-256 above. The other 99 ignored checkpoints were not uploaded, and no training was rerun during
the evidence-hardening cleanup. **W2 and G-7 are complete.** Implementation commit
`26b631ede27a6f88f1d004a66b845c52a658e07c` provides native-complex AWGN, per-image
unit-power normalization, keyed complex noise, symbol-domain PAPR and capped-power projection,
`djscc_residual_v1`, the task-head registry, config-derived loss, and parameter caps. The clean
corrected, implementation-bound Imagenette `r_1_2` CUDA profile completed batch 32 in 48.684 s,
reserved 1.004 GiB, projects 100 epochs to 1.352 h, and measured 1,640,957 parameters. Every
critical imported project module is recorded by resolved path, executed-byte SHA-256 and immutable
W2 git blob SHA. The machine-readable report lives under `results/profiling/` and verifies offline.
The validation-only JPEG 2000 transparency-bitrate probe is also complete. It loaded the exact
frozen classifier above, reproduced **898/1000 = 0.898** on the uncompressed validation view, then
evaluated 1,000 stable validation IDs across 17 frozen byte budgets and the 160/128/96/64 encode
axes: 68,000 cells, with zero infeasible encodes and zero decode failures. OpenJPEG 2.5.4 through
Glymur 0.14.3 used raw codestreams, irreversible 9/7, RPCL, six resolutions, 64×64 code blocks and
whole-image tiles. The selection-aware paired bootstrap forecasts the 5 pp
`probe_efficiency_threshold` at **1,330 bytes** (axis 128, mean 0.408654 bpp, 0.870 accuracy,
one-sided 95% LCB −0.041) and the 2 pp `probe_crossover_threshold` at **3,200 bytes** (axis 160,
mean 0.987788 bpp, 0.886 accuracy, LCB −0.018). Neither result is censored. These are engineering
forecasts, not G-8 operating-point selections: G-8 remains unresolved, no training ran, and the
test split stayed sealed. Evidence lives under `results/probes/transparency_bitrate/` and verifies
with `tools/verify_transparency_bitrate_probe.py`.
**W3 is complete and G-2 passed.** Implementation commit
`968e907237bbe571adf6ec48e4711ea021831719` provides the local transport/segmentation/CRC layer,
Sionna `2.0.1` adapter, BPSK/QPSK/16-QAM mapping and soft demapping, exact modulation interleaving,
the independent flooding offset-min-sum reconstruction and executable runtime packetisation.
srsRAN `release_25_10` exact vectors and the project-owned BG2/Z2 offline floor match bit-exactly.
At BLER `1e-2`, measured waterfall displacements were **0.0 dB BPSK**,
**+0.0036302379 dB QPSK**, and **0.0 dB 16-QAM**, each inside the 0.5 dB gate. Runtime metadata
reconciled all 216 configured packetisation cells, with all 144 headline obligations feasible and
the one preregistered smoke infeasibility classified explicitly. Evidence lives under
`results/baseline/g2/` and verifies with `tools/verify_g2_adjudication.py`. No image sweep, training,
G-8 selection or test access occurred.

The single next engineering task is **bounded W4 classical-baseline integration required before
G-8**. G-8 has not started.
Gate G-9 passed on 2026-07-27: the LDPC spike ran clean on the target hardware, and the golden
vectors match an independent MATLAB-derived reference bit-exactly. The spec has been through
repeated independent adversarial review and revised accordingly — [`spec/SPEC.md`](spec/SPEC.md) §17
records **sixteen amendment rounds** across 79 `AM` entries, and is the file to read before
re-litigating any decision. §16 records what is still provisional and which risks are being carried.
The 2026-07-28 rounds answered the pre-implementation gate audit in [`audit/`](audit/), built the
environment and config foundation, tightened all four preregistered hypotheses into uniquely
executable form, rewrote packetisation evidence that had passed while breaking four rules, and
resolved the academic calendar.

Measured claims are backed by [`spec/evidence/`](spec/evidence/) rather than asserted: the W0 spike
record, the golden-vector cross-check, and a TS 38.212 packetisation conformance check that runs in
under a second with no GPU and no network. The repository's checks are meant to be run, not trusted:

```bash
.venv/bin/python tools/gen_spec_views.py --check       # 187 requirements, 10 generated files
.venv/bin/python tools/check_doc_consistency.py        # current hand-written documentation agrees
.venv/bin/python tools/check_literals.py               # no parameter-valued source literals
.venv/bin/python spec/evidence/check_packetisation.py  # 215 feasible, 144 obligation, 0 failures
.venv/bin/python tools/verify_cpu_lock.py --clean-install
.venv/bin/python tools/fetch_datasets.py --check       # exact archive length + SHA-256
.venv/bin/python tools/materialize_manifests.py --check
.venv/bin/python tools/verify_datasets.py              # real train/val smoke; zero test decode/canonicalization
.venv/bin/python tools/train_reference_classifier.py --config configs/reference-classifier-clean.yaml --dataset imagenette160 --device cuda --smoke-steps 3 --smoke-val-batches 2  # bounded smoke; never G-1 evidence
.venv/bin/python tools/verify_g1_adjudication.py        # offline: epochs, counts, hashes, floor, lineage and checkpoint identity
.venv/bin/python tools/verify_g7_profile.py             # offline: clean commit, CUDA profile, caps, limits and training-only scope
.venv/bin/python tools/verify_transparency_bitrate_probe.py
.venv/bin/python tools/verify_g2_adjudication.py
.venv/bin/python -m pytest
```

The tracked manifests and their exact SHA-256 values are:

| Dataset | Manifest | Train / val / test | SHA-256 |
|---|---|---:|---|
| Imagenette-160 | `data/manifests/imagenette160.csv` | 8469 / 1000 / 3925 | `224309422f15bf89460559381aea4b00c4779c52d3652f7f679a213369f3f889` |
| STL-10 | `data/manifests/stl10.csv` | 4500 / 500 / 8000 | `67936da779dc0010160b37b3b40001490304a5873eb978d261e3a57947387b47` |
| CIFAR-10 | `data/manifests/cifar10.csv` | 45000 / 5000 / 10000 | `09e9debf4743831ca61f17154a997e60becdd7046a585bdbd94b5db4bf12a537` |

Downloaded archives and extracted datasets stay ignored. Their normative URL, filename, exact byte
length and SHA-256 are pinned under `params.datasets` in the generated datasheet and verified before
any sample or manifest scan is allowed.

[`NEXT.md`](NEXT.md) is the short-lived working file for what happens next — read it first.
See [`AGENTS.md`](AGENTS.md) for how the repo is organized.

## License

MIT — see [`LICENSE`](LICENSE).
