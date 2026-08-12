# Project Knowledge Transfer

This document is for someone who has just joined the project and knows nothing about it.

It explains the idea first, then the experiment, then the codebase. It deliberately uses simple
language. You do not need a communications background, a machine-learning background, or knowledge
of the repository to begin here.

## 1. The project in one minute

Imagine a small camera at the edge of a network. The camera sees an image. A server on the other
side of a wireless link must classify that image.

A normal communication system does this:

```text
image
  -> compress the image
  -> turn it into protected bits
  -> send the bits through noise
  -> recover the bits
  -> rebuild the image
  -> classify the rebuilt image
```

This project asks whether we can do this instead:

```text
image
  -> learn the information that is useful for classification
  -> send that representation through noise
  -> classify it at the receiver
```

The learned system may not need to preserve every pixel perfectly. It only needs to preserve enough
information for the receiver to do its task.

The project tests this idea using image classification over a simulated noisy channel. It compares
three systems at the same communication budget:

1. A strong conventional image-transmission system.
2. A digital system that transmits learned task features.
3. A learned joint source-channel coding system.

The project is successful if this comparison is implemented, run, and reported correctly. The
learned system is not required to win.

## 2. What problem are we solving?

An edge device can have limited bandwidth, limited power, and an unreliable connection. Sending a
complete high-quality image can be expensive. It may also be unnecessary when the receiver only
needs a narrow answer such as an image class.

For example, the receiver may only need to decide whether an image contains a dog, a truck, or a
building. It may not need every texture and background detail.

The central question is:

> At the same bandwidth and channel conditions, can a task-aware learned communication system
> preserve classification accuracy better than a properly tuned conventional system?

This is an experimental question. The repository does not assume the answer is yes.

## 3. What does “semantic communication” mean here?

The word “semantic” can sound vague. In this project it has a specific meaning:

> Transmit information that is useful for the receiver's task, rather than treating perfect source
> reconstruction as the only objective.

The task is image classification. The sender sees the image. The receiver owns the classifier. The
sender and receiver are separated by a noisy, bandwidth-limited channel.

This project does not study language semantics, large language models, agents, or reinforcement
learning.

It uses supervised learning. A neural encoder and neural decoder are trained end to end through a
differentiable channel model.

## 4. Why not send only the class label?

This is an important objection.

There are ten image classes. A complete class label needs only four bits. If the sender were allowed
to run the entire classifier, it could send only the label. That would be much cheaper than sending
an image or a learned representation.

The project therefore fixes the deployment split:

- The sender runs an encoder.
- The receiver owns the task head.
- The sender is not allowed to replace the communication problem by running the full receiver task.

This models cases where the receiver's task changes, where the task head is private or centrally
managed, or where the edge device cannot run the full model.

The project reports the label-only bound so this assumption is visible. It does not pretend the
objection does not exist.

## 5. The three systems

The three-way comparison is the core of the project.

### 5.1 System A: conventional image transmission

This system sends a compressed image through a digital communication chain.

```text
canonical image
  -> JPEG 2000 compression
  -> packet framing and CRC checks
  -> 5G NR LDPC channel coding
  -> BPSK, QPSK, or 16-QAM modulation
  -> simulated AWGN channel
  -> soft demodulation and LDPC decoding
  -> JPEG 2000 reconstruction
  -> frozen image classifier
```

The system is allowed to tune its JPEG 2000 quality, LDPC rate, modulation, and image downsampling
on validation data. It is not intentionally weak.

This system is called the classical or separated baseline because image compression and channel
protection are separate stages.

### 5.2 System B: task-aware digital features

This system transmits learned features instead of image pixels.

```text
canonical image
  -> learned feature encoder
  -> quantization into digital values
  -> the same digital channel-coding and modulation chain
  -> receiver task head
```

This system answers an attribution question.

Suppose the learned joint system beats the conventional image system. That difference could come
from either of two advantages:

- It sends task-aware information instead of reconstruction-oriented information.
- It learns source representation and channel protection jointly.

System B has the first advantage but not the second. Comparing against it helps determine where a
gain came from.

### 5.3 System C: learned joint source-channel coding

This is the main learned system. It is called DJSCC: deep joint source-channel coding.

```text
canonical image
  -> neural encoder
  -> fixed number of complex channel symbols
  -> power normalization
  -> simulated AWGN channel
  -> neural decoder
       -> reconstructed image
       -> classification logits
```

The noisy channel is inside the neural network's training path. Gradients pass through it. The
encoder and decoder therefore learn together.

The decoder has two outputs:

- A reconstruction head, which produces an image.
- A task head, which predicts the class.

The training loss combines classification loss and reconstruction loss:

```text
total loss = classification loss + lambda * reconstruction loss
```

The value of `lambda` controls the trade-off. It is selected later using validation data under a
predefined rule.

## 6. What makes the comparison fair?

A learned system can appear better if the baseline is weak or if the systems receive different
resources. This project has explicit controls to prevent that.

### 6.1 Same communication budget

All systems are compared using the same number of complex channel symbols, called `k`.

A complex symbol contains a real part and an imaginary part. Wireless communication systems use
these two dimensions to carry information. The project treats one complex symbol as one channel
use.

The project has six bandwidth ratios, from relatively generous to very small. Each ratio maps to an
exact value of `k` for each dataset.

### 6.2 Same noise definition

All systems use the same signal-to-noise ratio definition:

```text
Es/N0 in dB per complex channel use
```

The transmitted symbols are normalized before noise is added. This makes the SNR request mean the
same thing across systems.

### 6.3 Same image and channel realization

Rows are paired by image and channel condition. When two systems are compared on one image, they
receive noise derived from the same stable identity.

The random noise is generated from content-based keys. It does not change when rows are reordered,
batched differently, or skipped by another system.

### 6.4 Strong conventional baseline

The baseline may tune all of these on validation data:

- JPEG 2000 quality and downsample size.
- LDPC code rate.
- BPSK, QPSK, or 16-QAM modulation.

This is important. The project does not compare the learned model with a fixed or deliberately poor
digital configuration.

### 6.5 Every failure remains in the denominator

A conventional row can end in one of four states:

- `structural_infeasibility`: the requested packet cannot be represented legally.
- `codec_infeasibility`: JPEG 2000 cannot fit an image into the available bytes.
- `decode_failure`: the transmission occurred, but the receiver failed its CRC checks.
- `delivered`: the image was recovered and decoded.

Failed rows are not deleted. They contribute through a predefined outage policy.

### 6.6 Exact byte and symbol accounting

The project counts more than the compressed image bytes. It also counts CRCs, code-block overhead,
filler bits, rate matching, and all emitted JPEG 2000 structure.

This prevents the baseline from silently sending more information than the learned system.

### 6.7 Validation and test are separate

Validation data is used to choose settings. Test data is used once, after every choice is frozen.

The test split is currently sealed in code. The only module allowed to load a test sample is
[`src/data/test_access.py`](../src/data/test_access.py), and that module refuses access until the
required freeze record exists at gate G-12.

## 7. The data

The repository supports three datasets, but they have different jobs.

| Dataset | Role | What to remember |
|---|---|---|
| Imagenette-160 | Headline scientific dataset | Ten classes; the reference classifier and primary experiment use this dataset. |
| STL-10 | Fallback headline dataset | Available if the main dataset cannot support the planned study. |
| CIFAR-10 | Smoke and plumbing only | Used to test transport, cache, accounting, and failure paths. |

The frozen reference classifier is an Imagenette-160 classifier. It must never be used to claim
CIFAR-10 task accuracy. Both datasets have ten numeric labels, but the labels mean different things.

Each sample has a stable ID derived from its original source bytes. Split manifests are committed
under [`data/manifests/`](../data/manifests/). This keeps image identity and train/validation/test
membership stable across machines and library versions.

## 8. The channel model

Tier 1 uses AWGN: additive white Gaussian noise.

In simple terms, the channel adds random complex noise to every transmitted symbol.

```text
received symbol = transmitted symbol + noise
```

AWGN is deliberately simple. It lets the project study the communication method without adding
timing errors, frequency offsets, multipath fading, radio clipping, or hardware calibration.

The project does not claim that an AWGN result automatically transfers to a real radio.

Real SDR replay is a later stretch goal, not a requirement for the main scientific result.

## 9. JPEG 2000, LDPC, modulation, and BLER

These terms appear throughout the repository.

### JPEG 2000

JPEG 2000 is the conventional image codec used by the baseline. The project uses OpenJPEG 2.5.4
through a Python binding. It emits raw JPEG 2000 codestreams rather than ordinary `.jp2` files.

JPEG 2000 was chosen because it supports low-rate image coding and is stronger than using a basic
JPEG setting as the main comparator.

### CRC

A cyclic redundancy check is a small checksum added to transmitted data. The receiver uses it to
detect whether decoding succeeded.

### LDPC

Low-density parity-check coding adds structured redundancy so corrupted data can be recovered. The
project uses a 5G NR LDPC coding and rate-matching chain derived from 3GPP TS 38.212.

The project does not implement a complete 5G radio link. It does not claim NR scheduling, OFDM,
HARQ, synchronization, or full 5G conformance.

### Modulation

Modulation maps bits to complex channel symbols.

- BPSK carries one bit per symbol.
- QPSK carries two bits per symbol.
- 16-QAM carries four bits per symbol.

Higher-order modulation carries more bits but normally needs a cleaner channel.

### BER and BLER

BER is bit error rate: the fraction of decoded information bits that are wrong.

BLER is block error rate: the fraction of code blocks that fail.

The baseline needs BLER measurements for every physical-layer configuration it may select. A
missing BLER value is unknown. It is never treated as zero.

## 10. What is the project trying to observe?

The expected behavior is a hypothesis, not a guaranteed result.

A separated digital system may show a decoding cliff. Above some SNR it works well. Below that
region, block decoding may fail sharply.

A learned joint system may degrade more gradually because it does not require exact bit recovery.
It may preserve task-relevant information even when its reconstruction becomes worse.

The project measures top-1 classification accuracy against SNR. It also records reconstruction
quality, transmission failures, symbol energy, PAPR, and system configuration.

## 11. The hypotheses in plain language

The exact statistical rules live in [`spec/SPEC.md`](../spec/SPEC.md) §2. Do not implement a
statistical decision from this summary alone.

### H1: low-SNR separation

At the main operating ratio, the learned system should outperform the adaptive classical baseline
over a sustained low-SNR region.

This is the primary confirmatory hypothesis. It uses paired per-image outcomes and a calibrated run
rule. One lucky SNR point is not enough.

### H2: graceful degradation versus a cliff

Over a validation-selected SNR window, a fixed classical system should show a large accuracy drop
while the learned system shows a smaller drop.

### H3: convergence at high SNR

As SNR improves, the accuracy difference between the learned and adaptive classical systems should
contract toward zero.

A curve crossing is reported if it happens. It is not required for project success.

### H4: attribution

The learned joint system is also compared with the task-aware digital feature system.

If it beats the image baseline but not the digital feature system, the gain is attributed mainly to
task-aware representation. Joint coding receives credit only for the remaining advantage over the
digital feature control.

## 12. What counts as success?

Project completion and scientific outcome are separate.

Tier 1 is complete when:

- The three systems are implemented.
- Their resources are matched.
- The conventional baseline is properly tuned on validation.
- The learned settings are frozen using validation only.
- One sealed test campaign is run.
- Paired results and all failures are reported.
- Positive, null, and negative findings are handled using the same protocol.

The project does not need to prove that the learned system is always better.

## 13. What has already been completed?

The repository has completed the foundation needed before the final comparison.

### Environment and data foundation

- Reproducible CUDA and CPU dependency locks exist.
- Dataset archives and manifests have pinned checksums.
- Canonical image preprocessing is implemented.
- Stable artifact IDs and deterministic keyed randomness are implemented.
- The test-access boundary is implemented and tested.

### G-1: reference classifier

The Imagenette-160 classifier was trained from scratch for 100 epochs. It achieved 898 correct
predictions out of 1,000 validation images, or 89.8%, above its preregistered validation floor.

The test split was not used.

### G-7: learned-system feasibility

The DJSCC architecture is implemented and has been profiled on the available RTX 4060 Laptop GPU.
Its parameter count, epoch time, and memory use fit the planned training limits.

This was a feasibility profile, not final DJSCC training.

### G-2: digital physical-layer conformance

The LDPC, packetisation, modulation, and reference BLER path passed its conformance gate. Golden
vectors and independent reference curves agree within the specified tolerance.

G-2 covers a small reference configuration. It is not the full BLER table needed by the adaptive
baseline.

### W4: conventional pipeline integration

The classical chain runs end to end through JPEG 2000, packetisation, LDPC, modulation, AWGN,
decoding, reconstruction, outage handling, classification records, and verification.

The committed W4 run is bounded integration evidence. It is not the final scientific sweep.

### G8_A and G8_B

G8_A froze the campaign structure, candidate grid, required BLER identities, state model, and
preflight rules.

G8_B built the authenticated runner, crash-safe evidence publication, resume machinery, independent
verifiers, and a bounded non-scientific smoke test.

### G8_C: current scientific phase

G8_C is measuring the full physical-layer BLER table needed by the adaptive classical baseline.
The campaign is paused at a durable checkpoint. No worker is currently running.

Exact coverage, the next legal attempt, and the permitted restart sequence change as the campaign
progresses. They are intentionally recorded only in [`instructions/RESUME.md`](../instructions/RESUME.md)
and the authenticated campaign state. Do not copy those counts into another status document.

## 14. What has not happened yet?

As of the current handoff:

- The final G8 BLER table has not been frozen.
- Classical operating points have not been selected.
- Full validation measurement has not run.
- Final DJSCC training has not run.
- The task-aware digital feature system has not been evaluated.
- The artifact-finetuned classifier has not been released.
- No learned-versus-classical headline result exists.
- The test split has not been opened.
- No SDR or Raspberry Pi experiment has run.

Do not describe bounded smoke rows, validation probes, or G-2 conformance curves as final comparison
results.

## 15. What happens next?

The high-level order is:

```text
G8_C
  complete full BLER characterization
  freeze the measured BLER table

G8_D
  build validation-measurement tooling
  run bounded smoke tests

G8_E
  run full validation measurement
  select classical operating points in pass one

G8_F
  build a training-only artifact corpus
  fine-tune the artifact classifier
  run the one permitted second selection pass

G8_G
  adjudicate and freeze the G-8 outputs

later work
  calibrate the learned loss
  train final learned systems
  build and evaluate the digital feature control
  freeze everything at G-12
  run one test campaign
  report results and build the demo
```

The live next action is always defined by [`NEXT.md`](../NEXT.md). For G8 work, the exact operational
cursor is [`instructions/RESUME.md`](../instructions/RESUME.md).

## 16. Why is the repository so strict?

The final experiment makes many choices: codec settings, code rate, modulation, SNR, model
checkpoint, operating ratio, failure policy, and statistical method.

If these choices are changed after seeing results, the comparison becomes hard to trust.

The repository therefore records:

- The exact configuration used for each run.
- The source files that produced important evidence.
- The Git commit and whether the tree was dirty.
- Content hashes for configurations, requests, results, and contracts.
- Stable image and noise identities.
- Phase transitions and permissions.
- Whether validation, inference, training, or test access occurred.

This machinery is not the scientific idea. It protects the scientific idea from accidental or
result-driven changes.

## 17. The four layers of the repository

It helps to think of the project in four layers.

### Layer 1: the idea

Task-oriented communication may use a limited noisy link more effectively than pixel-perfect image
transmission for remote classification.

Start with this document and [`README.md`](../README.md).

### Layer 2: the experiment

The experiment defines the three systems, resource matching, datasets, validation/test separation,
hypotheses, and completion criteria.

The authority is [`spec/SPEC.md`](../spec/SPEC.md).

### Layer 3: the implementation

The code implements preprocessing, models, channels, the classical pipeline, training, records, and
verification.

The main implementation is under [`src/`](../src/).

### Layer 4: governance and provenance

Campaign contracts, hashes, state machines, manifests, gates, and resume rules ensure that evidence
is complete and reproducible.

These live mainly under [`results/`](../results/), [`instructions/`](../instructions/), and
[`tools/`](../tools/).

Do not begin learning the project from Layer 4. Understand Layers 1 and 2 first.

## 18. Repository map

### Top-level files

| Path | Purpose |
|---|---|
| [`README.md`](../README.md) | Project summary, major completed evidence, and common commands. |
| [`NEXT.md`](../NEXT.md) | Short-lived handoff describing what happens next. Read this at the start of every session. |
| [`AGENTS.md`](../AGENTS.md) | Detailed rules for agents and contributors working in this repository. |
| [`requirements.lock`](../requirements.lock) | Exact CUDA runtime dependency lock. |
| [`requirements-cpu.lock`](../requirements-cpu.lock) | Exact CPU analysis dependency lock. |

### Specification

| Path | Purpose |
|---|---|
| [`spec/SPEC.md`](../spec/SPEC.md) | Normative source of truth. Requirements, parameters, decisions, hypotheses, gates, risks, and amendment history. |
| [`spec/params.generated.yaml`](../spec/params.generated.yaml) | Machine-readable parameters generated from the spec. Runtime code reads this file. |
| [`spec/DATASHEET.md`](../spec/DATASHEET.md) | Generated flattened view of all parameters. |
| [`spec/concerns/`](../spec/concerns/) | Generated requirement views grouped by topic. |
| [`spec/evidence/`](../spec/evidence/) | Small conformance records and scripts, especially packetisation and LDPC checks. |

Never edit generated spec views directly. Edit `spec/SPEC.md`, add an amendment when required, and
regenerate them.

### Source code

| Path | Purpose |
|---|---|
| [`src/config/`](../src/config/) | Loads parameters and produces complete run configurations and hashes. |
| [`src/data/`](../src/data/) | Dataset registry, source-byte decoding, manifests, preprocessing, classifier loading, and the test-access guard. |
| [`src/channels/`](../src/channels/) | AWGN, symbol-power normalization, PAPR, and the channel registry. |
| [`src/models/`](../src/models/) | DJSCC encoder/decoder, reference classifier, frozen classifier wrapper, and task heads. |
| [`src/training/`](../src/training/) | Reference-classifier training and DJSCC loss functions. |
| [`src/baseline/j2k.py`](../src/baseline/j2k.py) | JPEG 2000 codec wrapper and budget search. |
| [`src/baseline/ldpc/`](../src/baseline/ldpc/) | CRCs, segmentation, LDPC adapter, rate matching, modulation, and transport construction. |
| [`src/baseline/classical/`](../src/baseline/classical/) | End-to-end classical image path, records, outage handling, and operating-point composition. |
| `src/baseline/g8_*` | G8 campaign enumeration, work units, authenticated state, runner, resume logic, and characterization. |
| [`src/artifacts/`](../src/artifacts/) | Stable IDs and deterministic keyed random streams. |
| [`src/probes/`](../src/probes/) | Validation-only engineering probes. |

### Commands, tests, and evidence

| Path | Purpose |
|---|---|
| [`tools/`](../tools/) | Command-line entry points, evidence generators, verifiers, migrations, and campaign coordinators. |
| [`tests/`](../tests/) | Unit, integration, mutation, provenance, and failure-path tests. |
| [`configs/`](../configs/) | Human-written experiment choices. These are resolved against generated parameters before use. |
| [`results/`](../results/) | Committed evidence, adjudications, source manifests, and campaign state. |
| [`worklogs/`](../worklogs/) | Historical engineering records for completed work. |
| [`instructions/`](../instructions/) | Durable phase protocols and the live G8 recovery ledger. |
| [`docs/`](../docs/) | Human-written explanations, literature review, Gantt plan, deployment dossier, and this document. |
| [`deliverables/`](../deliverables/) | Review and final-delivery packages. |

## 19. How configuration works

Scientific constants do not belong as unexplained numbers in source code.

The intended flow is:

```text
spec/SPEC.md
  -> tools/gen_spec_views.py
  -> spec/params.generated.yaml
  -> human experiment config under configs/
  -> fully resolved RunConfig
  -> config_hash
  -> archived beside results
```

A human config chooses things such as dataset, ratio, channel, modulation, or code rate. The runtime
combines those choices with all relevant generated parameters. The resulting complete configuration
gets a content hash.

If a scientific parameter changes, the hash changes. Old evidence remains tied to the old
configuration.

## 20. How randomness works

Normal global random-number generators are sensitive to call order. That is dangerous when two
systems have different control flow.

This repository uses keyed random streams. A draw is a function of its purpose and identity.

Examples include:

- Model initialization from the training seed and component path.
- Batch order from the training seed and epoch.
- Augmentation from the image ID, training seed, and epoch.
- Channel noise from a content-derived `noise_id`.

This means that reordering rows does not silently change the noise assigned to an image.

## 21. How evidence works

Important evidence usually has several parts:

- Raw or aggregate output.
- A resolved configuration.
- A summary or adjudication file.
- Hashes of output files.
- A manifest of the source files that produced it.
- An offline verifier that recomputes important claims.

Do not trust a JSON field merely because it says `PASS`. Read the verifier to see what it recomputes.

Do not regenerate a historical source manifest to make changed code appear compatible with old
measurements. A source mismatch may mean the evidence no longer describes the current implementation.

## 22. Safe first-day setup

Read these files in order:

1. This document.
2. [`README.md`](../README.md).
3. [`NEXT.md`](../NEXT.md).
4. [`spec/SPEC.md`](../spec/SPEC.md) §1–3.
5. The relevant requirement section for the component you will change.
6. [`instructions/RESUME.md`](../instructions/RESUME.md) only if you are working on the active G8 campaign.

Create or sync the runtime environment using the commands in [`AGENTS.md`](../AGENTS.md). On this
machine, project commands run with `.venv/bin/python`.

On a fresh clone, fetch the ignored third-party LDPC fixture before running the full test suite:

```bash
.venv/bin/python tools/fetch_ldpc_golden_vectors.py
.venv/bin/python -m pytest
```

Useful read-only checks are:

```bash
.venv/bin/python tools/gen_spec_views.py --check
.venv/bin/python tools/check_doc_consistency.py -v
.venv/bin/python tools/check_literals.py -v
.venv/bin/python spec/evidence/check_packetisation.py
.venv/bin/python tools/fetch_datasets.py --check
.venv/bin/python tools/materialize_manifests.py --check
.venv/bin/python tools/verify_g1_adjudication.py
.venv/bin/python tools/verify_g7_profile.py
.venv/bin/python tools/verify_g2_adjudication.py
.venv/bin/python tools/verify_w4_baseline_integration.py
```

The full test suite requires the CUDA environment. This repository intentionally rejects a CPU-only
build in its main environment test.

This machine is WSL2. GPU availability is checked through `/dev/dxg` and PyTorch CUDA initialization,
not by looking for `/dev/nvidia*`.

## 23. Rules before changing code

### Read the authority first

Read the relevant requirements in `spec/SPEC.md`. The spec wins if another document disagrees with
it.

### Do not hard-code scientific values

Runtime code must read generated parameters or resolved configuration. Run
`tools/check_literals.py` after source changes.

### Do not open the test split

Test access is forbidden before G-12. Do not import or bypass `src/data/test_access.py`.

### Do not weaken the baseline

Do not remove modulation choices, reduce baseline tuning, drop failed rows, or substitute a simpler
codec to make the learned system look better.

### Do not treat smoke tests as science

Smoke tests prove that code paths work. They do not establish scientific performance.

### Do not rewrite history

Historical mistakes are recorded and corrected. They are not erased, rebased away, or silently
described as if they never happened.

### Amend the spec when the science changes

If you change a requirement, parameter, decision, or gate, append a new `AM-n` record in
`spec/SPEC.md` and add the amendment reference to the changed item.

Implementation fixes that restore already specified behavior may not need an amendment. Record the
reasoning either way.

### Preserve active campaign evidence

Do not edit raw G8 work-unit evidence, registered contracts, or frozen epoch-1 sources. Do not run a
G8 worker from an old command copied from history.

For G8_C, follow the exact inspect, reconcile, marker, push-parity, and restart sequence in
`instructions/RESUME.md`.

## 24. Things that sound reasonable but are wrong here

### “Semantic communication means an LLM understands the message.”

No. Here it means task-oriented image communication for classification.

### “This is reinforcement learning.”

No. The DJSCC model is trained with supervised losses through a differentiable channel.

### “The learned system only needs to beat JPEG.”

No. The headline baseline is JPEG 2000 plus a tuned digital physical layer, and the project also
includes a task-aware digital control.

### “If no BLER record exists, the link probably works at high SNR.”

No. Missing BLER evidence means uncharacterized. The candidate is ineligible.

### “The small G-2 BLER table can be reused everywhere.”

No. G-2 is a conformance check for one physical identity. G8_C measures the complete table needed by
the adaptive baseline.

### “CIFAR-10 also has ten classes, so the Imagenette classifier can score it.”

No. The class meanings differ. CIFAR-10 is transport smoke only in this repository.

### “Failed decodes can be excluded because no prediction was produced.”

No. That would reward a system for failing. Failed rows remain in the denominator through the outage
policy.

### “Validation and test are both held-out data, so either is fine for tuning.”

No. Validation is used for tuning. Test is used once for final reporting.

### “A crossover must exist for the project to pass.”

No. A crossover is descriptive. Learned dominance, classical dominance, or no clear difference can
all be reported.

### “The project claims lower energy use.”

No measured energy saving is currently claimed. Systems are compared at equal channel uses and
measured aggregate symbol energy under the simulation model.

### “Hardware is required to complete the capstone.”

The registered Tier 1 path is simulation-first. SDR and Raspberry Pi work are stretch goals. The
guide's dated acknowledgement of this path is still a human deliverable for the First Review.

### “A passing test means the science is correct.”

No. Tests show that declared properties hold. They do not prove that the research question,
assumptions, or statistical interpretation are scientifically valid.

## 25. Suggested path for a new contributor

### Day 1: understand the problem

- Read §§1–12 of this document.
- Read `spec/SPEC.md` §§1–3.
- Explain the three systems in your own words.
- Explain why System B is necessary.
- Explain why the test split is sealed.

Do not begin with G8 state contracts or runner code.

### Day 2: trace one sample through the code

For the learned path, read:

1. `src/data/preprocessing.py`
2. `src/models/djscc.py`
3. `src/channels/awgn.py`
4. `src/training/djscc_loss.py`

For the conventional path, read:

1. `src/data/preprocessing.py`
2. `src/baseline/j2k.py`
3. `src/baseline/ldpc/transport.py`
4. `src/baseline/classical/channel_transport.py`
5. `src/baseline/classical/pipeline.py`
6. `src/baseline/classical/records.py`

### Day 3: understand one evidence package

Start with G-1 or G-7. Read the adjudication JSON and its verifier together.

Then answer:

- Which claims are stored?
- Which claims are recomputed?
- Which source files are bound?
- Which dataset split was accessed?
- What would make verification fail?

### Day 4: run safe checks

Run the read-only checks in §22. Run a small relevant test module. Do not start a production campaign.

### Day 5: choose a bounded contribution

Good first contributions include:

- Improving plain-language documentation.
- Adding a focused unit test for an existing contract.
- Improving an error message without changing scientific behavior.
- Tracing and documenting a single data or configuration path.
- Fixing an isolated bug with a regression test after confirming its requirement.

Avoid choosing a first contribution that changes experiment parameters, active G8 evidence, test
access, the baseline search space, or hypothesis logic.

## 26. Questions every contributor should be able to answer

Before changing scientific code, you should be able to answer these:

1. What is the downstream task?
2. Why are there three systems rather than two?
3. What is held equal across systems?
4. What can be tuned on validation?
5. Why is the test split sealed?
6. What is the difference between G-2 and G8_C BLER evidence?
7. Why do failed rows remain in the denominator?
8. What does `k` mean?
9. What is AWGN?
10. What result would count as project completion?
11. Where is the normative specification?
12. Where is the live operational cursor?

Short answers:

1. Imagenette-160 image classification over a noisy link.
2. To separate task-aware representation from joint source-channel coding.
3. Image identity, split, channel uses, SNR convention, and paired noise identity.
4. Baseline settings, learned hyperparameters, checkpoints, and operating choices defined by the spec.
5. To prevent tuning to the final evaluation data.
6. G-2 is small conformance evidence; G8_C builds the complete measured table for selection.
7. Excluding them would bias accuracy upward.
8. The number of complex channel symbols available to transmit one image.
9. A simple channel that adds Gaussian noise to complex symbols.
10. A correct, frozen, fairly matched, one-time evaluation and honest report, regardless of winner.
11. `spec/SPEC.md`.
12. `NEXT.md`, and `instructions/RESUME.md` for active G8 execution.

## 27. Glossary

| Term | Simple meaning |
|---|---|
| AWGN | A channel that adds independent Gaussian noise. |
| Baseline | The conventional system used for comparison. |
| BER | Fraction of information bits decoded incorrectly. |
| BLER | Fraction of transmitted code blocks that fail. |
| Canonical image | The one fixed preprocessed image given to both scientific arms. |
| Channel use | One transmitted complex symbol. |
| Checkpoint | Saved neural-model state. |
| Classical adaptive | The conventional baseline tuned separately at each SNR on validation. |
| Codec | Software that compresses and reconstructs media. JPEG 2000 is the headline codec here. |
| Complex symbol | A transmitted value with real and imaginary parts. |
| Config hash | Content ID for a complete resolved experiment configuration. |
| CRC | Checksum used to detect a decoding failure. |
| DJSCC | Deep joint source-channel coding. A neural encoder and decoder trained through a channel. |
| Evidence manifest | A record of files, hashes, and source versions behind a result. |
| G-1, G-2, etc. | Gates that must pass before later work is allowed. |
| G8 | The campaign that fully characterizes and selects the conventional baseline. |
| Imagenette-160 | Ten-class image dataset used for the headline task. |
| Joint coding | Learning representation and channel protection together. |
| `k` | Exact number of complex symbols available for one image. |
| LDPC | Error-correcting code used by the digital systems. |
| Modulation | Rule that maps bits to complex transmitted symbols. |
| Outage | A failed digital delivery handled by a predefined fallback prediction. |
| PAPR | Ratio between peak and average symbol power. |
| Paired evaluation | Comparing systems on the same images and matched channel identities. |
| Preregistration | Fixing decisions and analysis rules before final results are observed. |
| SNR | Signal-to-noise ratio. Higher usually means a cleaner channel. |
| Source coding | Compressing the original image. |
| Task head | Part of the receiver model that predicts the class. |
| Test split | Final held-out data opened once after all choices are frozen. |
| Validation split | Held-out data used to choose settings before final evaluation. |
| Work unit | One independently executable BLER characterization job in G8_C. |

## 28. Where to go for more detail

- Scientific source of truth: [`spec/SPEC.md`](../spec/SPEC.md)
- Current next steps: [`NEXT.md`](../NEXT.md)
- Active campaign recovery: [`instructions/RESUME.md`](../instructions/RESUME.md)
- Literature synthesis: [`docs/literature-review.md`](literature-review.md)
- Schedule: [`docs/gantt-plan.md`](gantt-plan.md)
- Standards boundary: [`docs/standards-and-tools-register.md`](standards-and-tools-register.md)
- Deployment plan: [`docs/deployment-dossier.md`](deployment-dossier.md)
- Crossover explanation: [`docs/crossover-explained.md`](crossover-explained.md)
- Historical implementation details: [`worklogs/`](../worklogs/)
- First Review package: [`deliverables/review-1/`](../deliverables/review-1/)

If two documents disagree, use this priority:

1. `spec/SPEC.md` for scientific meaning and requirements.
2. `instructions/RESUME.md` for the active G8 execution cursor.
3. `NEXT.md` for the current general handoff.
4. This document for explanation.
5. Historical worklogs for background only.

